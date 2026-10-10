"""Blizzard model worker: runs one forward step per request over a JSON-lines protocol.

The parent writes one request per line to stdin and reads exactly one reply per request from
the original stdout. Closing stdin is the shutdown signal.
"""

import os
import sys

# claim the original stdout for the protocol before anything can print to it; print() now goes to stderr
_protocol = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", newline="\n", buffering=1)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = sys.stderr

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import argparse  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402
import traceback  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache  # noqa: E402
from transformers.cache_utils import DynamicLayer  # noqa: E402


def send(msg: dict[str, Any]) -> None:
    _protocol.write(json.dumps(msg) + "\n")
    _protocol.flush()


@dataclass
class SeqState:
    # Token IDs the model has produced so far. Each step appends one. The last one is not in the KV cache yet:
    # it is the input of the next decode step. It is also checked against the end-of-text tokens to set done.
    generated: list[int]
    # text already sent; a token that ends mid-character sends nothing until the next one completes it so this is used to track what was the last token sent back to Go.
    emitted: str = ""


class Worker:
    def __init__(self, model_id: str) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        # load the tokenizer and model for the given model ID
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype).to(self.device)
        self.model.eval()

        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}

        # padded positions are masked out, so any valid id works when the model defines no pad token
        pad = self.tokenizer.pad_token_id
        self.pad_id = pad if pad is not None else next(iter(self.eos_ids), 0)

        # Join and trim edit each layer's K/V tensors directly, which is only valid for plain full-attention layers.
        # Other layer types (sliding window, quantized, linear attention) keep extra state those edits would
        # silently corrupt, producing wrong text rather than an error.
        layer_types = {type(layer) for layer in DynamicCache(config=self.model.config).layers}
        if layer_types - {DynamicLayer}:
            names = sorted(t.__name__ for t in layer_types)
            raise ValueError(f"unsupported KV cache layers {names}: only full attention is supported")

        # per sequence cache of token IDs so that Go Engine doesnt hae to resent entire text each step.
        self.seqs: dict[int, SeqState] = {}

        # One KV cache shared by all running sequences, one row per sequence:
        #   cache  every layer holds K and V of shape [rows, kv_heads, width, head_dim]
        #   mask   [rows, width]: 1 = real token, 0 = left padding
        #   rows   sequence ID of each row
        self.cache: DynamicCache | None = None
        self.mask: torch.Tensor | None = None
        self.rows: list[int] = []

    def reset(self) -> None:
        self.seqs.clear()
        self.cache, self.mask, self.rows = None, None, []

    @torch.inference_mode()
    def step(self, seqs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # a prompt always means start over, discarding any existing state
        new = {s["id"]: self._tokenize(s["prompt"]) for s in seqs if s.get("prompt")}
        running = {s["id"] for s in seqs} - new.keys()
        if missing := running - self.seqs.keys():
            raise LookupError(f"sequences {sorted(missing)} have no state and no prompt")

        # any id missing from the batch has finished or been cancelled, so forget it
        for seq_id in self.seqs.keys() - running:
            del self.seqs[seq_id]
        self._keep_rows([i for i, seq_id in enumerate(self.rows) if seq_id in running])

        if self.rows:
            self._decode()
        if new:
            self._prefill(new)

        return [self._result(s["id"], self.seqs[s["id"]]) for s in seqs]

    def _decode(self) -> None:
        """Generates one token for every running sequence in a single [rows, 1] forward pass over the shared cache."""
        input_ids = torch.tensor([[self.seqs[seq_id].generated[-1]] for seq_id in self.rows], device=self.device)
        # a row's position is its count of real tokens, not the cache width, which includes its padding
        position_ids = self.mask.sum(dim=1, keepdim=True)
        self.mask = torch.cat([self.mask, self.mask.new_ones(len(self.rows), 1)], dim=1)

        logits = self.model(
            input_ids=input_ids,
            attention_mask=self.mask,
            position_ids=position_ids,
            past_key_values=self.cache,
            use_cache=True,
        ).logits

        for seq_id, token_id in zip(self.rows, logits[:, -1].argmax(dim=-1).tolist()):  # greedy
            self.seqs[seq_id].generated.append(token_id)

    def _prefill(self, prompts: dict[int, list[int]]) -> None:
        """Runs the new prompts as one padded batch into a fresh cache, then joins it to the shared cache.

        This is a separate forward pass from decode: sharing one pass would mean padding every running row
        to the prompt length, wasting compute and leaving padding in the middle of its cache.
        """
        input_ids, attention_mask, position_ids = self._pad_batch(list(prompts.values()))
        cache = DynamicCache(config=self.model.config)

        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        ).logits

        for seq_id, token_id in zip(prompts, logits[:, -1].argmax(dim=-1).tolist()):  # greedy
            self.seqs[seq_id] = SeqState(generated=[token_id])
        self._join(cache, attention_mask, list(prompts))

    def _join(self, cache: DynamicCache, mask: torch.Tensor, ids: list[int]) -> None:
        """Appends freshly prefilled rows to the shared cache, left-padding whichever side is narrower.

        Example: the shared cache holds A (5 tokens) and B (2 tokens) joins. In every layer, for K and V:

            shared   A: a1 a2 a3 a4 a5
            joined   B: b1 b2
            result   A: a1 a2 a3 a4 a5     mask  1 1 1 1 1
                     B: 0  0  0  b1 b2           0 0 0 1 1

        Left padding keeps every row's newest token in the last column, so each decode appends one column
        for all rows. Padding columns are safe: each key already encodes its token's real position, so the
        column it sits in doesn't matter, and the mask hides the padding.
        """
        if self.cache is None:
            self.cache, self.mask, self.rows = cache, mask, ids
            return

        width = max(self.mask.shape[1], mask.shape[1])

        def pad(kv: torch.Tensor) -> torch.Tensor:  # [rows, heads, tokens, dim] -> zero columns before the tokens
            return F.pad(kv, (0, 0, width - kv.shape[-2], 0))

        for shared, joined in zip(self.cache.layers, cache.layers):
            shared.keys = torch.cat([pad(shared.keys), pad(joined.keys)])
            shared.values = torch.cat([pad(shared.values), pad(joined.values)])
        self.mask = torch.cat([
            F.pad(self.mask, (width - self.mask.shape[1], 0)),
            F.pad(mask, (width - mask.shape[1], 0)),
        ])
        self.rows = self.rows + ids

    def _keep_rows(self, keep: list[int]) -> None:
        """Drops every row not in keep, then trims leading columns that are padding in all remaining rows.

        Without the trim, the cache would stay as wide as the longest sequence that ever ran:

            before   A: a1 a2 a3 a4 a5 a6
                     B: 0  0  0  b1 b2 b3
            A leaves
            after    B: b1 b2 b3
        """
        if len(keep) == len(self.rows):
            return
        if not keep:
            self.cache, self.mask, self.rows = None, None, []
            return

        self.cache.batch_select_indices(torch.tensor(keep, device=self.device))
        self.mask = self.mask[keep]
        self.rows = [self.rows[i] for i in keep]

        start = int(self.mask.any(dim=0).int().argmax())  # first column holding a real token in any row
        if start:
            for layer in self.cache.layers:
                layer.keys = layer.keys[:, :, start:]
                layer.values = layer.values[:, :, start:]
            self.mask = self.mask[:, start:]

    def _tokenize(self, prompt: str) -> list[int]:
        if self.tokenizer.chat_template:
            # instruct models only emit end-of-turn when the prompt is framed as a chat turn
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_dict=True,
            )["input_ids"]
        return self.tokenizer(prompt).input_ids

    def _pad_batch(self, rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Left-pads token rows into one batch so a single forward pass serves every sequence.

        When batching with HF forward pass, all sequences must be same length, hence the need for left-padding.
        A longer sequence joins: the existing rows get left pad.
        A shorter sequence joins: it's row gets left pad.

        This also means, KV Cache needs to be adjusted to account for these PAD tokens.
        That waste is exactly what vLLM's paged blocks avoid: no padding means no wasted rows in any layer.

        Example with pad id 0 and rows of length 2 and 4:

            rows            [[5, 6], [7, 8, 9, 4]]
            input_ids       [[0, 0, 5, 6], [7, 8, 9, 4]]
            attention_mask  [[0, 0, 1, 1], [1, 1, 1, 1]]
            position_ids    [[0, 0, 0, 1], [0, 1, 2, 3]]

        Padding goes on the left so every row's last real token sits in the final column, where the
        next-token logits are read. The mask stops real tokens attending to padding, and positions
        start at 0 on each row's first real token so padding doesn't shift them.
        """
        width = max(len(row) for row in rows)
        input_ids = torch.full((len(rows), width), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        for i, row in enumerate(rows):
            input_ids[i, width - len(row):] = torch.tensor(row)
            attention_mask[i, width - len(row):] = 1

        position_ids = (attention_mask.cumsum(dim=-1) - 1).clamp(min=0)
        return input_ids.to(self.device), attention_mask.to(self.device), position_ids.to(self.device)

    def _result(self, seq_id: int, state: SeqState) -> dict[str, Any]:
        done = state.generated[-1] in self.eos_ids

        # decoding tokens one by one breaks multi-byte characters, so decode the whole output and diff
        text = self.tokenizer.decode(
            state.generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        if text.endswith("\ufffd") and not done:
            return {"id": seq_id, "token": "", "done": False}

        token = text[len(state.emitted):]
        state.emitted = text
        return {"id": seq_id, "token": token, "done": done}


def main() -> int:
    # the parent owns this process's lifetime and stops it by closing stdin, so ignore Ctrl+C
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    try:
        worker = Worker(args.model)
    except Exception as e:
        traceback.print_exc()
        send({"ready": False, "error": f"load {args.model}: {e}"})
        return 1

    print(f"loaded {args.model} on {worker.device}")
    send({"ready": True})

    while line := sys.stdin.buffer.readline():
        step = None
        try:
            req = json.loads(line)
            step = req["step"]
            send({"step": step, "results": worker.step(req["seqs"])})
        except Exception as e:
            traceback.print_exc()
            # the engine ends every active sequence when a step fails, and the cache may be half-updated, so drop it all
            worker.reset()
            send({"step": step, "results": [], "error": str(e)})

            # an OOM can leave the CUDA context unusable; exit so the parent starts a clean process
            if isinstance(e, torch.cuda.OutOfMemoryError):
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
