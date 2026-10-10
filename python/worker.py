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
from dataclasses import dataclass, field  # noqa: E402
from typing import Any  # noqa: E402

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def send(msg: dict[str, Any]) -> None:
    _protocol.write(json.dumps(msg) + "\n")
    _protocol.flush()


@dataclass
class SeqState:
    # The prompt's token IDs, after the chat template. Tokenized once on the first step and reused every step to rebuild the row.
    prompt_ids: list[int] 
    # Token IDs the model has produced so far. Each step appends one, and prompt_ids + generated is the row fed to the next forward pass. Its last entry is checked against the end-of-text tokens to set done.
    generated: list[int] = field(default_factory=list)
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

        # per sequence cache of token IDs so that Go Engine doesnt hae to resent entire text each step.
        self.seqs: dict[int, SeqState] = {}

    @torch.inference_mode()
    def step(self, seqs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # delete stale sequences; that is any id missing from the batch has finished or been cancelled, so forget it
        batch_ids = {s["id"] for s in seqs}
        for stale in self.seqs.keys() - batch_ids:
            del self.seqs[stale]

        states = []
        for s in seqs:
            seq_id = s["id"]
            prompt = s.get("prompt")
            if prompt:
                # a prompt always means start over, discarding any existing state
                self.seqs[seq_id] = SeqState(prompt_ids=self._tokenize(prompt))
            elif seq_id not in self.seqs:
                raise LookupError(f"sequence {seq_id} has no state and no prompt")
            states.append(self.seqs[seq_id])

        # Transformer batched forward pass requires left padding all sequences to the same length.
        # Given it also gives back attention masks, both the batch and the mask are fed into the model for proper attention computation.
        input_ids, attention_mask, position_ids = self._pad_batch(
            [st.prompt_ids + st.generated for st in states]
        )

        # forward pass
        logits = self.model(
            input_ids=input_ids, # shape [batch_size, seq_length]
            attention_mask=attention_mask,
            position_ids=position_ids, # positions for each token in the sequence
            use_cache=False,
        ).logits

        next_ids = logits[:, -1].argmax(dim=-1).tolist()  # greedy

        results = []
        for s, st, token_id in zip(seqs, states, next_ids):
            st.generated.append(token_id)
            results.append(self._result(s["id"], st))
        return results

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
            send({"step": step, "results": [], "error": str(e)})

            # an OOM can leave the CUDA context unusable; exit so the parent starts a clean process
            if isinstance(e, torch.cuda.OutOfMemoryError):
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
