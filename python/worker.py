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
    cache: Any  # past_key_values (i.e., the KV cache) from the last forward pass step
    generated: list[int] = field(default_factory=list)
    emitted: str = ""  # text already sent to the parent


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

        self.seqs: dict[int, SeqState] = {}

    @torch.inference_mode()
    def step(self, seqs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # any id missing from the batch has finished or been cancelled, so free its KV cache
        batch_ids = {s["id"] for s in seqs}
        for stale in self.seqs.keys() - batch_ids:
            del self.seqs[stale]

        # sequences run one at a time; batched decode needs padded caches and attention masks
        results = []
        for s in seqs:
            seq_id = s["id"]
            prompt = s.get("prompt")

            if prompt:
                # a prompt always means prefill from scratch, discarding any existing state
                self.seqs[seq_id] = self._prefill(prompt)
            else:
                state = self.seqs.get(seq_id)
                if state is None:
                    raise LookupError(f"sequence {seq_id} has no cached state and no prompt")
                self._decode(state)

            results.append(self._result(seq_id, self.seqs[seq_id]))

        return results

    def _prefill(self, prompt: str) -> SeqState:
        if self.tokenizer.chat_template:
            # instruct models only emit end-of-turn when the prompt is framed as a chat turn
            input_ids = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )["input_ids"].to(self.device)
        else:
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        token_id, cache = self._forward(input_ids, None)
        return SeqState(cache=cache, generated=[token_id])

    def _decode(self, state: SeqState) -> None:
        input_ids = torch.tensor([[state.generated[-1]]], device=self.device)
        # Pass along the previous KV cache to continue decoding from the last state
        token_id, state.cache = self._forward(input_ids, state.cache)
        state.generated.append(token_id)

    def _forward(self, input_ids: torch.Tensor, cache: Any) -> tuple[int, Any]:
        out = self.model(input_ids=input_ids, past_key_values=cache, use_cache=True)
        token_id = int(out.logits[0, -1].argmax())  # greedy
        # return the predicted token ID and the updated past key values (KV cache basically)
        return token_id, out.past_key_values

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
