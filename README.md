# Blizzard

An LLM inference server written in Go. It accepts OpenAI-compatible chat completion requests over HTTP, schedules them with continuous batching, and streams generated tokens back over Server-Sent Events. 
A Python worker process, wraps the Hugging Face Transformers library for actual inference, and uses stdin and stdout to communicate with LLM engine.

## How it works

```
HTTP client ──► apiserver ──► scheduler ──► llmengine ──► modelexecutor ──► modelworker ──► python/worker.py(worker process)
   ◄──── SSE tokens ◄──── stream ◄─────────────┘
```

| Package | Responsibility |
|---|---|
| `internal/apiserver` | OpenAI-compatible `POST /v1/chat/completions` (streaming only): request validation and SSE chunks ending in `data: [DONE]` |
| `internal/scheduler` | Bounded FIFO queue of pending requests |
| `internal/llmengine` | Continuous-batching loop: admits requests, runs one step at a time, delivers tokens, enforces each request's token limit and cancellation |
| `internal/stream` | Non-blocking token channel between the engine and each HTTP handler |
| `internal/modelexecutor` | Step protocol: sends each conversation once, then only sequence IDs; detects out-of-sync replies |
| `internal/modelworker` | Starts and stops the Python process; JSON request/response over stdin/stdout |
| `python/worker.py` | Loads the model, holds one KV cache shared by all running sequences, and decodes them in one batched forward pass per step |

### Model worker: batching and KV cache

The worker keeps a single KV cache for the whole batch, one row per running sequence. Rows have different lengths, so shorter rows are left-padded and a mask marks which columns are real tokens:

```
row A:  a1 a2 a3 a4 a5     mask 1 1 1 1 1
row B:  0  0  0  b1 b2          0 0 0 1 1
```

Each step:

1. **Leave**: rows for finished or cancelled sequences are dropped, and leading columns that are padding in every row are trimmed.
2. **Decode**: every running sequence generates one token in a single `[rows, 1]` forward pass over the shared cache.
3. **Prefill**: new prompts run as one padded batch in a separate forward pass, which also produces their first token.
4. **Join**: the new rows are left-padded to the cache width (or the cache to theirs) and appended.

Limitations:

- Only full-attention models are supported; the worker refuses sliding-window and other cache layer types at startup.
- While a new prompt is prefilled, running streams pause for that step (about 2 s on CPU for Qwen2.5-1.5B). Chunked prefill is planned.
- Decoding is greedy.

## Run locally

Requires Go 1.26+ and [uv](https://docs.astral.sh/uv/).

**1. Set up the Python environment** (once):

```powershell
cd python
uv sync
cd ..
```

**2. Start the inferencing server** from the repo root:

```powershell
$env:BLIZZARD_PYTHON = "$PWD\python\.venv\Scripts\python.exe"   # Linux: $PWD/python/.venv/bin/python
go run ./cmd
```

The first start downloads the model. The server is ready when it logs `listening on :8080`.

**3. Send a chat completion request:**

Bash:

```bash
curl -N -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-1.5B-Instruct","stream":true,"max_completion_tokens":100,"messages":[{"role":"system","content":"Be brief."},{"role":"user","content":"Write a short story about a fox."}]}'
```

PowerShell:

```powershell
curl.exe --% -N -X POST http://localhost:8080/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"Qwen/Qwen2.5-1.5B-Instruct\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":\"Tell me about Seattle\"}]}"
```

`--%` passes the rest of the line to curl unchanged, so the escaped JSON quotes survive on both Windows PowerShell 5.1 and PowerShell 7.

The request follows OpenAI chat completions: `model` must match the served model, `messages` takes `system`, `user` and `assistant` turns with string content, and `max_completion_tokens` (or the older `max_tokens`) defaults to the server cap. Only `"stream": true` is supported so far.

The response is a stream of `chat.completion.chunk` events: the first carries `delta.role`, the next ones carry `delta.content`, and the last has an empty `delta` and a `finish_reason` of `stop` or `length`. The stream ends with `data: [DONE]`.

Errors:

- **Before streaming starts**, the server replies with a plain-text status: `400` for an invalid request, `404` for an unknown model.
- **During streaming**, the status is already `200`, so a failure arrives in-band as `data: {"error": {"message": ..., "type": "server_error", ...}}` followed by `data: [DONE]`, which OpenAI SDKs raise as an error.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `BLIZZARD_PYTHON` | `python` | Python interpreter; point it at the venv |
| `BLIZZARD_WORKER` | `python/worker.py` | Worker script path |
| `BLIZZARD_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Hugging Face model ID (full-attention models with a chat template only) |
| `HF_HUB_OFFLINE` | unset | Set to `1` once the model is cached, to skip network checks at startup |

The listen address, queue size, batch size, and token limit are constants in `cmd/main.go`.
