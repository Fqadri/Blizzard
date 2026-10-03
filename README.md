# Blizzard

An LLM inference server written in Go. It accepts prompts over HTTP, schedules them with continuous batching, and streams generated tokens back over Server-Sent Events. A Python worker process loads and executes any HF Causal Language model.

## How it works

```
HTTP client ──► apiserver ──► scheduler ──► llmengine ──► modelexecutor ──► modelworker ──► python/worker.py
   ◄──── SSE tokens ◄──── stream ◄─────────────┘
```

| Package | Responsibility |
|---|---|
| `internal/apiserver` | HTTP endpoint, SSE streaming, terminal `done` / `error` events |
| `internal/scheduler` | Bounded FIFO queue of pending requests |
| `internal/llmengine` | Continuous-batching loop: admits requests, runs one step at a time, delivers tokens, enforces `maxTokens` and cancellation |
| `internal/stream` | Non-blocking token channel between the engine and each HTTP handler |
| `internal/modelexecutor` | Step protocol: sends each prompt once, then only sequence IDs; detects out-of-sync replies |
| `internal/modelworker` | Starts and stops the Python process; JSON request/response over stdin/stdout |
| `python/worker.py` | Loads the model, holds the KV cache per sequence, and generates one token per sequence per step |

## Run locally

Requires Go 1.25+ and [uv](https://docs.astral.sh/uv/).

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

**3. Send a prompt:**

Bash:

```bash
curl -N -X POST http://localhost:8080/stream_generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Write a short story about a fox."}'
```

PowerShell:

```powershell
curl.exe --% -N -X POST http://localhost:8080/stream_generate -H "Content-Type: application/json" -d "{\"prompt\":\"Tell me about Seattle\"}"
```

`--%` passes the rest of the line to curl unchanged, so the escaped JSON quotes survive on both Windows PowerShell 5.1 and PowerShell 7.

The response is a stream of `data: {"generated_text": "..."}` events, ending with `event: done` and a `finish_reason` of `stop` or `length`.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `BLIZZARD_PYTHON` | `python` | Python interpreter; point it at the venv |
| `BLIZZARD_WORKER` | `python/worker.py` | Worker script path |
| `BLIZZARD_MODEL` | `Qwen/Qwen2.5-1.5B-Instruct` | Hugging Face model ID |
| `HF_HUB_OFFLINE` | unset | Set to `1` once the model is cached, to skip network checks at startup |

The listen address, queue size, batch size, and token limit are constants in `cmd/main.go`.
