# Section 4 — Model Deployment

A quantized LLM (Qwen2.5-1.5B-Instruct, Q4_K_M GGUF) served CPU-only behind a
FastAPI REST API with token-by-token SSE streaming. It is containerized with
the model baked into the image, and ships a concurrency load test with real
measured results in `results/`. See `NOTES.md` for design reasoning and the
scaling discussion.

## Run with Docker (recommended)

```bash
docker build -t electro-pi-llm:latest .
docker run --rm -p 8000:8000 electro-pi-llm:latest
```

No volumes, no env vars, no GPU — the model is inside the image. The API is
ready when `docker ps` shows the container healthy (model load takes a few
seconds after start).

```bash
# Health
curl http://localhost:8000/health

# Non-streaming completion
curl -X POST http://localhost:8000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is a quantized LLM?", "max_tokens": 64}'

# Streaming (SSE, one event per token, ends with data: [DONE])
curl -N -X POST http://localhost:8000/v1/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Count to ten.", "max_tokens": 64}'
```

(On Windows PowerShell use `curl.exe`, not the `curl` alias.)

## Run natively (no Docker)

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir .\models
uvicorn server:app --host 0.0.0.0 --port 8000
```

Quick check: `python smoke_test.py`

## Load test

With the server (native or container) running on port 8000:

```powershell
python load_test.py http://localhost:8000 <label>
```

Runs a warm single-request baseline, then 10 concurrent streaming requests,
measuring per-request TTFT (first token event), total latency, and tokens/sec.
Results land in `results/load_test_<label>.json` and `.md`. Committed results:
`native` (bare Windows) and `docker` (container under WSL2).

## Design decisions

- **FastAPI + llama-cpp-python over vLLM/TGI** — the grader must be able to
  `docker build && docker run` on any machine with no GPU. CPU-only GGUF is
  the zero-friction option; vLLM/TGI on GPU is the production path (NOTES.md).
- **Model baked into the image** — instant, flag-free `docker run` at the cost
  of a ~2GB image.
- **Single-inference lock** — llama.cpp handles one inference at a time, so
  requests serialize behind an `asyncio.Lock` while the event loop stays free
  (health checks and streaming keep working). The load test deliberately
  measures this queueing; NOTES.md discusses how production serving removes it.
