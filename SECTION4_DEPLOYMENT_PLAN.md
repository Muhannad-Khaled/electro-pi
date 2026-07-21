# Section 4 — Model Deployment: Implementation Plan

> **Audience:** Claude Code (terminal agent) executing on the developer's Windows machine.
> **Goal:** Serve a quantized LLM behind a FastAPI REST API with streaming, containerize it, and load-test it. This is Section 4 of the Electro Pi AI Engineer take-home test (25 points, highest-weighted section, scored on "production-mindedness").

---

## 0. Context & Hard Constraints

- **Repo:** existing repo `electro-pi`, organized one folder per section. All work for this section goes in `section4_deployment/`. **Do NOT modify anything in `section1_*`, `section2_*`, or `section3*` folders.**
- **Host machine:** Windows 10/11, 8GB RAM, GTX 1650 (4GB VRAM — deliberately NOT used), PowerShell, Python 3.10+, Docker Desktop (WSL2 backend, memory-capped at 4GB via `.wslconfig`).
- **Locked architecture decision (do not revisit):** FastAPI + `llama-cpp-python` + **CPU-only** inference, using the official pre-quantized GGUF of the same model family used in Section 3.
  - Model: `Qwen/Qwen2.5-1.5B-Instruct-GGUF`, file `qwen2.5-1.5b-instruct-q4_k_m.gguf` (~1.1GB).
  - Rationale (goes in NOTES.md): the grader must be able to run `docker build && docker run` on any machine with no GPU guarantees; CPU-only GGUF is the only zero-friction option. vLLM/TGI are discussed in the write-up as the production path, not implemented.
- **The model file may already exist** at `section4_deployment/models/qwen2.5-1.5b-instruct-q4_k_m.gguf`. If present, reuse it. If absent, download it (Phase A, step 2).
- **Two phases:** Phase A runs everything natively on Windows (no Docker). Phase B is Docker, done last, in one session. Do not start Phase B until Phase A is fully verified.
- **Git hygiene:** add/extend `.gitignore` so that `models/`, `venv/`, `__pycache__/`, and `.env` are never committed. The GGUF file (1.1GB) must NEVER be committed. Commit locally with clear messages, but **do NOT push to remote without explicit user confirmation.**
- **Ask before deviating.** If a step fails and the fix requires changing an architectural decision, stop and ask the user instead of improvising.

---

## 1. Target Directory Structure

```
section4_deployment/
├── server.py             # FastAPI app
├── load_test.py          # concurrency benchmark (httpx + asyncio)
├── smoke_test.py         # minimal "does the model respond" check (may already exist)
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── models/               # GGUF lives here locally (gitignored)
├── results/              # committed: load test outputs (JSON + markdown table)
└── NOTES.md              # half-page write-up (see §6)
```

---

## 2. Phase A — Native (no Docker)

### Step A1 — Environment

If `venv/` doesn't exist in `section4_deployment/`:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install fastapi "uvicorn[standard]" llama-cpp-python httpx "huggingface_hub[cli]" sse-starlette
```

Write `requirements.txt` with **pinned versions** (`pip freeze` filtered to direct deps is fine).

**Known risk:** `llama-cpp-python` may fail to build on Windows without a C++ toolchain. If `pip install llama-cpp-python` fails with CMake/Visual Studio errors, install the prebuilt CPU wheel instead:

```powershell
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

If both fail, stop and report the error to the user.

### Step A2 — Model download (skip if file exists)

```powershell
huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct-GGUF qwen2.5-1.5b-instruct-q4_k_m.gguf --local-dir .\models
```

Verify file exists and size is ~1.1GB before proceeding.

### Step A3 — `server.py`

Requirements:

- **Load the model once** at startup using FastAPI lifespan (not per-request, not at import time of route handlers).
  - `Llama(model_path=..., n_ctx=2048, n_threads=<physical cores, e.g. 4>, verbose=False)`
  - Read model path from env var `MODEL_PATH` with default `./models/qwen2.5-1.5b-instruct-q4_k_m.gguf` (Docker will override this).
- **Endpoints:**
  1. `GET /health` → `{"status": "ok", "model_loaded": true}`. Must respond even while an inference is running (used by Docker HEALTHCHECK).
  2. `POST /v1/generate` — request body: `{"prompt": str, "max_tokens": int = 256, "temperature": float = 0.7}`. Returns full completion JSON: generated text, token counts, and latency_ms.
  3. `POST /v1/generate/stream` — same body; returns **Server-Sent Events**, one event per token chunk, terminated by a final `data: [DONE]` event. Each token event payload: `{"token": str}`.
- **Concurrency model (critical, this is a scoring point):**
  - `llama-cpp-python` handles ONE inference at a time and its calls are blocking/CPU-bound.
  - Use a single global `asyncio.Lock`. Each request acquires the lock, then runs the blocking llama call in a thread (`await asyncio.to_thread(...)` for non-streaming; for streaming, iterate the llama token generator inside a thread and hand tokens to the async side via an `asyncio.Queue`).
  - This means concurrent requests **serialize** — that is expected and intentional. Do NOT try to work around it with multiple model instances (RAM won't allow it). The queueing behavior is deliberately surfaced in the load test and discussed in NOTES.md.
- Use Pydantic models for request/response validation. Cap `max_tokens` at 512 server-side.
- Basic error handling: return proper 4xx for validation issues, 500 with a clean message (no stack trace leak) for inference failures.
- Keep the file clean and commented — trade-off comments matter for scoring ("we care as much about your reasoning as your code").

Run locally with: `uvicorn server:app --host 0.0.0.0 --port 8000`

### Step A4 — Verify manually

- `curl http://localhost:8000/health`
- One non-streaming request, one streaming request (PowerShell: `curl.exe -N -X POST ...` to see SSE flowing token by token). Confirm tokens arrive incrementally, not in one burst.

### Step A5 — `load_test.py`

- Pure Python, `httpx.AsyncClient` + `asyncio.gather`, no locust.
- Sends **10 concurrent requests** to `/v1/generate/stream` with a fixed prompt and `max_tokens=128`.
- Per request, measure:
  - **TTFT** = time from request send until the **first token SSE event** is received (NOT first HTTP byte/headers).
  - **Total latency** = until `[DONE]`.
  - Tokens received → derive tokens/sec.
- Output:
  - `results/load_test_native.json` (raw per-request numbers)
  - `results/load_test_native.md` — a markdown table: per-request TTFT + total, then summary row with min / median / max TTFT and total latency.
- Also run a single-request baseline (1 request alone) first and record it — the contrast between 1-request TTFT and the 10th queued request's TTFT is the headline number for the write-up.
- **Run it and save real results.** The test explicitly scores "real measurements (not guesses)".
- Expectation to sanity-check against: on CPU, single-request TTFT should be well under ~2s once warm; under 10-way concurrency, the last queued requests will have TTFT of tens of seconds. That is the expected, honest result.

---

## 3. Phase B — Docker (only after Phase A is green)

### Step B1 — `Dockerfile`

- Base: `python:3.11-slim`.
- Install deps from `requirements.txt`. For `llama-cpp-python` inside Linux, the prebuilt CPU wheel index (`--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu`) avoids a long native compile; use it. If it's unavailable for the platform, fall back to compiling (add `build-essential cmake` in a build stage, use multi-stage so the final image stays slim).
- **Bake the model into the image at build time** (locked decision): download the GGUF in a `RUN` step via `curl -L` from the HuggingFace resolve URL, into `/app/models/`. This makes `docker run` work instantly for the grader with zero volumes/env setup. Expected image size ~1.6–2GB — acceptable, documented in NOTES.md.
- Non-root user, `WORKDIR /app`, `EXPOSE 8000`.
- `HEALTHCHECK` hitting `/health`.
- `ENV MODEL_PATH=/app/models/qwen2.5-1.5b-instruct-q4_k_m.gguf`
- `CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]`
- Write `.dockerignore`: `venv/`, `models/`, `__pycache__/`, `results/`, `.git` — the local model dir must not be sent in the build context (it's downloaded inside the build instead; context upload of 1.1GB would be slow and redundant).

### Step B2 — Build, run, verify

```powershell
docker build -t electro-pi-llm:latest .
docker run --rm -p 8000:8000 electro-pi-llm:latest
```

- Repeat Step A4 verification against the container.
- Run `load_test.py` against the container → save as `results/load_test_docker.json` / `.md`. (Container numbers may be slightly slower than native due to the 4GB WSL2 cap — report both honestly.)
- Record the final image size (`docker images`) for NOTES.md.

---

## 4. README section

Add a `section4_deployment/README.md` (or a section in the repo's top-level README, matching how previous sections did it) with:

- What this is, in 3 sentences.
- Exact run instructions, **Docker path first** (`docker build` / `docker run` / example `curl` for both endpoints), then the native path.
- How to run the load test.
- A short "Design decisions" list: FastAPI over vLLM (with the CPU/grader-portability justification), model baked into image, single-inference lock.

Target: a grader must get a working streaming response within 10 minutes of cloning.

---

## 5. Acceptance checklist (verify ALL before declaring done)

- [ ] `python smoke_test.py` returns model output.
- [ ] `/health`, `/v1/generate`, `/v1/generate/stream` all work natively.
- [ ] Streaming is genuinely incremental (visible token-by-token with `curl -N`).
- [ ] `load_test.py` produced real numbers in `results/` for native run.
- [ ] `docker build` succeeds from a clean context; `docker run` serves immediately with no extra flags.
- [ ] Load test re-run against the container, results saved separately.
- [ ] `.gitignore` protects `models/` and `venv/`; `git status` shows no large files staged.
- [ ] NOTES.md written (see §6), containing the actual measured numbers, not placeholders.
- [ ] Local commits made; **nothing pushed** (user pushes manually).

---

## 6. NOTES.md write-up (draft it, ~half page)

Structure it as:

1. **What was built** — 3–4 lines: FastAPI + llama-cpp-python serving Qwen2.5-1.5B Q4_K_M GGUF on CPU, streaming via SSE, containerized, model baked into the image.
2. **Why this stack and not vLLM/TGI** — grader portability (no GPU assumption, `docker build && run` guaranteed), model size doesn't need continuous batching to demo, available hardware (4GB VRAM) rules out serious GPU serving. vLLM/TGI positioned as the production choice, not the take-home choice.
3. **Measured results** — small table: single-request TTFT & tokens/sec vs 10-concurrent min/median/max TTFT (native + Docker). One honest paragraph: llama.cpp serves one request at a time, so concurrent requests queue; the degradation curve in our own numbers demonstrates exactly why production serving needs continuous batching.
4. **Scaling to 50 concurrent users** (the required question) — answer in priority order: (1) switch the engine to vLLM/TGI on a GPU instance for continuous batching + PagedAttention (single biggest win — turns the queueing we measured into parallel decoding); (2) horizontal autoscaling behind a load balancer, scaled on queue depth/TTFT, not CPU%; (3) a real request queue with backpressure + per-request timeouts and rate limiting so one slow client can't starve the rest; (4) caching where traffic allows (prefix/KV cache for shared system prompts; response cache only if queries repeat); (5) observability first-class: TTFT, tokens/sec, queue depth as Prometheus metrics — you cannot autoscale on signals you don't measure. Tie each item back to the measured bottleneck rather than listing buzzwords.

Tone: first person, plain engineering English, honest about trade-offs. No marketing language.

---

## 7. Execution order for the agent

1. Read this file fully. Inspect `section4_deployment/` for what already exists (venv? model? smoke_test.py?). Reuse, don't recreate.
2. Phase A steps A1→A5 in order. Stop and report if `llama-cpp-python` install fails after both strategies.
3. Show the user the native load-test table before starting Phase B (checkpoint).
4. Phase B, verification, results.
5. README + NOTES.md with real numbers.
6. Final acceptance checklist pass, local commit, report summary to user.
