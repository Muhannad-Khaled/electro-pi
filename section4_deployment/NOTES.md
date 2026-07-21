# Section 4 — Notes

## What was built

A FastAPI server running Qwen2.5-1.5B-Instruct (Q4_K_M GGUF, ~1.1GB) on CPU via
llama-cpp-python. Two inference endpoints — `/v1/generate` (full JSON response)
and `/v1/generate/stream` (SSE, one event per token) — plus `/health`, which
stays responsive during inference. The whole thing is containerized with the
model baked into the image (1.47GB final size): `docker build && docker run`
works on any machine with no GPU, volumes, or env setup.

## Why this stack and not vLLM/TGI

The grader must be able to run this anywhere. vLLM/TGI assume a GPU; my
machine's 4GB VRAM can't serve seriously anyway, and a 1.5B model doesn't need
continuous batching to demo an API. CPU-only GGUF is the zero-friction choice
for a take-home; vLLM/TGI on GPU is what I'd deploy in production (below).

llama.cpp processes one inference at a time, so the server serializes requests
behind a single `asyncio.Lock`, running the blocking call in a worker thread to
keep the event loop (and `/health`) responsive. Multiple model instances were
ruled out: ~1.3GB RAM each on an 8GB host. The queueing this causes is
deliberately measured rather than hidden.

## Measured results (10 concurrent streaming requests, max_tokens=128)

| Metric | Native (Windows) | Docker (WSL2, 4GB cap) |
|---|---:|---:|
| Single-request TTFT (warm) | 0.35s | 0.54s |
| Single-request tokens/s | 12.9 | 9.6 |
| TTFT min / median / max | 0.36 / 43.0 / 82.3s | 0.41 / 144.2 / 192.7s |
| Total latency max | 92.4s | 201.6s |

A single warm request is fine: sub-second first token, ~13 tok/s native.
Under 10-way concurrency the picture is honest and ugly: requests serialize,
so each queued request's TTFT is the sum of every generation ahead of it —
the last native request waits 82s for its first token. Per-request decode
speed stays flat (~12–13 tok/s native), proving the bottleneck is pure
queueing, not compute degradation. The container is ~2.3x worse under load,
with early requests dropping to 1.7–7 tok/s before recovering. That pattern is
consistent with memory pressure under the 4GB WSL2 cap (likely page-cache
eviction of the mmap'd model file when 10 connections arrive at once), though
we did not profile inside the container to confirm the cause. This curve is
the whole argument for continuous
batching in production: the compute is there, the serving model wastes it.

## Scaling to 50 concurrent users

In priority order, each tied to what the numbers above show:

1. **Swap the engine: vLLM or TGI on a GPU instance.** The single biggest win.
   Continuous batching + PagedAttention decode many requests in parallel on
   one device — it turns the 82s queued TTFT we measured into roughly
   constant TTFT across the batch. Nothing else on this list matters until
   the one-request-at-a-time engine is gone.
2. **Horizontal autoscaling behind a load balancer**, scaled on queue depth /
   TTFT, not CPU%. Our own data shows why: during the load test the CPU was
   ~100% busy the entire time — CPU% says "healthy, fully utilized" while the
   10th user waits 82 seconds. Queue depth is the signal that actually tracks
   user pain.
3. **A real request queue with backpressure**: bounded queue, per-request
   timeouts, rate limiting, and 429s when full. Today a burst of 50 requests
   would all hang open for minutes (we watched 10 do exactly that); it's
   better to fail fast for the tail than time out for everyone.
4. **Caching where traffic allows**: prefix/KV-cache reuse for shared system
   prompts (cuts prefill from every request); a response cache only if
   queries genuinely repeat.
5. **Observability first**: TTFT, tokens/sec, and queue depth as Prometheus
   metrics. Items 2 and 3 autoscale and shed load on these signals — you
   cannot react to a queue you don't measure.
