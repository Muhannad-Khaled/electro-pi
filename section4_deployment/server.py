"""FastAPI server exposing a quantized Qwen2.5-1.5B-Instruct GGUF over REST.

Design notes (why, not just what):

- The model is loaded ONCE in the lifespan handler. Loading per-request would
  add ~seconds of latency and multiply RAM usage; loading at module import
  would make the app untestable and break tooling that imports the module.

- llama-cpp-python is blocking and CPU-bound, and a single Llama instance is
  NOT safe for concurrent calls. We therefore serialize all inference behind
  one asyncio.Lock and run the blocking call in a worker thread so the event
  loop stays free — /health keeps responding while a generation is running.
  Concurrent requests queue; that is intentional. Running multiple model
  instances is not an option on an 8GB host (each instance costs ~1.3GB RAM),
  and the queueing behavior is exactly what the load test is meant to surface.
"""

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from llama_cpp import Llama
from pydantic import BaseModel, Field

logger = logging.getLogger("uvicorn.error")

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
)
# Physical cores on the target machine. More threads than physical cores hurts
# llama.cpp throughput (hyperthread contention on the compute-bound matmuls).
N_THREADS = int(os.environ.get("N_THREADS", "4"))
N_CTX = 2048

llm: Llama | None = None
# Single global lock: one inference at a time (see module docstring).
inference_lock = asyncio.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global llm
    logger.info("Loading model from %s ...", MODEL_PATH)
    t0 = time.perf_counter()
    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        verbose=False,
    )
    logger.info("Model loaded in %.1fs", time.perf_counter() - t0)
    yield
    llm = None


app = FastAPI(title="electro-pi LLM server", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=8000)
    # Hard server-side cap at 512: protects the shared single-inference queue
    # from one client monopolizing it with a huge generation.
    max_tokens: int = Field(default=256, ge=1, le=512)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


def _chat_args(req: GenerateRequest) -> dict:
    # create_chat_completion applies Qwen's chat template for us; raw
    # create_completion on an instruct model yields much worse output.
    return dict(
        messages=[{"role": "user", "content": req.prompt}],
        max_tokens=req.max_tokens,
        temperature=req.temperature,
    )


@app.get("/health")
async def health():
    # Must respond even while an inference is running: we never touch the
    # inference lock here, and inference itself runs in a worker thread, so
    # the event loop is free to serve this. Used by Docker HEALTHCHECK.
    return {"status": "ok", "model_loaded": llm is not None}


@app.post("/v1/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    t0 = time.perf_counter()
    async with inference_lock:
        try:
            result = await asyncio.to_thread(llm.create_chat_completion, **_chat_args(req))
        except Exception:
            logger.exception("Inference failed")
            raise HTTPException(status_code=500, detail="Inference failed")
    return GenerateResponse(
        text=result["choices"][0]["message"]["content"],
        prompt_tokens=result["usage"]["prompt_tokens"],
        completion_tokens=result["usage"]["completion_tokens"],
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


@app.post("/v1/generate/stream")
async def generate_stream(req: GenerateRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    async def event_source():
        # The llama token generator is blocking, so it must run in a thread.
        # Tokens cross back to the async side through a queue; None sentinels
        # signal completion or failure.
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def produce():
            try:
                for chunk in llm.create_chat_completion(stream=True, **_chat_args(req)):
                    delta = chunk["choices"][0]["delta"]
                    token = delta.get("content")
                    if token:
                        loop.call_soon_threadsafe(queue.put_nowait, ("token", token))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception:
                logger.exception("Streaming inference failed")
                loop.call_soon_threadsafe(queue.put_nowait, ("error", None))

        # Hold the lock for the whole stream: the generator occupies the
        # model until it is exhausted, so releasing earlier would just let a
        # second request crash into it.
        async with inference_lock:
            producer = asyncio.create_task(asyncio.to_thread(produce))
            try:
                while True:
                    kind, token = await queue.get()
                    if kind == "token":
                        yield f"data: {json.dumps({'token': token})}\n\n"
                    elif kind == "done":
                        yield "data: [DONE]\n\n"
                        break
                    else:  # error — clean message, no stack trace to client
                        yield f"data: {json.dumps({'error': 'Inference failed'})}\n\n"
                        break
            finally:
                await producer

    return StreamingResponse(event_source(), media_type="text/event-stream")
