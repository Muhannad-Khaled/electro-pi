# Task: Finalize the top-level README.md

Decision on your two earlier flags: treat the opening sentence as a factual
fix — the corrected wording is already applied in the full draft below. Do
not create a README for Section 3; the top-level Quick start row covers it.

Below is the COMPLETE draft, already updated with every fact you verified
(layout, commands, filenames, numbers, Section 2 model). Your remaining job:

1. Resolve the ONE remaining `<!-- VERIFY -->` item: determine the actual LLM
   model string Section 1 uses (check its code/config). Fix the prose in
   "Model strategy" accordingly. If Section 1 uses a different model than
   Section 2, state each accurately — do not blur the difference. If Section 1
   still targets `gemini-2.0-flash`, flag it in your report as a likely
   runtime failure for the evaluator (zero free-tier quota) — report only, do
   not change Section 1 code.
2. Sanity-check that all numbers below match the committed results files one
   final time.
3. Write the file as root `README.md`, remove the VERIFY comment, commit
   locally: `docs: add top-level README`. No push.
4. Report back: the Section 1 model finding, plus any remaining mismatch.

All other content is final — no rewording beyond item 1. Do not modify any
file other than the root README.md. Delete this plan file's content from the
README if any of it leaks in — the README starts at the first line inside the
BEGIN/END markers below and contains nothing else.

---

<!-- BEGIN README CONTENT -->

# Electro Pi — AI Engineer Technical Test

Submission for the Mid-Level AI Engineer practical test. Four sections, one
folder each. Sections 1, 2, and 4 have their own READMEs with full setup
steps; Section 3's run steps are below. Each section's half-page write-up is
in its notes file (`NOTES.md`, or `NOTES_section3.md` for Section 3).

## Repository layout

```
section1_livekit/        LiveKit agent with a function tool (20 pts + 5 bonus)
section2_rag/            LangGraph RAG pipeline with citations (20 pts)
section3/                fp16 vs NF4 benchmark on Qwen2.5-1.5B (20 pts)
section4_deployment/     Dockerized FastAPI + llama.cpp serving (25 pts)
files/                   Source corpus for Section 2
```

## Model strategy (read this first)

Two models are used deliberately, not by accident:

- **Sections 1–2 use managed Gemini APIs.** Tool calling and relevance
  grading are the core evaluation criteria in these sections, and small
  self-hosted models are unreliable at both. Section 1 uses
  `<!-- VERIFY: actual Section 1 model string -->`; Section 2 uses
  `gemini-3.1-flash-lite` (chosen after free-tier quota changes retired the
  originally planned `gemini-2.0-flash` — details in section2_rag/README.md).
- **Sections 3–4 use Qwen2.5-1.5B-Instruct (self-hosted).** Quantization and
  deployment only make sense on a model whose weights you control. Section 4
  serves the same model family Section 3 benchmarked, so the two sections
  form one thread: quantize, measure, deploy.

The trade-off between managed and self-hosted serving is discussed in
`section4_deployment/NOTES.md`.

## Quick start

Requirements: Python 3.10+, Docker (Section 4 only), a free Gemini API key
(Sections 1–2 only; https://aistudio.google.com/apikey).

```bash
git clone <repo>
cd electro-pi
export GOOGLE_API_KEY=...   # sections 1–2 read it via dotenv (.env.example provided)
```

Then per section:

| Section | Run | Notes |
|---|---|---|
| 1 | `cd section1_livekit && pip install -r requirements.txt && python run_simulation.py` | Text I/O session; logs show the tool call |
| 2 | `cd section2_rag && pip install -r requirements.txt && python -m src.ingest && python -m src.cli "your question"` | Builds FAISS index, then answers with citations |
| 3 | GPU needed — run `python benchmark.py` in `section3/` on a Colab T4 | Committed results already in `section3/results/` (results.json, outputs.md, summary.md) |
| 4 | `cd section4_deployment && docker build -t electro-pi-llm:latest . && docker run --rm -p 8000:8000 electro-pi-llm:latest` | Then `python load_test.py http://localhost:8000 <label>` |

Sections 1, 2, and 4 have full steps in their own READMEs. Target: each
section runnable within 10 minutes of cloning (Section 3 excepted — it needs
a GPU, which is why its measured results are committed).

## What was built

**Section 1 — LiveKit Agents.** An airline support agent on `livekit-agents`
with an explicit STT → LLM → TTS `AgentSession` pipeline. STT/TTS are stubbed
with text I/O (explicitly permitted by the test); the LLM and tool-calling
logic are real. The agent exposes a `@function_tool`
`get_flight_status(flight_number)` with a mocked lookup, and the committed
transcript shows the LLM invoking it across three turns, including the error
path for an unknown flight. The write-up covers barge-in handling and adding
a second tool safely.

**Section 2 — LangChain / RAG.** A minimal 3-node LangGraph
(`retrieve → grade → answer | abstain`) over a FAISS index of 4 authored
support documents for a fictional food-delivery app ("Wasl Eats"). The
hallucination guardrail is structural, not prompt-based: an LLM grader gates
generation, and the abstain path never calls the generation model. Citations
are program-verified: the pipeline re-extracts cited chunk numbers with a
regex, maps them back to `source_file#chunk_id`, and drops any number the
model invented. Gemini embeddings use split task types (`retrieval_document`
at index time, `retrieval_query` at query time). Four example questions with
actual pipeline output, including the abstain case, are in
`section2_rag/examples/answers.md`.

**Section 3 — Quantization.** Qwen2.5-1.5B-Instruct benchmarked on a Colab T4
at fp16 and at 4-bit NF4 via bitsandbytes, same 5 fixed prompts on both
(`prompts.json`). Measured results (`section3/results/summary.md`):

| Metric | fp16 | NF4 (4-bit) | Delta |
|---|---|---|---|
| Weights VRAM | 2.88 GB | 1.08 GB | −62% |
| Peak VRAM | 2.90 GB | 1.15 GB | −60% |
| Throughput | 27.55 tok/s | 12.65 tok/s | ×0.46 |
| Load time | 3.9 s | 6.2 s | slower |

NF4 is slower here, not faster: bitsandbytes dequantizes on the fly, trading
throughput for memory. Quality degradation concentrated in code generation
and Arabic output; factual and summarization prompts stayed comparable.
Perplexity evaluation was deliberately skipped and is listed as a limitation
in `NOTES_section3.md`.

**Section 4 — Deployment.** The measured NF4 slowdown in Section 3 drove the
serving choice: FastAPI + `llama-cpp-python` with a GGUF Q4_K_M build of the
same model, in a CPU-only Docker image (1.47 GB) so
`docker build && docker run` works on any evaluator machine with no GPU
assumptions. Streaming via SSE, `/health` endpoint, model baked in at build
time. Inference is serialized behind a lock because llama.cpp has no
continuous batching, and the load test was designed to expose that on
purpose. Native (outside Docker): warm single-request TTFT 0.35 s at
12.9 tok/s; under 10 concurrent requests TTFT spreads to 0.36 / 43.0 / 82.3 s
(min/median/max) with a 92.4 s max total — while per-request decode speed
stays flat and aggregate system throughput (~12.2 tok/s) equals
single-request throughput. Concurrency adds zero throughput; the bottleneck
is queueing, not inference speed. That measurement is the empirical case for
the 50-user write-up: continuous batching (vLLM on a real GPU) is the fix.
In-container results add ~25% overhead on single requests (0.54 s TTFT,
9.6 tok/s, expected under WSL2) and degrade ~2.3x under burst load
(concurrent TTFT 0.41 / 144.2 / 192.7 s, max total 201.6 s), with a transient
decode dip attributed — as an unverified hypothesis, stated as such — to
page-cache pressure under the 4 GB memory cap.

## Known limitations

Stated per section in each notes file. The main ones: STT/TTS stubbed in
Section 1; Section 2's cross-document citation example has one documented
deviation (see its examples/answers.md §2); no perplexity metric and no
Section 3 README (run steps are above; its results are committed); Section 4
intentionally serves one request at a time, and its container-slowdown
explanation is a hypothesis, not a profiled result.

<!-- END README CONTENT -->
