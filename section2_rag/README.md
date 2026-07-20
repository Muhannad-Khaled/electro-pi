# Section 2 — RAG Pipeline with Citations & Hallucination Guardrails

A LangGraph RAG pipeline over a small set of food-delivery support documents
(Wasl Eats — same persona as Section 1's voice agent). It retrieves relevant
chunks from a local FAISS index, answers with **program-verified citations**
back to the source chunks, and **explicitly abstains** when no retrieved
context is actually relevant — the model never invents an answer.

```
                        ┌──────────┐     ┌───────┐
 question ──▶ retrieve ─▶  grade   ├──▶──┤answer ├──▶ END   (any relevant chunk)
             (FAISS      (1 LLM    │     └───────┘
              top-4)      call)    │     ┌───────┐
                                   └──▶──┤abstain├──▶ END   (zero relevant chunks;
                                         └───────┘           NO LLM call — fixed message)
```

- **retrieve** — embeds the question (query task type) and pulls the top-4
  chunks from FAISS. Raw similarity scores are logged but never used as a
  relevance gate: FAISS distances aren't comparable across corpora, and
  "closest" can still be irrelevant.
- **grade** — ONE batched LLM call sees the question plus all numbered chunks
  and returns JSON listing which chunks actually help answer it. This is the
  hallucination guardrail the conditional edge branches on.
- **answer** — one LLM call over only the graded-relevant chunks, instructed
  to cite every claim as `[n]`. The pipeline then re-extracts the cited
  numbers with a regex, maps them back to `source_file#chunk_id`, and drops
  any number the model invented. Citations are verified by the program, not
  trusted from the model.
- **abstain** — no LLM call at all. Returns a fixed, honest message.

## Setup

Requires Python 3.10+.

```bash
cd section2_rag
python -m venv venv
venv\Scripts\activate          # Windows   (Linux/macOS: source venv/bin/activate)
pip install -r requirements.txt
copy .env.example .env         # then put your GOOGLE_API_KEY in .env
```

Get a free key at https://aistudio.google.com/apikey.

## Run

```bash
# 1. Build the FAISS index (required once; idempotent)
python -m src.ingest

# 2. Ask a question
python -m src.cli "Do I get a refund if I cancel after 10 minutes?"

# Show retrieval scores + grader decision
python -m src.cli --verbose "What's the best pizza place in Cairo?"

# REPL mode
python -m src.cli --interactive

# Tests (skips cleanly if GOOGLE_API_KEY is unset)
pytest
```

> **Free-tier note:** the Gemini free tier allows only ~5–20 requests/minute.
> Each question makes 2 LLM calls (grader + answer). The pipeline retries 429s
> with long backoff automatically, so a rate-limited question may take up to a
> minute or two — it recovers on its own. If you ask questions back-to-back,
> pace them ~30s apart.

Real pipeline transcripts for 4 example questions (including the abstain
case) are in [`examples/answers.md`](examples/answers.md).

## Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| LLM | Gemini `gemini-3.1-flash-lite` | Fast, cheap, good JSON adherence, stable (non-preview), comfortable free-tier daily quota. Originally planned `gemini-2.0-flash`, but the free tier now has zero quota for it; `gemini-2.5-flash` is retired for new users; and `gemini-3.5-flash`'s free-tier daily cap proved too small even for this test suite |
| Embeddings | Gemini `models/gemini-embedding-001` | Same provider, strong retrieval quality; separate `retrieval_document` / `retrieval_query` task types for index vs. query (measurably better recall than one type for both). The originally planned `text-embedding-004` has been retired by Google (API returns 404), so its stable GA successor is used |
| Vector store | FAISS, persisted to `storage/` | No server to run; corpus is tiny; single-command rebuild |
| Orchestration | LangGraph, 3 nodes + 1 conditional edge | The relevant/irrelevant branch is a real graph edge — clearer and more testable than an if-statement buried inside a chain |
| Relevance gate | LLM grader node (one batched call per question) | Similarity scores alone can't detect "closest but still irrelevant"; grading all chunks in one call keeps latency and rate-limit cost flat |
| Chunking | `RecursiveCharacterTextSplitter`, 800 chars / 120 overlap, heading-first separators | Keeps markdown sections intact where possible |

Note on `allow_dangerous_deserialization=True` in `src/retriever.py`: the flag
is required by LangChain to unpickle a FAISS index. It is safe here because
the index is always built locally by `src.ingest` on your machine — it is
never downloaded or shared.

## Known limitations

- The grader adds one extra LLM call (~300–700 ms) per question — an accepted
  trade-off for a hard hallucination gate.
- The grader is itself an LLM and can misjudge borderline chunks. Mitigations:
  fail-open JSON parsing (on repeated parse failure all chunks pass to the
  grounded answer prompt, which is the second line of defense), and the answer
  prompt forbids outside knowledge.
- FAISS flat (exact) index — fine at this scale; production would need
  IVF/HNSW or a vector-DB service, plus periodic re-indexing.
- The corpus was authored for this test; real support docs would need a
  cleaning/normalization pass.
- Single-turn only: no conversational memory or follow-up question rewriting.
- The cross-document example (cash + missing items) retrieves from both
  `refund_policy.md` and `payments_and_billing.md`, but its citations land in
  a single file: both decisive facts happen to live in `refund_policy.md`,
  and the grader (verified against a stronger model) correctly judges the
  payments chunk unnecessary for the answer. See the honest note in
  `examples/answers.md` §2.
- See [`NOTES.md`](NOTES.md) for what I'd change about chunking/retrieval if
  answer quality on longer documents were poor.
