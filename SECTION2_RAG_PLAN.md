# Section 2 — LangChain RAG Pipeline: Implementation Plan

> **Audience:** This document is an execution spec for an AI coding agent (Claude Code).
> Follow it as written. Architecture decisions are already made — do not substitute
> providers, frameworks, or structure without an explicit blocker.

---

## 1. Context

This is Section 2 of a take-home technical test for a Mid-Level AI Engineer role.
The section is worth **20 points** and is graded on:

- **Retrieval quality** — relevant chunks actually get retrieved.
- **Hallucination guardrails** — the "no relevant context found" case is handled
  explicitly. The model must never invent an answer. This is the single most
  important grading criterion.
- **Chain design** — clean, readable LangChain/LangGraph composition.
- **Communication** — README clarity, honest write-up, stated limitations.

The grader must be able to clone the repo and run everything **within 10 minutes**.

## 2. Requirements (verbatim from the test)

1. Chunk and embed a small document set (3–5 files) into a vector store.
2. Build a LangChain **or LangGraph** chain that retrieves relevant chunks and
   answers a user question **with citations back to the source chunk**.
3. Handle the **"no relevant context found"** case explicitly (no hallucination).
4. Provide **3 example questions with the actual answers the pipeline produced**.
5. Half-page write-up: what to change about chunking/retrieval (hybrid search,
   re-ranking) if answer quality on longer documents were poor.

## 3. Locked Architecture Decisions

| Decision | Choice | Rationale (record in README) |
|---|---|---|
| LLM | Gemini `gemini-2.0-flash` via `langchain-google-genai` | Available API key, fast, cheap, good tool/JSON adherence |
| Embeddings | Gemini `models/text-embedding-004` | Same provider, strong retrieval quality |
| Vector store | **FAISS** (local, persisted to disk) | No server, single-file persistence, corpus is tiny |
| Orchestration | **LangGraph** (minimal 3-node graph) | The relevant/irrelevant branch is a real conditional edge — clearer and more testable than an if-statement buried in a chain |
| Relevance gate | **LLM grader node** (single batched call) | Similarity scores alone cannot detect "closest but still irrelevant". Explicitly required by the test |
| Corpus domain | Food-delivery support docs (authored for this test) | Coheres with Section 1's food-delivery voice-agent persona; realistic support-RAG use case |
| Python | 3.10+ | Test requirement |

### Critical implementation constraints (do not violate)

1. **Gemini embeddings `task_type`:** use `task_type="retrieval_document"` when
   embedding chunks at index time and `task_type="retrieval_query"` when
   embedding the user question. Using one type for both measurably hurts recall.
   With `langchain-google-genai`, pass `task_type` to
   `GoogleGenerativeAIEmbeddings`; if the installed version doesn't expose a
   per-call override, instantiate **two** embedding objects (one per task type)
   and use the query-side one inside the retriever.
2. **Grader = ONE LLM call per question**, grading all retrieved chunks together
   and returning structured JSON. Never one call per chunk (rate limits + latency).
3. **The abstain path must NOT call the generation LLM.** It returns a fixed,
   honest message. This is the whole point of the guardrail.
4. **Citations are program-verified, not model-trusted.** Number the chunks
   `[1]..[k]` in the prompt; the model cites numbers; the pipeline maps numbers
   back to `source_file#chunk_id` in the final output. Strip/flag any citation
   number the model invents that doesn't exist in the retrieved set.
5. **Rate limits:** Gemini free tier is RPM-limited. Batch embedding calls during
   ingestion (the LangChain FAISS `from_documents` path batches internally — fine).
   Add a simple retry with exponential backoff (3 attempts) around LLM calls.
6. **No API key in code.** Read `GOOGLE_API_KEY` from environment / `.env`
   (python-dotenv). Provide `.env.example`. Add `.env` to `.gitignore`.

## 4. Repository Structure

```
section2_rag/
├── data/                          # corpus (4 markdown files, authored)
│   ├── refund_policy.md
│   ├── delivery_and_tracking.md
│   ├── payments_and_billing.md
│   └── account_and_orders.md
├── src/
│   ├── __init__.py
│   ├── config.py                  # all tunables in one place
│   ├── ingest.py                  # load → chunk → embed → persist FAISS
│   ├── retriever.py               # load index, retrieve with scores
│   ├── graph.py                   # LangGraph: retrieve → grade → answer|abstain
│   ├── prompts.py                 # grader + answer prompt templates
│   └── cli.py                     # `python -m src.cli "question"` and --interactive
├── examples/
│   └── answers.md                 # 3 real Q&A transcripts + 1 abstain demo
├── tests/
│   └── test_pipeline.py           # smoke tests (see §7)
├── storage/                       # FAISS index output (gitignored, rebuilt by ingest)
├── .env.example
├── .gitignore
├── requirements.txt
├── NOTES.md                       # half-page write-up
└── README.md
```

## 5. Phase-by-Phase Tasks

### Phase 0 — Scaffolding
- Create the structure above.
- `requirements.txt` (pin major versions):
  `langchain`, `langchain-google-genai`, `langgraph`, `langchain-community`,
  `faiss-cpu`, `python-dotenv`, `pydantic`.
- `.gitignore`: `storage/`, `.env`, `__pycache__/`, `.pytest_cache/`.

### Phase 1 — Corpus (`data/`) — PROVIDED, DO NOT AUTHOR
The 4 corpus files are **already written and provided** in `data/`:

- `refund_policy.md`
- `delivery_and_tracking.md`
- `payments_and_billing.md`
- `account_and_orders.md`

**Do not modify, rewrite, "improve", or regenerate these files.** They contain
deliberately engineered properties the example questions depend on:

- **Cross-document coupling:** e.g. cash-on-delivery refunds are wallet-only
  (refund_policy) while cash-payment eligibility rules live in
  payments_and_billing — a cash-refund question must retrieve from both files.
- **Conditional facts:** the refund amount depends on payment method AND order
  stage; order modification rules depend on order stage; the 5-minute
  cancellation window only applies before restaurant acceptance.
- **Body-level facts (not in headings):** e.g. the 20-minute-late →
  automatic wallet credit rule, the 500-meter address-change limit, the
  24-hour-deactivation vs 30-day-deletion distinction. These force real
  retrieval rather than heading matching.

Your only Phase 1 task: verify the 4 files exist in `data/` and are readable.
If they are missing, STOP and ask the user — do not generate substitutes.

### Phase 2 — `src/config.py`
Single dataclass / constants module:

```python
CHUNK_SIZE = 800          # characters
CHUNK_OVERLAP = 120
TOP_K = 4
LLM_MODEL = "gemini-2.0-flash"
EMBEDDING_MODEL = "models/text-embedding-004"
INDEX_DIR = "storage/faiss_index"
DATA_DIR = "data"
ABSTAIN_MESSAGE = (
    "I couldn't find information about this in the support documents. "
    "This question may be outside the scope of what I can answer reliably."
)
```

### Phase 3 — `src/ingest.py`
- Load all `data/*.md` with `TextLoader` (or `DirectoryLoader`).
- Split with `RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE,
  chunk_overlap=CHUNK_OVERLAP, separators=["\n## ", "\n\n", "\n", " "])` —
  heading-first separators keep sections intact.
- Attach metadata to every chunk: `source` (filename), `chunk_id`
  (running int per file, e.g. `refund_policy.md#2`).
- Embed with the **document-task-type** embeddings object; build FAISS via
  `FAISS.from_documents`; `save_local(INDEX_DIR)`.
- Print a summary: files loaded, chunks created, index path.
- Runnable as `python -m src.ingest`. Idempotent (overwrites old index).

### Phase 4 — `src/retriever.py`
- `load_index()` → FAISS via `load_local(..., allow_dangerous_deserialization=True)`
  (document in README why this flag is safe here: index is built locally by the
  user, never downloaded).
- `retrieve(question: str) -> list[RetrievedChunk]` using
  `similarity_search_with_score(question, k=TOP_K)` with the
  **query-task-type** embeddings.
- `RetrievedChunk` (pydantic): `text`, `source`, `chunk_id`, `score`.
- Do **not** hard-filter on score — pass everything to the grader. Scores are
  kept for logging/transparency only. (FAISS L2 scores are not comparable
  across corpora; the grader is the gate.)

### Phase 5 — `src/graph.py` — the LangGraph
State (TypedDict): `question`, `chunks: list[RetrievedChunk]`,
`relevant_ids: list[int]`, `answer`, `citations: list[str]`, `abstained: bool`.

Three nodes + one conditional edge:

```
retrieve ──▶ grade ──▶ (conditional)
                         ├── any relevant ──▶ answer ──▶ END
                         └── none relevant ─▶ abstain ─▶ END
```

**`grade` node** — one LLM call. Prompt (in `prompts.py`):
- Input: the question + numbered chunks `[1]..[k]`.
- Instruction: "Return a JSON object: {\"relevant\": [list of chunk numbers that
  contain information directly useful for answering the question]}. Return an
  empty list if none are relevant. A chunk being on a similar topic is NOT
  enough — it must help answer this specific question."
- Use `llm.with_structured_output` (or JSON mode + robust parse with fallback:
  on parse failure, retry once; on second failure, treat as all-relevant and
  log a warning — fail open on parsing, the answer prompt still constrains
  grounding).

**`answer` node** — one LLM call with only the graded-relevant chunks,
renumbered `[1]..[m]`. Prompt requirements:
- "Answer ONLY from the provided context. Cite every factual claim with the
  chunk number in square brackets, e.g. [1]. If the context only partially
  answers, say what is missing. Do not use outside knowledge."
- Post-process: extract cited numbers with a regex, map to
  `source#chunk_id`, drop hallucinated citation numbers (log if any), and
  attach the mapping as a `Sources:` block in the final output.

**`abstain` node** — no LLM call. Sets `answer = ABSTAIN_MESSAGE`,
`abstained = True`, empty citations.

Wrap LLM calls in retry (3 attempts, exponential backoff, catch 429s).

### Phase 6 — `src/cli.py`
- `python -m src.cli "Do I get a refund if I cancel after 10 minutes?"` →
  prints answer, then `Sources:` block, then (with `--verbose`) retrieval
  scores and grader decision.
- `--interactive` flag for a REPL loop.
- If the index is missing, print a clear instruction to run ingest first
  (do not auto-ingest silently).

### Phase 7 — `examples/answers.md`
Run the pipeline for real and paste **actual outputs** (not hand-written).
Use exactly these four questions — they were designed against the provided
corpus:

1. *"I cancelled my order 8 minutes after placing it. Do I get a full
   refund?"* — tests conditional logic (5-minute window + restaurant
   acceptance rule; correct answer: it depends on whether the restaurant
   accepted, and a late cancellation refunds 50% of the subtotal only).
2. *"I paid cash for an order and half the items were missing. How do I get
   my money back?"* — cross-document: refund rules for >50%-missing orders
   (refund_policy) + cash refunds issued as wallet credit only. Citations
   must span ≥ 2 source files.
3. *"Can I change my delivery address after the courier picks up the
   order?"* — body-level detail (500-meter limit; after "On the way" it is
   at the courier's discretion via the in-app call).
4. *"What's the best pizza place in Cairo?"* — out of scope; must return the
   abstain message verbatim, with the grader marking zero chunks relevant.

Label each with: question, answer, citations, and grader decision. If an
answer comes out wrong or uncited, fix the pipeline — do not edit the corpus
and do not swap the question for an easier one.

### Phase 8 — Tests (`tests/test_pipeline.py`)
Lightweight, no mocking framework needed:
- `test_ingest_creates_index` — run ingest, assert index files exist and
  chunk count > 0.
- `test_relevant_question_answers_with_citations` — assert `abstained is False`
  and `len(citations) > 0`.
- `test_out_of_scope_abstains` — assert `abstained is True` and answer equals
  `ABSTAIN_MESSAGE`.
- `test_citation_numbers_map_to_real_chunks` — no dangling citations.
- Skip (with clear message) if `GOOGLE_API_KEY` is unset, so the suite doesn't
  hard-fail for a grader without a key.

### Phase 9 — `README.md`
Must include, in order:
1. One-paragraph overview + architecture diagram (ASCII is fine) of the graph.
2. Setup: `python -m venv`, `pip install -r requirements.txt`, copy
   `.env.example` → `.env`, add key.
3. Run: ingest command, one-shot question command, interactive mode, tests.
4. Design decisions table (copy from §3 of this plan, adapted).
5. Known limitations (see §8 below) — honesty is explicitly scored.

### Phase 10 — `NOTES.md` (the half-page write-up)
Address the exact question asked: *what would you change about chunking or
retrieval if answer quality on longer documents were poor?* Cover, concisely
and from an engineering (not documentation-recital) standpoint:
- **Chunking:** move from fixed-size to structure-aware/semantic chunking;
  parent-document retrieval (retrieve small, feed the LLM the parent section);
  tune overlap; add heading-path metadata into the chunk text itself.
- **Retrieval:** hybrid search (BM25 + dense, e.g. `EnsembleRetriever`) for
  exact-term queries (IDs, codes, product names) that embeddings miss;
  cross-encoder re-ranking (e.g. bge-reranker) over a larger candidate pool
  (retrieve 20 → re-rank → top 4); query rewriting/multi-query for vague
  questions.
- One sentence on evaluation: you'd validate any change with a small golden
  Q&A set + retrieval-hit-rate metric before/after, not by eyeballing.

## 6. Acceptance Criteria (self-verify before finishing)

- [ ] `python -m src.ingest` builds the index from scratch with a summary printed.
- [ ] A relevant question returns an answer where **every factual sentence has
      a citation**, and the `Sources:` block maps each number to `file#chunk_id`.
- [ ] The out-of-scope question returns the abstain message **without** a
      generation LLM call (verify via `--verbose` log showing grader → abstain).
- [ ] The cross-document question cites ≥ 2 different source files.
- [ ] `pytest` passes (or cleanly skips without an API key).
- [ ] Fresh-clone-to-first-answer takes < 10 minutes following README only.
- [ ] No secrets committed; `.env.example` present.
- [ ] `examples/answers.md` contains real pipeline output, including the
      grader's decision per example.

## 7. Explicit "Do NOT" List

- Do NOT add a web UI, Docker, or a server for this section — CLI is the
  deliverable. (Deployment is Section 4's job.)
- Do NOT filter chunks by raw FAISS score as the relevance gate.
- Do NOT let the abstain path go through the generation LLM.
- Do NOT call the grader once per chunk.
- Do NOT hand-write the example answers — run the pipeline and paste output.
- Do NOT add extra dependencies (re-rankers, BM25, Chroma, LangSmith) — they
  belong in the write-up as future work, not in the code.
- Do NOT swallow API errors silently — log and surface them.
- Do NOT modify, regenerate, or "improve" the provided corpus files in
  `data/` — the example questions depend on their exact cross-document
  structure. If a question fails, fix the pipeline, not the data.

## 8. Known Limitations (to state honestly in README)

- Grader adds one extra LLM call (~300–700ms) per question — accepted trade-off
  for a hard hallucination gate.
- Grader itself is an LLM and can misjudge borderline chunks; mitigated by
  fail-open parsing plus a grounded answer prompt as second line of defense.
- FAISS flat index — fine at this scale; would need IVF/HNSW + a vector DB
  service at production scale.
- Corpus is authored for the test; real support docs would need a cleaning/
  normalization pass and periodic re-indexing.
- Single-turn only — no conversational memory / follow-up question rewriting.
