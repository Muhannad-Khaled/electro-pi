# Notes — improving chunking & retrieval for longer documents

*The question: what would I change about chunking or retrieval if answer
quality on longer documents were poor?*

**Chunking.** Fixed-size character splitting is the first thing to replace.
On long documents it cuts mid-section, splitting a condition from its rule
(e.g. a refund percentage from the eligibility clause it belongs to). I'd move
to structure-aware chunking — split on the markdown/heading hierarchy first,
then only sub-split sections that exceed the size budget — and prepend the
heading path (`Refund Policy > Cancellation Window`) into each chunk's text so
the embedding carries its context. If chunks must stay small for retrieval
precision, I'd add parent-document retrieval: index small chunks, but feed the
LLM the enclosing parent section, so retrieval stays sharp while generation
gets full context. Overlap is a tunable, not a fix — it papers over bad split
points rather than removing them.

**Retrieval.** Dense embeddings alone miss exact-term queries — order IDs,
error codes, product names — that lexical search catches trivially. I'd add
hybrid search (BM25 + dense, e.g. LangChain's `EnsembleRetriever` with
reciprocal-rank fusion). Next, widen the candidate pool and re-rank: retrieve
~20 chunks, score them with a cross-encoder re-ranker (e.g. `bge-reranker`),
keep the top 4. Cross-encoders read the query and chunk *together*, so they
resolve "similar topic vs. actually answers this" far better than bi-encoder
distance — which would also let the re-ranker share load with the LLM grader.
For vague or multi-part user questions, query rewriting / multi-query
expansion (generate 2–3 paraphrases, union the results) recovers chunks the
original phrasing misses.

**Evaluation.** I would validate any of these changes against a small golden
Q&A set with a retrieval-hit-rate metric (did the gold chunk land in top-k?)
measured before and after — not by eyeballing individual answers.
