"""Load the persisted FAISS index and retrieve chunks for a question."""

import sys

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel

from src.config import EMBEDDING_MODEL, INDEX_DIR, TOP_K


class RetrievedChunk(BaseModel):
    text: str
    source: str
    chunk_id: int
    score: float  # FAISS L2 distance — logged for transparency, never a gate

    @property
    def ref(self) -> str:
        return f"{self.source}#{self.chunk_id}"


def load_index() -> FAISS:
    if not INDEX_DIR.exists():
        sys.exit(
            f"FAISS index not found at {INDEX_DIR}.\n"
            "Build it first with: python -m src.ingest"
        )
    # Query-side task type: the user question is embedded as a retrieval *query*
    # (chunks were embedded with task_type="retrieval_document" in src/ingest.py).
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, task_type="retrieval_query"
    )
    # Safe here: the pickle is produced locally by src.ingest on this machine,
    # never downloaded from an untrusted source.
    return FAISS.load_local(
        str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True
    )


def retrieve(index: FAISS, question: str) -> list[RetrievedChunk]:
    """Top-K similarity search. No score filtering — the LLM grader is the
    relevance gate (raw FAISS distances are not comparable across corpora)."""
    results = index.similarity_search_with_score(question, k=TOP_K)
    return [
        RetrievedChunk(
            text=doc.page_content,
            source=doc.metadata["source"],
            chunk_id=doc.metadata["chunk_id"],
            score=float(score),
        )
        for doc, score in results
    ]
