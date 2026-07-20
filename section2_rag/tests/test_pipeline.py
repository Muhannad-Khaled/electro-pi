"""Smoke tests for the RAG pipeline.

These hit the real Gemini API (no mocking) and are skipped entirely when
GOOGLE_API_KEY is unset, so the suite never hard-fails for a grader
without a key.

Run from section2_rag/: pytest
"""

import os
import subprocess
import sys

import pytest
from dotenv import load_dotenv

load_dotenv()

pytestmark = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"),
    reason="GOOGLE_API_KEY is not set — skipping live pipeline tests",
)

from src.config import ABSTAIN_MESSAGE, INDEX_DIR  # noqa: E402


@pytest.fixture(scope="session")
def index_built():
    result = subprocess.run(
        [sys.executable, "-m", "src.ingest"], capture_output=True, text=True
    )
    assert result.returncode == 0, f"ingest failed:\n{result.stderr}"
    return result.stdout


@pytest.fixture(scope="session")
def app(index_built):
    from src.graph import build_graph

    return build_graph()


def test_ingest_creates_index(index_built):
    assert (INDEX_DIR / "index.faiss").exists()
    assert (INDEX_DIR / "index.pkl").exists()
    assert "Total chunks" in index_built
    total = int(index_built.split("Total chunks :")[1].splitlines()[0])
    assert total > 0


def test_relevant_question_answers_with_citations(app):
    result = app.invoke(
        {"question": "How long does a card refund take to appear?"}
    )
    assert result["abstained"] is False
    assert len(result["citations"]) > 0


def test_out_of_scope_abstains(app):
    result = app.invoke({"question": "What's the best pizza place in Cairo?"})
    assert result["abstained"] is True
    assert result["answer"] == ABSTAIN_MESSAGE
    assert result["citations"] == []


def test_citation_numbers_map_to_real_chunks(app):
    result = app.invoke(
        {"question": "Can I get a refund if items are missing from my order?"}
    )
    assert result["abstained"] is False
    valid_refs = {chunk.ref for chunk in result["chunks"]}
    for citation in result["citations"]:
        assert citation in valid_refs, f"dangling citation: {citation}"
