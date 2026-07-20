"""LangGraph pipeline: retrieve -> grade -> (answer | abstain).

The grade node is the hallucination guardrail: one batched LLM call decides
which retrieved chunks are actually relevant. If none are, the abstain node
returns a fixed message WITHOUT calling the generation LLM.
"""

import json
import logging
import re
import time
from typing import TypedDict

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from src.config import (
    ABSTAIN_MESSAGE,
    LLM_INNER_RETRIES,
    LLM_MAX_RETRIES,
    LLM_MODEL,
    LLM_RETRY_BASE_DELAY,
)
from src.prompts import ANSWER_PROMPT, GRADER_PROMPT, format_chunks
from src.retriever import RetrievedChunk, load_index, retrieve

logger = logging.getLogger(__name__)


class RagState(TypedDict, total=False):
    question: str
    chunks: list[RetrievedChunk]
    relevant_ids: list[int]  # 1-based indices into `chunks`
    answer: str
    citations: list[str]  # "source_file#chunk_id" refs actually cited
    abstained: bool


def _invoke_with_retry(llm, prompt: str) -> str:
    """Call the LLM with exponential backoff (Gemini free tier returns 429s)."""
    for attempt in range(LLM_MAX_RETRIES):
        try:
            return llm.invoke(prompt).content
        except Exception as err:  # noqa: BLE001 — retry any transient API error
            if attempt == LLM_MAX_RETRIES - 1:
                raise
            delay = LLM_RETRY_BASE_DELAY * (2**attempt)
            logger.warning(
                "LLM call failed (attempt %d/%d): %s — retrying in %.0fs",
                attempt + 1, LLM_MAX_RETRIES, err, delay,
            )
            time.sleep(delay)
    raise RuntimeError("unreachable")


def _parse_grader_json(raw: str) -> list[int]:
    """Extract {"relevant": [...]} from the grader output."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON object in grader output: {raw!r}")
    relevant = json.loads(match.group(0))["relevant"]
    return [int(n) for n in relevant]


def build_graph(index=None, llm=None):
    """Compile the RAG graph. `index`/`llm` injectable for tests."""
    if index is None:
        index = load_index()
    if llm is None:
        # Keep the client's built-in retries low: on the free tier every
        # rejected retry still consumes per-minute quota. Our own wrapper
        # (_invoke_with_retry) waits long enough for the window to reset.
        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL, temperature=0, max_retries=LLM_INNER_RETRIES
        )

    def retrieve_node(state: RagState) -> RagState:
        chunks = retrieve(index, state["question"])
        return {"chunks": chunks}

    def grade_node(state: RagState) -> RagState:
        """ONE batched LLM call grading all chunks together."""
        prompt = GRADER_PROMPT.format(
            question=state["question"],
            chunks=format_chunks([c.text for c in state["chunks"]]),
        )
        n = len(state["chunks"])
        for attempt in range(2):
            raw = _invoke_with_retry(llm, prompt)
            try:
                ids = _parse_grader_json(raw)
                # Drop out-of-range numbers the grader may have invented.
                return {"relevant_ids": [i for i in ids if 1 <= i <= n]}
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as err:
                logger.warning("Grader parse failure (attempt %d): %s", attempt + 1, err)
        # Fail open on parsing: treat all chunks as relevant — the grounded
        # answer prompt is the second line of defense.
        logger.warning("Grader unparseable twice — failing open (all chunks relevant)")
        return {"relevant_ids": list(range(1, n + 1))}

    def answer_node(state: RagState) -> RagState:
        relevant = [state["chunks"][i - 1] for i in state["relevant_ids"]]
        # Renumber the surviving chunks [1]..[m] for the answer prompt.
        prompt = ANSWER_PROMPT.format(
            question=state["question"],
            chunks=format_chunks([c.text for c in relevant]),
        )
        answer = _invoke_with_retry(llm, prompt)

        # Program-verified citations: map cited numbers back to source#chunk_id;
        # strip any number the model invented.
        cited_nums = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        invented = sorted(n for n in cited_nums if not 1 <= n <= len(relevant))
        if invented:
            logger.warning("Dropping hallucinated citation numbers: %s", invented)
        valid_nums = sorted(n for n in cited_nums if 1 <= n <= len(relevant))
        citations = [relevant[n - 1].ref for n in valid_nums]

        sources_block = "\n".join(
            f"  [{n}] {relevant[n - 1].ref}" for n in valid_nums
        )
        full_answer = f"{answer.strip()}\n\nSources:\n{sources_block}"
        return {"answer": full_answer, "citations": citations, "abstained": False}

    def abstain_node(state: RagState) -> RagState:
        """No LLM call — fixed honest message."""
        return {"answer": ABSTAIN_MESSAGE, "citations": [], "abstained": True}

    def route_after_grade(state: RagState) -> str:
        return "generate" if state["relevant_ids"] else "abstain"

    # Node named "generate" because LangGraph forbids node names that collide
    # with state keys ("answer").
    graph = StateGraph(RagState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_node)
    graph.add_node("generate", answer_node)
    graph.add_node("abstain", abstain_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade", route_after_grade, {"generate": "generate", "abstain": "abstain"}
    )
    graph.add_edge("generate", END)
    graph.add_edge("abstain", END)
    return graph.compile()


def ask(question: str, app=None) -> RagState:
    """Convenience entry point: run one question through the pipeline."""
    load_dotenv()
    if app is None:
        app = build_graph()
    return app.invoke({"question": question})
