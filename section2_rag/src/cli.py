"""CLI for the RAG pipeline.

Usage:
  python -m src.cli "Do I get a refund if I cancel after 10 minutes?"
  python -m src.cli --interactive
  python -m src.cli --verbose "..."   # also print retrieval scores + grader decision
"""

import argparse
import logging
import warnings

from dotenv import load_dotenv

# langgraph emits a PendingDeprecationWarning about serializer defaults on
# import — irrelevant to users of this CLI.
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

from src.graph import build_graph  # noqa: E402


def print_result(result: dict, verbose: bool) -> None:
    if verbose:
        print("--- Retrieval ---")
        for i, chunk in enumerate(result.get("chunks", []), start=1):
            print(f"  [{i}] {chunk.ref}  (L2 distance: {chunk.score:.4f})")
        relevant = result.get("relevant_ids", [])
        decision = f"relevant chunks: {relevant}" if relevant else "NO relevant chunks"
        print(f"--- Grader: {decision} ---")
        route = "abstain (no generation LLM call)" if result["abstained"] else "answer"
        print(f"--- Route: {route} ---\n")
    print(result["answer"])


def main():
    parser = argparse.ArgumentParser(description="Ask the Wasl Eats support docs.")
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument("--interactive", action="store_true", help="REPL mode")
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show retrieval scores and grader decision",
    )
    args = parser.parse_args()

    if not args.question and not args.interactive:
        parser.error("provide a question or use --interactive")

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    load_dotenv()
    app = build_graph()  # exits with instructions if the index is missing

    if args.question:
        print_result(app.invoke({"question": args.question}), args.verbose)

    if args.interactive:
        print("Interactive mode — empty line or Ctrl+C to exit.")
        while True:
            try:
                question = input("\nQ: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not question:
                break
            print_result(app.invoke({"question": question}), args.verbose)


if __name__ == "__main__":
    main()
