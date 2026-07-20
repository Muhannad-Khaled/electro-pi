"""Load the corpus, chunk it, embed it, and persist a FAISS index.

Run as: python -m src.ingest
Idempotent: overwrites any existing index.
"""

import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_MODEL,
    INDEX_DIR,
)


def load_and_chunk():
    """Load every data/*.md file and split it into chunks with source metadata."""
    files = sorted(DATA_DIR.glob("*.md"))
    if not files:
        sys.exit(f"No .md files found in {DATA_DIR}")

    # Heading-first separators keep markdown sections intact where possible.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n\n", "\n", " "],
    )

    chunks = []
    for path in files:
        docs = TextLoader(str(path), encoding="utf-8").load()
        for chunk_id, chunk in enumerate(splitter.split_documents(docs)):
            chunk.metadata["source"] = path.name
            chunk.metadata["chunk_id"] = chunk_id
            chunks.append(chunk)
    return files, chunks


def main():
    load_dotenv()
    files, chunks = load_and_chunk()

    # Document-side task type: chunks are embedded as retrieval *documents*.
    # The query side (src/retriever.py) uses task_type="retrieval_query".
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, task_type="retrieval_document"
    )
    index = FAISS.from_documents(chunks, embeddings)
    index.save_local(str(INDEX_DIR))

    print(f"Files loaded : {len(files)}")
    for path in files:
        n = sum(1 for c in chunks if c.metadata["source"] == path.name)
        print(f"  - {path.name}: {n} chunks")
    print(f"Total chunks : {len(chunks)}")
    print(f"Index saved  : {INDEX_DIR}")


if __name__ == "__main__":
    main()
