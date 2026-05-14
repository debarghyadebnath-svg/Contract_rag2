"""
ingest.py
---------
Single entry point for the contract RAG ingestion pipeline.

Usage:
    python ingest.py contract.pdf
    # or from Python:
    from ingest import ingest
    ingest(Path("contract.pdf"))

Runs sequentially:
  1. SectionAwareSplitter  → Layer 1 (definitions) + Layer 2 (sections) + Layer 3 (tables)
  2. contract_embedder     → BGE-large-en-v1.5 → Chroma + BM25
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from contract_chunker import SectionAwareSplitter
from contract_embedder import embed_and_index


def ingest(pdf_path: str | Path) -> None:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    print(f"\n=== Ingesting: {pdf_path.name} ===\n")

    splitter = SectionAwareSplitter()
    chunks = splitter.split(pdf_path)

    print(f"\n[ingest] Total chunks produced: {len(chunks)}")
    for chunk_type in ("definition", "section", "table"):
        count = sum(1 for c in chunks if c.metadata.get("chunk_type") == chunk_type)
        print(f"  {chunk_type:12s}: {count}")

    embed_and_index(chunks)
    print("\n=== Ingestion complete ===\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_contract.pdf>")
        sys.exit(1)
    ingest(sys.argv[1])
