"""
ingest.py
---------
Single entry point for the contract RAG ingestion pipeline.

Usage:
    python ingest.py contract.pdf                        # cloud (LlamaParse) — default
    python ingest.py contract.pdf --local                # local pdfplumber (no API key needed)
    python ingest.py contract.pdf --extract-features     # also enrich chunks with structured features via Groq

    # or from Python:
    from ingest import ingest
    ingest(Path("contract.pdf"), use_cloud=True, extract_features=True)

Extraction backends:
  Cloud  — LlamaParse: fast, handles complex tables/layouts, needs LLAMAINDEX_API_KEY
  Local  — pdfplumber: offline fallback, no API key required

Runs sequentially:
  1. SectionAwareSplitter     → Layer 1 (definitions) + Layer 2 (sections) + Layer 3 (tables)
  2. enrich_chunks_with_features (optional) → Groq LLM extracts structured metadata per chunk
  3. contract_embedder        → qwen3-embedding:0.6b → Chroma + BM25
"""

from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from contract_chunker import SectionAwareSplitter
from contract_embedder import embed_and_index
from contract_feature_extractor import enrich_chunks_with_features


def ingest(pdf_path: str | Path, use_cloud: bool = True, extract_features: bool = False) -> None:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    backend = "LlamaParse (cloud)" if use_cloud else "pdfplumber (local)"
    print(f"\n=== Ingesting: {pdf_path.name} | Backend: {backend} ===\n")

    splitter = SectionAwareSplitter()
    chunks = splitter.split(pdf_path, use_cloud=use_cloud)

    print(f"\n[ingest] Total chunks produced: {len(chunks)}")
    for chunk_type in ("definition", "section", "table"):
        count = sum(1 for c in chunks if c.metadata.get("chunk_type") == chunk_type)
        print(f"  {chunk_type:12s}: {count}")

    if extract_features:
        enrich_chunks_with_features(chunks)

    embed_and_index(chunks)
    print("\n=== Ingestion complete ===\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_contract.pdf> [--local] [--extract-features]")
        sys.exit(1)

    use_cloud = "--local" not in sys.argv
    extract_features = "--extract-features" in sys.argv
    ingest(sys.argv[1], use_cloud=use_cloud, extract_features=extract_features)

