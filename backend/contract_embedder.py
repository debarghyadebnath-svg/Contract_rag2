"""
contract_embedder.py
--------------------
Embeds contract chunks using Cohere's embed-english-v3.0 API,
stores dense vectors in a local Chroma persistent store, and builds a BM25
sparse index in-memory (serialised to disk for reuse).

Why Cohere instead of Ollama/Qwen?
  - No local server required (no `ollama serve` step).
  - embed-english-v3.0 produces 1024-dim vectors with strong legal domain accuracy.
  - Native input_type support: 'search_document' for indexing, 'search_query' for queries.
  - Free tier: 1,000 API calls/month, up to 96 texts per call.

Cohere Embed API:
  POST https://api.cohere.com/v2/embed
  Body: {"model": "embed-english-v3.0", "texts": [...], "input_type": "search_document", "embedding_types": ["float"]}

Chroma pattern:
  collection.upsert(ids, documents, embeddings, metadatas)
  collection.query(query_embeddings=[[...]], n_results=N)

BM25 pattern (rank-bm25):
  BM25Okapi(tokenized_corpus).get_scores(tokenized_query)
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import chromadb
from chromadb import PersistentClient
from rank_bm25 import BM25Okapi

from contract_chunker import Chunk

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_DIR              = Path(__file__).parent / "chroma_contract_store"
CHROMA_COLLECTION_NAME  = "ihda_contract"
BM25_INDEX_PATH         = Path(__file__).parent / "bm25_contract_index.pkl"

COHERE_MODEL            = "embed-english-v3.0"
COHERE_EMBED_DIM        = 1024

# Cohere allows up to 96 texts per request (free tier) — keep at 90 for safety
COHERE_BATCH_SIZE       = 90


# ---------------------------------------------------------------------------
# Lazy singleton — Chroma client created once per process
# ---------------------------------------------------------------------------

_chroma_client: PersistentClient | None = None
_collection: chromadb.Collection | None = None


def _get_collection() -> chromadb.Collection:
    global _chroma_client, _collection
    if _collection is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ---------------------------------------------------------------------------
# Cohere embedding helper
# ---------------------------------------------------------------------------

def _get_cohere_client():
    """Return a Cohere client, raising clearly if the key is missing."""
    try:
        import cohere  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "[embedder] cohere package not installed. Run: pip install cohere"
        )
    api_key = os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "[embedder] COHERE_API_KEY not set. Add it to your .env file."
        )
    return cohere.ClientV2(api_key)


def _embed_with_cohere(
    texts: list[str],
    input_type: str = "search_document",
) -> list[list[float]]:
    """
    Embed texts via the Cohere API in batches.

    Parameters
    ----------
    texts      : list of strings to embed
    input_type : 'search_document' for indexing, 'search_query' for queries
    """
    client = _get_cohere_client()
    all_embeddings: list[list[float]] = []
    total = len(texts)

    import time

    for start in range(0, total, COHERE_BATCH_SIZE):
        batch = texts[start : start + COHERE_BATCH_SIZE]
        
        # Retry loop for rate limits
        for attempt in range(5):
            try:
                response = client.embed(
                    texts=batch,
                    model=COHERE_MODEL,
                    input_type=input_type,
                    embedding_types=["float"],
                )
                break  # success, exit retry loop
            except Exception as exc:
                if "429" in str(exc) or "too_many_requests" in str(exc).lower():
                    wait = 20 * (2 ** attempt)  # Wait 20s, 40s, 80s... since limit is per minute
                    print(f"  [embedder] Cohere rate limit hit. Waiting {wait}s before retrying...")
                    time.sleep(wait)
                else:
                    raise  # Re-raise if it's a different error

        # Response: EmbedByTypeResponse → .embeddings.float_ is List[List[float]]
        batch_embeddings = response.embeddings.float_
        all_embeddings.extend(batch_embeddings)
        done = min(start + COHERE_BATCH_SIZE, total)
        print(f"  [embedder] {done}/{total} chunks encoded …", flush=True)

    return all_embeddings


# ---------------------------------------------------------------------------
# Wrapper to mimic SentenceTransformer.encode() interface (used in query.py)
# ---------------------------------------------------------------------------

def _get_model():
    """
    Returns a wrapper with .encode() that hits the Cohere API.
    Accepts the same interface as SentenceTransformer so contract_query.py
    doesn't need to know which backend is active.
    """
    class CohereEncoder:
        def encode(
            self,
            texts: str | list[str],
            normalize_embeddings: bool = True,  # kept for API compat; Cohere normalises by default
            input_type: str = "search_query",   # default: query mode
        ) -> list[float] | list[list[float]]:
            if isinstance(texts, str):
                return _embed_with_cohere([texts], input_type=input_type)[0]
            return _embed_with_cohere(texts, input_type=input_type)

    return CohereEncoder()


# ---------------------------------------------------------------------------
# Serialise metadata for Chroma (values must be str | int | float | bool)
# ---------------------------------------------------------------------------

def _flatten_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            flat[k] = v
        elif isinstance(v, list):
            flat[k] = json.dumps(v)
        else:
            flat[k] = str(v)
    return flat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_and_index(chunks: list[Chunk]) -> None:
    """
    Embed all chunks via Cohere, upsert into Chroma, build BM25 index.
    Safe to call multiple times (upsert is idempotent on chunk_id).
    """
    if not chunks:
        print("[embedder] No chunks to index.")
        return

    collection = _get_collection()
    texts     = [c.page_content for c in chunks]
    ids       = [c.metadata["chunk_id"] for c in chunks]
    metadatas = [_flatten_metadata(c.metadata) for c in chunks]

    print(f"[embedder] Encoding {len(texts)} chunks with Cohere/{COHERE_MODEL} …")
    embeddings = _embed_with_cohere(texts, input_type="search_document")

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"[embedder] Upserted {len(ids)} vectors into Chroma '{CHROMA_COLLECTION_NAME}'")

    _build_bm25_index(texts, ids)


def _build_bm25_index(texts: list[str], ids: list[str]) -> None:
    tokenized = [text.lower().split() for text in texts]
    bm25 = BM25Okapi(tokenized)
    index_data = {"bm25": bm25, "ids": ids, "texts": texts}
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(index_data, f)
    print(f"[embedder] BM25 index saved → {BM25_INDEX_PATH}")


def load_bm25_index() -> dict[str, Any] | None:
    if not BM25_INDEX_PATH.exists():
        return None
    with open(BM25_INDEX_PATH, "rb") as f:
        return pickle.load(f)


def collection_count() -> int:
    return _get_collection().count()
