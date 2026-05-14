"""
contract_embedder.py
--------------------
Embeds contract chunks with BAAI/bge-large-en-v1.5 via SentenceTransformers,
stores dense vectors in a local Chroma persistent store, and builds a BM25
sparse index in-memory (serialised to disk for reuse).

Why local Chroma (not the existing Qdrant)?
  The user said "use the current Qdrant db only" for the existing insurance RAG.
  For the contract RAG we use a separate local Chroma store so neither system
  pollutes the other's collection.

Chroma API pattern (from docs):
  client = chromadb.PersistentClient(path=...)
  collection = client.get_or_create_collection(name, embedding_function=...)
  collection.upsert(ids=[...], documents=[...], metadatas=[...])
  collection.query(query_embeddings=[[...]], n_results=N, where={...})

BM25 pattern (from rank-bm25 docs):
  from rank_bm25 import BM25Okapi
  tokenized_corpus = [doc.lower().split() for doc in documents]
  bm25 = BM25Okapi(tokenized_corpus)
  scores = bm25.get_scores(query.lower().split())
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import chromadb
from chromadb import PersistentClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from contract_chunker import Chunk

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_DIR = Path(__file__).parent / "chroma_contract_store"
CHROMA_COLLECTION_NAME = "ihda_contract"
BGE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
BM25_INDEX_PATH = Path(__file__).parent / "bm25_contract_index.pkl"

# BGE models expect an instruction prefix for asymmetric retrieval.
# For passage indexing: no prefix.  For query encoding: "Represent this sentence: "
BGE_QUERY_PREFIX = "Represent this sentence: "


# ---------------------------------------------------------------------------
# Lazy singletons — model and client created once per process
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None
_chroma_client: PersistentClient | None = None
_collection: chromadb.Collection | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"[embedder] Loading {BGE_MODEL_NAME} …")
        _model = SentenceTransformer(BGE_MODEL_NAME)
    return _model


def _get_collection() -> chromadb.Collection:
    global _chroma_client, _collection
    if _collection is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        # We provide embeddings manually, so embedding_function=None
        _collection = _chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ---------------------------------------------------------------------------
# Serialise metadata for Chroma (values must be str | int | float | bool)
# ---------------------------------------------------------------------------

def _flatten_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for k, v in meta.items():
        if isinstance(v, (str, int, float, bool)):
            flat[k] = v
        elif isinstance(v, list):
            flat[k] = json.dumps(v)  # Chroma doesn't accept lists
        else:
            flat[k] = str(v)
    return flat


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_and_index(chunks: list[Chunk]) -> None:
    """
    Embed all chunks with BGE-large, upsert into Chroma, build BM25 index.
    Safe to call multiple times (upsert is idempotent on chunk_id).
    """
    if not chunks:
        print("[embedder] No chunks to index.")
        return

    model = _get_model()
    collection = _get_collection()

    texts = [c.page_content for c in chunks]
    ids = [c.metadata["chunk_id"] for c in chunks]
    metadatas = [_flatten_metadata(c.metadata) for c in chunks]

    print(f"[embedder] Encoding {len(texts)} chunks with {BGE_MODEL_NAME} …")
    # show_progress_bar=True gives a tqdm progress bar during batch encoding
    embeddings: list[list[float]] = model.encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity via dot product
    ).tolist()

    # Chroma upsert handles duplicates gracefully (update in place)
    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    print(f"[embedder] Upserted {len(ids)} vectors into Chroma collection '{CHROMA_COLLECTION_NAME}'")

    # Build and persist BM25 index
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
