
"""
contract_embedder.py
--------------------
Embeds contract chunks using Ollama's local embedding API (qwen3-embedding:0.6b),
stores dense vectors in a local Chroma persistent store, and builds a BM25
sparse index in-memory (serialised to disk for reuse).

Why Ollama for embeddings?
  Ollama runs as a separate process, completely bypassing the Windows
  Intel MKL / OMP thread-pool deadlock that affects SentenceTransformers
  and raw HuggingFace inference on CPU.

Why local Chroma (not the existing Qdrant)?
  The user said "use the current Qdrant db only" for the existing insurance RAG.
  For the contract RAG we use a separate local Chroma store so neither system
  pollutes the other's collection.

Ollama Embed API:
  POST http://localhost:11434/api/embed
  Body: {"model": "qwen3-embedding:0.6b", "input": ["text1", "text2", ...]}
  Response: {"embeddings": [[...], [...], ...]}

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
import requests
from chromadb import PersistentClient
from rank_bm25 import BM25Okapi

from contract_chunker import Chunk

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CHROMA_DIR = Path(__file__).parent / "chroma_contract_store"
CHROMA_COLLECTION_NAME = "ihda_contract"

# Ollama settings — model must be pulled first: `ollama pull qwen3-embedding:0.6b`
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL_NAME = "qwen3-embedding:0.6b"

BM25_INDEX_PATH = Path(__file__).parent / "bm25_contract_index.pkl"

# Batch size for Ollama embedding requests (tune up/down based on RAM)
OLLAMA_BATCH_SIZE = 32

# Prefix required by Qwen2/3 embedding models for queries
QWEN_QUERY_PREFIX = "Represent this query for retrieving relevant documents: "


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
        # We provide embeddings manually, so no embedding_function needed
        _collection = _chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _get_model():
    """
    Returns a wrapper that mimics SentenceTransformer.encode() 
    but sends requests to the local Ollama API.
    """
    class OllamaEncoder:
        def encode(self, texts: str | list[str], normalize_embeddings: bool = True):
            if isinstance(texts, str):
                return _embed_batch([texts])[0]
            return _encode_texts(texts)
    return OllamaEncoder()


# ---------------------------------------------------------------------------
# Ollama embedding helper
# ---------------------------------------------------------------------------

def _check_ollama() -> None:
    """Raise a clear error if the Ollama server is not running."""
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        r.raise_for_status()
    except requests.ConnectionError:
        raise RuntimeError(
            "[embedder] Cannot reach Ollama at http://localhost:11434. "
            "Make sure Ollama is running (`ollama serve`) and the model is "
            f"pulled (`ollama pull {EMBED_MODEL_NAME}`)."
        )


def _embed_batch(texts: list[str]) -> list[list[float]]:
    """Send a batch of texts to Ollama and return their embeddings."""
    payload = {"model": EMBED_MODEL_NAME, "input": texts}
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json=payload,
        timeout=600,
    )
    response.raise_for_status()
    return response.json()["embeddings"]


def _encode_texts(texts: list[str]) -> list[list[float]]:
    """
    Encode all texts via Ollama in batches, printing progress.
    Returns a flat list of embedding vectors.
    """
    _check_ollama()
    all_embeddings: list[list[float]] = []
    total = len(texts)
    for start in range(0, total, OLLAMA_BATCH_SIZE):
        batch = texts[start : start + OLLAMA_BATCH_SIZE]
        embeddings = _embed_batch(batch)
        all_embeddings.extend(embeddings)
        done = min(start + OLLAMA_BATCH_SIZE, total)
        print(f"  [embedder] {done}/{total} chunks encoded …", flush=True)
    return all_embeddings


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
    Embed all chunks via Ollama, upsert into Chroma, build BM25 index.
    Safe to call multiple times (upsert is idempotent on chunk_id).
    """
    if not chunks:
        print("[embedder] No chunks to index.")
        return

    collection = _get_collection()

    texts = [c.page_content for c in chunks]
    ids = [c.metadata["chunk_id"] for c in chunks]
    metadatas = [_flatten_metadata(c.metadata) for c in chunks]

    print(f"[embedder] Encoding {len(texts)} chunks with Ollama/{EMBED_MODEL_NAME} …")
    embeddings = _encode_texts(texts)

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
