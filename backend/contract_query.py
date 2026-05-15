"""
contract_query.py
-----------------
Hybrid retrieval (Chroma dense + BM25 sparse) with Groq LLM answer generation.

Retrieval strategy:
  1. Dense: query Chroma with BGE embedding → top-K candidates
  2. Sparse: BM25 score all indexed documents → top-K candidates
  3. Merge with Reciprocal Rank Fusion (RRF) → final ranked list
  4. Feed top context into Groq (meta-llama/llama-4-scout) to generate answer

Uses ONLY the GROQ_API_KEY from .env — no other API keys needed at query time.
"""

from __future__ import annotations

import os
from typing import Any

from groq import Groq

from contract_embedder import QWEN_QUERY_PREFIX, _get_collection, _get_model, load_bm25_index

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

CONTRACT_SYSTEM_PROMPT = """\
You are a legal contract analysis assistant specializing in mortgage purchase agreements.
Your job is to give clear, precise answers about the IHDA Mortgage Purchase Agreement.

Rules:
1. Use ONLY the contract excerpts provided below to answer.
2. Cite specific section numbers and defined terms wherever possible.
3. Use **bold** for critical obligations, deadlines, and dollar amounts.
4. If a table is present in the context, interpret it numerically — do not just repeat it.
5. Always end with a ## Summary section that gives a one-sentence plain-English takeaway.
6. If the context is insufficient, state exactly what section or clause the user should check.

Contract Excerpts:
{context}
"""

# ---------------------------------------------------------------------------
# RRF (Reciprocal Rank Fusion) merge helper
# ---------------------------------------------------------------------------

def _rrf_merge(
    dense_ids: list[str],
    sparse_ids: list[str],
    k: int = 60,
) -> list[str]:
    """
    Combine two ranked lists using RRF score = Σ 1/(k + rank).
    Returns re-ranked list of ids.
    """
    scores: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    for rank, doc_id in enumerate(sparse_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda x: scores[x], reverse=True)


# ---------------------------------------------------------------------------
# Build context string from retrieved docs
# ---------------------------------------------------------------------------

def _build_context(
    merged_ids: list[str],
    id_to_doc: dict[str, dict[str, Any]],
    max_chunks: int = 8,
) -> str:
    parts: list[str] = []
    for i, doc_id in enumerate(merged_ids[:max_chunks], start=1):
        doc = id_to_doc.get(doc_id)
        if not doc:
            continue
        meta = doc.get("metadata", {}) or {}
        chunk_type = meta.get("chunk_type", "?")
        section = meta.get("section_number") or meta.get("section", "?")
        header = f"[Source {i} | type={chunk_type} | section={section} | id={doc_id}]"
        parts.append(f"{header}\n{doc['text']}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Main retrieval + generation
# ---------------------------------------------------------------------------

def retrieve_contract(
    query: str,
    top_k: int = 10,
    chunk_type_filter: str | None = None,
) -> list[dict[str, Any]]:
    """
    Returns a list of {id, text, metadata, rrf_score} dicts, ranked by RRF.
    Optional chunk_type_filter: 'definition' | 'section' | 'table'
    """
    model = _get_model()
    collection = _get_collection()

    if collection.count() == 0:
        return []

    # --- Dense retrieval via Chroma
    query_embedding = model.encode(
        QWEN_QUERY_PREFIX + query,
        normalize_embeddings=True,
    )

    where_filter: dict[str, Any] | None = None
    if chunk_type_filter:
        where_filter = {"chunk_type": {"$eq": chunk_type_filter}}

    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        where=where_filter,
        include=["documents", "metadatas", "distances"],
    )
    dense_ids: list[str] = dense_results["ids"][0] if dense_results["ids"] else []
    dense_docs_text: list[str] = dense_results["documents"][0] if dense_results["documents"] else []
    dense_metas: list[dict] = dense_results["metadatas"][0] if dense_results["metadatas"] else []

    id_to_doc: dict[str, dict[str, Any]] = {
        doc_id: {"text": text, "metadata": meta}
        for doc_id, text, meta in zip(dense_ids, dense_docs_text, dense_metas)
    }

    # --- Sparse retrieval via BM25
    bm25_data = load_bm25_index()
    sparse_ids: list[str] = []
    if bm25_data:
        tokenized_query = query.lower().split()
        scores = bm25_data["bm25"].get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for idx in ranked_indices[:top_k]:
            sid = bm25_data["ids"][idx]
            sparse_ids.append(sid)
            if sid not in id_to_doc:
                id_to_doc[sid] = {
                    "text": bm25_data["texts"][idx],
                    "metadata": {},
                }

    # --- RRF merge
    merged = _rrf_merge(dense_ids, sparse_ids)

    return [
        {
            "id": doc_id,
            "text": id_to_doc[doc_id]["text"],
            "metadata": id_to_doc[doc_id]["metadata"],
        }
        for doc_id in merged
        if doc_id in id_to_doc
    ]


def answer_contract_query(
    query: str,
    top_k: int = 10,
    chunk_type_filter: str | None = None,
) -> dict[str, Any]:
    """
    Full RAG pipeline: retrieve → build context → Groq LLM → return answer.
    """
    retrieved = retrieve_contract(query, top_k=top_k, chunk_type_filter=chunk_type_filter)

    if not retrieved:
        return {
            "answer": (
                "No contract content has been indexed yet. "
                "Run `ingest('contract.pdf')` first."
            ),
            "sources": [],
        }

    id_to_doc = {r["id"]: {"text": r["text"], "metadata": r["metadata"]} for r in retrieved}
    merged_ids = [r["id"] for r in retrieved]
    context = _build_context(merged_ids, id_to_doc)

    groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])

    completion = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": CONTRACT_SYSTEM_PROMPT.format(context=context)},
            {"role": "user", "content": query},
        ],
        temperature=0.2,
        max_tokens=2048,
    )

    answer = completion.choices[0].message.content

    sources = [
        {
            "chunk_id": r["id"],
            "chunk_type": r["metadata"].get("chunk_type", "?"),
            "section": r["metadata"].get("section_number") or r["metadata"].get("section", "?"),
            "snippet": r["text"][:300],
        }
        for r in retrieved[:8]
    ]

    return {"answer": answer, "sources": sources}
