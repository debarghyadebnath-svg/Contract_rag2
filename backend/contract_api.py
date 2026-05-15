"""
contract_api.py
---------------
FastAPI server that exposes the contract RAG backend to the Next.js frontend.

Endpoints:
  GET  /api/health              → health check
  GET  /api/documents           → list indexed documents (from Chroma metadata)
  POST /api/upload              → upload + ingest a PDF contract (background task)
  POST /api/query               → hybrid RAG query → answer + sources
  POST /api/feedback            → thumbs up/down on a logged answer

Run with:
  uvicorn contract_api:app --reload --port 8000

Frontend should proxy to:
  http://localhost:8000
"""

from __future__ import annotations

import os
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, BackgroundTasks, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from contract_query import answer_contract_query
from contract_embedder import _get_collection
from ingest import ingest

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

UPLOADS_DIR = Path(__file__).parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

# Simple in-memory store for upload status + feedback logs
# In production you'd use a DB; this is fine for a local demo.
_upload_status: dict[str, dict[str, Any]] = {}   # doc_id → status info
_query_log: dict[int, dict[str, Any]] = {}        # log_id → query+answer+feedback
_log_counter = 0

app = FastAPI(title="Contract RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    chunk_type_filter: str | None = None   # 'section' | 'table' | 'definition'


class FeedbackRequest(BaseModel):
    log_id: int
    feedback: int   # 1 = thumbs up, -1 = thumbs down


# ---------------------------------------------------------------------------
# Background ingestion task
# ---------------------------------------------------------------------------

def _run_ingestion(doc_id: str, pdf_path: Path) -> None:
    try:
        _upload_status[doc_id]["status"] = "indexing"
        ingest(pdf_path, use_cloud=True)
        _upload_status[doc_id]["status"] = "active"
    except Exception as exc:
        _upload_status[doc_id]["status"] = "failed"
        _upload_status[doc_id]["error"] = str(exc)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/documents")
def list_documents() -> list[dict[str, Any]]:
    """
    Returns the list of documents that have been uploaded via the API,
    plus their indexing status.
    """
    docs = []
    for doc_id, info in _upload_status.items():
        docs.append({
            "id":       doc_id,
            "filename": info.get("filename", "unknown"),
            "status":   info.get("status", "unknown"),
            "error":    info.get("error"),
        })

    # Also include a synthetic "active" entry for docs already pre-indexed
    # via the CLI (i.e., the manual we ingested before starting the API)
    try:
        collection = _get_collection()
        count = collection.count()
        if count > 0 and not _upload_status:
            docs.append({
                "id":       "pre-indexed",
                "filename": "Pre-indexed contract manual",
                "status":   "active",
                "chunks":   count,
                "error":    None,
            })
    except Exception:
        pass

    return docs


@app.post("/api/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Upload a PDF contract and ingest it in the background."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    doc_id = str(uuid.uuid4())
    save_path = UPLOADS_DIR / f"{doc_id}_{file.filename}"
    content = await file.read()
    save_path.write_bytes(content)

    _upload_status[doc_id] = {
        "filename": file.filename,
        "status":   "queued",
        "error":    None,
    }

    background_tasks.add_task(_run_ingestion, doc_id, save_path)

    return {
        "id":       doc_id,
        "filename": file.filename,
        "status":   "queued",
    }


@app.get("/api/documents/{doc_id}")
def get_document_status(doc_id: str) -> dict[str, Any]:
    if doc_id not in _upload_status:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"id": doc_id, **_upload_status[doc_id]}


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: str) -> dict[str, Any]:
    if doc_id not in _upload_status:
        raise HTTPException(status_code=404, detail="Document not found.")

    # Remove the uploaded file
    filename = _upload_status[doc_id].get("filename", "")
    for f in UPLOADS_DIR.glob(f"{doc_id}_*"):
        f.unlink(missing_ok=True)

    del _upload_status[doc_id]
    return {"deleted": doc_id}


@app.post("/api/query")
def query_contract(req: QueryRequest) -> dict[str, Any]:
    """Run hybrid RAG and return an LLM answer with cited sources."""
    global _log_counter
    try:
        result = answer_contract_query(
            query=req.query,
            chunk_type_filter=req.chunk_type_filter,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    _log_counter += 1
    log_id = _log_counter
    _query_log[log_id] = {
        "query":    req.query,
        "answer":   result["answer"],
        "sources":  result["sources"],
        "feedback": None,
    }

    return {
        "log_id":  log_id,
        "answer":  result["answer"],
        "sources": result["sources"],
    }


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest) -> dict[str, Any]:
    if req.log_id not in _query_log:
        raise HTTPException(status_code=404, detail="Log entry not found.")
    _query_log[req.log_id]["feedback"] = req.feedback
    return {"ok": True}
