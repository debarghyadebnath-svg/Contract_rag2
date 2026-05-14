import hashlib
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

from fastapi import FastAPI, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import database
import embedder
import pdf_parser
import query_engine

UPLOADS_DIR = Path(__file__).parent / "uploads"
UPLOADS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Insurance RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

database.initialize_database()


# ---------------------------------------------------------------------------
# Background task: parse → embed → mark active
# ---------------------------------------------------------------------------

def _process_pdf(
    pdf_path: Path,
    manual_id: int,
    insurer: str,
    category: str,
    filename: str,
    policy_name: str,
) -> None:
    try:
        pages = pdf_parser.extract_pages(pdf_path)
        embedder.index_pdf_pages(
            pages,
            manual_id,
            insurer,
            category,
            filename,
            policy_name,
        )
        database.set_manual_status(manual_id, "active")
    except Exception:
        database.set_manual_status(manual_id, "failed")
        raise


def _reconcile_stale_manuals() -> None:
    for manual in database.get_all_manuals():
        if manual["status"] not in {"indexing", "failed"}:
            continue

        pdf_path = UPLOADS_DIR / f'{manual["id"]}_{manual["filename"]}'
        if not pdf_path.exists():
            database.delete_manual_record(manual["id"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    manual_ids: list[int] | None = None


class FeedbackRequest(BaseModel):
    log_id: int
    feedback: int  # 1 = thumbs up, -1 = thumbs down


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/manuals")
def list_manuals() -> list[dict[str, Any]]:
    _reconcile_stale_manuals()
    return database.get_all_manuals()


@app.post("/api/manuals/upload")
def upload_manual(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    insurer: str = Form(...),
    category: str = Form(...),
) -> dict[str, Any]:
    content = file.file.read()
    file_hash = hashlib.sha256(content).hexdigest()
    policy_name = pdf_parser.normalize_policy_name(file.filename)

    if database.find_manual_by_hash(file_hash):
        raise HTTPException(status_code=409, detail="This document is already indexed.")

    manual_id = database.insert_manual(
        file_hash=file_hash,
        filename=file.filename,
        insurer=insurer,
        category=category,
        policy_name=policy_name,
        vector_collection_id=embedder.COLLECTION_NAME,
    )

    pdf_path = UPLOADS_DIR / f"{manual_id}_{file.filename}"
    pdf_path.write_bytes(content)

    background_tasks.add_task(
        _process_pdf,
        pdf_path,
        manual_id,
        insurer,
        category,
        file.filename,
        policy_name,
    )

    return {
        "id": manual_id,
        "status": "indexing",
        "filename": file.filename,
        "policy_name": policy_name,
    }


@app.get("/api/manuals/{manual_id}")
def get_manual(manual_id: int) -> dict[str, Any]:
    manual = database.find_manual_by_id(manual_id)
    if not manual:
        raise HTTPException(status_code=404, detail="Manual not found.")
    return manual


@app.delete("/api/manuals/{manual_id}")
def delete_manual(manual_id: int) -> dict[str, Any]:
    manual = database.find_manual_by_id(manual_id)
    if not manual:
        raise HTTPException(status_code=404, detail="Manual not found.")

    embedder.delete_manual_vectors(manual_id)

    for f in UPLOADS_DIR.glob(f"{manual_id}_*"):
        f.unlink(missing_ok=True)

    database.delete_manual_record(manual_id)
    return {"deleted": manual_id}


@app.post("/api/query")
def query_manuals(req: QueryRequest) -> dict[str, Any]:
    result = query_engine.answer_query(req.query, req.manual_ids)
    log_id = database.insert_query_log(
        query_text=req.query,
        response_text=result["answer"],
        sources_used=json.dumps(result["sources"]),
    )
    return {"log_id": log_id, "answer": result["answer"], "sources": result["sources"]}


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest) -> dict[str, Any]:
    database.set_query_feedback(req.log_id, req.feedback)
    return {"ok": True}
