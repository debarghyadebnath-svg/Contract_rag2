import sqlite3
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).parent / "insurance_rag.db"


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database() -> None:
    """Create tables on first startup if they don't exist."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS policy_manuals (
                id                   INTEGER PRIMARY KEY AUTOINCREMENT,
                file_hash            TEXT    UNIQUE NOT NULL,
                filename             TEXT    NOT NULL,
                insurer              TEXT    NOT NULL,
                category             TEXT    NOT NULL,
                policy_name          TEXT    NOT NULL DEFAULT '',
                status               TEXT    NOT NULL DEFAULT 'indexing',
                vector_collection_id TEXT,
                created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text    TEXT    NOT NULL,
                response_text TEXT    NOT NULL,
                sources_used  TEXT,       -- JSON array of {manual_id, page_number, snippet}
                feedback      INTEGER,    -- 1 = thumbs up, -1 = thumbs down
                created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        existing_columns = {
            row["name"]
            for row in conn.execute("PRAGMA table_info(policy_manuals)").fetchall()
        }
        if "policy_name" not in existing_columns:
            conn.execute(
                "ALTER TABLE policy_manuals ADD COLUMN policy_name TEXT NOT NULL DEFAULT ''"
            )
        conn.commit()


def get_all_manuals() -> list[dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM policy_manuals ORDER BY created_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def insert_manual(
    file_hash: str,
    filename: str,
    insurer: str,
    category: str,
    policy_name: str,
    vector_collection_id: str,
) -> int:
    """Insert a new manual record and return its id."""
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO policy_manuals
               (file_hash, filename, insurer, category, policy_name, status, vector_collection_id)
               VALUES (?, ?, ?, ?, ?, 'indexing', ?)""",
            (file_hash, filename, insurer, category, policy_name, vector_collection_id),
        )
        conn.commit()
        return cursor.lastrowid


def find_manual_by_hash(file_hash: str) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM policy_manuals WHERE file_hash = ?", (file_hash,)
        ).fetchone()
    return dict(row) if row else None


def find_manual_by_id(manual_id: int) -> dict[str, Any] | None:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM policy_manuals WHERE id = ?", (manual_id,)
        ).fetchone()
    return dict(row) if row else None


def set_manual_status(manual_id: int, status: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE policy_manuals SET status = ? WHERE id = ?",
            (status, manual_id),
        )
        conn.commit()


def set_manual_policy_name(manual_id: int, policy_name: str) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE policy_manuals SET policy_name = ? WHERE id = ?",
            (policy_name, manual_id),
        )
        conn.commit()


def delete_manual_record(manual_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM policy_manuals WHERE id = ?", (manual_id,))
        conn.commit()


def insert_query_log(
    query_text: str,
    response_text: str,
    sources_used: str,  # pre-serialized JSON string
) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO query_logs (query_text, response_text, sources_used)
               VALUES (?, ?, ?)""",
            (query_text, response_text, sources_used),
        )
        conn.commit()
        return cursor.lastrowid


def set_query_feedback(log_id: int, feedback: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE query_logs SET feedback = ? WHERE id = ?",
            (feedback, log_id),
        )
        conn.commit()
