from __future__ import annotations

import json
import sqlite3
from typing import Any

from app.config import settings
from app.models import JobStatus


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with get_conn() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                output_dir TEXT NOT NULL,
                voice TEXT NOT NULL,
                speed REAL NOT NULL,
                use_gpu INTEGER NOT NULL,
                status TEXT NOT NULL,
                progress REAL NOT NULL,
                error TEXT,
                meta_json TEXT NOT NULL
            )
            """
        )
        conn.commit()


def insert_job(
    job_id: str,
    source_path: str,
    output_dir: str,
    voice: str,
    speed: float,
    use_gpu: bool,
) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, source_path, output_dir, voice, speed, use_gpu, status, progress, error, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                source_path,
                output_dir,
                voice,
                speed,
                int(use_gpu),
                JobStatus.PENDING.value,
                0.0,
                None,
                "{}",
            ),
        )
        conn.commit()


def update_job(
    job_id: str,
    *,
    status: JobStatus | None = None,
    progress: float | None = None,
    error: str | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    fields: list[str] = []
    values: list[Any] = []
    if status is not None:
        fields.append("status = ?")
        values.append(status.value)
    if progress is not None:
        fields.append("progress = ?")
        values.append(progress)
    if error is not None:
        fields.append("error = ?")
        values.append(error)
    if meta is not None:
        fields.append("meta_json = ?")
        values.append(json.dumps(meta, ensure_ascii=False))
    if not fields:
        return

    values.append(job_id)
    with get_conn() as conn:
        conn.execute(f"UPDATE jobs SET {', '.join(fields)} WHERE id = ?", values)
        conn.commit()


def get_job(job_id: str) -> dict[str, Any] | None:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    if row is None:
        return None
    item = dict(row)
    item["meta"] = json.loads(item.pop("meta_json") or "{}")
    return item


def list_jobs() -> list[dict[str, Any]]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM jobs ORDER BY rowid DESC").fetchall()
    result: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["meta"] = json.loads(item.pop("meta_json") or "{}")
        result.append(item)
    return result


def clear_jobs() -> int:
    """Delete all jobs from DB. Returns number of deleted rows."""
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM jobs")
        conn.commit()
        return cur.rowcount
