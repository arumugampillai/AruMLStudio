"""SQLite persistence for prediction exec jobs / workers / checkpoints."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

from .prediction_job_schema import (
    DAY_CP_CANCELLED,
    DAY_CP_COMPLETED,
    DAY_CP_FAILED,
    DAY_CP_IN_PROGRESS,
    DAY_CP_PENDING,
    JOB_STATUS_CANCELLED,
    JOB_STATUS_COMPLETED,
    JOB_STATUS_FAILED,
    JOB_STATUS_PAUSED,
    JOB_STATUS_PENDING,
    JOB_STATUS_RUNNING,
    WORKER_STATUS_DONE,
    WORKER_STATUS_FAILED,
    WORKER_STATUS_IDLE,
    WORKER_STATUS_RUNNING,
    WORKER_STATUS_WAITING,
    create_prediction_exec_tables_sql,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


_SCHEMA_READY: set[str] = set()


def _connect(lab_db_path: str, *, timeout_sec: float = 30.0) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(lab_db_path) or ".", exist_ok=True)
    timeout_sec = max(0.05, float(timeout_sec))
    conn = sqlite3.connect(lab_db_path, timeout=timeout_sec)
    conn.execute(f"PRAGMA busy_timeout={int(timeout_sec * 1000)}")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def ensure_exec_schema(lab_db_path: str, *, timeout_sec: float = 30.0) -> None:
    key = os.path.abspath(lab_db_path)
    if key in _SCHEMA_READY:
        return
    conn = _connect(lab_db_path, timeout_sec=timeout_sec)
    try:
        conn.executescript(create_prediction_exec_tables_sql())
        conn.commit()
        _SCHEMA_READY.add(key)
    finally:
        conn.close()


def create_job(
    lab_db_path: str,
    *,
    job_id: str,
    lab_uuid: str,
    data_dir: str,
    worker_count: int,
    config: dict[str, Any],
    day_assignments: list[tuple[str, int]],
    overwrite: bool = False,
    resume: bool = True,
    row_limit: int | None = None,
    mark_day_complete: bool = True,
) -> dict[str, Any]:
    """
    Create job + worker rows + checkpoint rows.

    ``day_assignments`` is ``[(trading_day, worker_id), ...]`` — each day owned
    by exactly one worker (never split).
    """
    ensure_exec_schema(lab_db_path)
    now = _utc_now()
    conn = _connect(lab_db_path)
    try:
        conn.execute(
            """
            INSERT INTO prediction_exec_job (
                job_id, lab_uuid, lab_db_path, data_dir, worker_count, status,
                overwrite, resume, row_limit, mark_day_complete, config_json,
                days_total, days_completed, created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,0,?,?)
            """,
            (
                job_id,
                lab_uuid,
                os.path.abspath(lab_db_path),
                data_dir,
                int(worker_count),
                JOB_STATUS_PENDING,
                1 if overwrite else 0,
                1 if resume else 0,
                row_limit,
                1 if mark_day_complete else 0,
                json.dumps(config, ensure_ascii=False),
                len(day_assignments),
                now,
                now,
            ),
        )
        for wid in range(1, int(worker_count) + 1):
            conn.execute(
                """
                INSERT INTO prediction_exec_worker (
                    job_id, worker_id, status, updated_at
                ) VALUES (?,?,?,?)
                """,
                (job_id, wid, WORKER_STATUS_IDLE, now),
            )
        for day, wid in day_assignments:
            conn.execute(
                """
                INSERT INTO prediction_exec_checkpoint (
                    job_id, trading_day, worker_id, status, rows_committed, updated_at
                ) VALUES (?,?,?,?,0,?)
                """,
                (job_id, str(day), int(wid), DAY_CP_PENDING, now),
            )
        conn.commit()
    finally:
        conn.close()
    return get_job(lab_db_path, job_id) or {}


def get_job(lab_db_path: str, job_id: str, *, timeout_sec: float = 30.0) -> dict[str, Any] | None:
    ensure_exec_schema(lab_db_path, timeout_sec=timeout_sec)
    conn = _connect(lab_db_path, timeout_sec=timeout_sec)
    try:
        row = conn.execute(
            """
            SELECT job_id, lab_uuid, lab_db_path, data_dir, worker_count, status,
                   overwrite, resume, row_limit, mark_day_complete, config_json,
                   days_total, days_completed, error_message,
                   created_at, started_at, finished_at, updated_at
            FROM prediction_exec_job WHERE job_id = ?
            """,
            (job_id,),
        ).fetchone()
        if not row:
            return None
        cfg: dict[str, Any] = {}
        try:
            loaded = json.loads(str(row[10] or "{}"))
            if isinstance(loaded, dict):
                cfg = loaded
        except (TypeError, json.JSONDecodeError):
            cfg = {}
        return {
            "job_id": row[0],
            "lab_uuid": row[1],
            "lab_db_path": row[2],
            "data_dir": row[3],
            "worker_count": int(row[4] or 1),
            "status": row[5],
            "overwrite": bool(row[6]),
            "resume": bool(row[7]),
            "row_limit": row[8],
            "mark_day_complete": bool(row[9]),
            "config": cfg,
            "days_total": int(row[11] or 0),
            "days_completed": int(row[12] or 0),
            "error_message": row[13],
            "created_at": row[14],
            "started_at": row[15],
            "finished_at": row[16],
            "updated_at": row[17],
        }
    finally:
        conn.close()


def list_active_jobs(lab_db_path: str) -> list[dict[str, Any]]:
    ensure_exec_schema(lab_db_path)
    conn = _connect(lab_db_path)
    try:
        rows = conn.execute(
            """
            SELECT job_id FROM prediction_exec_job
            WHERE status IN ('pending', 'running', 'paused')
            ORDER BY created_at DESC
            """
        ).fetchall()
    finally:
        conn.close()
    out: list[dict[str, Any]] = []
    for (jid,) in rows:
        job = get_job(lab_db_path, str(jid))
        if job:
            out.append(job)
    return out


def set_job_status(
    lab_db_path: str,
    job_id: str,
    status: str,
    *,
    error_message: str | None = None,
    started: bool = False,
    finished: bool = False,
) -> None:
    ensure_exec_schema(lab_db_path)
    now = _utc_now()
    sets = ["status = ?", "updated_at = ?"]
    args: list[Any] = [status, now]
    if error_message is not None:
        sets.append("error_message = ?")
        args.append(error_message)
    if started:
        sets.append("started_at = COALESCE(started_at, ?)")
        args.append(now)
    if finished:
        sets.append("finished_at = ?")
        args.append(now)
    args.append(job_id)
    conn = _connect(lab_db_path)
    try:
        conn.execute(
            f"UPDATE prediction_exec_job SET {', '.join(sets)} WHERE job_id = ?",
            args,
        )
        conn.commit()
    finally:
        conn.close()


def request_job_pause(lab_db_path: str, job_id: str) -> None:
    set_job_status(lab_db_path, job_id, JOB_STATUS_PAUSED)


def request_job_cancel(lab_db_path: str, job_id: str) -> None:
    set_job_status(lab_db_path, job_id, JOB_STATUS_CANCELLED)


def request_job_resume(lab_db_path: str, job_id: str) -> None:
    set_job_status(lab_db_path, job_id, JOB_STATUS_RUNNING, started=True)


def refresh_job_day_counts(lab_db_path: str, job_id: str) -> None:
    conn = _connect(lab_db_path)
    try:
        row = conn.execute(
            """
            SELECT COUNT(*) FROM prediction_exec_checkpoint
            WHERE job_id = ? AND status = ?
            """,
            (job_id, DAY_CP_COMPLETED),
        ).fetchone()
        done = int(row[0] or 0) if row else 0
        conn.execute(
            """
            UPDATE prediction_exec_job
            SET days_completed = ?, updated_at = ?
            WHERE job_id = ?
            """,
            (done, _utc_now(), job_id),
        )
        conn.commit()
    finally:
        conn.close()


def list_workers(lab_db_path: str, job_id: str, *, timeout_sec: float = 30.0) -> list[dict[str, Any]]:
    ensure_exec_schema(lab_db_path, timeout_sec=timeout_sec)
    conn = _connect(lab_db_path, timeout_sec=timeout_sec)
    try:
        rows = conn.execute(
            """
            SELECT job_id, worker_id, pid, assigned_day, current_row, total_rows,
                   percent, eta_sec, status, log_path, last_message,
                   heartbeat_at, started_at, finished_at, updated_at
            FROM prediction_exec_worker
            WHERE job_id = ?
            ORDER BY worker_id
            """,
            (job_id,),
        ).fetchall()
        return [
            {
                "job_id": r[0],
                "worker_id": int(r[1]),
                "pid": r[2],
                "assigned_day": r[3],
                "current_row": int(r[4] or 0),
                "total_rows": int(r[5] or 0),
                "percent": float(r[6] or 0),
                "eta_sec": r[7],
                "status": r[8],
                "log_path": r[9],
                "last_message": r[10],
                "heartbeat_at": r[11],
                "started_at": r[12],
                "finished_at": r[13],
                "updated_at": r[14],
            }
            for r in rows
        ]
    finally:
        conn.close()


def update_worker(
    lab_db_path: str,
    job_id: str,
    worker_id: int,
    *,
    pid: int | None = None,
    assigned_day: str | None = None,
    current_row: int | None = None,
    total_rows: int | None = None,
    percent: float | None = None,
    eta_sec: float | None = None,
    status: str | None = None,
    log_path: str | None = None,
    last_message: str | None = None,
    heartbeat: bool = True,
    started: bool = False,
    finished: bool = False,
) -> None:
    ensure_exec_schema(lab_db_path)
    now = _utc_now()
    sets = ["updated_at = ?"]
    args: list[Any] = [now]
    if pid is not None:
        sets.append("pid = ?")
        args.append(int(pid))
    if assigned_day is not None:
        sets.append("assigned_day = ?")
        args.append(assigned_day or None)
    if current_row is not None:
        sets.append("current_row = ?")
        args.append(int(current_row))
    if total_rows is not None:
        sets.append("total_rows = ?")
        args.append(int(total_rows))
    if percent is not None:
        sets.append("percent = ?")
        args.append(float(percent))
    if eta_sec is not None:
        sets.append("eta_sec = ?")
        args.append(float(eta_sec) if eta_sec is not None else None)
    if status is not None:
        sets.append("status = ?")
        args.append(status)
    if log_path is not None:
        sets.append("log_path = ?")
        args.append(log_path)
    if last_message is not None:
        sets.append("last_message = ?")
        args.append(last_message)
    if heartbeat:
        sets.append("heartbeat_at = ?")
        args.append(now)
    if started:
        sets.append("started_at = COALESCE(started_at, ?)")
        args.append(now)
    if finished:
        sets.append("finished_at = ?")
        args.append(now)
    args.extend([job_id, int(worker_id)])
    conn = _connect(lab_db_path)
    try:
        conn.execute(
            f"""
            UPDATE prediction_exec_worker
            SET {', '.join(sets)}
            WHERE job_id = ? AND worker_id = ?
            """,
            args,
        )
        conn.commit()
    finally:
        conn.close()


def list_checkpoints(
    lab_db_path: str, job_id: str, *, timeout_sec: float = 30.0
) -> list[dict[str, Any]]:
    ensure_exec_schema(lab_db_path, timeout_sec=timeout_sec)
    conn = _connect(lab_db_path, timeout_sec=timeout_sec)
    try:
        rows = conn.execute(
            """
            SELECT job_id, trading_day, worker_id, status, rows_committed,
                   rows_expected, error_message, started_at, finished_at, updated_at
            FROM prediction_exec_checkpoint
            WHERE job_id = ?
            ORDER BY trading_day
            """,
            (job_id,),
        ).fetchall()
        return [
            {
                "job_id": r[0],
                "trading_day": r[1],
                "worker_id": r[2],
                "status": r[3],
                "rows_committed": int(r[4] or 0),
                "rows_expected": r[5],
                "error_message": r[6],
                "started_at": r[7],
                "finished_at": r[8],
                "updated_at": r[9],
            }
            for r in rows
        ]
    finally:
        conn.close()


def worker_pending_days(
    lab_db_path: str,
    job_id: str,
    worker_id: int,
) -> list[dict[str, Any]]:
    """Days owned by this worker still eligible to run (resume-safe).

    Excludes COMPLETED, CANCELLED, and FAILED so a hard failure cannot
    spin the worker in an infinite retry loop.
    """
    ensure_exec_schema(lab_db_path)
    conn = _connect(lab_db_path)
    try:
        rows = conn.execute(
            """
            SELECT trading_day, status, rows_committed, rows_expected
            FROM prediction_exec_checkpoint
            WHERE job_id = ? AND worker_id = ?
              AND status NOT IN (?, ?, ?)
            ORDER BY trading_day
            """,
            (
                job_id,
                int(worker_id),
                DAY_CP_COMPLETED,
                DAY_CP_CANCELLED,
                DAY_CP_FAILED,
            ),
        ).fetchall()
        return [
            {
                "trading_day": r[0],
                "status": r[1],
                "rows_committed": int(r[2] or 0),
                "rows_expected": r[3],
            }
            for r in rows
        ]
    finally:
        conn.close()


def update_checkpoint(
    lab_db_path: str,
    job_id: str,
    trading_day: str,
    *,
    status: str | None = None,
    rows_committed: int | None = None,
    rows_expected: int | None = None,
    error_message: str | None = None,
    started: bool = False,
    finished: bool = False,
) -> None:
    ensure_exec_schema(lab_db_path)
    now = _utc_now()
    sets = ["updated_at = ?"]
    args: list[Any] = [now]
    if status is not None:
        sets.append("status = ?")
        args.append(status)
    if rows_committed is not None:
        sets.append("rows_committed = ?")
        args.append(int(rows_committed))
    if rows_expected is not None:
        sets.append("rows_expected = ?")
        args.append(int(rows_expected))
    if error_message is not None:
        sets.append("error_message = ?")
        args.append(error_message)
    if started:
        sets.append("started_at = COALESCE(started_at, ?)")
        args.append(now)
    if finished:
        sets.append("finished_at = ?")
        args.append(now)
    args.extend([job_id, trading_day])
    conn = _connect(lab_db_path)
    try:
        conn.execute(
            f"""
            UPDATE prediction_exec_checkpoint
            SET {', '.join(sets)}
            WHERE job_id = ? AND trading_day = ?
            """,
            args,
        )
        conn.commit()
    finally:
        conn.close()


def assign_days_round_robin(
    days: list[str],
    worker_count: int,
) -> list[tuple[str, int]]:
    """Each day owned by exactly one worker (round-robin). Never splits a day."""
    n = max(1, int(worker_count))
    out: list[tuple[str, int]] = []
    for i, day in enumerate(days):
        out.append((str(day), (i % n) + 1))
    return out


def job_progress_snapshot(
    lab_db_path: str,
    job_id: str,
    *,
    timeout_sec: float = 2.0,
) -> dict[str, Any]:
    """Lightweight job progress. Use short timeout_sec from the GUI poll path."""
    try:
        job = get_job(lab_db_path, job_id, timeout_sec=timeout_sec)
    except sqlite3.OperationalError as exc:
        return {"available": False, "error": f"database busy: {exc}", "status": "running"}
    if not job:
        return {"available": False, "error": "job not found"}
    try:
        workers = list_workers(lab_db_path, job_id, timeout_sec=timeout_sec)
        checkpoints = list_checkpoints(lab_db_path, job_id, timeout_sec=timeout_sec)
    except sqlite3.OperationalError as exc:
        return {
            "available": True,
            "job": job,
            "workers": [],
            "checkpoints": [],
            "days_completed": int(job.get("days_completed") or 0),
            "days_failed": 0,
            "days_total": int(job.get("days_total") or 0),
            "percent": 0.0,
            "current_day": None,
            "status": job.get("status"),
            "error": f"database busy: {exc}",
        }
    done = sum(1 for c in checkpoints if c.get("status") == DAY_CP_COMPLETED)
    failed = sum(1 for c in checkpoints if c.get("status") == DAY_CP_FAILED)
    running_day = next(
        (w.get("assigned_day") for w in workers if w.get("status") == WORKER_STATUS_RUNNING),
        None,
    )
    total = int(job.get("days_total") or len(checkpoints) or 0)
    # Prefer day-progress from the active worker while a day is in flight
    worker_pct = None
    for w in workers:
        if w.get("status") == WORKER_STATUS_RUNNING and w.get("percent") is not None:
            worker_pct = float(w.get("percent") or 0.0)
            break
    if total > 0 and worker_pct is not None:
        pct = (100.0 * done / total) + (worker_pct / max(total, 1))
        pct = min(99.9, pct)
    else:
        pct = (100.0 * done / total) if total else 0.0
    return {
        "available": True,
        "job": job,
        "workers": workers,
        "checkpoints": checkpoints,
        "days_completed": done,
        "days_failed": failed,
        "days_total": total,
        "percent": pct,
        "current_day": running_day,
        "status": job.get("status"),
    }


__all__ = [
    "JOB_STATUS_PENDING",
    "JOB_STATUS_RUNNING",
    "JOB_STATUS_PAUSED",
    "JOB_STATUS_CANCELLED",
    "JOB_STATUS_COMPLETED",
    "JOB_STATUS_FAILED",
    "WORKER_STATUS_IDLE",
    "WORKER_STATUS_RUNNING",
    "WORKER_STATUS_WAITING",
    "WORKER_STATUS_DONE",
    "WORKER_STATUS_FAILED",
    "DAY_CP_PENDING",
    "DAY_CP_IN_PROGRESS",
    "DAY_CP_COMPLETED",
    "DAY_CP_FAILED",
    "DAY_CP_CANCELLED",
    "ensure_exec_schema",
    "create_job",
    "get_job",
    "list_active_jobs",
    "set_job_status",
    "request_job_pause",
    "request_job_cancel",
    "request_job_resume",
    "refresh_job_day_counts",
    "list_workers",
    "update_worker",
    "list_checkpoints",
    "worker_pending_days",
    "update_checkpoint",
    "assign_days_round_robin",
    "job_progress_snapshot",
]
