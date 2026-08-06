"""Persistent replay session index (SQLite) + per-tab JSON artifacts on disk."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from datetime import datetime, timezone
from typing import Any

from .replay_engine_version import replay_engine_version

_DB_FILENAME = "replay_session.db"
_CACHE_DIRNAME = "replay_cache"

ARTIFACT_TRADE_REPORT = "trade_report"
ARTIFACT_MODEL_AUDIT = "model_audit"
ARTIFACT_PREDICTION_ANALYSIS = "prediction_analysis"
ARTIFACT_SCORING_COVERAGE = "scoring_coverage"
ARTIFACT_POSITION_LIMITS = "position_limits"
ARTIFACT_SUMMARY = "summary"

KNOWN_ARTIFACTS = (
    ARTIFACT_TRADE_REPORT,
    ARTIFACT_MODEL_AUDIT,
    ARTIFACT_PREDICTION_ANALYSIS,
    ARTIFACT_SCORING_COVERAGE,
    ARTIFACT_POSITION_LIMITS,
    ARTIFACT_SUMMARY,
)

_STATUS_MISSING = "missing"
_STATUS_READY = "ready"
_STATUS_ERROR = "error"
_STATUS_STALE = "stale"


def replay_session_db_path(data_dir: str) -> str:
    return os.path.join(data_dir, _DB_FILENAME)


def replay_cache_root(data_dir: str) -> str:
    return os.path.join(data_dir, _CACHE_DIRNAME)


def session_artifact_dir(data_dir: str, session_id: str) -> str:
    return os.path.join(replay_cache_root(data_dir), session_id)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_session_id(
    *,
    model_name: str,
    trading_day: str,
    expiry: str,
    underlying: str,
    engine_version: str | None = None,
) -> str:
    ev = str(engine_version or replay_engine_version())
    blob = "|".join([
        str(model_name or "").strip(),
        str(trading_day or "").strip(),
        str(expiry or "").strip(),
        str(underlying or "").strip().upper(),
        ev,
    ])
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _connect(data_dir: str) -> sqlite3.Connection:
    path = replay_session_db_path(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_replay_session_db(data_dir: str) -> None:
    with _connect(data_dir) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS replay_session (
                session_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                trading_day TEXT NOT NULL,
                expiry TEXT NOT NULL,
                underlying TEXT NOT NULL,
                engine_version TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_replay_session_natural
                ON replay_session(model_name, trading_day, expiry, underlying, engine_version);

            CREATE TABLE IF NOT EXISTS replay_artifact (
                session_id TEXT NOT NULL,
                artifact_name TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'missing',
                file_path TEXT,
                checksum TEXT,
                created_at TEXT,
                compute_ms REAL,
                error_message TEXT,
                PRIMARY KEY (session_id, artifact_name),
                FOREIGN KEY (session_id) REFERENCES replay_session(session_id) ON DELETE CASCADE
            );
            """
        )
        conn.commit()


def ensure_replay_session(
    data_dir: str,
    *,
    model_name: str,
    trading_day: str,
    expiry: str,
    underlying: str,
    engine_version: str | None = None,
) -> dict[str, Any]:
    init_replay_session_db(data_dir)
    ev = str(engine_version or replay_engine_version())
    sid = make_session_id(
        model_name=model_name,
        trading_day=trading_day,
        expiry=expiry,
        underlying=underlying,
        engine_version=ev,
    )
    now = _utc_now_iso()
    with _connect(data_dir) as conn:
        row = conn.execute(
            "SELECT * FROM replay_session WHERE session_id = ?",
            (sid,),
        ).fetchone()
        if row is None:
            conn.execute(
                """
                INSERT INTO replay_session (
                    session_id, model_name, trading_day, expiry, underlying,
                    engine_version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sid,
                    str(model_name).strip(),
                    str(trading_day).strip(),
                    str(expiry).strip(),
                    str(underlying).strip().upper(),
                    ev,
                    now,
                    now,
                ),
            )
        else:
            conn.execute(
                "UPDATE replay_session SET updated_at = ? WHERE session_id = ?",
                (now, sid),
            )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM replay_session WHERE session_id = ?",
            (sid,),
        ).fetchone()
    os.makedirs(session_artifact_dir(data_dir, sid), exist_ok=True)
    return dict(row) if row else {}


def list_replay_sessions_for_model(
    data_dir: str,
    *,
    model_name: str,
    underlying: str,
    expiry: str | None = None,
) -> list[dict[str, Any]]:
    """All replay sessions for one model + underlying (latest per trading day)."""
    init_replay_session_db(data_dir)
    name = str(model_name or "").strip()
    ul = str(underlying or "NIFTY").strip().upper()
    exp = str(expiry or "").strip()
    sql = """
        SELECT * FROM replay_session
        WHERE model_name = ? AND underlying = ?
    """
    params: list[Any] = [name, ul]
    if exp:
        sql += " AND expiry = ?"
        params.append(exp)
    sql += " ORDER BY trading_day DESC, updated_at DESC"
    with _connect(data_dir) as conn:
        rows = conn.execute(sql, params).fetchall()
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        day = str(row["trading_day"] or "").strip()
        if not day or day in seen:
            continue
        seen.add(day)
        out.append(dict(row))
    return out


def list_replay_sessions_for_day(
    data_dir: str,
    *,
    trading_day: str,
    underlying: str,
    expiry: str | None = None,
) -> list[dict[str, Any]]:
    """Latest replay session per model for a trading day (SQLite index only)."""
    init_replay_session_db(data_dir)
    day = str(trading_day or "").strip()
    ul = str(underlying or "NIFTY").strip().upper()
    exp = str(expiry or "").strip()
    sql = """
        SELECT * FROM replay_session
        WHERE trading_day = ? AND underlying = ?
    """
    params: list[Any] = [day, ul]
    if exp:
        sql += " AND expiry = ?"
        params.append(exp)
    sql += " ORDER BY updated_at DESC"
    with _connect(data_dir) as conn:
        rows = conn.execute(sql, params).fetchall()
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        name = str(row["model_name"] or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(dict(row))
    return out


def find_replay_session(
    data_dir: str,
    *,
    model_name: str,
    trading_day: str,
    expiry: str,
    underlying: str,
    engine_version: str | None = None,
) -> dict[str, Any] | None:
    init_replay_session_db(data_dir)
    ev = str(engine_version or replay_engine_version())
    sid = make_session_id(
        model_name=model_name,
        trading_day=trading_day,
        expiry=expiry,
        underlying=underlying,
        engine_version=ev,
    )
    with _connect(data_dir) as conn:
        row = conn.execute(
            "SELECT * FROM replay_session WHERE session_id = ?",
            (sid,),
        ).fetchone()
    return dict(row) if row else None


def list_artifacts(data_dir: str, session_id: str) -> list[dict[str, Any]]:
    init_replay_session_db(data_dir)
    with _connect(data_dir) as conn:
        rows = conn.execute(
            """
            SELECT artifact_name, status, file_path, checksum, created_at, compute_ms, error_message
            FROM replay_artifact
            WHERE session_id = ?
            ORDER BY artifact_name
            """,
            (str(session_id),),
        ).fetchall()
    by_name = {str(r["artifact_name"]): dict(r) for r in rows}
    out: list[dict[str, Any]] = []
    for name in KNOWN_ARTIFACTS:
        row = by_name.get(name)
        if row:
            out.append(row)
        else:
            out.append({
                "artifact_name": name,
                "status": _STATUS_MISSING,
                "file_path": None,
                "checksum": None,
                "created_at": None,
                "compute_ms": None,
                "error_message": None,
            })
    for name, row in sorted(by_name.items()):
        if name not in KNOWN_ARTIFACTS:
            out.append(row)
    return out


def get_session_status_doc(
    data_dir: str,
    *,
    model_name: str,
    trading_day: str,
    expiry: str,
    underlying: str,
    engine_version: str | None = None,
) -> dict[str, Any]:
    ev = str(engine_version or replay_engine_version())
    session = find_replay_session(
        data_dir,
        model_name=model_name,
        trading_day=trading_day,
        expiry=expiry,
        underlying=underlying,
        engine_version=ev,
    )
    if not session:
        return {
            "found": False,
            "engine_version": ev,
            "model_name": str(model_name).strip(),
            "trading_day": str(trading_day).strip(),
            "expiry": str(expiry).strip(),
            "underlying": str(underlying).strip().upper(),
            "artifacts": [
                {"artifact_name": n, "status": _STATUS_MISSING, "cached": False, "created_at": None}
                for n in KNOWN_ARTIFACTS
            ],
            "cache_state": "none",
        }
    artifacts = list_artifacts(data_dir, session["session_id"])
    ready = sum(1 for a in artifacts if a.get("status") == _STATUS_READY)
    total = len(KNOWN_ARTIFACTS)
    if ready == 0:
        cache_state = "none"
    elif ready >= total:
        cache_state = "complete"
    else:
        cache_state = "partial"
    return {
        "found": True,
        "cache_state": cache_state,
        "engine_version": ev,
        "session": session,
        "artifacts": [
            {
                **a,
                "cached": a.get("status") == _STATUS_READY,
            }
            for a in artifacts
        ],
    }


def _artifact_file_path(data_dir: str, session_id: str, artifact_name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(artifact_name))
    return os.path.join(session_artifact_dir(data_dir, session_id), f"{safe}.json")


def _file_checksum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_artifact_json(
    data_dir: str,
    session_id: str,
    artifact_name: str,
    payload: Any,
    *,
    compute_ms: float | None = None,
) -> str:
    init_replay_session_db(data_dir)
    path = _artifact_file_path(data_dir, session_id, artifact_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
    checksum = _file_checksum(path)
    now = _utc_now_iso()
    rel_path = os.path.relpath(path, data_dir).replace("\\", "/")
    with _connect(data_dir) as conn:
        conn.execute(
            """
            INSERT INTO replay_artifact (
                session_id, artifact_name, status, file_path, checksum,
                created_at, compute_ms, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL)
            ON CONFLICT(session_id, artifact_name) DO UPDATE SET
                status = excluded.status,
                file_path = excluded.file_path,
                checksum = excluded.checksum,
                created_at = excluded.created_at,
                compute_ms = excluded.compute_ms,
                error_message = NULL
            """,
            (
                str(session_id),
                str(artifact_name),
                _STATUS_READY,
                rel_path,
                checksum,
                now,
                float(compute_ms) if compute_ms is not None else None,
            ),
        )
        conn.execute(
            "UPDATE replay_session SET updated_at = ? WHERE session_id = ?",
            (now, str(session_id)),
        )
        conn.commit()
    return path


def load_artifact_json(
    data_dir: str,
    session_id: str,
    artifact_name: str,
) -> Any | None:
    init_replay_session_db(data_dir)
    with _connect(data_dir) as conn:
        row = conn.execute(
            """
            SELECT status, file_path FROM replay_artifact
            WHERE session_id = ? AND artifact_name = ?
            """,
            (str(session_id), str(artifact_name)),
        ).fetchone()
    if not row or row["status"] != _STATUS_READY or not row["file_path"]:
        return None
    path = row["file_path"]
    if not os.path.isabs(path):
        path = os.path.join(data_dir, path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def artifact_is_ready(data_dir: str, session_id: str, artifact_name: str) -> bool:
    return load_artifact_json(data_dir, session_id, artifact_name) is not None


def bust_replay_session(data_dir: str, session_id: str) -> None:
    init_replay_session_db(data_dir)
    art_dir = session_artifact_dir(data_dir, session_id)
    if os.path.isdir(art_dir):
        for name in os.listdir(art_dir):
            try:
                os.remove(os.path.join(art_dir, name))
            except OSError:
                pass
    now = _utc_now_iso()
    with _connect(data_dir) as conn:
        conn.execute(
            """
            UPDATE replay_artifact
            SET status = ?, file_path = NULL, checksum = NULL,
                created_at = NULL, compute_ms = NULL, error_message = NULL
            WHERE session_id = ?
            """,
            (_STATUS_MISSING, str(session_id)),
        )
        conn.execute(
            "UPDATE replay_session SET updated_at = ? WHERE session_id = ?",
            (now, str(session_id)),
        )
        conn.commit()


def delete_replay_sessions_for_model(data_dir: str, model_name: str) -> dict[str, Any]:
    """Remove all replay_session rows and on-disk cache artifacts for a model."""
    init_replay_session_db(data_dir)
    name = str(model_name or "").strip()
    if not name:
        return {"deleted_sessions": 0, "session_ids": []}

    with _connect(data_dir) as conn:
        rows = conn.execute(
            "SELECT session_id FROM replay_session WHERE model_name = ?",
            (name,),
        ).fetchall()
    session_ids = [str(r["session_id"]) for r in rows if r["session_id"]]

    for sid in session_ids:
        art_dir = session_artifact_dir(data_dir, sid)
        if os.path.isdir(art_dir):
            try:
                shutil.rmtree(art_dir)
            except OSError:
                pass

    if session_ids:
        with _connect(data_dir) as conn:
            conn.execute("DELETE FROM replay_session WHERE model_name = ?", (name,))
            conn.commit()

    return {"deleted_sessions": len(session_ids), "session_ids": session_ids}
