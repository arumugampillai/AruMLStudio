"""Research Sessions — group prediction runs, strategies, and results."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from .paths import research_sessions_db_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ResearchSessionStore:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.db_path = research_sessions_db_path(data_dir)
        self._conn: sqlite3.Connection | None = None

    def open(self) -> None:
        import os

        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> ResearchSessionStore:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("ResearchSessionStore not open")
        return self._conn

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS research_sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                notes TEXT,
                prediction_run_ids_json TEXT,
                strategy_version_ids_json TEXT,
                strategy_run_ids_json TEXT,
                created_on TEXT NOT NULL,
                updated_on TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_research_sessions_updated
                ON research_sessions(updated_on DESC);
            """
        )

    def _row_to_dict(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        for key, field in (
            ("prediction_run_ids", "prediction_run_ids_json"),
            ("strategy_version_ids", "strategy_version_ids_json"),
            ("strategy_run_ids", "strategy_run_ids_json"),
        ):
            raw = d.pop(field, None)
            try:
                d[key] = json.loads(raw) if raw else []
            except json.JSONDecodeError:
                d[key] = []
        return d

    def create_session(self, *, title: str, notes: str | None = None) -> dict[str, Any]:
        session_id = uuid.uuid4().hex
        now = _utc_now()
        self.conn.execute(
            """
            INSERT INTO research_sessions (
                session_id, title, notes,
                prediction_run_ids_json, strategy_version_ids_json, strategy_run_ids_json,
                created_on, updated_on
            ) VALUES (?, ?, ?, '[]', '[]', '[]', ?, ?)
            """,
            (session_id, title, notes, now, now),
        )
        self.conn.commit()
        return self.get_session(session_id) or {"session_id": session_id}

    def update_session(self, session_id: str, doc: dict[str, Any]) -> dict[str, Any] | None:
        existing = self.get_session(session_id)
        if not existing:
            return None
        title = doc.get("title", existing.get("title"))
        notes = doc.get("notes", existing.get("notes"))
        pred_ids = doc.get("prediction_run_ids", existing.get("prediction_run_ids"))
        ver_ids = doc.get("strategy_version_ids", existing.get("strategy_version_ids"))
        run_ids = doc.get("strategy_run_ids", existing.get("strategy_run_ids"))
        self.conn.execute(
            """
            UPDATE research_sessions
            SET title = ?, notes = ?,
                prediction_run_ids_json = ?,
                strategy_version_ids_json = ?,
                strategy_run_ids_json = ?,
                updated_on = ?
            WHERE session_id = ?
            """,
            (
                title,
                notes,
                json.dumps(list(pred_ids or [])),
                json.dumps(list(ver_ids or [])),
                json.dumps(list(run_ids or [])),
                _utc_now(),
                session_id,
            ),
        )
        self.conn.commit()
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM research_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
        return self._row_to_dict(row)

    def list_sessions(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT * FROM research_sessions ORDER BY updated_on DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows if r]

    def delete_session(self, session_id: str) -> bool:
        cur = self.conn.execute(
            "DELETE FROM research_sessions WHERE session_id = ?", (session_id,)
        )
        self.conn.commit()
        return cur.rowcount > 0
