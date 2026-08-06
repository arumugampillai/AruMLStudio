"""Research bookmarks — save replay moments for later review."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from chain_replay_ml.research_lab.paths import research_sessions_db_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bookmarks_db_path(data_dir: str) -> str:
    import os

    return os.path.join(os.path.dirname(research_sessions_db_path(data_dir)), "research_bookmarks.db")


class ResearchBookmarksStore:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.db_path = _bookmarks_db_path(data_dir)
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

    def __enter__(self) -> ResearchBookmarksStore:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("ResearchBookmarksStore not open")
        return self._conn

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS research_bookmarks (
                bookmark_id TEXT PRIMARY KEY,
                prediction_run_id TEXT NOT NULL,
                fold_id TEXT,
                trade_id TEXT,
                timestamp REAL,
                sequence INTEGER,
                title TEXT,
                reason TEXT NOT NULL,
                context_json TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_bookmarks_run
                ON research_bookmarks(prediction_run_id, fold_id, created_at DESC);
            """
        )
        cols = {row[1] for row in self.conn.execute("PRAGMA table_info(research_bookmarks)").fetchall()}
        if "tags_json" not in cols:
            self.conn.execute("ALTER TABLE research_bookmarks ADD COLUMN tags_json TEXT")
            self.conn.commit()

    def save_bookmark(
        self,
        *,
        prediction_run_id: str,
        reason: str,
        fold_id: str | None = None,
        trade_id: str | None = None,
        timestamp: float | None = None,
        sequence: int | None = None,
        title: str | None = None,
        context: dict[str, Any] | None = None,
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        bid = str(uuid.uuid4())
        now = _utc_now()
        tag_list = [str(t).strip() for t in (tags or []) if str(t).strip()]
        ctx = dict(context or {})
        if tag_list and "tags" not in ctx:
            ctx["tags"] = tag_list
        self.conn.execute(
            """
            INSERT INTO research_bookmarks (
                bookmark_id, prediction_run_id, fold_id, trade_id,
                timestamp, sequence, title, reason, context_json, tags_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                bid,
                prediction_run_id,
                fold_id,
                trade_id,
                timestamp,
                sequence,
                title,
                reason,
                json.dumps(ctx, default=str),
                json.dumps(tag_list),
                now,
            ),
        )
        self.conn.commit()
        return self.get_bookmark(bid) or {"bookmark_id": bid}

    def get_bookmark(self, bookmark_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM research_bookmarks WHERE bookmark_id = ?",
            (bookmark_id,),
        ).fetchone()
        return self._row_to_dict(row)

    def list_bookmarks(
        self,
        *,
        prediction_run_id: str | None = None,
        fold_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if prediction_run_id:
            clauses.append("prediction_run_id = ?")
            params.append(prediction_run_id)
        if fold_id:
            clauses.append("fold_id = ?")
            params.append(fold_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(
            f"""
            SELECT * FROM research_bookmarks
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows if r]

    def search_bookmarks(
        self,
        query: str,
        *,
        prediction_run_id: str | None = None,
        fold_id: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        q = (query or "").strip()
        if not q:
            return self.list_bookmarks(
                prediction_run_id=prediction_run_id,
                fold_id=fold_id,
                limit=limit,
            )
        like = f"%{q}%"
        clauses = ["(reason LIKE ? OR title LIKE ? OR tags_json LIKE ? OR context_json LIKE ?)"]
        params: list[Any] = [like, like, like, like]
        if prediction_run_id:
            clauses.append("prediction_run_id = ?")
            params.append(prediction_run_id)
        if fold_id:
            clauses.append("fold_id = ?")
            params.append(fold_id)
        where = f"WHERE {' AND '.join(clauses)}"
        rows = self.conn.execute(
            f"""
            SELECT * FROM research_bookmarks
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows if r]

    def _row_to_dict(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        try:
            d["context"] = json.loads(d.pop("context_json") or "{}")
        except json.JSONDecodeError:
            d["context"] = {}
        try:
            d["tags"] = json.loads(d.pop("tags_json") or "[]")
        except (json.JSONDecodeError, KeyError):
            d["tags"] = d.get("context", {}).get("tags") or []
        return d


def save_research_bookmark(data_dir: str, **kwargs: Any) -> dict[str, Any]:
    with ResearchBookmarksStore(data_dir) as store:
        return store.save_bookmark(**kwargs)


def list_research_bookmarks(data_dir: str, **kwargs: Any) -> list[dict[str, Any]]:
    with ResearchBookmarksStore(data_dir) as store:
        return store.list_bookmarks(**kwargs)


def search_research_bookmarks(data_dir: str, query: str, **kwargs: Any) -> list[dict[str, Any]]:
    with ResearchBookmarksStore(data_dir) as store:
        return store.search_bookmarks(query, **kwargs)
