"""Research notebook — searchable notes for folds, models, and strategies."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from chain_replay_ml.research_lab.paths import research_sessions_db_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _notes_db_path(data_dir: str) -> str:
    import os

    return os.path.join(os.path.dirname(research_sessions_db_path(data_dir)), "research_notes.db")


class ResearchNotesStore:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.db_path = _notes_db_path(data_dir)
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

    def __enter__(self) -> ResearchNotesStore:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("ResearchNotesStore not open")
        return self._conn

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS research_notes (
                note_id TEXT PRIMARY KEY,
                scope_type TEXT NOT NULL,
                scope_key TEXT NOT NULL,
                prediction_run_id TEXT,
                fold_id TEXT,
                model_id TEXT,
                title TEXT,
                body TEXT NOT NULL,
                tags_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_research_notes_scope
                ON research_notes(scope_type, scope_key);
            CREATE INDEX IF NOT EXISTS idx_research_notes_run
                ON research_notes(prediction_run_id, fold_id);
            """
        )

    def save_note(
        self,
        *,
        scope_type: str,
        scope_key: str,
        body: str,
        title: str | None = None,
        prediction_run_id: str | None = None,
        fold_id: str | None = None,
        model_id: str | None = None,
        tags: list[str] | None = None,
        note_id: str | None = None,
    ) -> dict[str, Any]:
        now = _utc_now()
        nid = note_id or str(uuid.uuid4())
        existing = self.conn.execute(
            "SELECT note_id FROM research_notes WHERE note_id = ?", (nid,)
        ).fetchone()
        if existing:
            self.conn.execute(
                """
                UPDATE research_notes
                SET body = ?, title = ?, tags_json = ?, updated_at = ?
                WHERE note_id = ?
                """,
                (body, title, json.dumps(tags or []), now, nid),
            )
        else:
            self.conn.execute(
                """
                INSERT INTO research_notes (
                    note_id, scope_type, scope_key, prediction_run_id, fold_id,
                    model_id, title, body, tags_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    nid,
                    scope_type,
                    scope_key,
                    prediction_run_id,
                    fold_id,
                    model_id,
                    title,
                    body,
                    json.dumps(tags or []),
                    now,
                    now,
                ),
            )
        self.conn.commit()
        return self.get_note(nid) or {"note_id": nid}

    def get_note(self, note_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM research_notes WHERE note_id = ?", (note_id,)
        ).fetchone()
        return self._row_to_dict(row)

    def list_notes(
        self,
        *,
        scope_type: str | None = None,
        scope_key: str | None = None,
        prediction_run_id: str | None = None,
        fold_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if scope_type:
            clauses.append("scope_type = ?")
            params.append(scope_type)
        if scope_key:
            clauses.append("scope_key = ?")
            params.append(scope_key)
        if prediction_run_id:
            clauses.append("prediction_run_id = ?")
            params.append(prediction_run_id)
        if fold_id:
            clauses.append("fold_id = ?")
            params.append(fold_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self.conn.execute(
            f"""
            SELECT * FROM research_notes
            {where}
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (*params, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows if r]

    def search_notes(self, query: str, *, limit: int = 50) -> list[dict[str, Any]]:
        q = f"%{query.strip()}%"
        rows = self.conn.execute(
            """
            SELECT * FROM research_notes
            WHERE body LIKE ? OR title LIKE ? OR tags_json LIKE ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (q, q, q, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows if r]

    def _row_to_dict(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        try:
            d["tags"] = json.loads(d.pop("tags_json") or "[]")
        except json.JSONDecodeError:
            d["tags"] = []
        return d


def save_fold_note(
    data_dir: str,
    *,
    prediction_run_id: str,
    fold_id: str,
    body: str,
    title: str | None = None,
    model_id: str | None = None,
    tags: list[str] | None = None,
) -> dict[str, Any]:
    scope_key = f"{prediction_run_id}:{fold_id}"
    with ResearchNotesStore(data_dir) as store:
        return store.save_note(
            scope_type="fold",
            scope_key=scope_key,
            body=body,
            title=title,
            prediction_run_id=prediction_run_id,
            fold_id=fold_id,
            model_id=model_id,
            tags=tags,
        )


def list_fold_notes(
    data_dir: str,
    *,
    prediction_run_id: str,
    fold_id: str,
) -> list[dict[str, Any]]:
    with ResearchNotesStore(data_dir) as store:
        return store.list_notes(prediction_run_id=prediction_run_id, fold_id=fold_id)


def search_research_notes(data_dir: str, query: str, *, limit: int = 50) -> list[dict[str, Any]]:
    with ResearchNotesStore(data_dir) as store:
        return store.search_notes(query, limit=limit)
