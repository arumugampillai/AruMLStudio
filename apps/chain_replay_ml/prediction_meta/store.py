"""SQLite store for prediction meta dataset (resume + batch insert)."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

DEFAULT_BATCH_SIZE = 1000

_TEXT_COLUMNS = frozenset({
    "prediction_id",
    "trading_day",
    "token",
    "strike",
    "option_type",
    "symbol",
    "market",
    "expiry",
    "feature_version",
    "model_registry_version",
})


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BuilderProgress:
    status: str
    rows_done: int
    rows_total: int | None
    last_trading_day: str | None
    last_timestamp: float | None
    last_token: str | None
    started_at: str | None
    finished_at: str | None
    error_message: str | None


class PredictionMetaStore:
    def __init__(self, db_path: str, *, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self.db_path = db_path
        self.batch_size = max(1, int(batch_size))
        self._conn: sqlite3.Connection | None = None
        self._insert_sql: str | None = None
        self._insert_cols: list[str] = []

    def open(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_tables()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> PredictionMetaStore:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("PredictionMetaStore not open")
        return self._conn

    def _ensure_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS samples (
                prediction_id TEXT PRIMARY KEY,
                trading_day TEXT NOT NULL,
                timestamp   REAL NOT NULL,
                token       TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_samples_grid
                ON samples(trading_day, timestamp, token);

            CREATE TABLE IF NOT EXISTS builder_progress (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                status TEXT NOT NULL DEFAULT 'idle',
                rows_done INTEGER NOT NULL DEFAULT 0,
                rows_total INTEGER,
                last_trading_day TEXT,
                last_timestamp REAL,
                last_token TEXT,
                started_at TEXT,
                finished_at TEXT,
                error_message TEXT
            );

            CREATE TABLE IF NOT EXISTS dataset_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            INSERT OR IGNORE INTO builder_progress (id, status) VALUES (1, 'idle');
            """
        )
        from .model_registry import ensure_registry_tables

        ensure_registry_tables(self.conn)
        self.conn.commit()

    def ensure_columns(self, columns: Sequence[str]) -> None:
        existing = {row[1] for row in self.conn.execute("PRAGMA table_info(samples)").fetchall()}
        pk = {"prediction_id"}
        for col in columns:
            if col in pk or col in existing:
                continue
            sql_type = "TEXT" if col in _TEXT_COLUMNS else "REAL"
            self.conn.execute(f'ALTER TABLE samples ADD COLUMN "{col}" {sql_type}')
            existing.add(col)
        self.conn.commit()

    def set_meta(self, key: str, value: Any) -> None:
        blob = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        self.conn.execute(
            "INSERT INTO dataset_meta(key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, blob),
        )
        self.conn.commit()

    def get_meta(self, key: str) -> Any:
        row = self.conn.execute("SELECT value FROM dataset_meta WHERE key = ?", (key,)).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except (TypeError, json.JSONDecodeError):
            return row[0]

    def read_progress(self) -> BuilderProgress:
        row = self.conn.execute(
            """
            SELECT status, rows_done, rows_total, last_trading_day, last_timestamp, last_token,
                   started_at, finished_at, error_message
            FROM builder_progress WHERE id = 1
            """
        ).fetchone()
        if not row:
            return BuilderProgress("idle", 0, None, None, None, None, None, None, None)
        return BuilderProgress(*row)

    def start_job(self, *, rows_total: int | None, resume: bool) -> BuilderProgress:
        prog = self.read_progress()
        rows_done = prog.rows_done if resume and prog.status in (
            "running", "paused", "complete", "failed",
        ) else 0
        if not resume:
            self.conn.execute("DELETE FROM samples")
            rows_done = 0
        self.conn.execute(
            """
            UPDATE builder_progress SET
                status = 'running',
                rows_done = ?,
                rows_total = ?,
                started_at = ?,
                finished_at = NULL,
                error_message = NULL
            WHERE id = 1
            """,
            (rows_done, rows_total, _utc_now()),
        )
        self.conn.commit()
        return self.read_progress()

    def update_checkpoint(
        self,
        *,
        rows_done: int,
        trading_day: str,
        timestamp: float,
        token: str,
    ) -> None:
        self.conn.execute(
            """
            UPDATE builder_progress SET
                rows_done = ?,
                last_trading_day = ?,
                last_timestamp = ?,
                last_token = ?
            WHERE id = 1
            """,
            (rows_done, trading_day, float(timestamp), str(token)),
        )
        self.conn.commit()

    def mark_complete(self, rows_done: int) -> None:
        self.conn.execute(
            """
            UPDATE builder_progress SET
                status = 'complete',
                rows_done = ?,
                finished_at = ?,
                error_message = NULL
            WHERE id = 1
            """,
            (rows_done, _utc_now()),
        )
        self.conn.commit()

    def mark_failed(self, error_message: str) -> None:
        self.conn.execute(
            """
            UPDATE builder_progress SET
                status = 'failed',
                error_message = ?,
                finished_at = ?
            WHERE id = 1
            """,
            (error_message[:2000], _utc_now()),
        )
        self.conn.commit()

    def row_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM samples").fetchone()
        return int(row[0]) if row else 0

    def prepare_insert(self, columns: Sequence[str]) -> None:
        self.ensure_columns(columns)
        self._insert_cols = list(columns)
        placeholders = ", ".join("?" for _ in self._insert_cols)
        col_sql = ", ".join(f'"{c}"' for c in self._insert_cols)
        self._insert_sql = f"INSERT OR IGNORE INTO samples ({col_sql}) VALUES ({placeholders})"

    def insert_rows(self, rows: Iterable[dict[str, Any]]) -> int:
        if not self._insert_sql:
            raise RuntimeError("Call prepare_insert() first")
        batch: list[tuple[Any, ...]] = []
        n = 0
        cols = self._insert_cols
        for row in rows:
            batch.append(tuple(row.get(c) for c in cols))
            if len(batch) >= self.batch_size:
                self.conn.executemany(self._insert_sql, batch)
                self.conn.commit()
                n += len(batch)
                batch.clear()
        if batch:
            self.conn.executemany(self._insert_sql, batch)
            self.conn.commit()
            n += len(batch)
        return n
