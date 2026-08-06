"""SQLite master dataset store — one day per transaction, builder checkpoint."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import time
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any, Iterable, Sequence

_store_registry_lock = RLock()
_store_registry: dict[str, list[weakref.ref]] = {}


def _norm_db_path(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _register_store(store: "MasterStore") -> None:
    key = _norm_db_path(store.db_path)
    with _store_registry_lock:
        refs = _store_registry.setdefault(key, [])
        refs.append(weakref.ref(store))
        _store_registry[key] = [r for r in refs if r() is not None]


def _unregister_store(store: "MasterStore") -> None:
    key = _norm_db_path(store.db_path)
    with _store_registry_lock:
        refs = _store_registry.get(key)
        if not refs:
            return
        live = [r for r in refs if r() is not None and r() is not store]
        if live:
            _store_registry[key] = live
        else:
            _store_registry.pop(key, None)


def close_all_stores_for_path(db_path: str) -> int:
    """Close every open MasterStore handle for a database path (e.g. before delete)."""
    key = _norm_db_path(db_path)
    closed = 0
    with _store_registry_lock:
        refs = list(_store_registry.get(key, []))
    for ref in refs:
        store = ref()
        if store is None:
            continue
        try:
            store.close()
            closed += 1
        except Exception:
            pass
    with _store_registry_lock:
        _store_registry.pop(key, None)
    return closed

DEFAULT_BATCH_SIZE = 1000

_TEXT_COLUMNS = frozenset({
    "trading_day",
    "market",
    "expiry",
    "option_type",
    "token",
    "symbol",
})

_SAFE_COL = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

_META_EXTRA_COLUMNS: dict[str, str] = {
    "market": "TEXT",
    "builder_version": "TEXT",
    "created_from": "TEXT",
    "feature_registry_version": "TEXT",
    "feature_hash": "TEXT",
    "target_hash": "TEXT",
    "dataset_fingerprint": "TEXT",
    "last_day_added": "TEXT",
    "last_day_deleted": "TEXT",
    "last_modified": "TEXT",
    "last_backfill_ms": "INTEGER",
    "last_refresh_ms": "INTEGER",
}

_DAYS_EXTRA_COLUMNS: dict[str, str] = {
    "dominant_expiry": "TEXT",
    "is_expiry_day": "INTEGER",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sql_type(col: str) -> str:
    if col in _TEXT_COLUMNS:
        return "TEXT"
    return "REAL"


@dataclass
class BuilderProgress:
    last_completed_day: str | None
    current_day: str | None
    status: str
    started_at: str | None
    finished_at: str | None
    job_id: str | None
    error_message: str | None
    days_total: int | None
    days_done: int | None


@dataclass
class MasterDatasetMetaRow:
    metadata_version: int
    total_rows: int
    total_days: int
    feature_count: int | None
    target_count: int | None
    schema_hash: str | None
    sampling_interval_sec: int | None
    database_size: int | None
    wal_size: int | None
    created_at: str | None
    updated_at: str | None
    last_metadata_refresh: str | None
    last_integrity_check: str | None
    metadata_status: str


class MasterStore:
    """Master SQLite dataset with day-atomic writes and builder_progress checkpoint."""

    def __init__(self, db_path: str, *, batch_size: int = DEFAULT_BATCH_SIZE) -> None:
        self.db_path = db_path
        self.batch_size = max(1, int(batch_size))
        self._conn: sqlite3.Connection | None = None
        self._day_txn: bool = False
        self._insert_sql: str | None = None
        self._insert_cols: list[str] = []
        self._next_master_row_id: int | None = None

    def open(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, timeout=30.0)
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_meta_tables()
        _register_store(self)

    def close(self) -> None:
        if self._conn is not None:
            if self._day_txn:
                self._conn.rollback()
                self._day_txn = False
            self._conn.close()
            self._conn = None
        _unregister_store(self)

    def __enter__(self) -> MasterStore:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("MasterStore not open")
        return self._conn

    def _ensure_meta_tables(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS samples (
                trading_day      TEXT NOT NULL,
                timestamp        REAL NOT NULL,
                token            TEXT NOT NULL,
                PRIMARY KEY (trading_day, timestamp, token)
            );

            CREATE TABLE IF NOT EXISTS builder_progress (
                id                   INTEGER PRIMARY KEY CHECK (id = 1),
                last_completed_day   TEXT,
                current_day          TEXT,
                status               TEXT NOT NULL DEFAULT 'idle',
                started_at           TEXT,
                finished_at          TEXT,
                job_id               TEXT,
                error_message        TEXT,
                days_total           INTEGER,
                days_done            INTEGER
            );

            CREATE TABLE IF NOT EXISTS dataset_meta (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS master_dataset_meta (
                id                      INTEGER PRIMARY KEY CHECK (id = 1),
                metadata_version        INTEGER NOT NULL DEFAULT 0,
                total_rows              INTEGER NOT NULL DEFAULT 0,
                total_days              INTEGER NOT NULL DEFAULT 0,
                feature_count           INTEGER,
                target_count            INTEGER,
                schema_hash             TEXT,
                sampling_interval_sec   INTEGER,
                database_size           INTEGER,
                wal_size                INTEGER,
                created_at              TEXT,
                updated_at              TEXT,
                last_metadata_refresh   TEXT,
                last_integrity_check    TEXT,
                metadata_status         TEXT NOT NULL DEFAULT 'OK'
            );

            CREATE TABLE IF NOT EXISTS master_dataset_days (
                trading_day       TEXT PRIMARY KEY,
                row_count         INTEGER NOT NULL DEFAULT 0,
                token_count       INTEGER,
                expiry_count      INTEGER,
                dominant_expiry   TEXT,
                is_expiry_day     INTEGER,
                first_timestamp   REAL,
                last_timestamp    REAL,
                coverage_percent  REAL,
                rejected_rows     INTEGER,
                status            TEXT,
                last_updated      TEXT
            );

            CREATE TABLE IF NOT EXISTS master_dataset_meta_history (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                TEXT NOT NULL,
                reason            TEXT NOT NULL,
                metadata_version  INTEGER,
                total_rows        INTEGER,
                total_days        INTEGER
            );

            CREATE TABLE IF NOT EXISTS master_dataset_distribution (
                distribution_type TEXT NOT NULL,
                bucket            TEXT NOT NULL,
                rows              INTEGER NOT NULL DEFAULT 0,
                tokens            INTEGER,
                updated_at        TEXT,
                PRIMARY KEY (distribution_type, bucket)
            );

            INSERT OR IGNORE INTO builder_progress (id, status) VALUES (1, 'idle');
            INSERT OR IGNORE INTO master_dataset_meta (id, metadata_status) VALUES (1, 'VALID');
            """
        )
        self.conn.commit()
        self._migrate_meta_schema()
        self._migrate_days_schema()
        self.ensure_master_row_id()
        self._maybe_backfill_metadata()
        try:
            from .day_metadata import ensure_day_metadata_tables

            ensure_day_metadata_tables(self.conn)
        except Exception:
            pass

    def ensure_master_row_id(self) -> None:
        """
        Immutable sample identity for FK joins from Research Lab prediction rows.

        Assigned once at insert (or backfilled from ROWID for legacy DBs). Never
        renumber existing values; regenerate = new master database.

        Important: when ``idx_samples_master_row_id`` already exists, skip the
        ``UPDATE … WHERE master_row_id IS NULL`` scan. That UPDATE walks every
        sample row (~25s on a 10GB DB) even when zero NULLs remain, and was
        making Master Dataset open / interval switches unusably slow.
        """
        existing = {
            str(row[1])
            for row in self.conn.execute("PRAGMA table_info(samples)").fetchall()
        }
        if "master_row_id" not in existing:
            self.conn.execute(
                "ALTER TABLE samples ADD COLUMN master_row_id INTEGER"
            )
            self.conn.commit()

        index_row = self.conn.execute(
            """
            SELECT 1 FROM sqlite_master
            WHERE type = 'index' AND name = 'idx_samples_master_row_id'
            """
        ).fetchone()
        if index_row:
            # Prior ensure completed (index is created only after backfill).
            return

        # One-time backfill: copy stable ROWID into the stored column.
        self.conn.execute(
            """
            UPDATE samples
            SET master_row_id = rowid
            WHERE master_row_id IS NULL
            """
        )
        self.conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_samples_master_row_id
            ON samples(master_row_id)
            WHERE master_row_id IS NOT NULL
            """
        )
        self.conn.commit()

    def next_master_row_id(self) -> int:
        """Next unused master_row_id (max+1). Safe after ensure_master_row_id()."""
        row = self.conn.execute(
            "SELECT COALESCE(MAX(master_row_id), 0) FROM samples"
        ).fetchone()
        return int(row[0] or 0) + 1

    def _migrate_meta_schema(self) -> None:
        existing = {
            str(row[1])
            for row in self.conn.execute("PRAGMA table_info(master_dataset_meta)").fetchall()
        }
        for col, sql_type in _META_EXTRA_COLUMNS.items():
            if col not in existing:
                self.conn.execute(
                    f'ALTER TABLE master_dataset_meta ADD COLUMN "{col}" {sql_type}'
                )
        self.conn.execute(
            """
            UPDATE master_dataset_meta SET metadata_status = 'VALID'
            WHERE metadata_status = 'OK' OR metadata_status IS NULL
            """
        )
        self.conn.commit()

    def _migrate_days_schema(self) -> None:
        existing = {
            str(row[1])
            for row in self.conn.execute("PRAGMA table_info(master_dataset_days)").fetchall()
        }
        for col, sql_type in _DAYS_EXTRA_COLUMNS.items():
            if col not in existing:
                self.conn.execute(
                    f'ALTER TABLE master_dataset_days ADD COLUMN "{col}" {sql_type}'
                )
        self.conn.commit()
        if self.get_meta("day_tags_version") != 1:
            self._ensure_expiry_day_flags()
            self.set_meta("day_tags_version", 1)

    def _ensure_expiry_day_flags(self) -> None:
        """Fill is_expiry_day from day-level samples (same-day expiry present)."""
        need = self.conn.execute(
            """
            SELECT 1 FROM master_dataset_days
            WHERE is_expiry_day IS NULL
            LIMIT 1
            """
        ).fetchone()
        if not need:
            return
        sample_cols = {
            str(row[1])
            for row in self.conn.execute("PRAGMA table_info(samples)").fetchall()
        }
        if "expiry" not in sample_cols or "trading_day" not in sample_cols:
            self.conn.execute(
                "UPDATE master_dataset_days SET is_expiry_day = 0 WHERE is_expiry_day IS NULL"
            )
            self.conn.commit()
            return
        expiry_days = {
            str(r[0])
            for r in self.conn.execute(
                """
                SELECT DISTINCT trading_day
                FROM samples
                WHERE expiry IS NOT NULL AND expiry = trading_day
                """
            ).fetchall()
            if r and r[0]
        }
        days = [
            str(r[0])
            for r in self.conn.execute(
                "SELECT trading_day FROM master_dataset_days WHERE is_expiry_day IS NULL"
            ).fetchall()
            if r and r[0]
        ]
        for td in days:
            self.conn.execute(
                """
                UPDATE master_dataset_days SET is_expiry_day = ?
                WHERE trading_day = ?
                """,
                (1 if td in expiry_days else 0, td),
            )
        self.conn.commit()

    def ensure_columns(self, columns: Sequence[str]) -> None:
        """Add missing sample columns (features, targets, metadata)."""
        existing = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(samples)").fetchall()
        }
        pk = {"trading_day", "timestamp", "token"}
        for col in columns:
            if col in pk or col in existing:
                continue
            if not _SAFE_COL.match(col):
                raise ValueError(f"Unsafe column name: {col}")
            self.conn.execute(
                f'ALTER TABLE samples ADD COLUMN "{col}" {_sql_type(col)}'
            )
            existing.add(col)
        self.conn.commit()

    def get_coverage_by_day(self) -> dict[str, Any]:
        raw = self.get_meta("coverage_by_day")
        return dict(raw) if isinstance(raw, dict) else {}

    def set_day_coverage(self, trading_day: str, coverage: dict[str, Any]) -> None:
        all_cov = self.get_coverage_by_day()
        stored = {
            k: v
            for k, v in coverage.items()
            if k not in ("first_tick_ts", "last_tick_ts")
        }
        all_cov[str(trading_day)] = stored
        self.set_meta("coverage_by_day", all_cov)
        rejected = int(stored.get("rejected_samples") or 0)
        cov_pct = stored.get("coverage_pct")
        cov_val = float(cov_pct) if cov_pct is not None else None
        now = _utc_now()
        self.conn.execute(
            """
            UPDATE master_dataset_days SET
                rejected_rows = ?,
                coverage_percent = ?,
                last_updated = ?
            WHERE trading_day = ?
            """,
            (rejected, cov_val, now, str(trading_day)),
        )
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
        row = self.conn.execute(
            "SELECT value FROM dataset_meta WHERE key = ?", (key,)
        ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except (TypeError, json.JSONDecodeError):
            return row[0]

    def read_builder_progress(self) -> BuilderProgress:
        row = self.conn.execute(
            """
            SELECT last_completed_day, current_day, status, started_at, finished_at,
                   job_id, error_message, days_total, days_done
            FROM builder_progress WHERE id = 1
            """
        ).fetchone()
        if not row:
            return BuilderProgress(None, None, "idle", None, None, None, None, None, None)
        return BuilderProgress(*row)

    def start_build_job(
        self,
        *,
        job_id: str,
        days_total: int,
        resume_after: str | None = None,
    ) -> BuilderProgress:
        last = resume_after
        if last is None:
            prog = self.read_builder_progress()
            last = prog.last_completed_day
        days_done = self.count_metadata_days() if last else 0
        self.conn.execute(
            """
            UPDATE builder_progress SET
                last_completed_day = ?,
                current_day = NULL,
                status = 'running',
                started_at = ?,
                finished_at = NULL,
                job_id = ?,
                error_message = NULL,
                days_total = ?,
                days_done = ?
            WHERE id = 1
            """,
            (last, _utc_now(), job_id, days_total, days_done),
        )
        self._set_metadata_status("BUILDING", commit=True)
        return self.read_builder_progress()

    def should_skip_day(self, trading_day: str, *, resume: bool = True) -> bool:
        """True if day is already fully committed (resume mode)."""
        if not resume:
            return False
        prog = self.read_builder_progress()
        if prog.last_completed_day and trading_day <= prog.last_completed_day:
            return self.row_count_for_day(trading_day) > 0
        return False

    def _mark_day_started_txn(self, trading_day: str) -> None:
        self.conn.execute(
            """
            UPDATE builder_progress SET current_day = ?, status = 'running'
            WHERE id = 1
            """,
            (trading_day,),
        )

    def mark_day_started(self, trading_day: str) -> None:
        self._mark_day_started_txn(trading_day)
        self.conn.commit()

    def mark_day_completed(self, trading_day: str) -> None:
        self.conn.execute(
            """
            UPDATE builder_progress SET
                last_completed_day = ?,
                current_day = NULL,
                days_done = COALESCE(days_done, 0) + 1
            WHERE id = 1
            """,
            (trading_day,),
        )
        self.conn.commit()

    def mark_build_failed(self, error_message: str) -> None:
        self.conn.execute(
            """
            UPDATE builder_progress SET
                status = 'failed',
                error_message = ?,
                current_day = NULL
            WHERE id = 1
            """,
            (error_message[:2000],),
        )
        self._set_metadata_status("ERROR", commit=True)

    def mark_build_complete(self) -> None:
        self.conn.execute(
            """
            UPDATE builder_progress SET
                status = 'complete',
                finished_at = ?,
                current_day = NULL,
                error_message = NULL
            WHERE id = 1
            """,
            (_utc_now(),),
        )
        self.conn.commit()

    def resume_start_day(self) -> str | None:
        """Next trading_day to process after last_completed_day (lexicographic ISO dates)."""
        prog = self.read_builder_progress()
        return prog.last_completed_day

    def row_count_for_day(self, trading_day: str) -> int:
        row = self.conn.execute(
            "SELECT row_count FROM master_dataset_days WHERE trading_day = ?",
            (trading_day,),
        ).fetchone()
        if row:
            return int(row[0] or 0)
        row = self.conn.execute(
            "SELECT COUNT(*) FROM samples WHERE trading_day = ?",
            (trading_day,),
        ).fetchone()
        return int(row[0]) if row else 0

    def delete_day(self, trading_day: str) -> int:
        td = str(trading_day)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            meta_row = self.conn.execute(
                "SELECT row_count FROM master_dataset_days WHERE trading_day = ?",
                (td,),
            ).fetchone()
            deleted = int(meta_row[0]) if meta_row else 0
            if deleted <= 0:
                row = self.conn.execute(
                    "SELECT COUNT(*) FROM samples WHERE trading_day = ?",
                    (td,),
                ).fetchone()
                deleted = int(row[0]) if row else 0

            from .master_distribution import apply_day_distribution_delta

            apply_day_distribution_delta(self.conn, td, sign=-1)

            self.conn.execute("DELETE FROM samples WHERE trading_day = ?", (td,))
            self.conn.execute("DELETE FROM master_dataset_days WHERE trading_day = ?", (td,))
            try:
                from .day_metadata import delete_day_metadata

                delete_day_metadata(self.conn, td)
            except Exception:
                pass

            all_cov = self.get_coverage_by_day()
            if td in all_cov:
                del all_cov[td]
                blob = json.dumps(all_cov, ensure_ascii=False)
                self.conn.execute(
                    "INSERT INTO dataset_meta(key, value) VALUES (?, ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    ("coverage_by_day", blob),
                )

            self._decrement_meta_after_day_delete(deleted)
            now = _utc_now()
            self.conn.execute(
                """
                UPDATE master_dataset_meta SET
                    last_day_deleted = ?,
                    last_modified = ?
                WHERE id = 1
                """,
                (td, now),
            )
            self._set_metadata_status("VALID", commit=False)
            self._append_meta_history("DAY_DELETED")
            self._refresh_meta_file_sizes()
            self.conn.commit()
            return deleted
        except Exception:
            self.conn.rollback()
            raise

    def remove_day_coverage(self, trading_day: str) -> None:
        all_cov = self.get_coverage_by_day()
        if trading_day in all_cov:
            del all_cov[trading_day]
            self.set_meta("coverage_by_day", all_cov)

    def begin_day(self, trading_day: str, columns: Sequence[str]) -> None:
        if self._day_txn:
            raise RuntimeError("Day transaction already open")
        self.ensure_master_row_id()
        col_list = list(columns)
        if "master_row_id" not in col_list:
            col_list.append("master_row_id")
        self.ensure_columns([c for c in col_list if c != "master_row_id"])
        self._insert_cols = col_list
        self._next_master_row_id = self.next_master_row_id()
        placeholders = ", ".join("?" for _ in self._insert_cols)
        col_sql = ", ".join(f'"{c}"' for c in self._insert_cols)
        self._insert_sql = f"INSERT INTO samples ({col_sql}) VALUES ({placeholders})"
        self.conn.execute("BEGIN IMMEDIATE")
        self._day_txn = True
        self._mark_day_started_txn(trading_day)
        self._set_metadata_status("BUILDING", commit=False)

    def insert_rows(self, rows: Iterable[dict[str, Any]]) -> int:
        if not self._day_txn or not self._insert_sql:
            raise RuntimeError("Call begin_day() before insert_rows()")
        batch: list[tuple[Any, ...]] = []
        n = 0
        cols = self._insert_cols
        next_id = self._next_master_row_id
        if next_id is None:
            next_id = self.next_master_row_id()
        for row in rows:
            values = dict(row)
            if values.get("master_row_id") is None and "master_row_id" in cols:
                values["master_row_id"] = int(next_id)
                next_id = int(next_id) + 1
            batch.append(tuple(values.get(c) for c in cols))
            if len(batch) >= self.batch_size:
                self.conn.executemany(self._insert_sql, batch)
                n += len(batch)
                batch.clear()
        self._next_master_row_id = int(next_id)
        if batch:
            self.conn.executemany(self._insert_sql, batch)
            n += len(batch)
        return n

    def commit_day(self, trading_day: str) -> None:
        if not self._day_txn:
            raise RuntimeError("No open day transaction")
        td = str(trading_day)
        stats = self._aggregate_day_stats(td)
        had_day = self.conn.execute(
            "SELECT 1 FROM master_dataset_days WHERE trading_day = ?",
            (td,),
        ).fetchone() is not None
        now = _utc_now()
        self.conn.execute(
            """
            INSERT INTO master_dataset_days (
                trading_day, row_count, token_count, expiry_count,
                dominant_expiry, is_expiry_day,
                first_timestamp, last_timestamp, status, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(trading_day) DO UPDATE SET
                row_count = excluded.row_count,
                token_count = excluded.token_count,
                expiry_count = excluded.expiry_count,
                dominant_expiry = excluded.dominant_expiry,
                is_expiry_day = excluded.is_expiry_day,
                first_timestamp = excluded.first_timestamp,
                last_timestamp = excluded.last_timestamp,
                status = excluded.status,
                last_updated = excluded.last_updated
            """,
            (
                td,
                int(stats["row_count"]),
                stats.get("token_count"),
                stats.get("expiry_count"),
                stats.get("dominant_expiry"),
                stats.get("is_expiry_day"),
                stats.get("first_timestamp"),
                stats.get("last_timestamp"),
                "complete",
                now,
            ),
        )
        if not had_day:
            self._increment_meta_after_day_add(int(stats["row_count"]))
        else:
            self._recompute_meta_totals()
        sample_cols = {
            str(row[1])
            for row in self.conn.execute("PRAGMA table_info(samples)").fetchall()
        }
        if had_day:
            from .master_distribution import rebuild_all_distributions

            rebuild_all_distributions(self.conn)
        else:
            from .master_distribution import apply_day_distribution_delta

            apply_day_distribution_delta(self.conn, td, sign=1, sample_cols=sample_cols)
        self.conn.execute(
            """
            UPDATE master_dataset_meta SET
                last_day_added = ?,
                last_modified = ?
            WHERE id = 1
            """,
            (td, now),
        )
        self._set_metadata_status("VALID", commit=False)
        self.conn.execute(
            """
            UPDATE builder_progress SET
                last_completed_day = ?,
                current_day = NULL,
                days_done = COALESCE(days_done, 0) + ?
            WHERE id = 1
            """,
            (td, 0 if had_day else 1),
        )
        self._append_meta_history("DAY_ADDED")
        self._refresh_meta_file_sizes()
        self.conn.commit()
        self._day_txn = False
        self._insert_sql = None
        self._insert_cols = []
        self._next_master_row_id = None

    def rollback_day(self) -> None:
        if self._day_txn:
            self.conn.rollback()
            self._day_txn = False
            self._insert_sql = None
            self._insert_cols = []
            self._next_master_row_id = None
        self.conn.execute(
            "UPDATE builder_progress SET current_day = NULL WHERE id = 1"
        )
        self.conn.commit()

    def distinct_trading_days(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT trading_day FROM master_dataset_days ORDER BY trading_day"
        ).fetchall()
        if rows:
            return [str(r[0]) for r in rows]
        rows = self.conn.execute(
            "SELECT DISTINCT trading_day FROM samples ORDER BY trading_day"
        ).fetchall()
        return [str(r[0]) for r in rows]

    def count_metadata_days(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM master_dataset_days").fetchone()
        return int(row[0]) if row else 0

    def read_master_meta(self) -> MasterDatasetMetaRow:
        row = self.conn.execute(
            """
            SELECT metadata_version, total_rows, total_days, feature_count, target_count,
                   schema_hash, sampling_interval_sec, database_size, wal_size,
                   created_at, updated_at, last_metadata_refresh, last_integrity_check,
                   metadata_status
            FROM master_dataset_meta WHERE id = 1
            """
        ).fetchone()
        if not row:
            return MasterDatasetMetaRow(0, 0, 0, None, None, None, None, None, None, None, None, None, None, "VALID")
        return MasterDatasetMetaRow(*row)

    def read_master_days(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT trading_day, row_count, token_count, expiry_count,
                   first_timestamp, last_timestamp, coverage_percent,
                   rejected_rows, status, last_updated,
                   dominant_expiry, is_expiry_day
            FROM master_dataset_days
            ORDER BY trading_day DESC
            """
        ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append({
                "trading_day": str(r[0]),
                "row_count": int(r[1] or 0),
                "token_count": int(r[2]) if r[2] is not None else None,
                "expiry_count": int(r[3]) if r[3] is not None else None,
                "first_timestamp": float(r[4]) if r[4] is not None else None,
                "last_timestamp": float(r[5]) if r[5] is not None else None,
                "coverage_percent": float(r[6]) if r[6] is not None else None,
                "rejected_rows": int(r[7]) if r[7] is not None else None,
                "status": str(r[8]) if r[8] is not None else None,
                "last_updated": str(r[9]) if r[9] is not None else None,
                "dominant_expiry": str(r[10]) if r[10] is not None else None,
                "is_expiry_day": int(r[11]) if r[11] is not None else None,
            })
        return out

    def row_counts_by_day(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT trading_day, row_count FROM master_dataset_days ORDER BY trading_day"
        ).fetchall()
        if rows:
            return {str(day): int(count) for day, count in rows}
        rows = self.conn.execute(
            "SELECT trading_day, COUNT(*) FROM samples GROUP BY trading_day ORDER BY trading_day"
        ).fetchall()
        return {str(day): int(count) for day, count in rows}

    def total_row_count(self) -> int:
        meta = self.read_master_meta()
        if meta.total_rows > 0 or self.count_metadata_days() > 0:
            return int(meta.total_rows)
        row = self.conn.execute("SELECT COUNT(*) FROM samples").fetchone()
        return int(row[0]) if row else 0

    def refresh_metadata_from_samples(self, *, reason: str = "BACKFILL") -> MasterDatasetMetaRow:
        """Full scan of samples — admin/backfill only."""
        from .master_distribution import rebuild_all_distributions

        t0 = time.perf_counter()
        self._set_metadata_status("REPAIRING", commit=True)
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            self.conn.execute("DELETE FROM master_dataset_days")
            sample_cols = {
                row[1]
                for row in self.conn.execute("PRAGMA table_info(samples)").fetchall()
            }
            expiry_sql = "COUNT(DISTINCT expiry)" if "expiry" in sample_cols else "NULL"
            day_rows = self.conn.execute(
                f"""
                SELECT trading_day,
                       COUNT(*) AS row_count,
                       COUNT(DISTINCT token) AS token_count,
                       {expiry_sql} AS expiry_count,
                       MIN(timestamp) AS first_timestamp,
                       MAX(timestamp) AS last_timestamp
                FROM samples
                GROUP BY trading_day
                ORDER BY trading_day
                """
            ).fetchall()
            expiry_days: set[str] = set()
            if "expiry" in sample_cols:
                expiry_days = {
                    str(r[0])
                    for r in self.conn.execute(
                        """
                        SELECT DISTINCT trading_day
                        FROM samples
                        WHERE expiry IS NOT NULL AND expiry = trading_day
                        """
                    ).fetchall()
                    if r and r[0]
                }
            now = _utc_now()
            total_rows = 0
            for r in day_rows:
                td = str(r[0])
                row_count = int(r[1] or 0)
                total_rows += row_count
                cov = self.get_coverage_by_day().get(td) or {}
                rejected = int(cov.get("rejected_samples") or 0) if isinstance(cov, dict) else 0
                cov_pct = cov.get("coverage_pct") if isinstance(cov, dict) else None
                cov_val = float(cov_pct) if cov_pct is not None else None
                self.conn.execute(
                    """
                    INSERT INTO master_dataset_days (
                        trading_day, row_count, token_count, expiry_count,
                        dominant_expiry, is_expiry_day,
                        first_timestamp, last_timestamp, coverage_percent,
                        rejected_rows, status, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        td, row_count,
                        int(r[2]) if r[2] is not None else None,
                        int(r[3]) if r[3] is not None else None,
                        None,
                        1 if td in expiry_days else 0,
                        float(r[4]) if r[4] is not None else None,
                        float(r[5]) if r[5] is not None else None,
                        cov_val, rejected, "complete", now,
                    ),
                )
            schema_fields = self._schema_fields_from_dataset_meta()
            sizes = self._file_sizes()
            meta = self.read_master_meta()
            version = int(meta.metadata_version or 0) + 1
            created = meta.created_at or now
            rebuild_all_distributions(self.conn)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            timing_col = "last_backfill_ms" if reason == "BACKFILL" else "last_refresh_ms"
            self.conn.execute(
                f"""
                UPDATE master_dataset_meta SET
                    metadata_version = ?,
                    total_rows = ?,
                    total_days = ?,
                    feature_count = ?,
                    target_count = ?,
                    schema_hash = ?,
                    sampling_interval_sec = ?,
                    database_size = ?,
                    wal_size = ?,
                    created_at = ?,
                    updated_at = ?,
                    last_metadata_refresh = ?,
                    metadata_status = 'VALID',
                    last_modified = ?,
                    {timing_col} = ?
                WHERE id = 1
                """,
                (
                    version,
                    total_rows,
                    len(day_rows),
                    schema_fields.get("feature_count"),
                    schema_fields.get("target_count"),
                    schema_fields.get("schema_hash"),
                    schema_fields.get("sampling_interval_sec"),
                    sizes.get("db_bytes"),
                    sizes.get("wal_bytes"),
                    created,
                    now,
                    now,
                    now,
                    elapsed_ms,
                ),
            )
            self._sync_fingerprint_from_meta_fields(schema_fields)
            self._append_meta_history(reason, version=version, total_rows=total_rows, total_days=len(day_rows))
            self.conn.commit()
            return self.read_master_meta()
        except Exception:
            self.conn.rollback()
            self._set_metadata_status("ERROR", commit=True)
            raise

    def update_build_identity(self, identity: dict[str, Any]) -> None:
        """Persist registry snapshot and dataset fingerprint after a build."""
        from .master_fingerprint import fingerprint_json_blob

        fp = identity.get("dataset_fingerprint") or {}
        now = _utc_now()
        self.conn.execute(
            """
            UPDATE master_dataset_meta SET
                market = COALESCE(?, market),
                builder_version = COALESCE(?, builder_version),
                created_from = COALESCE(?, created_from),
                feature_registry_version = COALESCE(?, feature_registry_version),
                feature_hash = COALESCE(?, feature_hash),
                target_hash = COALESCE(?, target_hash),
                schema_hash = COALESCE(?, schema_hash),
                feature_count = COALESCE(?, feature_count),
                target_count = COALESCE(?, target_count),
                sampling_interval_sec = COALESCE(?, sampling_interval_sec),
                dataset_fingerprint = ?,
                updated_at = ?,
                last_modified = ?,
                metadata_status = 'VALID'
            WHERE id = 1
            """,
            (
                identity.get("market"),
                identity.get("builder_version"),
                identity.get("created_from"),
                identity.get("feature_registry_version"),
                identity.get("feature_hash"),
                identity.get("target_hash"),
                identity.get("schema_hash"),
                identity.get("feature_count"),
                identity.get("target_count"),
                identity.get("sampling_interval_sec"),
                fingerprint_json_blob(fp) if fp else None,
                now,
                now,
            ),
        )
        self.conn.commit()

    def read_master_distributions(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT distribution_type, bucket, rows, tokens, updated_at
            FROM master_dataset_distribution
            ORDER BY distribution_type, bucket
            """
        ).fetchall()
        return [
            {
                "distribution_type": str(r[0]),
                "bucket": str(r[1]),
                "rows": int(r[2] or 0),
                "tokens": int(r[3]) if r[3] is not None else None,
                "updated_at": str(r[4]) if r[4] is not None else None,
            }
            for r in rows
        ]

    def read_master_meta_dict(self) -> dict[str, Any]:
        row = self.conn.execute("SELECT * FROM master_dataset_meta WHERE id = 1").fetchone()
        if not row:
            return {}
        cols = [d[1] for d in self.conn.execute("PRAGMA table_info(master_dataset_meta)").fetchall()]
        out = {cols[i]: row[i] for i in range(len(cols))}
        fp_raw = out.get("dataset_fingerprint")
        if isinstance(fp_raw, str) and fp_raw:
            try:
                out["dataset_fingerprint"] = json.loads(fp_raw)
            except json.JSONDecodeError:
                pass
        from .master_fingerprint import normalize_metadata_status

        out["metadata_status"] = normalize_metadata_status(out.get("metadata_status"))
        return out

    def read_dataset_fingerprint(self) -> dict[str, Any]:
        meta = self.read_master_meta_dict()
        fp = meta.get("dataset_fingerprint")
        if isinstance(fp, dict):
            return fp
        from .master_fingerprint import build_dataset_fingerprint_blob

        return build_dataset_fingerprint_blob(
            sampling_interval_sec=meta.get("sampling_interval_sec"),
            feature_registry_version=meta.get("feature_registry_version"),
            feature_count=meta.get("feature_count"),
            target_count=meta.get("target_count"),
            schema_hash=meta.get("schema_hash"),
            feature_hash=meta.get("feature_hash"),
            target_hash=meta.get("target_hash"),
            builder_version=meta.get("builder_version"),
            market=meta.get("market"),
        )

    def sync_schema_meta_fields(self) -> None:
        """Push feature/target/schema fields from dataset_meta JSON into master_dataset_meta."""
        fields = self._schema_fields_from_dataset_meta()
        if not any(fields.values()):
            return
        now = _utc_now()
        self.conn.execute(
            """
            UPDATE master_dataset_meta SET
                feature_count = COALESCE(?, feature_count),
                target_count = COALESCE(?, target_count),
                schema_hash = COALESCE(?, schema_hash),
                sampling_interval_sec = COALESCE(?, sampling_interval_sec),
                updated_at = ?
            WHERE id = 1
            """,
            (
                fields.get("feature_count"),
                fields.get("target_count"),
                fields.get("schema_hash"),
                fields.get("sampling_interval_sec"),
                now,
            ),
        )
        self.conn.commit()

    def _maybe_backfill_metadata(self) -> None:
        meta = self.read_master_meta()
        if meta.total_rows > 0 and self.count_metadata_days() > 0:
            return
        row = self.conn.execute("SELECT 1 FROM samples LIMIT 1").fetchone()
        if not row:
            return
        self.refresh_metadata_from_samples(reason="BACKFILL")

    def _aggregate_day_stats(self, trading_day: str) -> dict[str, Any]:
        sample_cols = {
            row[1]
            for row in self.conn.execute("PRAGMA table_info(samples)").fetchall()
        }
        expiry_sql = "COUNT(DISTINCT expiry)" if "expiry" in sample_cols else "NULL"
        row = self.conn.execute(
            f"""
            SELECT COUNT(*) AS row_count,
                   COUNT(DISTINCT token) AS token_count,
                   {expiry_sql} AS expiry_count,
                   MIN(timestamp) AS first_timestamp,
                   MAX(timestamp) AS last_timestamp
            FROM samples
            WHERE trading_day = ?
            """,
            (trading_day,),
        ).fetchone()
        if not row:
            return {
                "row_count": 0,
                "token_count": None,
                "expiry_count": None,
                "dominant_expiry": None,
                "is_expiry_day": 0,
                "first_timestamp": None,
                "last_timestamp": None,
            }
        dominant_expiry: str | None = None
        is_expiry_day = 0
        if "expiry" in sample_cols:
            dom = self.conn.execute(
                """
                SELECT expiry, COUNT(*) AS n
                FROM samples
                WHERE trading_day = ? AND expiry IS NOT NULL
                GROUP BY expiry
                ORDER BY n DESC
                LIMIT 1
                """,
                (trading_day,),
            ).fetchone()
            if dom and dom[0] is not None:
                dominant_expiry = str(dom[0])
            same = self.conn.execute(
                """
                SELECT 1 FROM samples
                WHERE trading_day = ? AND expiry = ?
                LIMIT 1
                """,
                (trading_day, trading_day),
            ).fetchone()
            is_expiry_day = 1 if same else 0
        return {
            "row_count": int(row[0] or 0),
            "token_count": int(row[1]) if row[1] is not None else None,
            "expiry_count": int(row[2]) if row[2] is not None else None,
            "dominant_expiry": dominant_expiry,
            "is_expiry_day": is_expiry_day,
            "first_timestamp": float(row[3]) if row[3] is not None else None,
            "last_timestamp": float(row[4]) if row[4] is not None else None,
        }

    def _increment_meta_after_day_add(self, row_count: int) -> None:
        now = _utc_now()
        meta = self.read_master_meta()
        version = int(meta.metadata_version or 0) + 1
        created = meta.created_at or now
        fields = self._schema_fields_from_dataset_meta()
        self.conn.execute(
            """
            UPDATE master_dataset_meta SET
                metadata_version = ?,
                total_rows = total_rows + ?,
                total_days = total_days + 1,
                feature_count = COALESCE(?, feature_count),
                target_count = COALESCE(?, target_count),
                schema_hash = COALESCE(?, schema_hash),
                sampling_interval_sec = COALESCE(?, sampling_interval_sec),
                created_at = COALESCE(created_at, ?),
                updated_at = ?,
                metadata_status = 'VALID'
            WHERE id = 1
            """,
            (
                version,
                int(row_count),
                fields.get("feature_count"),
                fields.get("target_count"),
                fields.get("schema_hash"),
                fields.get("sampling_interval_sec"),
                created,
                now,
            ),
        )

    def _decrement_meta_after_day_delete(self, deleted_rows: int) -> None:
        now = _utc_now()
        meta = self.read_master_meta()
        version = int(meta.metadata_version or 0) + 1
        self.conn.execute(
            """
            UPDATE master_dataset_meta SET
                metadata_version = ?,
                total_rows = MAX(0, total_rows - ?),
                total_days = MAX(0, total_days - 1),
                updated_at = ?,
                metadata_status = 'VALID'
            WHERE id = 1
            """,
            (version, int(deleted_rows), now),
        )

    def _recompute_meta_totals(self) -> None:
        row = self.conn.execute(
            "SELECT COALESCE(SUM(row_count), 0), COUNT(*) FROM master_dataset_days"
        ).fetchone()
        total_rows = int(row[0] or 0) if row else 0
        total_days = int(row[1] or 0) if row else 0
        now = _utc_now()
        meta = self.read_master_meta()
        version = int(meta.metadata_version or 0) + 1
        self.conn.execute(
            """
            UPDATE master_dataset_meta SET
                metadata_version = ?,
                total_rows = ?,
                total_days = ?,
                updated_at = ?,
                metadata_status = 'VALID'
            WHERE id = 1
            """,
            (version, total_rows, total_days, now),
        )

    def _set_metadata_status(self, status: str, *, commit: bool = False) -> None:
        from .master_fingerprint import normalize_metadata_status

        now = _utc_now()
        self.conn.execute(
            """
            UPDATE master_dataset_meta SET
                metadata_status = ?,
                last_modified = ?
            WHERE id = 1
            """,
            (normalize_metadata_status(status), now),
        )
        if commit:
            self.conn.commit()

    def _sync_fingerprint_from_meta_fields(self, fields: dict[str, Any]) -> None:
        from .master_fingerprint import build_dataset_fingerprint_blob, fingerprint_json_blob

        meta = self.read_master_meta_dict()
        fp = build_dataset_fingerprint_blob(
            sampling_interval_sec=fields.get("sampling_interval_sec") or meta.get("sampling_interval_sec"),
            feature_registry_version=meta.get("feature_registry_version"),
            feature_count=fields.get("feature_count") or meta.get("feature_count"),
            target_count=fields.get("target_count") or meta.get("target_count"),
            schema_hash=fields.get("schema_hash") or meta.get("schema_hash"),
            feature_hash=meta.get("feature_hash"),
            target_hash=meta.get("target_hash"),
            builder_version=meta.get("builder_version"),
            market=meta.get("market"),
        )
        self.conn.execute(
            "UPDATE master_dataset_meta SET dataset_fingerprint = ? WHERE id = 1",
            (fingerprint_json_blob(fp),),
        )

    def _append_meta_history(
        self,
        reason: str,
        *,
        version: int | None = None,
        total_rows: int | None = None,
        total_days: int | None = None,
    ) -> None:
        meta = self.read_master_meta()
        self.conn.execute(
            """
            INSERT INTO master_dataset_meta_history (
                ts, reason, metadata_version, total_rows, total_days
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                _utc_now(),
                str(reason),
                int(version if version is not None else meta.metadata_version or 0),
                int(total_rows if total_rows is not None else meta.total_rows or 0),
                int(total_days if total_days is not None else meta.total_days or 0),
            ),
        )

    def _refresh_meta_file_sizes(self) -> None:
        sizes = self._file_sizes()
        now = _utc_now()
        self.conn.execute(
            """
            UPDATE master_dataset_meta SET
                database_size = ?,
                wal_size = ?,
                updated_at = ?
            WHERE id = 1
            """,
            (sizes.get("db_bytes"), sizes.get("wal_bytes"), now),
        )

    def _file_sizes(self) -> dict[str, int]:
        db_bytes = os.path.getsize(self.db_path) if os.path.isfile(self.db_path) else 0
        wal_bytes = os.path.getsize(f"{self.db_path}-wal") if os.path.isfile(f"{self.db_path}-wal") else 0
        return {"db_bytes": db_bytes, "wal_bytes": wal_bytes}

    def _schema_fields_from_dataset_meta(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "feature_count": None,
            "target_count": None,
            "schema_hash": None,
            "sampling_interval_sec": None,
        }
        cfg = self.get_meta("master_config")
        if isinstance(cfg, dict):
            if cfg.get("feature_count") is not None:
                out["feature_count"] = int(cfg["feature_count"])
            if cfg.get("target_count") is not None:
                out["target_count"] = int(cfg["target_count"])
            if cfg.get("sampling_interval_sec") is not None:
                out["sampling_interval_sec"] = int(cfg["sampling_interval_sec"])
            if cfg.get("schema_hash"):
                out["schema_hash"] = str(cfg["schema_hash"])
        schema = self.get_meta("build_schema")
        if isinstance(schema, dict):
            if schema.get("feature_count") is not None:
                out["feature_count"] = int(schema["feature_count"])
            if schema.get("target_count") is not None:
                out["target_count"] = int(schema["target_count"])
            if schema.get("schema_hash"):
                out["schema_hash"] = str(schema["schema_hash"])
        return out
