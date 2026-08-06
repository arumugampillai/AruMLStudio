"""SQLite registry for Prediction Runs, Folds, and Rows."""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

from .paths import prediction_runs_db_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PredictionRunStore:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.db_path = prediction_runs_db_path(data_dir)
        self._conn: sqlite3.Connection | None = None

    def open(self) -> None:
        import os

        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> PredictionRunStore:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("PredictionRunStore not open")
        return self._conn

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS prediction_runs (
                run_id TEXT PRIMARY KEY,
                model_id TEXT NOT NULL,
                model_version TEXT,
                dataset_name TEXT,
                target TEXT,
                dataset_fingerprint TEXT,
                feature_snapshot_hash TEXT,
                walk_forward_config_hash TEXT,
                training_config_hash TEXT,
                status TEXT NOT NULL DEFAULT 'running',
                run_kind TEXT NOT NULL DEFAULT 'walk_forward_production',
                created_at TEXT NOT NULL,
                finished_at TEXT,
                training_duration_sec REAL,
                prediction_count INTEGER DEFAULT 0,
                fold_count INTEGER DEFAULT 0,
                package_dir TEXT,
                meta_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_prediction_runs_model
                ON prediction_runs(model_id, created_at DESC);

            CREATE TABLE IF NOT EXISTS prediction_folds (
                fold_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                fold_number INTEGER NOT NULL,
                train_start INTEGER,
                train_end INTEGER,
                validation_start INTEGER,
                validation_end INTEGER,
                train_rows INTEGER,
                validation_rows INTEGER,
                mae REAL,
                rmse REAL,
                directional_accuracy_pct REAL,
                prediction_count INTEGER DEFAULT 0,
                meta_json TEXT,
                FOREIGN KEY (run_id) REFERENCES prediction_runs(run_id)
            );
            CREATE INDEX IF NOT EXISTS idx_prediction_folds_run
                ON prediction_folds(run_id, fold_number);

            CREATE TABLE IF NOT EXISTS prediction_rows (
                prediction_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                fold_id TEXT NOT NULL,
                row_index INTEGER NOT NULL,
                timestamp REAL,
                trading_day TEXT,
                token TEXT,
                strike REAL,
                option_type TEXT,
                spot REAL,
                ltp REAL,
                predicted_ltp REAL,
                actual_ltp REAL,
                prediction_error REAL,
                direction_correct INTEGER,
                confidence REAL,
                meta_json TEXT,
                FOREIGN KEY (run_id) REFERENCES prediction_runs(run_id),
                FOREIGN KEY (fold_id) REFERENCES prediction_folds(fold_id)
            );
            CREATE INDEX IF NOT EXISTS idx_prediction_rows_run_fold
                ON prediction_rows(run_id, fold_id, row_index);
            CREATE INDEX IF NOT EXISTS idx_prediction_rows_run_ts
                ON prediction_rows(run_id, timestamp);
            """
        )

    def create_run(self, doc: dict[str, Any]) -> dict[str, Any]:
        run_id = str(doc.get("run_id") or uuid.uuid4().hex)
        now = _utc_now()
        meta = doc.get("meta") or {}
        self.conn.execute(
            """
            INSERT INTO prediction_runs (
                run_id, model_id, model_version, dataset_name, target,
                dataset_fingerprint, feature_snapshot_hash, walk_forward_config_hash,
                training_config_hash, status, run_kind, created_at, package_dir, meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                doc["model_id"],
                doc.get("model_version"),
                doc.get("dataset_name"),
                doc.get("target"),
                doc.get("dataset_fingerprint"),
                doc.get("feature_snapshot_hash"),
                doc.get("walk_forward_config_hash"),
                doc.get("training_config_hash"),
                doc.get("status") or "running",
                doc.get("run_kind") or "walk_forward_production",
                now,
                doc.get("package_dir"),
                json.dumps(meta, default=str),
            ),
        )
        self.conn.commit()
        return self.get_run(run_id) or {"run_id": run_id}

    def finalize_run(
        self,
        run_id: str,
        *,
        status: str = "completed",
        training_duration_sec: float | None = None,
        prediction_count: int | None = None,
        fold_count: int | None = None,
    ) -> None:
        self.conn.execute(
            """
            UPDATE prediction_runs
            SET status = ?, finished_at = ?, training_duration_sec = ?,
                prediction_count = COALESCE(?, prediction_count),
                fold_count = COALESCE(?, fold_count)
            WHERE run_id = ?
            """,
            (status, _utc_now(), training_duration_sec, prediction_count, fold_count, run_id),
        )
        self.conn.commit()

    def insert_fold(self, doc: dict[str, Any]) -> str:
        fold_id = str(doc.get("fold_id") or f"{doc['run_id']}_fold_{doc['fold_number']}")
        self.conn.execute(
            """
            INSERT OR REPLACE INTO prediction_folds (
                fold_id, run_id, fold_number, train_start, train_end,
                validation_start, validation_end, train_rows, validation_rows,
                mae, rmse, directional_accuracy_pct, prediction_count, meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fold_id,
                doc["run_id"],
                int(doc["fold_number"]),
                doc.get("train_start"),
                doc.get("train_end"),
                doc.get("validation_start"),
                doc.get("validation_end"),
                doc.get("train_rows"),
                doc.get("validation_rows"),
                doc.get("mae"),
                doc.get("rmse"),
                doc.get("directional_accuracy_pct"),
                doc.get("prediction_count") or 0,
                json.dumps(doc.get("meta") or {}, default=str),
            ),
        )
        self.conn.commit()
        return fold_id

    def insert_rows_batch(self, rows: Sequence[dict[str, Any]]) -> int:
        if not rows:
            return 0
        self.conn.executemany(
            """
            INSERT OR REPLACE INTO prediction_rows (
                prediction_id, run_id, fold_id, row_index,
                timestamp, trading_day, token, strike, option_type,
                spot, ltp, predicted_ltp, actual_ltp, prediction_error,
                direction_correct, confidence, meta_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r["prediction_id"],
                    r["run_id"],
                    r["fold_id"],
                    r["row_index"],
                    r.get("timestamp"),
                    r.get("trading_day"),
                    r.get("token"),
                    r.get("strike"),
                    r.get("option_type"),
                    r.get("spot"),
                    r.get("ltp"),
                    r.get("predicted_ltp"),
                    r.get("actual_ltp"),
                    r.get("prediction_error"),
                    r.get("direction_correct"),
                    r.get("confidence"),
                    json.dumps(r.get("meta") or {}, default=str),
                )
                for r in rows
            ],
        )
        self.conn.commit()
        return len(rows)

    def _row_to_dict(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        d = dict(row)
        if d.get("meta_json"):
            try:
                d["meta"] = json.loads(d["meta_json"])
            except json.JSONDecodeError:
                d["meta"] = {}
            del d["meta_json"]
        return d

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM prediction_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return self._row_to_dict(row)

    def list_runs_for_model(self, model_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM prediction_runs
            WHERE model_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (model_id, limit),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows if r]

    def list_all_runs(self, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM prediction_runs
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows if r]

    def list_folds(self, run_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT * FROM prediction_folds
            WHERE run_id = ?
            ORDER BY fold_number
            """,
            (run_id,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows if r]

    def list_rows(
        self,
        run_id: str,
        *,
        fold_id: str | None = None,
        limit: int = 500,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        if fold_id:
            rows = self.conn.execute(
                """
                SELECT * FROM prediction_rows
                WHERE run_id = ? AND fold_id = ?
                ORDER BY row_index
                LIMIT ? OFFSET ?
                """,
                (run_id, fold_id, limit, offset),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT * FROM prediction_rows
                WHERE run_id = ?
                ORDER BY fold_id, row_index
                LIMIT ? OFFSET ?
                """,
                (run_id, limit, offset),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows if r]

    def list_all_rows(self, run_id: str, *, fold_id: str | None = None) -> list[dict[str, Any]]:
        if fold_id:
            rows = self.conn.execute(
                """
                SELECT * FROM prediction_rows
                WHERE run_id = ? AND fold_id = ?
                ORDER BY timestamp, row_index
                """,
                (run_id, fold_id),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT * FROM prediction_rows
                WHERE run_id = ?
                ORDER BY fold_id, timestamp, row_index
                """,
                (run_id,),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows if r]

    def get_prediction_row(self, prediction_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM prediction_rows WHERE prediction_id = ?",
            (prediction_id,),
        ).fetchone()
        return self._row_to_dict(row)

    def count_rows(self, run_id: str, *, fold_id: str | None = None) -> int:
        if fold_id:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM prediction_rows WHERE run_id = ? AND fold_id = ?",
                (run_id, fold_id),
            ).fetchone()
        else:
            row = self.conn.execute(
                "SELECT COUNT(*) FROM prediction_rows WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return int(row[0]) if row else 0
