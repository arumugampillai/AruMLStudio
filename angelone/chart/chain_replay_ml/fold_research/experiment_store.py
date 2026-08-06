"""Persist planned and completed research experiments."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from chain_replay_ml.research_lab.paths import research_sessions_db_path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _experiments_db_path(data_dir: str) -> str:
    return os.path.join(os.path.dirname(research_sessions_db_path(data_dir)), "experiments.db")


class ExperimentStore:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.db_path = _experiments_db_path(data_dir)
        self._conn: sqlite3.Connection | None = None

    def open(self) -> None:
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> ExperimentStore:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("ExperimentStore not open")
        return self._conn

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id TEXT PRIMARY KEY,
                experiment_number INTEGER NOT NULL,
                title TEXT NOT NULL,
                goal TEXT,
                status TEXT NOT NULL,
                research_report_id TEXT,
                prediction_run_id TEXT,
                strategy_run_id TEXT,
                model_id TEXT,
                strategy_label TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                launched_at TEXT,
                completed_at TEXT,
                experiment_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_experiments_number
                ON experiments(experiment_number DESC);
            CREATE INDEX IF NOT EXISTS idx_experiments_report
                ON experiments(research_report_id, created_at DESC);
            """
        )

    def next_experiment_number(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(experiment_number), 0) + 1 AS n FROM experiments").fetchone()
        return int(row["n"] if row else 1)

    def save_experiment(self, experiment: dict[str, Any]) -> dict[str, Any]:
        exp_id = str(experiment.get("experiment_id") or uuid.uuid4().hex)
        now = _utc_now()
        number = int(experiment.get("experiment_number") or self.next_experiment_number())
        prov = experiment.get("provenance") or {}
        payload = dict(experiment)
        payload["experiment_id"] = exp_id
        payload["experiment_number"] = number
        payload.setdefault("title", f"Experiment #{number}")
        payload.setdefault("status", "pending")
        payload.setdefault("created_at", now)
        payload["updated_at"] = now

        self.conn.execute(
            """
            INSERT OR REPLACE INTO experiments (
                experiment_id, experiment_number, title, goal, status,
                research_report_id, prediction_run_id, strategy_run_id,
                model_id, strategy_label, created_at, updated_at,
                launched_at, completed_at, experiment_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                exp_id,
                number,
                payload.get("title"),
                payload.get("goal"),
                payload.get("status"),
                prov.get("research_report_id") or experiment.get("research_report_id"),
                prov.get("prediction_run_id") or experiment.get("prediction_run_id"),
                prov.get("strategy_run_id") or experiment.get("strategy_run_id"),
                prov.get("model_id") or experiment.get("model_id"),
                prov.get("strategy_label") or experiment.get("strategy_label"),
                payload.get("created_at"),
                payload.get("updated_at"),
                payload.get("launched_at"),
                payload.get("completed_at"),
                json.dumps(payload, default=str),
            ),
        )
        self.conn.commit()
        return payload

    def list_experiments(self, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT experiment_id, experiment_number, title, goal, status,
                   research_report_id, prediction_run_id, model_id, strategy_label,
                   created_at, completed_at, experiment_json
            FROM experiments
            ORDER BY experiment_number DESC
            LIMIT ?
            """,
            (max(1, min(limit, 200)),),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            doc = json.loads(row["experiment_json"])
            results = doc.get("results") or {}
            out.append({
                "experiment_id": row["experiment_id"],
                "experiment_number": row["experiment_number"],
                "title": row["title"],
                "goal": row["goal"],
                "status": row["status"],
                "research_report_id": row["research_report_id"],
                "prediction_run_id": row["prediction_run_id"],
                "model_id": row["model_id"],
                "strategy_label": row["strategy_label"],
                "created_at": row["created_at"],
                "completed_at": row["completed_at"],
                "result_grade": results.get("grade"),
                "result_pf_after": results.get("profit_factor_after"),
            })
        return out

    def get_experiment(self, experiment_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT experiment_json FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        if not row:
            return None
        return json.loads(row["experiment_json"])


def save_experiment(data_dir: str, experiment: dict[str, Any]) -> dict[str, Any]:
    with ExperimentStore(data_dir) as store:
        return store.save_experiment(experiment)


def list_experiments(data_dir: str, *, limit: int = 50) -> list[dict[str, Any]]:
    with ExperimentStore(data_dir) as store:
        return store.list_experiments(limit=limit)


def get_experiment(data_dir: str, experiment_id: str) -> dict[str, Any] | None:
    with ExperimentStore(data_dir) as store:
        return store.get_experiment(experiment_id)
