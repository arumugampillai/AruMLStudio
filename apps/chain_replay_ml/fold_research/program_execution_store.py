"""Program execution on a model — reusable program runs (Phase F1/F3)."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from typing import Any

from chain_replay_ml.research_lab.paths import research_sessions_db_path

from .research_objective import PROGRAM_RUN_STATUSES


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _program_run_db_path(data_dir: str) -> str:
    return os.path.join(os.path.dirname(research_sessions_db_path(data_dir)), "program_execution.db")


class ProgramExecutionStore:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.db_path = _program_run_db_path(data_dir)
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

    def __enter__(self) -> ProgramExecutionStore:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("ProgramExecutionStore not open")
        return self._conn

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS program_runs (
                run_id TEXT PRIMARY KEY,
                run_number INTEGER NOT NULL,
                model_id TEXT NOT NULL,
                program_id TEXT NOT NULL,
                prediction_run_id TEXT,
                strategy_run_id TEXT,
                research_report_id TEXT,
                status TEXT NOT NULL,
                checkpoint_json TEXT NOT NULL,
                manifest_json TEXT NOT NULL,
                event_log_json TEXT NOT NULL,
                summary_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_program_runs_model
                ON program_runs(model_id, run_number DESC);
            CREATE INDEX IF NOT EXISTS idx_program_runs_program
                ON program_runs(program_id, status);
            """
        )

    def _next_run_number(self) -> int:
        row = self.conn.execute("SELECT COALESCE(MAX(run_number), 0) + 1 AS n FROM program_runs").fetchone()
        return int(row["n"] if row else 1)

    def save_run(self, run: dict[str, Any]) -> dict[str, Any]:
        rid = str(run.get("run_id") or uuid.uuid4().hex)
        now = _utc_now()
        payload = dict(run)
        payload["run_id"] = rid
        payload.setdefault("run_number", self._next_run_number())
        payload.setdefault("status", "waiting")
        payload.setdefault("checkpoint", {})
        payload.setdefault("manifest", {})
        payload.setdefault("event_log", [])
        payload.setdefault("created_at", now)
        payload["updated_at"] = now
        self.conn.execute(
            """
            INSERT OR REPLACE INTO program_runs (
                run_id, run_number, model_id, program_id,
                prediction_run_id, strategy_run_id, research_report_id,
                status, checkpoint_json, manifest_json, event_log_json, summary_json,
                created_at, updated_at, started_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                rid,
                payload["run_number"],
                payload.get("model_id"),
                payload.get("program_id"),
                payload.get("prediction_run_id"),
                payload.get("strategy_run_id"),
                payload.get("research_report_id"),
                payload.get("status"),
                json.dumps(payload.get("checkpoint") or {}, default=str),
                json.dumps(payload.get("manifest") or {}, default=str),
                json.dumps(payload.get("event_log") or [], default=str),
                json.dumps(payload.get("summary") or {}, default=str) if payload.get("summary") else None,
                payload.get("created_at"),
                payload["updated_at"],
                payload.get("started_at"),
                payload.get("completed_at"),
            ),
        )
        self.conn.commit()
        return self.get_run(rid) or payload

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM program_runs WHERE run_id = ?", (run_id,)).fetchone()
        if not row:
            return None
        return self._row(row)

    def list_runs(
        self,
        *,
        model_id: str | None = None,
        program_id: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses = ["1=1"]
        params: list[Any] = []
        if model_id:
            clauses.append("model_id = ?")
            params.append(model_id)
        if program_id:
            clauses.append("program_id = ?")
            params.append(program_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        params.append(max(1, min(limit, 200)))
        rows = self.conn.execute(
            f"""
            SELECT * FROM program_runs
            WHERE {' AND '.join(clauses)}
            ORDER BY run_number DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._row(r) for r in rows]

    def append_event(self, run_id: str, event: dict[str, Any]) -> dict[str, Any] | None:
        run = self.get_run(run_id)
        if not run:
            return None
        log = list(run.get("event_log") or [])
        log.append({**event, "ts": event.get("ts") or _utc_now()})
        run["event_log"] = log[-500:]
        run["updated_at"] = _utc_now()
        return self.save_run(run)

    def _row(self, row: sqlite3.Row) -> dict[str, Any]:
        doc = dict(row)
        doc["checkpoint"] = json.loads(doc.pop("checkpoint_json") or "{}")
        doc["manifest"] = json.loads(doc.pop("manifest_json") or "{}")
        doc["event_log"] = json.loads(doc.pop("event_log_json") or "[]")
        summary_raw = doc.pop("summary_json", None)
        doc["summary"] = json.loads(summary_raw) if summary_raw else None
        return doc


def create_program_run(
    data_dir: str,
    *,
    model_id: str,
    program_id: str,
    prediction_run_id: str | None = None,
    strategy_run_id: str | None = None,
    research_report_id: str | None = None,
) -> dict[str, Any]:
    from .research_program_store import ResearchProgramStore

    if not str(model_id or "").strip():
        return {"ok": False, "error": "model_id is required"}
    with ResearchProgramStore(data_dir) as store:
        program = store._load_program(program_id)
    if not program:
        return {"ok": False, "error": "program not found"}

    campaigns = []
    with ResearchProgramStore(data_dir) as store:
        for c in store.list_campaigns(program_id=program_id, include_retired=False, limit=100):
            campaigns.append({
                "campaign_id": c.get("campaign_id"),
                "campaign_number": c.get("campaign_number"),
                "name": c.get("name"),
                "status": "waiting",
            })

    manifest = {
        "model_id": model_id,
        "program_id": program_id,
        "program_name": program.get("name"),
        "program_type": program.get("program_type") or "strategy",
        "campaigns": campaigns,
        "completed_campaigns": 0,
        "total_campaigns": len(campaigns),
    }

    with ProgramExecutionStore(data_dir) as store:
        saved = store.save_run({
            "model_id": model_id,
            "program_id": program_id,
            "prediction_run_id": prediction_run_id,
            "strategy_run_id": strategy_run_id,
            "research_report_id": research_report_id,
            "status": "waiting",
            "manifest": manifest,
            "checkpoint": {"current_campaign_index": 0},
            "event_log": [{"action": "program_run_created", "ts": _utc_now()}],
        })
    return {"ok": True, "run": saved}


def get_program_run(data_dir: str, run_id: str) -> dict[str, Any] | None:
    with ProgramExecutionStore(data_dir) as store:
        return store.get_run(run_id)


def list_program_runs_for_model(data_dir: str, model_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
    with ProgramExecutionStore(data_dir) as store:
        return store.list_runs(model_id=model_id, limit=limit)
