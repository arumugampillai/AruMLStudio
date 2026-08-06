"""Persist research reports for prediction runs."""

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


def _reports_db_path(data_dir: str) -> str:
    return os.path.join(os.path.dirname(research_sessions_db_path(data_dir)), "research_reports.db")


class ResearchReportStore:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.db_path = _reports_db_path(data_dir)
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

    def __enter__(self) -> ResearchReportStore:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("ResearchReportStore not open")
        return self._conn

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS research_reports (
                report_id TEXT PRIMARY KEY,
                prediction_run_id TEXT NOT NULL,
                strategy_run_id TEXT,
                model_id TEXT,
                grade TEXT,
                score INTEGER,
                trade_count INTEGER,
                created_at TEXT NOT NULL,
                report_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_research_reports_run
                ON research_reports(prediction_run_id, created_at DESC);
            """
        )

    def save_report(self, report: dict[str, Any]) -> dict[str, Any]:
        report_id = str(report.get("report_id") or uuid.uuid4().hex)
        exec_sum = report.get("executive_summary") or {}
        created_at = report.get("created_at") or _utc_now()
        payload = dict(report)
        payload["report_id"] = report_id
        payload["created_at"] = created_at
        self.conn.execute(
            """
            INSERT OR REPLACE INTO research_reports (
                report_id, prediction_run_id, strategy_run_id, model_id,
                grade, score, trade_count, created_at, report_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                report_id,
                str(report.get("prediction_run_id") or ""),
                report.get("strategy_run_id"),
                exec_sum.get("model_id"),
                exec_sum.get("overall_grade"),
                exec_sum.get("overall_score"),
                exec_sum.get("trade_count"),
                created_at,
                json.dumps(payload, default=str),
            ),
        )
        self.conn.commit()
        return payload

    def list_reports(
        self,
        *,
        prediction_run_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if prediction_run_id:
            clauses.append("prediction_run_id = ?")
            params.append(prediction_run_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(limit, 200)))
        rows = self.conn.execute(
            f"""
            SELECT report_id, prediction_run_id, strategy_run_id, model_id,
                   grade, score, trade_count, created_at
            FROM research_reports
            {where}
            ORDER BY created_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_report(self, report_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT report_json FROM research_reports WHERE report_id = ?",
            (report_id,),
        ).fetchone()
        if not row:
            return None
        return json.loads(row["report_json"])


def save_research_report(data_dir: str, report: dict[str, Any]) -> dict[str, Any]:
    with ResearchReportStore(data_dir) as store:
        return store.save_report(report)


def list_research_reports(
    data_dir: str,
    *,
    prediction_run_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    with ResearchReportStore(data_dir) as store:
        return store.list_reports(prediction_run_id=prediction_run_id, limit=limit)


def get_research_report(data_dir: str, report_id: str) -> dict[str, Any] | None:
    with ResearchReportStore(data_dir) as store:
        return store.get_report(report_id)


def resolve_research_report_id(data_dir: str, report_id: str) -> str | None:
    """Exact match first, then unique prefix match (e.g. first 8 chars from UI hint)."""
    rid = str(report_id or "").strip()
    if not rid:
        return None
    if get_research_report(data_dir, rid):
        return rid
    reports = list_research_reports(data_dir, limit=100)
    matches = [str(r.get("report_id") or "") for r in reports if str(r.get("report_id") or "").startswith(rid)]
    if len(matches) == 1:
        return matches[0]
    return None
