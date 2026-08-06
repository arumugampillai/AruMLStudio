"""Phase D1 — Research Program and Campaign persistence."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from chain_replay_ml.research_lab.paths import research_sessions_db_path

from .research_objective import (
    CAMPAIGN_STATUSES,
    PROGRAM_STATUSES,
    default_budget,
    default_failure_criteria,
    default_objective,
    default_stopping_policy,
    default_success_criteria,
    merge_budget,
    merge_objective,
    merge_stopping,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _research_program_db_path(data_dir: str) -> str:
    return os.path.join(os.path.dirname(research_sessions_db_path(data_dir)), "research_program.db")


class ResearchProgramStore:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.db_path = _research_program_db_path(data_dir)
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

    def __enter__(self) -> ResearchProgramStore:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("ResearchProgramStore not open")
        return self._conn

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS research_programs (
                program_id TEXT PRIMARY KEY,
                program_number INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                status TEXT NOT NULL,
                importance TEXT NOT NULL DEFAULT 'medium',
                objective_json TEXT NOT NULL,
                budget_json TEXT NOT NULL,
                champion_json TEXT,
                retired_reason TEXT,
                stats_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_programs_status
                ON research_programs(status, program_number DESC);

            CREATE TABLE IF NOT EXISTS research_campaigns (
                campaign_id TEXT PRIMARY KEY,
                campaign_number INTEGER NOT NULL,
                program_id TEXT NOT NULL,
                name TEXT NOT NULL,
                research_question TEXT NOT NULL,
                description TEXT,
                status TEXT NOT NULL,
                importance TEXT,
                objective_json TEXT,
                budget_json TEXT,
                memory_json TEXT,
                dependencies_json TEXT,
                best_template_id TEXT,
                retired_reason TEXT,
                budget_used_json TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (program_id) REFERENCES research_programs(program_id)
            );
            CREATE INDEX IF NOT EXISTS idx_campaigns_program
                ON research_campaigns(program_id, campaign_number DESC);
            CREATE INDEX IF NOT EXISTS idx_campaigns_status
                ON research_campaigns(status, updated_at DESC);
            """
        )
        self._migrate_v2_columns()

    def _migrate_v2_columns(self) -> None:
        program_cols = {
            "program_type": "TEXT NOT NULL DEFAULT 'strategy'",
            "stopping_json": "TEXT",
        }
        campaign_cols = {
            "hypothesis": "TEXT",
            "success_criteria_json": "TEXT",
            "failure_criteria_json": "TEXT",
            "stopping_json": "TEXT",
            "manifest_json": "TEXT",
        }
        existing_prog = {r[1] for r in self.conn.execute("PRAGMA table_info(research_programs)")}
        for col, typedef in program_cols.items():
            if col not in existing_prog:
                self.conn.execute(f"ALTER TABLE research_programs ADD COLUMN {col} {typedef}")
        existing_camp = {r[1] for r in self.conn.execute("PRAGMA table_info(research_campaigns)")}
        for col, typedef in campaign_cols.items():
            if col not in existing_camp:
                self.conn.execute(f"ALTER TABLE research_campaigns ADD COLUMN {col} {typedef}")
        self.conn.commit()

    def _next_number(self, table: str, column: str) -> int:
        row = self.conn.execute(
            f"SELECT COALESCE(MAX({column}), 0) + 1 AS n FROM {table}",
        ).fetchone()
        return int(row["n"] if row else 1)

    def save_program(self, program: dict[str, Any]) -> dict[str, Any]:
        pid = str(program.get("program_id") or uuid.uuid4().hex)
        now = _utc_now()
        num = int(program.get("program_number") or self._next_number("research_programs", "program_number"))
        payload = dict(program)
        payload["program_id"] = pid
        payload["program_number"] = num
        payload.setdefault("status", "draft")
        payload.setdefault("importance", "medium")
        payload.setdefault("objective", default_objective())
        payload.setdefault("budget", default_budget())
        payload.setdefault("program_type", "strategy")
        payload.setdefault("stopping", default_stopping_policy())
        payload.setdefault("created_at", now)
        payload["updated_at"] = now
        self.conn.execute(
            """
            INSERT OR REPLACE INTO research_programs (
                program_id, program_number, name, description, status, importance,
                program_type, objective_json, budget_json, stopping_json,
                champion_json, retired_reason, stats_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pid,
                num,
                payload.get("name"),
                payload.get("description"),
                payload.get("status"),
                payload.get("importance"),
                payload.get("program_type") or "strategy",
                json.dumps(payload.get("objective") or {}, default=str),
                json.dumps(payload.get("budget") or {}, default=str),
                json.dumps(payload.get("stopping") or {}, default=str),
                json.dumps(payload.get("champion") or {}, default=str) if payload.get("champion") else None,
                payload.get("retired_reason"),
                json.dumps(payload.get("stats") or {}, default=str),
                payload.get("created_at"),
                payload["updated_at"],
            ),
        )
        self.conn.commit()
        return self._load_program(pid) or payload

    def _program_row(self, row: sqlite3.Row) -> dict[str, Any]:
        doc = dict(row)
        doc["objective"] = json.loads(doc.pop("objective_json") or "{}")
        doc["budget"] = json.loads(doc.pop("budget_json") or "{}")
        stop_raw = doc.pop("stopping_json", None)
        doc["stopping"] = json.loads(stop_raw) if stop_raw else default_stopping_policy()
        champion_raw = doc.pop("champion_json", None)
        doc["champion"] = json.loads(champion_raw) if champion_raw else None
        doc["stats"] = json.loads(doc.pop("stats_json") or "{}")
        doc.setdefault("program_type", "strategy")
        return doc

    def _load_program(self, program_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM research_programs WHERE program_id = ?",
            (program_id,),
        ).fetchone()
        if not row:
            return None
        return self._program_row(row)

    def list_programs(self, *, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if status:
            rows = self.conn.execute(
                """
                SELECT * FROM research_programs
                WHERE status = ?
                ORDER BY program_number DESC
                LIMIT ?
                """,
                (status, max(1, min(limit, 200))),
            ).fetchall()
        else:
            rows = self.conn.execute(
                """
                SELECT * FROM research_programs
                WHERE status NOT IN ('retired', 'archived')
                ORDER BY program_number DESC
                LIMIT ?
                """,
                (max(1, min(limit, 200)),),
            ).fetchall()
        return [self._program_row(r) for r in rows]

    def save_campaign(self, campaign: dict[str, Any]) -> dict[str, Any]:
        cid = str(campaign.get("campaign_id") or uuid.uuid4().hex)
        now = _utc_now()
        num = int(campaign.get("campaign_number") or self._next_number("research_campaigns", "campaign_number"))
        payload = dict(campaign)
        payload["campaign_id"] = cid
        payload["campaign_number"] = num
        payload.setdefault("status", "created")
        payload.setdefault("memory", {})
        payload.setdefault("dependencies", [])
        payload.setdefault("budget_used", {})
        payload.setdefault("stopping", default_stopping_policy())
        payload.setdefault("success_criteria", default_success_criteria())
        payload.setdefault("failure_criteria", default_failure_criteria())
        payload.setdefault("created_at", now)
        payload["updated_at"] = now
        self.conn.execute(
            """
            INSERT OR REPLACE INTO research_campaigns (
                campaign_id, campaign_number, program_id, name, research_question,
                hypothesis, description, status, importance, objective_json, budget_json,
                success_criteria_json, failure_criteria_json, stopping_json, manifest_json,
                memory_json, dependencies_json, best_template_id, retired_reason,
                budget_used_json, created_at, updated_at, started_at, completed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cid,
                num,
                payload.get("program_id"),
                payload.get("name"),
                payload.get("research_question"),
                payload.get("hypothesis"),
                payload.get("description"),
                payload.get("status"),
                payload.get("importance"),
                json.dumps(payload.get("objective") or {}, default=str) if payload.get("objective") else None,
                json.dumps(payload.get("budget") or {}, default=str) if payload.get("budget") else None,
                json.dumps(payload.get("success_criteria") or {}, default=str),
                json.dumps(payload.get("failure_criteria") or {}, default=str),
                json.dumps(payload.get("stopping") or {}, default=str),
                json.dumps(payload.get("manifest") or {}, default=str) if payload.get("manifest") else None,
                json.dumps(payload.get("memory") or {}, default=str),
                json.dumps(payload.get("dependencies") or [], default=str),
                payload.get("best_template_id"),
                payload.get("retired_reason"),
                json.dumps(payload.get("budget_used") or {}, default=str),
                payload.get("created_at"),
                payload["updated_at"],
                payload.get("started_at"),
                payload.get("completed_at"),
            ),
        )
        self.conn.commit()
        return self._load_campaign(cid) or payload

    def _campaign_row(self, row: sqlite3.Row) -> dict[str, Any]:
        doc = dict(row)
        obj_raw = doc.pop("objective_json", None)
        doc["objective"] = json.loads(obj_raw) if obj_raw else None
        bud_raw = doc.pop("budget_json", None)
        doc["budget"] = json.loads(bud_raw) if bud_raw else None
        doc["success_criteria"] = json.loads(doc.pop("success_criteria_json") or "{}")
        doc["failure_criteria"] = json.loads(doc.pop("failure_criteria_json") or "{}")
        stop_raw = doc.pop("stopping_json", None)
        doc["stopping"] = json.loads(stop_raw) if stop_raw else default_stopping_policy()
        man_raw = doc.pop("manifest_json", None)
        doc["manifest"] = json.loads(man_raw) if man_raw else None
        doc["memory"] = json.loads(doc.pop("memory_json") or "{}")
        doc["dependencies"] = json.loads(doc.pop("dependencies_json") or "[]")
        doc["budget_used"] = json.loads(doc.pop("budget_used_json") or "{}")
        return doc

    def _load_campaign(self, campaign_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM research_campaigns WHERE campaign_id = ?",
            (campaign_id,),
        ).fetchone()
        if not row:
            return None
        return self._campaign_row(row)

    def list_campaigns(
        self,
        *,
        program_id: str | None = None,
        status: str | None = None,
        include_retired: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses = ["1=1"]
        params: list[Any] = []
        if program_id:
            clauses.append("program_id = ?")
            params.append(program_id)
        if status:
            clauses.append("status = ?")
            params.append(status)
        elif not include_retired:
            clauses.append("status != 'retired'")
        params.append(max(1, min(limit, 200)))
        sql = f"""
            SELECT * FROM research_campaigns
            WHERE {' AND '.join(clauses)}
            ORDER BY campaign_number DESC
            LIMIT ?
        """
        rows = self.conn.execute(sql, params).fetchall()
        return [self._campaign_row(r) for r in rows]

    def count_campaigns_for_program(self, program_id: str) -> dict[str, int]:
        rows = self.conn.execute(
            """
            SELECT status, COUNT(*) AS n
            FROM research_campaigns
            WHERE program_id = ?
            GROUP BY status
            """,
            (program_id,),
        ).fetchall()
        counts = {str(r["status"]): int(r["n"]) for r in rows}
        total = sum(counts.values())
        return {"total": total, **counts}

    def resolve_campaign_config(self, campaign_id: str) -> dict[str, Any] | None:
        campaign = self._load_campaign(campaign_id)
        if not campaign:
            return None
        program = self._load_program(str(campaign.get("program_id") or ""))
        if not program:
            return None
        objective = merge_objective(program.get("objective"), campaign.get("objective"))
        budget = merge_budget(program.get("budget"), campaign.get("budget"))
        stopping = merge_stopping(program.get("stopping"), campaign.get("stopping"))
        importance = campaign.get("importance") or program.get("importance") or "medium"
        return {
            "program": program,
            "campaign": campaign,
            "resolved_objective": objective,
            "resolved_budget": budget,
            "resolved_stopping": stopping,
            "importance": importance,
        }
