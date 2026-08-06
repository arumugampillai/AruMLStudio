"""Phase A — Proposal → Template → Job persistence."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from chain_replay_ml.research_lab.paths import research_sessions_db_path

JOB_STEPS = (
    "preparing",
    "cloning",
    "training",
    "walk_forward",
    "simulation",
    "research_report",
    "knowledge_base",
    "complete",
)

STEP_EXPLANATIONS: dict[str, str] = {
    "preparing": (
        "Validating the frozen template, baseline prediction/strategy runs, and which phases "
        "(strategy simulation, model training, dataset migration) will execute."
    ),
    "cloning": (
        "Cloning the champion strategy version and/or model training config. "
        "Champions are never overwritten — each experiment gets its own version."
    ),
    "training": (
        "Training the model with walk-forward validation. Feature optimization and HPO run here. "
        "This step usually takes the longest."
    ),
    "walk_forward": (
        "Walk-forward folds were evaluated during training. Metrics from each fold feed the new prediction run."
    ),
    "simulation": (
        "Replaying strategy entry/exit rules on the prediction run to produce trade metrics "
        "(profit factor, win rate, trade count)."
    ),
    "research_report": (
        "Building a full research report comparing this run against baseline and surfacing recommendations."
    ),
    "knowledge_base": (
        "Auto-closure: verdict vs baseline, root cause, information gain score, knowledge-base extraction, "
        "and suggested isolated follow-up experiments."
    ),
    "complete": "All pipeline steps finished. Verdict and outputs are ready to review.",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pipeline_db_path(data_dir: str) -> str:
    return os.path.join(os.path.dirname(research_sessions_db_path(data_dir)), "experiment_pipeline.db")


class ExperimentPipelineStore:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.db_path = _pipeline_db_path(data_dir)
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

    def __enter__(self) -> ExperimentPipelineStore:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("ExperimentPipelineStore not open")
        return self._conn

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiment_proposals (
                proposal_id TEXT PRIMARY KEY,
                proposal_number INTEGER NOT NULL,
                status TEXT NOT NULL,
                research_report_id TEXT,
                prediction_run_id TEXT,
                strategy_run_id TEXT,
                model_id TEXT,
                strategy_label TEXT,
                goal TEXT,
                tags_json TEXT,
                available_json TEXT NOT NULL,
                selected_json TEXT NOT NULL,
                baseline_json TEXT,
                score_json TEXT,
                template_id TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_proposals_status
                ON experiment_proposals(status, created_at DESC);

            CREATE TABLE IF NOT EXISTS experiment_templates (
                template_id TEXT PRIMARY KEY,
                template_number INTEGER NOT NULL,
                proposal_id TEXT,
                research_report_id TEXT,
                prediction_run_id TEXT,
                strategy_run_id TEXT,
                model_id TEXT,
                strategy_label TEXT,
                goal TEXT,
                tags_json TEXT NOT NULL,
                accepted_changes_json TEXT NOT NULL,
                routing_json TEXT,
                baseline_json TEXT,
                score_json TEXT,
                status TEXT NOT NULL DEFAULT 'ready',
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_templates_number
                ON experiment_templates(template_number DESC);

            CREATE TABLE IF NOT EXISTS experiment_jobs (
                job_id TEXT PRIMARY KEY,
                job_number INTEGER NOT NULL,
                template_id TEXT NOT NULL,
                status TEXT NOT NULL,
                current_step TEXT,
                progress_json TEXT,
                overrides_json TEXT,
                outputs_json TEXT,
                results_json TEXT,
                comparison_json TEXT,
                error TEXT,
                started_at TEXT,
                completed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_jobs_template
                ON experiment_jobs(template_id, created_at DESC);
            """
        )
        self._migrate_columns()

    def _migrate_columns(self) -> None:
        migrations = {
            "experiment_proposals": (
                ("campaign_id", "TEXT"),
                ("program_id", "TEXT"),
                ("objective_score_json", "TEXT"),
            ),
            "experiment_templates": (
                ("campaign_id", "TEXT"),
                ("program_id", "TEXT"),
                ("objective_score_json", "TEXT"),
            ),
            "experiment_jobs": (
                ("campaign_id", "TEXT"),
                ("program_id", "TEXT"),
            ),
        }
        for table, cols in migrations.items():
            existing = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})").fetchall()}
            for name, col_type in cols:
                if name not in existing:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {col_type}")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_proposals_campaign ON experiment_proposals(campaign_id, status)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_templates_campaign ON experiment_templates(campaign_id, template_number DESC)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_campaign ON experiment_jobs(campaign_id, created_at DESC)"
        )
        self.conn.commit()

    def _next_number(self, table: str, column: str) -> int:
        row = self.conn.execute(
            f"SELECT COALESCE(MAX({column}), 0) + 1 AS n FROM {table}",
        ).fetchone()
        return int(row["n"] if row else 1)

    def save_proposal(self, proposal: dict[str, Any]) -> dict[str, Any]:
        pid = str(proposal.get("proposal_id") or uuid.uuid4().hex)
        now = _utc_now()
        num = int(proposal.get("proposal_number") or self._next_number("experiment_proposals", "proposal_number"))
        payload = dict(proposal)
        payload["proposal_id"] = pid
        payload["proposal_number"] = num
        payload.setdefault("status", "draft")
        payload.setdefault("created_at", now)
        payload["updated_at"] = now
        self.conn.execute(
            """
            INSERT OR REPLACE INTO experiment_proposals (
                proposal_id, proposal_number, status, research_report_id,
                prediction_run_id, strategy_run_id, model_id, strategy_label,
                goal, tags_json, available_json, selected_json, baseline_json,
                score_json, template_id, campaign_id, program_id, objective_score_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pid, num, payload.get("status"),
                payload.get("research_report_id"),
                payload.get("prediction_run_id"),
                payload.get("strategy_run_id"),
                payload.get("model_id"),
                payload.get("strategy_label"),
                payload.get("goal"),
                json.dumps(payload.get("tags") or [], default=str),
                json.dumps(payload.get("available_recommendations") or [], default=str),
                json.dumps(payload.get("selected_recommendations") or [], default=str),
                json.dumps(payload.get("baseline") or {}, default=str),
                json.dumps(payload.get("score") or {}, default=str),
                payload.get("template_id"),
                payload.get("campaign_id"),
                payload.get("program_id"),
                json.dumps(payload.get("objective_score") or {}, default=str) if payload.get("objective_score") else None,
                payload.get("created_at"),
                payload["updated_at"],
            ),
        )
        self.conn.commit()
        return self._load_proposal(pid) or payload

    def _load_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM experiment_proposals WHERE proposal_id = ?",
            (proposal_id,),
        ).fetchone()
        if not row:
            return None
        return self._proposal_row(row)

    def _proposal_row(self, row: sqlite3.Row) -> dict[str, Any]:
        doc = dict(row)
        doc["tags"] = json.loads(doc.pop("tags_json") or "[]")
        doc["available_recommendations"] = json.loads(doc.pop("available_json") or "[]")
        doc["selected_recommendations"] = json.loads(doc.pop("selected_json") or "[]")
        doc["baseline"] = json.loads(doc.pop("baseline_json") or "{}")
        doc["score"] = json.loads(doc.pop("score_json") or "{}")
        obj_raw = doc.pop("objective_score_json", None)
        doc["objective_score"] = json.loads(obj_raw) if obj_raw else None
        return doc

    def list_proposals(
        self,
        *,
        status: str | None = "draft",
        campaign_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if campaign_id:
            clauses.append("campaign_id = ?")
            params.append(campaign_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(limit, 200)))
        rows = self.conn.execute(
            f"SELECT * FROM experiment_proposals {where} ORDER BY proposal_number DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._proposal_row(r) for r in rows]

    def save_template(self, template: dict[str, Any]) -> dict[str, Any]:
        tid = str(template.get("template_id") or uuid.uuid4().hex)
        now = _utc_now()
        num = int(template.get("template_number") or self._next_number("experiment_templates", "template_number"))
        payload = dict(template)
        payload["template_id"] = tid
        payload["template_number"] = num
        payload.setdefault("status", "ready")
        payload.setdefault("created_at", now)
        self.conn.execute(
            """
            INSERT OR REPLACE INTO experiment_templates (
                template_id, template_number, proposal_id, research_report_id,
                prediction_run_id, strategy_run_id, model_id, strategy_label,
                goal, tags_json, accepted_changes_json, routing_json, baseline_json,
                score_json, status, campaign_id, program_id, objective_score_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tid, num, payload.get("proposal_id"),
                payload.get("research_report_id"),
                payload.get("prediction_run_id"),
                payload.get("strategy_run_id"),
                payload.get("model_id"),
                payload.get("strategy_label"),
                payload.get("goal"),
                json.dumps(payload.get("tags") or [], default=str),
                json.dumps(payload.get("accepted_changes") or [], default=str),
                json.dumps(payload.get("routing") or {}, default=str),
                json.dumps(payload.get("baseline") or {}, default=str),
                json.dumps(payload.get("score") or {}, default=str),
                payload.get("status"),
                payload.get("campaign_id"),
                payload.get("program_id"),
                json.dumps(payload.get("objective_score") or {}, default=str) if payload.get("objective_score") else None,
                payload.get("created_at"),
            ),
        )
        self.conn.commit()
        return self._load_template(tid) or payload

    def _load_template(self, template_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM experiment_templates WHERE template_id = ?",
            (template_id,),
        ).fetchone()
        if not row:
            return None
        return self._template_row(row)

    def _template_row(self, row: sqlite3.Row) -> dict[str, Any]:
        doc = dict(row)
        doc["tags"] = json.loads(doc.pop("tags_json") or "[]")
        doc["accepted_changes"] = json.loads(doc.pop("accepted_changes_json") or "[]")
        doc["routing"] = json.loads(doc.pop("routing_json") or "{}")
        doc["baseline"] = json.loads(doc.pop("baseline_json") or "{}")
        doc["score"] = json.loads(doc.pop("score_json") or "{}")
        obj_raw = doc.pop("objective_score_json", None)
        doc["objective_score"] = json.loads(obj_raw) if obj_raw else None
        return doc

    def list_templates(
        self,
        *,
        campaign_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses = ["status != 'archived'"]
        params: list[Any] = []
        if campaign_id:
            clauses.append("campaign_id = ?")
            params.append(campaign_id)
        params.append(max(1, min(limit, 200)))
        rows = self.conn.execute(
            f"""
            SELECT * FROM experiment_templates
            WHERE {' AND '.join(clauses)}
            ORDER BY template_number DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [self._template_row(r) for r in rows]

    def save_job(self, job: dict[str, Any]) -> dict[str, Any]:
        jid = str(job.get("job_id") or uuid.uuid4().hex)
        now = _utc_now()
        num = int(job.get("job_number") or self._next_number("experiment_jobs", "job_number"))
        payload = dict(job)
        payload["job_id"] = jid
        payload["job_number"] = num
        payload.setdefault("created_at", now)
        payload["updated_at"] = now
        self.conn.execute(
            """
            INSERT OR REPLACE INTO experiment_jobs (
                job_id, job_number, template_id, status, current_step,
                progress_json, overrides_json, outputs_json, results_json,
                comparison_json, error, started_at, completed_at,
                campaign_id, program_id, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                jid, num, payload.get("template_id"), payload.get("status"),
                payload.get("current_step"),
                json.dumps(payload.get("progress") or {}, default=str),
                json.dumps(payload.get("overrides") or {}, default=str),
                json.dumps(payload.get("outputs") or {}, default=str),
                json.dumps(payload.get("results") or {}, default=str),
                json.dumps(payload.get("comparison") or {}, default=str),
                payload.get("error"),
                payload.get("started_at"),
                payload.get("completed_at"),
                payload.get("campaign_id"),
                payload.get("program_id"),
                payload.get("created_at"),
                payload["updated_at"],
            ),
        )
        self.conn.commit()
        return self._load_job(jid) or payload

    def _load_job(self, job_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM experiment_jobs WHERE job_id = ?",
            (job_id,),
        ).fetchone()
        if not row:
            return None
        return self._job_row(row)

    def _job_row(self, row: sqlite3.Row) -> dict[str, Any]:
        doc = dict(row)
        doc["progress"] = json.loads(doc.pop("progress_json") or "{}")
        doc["overrides"] = json.loads(doc.pop("overrides_json") or "{}")
        doc["outputs"] = json.loads(doc.pop("outputs_json") or "{}")
        doc["results"] = json.loads(doc.pop("results_json") or "{}")
        doc["comparison"] = json.loads(doc.pop("comparison_json") or "{}")
        return doc

    def list_jobs(
        self,
        *,
        template_id: str | None = None,
        campaign_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if template_id:
            clauses.append("template_id = ?")
            params.append(template_id)
        if campaign_id:
            clauses.append("campaign_id = ?")
            params.append(campaign_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(limit, 200)))
        rows = self.conn.execute(
            f"SELECT * FROM experiment_jobs {where} ORDER BY job_number DESC LIMIT ?",
            params,
        ).fetchall()
        return [self._job_row(r) for r in rows]

    def count_jobs_for_template(self, template_id: str) -> dict[str, int]:
        rows = self.conn.execute(
            """
            SELECT status, COUNT(*) AS n FROM experiment_jobs
            WHERE template_id = ? GROUP BY status
            """,
            (template_id,),
        ).fetchall()
        counts = {str(r["status"]): int(r["n"]) for r in rows}
        total = sum(counts.values())
        completed = counts.get("complete", 0) + counts.get("completed", 0)
        failed = counts.get("failed", 0)
        return {"total": total, "completed": completed, "failed": failed, "running": max(0, total - completed - failed)}

    def get_running_job_for_campaign(self, campaign_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            """
            SELECT * FROM experiment_jobs
            WHERE campaign_id = ? AND status = 'running'
            ORDER BY job_number DESC
            LIMIT 1
            """,
            (campaign_id,),
        ).fetchone()
        return self._job_row(row) if row else None

    def list_running_jobs(self, *, campaign_id: str | None = None) -> list[dict[str, Any]]:
        if campaign_id:
            rows = self.conn.execute(
                "SELECT * FROM experiment_jobs WHERE status = 'running' AND campaign_id = ?",
                (campaign_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM experiment_jobs WHERE status = 'running' ORDER BY job_number DESC",
            ).fetchall()
        return [self._job_row(r) for r in rows]
