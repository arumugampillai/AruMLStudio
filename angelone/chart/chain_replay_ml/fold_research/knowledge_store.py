"""Knowledge Base — evidence-backed finding storage (Phase 1)."""

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


def _knowledge_db_path(data_dir: str) -> str:
    return os.path.join(os.path.dirname(research_sessions_db_path(data_dir)), "knowledge_base.db")


class KnowledgeStore:
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir
        self.db_path = _knowledge_db_path(data_dir)
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

    def __enter__(self) -> KnowledgeStore:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("KnowledgeStore not open")
        return self._conn

    def _ensure_schema(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS knowledge_findings (
                finding_id TEXT PRIMARY KEY,
                finding_key TEXT NOT NULL UNIQUE,
                finding TEXT NOT NULL,
                category TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'candidate',
                confidence TEXT NOT NULL DEFAULT 'low',
                evidence_count INTEGER NOT NULL DEFAULT 0,
                supporting_count INTEGER NOT NULL DEFAULT 0,
                contradicting_count INTEGER NOT NULL DEFAULT 0,
                trade_count INTEGER NOT NULL DEFAULT 0,
                experiment_count INTEGER NOT NULL DEFAULT 0,
                markets_json TEXT,
                time_span_days INTEGER,
                last_confirmed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                metadata_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_findings_key ON knowledge_findings(finding_key);
            CREATE INDEX IF NOT EXISTS idx_findings_status ON knowledge_findings(status, confidence);

            CREATE TABLE IF NOT EXISTS finding_evidence (
                evidence_id TEXT PRIMARY KEY,
                finding_id TEXT NOT NULL,
                experiment_id TEXT,
                experiment_number INTEGER,
                research_report_id TEXT,
                prediction_run_id TEXT,
                strategy_run_id TEXT,
                model_id TEXT,
                trade_count INTEGER,
                pf_change REAL,
                win_rate_change REAL,
                supports_finding INTEGER NOT NULL,
                evidence_quality TEXT NOT NULL DEFAULT 'moderate',
                notes TEXT,
                recorded_at TEXT NOT NULL,
                evidence_json TEXT,
                FOREIGN KEY (finding_id) REFERENCES knowledge_findings(finding_id)
            );
            CREATE INDEX IF NOT EXISTS idx_evidence_finding
                ON finding_evidence(finding_id, recorded_at DESC);
            CREATE INDEX IF NOT EXISTS idx_evidence_experiment
                ON finding_evidence(experiment_id);

            CREATE TABLE IF NOT EXISTS finding_links (
                link_id TEXT PRIMARY KEY,
                finding_id TEXT NOT NULL,
                link_type TEXT NOT NULL,
                link_ref TEXT NOT NULL,
                link_label TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (finding_id) REFERENCES knowledge_findings(finding_id)
            );
            CREATE INDEX IF NOT EXISTS idx_links_finding ON finding_links(finding_id);
            CREATE INDEX IF NOT EXISTS idx_links_ref ON finding_links(link_type, link_ref);
            """
        )

    def get_finding_by_key(self, finding_key: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM knowledge_findings WHERE finding_key = ?",
            (finding_key,),
        ).fetchone()
        return dict(row) if row else None

    def get_finding(self, finding_id: str) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM knowledge_findings WHERE finding_id = ?",
            (finding_id,),
        ).fetchone()
        if not row:
            return None
        doc = dict(row)
        doc["markets"] = json.loads(doc.pop("markets_json") or "[]")
        doc["metadata"] = json.loads(doc.pop("metadata_json") or "{}")
        return doc

    def upsert_finding(
        self,
        *,
        finding_key: str,
        finding: str,
        category: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = _utc_now()
        existing = self.get_finding_by_key(finding_key)
        if existing:
            return existing
        finding_id = uuid.uuid4().hex
        self.conn.execute(
            """
            INSERT INTO knowledge_findings (
                finding_id, finding_key, finding, category, status, confidence,
                evidence_count, supporting_count, contradicting_count,
                trade_count, experiment_count, markets_json, time_span_days,
                last_confirmed_at, created_at, updated_at, metadata_json
            ) VALUES (?, ?, ?, ?, 'candidate', 'low', 0, 0, 0, 0, 0, '[]', NULL, NULL, ?, ?, ?)
            """,
            (
                finding_id,
                finding_key,
                finding,
                category,
                now,
                now,
                json.dumps(metadata or {}, default=str),
            ),
        )
        self.conn.commit()
        return self.get_finding(finding_id) or {}

    def add_evidence(self, finding_id: str, evidence: dict[str, Any]) -> dict[str, Any]:
        evidence_id = str(evidence.get("evidence_id") or uuid.uuid4().hex)
        now = evidence.get("recorded_at") or _utc_now()
        payload = dict(evidence)
        payload["evidence_id"] = evidence_id
        payload["recorded_at"] = now
        self.conn.execute(
            """
            INSERT INTO finding_evidence (
                evidence_id, finding_id, experiment_id, experiment_number,
                research_report_id, prediction_run_id, strategy_run_id, model_id,
                trade_count, pf_change, win_rate_change, supports_finding,
                evidence_quality, notes, recorded_at, evidence_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evidence_id,
                finding_id,
                payload.get("experiment_id"),
                payload.get("experiment_number"),
                payload.get("research_report_id"),
                payload.get("prediction_run_id"),
                payload.get("strategy_run_id"),
                payload.get("model_id"),
                payload.get("trade_count"),
                payload.get("pf_change"),
                payload.get("win_rate_change"),
                1 if payload.get("supports_finding") else 0,
                payload.get("evidence_quality") or "moderate",
                payload.get("notes"),
                now,
                json.dumps(payload, default=str),
            ),
        )
        self.conn.commit()
        self._recompute_finding_aggregates(finding_id)
        return payload

    def add_link(
        self,
        finding_id: str,
        *,
        link_type: str,
        link_ref: str,
        link_label: str | None = None,
    ) -> None:
        existing = self.conn.execute(
            """
            SELECT link_id FROM finding_links
            WHERE finding_id = ? AND link_type = ? AND link_ref = ?
            """,
            (finding_id, link_type, link_ref),
        ).fetchone()
        if existing:
            return
        self.conn.execute(
            """
            INSERT INTO finding_links (link_id, finding_id, link_type, link_ref, link_label, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (uuid.uuid4().hex, finding_id, link_type, link_ref, link_label, _utc_now()),
        )
        self.conn.commit()

    def _recompute_finding_aggregates(self, finding_id: str) -> None:
        rows = self.conn.execute(
            """
            SELECT supports_finding, trade_count, recorded_at
            FROM finding_evidence WHERE finding_id = ?
            ORDER BY recorded_at ASC
            """,
            (finding_id,),
        ).fetchall()
        if not rows:
            return

        evidence_count = len(rows)
        supporting = sum(1 for r in rows if r["supports_finding"])
        contradicting = evidence_count - supporting
        trade_count = sum(int(r["trade_count"] or 0) for r in rows)
        experiment_count = self.conn.execute(
            """
            SELECT COUNT(DISTINCT experiment_id) AS n
            FROM finding_evidence WHERE finding_id = ? AND experiment_id IS NOT NULL
            """,
            (finding_id,),
        ).fetchone()["n"]

        timestamps = [r["recorded_at"] for r in rows if r["recorded_at"]]
        time_span_days = None
        if len(timestamps) >= 2:
            try:
                t0 = datetime.fromisoformat(timestamps[0].replace("Z", "+00:00"))
                t1 = datetime.fromisoformat(timestamps[-1].replace("Z", "+00:00"))
                time_span_days = max(1, (t1 - t0).days)
            except ValueError:
                pass

        status, confidence = _derive_lifecycle_and_confidence(
            evidence_count=evidence_count,
            supporting_count=supporting,
            contradicting_count=contradicting,
            trade_count=trade_count,
            experiment_count=int(experiment_count or 0),
        )

        last_confirmed = None
        if status == "confirmed":
            last_confirmed = _utc_now()
        else:
            prev = self.conn.execute(
                "SELECT last_confirmed_at FROM knowledge_findings WHERE finding_id = ?",
                (finding_id,),
            ).fetchone()
            if prev:
                last_confirmed = prev["last_confirmed_at"]

        self.conn.execute(
            """
            UPDATE knowledge_findings SET
                evidence_count = ?,
                supporting_count = ?,
                contradicting_count = ?,
                trade_count = ?,
                experiment_count = ?,
                time_span_days = ?,
                status = ?,
                confidence = ?,
                last_confirmed_at = COALESCE(?, last_confirmed_at),
                updated_at = ?
            WHERE finding_id = ?
            """,
            (
                evidence_count,
                supporting,
                contradicting,
                trade_count,
                experiment_count,
                time_span_days,
                status,
                confidence,
                last_confirmed,
                _utc_now(),
                finding_id,
            ),
        )
        self.conn.commit()

    def set_finding_status(self, finding_id: str, status: str) -> dict[str, Any] | None:
        self.conn.execute(
            "UPDATE knowledge_findings SET status = ?, updated_at = ? WHERE finding_id = ?",
            (status, _utc_now(), finding_id),
        )
        self.conn.commit()
        return self.get_finding(finding_id)

    def promote_to_knowledge(self, finding_id: str) -> dict[str, Any]:
        finding = self.get_finding(finding_id)
        if not finding:
            return {"ok": False, "error": "finding not found"}
        if finding.get("status") not in ("confirmed", "supported"):
            return {
                "ok": False,
                "error": f"finding status {finding.get('status')} not eligible for knowledge promotion",
            }
        if int(finding.get("experiment_count") or 0) < 2:
            return {"ok": False, "error": "insufficient experiments for knowledge promotion"}
        meta = dict(finding.get("metadata") or {})
        meta["promoted_to_knowledge_at"] = _utc_now()
        self.conn.execute(
            """
            UPDATE knowledge_findings
            SET status = 'knowledge', metadata_json = ?, updated_at = ?
            WHERE finding_id = ?
            """,
            (json.dumps(meta, default=str), _utc_now(), finding_id),
        )
        self.conn.commit()
        return {"ok": True, "finding": self.get_finding(finding_id)}

    def list_findings(
        self,
        *,
        status: str | None = None,
        category: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if category:
            clauses.append("category = ?")
            params.append(category)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(limit, 500)))
        rows = self.conn.execute(
            f"""
            SELECT finding_id, finding_key, finding, category, status, confidence,
                   evidence_count, supporting_count, contradicting_count,
                   trade_count, experiment_count, time_span_days, last_confirmed_at,
                   created_at, updated_at
            FROM knowledge_findings
            {where}
            ORDER BY evidence_count DESC, updated_at DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def list_evidence_for_finding(self, finding_id: str, *, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT evidence_id, experiment_id, experiment_number, prediction_run_id,
                   trade_count, pf_change, win_rate_change, supports_finding,
                   evidence_quality, notes, recorded_at
            FROM finding_evidence
            WHERE finding_id = ?
            ORDER BY recorded_at DESC
            LIMIT ?
            """,
            (finding_id, max(1, min(limit, 200))),
        ).fetchall()
        return [dict(r) for r in rows]

    def list_links_for_finding(self, finding_id: str) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            """
            SELECT link_type, link_ref, link_label, created_at
            FROM finding_links WHERE finding_id = ?
            ORDER BY created_at DESC
            """,
            (finding_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def _derive_lifecycle_and_confidence(
    *,
    evidence_count: int,
    supporting_count: int,
    contradicting_count: int,
    trade_count: int,
    experiment_count: int,
) -> tuple[str, str]:
    support_ratio = supporting_count / evidence_count if evidence_count else 0.0

    if contradicting_count >= 3 and contradicting_count > supporting_count:
        return "contradicted", "medium"

    if evidence_count >= 8 and support_ratio >= 0.75 and experiment_count >= 5:
        conf = "very_high" if trade_count >= 10000 else "high"
        return "confirmed", conf

    if evidence_count >= 3 and support_ratio >= 0.6:
        return "supported", "medium" if experiment_count >= 2 else "low"

    return "candidate", "low"


LIFECYCLE_STAGE_MAP = {
    "candidate": "evidence_linked",
    "supported": "finding",
    "confirmed": "finding",
    "knowledge": "knowledge",
    "contradicted": "finding",
}


def lifecycle_stage_for_status(status: str) -> str:
    return LIFECYCLE_STAGE_MAP.get(str(status or ""), "evidence_linked")


def list_knowledge_findings(
    data_dir: str,
    *,
    status: str | None = None,
    category: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    with KnowledgeStore(data_dir) as store:
        return store.list_findings(status=status, category=category, limit=limit)


def get_knowledge_finding(data_dir: str, finding_id: str) -> dict[str, Any] | None:
    with KnowledgeStore(data_dir) as store:
        finding = store.get_finding(finding_id)
        if not finding:
            return None
        finding["evidence"] = store.list_evidence_for_finding(finding_id)
        finding["links"] = store.list_links_for_finding(finding_id)
        return finding
