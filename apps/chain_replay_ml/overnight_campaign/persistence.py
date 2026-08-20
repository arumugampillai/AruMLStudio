"""Persistence layer for overnight research campaigns in analysis.db (Phase 4F.5)."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from chain_replay_ml.research_memory.db import connect_analysis_db, init_analysis_db
from .types import (
    CampaignConfig,
    CampaignState,
    CampaignStatus,
    CampaignStopReason,
    OvernightCampaignReport,
)


def init_campaign_tables(data_dir: str) -> None:
    """Initialize overnight campaign tables in analysis.db."""
    init_analysis_db(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS overnight_campaigns (
                campaign_id TEXT PRIMARY KEY,
                config_hash TEXT NOT NULL,
                config_json TEXT NOT NULL,
                status TEXT NOT NULL,
                stop_reason TEXT NOT NULL,
                current_generation INTEGER NOT NULL,
                total_candidates_generated INTEGER NOT NULL,
                total_candidates_trained INTEGER NOT NULL,
                total_candidates_evaluated INTEGER NOT NULL,
                total_candidates_excluded INTEGER NOT NULL,
                total_candidates_pruned INTEGER NOT NULL,
                total_failures INTEGER NOT NULL,
                best_candidate_id TEXT,
                best_signature_hash TEXT,
                best_composite_score REAL NOT NULL,
                best_trading_score REAL NOT NULL,
                best_model_score REAL NOT NULL,
                starting_best_score REAL NOT NULL,
                start_time_iso TEXT NOT NULL,
                last_update_iso TEXT NOT NULL,
                end_time_iso TEXT,
                warnings_json TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS overnight_campaign_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                campaign_id TEXT NOT NULL,
                generation_number INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                candidate_id TEXT,
                message TEXT,
                event_details_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (campaign_id) REFERENCES overnight_campaigns(campaign_id)
            );
            """
        )
        # Safe additive migrations for existing tables
        try:
            conn.execute("ALTER TABLE overnight_campaign_events ADD COLUMN candidate_id TEXT;")
        except Exception:
            pass
        try:
            conn.execute("ALTER TABLE overnight_campaign_events ADD COLUMN message TEXT;")
        except Exception:
            pass

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS campaign_candidate_specs (
                candidate_id TEXT PRIMARY KEY,
                signature_hash TEXT NOT NULL,
                context_key TEXT NOT NULL,
                algorithm TEXT NOT NULL,
                features_json TEXT NOT NULL,
                hyperparameters_json TEXT NOT NULL,
                parent_candidate_id TEXT,
                mutation_type TEXT,
                mutation_description TEXT,
                campaign_id TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_camp_events ON overnight_campaign_events (campaign_id, generation_number);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_camp_specs_ctx ON campaign_candidate_specs (context_key, campaign_id);"
        )
        conn.commit()
    finally:
        conn.close()



def persist_campaign_state(data_dir: str, config: CampaignConfig, state: CampaignState) -> None:
    """Persist or update campaign state in analysis.db."""
    init_campaign_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO overnight_campaigns (
                campaign_id, config_hash, config_json, status, stop_reason,
                current_generation, total_candidates_generated, total_candidates_trained,
                total_candidates_evaluated, total_candidates_excluded, total_candidates_pruned,
                total_failures, best_candidate_id, best_signature_hash, best_composite_score,
                best_trading_score, best_model_score, starting_best_score, start_time_iso,
                last_update_iso, end_time_iso, warnings_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                state.campaign_id,
                state.config_hash,
                json.dumps(config.to_dict()),
                state.status.value,
                state.stop_reason.value,
                state.current_generation,
                state.total_candidates_generated,
                state.total_candidates_trained,
                state.total_candidates_evaluated,
                state.total_candidates_excluded,
                state.total_candidates_pruned,
                state.total_failures,
                state.best_candidate_id,
                state.best_signature_hash,
                state.best_composite_score,
                state.best_trading_score,
                state.best_model_score,
                state.starting_best_score,
                state.start_time_iso,
                state.last_update_iso,
                state.end_time_iso,
                json.dumps(state.warnings),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_campaign_state(data_dir: str, campaign_id: str) -> tuple[CampaignConfig | None, CampaignState | None]:
    """Retrieve persisted campaign config and state from analysis.db."""
    init_campaign_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        row = conn.execute(
            "SELECT * FROM overnight_campaigns WHERE campaign_id = ?;",
            (campaign_id,),
        ).fetchone()

        if not row:
            return None, None

        cfg_dict = json.loads(row["config_json"])
        config = CampaignConfig(**cfg_dict)

        state = CampaignState(
            campaign_id=row["campaign_id"],
            config_hash=row["config_hash"],
            status=CampaignStatus(row["status"]),
            stop_reason=CampaignStopReason(row["stop_reason"]),
            current_generation=int(row["current_generation"]),
            total_candidates_generated=int(row["total_candidates_generated"]),
            total_candidates_trained=int(row["total_candidates_trained"]),
            total_candidates_evaluated=int(row["total_candidates_evaluated"]),
            total_candidates_excluded=int(row["total_candidates_excluded"]),
            total_candidates_pruned=int(row["total_candidates_pruned"]),
            total_failures=int(row["total_failures"]),
            best_candidate_id=row["best_candidate_id"],
            best_signature_hash=row["best_signature_hash"],
            best_composite_score=float(row["best_composite_score"]),
            best_trading_score=float(row["best_trading_score"]),
            best_model_score=float(row["best_model_score"]),
            starting_best_score=float(row["starting_best_score"]),
            start_time_iso=row["start_time_iso"],
            last_update_iso=row["last_update_iso"],
            end_time_iso=row["end_time_iso"],
            warnings=json.loads(row["warnings_json"] or "[]"),
        )
        return config, state
    finally:
        conn.close()


def persist_candidate_specs(
    data_dir: str,
    candidates: list[Any],
    *,
    campaign_id: str | None = None,
) -> int:
    """Persist CandidateSpec objects into campaign_candidate_specs table in analysis.db."""
    if not candidates:
        return 0
    init_campaign_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    count = 0
    try:
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        for c in candidates:
            p_id = c.lineage.parent_candidate_id if c.lineage else None
            m_type = c.lineage.mutation_type.value if c.lineage else "INITIAL_SPEC"
            m_desc = c.lineage.mutation_description if c.lineage else "Phase 4E seed candidate"
            camp_id = campaign_id or (c.lineage.campaign_id if c.lineage else None)

            conn.execute(
                """
                INSERT OR REPLACE INTO campaign_candidate_specs (
                    candidate_id, signature_hash, context_key, algorithm,
                    features_json, hyperparameters_json, parent_candidate_id,
                    mutation_type, mutation_description, campaign_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    c.candidate_id,
                    c.signature_hash,
                    c.context_key,
                    c.algorithm,
                    json.dumps(c.features),
                    json.dumps(c.hyperparameters),
                    p_id,
                    m_type,
                    m_desc,
                    camp_id,
                    now_iso,
                ),
            )
            count += 1
        conn.commit()
        return count
    finally:
        conn.close()


def load_candidate_specs_for_campaign(
    data_dir: str,
    campaign_id: str | None = None,
    *,
    context_key: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Retrieve persisted candidate specifications from analysis.db as candidate_id -> metadata dict."""
    init_campaign_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        query = "SELECT * FROM campaign_candidate_specs WHERE 1=1"
        params: list[Any] = []
        if campaign_id:
            query += " AND campaign_id = ?"
            params.append(campaign_id)
        if context_key:
            query += " AND context_key = ?"
            params.append(context_key)

        rows = conn.execute(query, tuple(params)).fetchall()
        result: dict[str, dict[str, Any]] = {}
        for r in rows:
            result[r["candidate_id"]] = {
                "candidate_id": r["candidate_id"],
                "signature_hash": r["signature_hash"],
                "context_key": r["context_key"],
                "algorithm": r["algorithm"],
                "features": json.loads(r["features_json"]),
                "hyperparameters": json.loads(r["hyperparameters_json"]),
                "parent_candidate_id": r["parent_candidate_id"],
                "mutation_type": r["mutation_type"],
                "mutation_description": r["mutation_description"],
                "campaign_id": r["campaign_id"],
                "created_at": r["created_at"],
            }
        return result
    finally:
        conn.close()


def persist_campaign_event(
    data_dir: str,
    *,
    campaign_id: str,
    generation_number: int,
    event_type: str,
    candidate_id: str | None = None,
    message: str = "",
    details: dict[str, Any] | None = None,
    created_at: str | None = None,
) -> int:
    """Insert a chronological campaign audit event into overnight_campaign_events."""
    init_campaign_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        from datetime import datetime, timezone
        now_iso = created_at or datetime.now(timezone.utc).isoformat()
        details_json = json.dumps(details or {}, ensure_ascii=False)
        cur = conn.execute(
            """
            INSERT INTO overnight_campaign_events (
                campaign_id, generation_number, event_type, candidate_id, message, event_details_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?);
            """,
            (
                str(campaign_id),
                int(generation_number),
                str(event_type).upper().strip(),
                str(candidate_id) if candidate_id else None,
                str(message),
                details_json,
                now_iso,
            ),
        )
        conn.commit()
        return cur.lastrowid or 0
    finally:
        conn.close()


def load_campaign_events(
    data_dir: str,
    campaign_id: str | None = None,
    *,
    event_type_filter: str | None = None,
    search_query: str | None = None,
    limit: int = 1000,
) -> list[dict[str, Any]]:
    """Retrieve chronological campaign execution audit events from analysis.db."""
    init_campaign_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        query = "SELECT * FROM overnight_campaign_events WHERE 1=1"
        params: list[Any] = []
        if campaign_id:
            query += " AND campaign_id = ?"
            params.append(str(campaign_id))
        if event_type_filter and event_type_filter.upper() != "ALL":
            filter_up = event_type_filter.upper().strip()
            if filter_up == "CANDIDATE":
                query += " AND (event_type LIKE '%CANDIDATE%' OR candidate_id IS NOT NULL)"
            elif filter_up == "METRICS":
                query += " AND (event_type LIKE '%EVAL%' OR event_type LIKE '%METRIC%')"
            elif filter_up == "CHAMPION":
                query += " AND (event_type LIKE '%CHAMP%')"
            elif filter_up == "DECISIONS":
                query += " AND (event_type LIKE '%VERDICT%' OR event_type LIKE '%PLATEAU%' OR event_type LIKE '%PRUN%')"
            elif filter_up == "WARNINGS":
                query += " AND (event_type LIKE '%WARN%' OR event_type LIKE '%ERROR%' OR message LIKE '%⚠️%' OR message LIKE '%WARN%')"
            else:
                query += " AND event_type = ?"
                params.append(filter_up)

        if search_query:
            sq = f"%{search_query.strip()}%"
            query += " AND (candidate_id LIKE ? OR message LIKE ? OR event_details_json LIKE ?)"
            params.extend([sq, sq, sq])

        query += " ORDER BY event_id ASC LIMIT ?"
        params.append(max(1, int(limit)))

        rows = conn.execute(query, tuple(params)).fetchall()
        events: list[dict[str, Any]] = []
        for r in rows:
            details = {}
            if r["event_details_json"]:
                try:
                    details = json.loads(r["event_details_json"])
                except Exception:
                    details = {}
            events.append({
                "event_id": r["event_id"],
                "campaign_id": r["campaign_id"],
                "generation": r["generation_number"],
                "candidate_id": r["candidate_id"] or details.get("candidate_id") or "—",
                "event_type": r["event_type"],
                "message": r["message"] or details.get("message") or "",
                "timestamp": r["created_at"],
                "details": details,
            })
        return events
    finally:
        conn.close()

