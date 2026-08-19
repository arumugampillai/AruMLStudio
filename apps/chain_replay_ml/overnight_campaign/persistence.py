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
                event_details_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (campaign_id) REFERENCES overnight_campaigns(campaign_id)
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_camp_events ON overnight_campaign_events (campaign_id, generation_number);"
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
