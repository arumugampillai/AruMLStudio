"""Persistence layer for Phase 4F.4 Fine-Tuning Trials in analysis.db."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from chain_replay_ml.candidate_generation.types import MutationType
from chain_replay_ml.research_memory.db import connect_analysis_db, init_analysis_db
from .types import (
    DescendantEvaluationRecord,
    FineTuningCampaignResult,
    FineTuningDecision,
)


def init_fine_tuning_tables(data_dir: str) -> None:
    """Initialize fine_tuning_trials table in analysis.db."""
    init_analysis_db(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS fine_tuning_trials (
                trial_id TEXT PRIMARY KEY,
                context_key TEXT NOT NULL,
                parent_candidate_id TEXT NOT NULL,
                parent_signature_hash TEXT NOT NULL,
                child_candidate_id TEXT NOT NULL,
                child_signature_hash TEXT NOT NULL,
                generation_number INTEGER NOT NULL,
                mutation_type TEXT NOT NULL,
                mutation_description TEXT NOT NULL,
                opportunity_id TEXT,
                parent_composite_score REAL NOT NULL,
                child_composite_score REAL NOT NULL,
                delta_composite_score REAL NOT NULL,
                delta_trading_score REAL NOT NULL,
                delta_model_score REAL NOT NULL,
                delta_risk_penalty REAL NOT NULL,
                decision_verdict TEXT NOT NULL,
                is_branch_pruned INTEGER NOT NULL,
                warnings_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ft_trials_ctx ON fine_tuning_trials (context_key, delta_composite_score DESC);"
        )
        conn.commit()
    finally:
        conn.close()


def persist_fine_tuning_records(
    data_dir: str,
    records: list[DescendantEvaluationRecord],
) -> int:
    """Persist fine-tuning trial records into analysis.db."""
    init_fine_tuning_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    written = 0
    try:
        for r in records:
            conn.execute(
                """
                INSERT OR REPLACE INTO fine_tuning_trials (
                    trial_id, context_key, parent_candidate_id, parent_signature_hash,
                    child_candidate_id, child_signature_hash, generation_number,
                    mutation_type, mutation_description, opportunity_id,
                    parent_composite_score, child_composite_score, delta_composite_score,
                    delta_trading_score, delta_model_score, delta_risk_penalty,
                    decision_verdict, is_branch_pruned, warnings_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    r.trial_id,
                    r.context_key,
                    r.parent_candidate_id,
                    r.parent_signature_hash,
                    r.child_candidate_id,
                    r.child_signature_hash,
                    r.generation_number,
                    r.mutation_type.value,
                    r.mutation_description,
                    r.opportunity_id,
                    r.parent_composite_score,
                    r.child_composite_score,
                    r.delta_composite_score,
                    r.delta_trading_score,
                    r.delta_model_score,
                    r.delta_risk_penalty,
                    r.decision_verdict.value,
                    1 if r.is_branch_pruned else 0,
                    json.dumps(r.warnings),
                    r.created_at,
                ),
            )
            written += 1
        conn.commit()
        return written
    finally:
        conn.close()


def load_fine_tuning_records_for_context(
    data_dir: str,
    context_key: str,
) -> list[DescendantEvaluationRecord]:
    """Retrieve fine-tuning trial records for a context key from analysis.db."""
    init_fine_tuning_tables(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        rows = conn.execute(
            """
            SELECT trial_id, context_key, parent_candidate_id, parent_signature_hash,
                   child_candidate_id, child_signature_hash, generation_number,
                   mutation_type, mutation_description, opportunity_id,
                   parent_composite_score, child_composite_score, delta_composite_score,
                   delta_trading_score, delta_model_score, delta_risk_penalty,
                   decision_verdict, is_branch_pruned, warnings_json, created_at
            FROM fine_tuning_trials
            WHERE context_key = ?
            ORDER BY delta_composite_score DESC;
            """,
            (context_key,),
        ).fetchall()

        records: list[DescendantEvaluationRecord] = []
        for r in rows:
            records.append(
                DescendantEvaluationRecord(
                    trial_id=r["trial_id"],
                    context_key=r["context_key"],
                    parent_candidate_id=r["parent_candidate_id"],
                    parent_signature_hash=r["parent_signature_hash"],
                    child_candidate_id=r["child_candidate_id"],
                    child_signature_hash=r["child_signature_hash"],
                    generation_number=int(r["generation_number"]),
                    mutation_type=MutationType(r["mutation_type"]),
                    mutation_description=r["mutation_description"],
                    opportunity_id=r["opportunity_id"],
                    parent_composite_score=float(r["parent_composite_score"]),
                    child_composite_score=float(r["child_composite_score"]),
                    delta_composite_score=float(r["delta_composite_score"]),
                    delta_trading_score=float(r["delta_trading_score"]),
                    delta_model_score=float(r["delta_model_score"]),
                    delta_risk_penalty=float(r["delta_risk_penalty"]),
                    decision_verdict=FineTuningDecision(r["decision_verdict"]),
                    is_branch_pruned=bool(r["is_branch_pruned"]),
                    warnings=json.loads(r["warnings_json"] or "[]"),
                    created_at=r["created_at"],
                )
            )
        return records
    finally:
        conn.close()
