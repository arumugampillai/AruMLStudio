"""Persistence layer for candidate evidence rankings in analysis.db (Phase 4F.3)."""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from chain_replay_ml.research_memory.db import connect_analysis_db, init_analysis_db
from .types import (
    CandidateEvidenceScore,
    ContextRankingReport,
    RecommendationClass,
)


def init_candidate_rankings_table(data_dir: str) -> None:
    """Initialize candidate_evidence_rankings table in analysis.db."""
    init_analysis_db(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS candidate_evidence_rankings (
                signature_hash TEXT NOT NULL,
                context_key TEXT NOT NULL,
                candidate_id TEXT NOT NULL,
                policy_id TEXT NOT NULL,
                policy_hash TEXT NOT NULL,
                composite_score REAL NOT NULL,
                model_evidence_score REAL NOT NULL,
                trading_evidence_score REAL NOT NULL,
                risk_penalty REAL NOT NULL,
                volume_confidence REAL NOT NULL,
                recommendation_class TEXT NOT NULL,
                model_metrics_json TEXT NOT NULL,
                trading_metrics_json TEXT NOT NULL,
                score_breakdown_json TEXT NOT NULL,
                warnings_json TEXT NOT NULL,
                parent_candidate_id TEXT,
                delta_vs_parent REAL,
                opportunity_id TEXT,
                evaluated_at TEXT NOT NULL,
                PRIMARY KEY (signature_hash, policy_id)
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_cand_rank_ctx ON candidate_evidence_rankings (context_key, composite_score DESC);"
        )
        conn.commit()
    finally:
        conn.close()


def persist_candidate_rankings(
    data_dir: str,
    report: ContextRankingReport,
) -> int:
    """Persist context ranking report scores into candidate_evidence_rankings table."""
    init_candidate_rankings_table(data_dir)
    conn = connect_analysis_db(data_dir)
    written = 0
    try:
        for cand in report.ranked_candidates:
            conn.execute(
                """
                INSERT OR REPLACE INTO candidate_evidence_rankings (
                    signature_hash, context_key, candidate_id, policy_id, policy_hash,
                    composite_score, model_evidence_score, trading_evidence_score,
                    risk_penalty, volume_confidence, recommendation_class,
                    model_metrics_json, trading_metrics_json, score_breakdown_json,
                    warnings_json, parent_candidate_id, delta_vs_parent,
                    opportunity_id, evaluated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    cand.signature_hash,
                    cand.context_key,
                    cand.candidate_id,
                    report.ranking_policy_id,
                    report.ranking_policy_hash,
                    cand.composite_score,
                    cand.model_evidence_score,
                    cand.trading_evidence_score,
                    cand.risk_penalty,
                    cand.volume_confidence,
                    cand.recommendation_class.value,
                    json.dumps(cand.model_metrics),
                    json.dumps(cand.trading_metrics),
                    json.dumps(cand.score_breakdown),
                    json.dumps(cand.warnings),
                    cand.parent_candidate_id,
                    cand.delta_vs_parent,
                    cand.opportunity_id,
                    cand.evaluated_at,
                ),
            )
            written += 1
        conn.commit()
        return written
    finally:
        conn.close()


def load_candidate_rankings_for_context(
    data_dir: str,
    context_key: str,
    *,
    policy_id: str = "RANK_POLICY_v1.0",
) -> list[CandidateEvidenceScore]:
    """Retrieve persisted candidate evidence rankings for a context key."""
    init_candidate_rankings_table(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        rows = conn.execute(
            """
            SELECT signature_hash, context_key, candidate_id, composite_score,
                   model_evidence_score, trading_evidence_score, risk_penalty,
                   volume_confidence, recommendation_class, model_metrics_json,
                   trading_metrics_json, score_breakdown_json, warnings_json,
                   parent_candidate_id, delta_vs_parent, opportunity_id, evaluated_at
            FROM candidate_evidence_rankings
            WHERE context_key = ? AND policy_id = ?
            ORDER BY composite_score DESC;
            """,
            (context_key, policy_id),
        ).fetchall()

        scores: list[CandidateEvidenceScore] = []
        for r in rows:
            scores.append(
                CandidateEvidenceScore(
                    candidate_id=r["candidate_id"],
                    signature_hash=r["signature_hash"],
                    context_key=r["context_key"],
                    composite_score=float(r["composite_score"]),
                    model_evidence_score=float(r["model_evidence_score"]),
                    trading_evidence_score=float(r["trading_evidence_score"]),
                    risk_penalty=float(r["risk_penalty"]),
                    volume_confidence=float(r["volume_confidence"]),
                    recommendation_class=RecommendationClass(r["recommendation_class"]),
                    model_metrics=json.loads(r["model_metrics_json"] or "{}"),
                    trading_metrics=json.loads(r["trading_metrics_json"] or "{}"),
                    score_breakdown=json.loads(r["score_breakdown_json"] or "{}"),
                    warnings=json.loads(r["warnings_json"] or "[]"),
                    parent_candidate_id=r["parent_candidate_id"],
                    delta_vs_parent=float(r["delta_vs_parent"]) if r["delta_vs_parent"] is not None else None,
                    opportunity_id=r["opportunity_id"],
                    evaluated_at=r["evaluated_at"],
                )
            )
        return scores
    finally:
        conn.close()
