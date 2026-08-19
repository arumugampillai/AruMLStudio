"""Context-scoped candidate ranking service & deterministic tie-breaking (Phase 4F.3)."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any, Sequence

from chain_replay_ml.research_memory.db import connect_analysis_db, init_analysis_db
from .scorer import evaluate_candidate_evidence
from .types import (
    CandidateEvidenceScore,
    CandidateRankingPolicy,
    ContextRankingReport,
    RANK_POLICY_v1_0,
    RecommendationClass,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def rank_candidates_in_context(
    data_dir: str,
    context_key: str,
    *,
    candidate_items: Sequence[dict[str, Any]] | None = None,
    policy: CandidateRankingPolicy | None = None,
    champion_composite_score: float | None = None,
) -> ContextRankingReport:
    """Rank all evaluated candidate models in a single context key using Pareto multi-objective scoring."""
    pol = policy or RANK_POLICY_v1_0
    init_analysis_db(data_dir)

    evaluated_scores: list[CandidateEvidenceScore] = []

    # 1. If candidate items provided directly, score them
    if candidate_items is not None:
        for item in candidate_items:
            cand_id = item.get("candidate_id", "CAND_UNKNOWN")
            sig_hash = item.get("signature_hash", "")
            c_key = item.get("context_key", context_key)
            m_metrics = item.get("model_metrics", {})
            t_metrics = item.get("trading_metrics", {})
            p_id = item.get("parent_candidate_id")
            p_score = item.get("parent_composite_score")
            opp_id = item.get("opportunity_id")

            ev_score = evaluate_candidate_evidence(
                candidate_id=cand_id,
                signature_hash=sig_hash,
                context_key=c_key,
                model_metrics=m_metrics,
                trading_metrics=t_metrics,
                parent_candidate_id=p_id,
                parent_composite_score=p_score,
                opportunity_id=opp_id,
                policy=pol,
                champion_composite_score=champion_composite_score,
            )
            evaluated_scores.append(ev_score)
    else:
        # 2. Query analysis.db for benchmark and trading metrics in context
        conn = connect_analysis_db(data_dir)
        try:
            # Query model_benchmarks joined with benchmark_metrics
            rows = conn.execute(
                """
                SELECT s.signature_hash, s.canonical_payload_json,
                       b.robustness_score, b.roc_auc, b.expected_calibration_error,
                       b.fold_mean, b.fold_std, b.worst_fold_drawdown,
                       t.metric_value AS win_rate_pct
                FROM experiment_signatures s
                LEFT JOIN model_benchmarks b ON s.signature_hash = b.signature_hash
                LEFT JOIN benchmark_metrics t ON s.signature_hash = t.signature_hash AND t.metric_name = 'win_rate_pct'
                WHERE s.context_key = ?
                ORDER BY s.first_executed_at ASC;
                """,
                (context_key,),
            ).fetchall()

            for r in rows:
                sig = r["signature_hash"]
                m_metrics = {
                    "roc_auc": r["roc_auc"] or 0.50,
                    "expected_calibration_error": r["expected_calibration_error"] or 0.0,
                    "fold_mean": r["fold_mean"] or (r["roc_auc"] or 0.50),
                    "fold_std": r["fold_std"] or 0.0,
                    "worst_fold_drawdown": r["worst_fold_drawdown"] or 0.0,
                }
                # Query additional trading metrics for this signature
                t_rows = conn.execute(
                    "SELECT metric_name, metric_value FROM benchmark_metrics WHERE signature_hash = ?;",
                    (sig,),
                ).fetchall()
                t_metrics = {tr["metric_name"]: tr["metric_value"] for tr in t_rows}

                ev_score = evaluate_candidate_evidence(
                    candidate_id=f"CAND_{sig[:8]}",
                    signature_hash=sig,
                    context_key=context_key,
                    model_metrics=m_metrics,
                    trading_metrics=t_metrics,
                    policy=pol,
                    champion_composite_score=champion_composite_score,
                )
                evaluated_scores.append(ev_score)
        finally:
            conn.close()

    # 3. Deterministic 5-Level Tie-Breaking Sort:
    # 1. Composite Score (desc)
    # 2. Trading Evidence Score (desc)
    # 3. ECE / Calibration Error (asc)
    # 4. Feature Count Parsimony (asc)
    # 5. Signature Hash (asc)
    def _sort_key(score: CandidateEvidenceScore) -> tuple[float, float, float, int, str]:
        ece_val = score.model_metrics.get("expected_calibration_error", 1.0)
        n_feats = int(score.model_metrics.get("total_features", 30))
        return (
            -score.composite_score,
            -score.trading_evidence_score,
            ece_val,
            n_feats,
            score.signature_hash,
        )

    evaluated_scores.sort(key=_sort_key)

    # 4. Identify Top, Champion, and Fine-Tune Candidates
    top_cand = evaluated_scores[0] if evaluated_scores else None
    champ_cand = next((c for c in evaluated_scores if c.recommendation_class == RecommendationClass.CHAMPION_CANDIDATE), None)

    # Fine-tune candidates: Top 3 candidates that are not rejected
    fine_tune_cands: list[CandidateEvidenceScore] = []
    for c in evaluated_scores:
        if c.recommendation_class != RecommendationClass.REJECTED and len(fine_tune_cands) < 3:
            # Overwrite classification to FINE_TUNE_CANDIDATE if not already CHAMPION_CANDIDATE
            if c.recommendation_class != RecommendationClass.CHAMPION_CANDIDATE:
                updated = CandidateEvidenceScore(
                    candidate_id=c.candidate_id,
                    signature_hash=c.signature_hash,
                    context_key=c.context_key,
                    composite_score=c.composite_score,
                    model_evidence_score=c.model_evidence_score,
                    trading_evidence_score=c.trading_evidence_score,
                    risk_penalty=c.risk_penalty,
                    volume_confidence=c.volume_confidence,
                    recommendation_class=RecommendationClass.FINE_TUNE_CANDIDATE,
                    model_metrics=c.model_metrics,
                    trading_metrics=c.trading_metrics,
                    score_breakdown=c.score_breakdown,
                    warnings=c.warnings,
                    parent_candidate_id=c.parent_candidate_id,
                    delta_vs_parent=c.delta_vs_parent,
                    opportunity_id=c.opportunity_id,
                    evaluated_at=c.evaluated_at,
                )
                fine_tune_cands.append(updated)
            else:
                fine_tune_cands.append(c)

    return ContextRankingReport(
        context_key=context_key,
        ranking_policy_id=pol.policy_id,
        ranking_policy_hash=pol.compute_policy_hash(),
        total_candidates_ranked=len(evaluated_scores),
        top_candidate=top_cand,
        champion_candidate=champ_cand,
        fine_tune_candidates=fine_tune_cands,
        ranked_candidates=evaluated_scores,
        generated_at=_utc_now_iso(),
    )
