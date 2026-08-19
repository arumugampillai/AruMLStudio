"""Comparative Child vs Parent Evaluation Engine (Phase 4F.4).

Computes multidimensional performance deltas and assigns deterministic fine-tuning decisions.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any

from chain_replay_ml.candidate_generation.types import MutationType
from chain_replay_ml.model_ranking.types import CandidateEvidenceScore
from .types import (
    DescendantEvaluationRecord,
    FineTuningBudget,
    FineTuningDecision,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def evaluate_child_vs_parent(
    *,
    child_score: CandidateEvidenceScore,
    parent_score: CandidateEvidenceScore,
    mutation_type: MutationType = MutationType.FEATURE_SUBSET_MUTATION,
    mutation_description: str = "Fine-tuning descendant mutation",
    generation_number: int = 1,
    opportunity_id: str | None = None,
    budget: FineTuningBudget | None = None,
) -> DescendantEvaluationRecord:
    """Evaluate child candidate against its parent candidate and assign fine-tuning decision."""
    b = budget or FineTuningBudget()
    trial_id = f"FT_{parent_score.candidate_id[:8]}_{child_score.candidate_id[:8]}"

    delta_comp = round(child_score.composite_score - parent_score.composite_score, 4)
    delta_trade = round(child_score.trading_evidence_score - parent_score.trading_evidence_score, 4)
    delta_model = round(child_score.model_evidence_score - parent_score.model_evidence_score, 4)
    delta_risk = round(child_score.risk_penalty - parent_score.risk_penalty, 4)

    warnings: list[str] = list(child_score.warnings)

    # Determine Decision Verdict
    if child_score.risk_penalty > b.max_risk_penalty_allowed or delta_comp <= b.prune_threshold:
        verdict = FineTuningDecision.PRUNED_MUTATION_PATH
        is_pruned = True
        warnings.append(f"PRUNED: Delta composite {delta_comp:.2f} <= {b.prune_threshold:.2f} or risk penalty {child_score.risk_penalty:.1f} > {b.max_risk_penalty_allowed:.1f}")
    elif delta_comp >= b.lift_threshold:
        verdict = FineTuningDecision.CONFIRMED_MUTATION_LIFT
        is_pruned = False
    elif delta_comp <= b.regression_threshold:
        verdict = FineTuningDecision.REGRESSION
        is_pruned = False
    else:
        verdict = FineTuningDecision.NEUTRAL_MUTATION
        is_pruned = False

    return DescendantEvaluationRecord(
        trial_id=trial_id,
        context_key=child_score.context_key,
        parent_candidate_id=parent_score.candidate_id,
        parent_signature_hash=parent_score.signature_hash,
        child_candidate_id=child_score.candidate_id,
        child_signature_hash=child_score.signature_hash,
        generation_number=generation_number,
        mutation_type=mutation_type,
        mutation_description=mutation_description,
        opportunity_id=opportunity_id,
        parent_composite_score=parent_score.composite_score,
        child_composite_score=child_score.composite_score,
        delta_composite_score=delta_comp,
        delta_trading_score=delta_trade,
        delta_model_score=delta_model,
        delta_risk_penalty=delta_risk,
        decision_verdict=verdict,
        is_branch_pruned=is_pruned,
        warnings=warnings,
        created_at=_utc_now_iso(),
    )
