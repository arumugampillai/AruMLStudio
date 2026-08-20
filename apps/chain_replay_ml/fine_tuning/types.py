"""Data types and schemas for Phase 4F.4: Automated Fine-Tuning & Descendant Mutation Controller.

Defines:
1. FineTuningDecision: Classification of child vs parent mutation outcome.
2. FineTuningBudget: Resource control, depth limits, and pruning thresholds.
3. DescendantEvaluationRecord: Detailed comparative trial record between child and parent.
4. FineTuningCampaignResult: Aggregated summary of a fine-tuning execution cycle.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from chain_replay_ml.candidate_generation.types import (
    CandidateEligibility,
    CandidateSpec,
    MutationType,
)
from chain_replay_ml.model_ranking.types import CandidateEvidenceScore


class FineTuningDecision(str, Enum):
    """Comparative decision verdict of a descendant candidate relative to its parent."""
    CONFIRMED_MUTATION_LIFT = "CONFIRMED_MUTATION_LIFT"  # Child composite score > Parent + lift_threshold
    NEUTRAL_MUTATION = "NEUTRAL_MUTATION"                # Child within [-prune_threshold, +lift_threshold]
    REGRESSION = "REGRESSION"                            # Child worse than parent
    PRUNED_MUTATION_PATH = "PRUNED_MUTATION_PATH"        # Significant degradation or high risk penalty; branch terminated
    EXCLUDED_BY_PRUNING = "EXCLUDED_BY_PRUNING"          # Pre-training pruning by Phase 4E.4 negative evidence


@dataclass(frozen=True)
class FineTuningBudget:
    """Resource control and evolutionary search parameters (16 GB RAM safety defaults)."""
    max_descendants_per_parent: int = 3
    max_generations: int = 4
    max_candidates_total: int = 30
    max_features_per_candidate: int = 2000
    lift_threshold: float = 1.5           # Delta composite score required for CONFIRMED_MUTATION_LIFT
    regression_threshold: float = -1.0    # Delta composite score that triggers REGRESSION
    prune_threshold: float = -3.0         # Delta composite score that triggers PRUNED_MUTATION_PATH
    max_risk_penalty_allowed: float = 20.0 # Risk penalty ceiling before forced pruning

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DescendantEvaluationRecord:
    """Complete comparative record evaluating a child candidate against its parent."""
    trial_id: str
    context_key: str
    parent_candidate_id: str
    parent_signature_hash: str
    child_candidate_id: str
    child_signature_hash: str
    generation_number: int
    mutation_type: MutationType
    mutation_description: str
    opportunity_id: str | None
    parent_composite_score: float
    child_composite_score: float
    delta_composite_score: float
    delta_trading_score: float
    delta_model_score: float
    delta_risk_penalty: float
    decision_verdict: FineTuningDecision
    is_branch_pruned: bool
    warnings: list[str] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mutation_type"] = self.mutation_type.value
        d["decision_verdict"] = self.decision_verdict.value
        return d


@dataclass(frozen=True)
class FineTuningCampaignResult:
    """Result summary of an automated fine-tuning generation cycle."""
    context_key: str
    campaign_id: str | None
    total_descendants_generated: int
    confirmed_lifts_count: int
    neutral_count: int
    regression_count: int
    pruned_paths_count: int
    active_lineage_roots: list[str]
    best_descendant: DescendantEvaluationRecord | None
    trial_records: list[DescendantEvaluationRecord]
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_key": self.context_key,
            "campaign_id": self.campaign_id,
            "total_descendants_generated": self.total_descendants_generated,
            "confirmed_lifts_count": self.confirmed_lifts_count,
            "neutral_count": self.neutral_count,
            "regression_count": self.regression_count,
            "pruned_paths_count": self.pruned_paths_count,
            "active_lineage_roots": self.active_lineage_roots,
            "best_descendant": self.best_descendant.to_dict() if self.best_descendant else None,
            "trial_records": [t.to_dict() for t in self.trial_records],
            "generated_at": self.generated_at,
        }
