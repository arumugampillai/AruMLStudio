"""Data types and schemas for Phase 4F.6: Morning Research Dossier & Presentation Layer.

Defines structured presentation models representing overnight campaign telemetry,
candidate rankings, lineage trees, feature governance audits, discovered feature intelligence,
feature synergy discoveries, and candidate mutation drill-down analytics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from typing import Any

from chain_replay_ml.fine_tuning.types import DescendantEvaluationRecord, FineTuningDecision
from chain_replay_ml.model_ranking.types import CandidateEvidenceScore, RecommendationClass
from chain_replay_ml.overnight_campaign.types import CampaignConfig, CampaignStatus, CampaignStopReason


class DiscoveredFeatureStatus(str, Enum):
    """Classification of discovered empirical feature utility from campaign evaluation."""
    STRONG_DISCOVERED = "STRONG_DISCOVERED"       # 🟢 Repeatedly improved candidates, positive delta & trading lift
    PROMISING = "PROMISING"                       # 🟡 Positive signals, needs further independent confirmation
    REJECTED_HARMFUL = "REJECTED_HARMFUL"         # 🔴 Negative delta, degrading, pruned or deprecated


@dataclass(frozen=True)
class DiscoveredFeatureRecord:
    """Discovered empirical evidence for an individual feature across campaign candidates."""
    feature_name: str
    times_tested: int
    positive_descendant_count: int
    top_candidates: list[str]
    best_composite_score: float
    best_trading_score: float
    avg_composite_score: float
    avg_trading_score: float
    best_delta_vs_parent: float
    avg_delta_vs_parent: float
    cross_regime_consistency: str               # "HIGH" / "MODERATE" / "LOW" / "UNKNOWN"
    phase4e_evidence_level: str                 # "HIGH" / "MODERATE" / "LOW" / "NEGATIVE"
    lifecycle_status: str                       # e.g. "ACTIVE" / "CANDIDATE" / "DEPRECATED"
    status: DiscoveredFeatureStatus             # STRONG_DISCOVERED / PROMISING / REJECTED_HARMFUL
    recommendation: str                         # "DISCOVERED — HUMAN REVIEW REQUIRED" / "REJECTED — AVOID"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "times_tested": self.times_tested,
            "positive_descendant_count": self.positive_descendant_count,
            "top_candidates": self.top_candidates,
            "best_composite_score": round(self.best_composite_score, 2),
            "best_trading_score": round(self.best_trading_score, 2),
            "avg_composite_score": round(self.avg_composite_score, 2),
            "avg_trading_score": round(self.avg_trading_score, 2),
            "best_delta_vs_parent": round(self.best_delta_vs_parent, 2),
            "avg_delta_vs_parent": round(self.avg_delta_vs_parent, 2),
            "cross_regime_consistency": self.cross_regime_consistency,
            "phase4e_evidence_level": self.phase4e_evidence_level,
            "lifecycle_status": self.lifecycle_status,
            "status": self.status.value,
            "recommendation": self.recommendation,
        }


@dataclass(frozen=True)
class DiscoveredFeatureSynergy:
    """Discovered empirical synergy pair across candidate feature mutations."""
    feature_a: str
    feature_b: str
    times_tested: int
    best_delta_composite: float
    best_delta_trading: float
    cross_regime_evidence: str
    status: str                                  # "VALIDATED_SYNERGY — REVIEW" / "CANDIDATE_SYNERGY"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateFeatureDeltaView:
    """Detailed feature delta and metrics for candidate drill-down in leaderboard / dossier."""
    candidate_id: str
    parent_candidate_id: str | None
    parent_features: list[str]
    child_features: list[str]
    added_features: list[str]
    removed_features: list[str]
    interaction_features: list[str]
    delta_composite: float
    delta_trading: float
    delta_model: float
    model_score: float
    trading_score: float
    composite_score: float
    win_rate_pct: float
    profit_factor: float
    max_drawdown_pct: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LineageNodeView:
    """Presentation view for an individual candidate in the generational lineage tree."""
    candidate_id: str
    parent_candidate_id: str | None
    generation_number: int
    mutation_type: str
    mutation_description: str
    composite_score: float
    trading_score: float
    model_score: float
    delta_vs_parent: float | None
    decision_verdict: str
    is_pruned: bool
    features: list[str] = field(default_factory=list)
    algorithm: str = "xgboost"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureGovernanceAuditSummary:
    """Summary of feature lifecycle governance during the campaign."""
    total_features_evaluated: int
    features_used: list[str]
    phase4e_recommended_features: list[str]
    deprecated_features_blocked: list[str]
    unknown_features_governed: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MorningResearchDossier:
    """Complete Morning Research Dossier compiled from persisted research memory."""
    campaign_id: str
    context_key: str
    generated_at: str
    campaign_status: CampaignStatus
    stop_reason: CampaignStopReason
    start_time_iso: str
    end_time_iso: str
    duration_seconds: float
    total_generations_completed: int
    total_candidates_generated: int
    total_candidates_trained: int
    total_candidates_evaluated: int
    total_candidates_excluded: int
    total_candidates_pruned: int
    starting_best_score: float
    best_composite_score: float
    total_score_improvement: float
    best_candidate_id: str | None
    best_candidate_class: str | None
    best_trading_score: float
    best_model_score: float
    best_win_rate_pct: float
    best_profit_factor: float
    best_max_drawdown_pct: float
    ranked_candidates: list[CandidateEvidenceScore]
    fine_tuning_trials: list[DescendantEvaluationRecord]
    lineage_tree: list[LineageNodeView]
    feature_governance_summary: FeatureGovernanceAuditSummary
    recommended_next_actions: list[str]
    discovered_features: list[DiscoveredFeatureRecord] = field(default_factory=list)
    discovered_synergies: list[DiscoveredFeatureSynergy] = field(default_factory=list)
    candidate_feature_deltas: list[CandidateFeatureDeltaView] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "context_key": self.context_key,
            "generated_at": self.generated_at,
            "campaign_status": self.campaign_status.value,
            "stop_reason": self.stop_reason.value,
            "start_time_iso": self.start_time_iso,
            "end_time_iso": self.end_time_iso,
            "duration_seconds": self.duration_seconds,
            "total_generations_completed": self.total_generations_completed,
            "total_candidates_generated": self.total_candidates_generated,
            "total_candidates_trained": self.total_candidates_trained,
            "total_candidates_evaluated": self.total_candidates_evaluated,
            "total_candidates_excluded": self.total_candidates_excluded,
            "total_candidates_pruned": self.total_candidates_pruned,
            "starting_best_score": self.starting_best_score,
            "best_composite_score": self.best_composite_score,
            "total_score_improvement": self.total_score_improvement,
            "best_candidate_id": self.best_candidate_id,
            "best_candidate_class": self.best_candidate_class,
            "best_trading_score": self.best_trading_score,
            "best_model_score": self.best_model_score,
            "best_win_rate_pct": self.best_win_rate_pct,
            "best_profit_factor": self.best_profit_factor,
            "best_max_drawdown_pct": self.best_max_drawdown_pct,
            "ranked_candidates": [c.to_dict() for c in self.ranked_candidates],
            "fine_tuning_trials": [t.to_dict() for t in self.fine_tuning_trials],
            "lineage_tree": [n.to_dict() for n in self.lineage_tree],
            "feature_governance_summary": self.feature_governance_summary.to_dict() if self.feature_governance_summary else None,
            "recommended_next_actions": self.recommended_next_actions,
            "discovered_features": [f.to_dict() for f in self.discovered_features],
            "discovered_synergies": [s.to_dict() for s in self.discovered_synergies],
            "candidate_feature_deltas": [d.to_dict() for d in self.candidate_feature_deltas],
            "warnings": self.warnings,
        }
