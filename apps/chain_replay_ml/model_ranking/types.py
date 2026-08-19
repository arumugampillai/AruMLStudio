"""Data types and schemas for Phase 4F.3: Model + Trading Evidence Ranking Engine.

Defines:
1. RecommendationClass: Multi-objective research recommendation classification.
2. CandidateRankingPolicy: Configurable research baseline weighting and penalty policy.
3. CandidateEvidenceScore: Complete evaluated model + trading evidence score record.
4. ContextRankingReport: Ranked candidate report for a single ModelContextKey.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any


class RecommendationClass(str, Enum):
    """Advisory recommendation classification for candidate models."""
    CHAMPION_CANDIDATE = "CHAMPION_CANDIDATE"      # Outperforms existing champion across model & trading dimensions
    STRONG_CONTENDER = "STRONG_CONTENDER"          # High overall robustness and trading evidence (Score >= 70)
    FINE_TUNE_CANDIDATE = "FINE_TUNE_CANDIDATE"    # Promising candidate selected for Phase 4F.4 descendant mutations
    BENCHMARK_ONLY = "BENCHMARK_ONLY"              # Moderate baseline performance (50 <= Score < 70)
    REJECTED = "REJECTED"                          # Poor performance or excessive risk penalty (Score < 50 or P_risk > 20)


@dataclass(frozen=True)
class CandidateRankingPolicy:
    """Research-configurable baseline ranking policy.
    
    All numeric parameters (0.40, 0.60, 5.0% DD, 3 losses) are configurable
    RESEARCH BASELINE HYPOTHESES, NOT authoritative production constants.
    """
    policy_id: str = "RANK_POLICY_v1.0"
    w_model: float = 0.40                          # Weight on statistical model robustness
    w_trade: float = 0.60                          # Weight on strategy replay trading evidence
    alpha_win_rate: float = 0.40                   # Win rate sub-weight in trading score
    alpha_profit_factor: float = 0.40              # Profit factor sub-weight in trading score
    alpha_mfe_mae: float = 0.20                    # MFE/MAE efficiency ratio sub-weight
    lambda_drawdown: float = 2.0                   # Drawdown penalty multiplier
    tau_safe_drawdown: float = 5.0                 # Safe drawdown threshold (%)
    lambda_loss_streak: float = 3.0                # Consecutive loss streak penalty multiplier
    tau_safe_loss_streak: int = 3                  # Safe consecutive loss streak threshold
    lambda_reg_spread: float = 0.50                # Cross-regime win rate spread penalty multiplier
    tau_safe_reg_spread: float = 25.0              # Safe regime spread threshold (%)
    min_trade_volume: int = 30                     # Minimum trade count for full evidence confidence
    champion_beat_margin: float = 2.0              # Required score margin to earn CHAMPION_CANDIDATE status
    champion_min_score: float = 75.0               # Minimum composite score for champion candidacy

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def compute_policy_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Canonical Default Policy instance
RANK_POLICY_v1_0 = CandidateRankingPolicy()


@dataclass(frozen=True)
class CandidateEvidenceScore:
    """Evaluated evidence score record for a single candidate model."""
    candidate_id: str
    signature_hash: str
    context_key: str
    composite_score: float
    model_evidence_score: float
    trading_evidence_score: float
    risk_penalty: float
    volume_confidence: float
    recommendation_class: RecommendationClass
    model_metrics: dict[str, float]
    trading_metrics: dict[str, float]
    score_breakdown: dict[str, float]
    warnings: list[str] = field(default_factory=list)
    parent_candidate_id: str | None = None
    delta_vs_parent: float | None = None
    opportunity_id: str | None = None
    evaluated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["recommendation_class"] = self.recommendation_class.value
        return d


@dataclass(frozen=True)
class ContextRankingReport:
    """Aggregated, Pareto-ranked candidate report for a single ModelContextKey."""
    context_key: str
    ranking_policy_id: str
    ranking_policy_hash: str
    total_candidates_ranked: int
    top_candidate: CandidateEvidenceScore | None
    champion_candidate: CandidateEvidenceScore | None
    fine_tune_candidates: list[CandidateEvidenceScore]
    ranked_candidates: list[CandidateEvidenceScore]
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_key": self.context_key,
            "ranking_policy_id": self.ranking_policy_id,
            "ranking_policy_hash": self.ranking_policy_hash,
            "total_candidates_ranked": self.total_candidates_ranked,
            "top_candidate": self.top_candidate.to_dict() if self.top_candidate else None,
            "champion_candidate": self.champion_candidate.to_dict() if self.champion_candidate else None,
            "fine_tune_candidates": [c.to_dict() for c in self.fine_tune_candidates],
            "ranked_candidates": [c.to_dict() for c in self.ranked_candidates],
            "generated_at": self.generated_at,
        }
