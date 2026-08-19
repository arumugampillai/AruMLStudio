"""Data types and schemas for Phase 4F.5: Autonomous Overnight Research Campaign Controller.

Defines:
1. CampaignStatus: Finite state machine lifecycle states.
2. CampaignStopReason: Deterministic stop causes.
3. CampaignConfig: Configurable research campaign parameters and safety limits.
4. CampaignState: Live execution telemetry and recovery checkpoint.
5. OvernightCampaignReport: Machine-readable summary for Phase 4F.6 Morning Research Dossier.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
from typing import Any

from chain_replay_ml.fine_tuning.types import DescendantEvaluationRecord
from chain_replay_ml.model_ranking.types import CandidateEvidenceScore


class CampaignStatus(str, Enum):
    """Lifecycle states of an autonomous overnight campaign."""
    CREATED = "CREATED"
    RUNNING = "RUNNING"
    GENERATING_CANDIDATES = "GENERATING_CANDIDATES"
    TRAINING = "TRAINING"
    OOS_EVALUATION = "OOS_EVALUATION"
    TRADING_EVALUATION = "TRADING_EVALUATION"
    RANKING = "RANKING"
    FINE_TUNING = "FINE_TUNING"
    NEXT_GENERATION = "NEXT_GENERATION"
    RESOURCE_PAUSED = "RESOURCE_PAUSED"
    COMPLETED = "COMPLETED"
    CAMPAIGN_STOPPED = "CAMPAIGN_STOPPED"
    CAMPAIGN_FAILED = "CAMPAIGN_FAILED"


class CampaignStopReason(str, Enum):
    """Deterministic reasons for halting an overnight research campaign."""
    NOT_STOPPED = "NOT_STOPPED"
    MAX_DURATION_EXCEEDED = "MAX_DURATION_EXCEEDED"
    MAX_CANDIDATES_REACHED = "MAX_CANDIDATES_REACHED"
    MAX_GENERATIONS_REACHED = "MAX_GENERATIONS_REACHED"
    PLATEAU_DETECTED = "PLATEAU_DETECTED"
    NO_ELIGIBLE_CANDIDATES = "NO_ELIGIBLE_CANDIDATES"
    RESOURCE_LIMIT_EXCEEDED = "RESOURCE_LIMIT_EXCEEDED"
    EXCESSIVE_FAILURES = "EXCESSIVE_FAILURES"
    USER_CANCELLED = "USER_CANCELLED"
    COMPLETED_SUCCESSFULLY = "COMPLETED_SUCCESSFULLY"


@dataclass(frozen=True)
class CampaignConfig:
    """Configurable overnight research campaign parameters and safety limits."""
    campaign_id: str
    context_keys: list[str]
    max_duration_hours: float = 8.0              # Maximum campaign runtime (e.g. overnight window)
    max_candidates_total: int = 40              # Global candidate budget across all generations
    max_generations: int = 4                    # Maximum generational mutation depth
    max_descendants_per_parent: int = 3         # Branching factor per parent
    max_candidates_per_generation: int = 10     # Max candidates to train in a single generation
    plateau_patience_generations: int = 2       # Consecutive generations without lift before plateau stop
    plateau_min_lift: float = 1.0               # Minimum composite score delta considered meaningful lift
    max_consecutive_failures: int = 5           # Max candidate training/eval errors before aborting
    max_memory_mb: int = 12288                  # 12 GB RAM ceiling on 16 GB workstation
    min_trade_volume: int = 30                  # Minimum trades for full confidence
    policy_id: str = "RANK_POLICY_v1.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def compute_config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class CampaignState:
    """Live state and telemetry tracking for an overnight research campaign."""
    campaign_id: str
    config_hash: str
    status: CampaignStatus = CampaignStatus.CREATED
    stop_reason: CampaignStopReason = CampaignStopReason.NOT_STOPPED
    current_generation: int = 0
    total_candidates_generated: int = 0
    total_candidates_trained: int = 0
    total_candidates_evaluated: int = 0
    total_candidates_excluded: int = 0
    total_candidates_pruned: int = 0
    total_failures: int = 0
    consecutive_failures: int = 0
    consecutive_plateau_generations: int = 0
    best_candidate_id: str | None = None
    best_signature_hash: str | None = None
    best_composite_score: float = 0.0
    best_trading_score: float = 0.0
    best_model_score: float = 0.0
    starting_best_score: float = 0.0
    start_time_iso: str = ""
    last_update_iso: str = ""
    end_time_iso: str | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["stop_reason"] = self.stop_reason.value
        return d


@dataclass(frozen=True)
class OvernightCampaignReport:
    """Complete machine-readable campaign dossier for Phase 4F.6 Morning Research Dossier."""
    campaign_id: str
    config: CampaignConfig
    status: CampaignStatus
    stop_reason: CampaignStopReason
    contexts_researched: list[str]
    total_generations_completed: int
    total_candidates_generated: int
    total_candidates_trained: int
    total_candidates_evaluated: int
    total_candidates_excluded: int
    total_candidates_pruned: int
    best_candidate: CandidateEvidenceScore | None
    starting_best_score: float
    best_composite_score: float
    total_score_improvement: float
    best_trading_score: float
    best_model_score: float
    fine_tuning_trials: list[DescendantEvaluationRecord]
    ranked_candidates: list[CandidateEvidenceScore]
    start_time_iso: str
    end_time_iso: str
    duration_seconds: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "stop_reason": self.stop_reason.value,
            "contexts_researched": self.contexts_researched,
            "total_generations_completed": self.total_generations_completed,
            "total_candidates_generated": self.total_candidates_generated,
            "total_candidates_trained": self.total_candidates_trained,
            "total_candidates_evaluated": self.total_candidates_evaluated,
            "total_candidates_excluded": self.total_candidates_excluded,
            "total_candidates_pruned": self.total_candidates_pruned,
            "best_candidate": self.best_candidate.to_dict() if self.best_candidate else None,
            "starting_best_score": self.starting_best_score,
            "best_composite_score": self.best_composite_score,
            "total_score_improvement": self.total_score_improvement,
            "best_trading_score": self.best_trading_score,
            "best_model_score": self.best_model_score,
            "fine_tuning_trials": [t.to_dict() for t in self.fine_tuning_trials],
            "ranked_candidates": [c.to_dict() for c in self.ranked_candidates],
            "start_time_iso": self.start_time_iso,
            "end_time_iso": self.end_time_iso,
            "duration_seconds": self.duration_seconds,
            "warnings": self.warnings,
        }
