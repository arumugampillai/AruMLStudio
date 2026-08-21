"""Data types and schemas for Phase 4F.2: Automated Candidate Generator & Lineage Tracker.

Defines:
1. MutationType: Enumeration of deterministic candidate mutation categories.
2. CandidateEligibility: Eligibility classification (ELIGIBLE, CAUTION, EXCLUDED, ALREADY_EVALUATED).
3. CandidateGenerationBudget: Configurable workstation resource and search limits.
4. CandidateLineageRecord: Cryptographic lineage tracking (parent-child relationship & opportunity provenance).
5. CandidateSpec: Full candidate model specification ready for downstream training.
6. CandidateGenerationResult: Result payload from the candidate generation service.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class MutationType(str, Enum):
    """Categories of candidate specification mutations."""
    FULL_FEATURE_BASELINE = "FULL_FEATURE_BASELINE"
    COLD_START = "COLD_START"
    ALGORITHM_MUTATION = "ALGORITHM_MUTATION"
    HYPERPARAMETER_MUTATION = "HYPERPARAMETER_MUTATION"
    FEATURE_SUBSET_MUTATION = "FEATURE_SUBSET_MUTATION"
    FEATURE_ELIMINATION = "FEATURE_ELIMINATION"
    TARGET_HORIZON_MUTATION = "TARGET_HORIZON_MUTATION"
    REGIME_SPECIALIZATION = "REGIME_SPECIALIZATION"


class CandidateEligibility(str, Enum):
    """Eligibility classification for generated candidates."""
    ELIGIBLE = "ELIGIBLE"
    CAUTION = "CAUTION"
    EXCLUDED = "EXCLUDED"
    ALREADY_EVALUATED = "ALREADY_EVALUATED"


@dataclass(frozen=True)
class CandidateGenerationBudget:
    """Research-configurable resource budget and search depth limits (16 GB RAM safety)."""
    max_candidates_per_campaign: int = 30       # Maximum total candidates in one generation pass
    max_generations: int = 4                    # Maximum descendant depth (0 -> 1 -> 2 -> 3)
    max_descendants_per_parent: int = 3         # Branching factor ceiling per parent
    max_features_per_candidate: int = 2000      # Ceiling to allow complete eligible dataset universes
    max_concurrent_workers: int = 1             # Strict serial execution default

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateLineageRecord:
    """Cryptographic lineage tracking record linking child candidate to its parent and Phase 4E opportunity."""
    candidate_id: str
    signature_hash: str
    parent_candidate_id: str | None = None
    parent_signature_hash: str | None = None
    generation_number: int = 0
    campaign_id: str | None = None
    context_key: str = "CONTEXT_KEY"
    mutation_type: MutationType = MutationType.COLD_START
    mutation_description: str = "Initial baseline candidate"
    opportunity_id: str | None = None          # Phase 4E opportunity ID (e.g. OPP_NIFTY_3s_DIR_5m_R001_01)
    opportunity_type: str | None = None        # Phase 4E opportunity archetype (e.g. INTERACTION_SYNERGY_TEST)
    priority_score: float | None = None        # Phase 4E multi-objective priority score
    feature_elimination_strategy: str | None = None # NONE | SHAP | RFE | PERMUTATION
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mutation_type"] = self.mutation_type.value
        return d


@dataclass
class CandidateSpec:
    """Complete candidate model specification ready for downstream execution."""
    candidate_id: str
    context_key: str
    market: str
    sampling_interval_sec: int
    task_type: str
    prediction_horizon: str
    regime_id: str
    regime_definition_hash: str
    dataset_snapshot_hash: str
    features: list[str]
    algorithm: str
    hyperparameters: dict[str, Any]
    walk_forward_config: dict[str, Any]
    random_seed: int = 42
    signature_hash: str = ""
    lineage: CandidateLineageRecord | None = None
    eligibility: CandidateEligibility = CandidateEligibility.ELIGIBLE
    exclusion_reasons: list[str] = field(default_factory=list)
    caution_warnings: list[str] = field(default_factory=list)
    feature_elimination_strategy: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["eligibility"] = self.eligibility.value
        if self.lineage:
            d["lineage"] = self.lineage.to_dict()
        return d

    def to_experiment_spec(self) -> dict[str, Any]:
        """Convert to canonical experiment payload spec format for Phase 4D signature calculation."""
        return {
            "market": self.market,
            "sampling_interval_sec": self.sampling_interval_sec,
            "task_type": self.task_type,
            "prediction_horizon": self.prediction_horizon,
            "regime_id": self.regime_id,
            "regime_definition_hash": self.regime_definition_hash,
            "dataset_snapshot_hash": self.dataset_snapshot_hash,
            "features": list(self.features),
            "algorithm": self.algorithm,
            "hyperparameters": dict(self.hyperparameters),
            "walk_forward_config": dict(self.walk_forward_config),
            "random_seed": int(self.random_seed),
        }


@dataclass
class CandidateGenerationResult:
    """Result returned by the candidate generation service."""
    context_key: str
    campaign_id: str | None
    candidates: list[CandidateSpec]
    total_generated: int
    eligible_count: int
    caution_count: int
    excluded_count: int
    already_evaluated_count: int
    budget_exhausted: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_key": self.context_key,
            "campaign_id": self.campaign_id,
            "candidates": [c.to_dict() for c in self.candidates],
            "total_generated": self.total_generated,
            "eligible_count": self.eligible_count,
            "caution_count": self.caution_count,
            "excluded_count": self.excluded_count,
            "already_evaluated_count": self.already_evaluated_count,
            "budget_exhausted": self.budget_exhausted,
        }
