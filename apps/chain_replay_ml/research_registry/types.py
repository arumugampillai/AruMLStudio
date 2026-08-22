"""Data types and schemas for Phase 4F.7: Autonomous Research Registry (Doc 16).

Defines structured data models representing permanent historical research executions,
generational snapshot linkages, and cross-research formula memory priors.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
from typing import Any


class ResearchStatus(str, Enum):
    """Lifecycle execution status of an autonomous research campaign."""
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"
    PAUSED = "PAUSED"
    CRASHED = "CRASHED"


class FormulaGlobalStatus(str, Enum):
    """Longitudinal status of a synthesized formula across all historical research."""
    PROMISING = "PROMISING"           # Repeated KEEP / positive lift
    WATCH = "WATCH"                   # Moderate performance / watch status
    REJECTED_DRIFT = "REJECTED_DRIFT" # Severe KS distribution drift (D_KS > 0.35)
    REJECTED_NOISE = "REJECTED_NOISE" # Negative lift / low consistency
    PROMOTED = "PROMOTED"             # Promoted to permanent Feature Registry (FR_xxxx)


@dataclass(frozen=True)
class ResearchRegistryRecord:
    """Immutable metadata record representing a completed or active autonomous research run."""
    research_id: str
    campaign_id: str
    context_key: str
    context_id: str
    
    dataset_name: str
    dataset_snapshot_hash: str
    base_pipeline_id: str = "PL_0001"
    base_feature_count: int = 171
    registry_feature_count: int = 211
    
    started_at: str = ""
    finished_at: str | None = None
    duration_seconds: float = 0.0
    status: ResearchStatus = ResearchStatus.RUNNING
    stop_reason: str = "IN_PROGRESS"
    
    algorithms_used: list[str] = field(default_factory=list)
    elimination_strategy: str = "SHAP_AND_EVIDENCE"
    max_generations_configured: int = 100
    actual_generations_completed: int = 0
    max_candidates_configured: int = 500
    candidates_generated: int = 0
    candidates_evaluated: int = 0
    candidates_pruned: int = 0
    
    best_candidate_id: str | None = None
    best_composite_score: float = 0.0
    best_trading_score: float = 0.0
    best_model_score: float = 0.0
    best_win_rate_pct: float = 0.0
    best_profit_factor: float = 0.0
    best_max_drawdown_pct: float = 0.0
    starting_best_score: float = 0.0
    total_score_lift: float = 0.0
    
    discovery_pipeline_id: str = ""
    final_discovery_snapshot_hash: str | None = None
    total_df_features_created: int = 0
    unique_formula_count: int = 0
    keep_count: int = 0
    watch_count: int = 0
    remove_count: int = 0
    active_discovery_pool: int = 0
    promoted_feature_count: int = 0
    
    research_config_json: str = "{}"
    research_outcome_json: str = "{}"
    failure_reason: str | None = None
    architecture_version: str = "2.2.0"
    code_version: str = "1.0.0"
    
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value if isinstance(self.status, ResearchStatus) else str(self.status)
        return d


@dataclass(frozen=True)
class ResearchGenerationLinkage:
    """Lightweight linkage between a research run, candidate milestone, and discovery snapshot."""
    snapshot_record_id: int | None
    research_id: str
    campaign_id: str
    generation_number: int
    discovery_snapshot_hash: str
    candidates_evaluated: int
    generation_best_score: float
    generation_best_candidate_id: str
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FormulaMemoryRecord:
    """Longitudinal empirical prior record for a synthesized mathematical formula."""
    formula_hash: str
    canonical_formula_expression: str
    generator_strategy: str
    parent_features: list[str]
    first_discovered_research_id: str
    first_discovered_at: str
    last_evaluated_research_id: str
    last_evaluated_at: str
    total_researches_tested: int
    total_evaluations_count: int
    highest_evidence_score: float
    lowest_ks_drift: float
    best_marginal_delta_auc: float
    global_status: FormulaGlobalStatus
    last_governance_verdict: str
    last_governance_reason: str
    context_lock: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["global_status"] = self.global_status.value if isinstance(self.global_status, FormulaGlobalStatus) else str(self.global_status)
        return d
