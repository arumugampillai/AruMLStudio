"""Phase 4B.0: Feature Selection & Attribution Data Contracts, Enums, and Configuration.

Authoritative contracts for composite non-linear feature selection, multi-method
attribution harmonization (MI + Permutation + TreeSHAP), and canonical feature
governance (KEEP / WATCH / REMOVE).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping, Sequence


# =============================================================================
# 1. CANONICAL LIFECYCLE & DIAGNOSTIC ENUMS
# =============================================================================

class CanonicalFeatureAction(str, Enum):
    """Authoritative macro-lifecycle governance action for a feature."""
    KEEP = "KEEP"
    WATCH = "WATCH"
    REMOVE = "REMOVE"


class DiscoveryDiagnosticAction(str, Enum):
    """Stage 1 (Pre-Training / Discovery) granular diagnostic action."""
    KEEP = "KEEP"
    REVIEW_FAMILY = "REVIEW FAMILY"
    MERGE_CANDIDATE = "MERGE CANDIDATE"
    RETIRE_CANDIDATE = "RETIRE CANDIDATE"


class ValidationDiagnosticAction(str, Enum):
    """Stage 2 (Post-Training / Validation) model-quality diagnostic action."""
    PRODUCTION_READY = "PRODUCTION READY"
    NEEDS_REVIEW = "NEEDS REVIEW"
    UNSTABLE = "UNSTABLE"


class AttributionStage(str, Enum):
    """Attribution evaluation stage distinguishing pre-training from post-training."""
    STAGE_DISCOVERY = "discovery"
    STAGE_VALIDATION = "validation"


class CompositeStrategy(str, Enum):
    """Selection strategy identifier for candidate dataset resolution."""
    COMPOSITE_NONLINEAR = "composite_nonlinear"
    MI_ONLY = "mi_only"
    PERMUTATION_ONLY = "permutation_only"
    HCA_CORR = "hca_corr"


# =============================================================================
# 2. ACTION RECONCILIATION & MAPPING HELPERS
# =============================================================================

_DISCOVERY_TO_CANONICAL_MAP: dict[str, CanonicalFeatureAction] = {
    DiscoveryDiagnosticAction.KEEP.value: CanonicalFeatureAction.KEEP,
    DiscoveryDiagnosticAction.REVIEW_FAMILY.value: CanonicalFeatureAction.WATCH,
    DiscoveryDiagnosticAction.MERGE_CANDIDATE.value: CanonicalFeatureAction.WATCH,
    DiscoveryDiagnosticAction.RETIRE_CANDIDATE.value: CanonicalFeatureAction.REMOVE,
    # Aliases
    "KEEP": CanonicalFeatureAction.KEEP,
    "REVIEW": CanonicalFeatureAction.WATCH,
    "REVIEW FAMILY": CanonicalFeatureAction.WATCH,
    "FAMILY DECISION REQUIRED": CanonicalFeatureAction.WATCH,
    "MERGE": CanonicalFeatureAction.WATCH,
    "MERGE CANDIDATE": CanonicalFeatureAction.WATCH,
    "RETIRE": CanonicalFeatureAction.REMOVE,
    "RETIRE CANDIDATE": CanonicalFeatureAction.REMOVE,
}

_VALIDATION_TO_CANONICAL_MAP: dict[str, CanonicalFeatureAction] = {
    ValidationDiagnosticAction.PRODUCTION_READY.value: CanonicalFeatureAction.KEEP,
    ValidationDiagnosticAction.NEEDS_REVIEW.value: CanonicalFeatureAction.WATCH,
    ValidationDiagnosticAction.UNSTABLE.value: CanonicalFeatureAction.REMOVE,
    # Aliases
    "PRODUCTION READY": CanonicalFeatureAction.KEEP,
    "NEEDS REVIEW": CanonicalFeatureAction.WATCH,
    "UNSTABLE": CanonicalFeatureAction.REMOVE,
}


def map_discovery_action_to_canonical(action: DiscoveryDiagnosticAction | str) -> CanonicalFeatureAction:
    """Map an internal Stage-1 discovery diagnostic action to the canonical KEEP/WATCH/REMOVE lifecycle."""
    act_str = str(action.value if isinstance(action, DiscoveryDiagnosticAction) else action).strip().upper()
    return _DISCOVERY_TO_CANONICAL_MAP.get(act_str, CanonicalFeatureAction.WATCH)


def map_validation_action_to_canonical(action: ValidationDiagnosticAction | str) -> CanonicalFeatureAction:
    """Map an internal Stage-2 validation diagnostic action to the canonical KEEP/WATCH/REMOVE lifecycle."""
    act_str = str(action.value if isinstance(action, ValidationDiagnosticAction) else action).strip().upper()
    return _VALIDATION_TO_CANONICAL_MAP.get(act_str, CanonicalFeatureAction.WATCH)


# =============================================================================
# 3. CONFIGURATION DATA CONTRACTS
# =============================================================================

@dataclass(frozen=True)
class CompositeSelectionConfig:
    """Configuration container for Phase 4B Composite Non-Linear Feature Selection.
    
    Encapsulates all pre-training, post-training, correlation, and data-health thresholds
    with explicit, configurable defaults.
    """
    # Pre-Training Weights (Discovery Stage: SHAP is strictly prohibited)
    weight_mi_pre: float = 0.45
    weight_perm_pre: float = 0.40

    # Post-Training Weights (Validation Stage: TreeSHAP + OOS Permutation + MI)
    weight_shap_post: float = 0.45
    weight_perm_post: float = 0.30
    weight_mi_post: float = 0.25

    # Correlation & Redundancy Thresholds
    corr_threshold: float = 0.95
    extreme_duplicate_threshold: float = 0.999
    moderate_corr_threshold: float = 0.85

    # Data Health & Missingness Gates
    max_null_pct: float = 5.0
    min_coverage_pct: float = 90.0

    # Rating Band Percentiles
    high_band_pct: float = 66.67
    low_band_pct: float = 33.33

    # Resource & Safety Bounds (16 GB Workstation)
    max_subsample_rows: int = 10_000
    random_seed: int = 42

    def __post_init__(self) -> None:
        # Validate pre-training weights
        pre_sum = self.weight_mi_pre + self.weight_perm_pre
        if pre_sum <= 0.0:
            raise ValueError(f"Pre-training weights sum must be > 0, got {pre_sum}")

        # Validate post-training weights
        post_sum = self.weight_shap_post + self.weight_perm_post + self.weight_mi_post
        if post_sum <= 0.0:
            raise ValueError(f"Post-training weights sum must be > 0, got {post_sum}")

        # Validate thresholds
        if not (0.0 <= self.corr_threshold <= 1.0):
            raise ValueError(f"corr_threshold must be in [0, 1], got {self.corr_threshold}")
        if not (0.0 <= self.max_null_pct <= 100.0):
            raise ValueError(f"max_null_pct must be in [0, 100], got {self.max_null_pct}")
        if self.max_subsample_rows <= 0:
            raise ValueError(f"max_subsample_rows must be > 0, got {self.max_subsample_rows}")


DEFAULT_COMPOSITE_SELECTION_CONFIG = CompositeSelectionConfig()


# =============================================================================
# 4. SCORE CONTAINERS & PROVENANCE RECORDS
# =============================================================================

@dataclass(frozen=True)
class FeatureAttributionRecord:
    """Individual feature attribution and score record across methods."""
    feature_name: str
    stage: AttributionStage
    # Raw metrics
    mi_raw: float | None = None
    perm_importance_raw: float | None = None
    shap_importance_raw: float | None = None
    abs_corr_peer: float | None = None
    peer_feature_name: str | None = None
    coverage_pct: float = 100.0
    # Normalized percentiles (0.0 to 100.0)
    mi_pct: float | None = None
    perm_pct: float | None = None
    shap_pct: float | None = None
    # Final synthesized score and actions
    composite_score: float = 0.0
    composite_rank: int = 0
    diagnostic_action: str = ""
    canonical_action: CanonicalFeatureAction = CanonicalFeatureAction.WATCH
    confidence: str = "Medium"
    reason: str = ""
    family_id: str | None = None


@dataclass(frozen=True)
class CompositeAttributionResult:
    """Run-level summary of composite feature selection and attributions."""
    run_id: str
    dataset_id: str
    target_column: str
    stage: AttributionStage
    strategy: CompositeStrategy
    total_features_evaluated: int
    selected_feature_count: int
    selected_features: list[str]
    quarantined_features: list[str]
    pruned_collinear_features: list[str]
    attributions: dict[str, FeatureAttributionRecord] = field(default_factory=dict)
    config: CompositeSelectionConfig = field(default_factory=CompositeSelectionConfig)


@dataclass(frozen=True)
class CompositeProvenanceRecord:
    """Cryptographic provenance record documenting selection parameters and outcome."""
    run_id: str
    dataset_id: str
    strategy: str
    config_json: str
    sha256_hash: str
    input_feature_count: int
    selected_feature_count: int
    selected_features: list[str]
    created_at_iso: str
