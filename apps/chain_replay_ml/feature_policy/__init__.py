"""Feature Policy Engine — metadata-driven warm-up, gap reset, and NULL propagation.

Reusable across dataset builder, replay, paper/live trading, and research.
"""

from __future__ import annotations

from .engine import EngineStats, FeaturePolicyEngine, FeatureRuntimeState
from .health import FeatureHealthRow, FeatureHealthTracker
from .lifecycle import should_reset_on_gap, should_reset_on_session_start
from .build_readiness import (
    FEATURE_READINESS_POLICY_VERSION,
    build_feature_readiness_manifest,
    enforce_readiness_on_rows,
    merge_readiness_stats,
    validate_readiness_compliance,
)
from .build_validation import (
    build_policy_report,
    build_validation_preview,
    compute_feature_health_from_rows,
    finalize_build_policy_manifest,
    sample_rows_from_sqlite,
)
from .manifest import build_dataset_policy_manifest, build_report_from_manifest
from .metadata import (
    FeaturePolicyMetadata,
    build_feature_policy_metadata,
    parse_warmup_string,
    resolve_effective_warmup,
)
from .registry import FeaturePolicyRegistry, load_feature_policy_registry, registry_version_info
from .types import (
    DEFAULT_GAP_MAX_SEC,
    FEATURE_POLICY_VERSION,
    FeatureCategory,
    FeatureLifecycle,
    RollingType,
    WarmupMode,
)
from .warmup_simulator import (
    WarmupSimulationResult,
    compare_ema_readiness,
    expand_timestamps_with_gaps,
    list_trading_days,
    simulate_warmup,
)

__all__ = [
    "DEFAULT_GAP_MAX_SEC",
    "FEATURE_POLICY_VERSION",
    "FEATURE_READINESS_POLICY_VERSION",
    "EngineStats",
    "FeatureCategory",
    "FeatureHealthRow",
    "FeatureHealthTracker",
    "FeatureLifecycle",
    "FeaturePolicyEngine",
    "FeaturePolicyMetadata",
    "FeaturePolicyRegistry",
    "FeatureRuntimeState",
    "RollingType",
    "WarmupMode",
    "build_dataset_policy_manifest",
    "build_feature_policy_metadata",
    "build_feature_readiness_manifest",
    "build_policy_report",
    "build_report_from_manifest",
    "build_validation_preview",
    "compute_feature_health_from_rows",
    "enforce_readiness_on_rows",
    "finalize_build_policy_manifest",
    "load_feature_policy_registry",
    "sample_rows_from_sqlite",
    "merge_readiness_stats",
    "parse_warmup_string",
    "registry_version_info",
    "resolve_effective_warmup",
    "should_reset_on_gap",
    "should_reset_on_session_start",
    "WarmupSimulationResult",
    "compare_ema_readiness",
    "expand_timestamps_with_gaps",
    "list_trading_days",
    "simulate_warmup",
    "validate_readiness_compliance",
]
