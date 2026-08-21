"""Autonomous Research Discovery Pipeline Module.

Provides isolated, campaign-scoped experimental sandbox types, persistence layer,
mathematical feature synthesis engine, real walk-forward evaluation,
Feature Studio Evidence DB Bridge, Discovery Governance, Evolutionary Loop,
Next-Day Multi-Session Continuity (Warm-Start), and Promotion Gate to Permanent Feature Registry.
"""

from .bridge import (
    bridge_discovery_evaluation_to_evidence_db,
    resolve_discovery_dataset_context,
)
from .continuity import (
    list_available_discovery_snapshots,
    load_discovery_snapshot_bundle,
    warm_start_discovery_pipeline,
)
from .evaluator import (
    DiscoveryFeatureEvaluator,
    generate_chronological_splits,
)
from .governance import (
    evaluate_discovery_governance_decision,
    run_discovery_pipeline_governance,
)
from .loop import (
    run_autonomous_discovery_loop,
    run_discovery_generation,
)
from .persistence import (
    DISCOVERY_PIPELINES_DDL,
    get_discovery_pipeline_summary,
    init_discovery_pipeline_tables,
    load_discovered_feature_by_hash,
    load_discovered_features,
    load_discovery_pipeline,
    load_discovery_pipeline_by_campaign,
    load_discovery_snapshot,
    load_discovery_snapshots_for_pipeline,
    persist_discovered_features,
    persist_discovery_pipeline,
    persist_discovery_snapshot,
    update_discovered_feature_status,
)
from .promotion import (
    PromotionEligibilityError,
    check_discovery_feature_promotion_eligibility,
    promote_discovery_feature_to_registry,
)
from .synthesizer import (
    DiscoveryFeatureSynthesizer,
    evaluate_discovery_formula,
    generate_discovery_features_from_dataset,
    is_eligible_base_feature,
    zscore,
)
from .types import (
    DiscoveredFeatureSpec,
    DiscoveryLifecycleStatus,
    DiscoveryPipelineBudget,
    DiscoveryPipelineSnapshot,
    DiscoveryPipelineSpec,
    GeneratorStrategy,
    compute_discovery_snapshot_hash,
    compute_formula_hash,
    format_discovered_feature_id,
    format_discovery_pipeline_id,
    normalize_formula_expression,
)

__all__ = [
    "DISCOVERY_PIPELINES_DDL",
    "DiscoveredFeatureSpec",
    "DiscoveryFeatureEvaluator",
    "DiscoveryFeatureSynthesizer",
    "DiscoveryLifecycleStatus",
    "DiscoveryPipelineBudget",
    "DiscoveryPipelineSnapshot",
    "DiscoveryPipelineSpec",
    "GeneratorStrategy",
    "PromotionEligibilityError",
    "bridge_discovery_evaluation_to_evidence_db",
    "check_discovery_feature_promotion_eligibility",
    "compute_discovery_snapshot_hash",
    "compute_formula_hash",
    "evaluate_discovery_formula",
    "evaluate_discovery_governance_decision",
    "format_discovered_feature_id",
    "format_discovery_pipeline_id",
    "generate_chronological_splits",
    "generate_discovery_features_from_dataset",
    "get_discovery_pipeline_summary",
    "init_discovery_pipeline_tables",
    "is_eligible_base_feature",
    "list_available_discovery_snapshots",
    "load_discovered_feature_by_hash",
    "load_discovered_features",
    "load_discovery_pipeline",
    "load_discovery_pipeline_by_campaign",
    "load_discovery_snapshot",
    "load_discovery_snapshot_bundle",
    "load_discovery_snapshots_for_pipeline",
    "normalize_formula_expression",
    "persist_discovered_features",
    "persist_discovery_pipeline",
    "persist_discovery_snapshot",
    "promote_discovery_feature_to_registry",
    "resolve_discovery_dataset_context",
    "run_autonomous_discovery_loop",
    "run_discovery_generation",
    "run_discovery_pipeline_governance",
    "update_discovered_feature_status",
    "warm_start_discovery_pipeline",
    "zscore",
]
