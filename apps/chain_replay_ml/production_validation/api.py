"""Production Validation public API (Phase A resolve + Phase B compute)."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from .compute import run_production_validation_compute
from .recommendation_store import (
    get_population_recommendations,
    get_recommendation_summary,
    ignore_recommendation,
    list_recommendation_history,
    recommended_for_removal,
    unignore_recommendation,
    update_registry_recommendations,
)
from .dataset_context import (
    DatasetContext,
    build_dataset_context,
    resolve_context_from_model_package,
    resolve_context_or_legacy,
)
from .evidence_store import (
    evidence_db_path,
    get_connection as get_evidence_connection,
    get_experimental_lineage_summaries,
    get_feature_context_summaries,
    preview_policy_impact,
    query_blocked_candidates,
    rebuild_all_projections,
)
from .recommendation_migration import migrate_legacy_recommendation_json
from .recommendation_policy import (
    BasePipelinePolicy,
    ExperimentalLifecyclePolicy,
    FeatureRegistryPolicy,
    RecommendationPolicy,
    ScoringPolicy,
    TrainingDecisionPolicy,
    compute_context_generalization,
    compute_evidence_confidence,
    compute_evidence_score,
    compute_model_consensus,
    compute_recency_staleness,
    compute_score_volatility,
    derive_risk_badges,
    list_policy_history,
    load_policy_store,
    load_recommendation_policy,
    restore_policy_version,
    save_policy_store,
    save_recommendation_policy,
    validate_recommendation_policy,
)
from .model_builder_handoff import (
    build_model_builder_training_bundle,
    export_training_candidates_preset,
)
from .lifecycle_traceability import (
    audit_model_training_feedback_loop,
    get_model_recommendation_provenance,
)
from .feature_graduation import (
    compile_feature_evidence_dossier,
    evaluate_base_pipeline_eligibility,
    evaluate_deprecation_prerequisites,
    evaluate_graduation_prerequisites,
    execute_base_pipeline_promotion,
    execute_feature_deprecation,
    execute_registry_graduation,
    feature_graduation_audit_log_path,
    get_feature_graduation_audit_log,
    is_feature_in_base_pipeline,
)
from .training_decision_engine import (
    TrainingDecisionResult,
    TrainingDecisionState,
    evaluate_candidate_training_eligibility,
    evaluate_population_training_decisions,
    evaluate_training_decision,
    rank_features_for_candidate_generation,
)
from .types import ProductionValidationResult, UnseenDatasetResolveResult
from .unseen_dataset import load_unseen_dataset_status, resolve_unseen_dataset
from .writer import load_validation_artifacts

ProgressCb = Callable[[str, int, int], None]
ComputeProgressCb = Callable[[dict[str, Any]], None]


def resolve_unseen_dataset_for_model(
    *,
    data_dir: str,
    model_name: str,
    create_if_missing: bool = True,
    on_progress: ProgressCb | None = None,
) -> UnseenDatasetResolveResult:
    """Resolve Seen/Unseen and reuse or create ``unseen_*`` Dataset Registry entry."""
    return resolve_unseen_dataset(
        data_dir=data_dir,
        model_name=model_name,
        create_if_missing=create_if_missing,
        on_progress=on_progress,
    )


def run_production_validation(
    *,
    data_dir: str,
    model_name: str,
    package_dir: str | None = None,
    holdout_max_rows: int | None = 50_000,
    unseen_max_rows: int | None = None,
    permutation_n_repeats: int = 5,
    resolve_unseen_if_needed: bool = True,
    progress: ComputeProgressCb | None = None,
) -> ProductionValidationResult:
    """Phase B: Holdout vs Unseen permutation importance + dual confidence."""
    return run_production_validation_compute(
        data_dir=data_dir,
        model_name=model_name,
        package_dir=package_dir,
        holdout_max_rows=holdout_max_rows,
        unseen_max_rows=unseen_max_rows,
        permutation_n_repeats=permutation_n_repeats,
        resolve_unseen_if_needed=resolve_unseen_if_needed,
        progress=progress,
    )


def persist_registry_recommendations(
    *,
    data_dir: str,
    model_name: str,
    package_dir: str | None = None,
    recommendations: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Persist PV recommendation rows into the cumulative recommendation store.

    Does **not** retire registry features or delete pipeline features.
    """
    return update_registry_recommendations(
        data_dir,
        model_name=model_name,
        package_dir=package_dir,
        recommendations=recommendations,
    )


persist_validation_evidence = persist_registry_recommendations


__all__ = [
    "BasePipelinePolicy",
    "DatasetContext",
    "TrainingDecisionPolicy",
    "TrainingDecisionResult",
    "TrainingDecisionState",
    "UnseenDatasetResolveResult",
    "audit_model_training_feedback_loop",
    "build_dataset_context",
    "build_model_builder_training_bundle",
    "compile_feature_evidence_dossier",
    "compute_context_generalization",
    "compute_evidence_confidence",
    "compute_evidence_score",
    "compute_model_consensus",
    "compute_recency_staleness",
    "compute_score_volatility",
    "derive_risk_badges",
    "evaluate_base_pipeline_eligibility",
    "evaluate_candidate_training_eligibility",
    "evaluate_deprecation_prerequisites",
    "evaluate_graduation_prerequisites",
    "evaluate_population_training_decisions",
    "evaluate_training_decision",
    "evidence_db_path",
    "execute_base_pipeline_promotion",
    "execute_feature_deprecation",
    "execute_registry_graduation",
    "export_training_candidates_preset",
    "feature_graduation_audit_log_path",
    "get_evidence_connection",
    "get_experimental_lineage_summaries",
    "get_feature_context_summaries",
    "get_feature_graduation_audit_log",
    "get_model_recommendation_provenance",
    "is_feature_in_base_pipeline",
    "get_population_recommendations",
    "get_recommendation_summary",
    "ignore_recommendation",
    "list_policy_history",
    "list_recommendation_history",
    "load_policy_store",
    "load_recommendation_policy",
    "load_unseen_dataset_status",
    "load_validation_artifacts",
    "migrate_legacy_recommendation_json",
    "persist_registry_recommendations",
    "persist_validation_evidence",
    "preview_policy_impact",
    "query_blocked_candidates",
    "rank_features_for_candidate_generation",
    "rebuild_all_projections",
    "recommended_for_removal",
    "resolve_context_from_model_package",
    "resolve_context_or_legacy",
    "resolve_unseen_dataset_for_model",
    "restore_policy_version",
    "run_production_validation",
    "save_policy_store",
    "save_recommendation_policy",
    "unignore_recommendation",
    "update_registry_recommendations",
    "validate_recommendation_policy",
]
