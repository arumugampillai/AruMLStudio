"""Persistent Research Memory Subsystem (Phase 4D).

Manages `<data_dir>/analysis.db` storing longitudinal research campaigns,
deterministic experiment signatures, multi-model benchmarks, granular metrics,
regime evaluations, and champion transition history.
"""

from __future__ import annotations

from .benchmarks import (
    create_benchmark_run,
    get_benchmark_metrics,
    get_benchmark_run,
    get_model_benchmark_by_id,
    get_model_benchmarks_for_context,
    record_benchmark_metrics,
    record_model_benchmark,
)
from .campaigns import (
    CampaignStatus,
    allocate_experiment_slot,
    cancel_campaign,
    complete_campaign,
    create_campaign,
    fail_campaign,
    get_campaign,
    get_campaign_experiments,
    link_experiment_to_campaign,
    list_campaigns_for_context,
    pause_campaign,
    resume_campaign,
    start_campaign,
)
from .champion_history import (
    get_champion_history_for_context,
    get_latest_champion_transition,
    reconstruct_champion_at_timestamp,
    record_champion_transition,
)
from .db import (
    analysis_db_path,
    connect_analysis_db,
    init_analysis_db,
    verify_analysis_db_schema,
)
from .feature_comp import (
    analyze_feature_set_composition,
    classify_feature_population,
    get_feature_set_evaluation,
    record_feature_set_evaluation,
)
from .ranking import (
    ROB_POLICY_v1_0,
    RobustnessRankingPolicy,
    compute_pareto_frontier,
    compute_robustness_score,
    normalize_metric,
    persist_context_rankings,
    rank_models_in_context,
)
from .regime_eval import (
    calculate_regime_degradation,
    get_regime_evaluations_for_model,
    get_regime_evaluations_for_regime,
    record_multi_regime_evaluations,
    record_regime_evaluation,
    summarize_regime_feature_affinity,
)
from .schema import (
    ANALYSIS_DB_TABLES_DDL,
    EXPECTED_INDICES,
    EXPECTED_TABLES,
)
from .signature import (
    build_canonical_experiment_payload,
    canonicalize_json,
    check_experiment_exists,
    compute_experiment_signature,
    compute_subcomponent_hash,
    get_experiment_by_signature,
    list_experiments_for_context,
    register_or_get_experiment,
)

__all__ = [
    "ANALYSIS_DB_TABLES_DDL",
    "EXPECTED_INDICES",
    "EXPECTED_TABLES",
    "analysis_db_path",
    "connect_analysis_db",
    "init_analysis_db",
    "verify_analysis_db_schema",
    "build_canonical_experiment_payload",
    "canonicalize_json",
    "check_experiment_exists",
    "compute_experiment_signature",
    "compute_subcomponent_hash",
    "get_experiment_by_signature",
    "list_experiments_for_context",
    "register_or_get_experiment",
    "create_benchmark_run",
    "record_model_benchmark",
    "record_benchmark_metrics",
    "get_benchmark_run",
    "get_model_benchmarks_for_context",
    "get_model_benchmark_by_id",
    "get_benchmark_metrics",
    "classify_feature_population",
    "analyze_feature_set_composition",
    "record_feature_set_evaluation",
    "get_feature_set_evaluation",
    "calculate_regime_degradation",
    "record_regime_evaluation",
    "record_multi_regime_evaluations",
    "get_regime_evaluations_for_model",
    "get_regime_evaluations_for_regime",
    "summarize_regime_feature_affinity",
    "RobustnessRankingPolicy",
    "ROB_POLICY_v1_0",
    "normalize_metric",
    "compute_robustness_score",
    "compute_pareto_frontier",
    "rank_models_in_context",
    "persist_context_rankings",
    "CampaignStatus",
    "create_campaign",
    "get_campaign",
    "list_campaigns_for_context",
    "start_campaign",
    "pause_campaign",
    "resume_campaign",
    "complete_campaign",
    "fail_campaign",
    "cancel_campaign",
    "allocate_experiment_slot",
    "link_experiment_to_campaign",
    "get_campaign_experiments",
    "record_champion_transition",
    "get_champion_history_for_context",
    "get_latest_champion_transition",
    "reconstruct_champion_at_timestamp",
]
