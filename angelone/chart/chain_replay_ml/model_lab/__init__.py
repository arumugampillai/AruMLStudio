"""Model Lab — frozen-model research workspace (Phase 1 meta + Phase 2 predictions)."""

from __future__ import annotations

from .paths import (
    DEFAULT_MODEL_RESEARCH_DIR,
    lab_db_path,
    latest_lab_path,
    list_lab_db_paths,
    next_lab_version,
    resolve_model_research_dir,
)
from .prediction_builder import (
    build_prediction_dataset,
    benchmark_prediction_workers,
    enrich_prediction_day_metadata,
    parent_registry_day_rows,
    prediction_build_summary,
    prediction_dataset_status,
    prediction_days_ui_skeleton,
    sync_prediction_build_catalog,
    validate_prediction_inputs,
    validate_prediction_output,
)
from .prediction_metadata import (
    prediction_metadata_path,
    read_prediction_metadata,
    rebuild_prediction_metadata_from_db,
    write_prediction_metadata,
)
from .prediction_export import export_prediction_dataset
from .prediction_parallel import DEFAULT_PREDICTION_WORKERS
from .research_dashboard import (
    OVERALL_STAT_ROWS,
    compute_overall_statistics,
    compute_research_dashboard,
    refresh_research_dashboard_cache,
)
from .feature_research import analyze_feature, list_research_features
from .service import (
    create_model_lab,
    default_lab_display_name,
    find_latest_lab,
    list_labs_for_model,
    list_research_lab_summaries,
    load_lab,
)
from .store import (
    LAB_PHASE,
    LAB_SCHEMA_VERSION,
    STATUS_ARCHIVED,
    STATUS_CREATED,
    STATUS_ERROR,
    STATUS_READY,
    ModelLabInfo,
    ModelLabStore,
)

__all__ = [
    "DEFAULT_MODEL_RESEARCH_DIR",
    "DEFAULT_PREDICTION_WORKERS",
    "LAB_PHASE",
    "LAB_SCHEMA_VERSION",
    "STATUS_ARCHIVED",
    "STATUS_CREATED",
    "STATUS_ERROR",
    "STATUS_READY",
    "ModelLabInfo",
    "ModelLabStore",
    "benchmark_prediction_workers",
    "build_prediction_dataset",
    "create_model_lab",
    "default_lab_display_name",
    "export_prediction_dataset",
    "find_latest_lab",
    "lab_db_path",
    "latest_lab_path",
    "list_lab_db_paths",
    "list_labs_for_model",
    "list_research_lab_summaries",
    "load_lab",
    "next_lab_version",
    "enrich_prediction_day_metadata",
    "prediction_build_summary",
    "prediction_dataset_status",
    "prediction_days_ui_skeleton",
    "prediction_metadata_path",
    "parent_registry_day_rows",
    "read_prediction_metadata",
    "rebuild_prediction_metadata_from_db",
    "OVERALL_STAT_ROWS",
    "compute_overall_statistics",
    "compute_research_dashboard",
    "refresh_research_dashboard_cache",
    "analyze_feature",
    "list_research_features",
    "resolve_model_research_dir",
    "sync_prediction_build_catalog",
    "validate_prediction_inputs",
    "validate_prediction_output",
    "write_prediction_metadata",
]
