"""Phase 7 — Artifact Catalog, Timeline, Metrics, Experiment Contracts."""

from __future__ import annotations

from .experiment_contracts import (
    build_contract_from_suggestion,
    default_pipeline_from_actions,
    is_runnable,
    load_contract,
    mint_experiment_id,
    register_contract_artifact,
    save_contract,
)
from .executor import ExperimentExecutor, ExperimentRunResult, run_experiment
from .indexer import rebuild_catalog_index
from .metrics import compute_research_metrics, format_evidence_summary
from .service import ArtifactCatalogService
from .store import ArtifactCatalogError, ArtifactCatalogStore
from .timeline import lineage_ancestors, lineage_chain_uris, timeline_chronological
from .types import (
    KNOWN_CAPABILITIES,
    ArtifactRecord,
    ExperimentContract,
    PipelineStep,
    ResearchMetrics,
    TimelineEvent,
)
from .uri import (
    ArtifactUriError,
    diagnostics_uri,
    eval_uri,
    experiment_uri,
    feature_studio_uri,
    is_artifact_uri,
    master_day_uri,
    mint_uri,
    model_uri,
    parse_uri,
    prediction_uri,
    training_uri,
)

__all__ = [
    "KNOWN_CAPABILITIES",
    "ArtifactCatalogError",
    "ArtifactCatalogService",
    "ArtifactCatalogStore",
    "ArtifactRecord",
    "ArtifactUriError",
    "ExperimentContract",
    "ExperimentExecutor",
    "ExperimentRunResult",
    "PipelineStep",
    "ResearchMetrics",
    "TimelineEvent",
    "build_contract_from_suggestion",
    "compute_research_metrics",
    "default_pipeline_from_actions",
    "diagnostics_uri",
    "eval_uri",
    "experiment_uri",
    "feature_studio_uri",
    "format_evidence_summary",
    "is_artifact_uri",
    "is_runnable",
    "lineage_ancestors",
    "lineage_chain_uris",
    "load_contract",
    "master_day_uri",
    "mint_experiment_id",
    "mint_uri",
    "model_uri",
    "parse_uri",
    "prediction_uri",
    "rebuild_catalog_index",
    "register_contract_artifact",
    "run_experiment",
    "save_contract",
    "timeline_chronological",
    "training_uri",
]
