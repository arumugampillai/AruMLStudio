"""Production Validation — Holdout → True Unseen Days.

Phase A: resolve/create ``unseen_*`` Dataset Registry entries.
Phase B: Holdout vs Unseen rank-based importance compare + dual-confidence.

Freeze: docs/antigravity-doc/production_validation_frozen_v1.md
"""

from __future__ import annotations

from .api import (
    get_recommendation_summary,
    ignore_recommendation,
    list_recommendation_history,
    load_unseen_dataset_status,
    load_validation_artifacts,
    persist_registry_recommendations,
    persist_validation_evidence,
    recommended_for_removal,
    resolve_unseen_dataset_for_model,
    run_production_validation,
    unignore_recommendation,
    update_registry_recommendations,
)
from .types import ProductionValidationResult, UnseenDatasetResolveResult

__all__ = [
    "ProductionValidationResult",
    "UnseenDatasetResolveResult",
    "get_recommendation_summary",
    "ignore_recommendation",
    "list_recommendation_history",
    "load_unseen_dataset_status",
    "load_validation_artifacts",
    "persist_registry_recommendations",
    "persist_validation_evidence",
    "recommended_for_removal",
    "resolve_unseen_dataset_for_model",
    "run_production_validation",
    "unignore_recommendation",
    "update_registry_recommendations",
]
