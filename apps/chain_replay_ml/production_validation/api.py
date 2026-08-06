"""Production Validation public API (Phase A resolve + Phase B compute)."""

from __future__ import annotations

from typing import Any, Callable, Iterable

from .compute import run_production_validation_compute
from .recommendation_store import (
    get_recommendation_summary,
    ignore_recommendation,
    list_recommendation_history,
    recommended_for_removal,
    unignore_recommendation,
    update_registry_recommendations,
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


__all__ = [
    "ProductionValidationResult",
    "UnseenDatasetResolveResult",
    "get_recommendation_summary",
    "ignore_recommendation",
    "list_recommendation_history",
    "load_unseen_dataset_status",
    "load_validation_artifacts",
    "persist_registry_recommendations",
    "recommended_for_removal",
    "resolve_unseen_dataset_for_model",
    "run_production_validation",
    "unignore_recommendation",
    "update_registry_recommendations",
]
