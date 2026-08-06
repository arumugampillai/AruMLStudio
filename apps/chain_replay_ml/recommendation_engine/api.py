"""Recommendation Engine API — Phase 5.3 (artifact-only suggestions)."""

from __future__ import annotations

from typing import Any, Callable

from .compute import run_compute
from .types import RecommendationEngineResult

ProgressCb = Callable[[dict[str, Any]], None]


def run_recommendation_engine(
    *,
    data_dir: str,
    model_name: str,
    package_dir: str | None = None,
    thresholds: dict[str, Any] | None = None,
    family_by_name: dict[str, str] | None = None,
    require: tuple[str, ...] = (),
    progress: ProgressCb | None = None,
) -> RecommendationEngineResult:
    """Join existing Feature Studio artifacts into Experiment Planner suggestions.

    Writes under ``models/<Model>/experiment_planner/``. Does not retrain,
    recompute Importance/Distribution/Drift, or touch Dataset Engine.
    """
    return run_compute(
        data_dir=data_dir,
        model_name=model_name,
        package_dir=package_dir,
        thresholds=thresholds,
        family_by_name=family_by_name,
        require=require,
        progress=progress,
    )
