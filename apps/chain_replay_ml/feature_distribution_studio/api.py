"""Feature Distribution Studio API — Phase 4.2.

Compute layer only (no UI). Writes JSON artifacts under the model package.
"""

from __future__ import annotations

from typing import Any, Callable

from .compute import run_compute
from .types import DistributionStudioResult

ProgressCb = Callable[[dict[str, Any]], None]


def run_feature_distribution_studio(
    *,
    data_dir: str,
    model_name: str,
    package_dir: str | None = None,
    holdout_max_rows: int | None = 50_000,
    progress: ProgressCb | None = None,
    X: Any = None,
    y: Any = None,
) -> DistributionStudioResult:
    """Analyze holdout feature distributions for an existing model package.

    v1: univariate stats + optional Importance rank join.
    Artifacts under ``feature_distribution_studio/``.

    Optional ``X``/``y`` skip the post-train parquet reload (Create Model).
    """
    return run_compute(
        data_dir=data_dir,
        model_name=model_name,
        package_dir=package_dir,
        holdout_max_rows=holdout_max_rows,
        progress=progress,
        X=X,
        y=y,
    )
