"""Feature Importance Studio API — Phase 4.1.

Compute layer only (no UI). Writes JSON artifacts under the model package.
"""

from __future__ import annotations

from typing import Any, Callable

from .compute import run_compute
from .types import ImportanceStudioResult

ProgressCb = Callable[[dict[str, Any]], None]


def run_feature_importance_studio(
    *,
    data_dir: str,
    model_name: str,
    package_dir: str | None = None,
    holdout_max_rows: int | None = 50_000,
    permutation_n_repeats: int = 5,
    shap_sample_size: int = 400,
    progress: ProgressCb | None = None,
    X: Any = None,
    y: Any = None,
) -> ImportanceStudioResult:
    """Analyze an existing trained model (no retraining).

    v1: Gain/Weight/Cover + holdout Permutation + holdout TreeSHAP +
    comparison table. Artifacts under ``feature_importance_studio/``.

    Optional ``X``/``y`` skip the post-train parquet reload (Create Model).
    """
    return run_compute(
        data_dir=data_dir,
        model_name=model_name,
        package_dir=package_dir,
        holdout_max_rows=holdout_max_rows,
        permutation_n_repeats=permutation_n_repeats,
        shap_sample_size=shap_sample_size,
        progress=progress,
        X=X,
        y=y,
    )
