"""Feature Distribution Studio compute pipeline (no UI)."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

from chain_replay_ml.feature_distribution_studio.comparison import (
    build_comparison_rows,
    index_importance_rows,
)
from chain_replay_ml.feature_distribution_studio.stats import compute_holdout_stats
from chain_replay_ml.feature_distribution_studio.types import DistributionStudioResult
from chain_replay_ml.feature_distribution_studio.writer import write_studio_artifacts
from chain_replay_ml.feature_importance_studio.compute import _load_holdout_xy
from chain_replay_ml.feature_importance_studio.writer import (
    ARTIFACT_DIRNAME as IMPORTANCE_DIRNAME,
)
from chain_replay_ml.training.paths import model_package_dir, safe_model_name
from chain_replay_ml.training.registry import load_model_detail

ProgressCb = Callable[[dict[str, Any]], None]


def _read_json(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc if isinstance(doc, dict) else {}


def _load_importance_join(package_dir: str) -> tuple[dict[str, dict[str, Any]], bool]:
    path = os.path.join(package_dir, IMPORTANCE_DIRNAME, "comparison.json")
    if not os.path.isfile(path):
        return {}, False
    doc = _read_json(path)
    rows = doc.get("rows") if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        return {}, False
    return index_importance_rows(rows), True


def run_compute(
    *,
    data_dir: str,
    model_name: str,
    package_dir: str | None = None,
    holdout_max_rows: int | None = 50_000,
    progress: ProgressCb | None = None,
    X: Any = None,
    y: Any = None,
) -> DistributionStudioResult:
    """Full compute pipeline → artifacts under the model package."""

    def _tick(stage: str, **extra: Any) -> None:
        if progress:
            progress({"stage": stage, **extra})

    safe = safe_model_name(model_name)
    pkg = package_dir or model_package_dir(data_dir, safe)
    if not os.path.isdir(pkg):
        return DistributionStudioResult(
            ok=False,
            model_name=safe,
            package_dir=pkg,
            artifacts_dir="",
            error=f"Model package not found: {pkg}",
        )

    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    try:
        doc = load_model_detail(data_dir, safe)
    except Exception:
        doc = {"model_name": safe, "config": _read_json(os.path.join(pkg, "config.json"))}

    _tick("load_holdout")
    t_ho = time.perf_counter()
    try:
        X_ho, _y_ho, features, load_meta = _load_holdout_xy(
            data_dir=data_dir,
            package_dir=pkg,
            model_name=safe,
            doc=doc,
            holdout_max_rows=holdout_max_rows,
            X=X,
            y=y,
        )
    except Exception as exc:
        return DistributionStudioResult(
            ok=False,
            model_name=safe,
            package_dir=pkg,
            artifacts_dir="",
            error=f"Failed to load holdout: {exc}",
        )
    timings["load_holdout_sec"] = round(time.perf_counter() - t_ho, 3)

    _tick("stats")
    t_s = time.perf_counter()
    holdout_stats = compute_holdout_stats(X_ho, features)
    timings["stats_sec"] = round(time.perf_counter() - t_s, 3)

    _tick("importance_join")
    importance_by_feature, importance_joined = _load_importance_join(pkg)

    _tick("comparison")
    comparison = build_comparison_rows(
        holdout_stats,
        importance_by_feature=importance_by_feature,
    )

    dataset_load = load_meta.get("dataset_load") or {}
    config_raw = _read_json(os.path.join(pkg, "config.json"))
    run_meta = {
        "model_name": safe,
        "package_dir": pkg,
        "dataset": load_meta.get("dataset"),
        "target": load_meta.get("target"),
        "prediction_type": load_meta.get("prediction_type"),
        "holdout_row_count": load_meta.get("holdout_rows"),
        "holdout_start": load_meta.get("holdout_start"),
        "holdout_stop": load_meta.get("holdout_stop"),
        "feature_count": load_meta.get("feature_count"),
        "dataset_engine_backend": dataset_load.get("backend"),
        "dataset_load": dataset_load,
        "holdout_max_rows": holdout_max_rows,
        "importance_joined": importance_joined,
        "model_version": config_raw.get("model_version") or config_raw.get("modelVersion"),
        "timings_sec": timings,
        "wall_time_sec": round(time.perf_counter() - t0, 3),
        "studio_version": "4.2.0",
    }

    _tick("write_artifacts")
    artifacts_dir = write_studio_artifacts(
        pkg,
        holdout_stats=holdout_stats,
        comparison=comparison,
        run_meta=run_meta,
    )

    return DistributionStudioResult(
        ok=True,
        model_name=safe,
        package_dir=pkg,
        artifacts_dir=artifacts_dir,
        holdout_stats=holdout_stats,
        comparison=comparison,
        meta=run_meta,
    )
