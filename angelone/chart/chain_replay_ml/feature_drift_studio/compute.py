"""Feature Drift Studio compute pipeline (no UI)."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

import numpy as np

from chain_replay_ml.feature_drift_studio.comparison import (
    build_comparison_rows,
    importance_weights_from_rows,
    index_rows,
)
from chain_replay_ml.feature_drift_studio.load import load_wf_holdout_xy
from chain_replay_ml.feature_drift_studio.types import DriftStudioResult
from chain_replay_ml.feature_drift_studio.writer import write_studio_artifacts
from chain_replay_ml.feature_distribution_studio.writer import (
    ARTIFACT_DIRNAME as DISTRIBUTION_DIRNAME,
)
from chain_replay_ml.feature_importance_studio.writer import (
    ARTIFACT_DIRNAME as IMPORTANCE_DIRNAME,
)
from chain_replay_ml.training.holdout_performance import (
    _importance_map_from_doc,
    build_feature_drift_ranking,
    compute_drift_scores,
    compute_similarity_score,
    distribution_summary,
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


def _load_rows_artifact(package_dir: str, dirname: str) -> tuple[list[dict[str, Any]], bool]:
    path = os.path.join(package_dir, dirname, "comparison.json")
    if not os.path.isfile(path):
        return [], False
    doc = _read_json(path)
    rows = doc.get("rows") if isinstance(doc, dict) else None
    if not isinstance(rows, list):
        return [], False
    return [r for r in rows if isinstance(r, dict)], True


def _enrich_stds(
    rows: list[dict[str, Any]],
    X_wf: Any,
    X_ho: Any,
) -> list[dict[str, Any]]:
    """Fill wf_std / holdout_std when missing (ranking usually already sets them)."""
    out: list[dict[str, Any]] = []
    for row in rows:
        feat = str(row.get("feature") or "")
        enriched = dict(row)
        if feat and feat in X_wf.columns and feat in X_ho.columns:
            if enriched.get("wf_std") is None or enriched.get("holdout_std") is None:
                wf_s = distribution_summary(X_wf[feat])
                ho_s = distribution_summary(X_ho[feat])
                if enriched.get("wf_std") is None:
                    enriched["wf_std"] = wf_s.get("std")
                if enriched.get("holdout_std") is None:
                    enriched["holdout_std"] = ho_s.get("std")
        out.append(enriched)
    return out


def _mean_finite(values: list[float | None]) -> float | None:
    nums = [float(v) for v in values if v is not None and np.isfinite(float(v))]
    if not nums:
        return None
    return float(round(sum(nums) / len(nums), 4))


def run_compute(
    *,
    data_dir: str,
    model_name: str,
    package_dir: str | None = None,
    holdout_max_rows: int | None = 50_000,
    wf_max_rows: int | None = 100_000,
    progress: ProgressCb | None = None,
    X: Any = None,
    y: Any = None,
) -> DriftStudioResult:
    def _tick(stage: str, **extra: Any) -> None:
        if progress:
            progress({"stage": stage, **extra})

    safe = safe_model_name(model_name)
    pkg = package_dir or model_package_dir(data_dir, safe)
    if not os.path.isdir(pkg):
        return DriftStudioResult(
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

    _tick("load_slices")
    t_load = time.perf_counter()
    try:
        X_wf, y_wf, X_ho, y_ho, features, load_meta = load_wf_holdout_xy(
            data_dir=data_dir,
            package_dir=pkg,
            model_name=safe,
            doc=doc,
            holdout_max_rows=holdout_max_rows,
            wf_max_rows=wf_max_rows,
            X=X,
            y=y,
        )
    except Exception as exc:
        return DriftStudioResult(
            ok=False,
            model_name=safe,
            package_dir=pkg,
            artifacts_dir="",
            error=f"Failed to load WF/holdout: {exc}",
        )
    timings["load_slices_sec"] = round(time.perf_counter() - t_load, 3)

    if len(X_wf) == 0 or len(X_ho) == 0:
        return DriftStudioResult(
            ok=False,
            model_name=safe,
            package_dir=pkg,
            artifacts_dir="",
            error="Empty WF or holdout slice — cannot compute drift",
        )

    imp_rows, importance_joined = _load_rows_artifact(pkg, IMPORTANCE_DIRNAME)
    dist_rows, distribution_joined = _load_rows_artifact(pkg, DISTRIBUTION_DIRNAME)

    importance_map = importance_weights_from_rows(imp_rows) if importance_joined else {}
    if not importance_map:
        importance_map = _importance_map_from_doc(doc)

    _tick("drift_ranking")
    t_d = time.perf_counter()
    drift_rows = build_feature_drift_ranking(
        X_wf,
        X_ho,
        features,
        importance_map=importance_map,
        limit=max(len(features), 1),
    )
    drift_rows = _enrich_stds(drift_rows, X_wf, X_ho)
    timings["drift_ranking_sec"] = round(time.perf_counter() - t_d, 3)

    _tick("aggregates")
    drift_scores = compute_drift_scores(
        target_wf=y_wf,
        target_holdout=y_ho,
        baseline_wf=None,
        baseline_holdout=None,
        vol_wf=None,
        vol_holdout=None,
        feature_ranking=drift_rows,
    )
    similarity_pct = compute_similarity_score(drift_scores)

    _tick("comparison")
    comparison = build_comparison_rows(
        drift_rows,
        importance_by_feature=index_rows(imp_rows) if importance_joined else None,
        distribution_by_feature=index_rows(dist_rows) if distribution_joined else None,
    )

    dataset_load = load_meta.get("dataset_load") or {}
    config_raw = _read_json(os.path.join(pkg, "config.json"))
    avg_drift_pct = _mean_finite(
        [r.get("drift_pct") for r in drift_rows if isinstance(r, dict)]
    )
    # Prefer absolute drift % for "Average Drift" so opposite signs don't cancel.
    avg_abs_drift_pct = _mean_finite(
        [
            abs(float(r["drift_pct"]))
            for r in drift_rows
            if isinstance(r, dict) and r.get("drift_pct") is not None
        ]
    )
    avg_ks = _mean_finite(
        [r.get("ks_statistic") for r in drift_rows if isinstance(r, dict)]
    )
    avg_wasserstein = _mean_finite(
        [r.get("wasserstein_distance") for r in drift_rows if isinstance(r, dict)]
    )
    avg_wasserstein_normalized = _mean_finite(
        [r.get("wasserstein_normalized") for r in drift_rows if isinstance(r, dict)]
    )
    avg_risk_score = _mean_finite(
        [r.get("risk_score") for r in drift_rows if isinstance(r, dict)]
    )
    run_meta = {
        "model_name": safe,
        "package_dir": pkg,
        "dataset": load_meta.get("dataset"),
        "target": load_meta.get("target"),
        "prediction_type": load_meta.get("prediction_type"),
        "holdout_start": load_meta.get("holdout_start"),
        "holdout_stop": load_meta.get("holdout_stop"),
        "wf_row_count": load_meta.get("wf_rows"),
        "holdout_row_count": load_meta.get("holdout_rows"),
        "feature_count": load_meta.get("feature_count"),
        "dataset_engine_backend": dataset_load.get("backend"),
        "dataset_load": dataset_load,
        "holdout_max_rows": holdout_max_rows,
        "wf_max_rows": wf_max_rows,
        "importance_joined": importance_joined,
        "distribution_joined": distribution_joined,
        "drift_scores": drift_scores,
        "feature_drift_pct": drift_scores.get("feature"),
        "target_drift_pct": drift_scores.get("target"),
        "similarity_pct": similarity_pct,
        "average_drift_pct": avg_abs_drift_pct if avg_abs_drift_pct is not None else avg_drift_pct,
        "average_ks": avg_ks,
        "average_wasserstein": avg_wasserstein,
        "average_wasserstein_normalized": avg_wasserstein_normalized,
        "average_risk_score": avg_risk_score,
        "model_version": config_raw.get("model_version") or config_raw.get("modelVersion"),
        "timings_sec": timings,
        "wall_time_sec": round(time.perf_counter() - t0, 3),
        "studio_version": "5.2.0",
        "schema_version": 2,
        "drift_schema": "v2",
        "high_risk_count": sum(1 for r in drift_rows if r.get("risk") == "high"),
        "medium_risk_count": sum(1 for r in drift_rows if r.get("risk") == "medium"),
    }

    _tick("write_artifacts")
    artifacts_dir = write_studio_artifacts(
        pkg,
        drift_rows=drift_rows,
        comparison=comparison,
        run_meta=run_meta,
    )

    return DriftStudioResult(
        ok=True,
        model_name=safe,
        package_dir=pkg,
        artifacts_dir=artifacts_dir,
        drift_rows=drift_rows,
        comparison=comparison,
        meta=run_meta,
    )
