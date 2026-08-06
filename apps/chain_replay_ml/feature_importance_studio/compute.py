"""Feature Importance Studio compute pipeline (no UI)."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

import pandas as pd

from chain_replay_ml.feature_importance_studio.comparison import build_comparison_rows
from chain_replay_ml.feature_importance_studio.native import compute_native_xgb_importance
from chain_replay_ml.feature_importance_studio.permutation import compute_permutation_importance
from chain_replay_ml.feature_importance_studio.shap import compute_shap_holdout
from chain_replay_ml.feature_importance_studio.types import ImportanceStudioResult
from chain_replay_ml.feature_importance_studio.writer import write_studio_artifacts
from chain_replay_ml.training.config import normalize_training_config
from chain_replay_ml.training.dataset_loader import load_training_xy
from chain_replay_ml.training.holdout_performance import resolve_holdout_slice_with_fallback
from chain_replay_ml.training.model_runtime import (
    load_prediction_model,
    resolve_production_model_path,
)
from chain_replay_ml.training.paths import model_package_dir, safe_model_name
from chain_replay_ml.training.registry import _selected_feature_names, load_model_detail
ProgressCb = Callable[[dict[str, Any]], None]


def _read_json(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc if isinstance(doc, dict) else {}


def _prediction_kind(config: dict[str, Any]) -> str:
    raw = str(
        config.get("prediction_type")
        or config.get("predictionType")
        or ""
    ).strip().lower()
    if raw in ("binary", "classification", "classifier"):
        return "binary"
    return "regression"


def _load_holdout_xy(
    *,
    data_dir: str,
    package_dir: str,
    model_name: str,
    doc: dict[str, Any],
    holdout_max_rows: int | None,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, Any]]:
    """Load training matrix via Dataset Engine path, then slice holdout.

    When ``X``/``y`` are provided (Create Model in-memory pass-through), skip
    the parquet reload and reuse the training matrix already in RAM.
    """
    config_path = os.path.join(package_dir, "config.json")
    if not os.path.isfile(config_path):
        config_path = os.path.join(package_dir, "training_config.json")
    config_raw = _read_json(config_path)
    if not config_raw and isinstance(doc.get("config"), dict):
        # load_model_detail nests config under config.data sometimes
        nested = doc["config"]
        config_raw = nested.get("data") if isinstance(nested.get("data"), dict) else nested

    training_cfg = normalize_training_config(config_raw)
    paths = {
        "config_json": os.path.join(package_dir, "config.json"),
        "package_dir": package_dir,
    }
    selected = _selected_feature_names(data_dir, safe_model_name(model_name), paths)
    if selected:
        training_cfg.features = list(selected)

    passthrough = X is not None and y is not None
    if passthrough:
        assert X is not None and y is not None
        features = list(training_cfg.features) or list(X.columns)
        use_features = [f for f in features if f in X.columns]
        if not use_features:
            raise ValueError("In-memory matrix has none of the selected features")
        X_mat = X[use_features]
        y_mat = y
        metadata: dict[str, Any] = {
            "dataset_load": {"backend": "in_memory_passthrough", "rows_returned": int(len(X_mat))},
        }
    else:
        X_mat, y_mat, features, metadata, _expected, _ctx = load_training_xy(
            data_dir, training_cfg
        )
        use_features = [f for f in features if f in X_mat.columns]
        X_mat = X_mat[use_features]

    n_rows = len(X_mat)
    start, stop = resolve_holdout_slice_with_fallback(doc, n_rows, training_cfg)

    X_ho = X_mat.iloc[start:stop].copy()
    y_ho = y_mat.iloc[start:stop].copy()
    if holdout_max_rows is not None and len(X_ho) > int(holdout_max_rows):
        X_ho = X_ho.iloc[: int(holdout_max_rows)]
        y_ho = y_ho.iloc[: int(holdout_max_rows)]

    load_meta = {
        "dataset": training_cfg.dataset,
        "target": training_cfg.target,
        "holdout_start": int(start),
        "holdout_stop": int(stop),
        "holdout_rows": int(len(X_ho)),
        "feature_count": len(use_features),
        "dataset_load": dict((metadata or {}).get("dataset_load") or {}),
        "prediction_type": _prediction_kind(config_raw),
        "matrix_source": "in_memory" if passthrough else "disk",
    }
    return X_ho, y_ho, use_features, load_meta


def run_compute(
    *,
    data_dir: str,
    model_name: str,
    package_dir: str | None = None,
    holdout_max_rows: int | None = 50_000,
    permutation_n_repeats: int = 5,
    shap_sample_size: int = 400,
    progress: ProgressCb | None = None,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
) -> ImportanceStudioResult:
    """Full compute pipeline → artifacts under the model package."""

    def _tick(stage: str, **extra: Any) -> None:
        if progress:
            progress({"stage": stage, **extra})

    safe = safe_model_name(model_name)
    pkg = package_dir or model_package_dir(data_dir, safe)
    if not os.path.isdir(pkg):
        return ImportanceStudioResult(
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

    _tick("load_model")
    t_model = time.perf_counter()
    config_raw = _read_json(os.path.join(pkg, "config.json"))
    algorithm = str(config_raw.get("algorithm") or "xgboost")
    try:
        model_path = resolve_production_model_path(pkg, algorithm=algorithm)
        model = load_prediction_model(model_path, algorithm)
    except Exception as exc:
        return ImportanceStudioResult(
            ok=False,
            model_name=safe,
            package_dir=pkg,
            artifacts_dir="",
            error=f"Failed to load model: {exc}",
        )
    timings["load_model_sec"] = round(time.perf_counter() - t_model, 3)

    _tick("load_holdout")
    t_ho = time.perf_counter()
    try:
        X_ho, y_ho, features, load_meta = _load_holdout_xy(
            data_dir=data_dir,
            package_dir=pkg,
            model_name=safe,
            doc=doc,
            holdout_max_rows=holdout_max_rows,
            X=X,
            y=y,
        )
    except Exception as exc:
        return ImportanceStudioResult(
            ok=False,
            model_name=safe,
            package_dir=pkg,
            artifacts_dir="",
            error=f"Failed to load holdout: {exc}",
        )
    timings["load_holdout_sec"] = round(time.perf_counter() - t_ho, 3)

    kind = str(load_meta.get("prediction_type") or "regression")
    if kind not in ("binary", "regression"):
        kind = "regression"

    _tick("native")
    t_n = time.perf_counter()
    native = compute_native_xgb_importance(model, features)
    timings["native_sec"] = round(time.perf_counter() - t_n, 3)

    _tick("permutation")
    t_p = time.perf_counter()
    permutation = compute_permutation_importance(
        model,
        X_ho,
        y_ho,
        features,
        n_repeats=permutation_n_repeats,
        kind=kind,  # type: ignore[arg-type]
        progress=progress,
    )
    timings["permutation_sec"] = round(time.perf_counter() - t_p, 3)

    _tick("shap")
    t_s = time.perf_counter()
    shap_rows = compute_shap_holdout(
        model, X_ho, features, sample_size=shap_sample_size
    )
    timings["shap_sec"] = round(time.perf_counter() - t_s, 3)

    _tick("comparison")
    comparison = build_comparison_rows(
        features=features,
        native=native,
        permutation=permutation,
        shap=shap_rows,
    )

    dataset_load = load_meta.get("dataset_load") or {}
    run_meta = {
        "model_name": safe,
        "package_dir": pkg,
        "dataset": load_meta.get("dataset"),
        "target": load_meta.get("target"),
        "prediction_type": kind,
        "holdout_row_count": load_meta.get("holdout_rows"),
        "holdout_start": load_meta.get("holdout_start"),
        "holdout_stop": load_meta.get("holdout_stop"),
        "feature_count": load_meta.get("feature_count"),
        "dataset_engine_backend": dataset_load.get("backend"),
        "dataset_load": dataset_load,
        "permutation_n_repeats": permutation_n_repeats,
        "shap_sample_size": shap_sample_size,
        "holdout_max_rows": holdout_max_rows,
        "model_version": config_raw.get("model_version") or config_raw.get("modelVersion"),
        "package_version": config_raw.get("model_version") or config_raw.get("version"),
        "timings_sec": timings,
        "wall_time_sec": round(time.perf_counter() - t0, 3),
        "studio_version": "4.1.0",
    }

    _tick("write_artifacts")
    artifacts_dir = write_studio_artifacts(
        pkg,
        native=native,
        permutation=permutation,
        shap=shap_rows,
        comparison=comparison,
        run_meta=run_meta,
    )

    return ImportanceStudioResult(
        ok=True,
        model_name=safe,
        package_dir=pkg,
        artifacts_dir=artifacts_dir,
        native=native,
        permutation=permutation,
        shap=shap_rows,
        comparison=comparison,
        meta=run_meta,
    )
