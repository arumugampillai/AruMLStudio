"""Layer 3 — package regression/classification predictors (no feature engineering)."""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np
import pandas as pd

from chain_replay_ml.training.model_runtime import load_prediction_model_cached

from .snapshot import FeatureSnapshot, PredictionResult, PredictionSnapshot
from .versions import feature_version, prediction_version


def _feature_value(shared: dict[str, Any], name: str) -> Any:
    """Map snapshot value to model input; None/NaN kept for tree models that handle missing."""
    if name not in shared:
        return None
    val = shared.get(name)
    if val is None:
        return np.nan
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return np.nan
    if isinstance(val, (int, float, bool, str)):
        return val
    try:
        f = float(val)
        return np.nan if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return val


def _empty_timing() -> dict[str, Any]:
    return {
        "mode": "sequential",
        "model_count": 0,
        "load_ms_total": 0.0,
        "load_ms_from_disk": 0.0,
        "predict_ms_total": 0.0,
        "dataframe_ms_total": 0.0,
        "shared_row_ms": 0.0,
        "dataframe_count": 0,
        "models_loaded_from_disk": 0,
        "models_cache_hit": 0,
        "dataframe_strategy": "shared_row_per_model_slice",
        "pre_first_model_ms": 0.0,
        "models": [],
    }


class RegressionModelEngine:
    """Each package member: select snapshot columns → predict (sequential)."""

    def __init__(self) -> None:
        self.last_timing: dict[str, Any] = _empty_timing()

    def predict_one(
        self,
        spec: dict[str, Any],
        snapshot: FeatureSnapshot | None,
        *,
        build_err: str | None = None,
        row_values: dict[str, Any] | None = None,
        timing: dict[str, Any] | None = None,
        model_timing: dict[str, Any] | None = None,
    ) -> PredictionResult:
        model_id = str(spec.get("model_name") or "")
        registry_row = spec.get("registry") or {}
        features = list(spec.get("features") or [])
        mae = spec.get("mae")
        rmse = spec.get("rmse")
        target = str(spec.get("target") or registry_row.get("target") or "")
        tier = str(spec.get("tier") or "regression")
        model_feature_version = str(spec.get("feature_version") or snapshot.feature_version if snapshot else feature_version())

        if build_err:
            return PredictionResult(
                prediction=None,
                model_id=model_id,
                mae=mae,
                rmse=rmse,
                prediction_time_ms=0.0,
                status="failed",
                feature_version=model_feature_version,
                target=target,
                error=build_err,
                tier=tier,
            )
        if snapshot is None:
            return PredictionResult(
                prediction=None,
                model_id=model_id,
                mae=mae,
                rmse=rmse,
                prediction_time_ms=0.0,
                status="failed",
                feature_version=model_feature_version,
                target=target,
                error="no_feature_snapshot",
                tier=tier,
            )
        if not features:
            return PredictionResult(
                prediction=None,
                model_id=model_id,
                mae=mae,
                rmse=rmse,
                prediction_time_ms=0.0,
                status="failed",
                feature_version=model_feature_version,
                target=target,
                error="no_features",
                tier=tier,
            )

        shared = dict(snapshot.features)
        missing_keys = [c for c in features if c not in shared]
        if missing_keys:
            return PredictionResult(
                prediction=None,
                model_id=model_id,
                mae=mae,
                rmse=rmse,
                prediction_time_ms=0.0,
                status="failed",
                feature_version=model_feature_version,
                target=target,
                error=f"missing_features:{','.join(missing_keys[:4])}",
                tier=tier,
            )

        t_df0 = time.perf_counter()
        if row_values is not None:
            x = pd.DataFrame([{c: row_values.get(c, _feature_value(shared, c)) for c in features}])
        else:
            x = pd.DataFrame([{c: _feature_value(shared, c) for c in features}])
        df_ms = round((time.perf_counter() - t_df0) * 1000.0, 3)
        if model_timing is not None:
            model_timing["dataframe_ms"] = df_ms
        if timing is not None:
            timing["dataframe_ms_total"] = round(float(timing.get("dataframe_ms_total", 0.0)) + df_ms, 3)
            timing["dataframe_count"] = int(timing.get("dataframe_count", 0)) + 1

        if not len(x.columns):
            return PredictionResult(
                prediction=None,
                model_id=model_id,
                mae=mae,
                rmse=rmse,
                prediction_time_ms=df_ms,
                status="failed",
                feature_version=model_feature_version,
                target=target,
                error="empty_feature_row",
                tier=tier,
            )

        model_path = spec.get("model_path")
        algorithm = spec.get("algorithm")
        if not model_path:
            return PredictionResult(
                prediction=None,
                model_id=model_id,
                mae=mae,
                rmse=rmse,
                prediction_time_ms=df_ms,
                status="failed",
                feature_version=model_feature_version,
                target=target,
                error="model_path_missing",
                tier=tier,
            )

        t0 = time.perf_counter()
        try:
            model, load_ms, from_disk = load_prediction_model_cached(model_path, algorithm)
            if model_timing is not None:
                model_timing["load_ms"] = load_ms
                model_timing["load_from_disk"] = from_disk
            if timing is not None:
                timing["load_ms_total"] = round(float(timing.get("load_ms_total", 0.0)) + load_ms, 3)
                if from_disk:
                    timing["load_ms_from_disk"] = round(float(timing.get("load_ms_from_disk", 0.0)) + load_ms, 3)
                    timing["models_loaded_from_disk"] = int(timing.get("models_loaded_from_disk", 0)) + 1
                else:
                    timing["models_cache_hit"] = int(timing.get("models_cache_hit", 0)) + 1

            t_pred0 = time.perf_counter()
            if tier == "classification" and hasattr(model, "predict_proba"):
                probabilities = np.asarray(model.predict_proba(x), dtype=float)
                pred = float(
                    probabilities[0, 1]
                    if probabilities.ndim == 2 and probabilities.shape[1] > 1
                    else probabilities.reshape(-1)[0]
                )
            else:
                # XGBoost binary:logistic and LightGBM Booster.predict return P(+).
                pred = float(model.predict(x)[0])
            predict_ms = round((time.perf_counter() - t_pred0) * 1000.0, 3)
            if model_timing is not None:
                model_timing["predict_ms"] = predict_ms
            if timing is not None:
                timing["predict_ms_total"] = round(float(timing.get("predict_ms_total", 0.0)) + predict_ms, 3)

            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            if not pd.notna(pred):
                return PredictionResult(
                    prediction=None,
                    model_id=model_id,
                    mae=mae,
                    rmse=rmse,
                    prediction_time_ms=elapsed_ms,
                    status="failed",
                    feature_version=model_feature_version,
                    target=target,
                    error="prediction_nan",
                    tier=tier,
                )
            return PredictionResult(
                prediction=(
                    round(pred, 6)
                    if tier == "classification"
                    else round(pred, 2)
                ),
                model_id=model_id,
                mae=mae,
                rmse=rmse,
                prediction_time_ms=elapsed_ms,
                status="ok",
                feature_version=snapshot.feature_version,
                target=target,
                tier=tier,
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            return PredictionResult(
                prediction=None,
                model_id=model_id,
                mae=mae,
                rmse=rmse,
                prediction_time_ms=elapsed_ms,
                status="failed",
                feature_version=model_feature_version,
                target=target,
                error=str(exc),
                tier=tier,
            )

    def predict_all(
        self,
        specs: list[dict[str, Any]],
        snapshot: FeatureSnapshot | None,
        *,
        build_err: str | None = None,
        on_model_progress: Callable[[int, int, str, PredictionResult], None] | None = None,
    ) -> PredictionSnapshot:
        results: dict[str, PredictionResult] = {}
        total = len(specs)
        timing = _empty_timing()
        timing["model_count"] = total

        row_values: dict[str, Any] | None = None
        if snapshot is not None and not build_err:
            union_features: set[str] = set()
            for spec in specs:
                union_features.update(spec.get("features") or [])
            if union_features:
                t_row0 = time.perf_counter()
                shared = dict(snapshot.features)
                row_values = {c: _feature_value(shared, c) for c in union_features}
                timing["shared_row_ms"] = round((time.perf_counter() - t_row0) * 1000.0, 3)

        t_pre_first = time.perf_counter()
        first = True
        for idx, spec in enumerate(specs, start=1):
            if first:
                timing["pre_first_model_ms"] = round((time.perf_counter() - t_pre_first) * 1000.0, 3)
                first = False
            model_id = str(spec.get("model_name") or "")
            model_timing: dict[str, Any] = {
                "model_id": model_id,
                "index": idx,
                "feature_count": len(spec.get("features") or []),
            }
            t_model0 = time.perf_counter()
            result = self.predict_one(
                spec,
                snapshot,
                build_err=build_err,
                row_values=row_values,
                timing=timing,
                model_timing=model_timing,
            )
            model_timing["total_ms"] = round((time.perf_counter() - t_model0) * 1000.0, 3)
            timing["models"].append(model_timing)
            results[result.model_id] = result
            if on_model_progress:
                on_model_progress(idx, total, result.model_id, result)

        self.last_timing = timing
        ts = float(snapshot.timestamp) if snapshot else 0.0
        token = str(snapshot.token) if snapshot else ""
        fv = snapshot.feature_version if snapshot else feature_version()
        return PredictionSnapshot.create(
            timestamp=ts,
            token=token,
            results=results,
            feature_version=fv,
            prediction_version=prediction_version(),
        )
