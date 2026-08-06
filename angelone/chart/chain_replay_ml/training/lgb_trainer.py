"""LightGBM regression training."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .xgb_trainer import TrainingCancelled, select_feature_columns


def _lgb_params(parameters: dict[str, Any], *, device_params: dict[str, Any]) -> dict[str, Any]:
    max_depth = int(parameters.get("max_depth", 6))
    pred = str(parameters.get("prediction_type") or "regression").strip().lower()
    n_classes = int(parameters.get("n_classes") or 0)

    if pred == "binary" or (pred in ("classification", "multiclass") and n_classes <= 2):
        objective = "binary"
        metric = "binary_logloss"
    elif pred in ("classification", "multiclass"):
        objective = "multiclass"
        metric = "multi_logloss"
    else:
        objective = "regression"
        metric = "rmse"

    params: dict[str, Any] = {
        "objective": objective,
        "metric": metric,
        "learning_rate": float(parameters.get("learning_rate", 0.05)),
        "max_depth": max_depth,
        "num_leaves": int(parameters.get("num_leaves", min(127, max(8, 2 ** max_depth)))),
        "subsample": float(parameters.get("subsample", 0.8)),
        "colsample_bytree": float(parameters.get("colsample_bytree", 0.8)),
        "min_child_samples": int(parameters.get("min_child_weight", 1)),
        "reg_alpha": float(parameters.get("reg_alpha", 0)),
        "reg_lambda": float(parameters.get("reg_lambda", 1)),
        "seed": int(parameters.get("random_seed", 42)),
        "verbosity": -1,
    }
    if objective == "multiclass":
        params["num_class"] = max(2, n_classes)
    params.update(dict(device_params or {}))
    return params


def _booster_wrapper(booster: Any, features: list[str], *, prediction_type: str = "regression") -> Any:
    class _Wrapper:
        def __init__(self, bst: Any, feat_names: list[str], pred_type: str) -> None:
            self._bst = bst
            self._prediction_type = pred_type
            self.feature_names_in_ = np.array(feat_names)
            gain = bst.feature_importance(importance_type="gain")
            self.feature_importances_ = np.asarray(gain, dtype=float)

        def save_model(self, path: str) -> None:
            self._bst.save_model(path)

        def predict(self, X_df: pd.DataFrame) -> np.ndarray:
            cols = list(self.feature_names_in_)
            raw = np.asarray(self._bst.predict(X_df.loc[:, cols]))
            # Binary objective already returns P(positive). Multiclass returns (n, k).
            if self._prediction_type in ("classification", "multiclass") and raw.ndim == 2:
                return raw  # caller / evaluator may use argmax; keep probabilities matrix
            return raw

    return _Wrapper(booster, features, prediction_type)


def train_lgb_regressor(
    *,
    train_X: pd.DataFrame,
    train_y: pd.Series,
    val_X: pd.DataFrame,
    val_y: pd.Series,
    features: list[str],
    parameters: dict[str, Any],
    cancel_check: Callable[[], bool] | None = None,
    on_iteration: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise RuntimeError("LightGBM is not installed. Run: pip install lightgbm") from exc

    from .model_device import LightGBMGpuUnavailableError, resolve_training_device

    # Raises LightGBMGpuUnavailableError when GPU requested but unsupported.
    device_plan = resolve_training_device("lightgbm", parameters)

    train_X, use_features = select_feature_columns(train_X, features)
    val_X, _ = select_feature_columns(val_X, features)

    pred_type = str(parameters.get("prediction_type") or "regression").strip().lower()
    train_y_fit = train_y
    val_y_fit = val_y
    params_in = dict(parameters)
    if pred_type in ("binary", "classification", "multiclass"):
        from .label_prep import adapt_target_for_prediction_type

        target_name = str(parameters.get("target") or "").strip() or None
        encoding = parameters.get("label_encoding") if isinstance(parameters.get("label_encoding"), dict) else None
        train_y_fit, adapt_meta = adapt_target_for_prediction_type(
            pd.to_numeric(train_y, errors="coerce"),
            prediction_type=pred_type,
            target=target_name,
            label_encoding=encoding,
        )
        val_y_fit, _ = adapt_target_for_prediction_type(
            pd.to_numeric(val_y, errors="coerce"),
            prediction_type=pred_type,
            target=target_name,
            label_encoding=encoding,
        )
        fit_pred = str(adapt_meta.get("adapted_prediction_type") or pred_type).strip().lower()
        n_classes = int(adapt_meta.get("n_classes") or 0)
        params_in["n_classes"] = n_classes
        params_in["prediction_type"] = "binary" if fit_pred == "binary" or n_classes <= 2 else "multiclass"
        pred_type = str(params_in["prediction_type"])
        train_y_fit = train_y_fit.fillna(0).astype(int)
        val_y_fit = val_y_fit.fillna(0).astype(int)

    n_estimators = int(parameters.get("n_estimators", 1000))
    early_stop = int(parameters.get("early_stopping_rounds", 100))

    def _cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    train_data = lgb.Dataset(train_X, label=train_y_fit, feature_name=use_features)
    val_data = lgb.Dataset(val_X, label=val_y_fit, reference=train_data, feature_name=use_features)

    eval_result: dict[str, dict[str, list[float]]] = {}
    train_start = time.monotonic()
    fit_params = _lgb_params(params_in, device_params=device_plan.library_params)
    metric_name = str(fit_params.get("metric") or "rmse")

    class _ProgressCallback:
        def __init__(self) -> None:
            self.last_emit = 0.0

        def __call__(self, env) -> None:  # type: ignore[no-untyped-def]
            if _cancelled():
                raise TrainingCancelled("Training cancelled")
            epoch = env.iteration
            now = time.monotonic()
            if on_iteration and (now - self.last_emit >= 0.35 or epoch >= n_estimators - 1):
                self.last_emit = now
                val_metric = None
                if env.evaluation_result_list:
                    for item in env.evaluation_result_list:
                        if item[0] == "validation":
                            val_metric = float(item[2])
                            break
                on_iteration({
                    "current_tree": epoch + 1,
                    "trees_total": n_estimators,
                    "validation_rmse": val_metric,
                    "train_progress_pct": round((epoch + 1) / n_estimators * 100, 1),
                })

    try:
        booster = lgb.train(
            fit_params,
            train_data,
            num_boost_round=n_estimators,
            valid_sets=[train_data, val_data],
            valid_names=["train", "validation"],
            callbacks=[
                lgb.early_stopping(stopping_rounds=early_stop, verbose=False),
                lgb.record_evaluation(eval_result),
                _ProgressCallback(),
            ],
        )
    except Exception as exc:
        if device_plan.use_gpu:
            raise LightGBMGpuUnavailableError(
                "LightGBM GPU training failed after a successful GPU probe.\n"
                f"Params: {fit_params}\n"
                f"Error: {exc.__class__.__name__}: {exc}\n"
                "Refusing to silently fall back to CPU. Fix the GPU build or set "
                "lgb_device=cpu to train on CPU intentionally."
            ) from exc
        raise

    used_gpu = bool(device_plan.use_gpu)
    fallback_reason = None if used_gpu else device_plan.fallback_reason

    best_iter = int(getattr(booster, "best_iteration", n_estimators - 1)) + 1
    train_curve = (eval_result.get("train") or {}).get(metric_name) or (eval_result.get("train") or {}).get("rmse") or []
    val_curve = (eval_result.get("validation") or {}).get(metric_name) or (eval_result.get("validation") or {}).get("rmse") or []
    best_idx0 = max(0, best_iter - 1)
    training_time_sec = round(time.monotonic() - train_start, 2)

    def _curve() -> list[dict[str, Any]]:
        n = max(len(train_curve), len(val_curve))
        rows = []
        for i in range(n):
            rows.append({
                "iteration": i + 1,
                "train_rmse": train_curve[i] if i < len(train_curve) else None,
                "validation_rmse": val_curve[i] if i < len(val_curve) else None,
            })
        return rows

    from .algorithm_runtime import (
        build_algorithm_runtime,
        detect_gpu_name,
        merge_runtime_into_training_meta,
    )

    algo_params = {
        "objective": fit_params.get("objective"),
        "metric": fit_params.get("metric"),
        "learning_rate": fit_params.get("learning_rate"),
        "max_depth": fit_params.get("max_depth"),
        "num_leaves": fit_params.get("num_leaves"),
        "subsample": fit_params.get("subsample"),
        "colsample_bytree": fit_params.get("colsample_bytree"),
        "min_child_samples": fit_params.get("min_child_samples"),
        "reg_alpha": fit_params.get("reg_alpha"),
        "reg_lambda": fit_params.get("reg_lambda"),
        "n_estimators": n_estimators,
        "early_stopping_rounds": early_stop,
        "device": fit_params.get("device") or "cpu",
        "gpu_use_dp": fit_params.get("gpu_use_dp"),
        "prediction_type": pred_type,
    }
    if "num_class" in fit_params:
        algo_params["num_class"] = fit_params["num_class"]
    if "num_threads" in fit_params:
        algo_params["num_threads"] = fit_params["num_threads"]
    if used_gpu:
        algo_params["gpu_params_passed"] = {
            k: fit_params[k] for k in ("device", "gpu_use_dp") if k in fit_params
        }
    runtime = build_algorithm_runtime(
        algorithm="lightgbm",
        implementation="LightGBM Booster",
        device="cuda" if used_gpu else "cpu",
        algorithm_parameters=algo_params,
        gpu_name=detect_gpu_name() if used_gpu else None,
        fallback_reason=fallback_reason if not used_gpu else None,
        training_time_sec=training_time_sec,
    )
    training_meta = merge_runtime_into_training_meta(
        {
            "best_iteration": best_iter,
            "early_stopping_rounds": early_stop,
            "best_validation_rmse": round(min(val_curve), 6) if val_curve else None,
            "train_rmse": round(train_curve[best_idx0], 6) if train_curve and best_idx0 < len(train_curve) else None,
            "device_plan": {
                "requested": device_plan.requested,
                "prefer_gpu": device_plan.prefer_gpu,
                "library_params": dict(device_plan.library_params),
            },
        },
        runtime,
    )
    if fallback_reason:
        warnings.warn(f"WARNING: {fallback_reason}", UserWarning, stacklevel=2)

    wrapper = _booster_wrapper(booster, use_features, prediction_type=pred_type)

    return {
        "booster": booster,
        "model": wrapper,
        "features": use_features,
        "trees_trained": best_iter,
        "early_stopped": best_iter < n_estimators,
        "training_meta": training_meta,
        "validation_loss_curve": _curve(),
        "training_time_sec": training_time_sec,
        "algorithm": "lightgbm",
    }
