"""CatBoost regression and classification training."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from .xgb_trainer import TrainingCancelled, select_feature_columns


def _metric_scalar(value: Any) -> float | None:
    """Coerce CatBoost callback metric (scalar or list) to float."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[-1]
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _resolve_task(parameters: dict[str, Any], train_y: pd.Series, val_y: pd.Series) -> tuple[str, int]:
    pred = str(parameters.get("prediction_type") or "regression").strip().lower()
    if pred not in ("binary", "classification", "multiclass"):
        return "regression", 0
    y_all = pd.concat([train_y, val_y], ignore_index=True)
    classes = sorted({int(round(float(v))) for v in y_all.dropna().tolist()})
    n_classes = len(classes)
    if pred == "binary" or n_classes <= 2:
        return "binary", max(2, n_classes)
    return "multiclass", max(2, n_classes)


def _catboost_params(
    parameters: dict[str, Any],
    *,
    task_type: str,
    task: str,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "iterations": int(parameters.get("n_estimators", 1000)),
        "learning_rate": float(parameters.get("learning_rate", 0.05)),
        "depth": int(parameters.get("max_depth", 6)),
        "l2_leaf_reg": float(parameters.get("reg_lambda", 1)),
        "subsample": float(parameters.get("subsample", 0.8)),
        "colsample_bylevel": float(parameters.get("colsample_bytree", 0.8)),
        "min_data_in_leaf": int(parameters.get("min_child_weight", 1)),
        "random_seed": int(parameters.get("random_seed", 42)),
        "early_stopping_rounds": int(parameters.get("early_stopping_rounds", 100)),
        "verbose": False,
        "allow_writing_files": False,
        "task_type": task_type,
    }
    if task == "binary":
        params["loss_function"] = "Logloss"
        params["eval_metric"] = "Logloss"
    elif task == "multiclass":
        params["loss_function"] = "MultiClass"
        params["eval_metric"] = "MultiClass"
    else:
        params["loss_function"] = "RMSE"
        params["eval_metric"] = "RMSE"
    return params


def _model_wrapper(model: Any, features: list[str], *, task: str = "regression") -> Any:
    class _Wrapper:
        def __init__(self, mdl: Any, feat_names: list[str], task_name: str) -> None:
            self._mdl = mdl
            self._task = task_name
            self.feature_names_in_ = np.array(feat_names)
            scores = mdl.get_feature_importance()
            self.feature_importances_ = np.asarray(scores, dtype=float)

        def save_model(self, path: str) -> None:
            self._mdl.save_model(path)

        def predict(self, X_df: pd.DataFrame) -> np.ndarray:
            cols = list(self.feature_names_in_)
            X = X_df.loc[:, cols]
            if self._task == "binary" and hasattr(self._mdl, "predict_proba"):
                proba = np.asarray(self._mdl.predict_proba(X))
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    return proba[:, 1]
                return proba.reshape(-1)
            if self._task == "multiclass" and hasattr(self._mdl, "predict_proba"):
                return np.asarray(self._mdl.predict_proba(X))
            return np.asarray(self._mdl.predict(X))

    return _Wrapper(model, features, task)


def train_catboost_regressor(
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
        from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    except ImportError as exc:
        raise RuntimeError("CatBoost is not installed. Run: pip install catboost") from exc

    from .model_device import resolve_training_device

    device_plan = resolve_training_device("catboost", parameters)
    task_type = str(
        device_plan.library_params.get("task_type")
        or ("GPU" if device_plan.use_gpu else "CPU")
    )

    train_X, use_features = select_feature_columns(train_X, features)
    val_X, _ = select_feature_columns(val_X, features)

    task, _n_classes = _resolve_task(parameters, train_y, val_y)
    train_y_fit = train_y
    val_y_fit = val_y
    if task in ("binary", "multiclass"):
        from .label_prep import adapt_target_for_prediction_type

        pred_type = str(parameters.get("prediction_type") or "regression").strip().lower()
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
        if fit_pred == "binary" or n_classes <= 2:
            task = "binary"
        else:
            task = "multiclass"
        train_y_fit = train_y_fit.fillna(0).astype(int)
        val_y_fit = val_y_fit.fillna(0).astype(int)

    n_estimators = int(parameters.get("n_estimators", 1000))

    def _cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    params = _catboost_params(parameters, task_type=task_type, task=task)
    early_stop = int(params.pop("early_stopping_rounds", 100))
    eval_metric_key = str(params.get("eval_metric") or "RMSE")

    train_pool = Pool(train_X, label=train_y_fit, feature_names=use_features)
    val_pool = Pool(val_X, label=val_y_fit, feature_names=use_features)

    class _ProgressCallback:
        def __init__(self) -> None:
            self.last_emit = 0.0

        def after_iteration(self, info) -> bool:  # type: ignore[no-untyped-def]
            if _cancelled():
                raise TrainingCancelled("Training cancelled")
            if on_iteration:
                now = time.monotonic()
                iteration = int(getattr(info, "iteration", 0))
                if now - self.last_emit >= 0.35 or iteration >= n_estimators - 1:
                    self.last_emit = now
                    metrics = getattr(info, "metrics", {}) or {}
                    val_metric = None
                    val_block = metrics.get("validation") or metrics.get("Validation") or {}
                    if isinstance(val_block, dict):
                        val_metric = _metric_scalar(
                            val_block.get(eval_metric_key)
                            or val_block.get("RMSE")
                            or val_block.get("Logloss")
                            or next(iter(val_block.values()), None)
                        )
                    on_iteration({
                        "current_tree": iteration + 1,
                        "trees_total": n_estimators,
                        "validation_rmse": val_metric,
                        "train_progress_pct": round((iteration + 1) / n_estimators * 100, 1),
                    })
            return True

    train_start = time.monotonic()
    want_gpu = str(params.get("task_type") or "CPU").upper() == "GPU"
    fallback_reason: str | None = None
    ModelCls = CatBoostClassifier if task in ("binary", "multiclass") else CatBoostRegressor
    try:
        model = ModelCls(**params, early_stopping_rounds=early_stop)
        model.fit(train_pool, eval_set=val_pool, callbacks=[_ProgressCallback()])
    except Exception as exc:
        if not want_gpu:
            raise
        fallback_reason = (
            f"CatBoost GPU failed ({exc.__class__.__name__}: {str(exc)[:160]}); "
            "falling back to CPU"
        )
        warnings.warn(f"WARNING: {fallback_reason}", UserWarning, stacklevel=2)
        params = dict(params)
        params["task_type"] = "CPU"
        model = ModelCls(**params, early_stopping_rounds=early_stop)
        model.fit(train_pool, eval_set=val_pool, callbacks=[_ProgressCallback()])
        want_gpu = False

    best_iter = int(model.get_best_iteration() or model.tree_count_ or n_estimators)
    evals = model.get_evals_result() or {}
    train_curve = list((evals.get("learn") or {}).get(eval_metric_key) or (evals.get("learn") or {}).get("RMSE") or [])
    val_curve = list((evals.get("validation") or {}).get(eval_metric_key) or (evals.get("validation") or {}).get("RMSE") or [])
    best_idx0 = max(0, best_iter - 1)
    training_time_sec = round(time.monotonic() - train_start, 2)

    def _curve() -> list[dict[str, Any]]:
        n = max(len(train_curve), len(val_curve))
        return [
            {
                "iteration": i + 1,
                "train_rmse": train_curve[i] if i < len(train_curve) else None,
                "validation_rmse": val_curve[i] if i < len(val_curve) else None,
            }
            for i in range(n)
        ]

    from .algorithm_runtime import (
        build_algorithm_runtime,
        detect_gpu_name,
        merge_runtime_into_training_meta,
    )

    used_gpu = str(params.get("task_type") or "CPU").upper() == "GPU"
    algo_params = {
        "loss_function": params.get("loss_function"),
        "eval_metric": params.get("eval_metric"),
        "iterations": params.get("iterations"),
        "learning_rate": params.get("learning_rate"),
        "depth": params.get("depth"),
        "l2_leaf_reg": params.get("l2_leaf_reg"),
        "subsample": params.get("subsample"),
        "colsample_bylevel": params.get("colsample_bylevel"),
        "min_data_in_leaf": params.get("min_data_in_leaf"),
        "task_type": params.get("task_type"),
        "early_stopping_rounds": early_stop,
        "random_seed": params.get("random_seed"),
        "prediction_type": task if task != "regression" else "regression",
        "gpu_params_passed": {"task_type": params.get("task_type")},
    }
    if not used_gpu and fallback_reason is None:
        fallback_reason = device_plan.fallback_reason or "CPU requested"
    impl = (
        "CatBoostClassifier" if task in ("binary", "multiclass") else "CatBoostRegressor"
    )
    runtime = build_algorithm_runtime(
        algorithm="catboost",
        implementation=impl,
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
            "train_rmse": (
                round(train_curve[best_idx0], 6)
                if train_curve and best_idx0 < len(train_curve)
                else None
            ),
            "device_plan": {
                "requested": device_plan.requested,
                "prefer_gpu": device_plan.prefer_gpu,
                "library_params": dict(device_plan.library_params),
            },
        },
        runtime,
    )
    wrapper = _model_wrapper(model, use_features, task=task)

    return {
        "booster": model,
        "model": wrapper,
        "features": use_features,
        "trees_trained": best_iter,
        "early_stopped": best_iter < n_estimators,
        "training_meta": training_meta,
        "validation_loss_curve": _curve(),
        "training_time_sec": training_time_sec,
        "algorithm": "catboost",
        "prediction_type": task if task != "regression" else "regression",
    }
