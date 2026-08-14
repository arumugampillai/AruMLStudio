"""Shared XGBoost regression training for single-split and walk-forward flows."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb


class TrainingCancelled(Exception):
    pass


def _resolve_xgb_runtime(parameters: dict[str, Any]) -> tuple[dict[str, Any], bool, Any]:
    """Resolve XGBoost device via the shared GPU factory."""
    from .model_device import resolve_training_device

    plan = resolve_training_device("xgboost", parameters)
    return dict(plan.library_params), bool(plan.use_gpu), plan


def select_feature_columns(X: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Subset X to requested features; order matches `features` list."""
    use = [f for f in features if f in X.columns]
    if len(use) != len(features):
        missing = [f for f in features if f not in X.columns]
        raise ValueError(
            f"Feature matrix missing {len(missing)} requested column(s): "
            f"{missing[:5]}{'…' if len(missing) > 5 else ''}"
        )
    return X.loc[:, use], use


def booster_wrapper(bst: xgb.Booster, features: list[str]) -> Any:
    class _BoosterWrapper:
        def __init__(self, booster: xgb.Booster, feat_names: list[str]) -> None:
            self._bst = booster
            self.feature_names_in_ = np.array(feat_names)
            scores = booster.get_score(importance_type="gain")
            self.feature_importances_ = np.array([float(scores.get(f, 0.0)) for f in feat_names])

        def save_model(self, path: str) -> None:
            self._bst.save_model(path)

        def predict(self, X_df: pd.DataFrame) -> np.ndarray:
            from .feature_matrix import sanitize_training_features

            cols = list(self.feature_names_in_)
            dm = xgb.DMatrix(sanitize_training_features(X_df.loc[:, cols]), feature_names=cols)
            return self._bst.predict(dm)

    return _BoosterWrapper(bst, features)


def feature_importance_df(model: Any, features: list[str]) -> pd.DataFrame:
    scores = model.feature_importances_
    total = float(scores.sum()) or 1.0
    rows = [
        {"feature": f, "importance": float(s), "importance_pct": round(float(s) / total * 100, 2)}
        for f, s in zip(features, scores)
    ]
    rows.sort(key=lambda r: r["importance"], reverse=True)
    return pd.DataFrame(rows)


def train_xgb_regressor(
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
    """Train one XGBoost regressor; returns booster, wrapper, meta, and loss curve."""
    train_X, use_features = select_feature_columns(train_X, features)
    val_X, _ = select_feature_columns(val_X, features)

    from .feature_matrix import sanitize_training_features

    train_X = sanitize_training_features(train_X)
    val_X = sanitize_training_features(val_X)
    train_y = pd.to_numeric(train_y, errors="coerce").astype("float32")
    val_y = pd.to_numeric(val_y, errors="coerce").astype("float32")

    from .label_prep import adapt_target_for_prediction_type

    pred_type = str(parameters.get("prediction_type") or "regression").strip().lower()
    target_name = str(parameters.get("target") or "").strip() or None
    encoding = parameters.get("label_encoding")
    if not isinstance(encoding, dict):
        encoding = None
    train_y, adapt_meta = adapt_target_for_prediction_type(
        train_y,
        prediction_type=pred_type,
        target=target_name,
        label_encoding=encoding,
    )
    val_y, _ = adapt_target_for_prediction_type(
        val_y,
        prediction_type=pred_type,
        target=target_name,
        label_encoding=encoding,
    )
    fit_pred = str(adapt_meta.get("adapted_prediction_type") or pred_type).strip().lower()
    n_classes = int(adapt_meta.get("n_classes") or 0)

    n_estimators = int(parameters.get("n_estimators", 1000))
    early_stop = int(parameters.get("early_stopping_rounds", 100))

    def _cancelled() -> bool:
        return bool(cancel_check and cancel_check())

    if fit_pred == "binary" or (fit_pred in ("classification", "multiclass") and n_classes <= 2):
        objective = "binary:logistic"
        eval_metric = "logloss"
    elif fit_pred in ("classification", "multiclass"):
        objective = "multi:softprob"
        eval_metric = "mlogloss"
    else:
        objective = "reg:squarederror"
        eval_metric = "rmse"

    xgb_params = {
        "objective": objective,
        "learning_rate": float(parameters.get("learning_rate", 0.05)),
        "max_depth": int(parameters.get("max_depth", 6)),
        "subsample": float(parameters.get("subsample", 0.8)),
        "colsample_bytree": float(parameters.get("colsample_bytree", 0.8)),
        "min_child_weight": float(parameters.get("min_child_weight", 1)),
        "reg_alpha": float(parameters.get("reg_alpha", 0)),
        "reg_lambda": float(parameters.get("reg_lambda", 1)),
        "gamma": float(parameters.get("gamma", 0)),
        "max_delta_step": float(parameters.get("max_delta_step", 0)),
        "seed": int(parameters.get("random_seed", 42)),
        "verbosity": 0,
        "eval_metric": eval_metric,
    }
    if objective == "multi:softprob":
        xgb_params["num_class"] = max(2, n_classes or 3)
    runtime_params, runtime_gpu, device_plan = _resolve_xgb_runtime(parameters)
    xgb_params.update(runtime_params)

    dtrain = xgb.DMatrix(train_X, label=train_y, feature_names=use_features)
    dval = xgb.DMatrix(val_X, label=val_y, feature_names=use_features)

    class _ProgressState:
        last_emit = 0.0

    class _EvalCapture:
        train_rmse: list[float] = []
        val_rmse: list[float] = []

    _EvalCapture.train_rmse = []
    _EvalCapture.val_rmse = []

    def _sync_eval_curves(evals_log: dict[str, Any] | None) -> None:
        if not evals_log:
            return
        for ds_name, metrics in evals_log.items():
            if not isinstance(metrics, dict):
                continue
            raw_rmse = metrics.get("rmse")
            if raw_rmse is None:
                raw_rmse = metrics.get("logloss")
            if raw_rmse is None:
                continue
            series = [float(v) for v in raw_rmse] if isinstance(raw_rmse, list) else [float(raw_rmse)]
            if ds_name == "train":
                _EvalCapture.train_rmse = series
            elif ds_name == "validation":
                _EvalCapture.val_rmse = series

    def _build_loss_curve(max_points: int = 250) -> list[dict[str, Any]]:
        n = max(len(_EvalCapture.train_rmse), len(_EvalCapture.val_rmse))
        if n <= 0:
            return []
        rows = []
        for i in range(n):
            rows.append({
                "iteration": i + 1,
                "train_rmse": _EvalCapture.train_rmse[i] if i < len(_EvalCapture.train_rmse) else None,
                "validation_rmse": _EvalCapture.val_rmse[i] if i < len(_EvalCapture.val_rmse) else None,
            })
        if len(rows) <= max_points:
            return rows
        step = max(1, len(rows) // max_points)
        sampled = rows[::step]
        if sampled[-1]["iteration"] != rows[-1]["iteration"]:
            sampled.append(rows[-1])
        return sampled

    def _training_meta(best_idx0: int) -> dict[str, Any]:
        best_val = min(_EvalCapture.val_rmse) if _EvalCapture.val_rmse else None
        train_at_best = None
        if _EvalCapture.train_rmse and 0 <= best_idx0 < len(_EvalCapture.train_rmse):
            train_at_best = _EvalCapture.train_rmse[best_idx0]
        return {
            "best_iteration": best_idx0 + 1,
            "early_stopping_rounds": early_stop,
            "best_validation_rmse": round(best_val, 6) if best_val is not None else None,
            "train_rmse": round(train_at_best, 6) if train_at_best is not None else None,
        }

    train_start = time.monotonic()

    def _after_iteration(model, epoch, evals_log):  # type: ignore[no-untyped-def]
        if _cancelled():
            raise TrainingCancelled("Training cancelled")
        _sync_eval_curves(evals_log)
        now = time.monotonic()
        if now - _ProgressState.last_emit < 0.35 and epoch < n_estimators - 1:
            return False
        _ProgressState.last_emit = now
        frac = (epoch + 1) / n_estimators
        eval_rmse = _EvalCapture.val_rmse[-1] if _EvalCapture.val_rmse else None
        if on_iteration:
            on_iteration({
                "current_tree": epoch + 1,
                "trees_total": n_estimators,
                "validation_rmse": eval_rmse,
                "train_progress_pct": round(frac * 100, 1),
            })
        return False

    callbacks = []
    try:
        from xgboost.callback import TrainingCallback

        class _Cb(TrainingCallback):
            def after_iteration(self, model, epoch, evals_log):  # type: ignore[no-untyped-def]
                return _after_iteration(model, epoch, evals_log)

        callbacks = [_Cb()]
    except ImportError:
        pass

    gpu_fallback_reason: str | None = None
    try:
        booster = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, "train"), (dval, "validation")],
            early_stopping_rounds=early_stop,
            callbacks=callbacks or None,
            verbose_eval=False,
        )
    except xgb.core.XGBoostError as exc:
        # CUDA runtime unavailable/misconfigured — explicit CPU fallback with warning.
        if not runtime_gpu:
            raise
        import warnings

        gpu_fallback_reason = str(exc)
        warn = (
            f"WARNING: XGBoost GPU failed; falling back to CPU "
            f"({exc.__class__.__name__}: {str(exc)[:160]})"
        )
        warnings.warn(warn, UserWarning, stacklevel=2)
        cpu_params = dict(xgb_params)
        cpu_params.update({
            "tree_method": "hist",
            "device": "cpu",
            "predictor": "cpu_predictor",
        })
        booster = xgb.train(
            cpu_params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, "train"), (dval, "validation")],
            early_stopping_rounds=early_stop,
            callbacks=callbacks or None,
            verbose_eval=False,
        )

    from .model_device import verify_xgboost_booster_device

    executed_device = verify_xgboost_booster_device(booster)
    if runtime_gpu and not gpu_fallback_reason and not str(executed_device).startswith("cuda"):
        import warnings

        gpu_fallback_reason = (
            f"XGBoost reported device={executed_device} after train "
            "(requested cuda) — treating run as CPU"
        )
        warnings.warn(
            f"WARNING: {gpu_fallback_reason}",
            UserWarning,
            stacklevel=2,
        )

    best_idx0 = int(getattr(booster, "best_iteration", n_estimators - 1))
    best_iter = best_idx0 + 1
    training_time_sec = round(time.monotonic() - train_start, 2)
    training_meta = _training_meta(best_idx0)
    wrapper = booster_wrapper(booster, use_features)

    from .algorithm_runtime import (
        build_algorithm_runtime,
        detect_gpu_name,
        merge_runtime_into_training_meta,
    )

    used_gpu = bool(runtime_gpu) and not gpu_fallback_reason and str(executed_device).startswith("cuda")
    final_xgb = dict(xgb_params)
    if not used_gpu:
        final_xgb.update({
            "tree_method": "hist",
            "device": "cpu",
            "predictor": "cpu_predictor",
        })
    algo_params = {
        "objective": final_xgb.get("objective"),
        "tree_method": final_xgb.get("tree_method"),
        "device": final_xgb.get("device"),
        "executed_device": executed_device,
        "eta": final_xgb.get("learning_rate"),
        "learning_rate": final_xgb.get("learning_rate"),
        "max_depth": final_xgb.get("max_depth"),
        "subsample": final_xgb.get("subsample"),
        "colsample_bytree": final_xgb.get("colsample_bytree"),
        "min_child_weight": final_xgb.get("min_child_weight"),
        "reg_alpha": final_xgb.get("reg_alpha"),
        "reg_lambda": final_xgb.get("reg_lambda"),
        "gamma": final_xgb.get("gamma"),
        "n_estimators": n_estimators,
        "early_stopping_rounds": early_stop,
        "seed": final_xgb.get("seed"),
    }
    fallback = None
    if gpu_fallback_reason:
        fallback = f"XGBoost GPU unavailable; using CPU ({gpu_fallback_reason[:160]})"
    elif not runtime_gpu:
        fallback = device_plan.fallback_reason or "CPU requested"
    runtime = build_algorithm_runtime(
        algorithm="xgboost",
        implementation="XGBoost Booster",
        device="cuda" if used_gpu else "cpu",
        algorithm_parameters=algo_params,
        gpu_name=detect_gpu_name() if used_gpu else None,
        fallback_reason=fallback if not used_gpu else None,
        training_time_sec=training_time_sec,
    )
    training_meta = merge_runtime_into_training_meta(training_meta, runtime)
    training_meta["device_plan"] = {
        "requested": device_plan.requested,
        "prefer_gpu": device_plan.prefer_gpu,
        "executed_device": executed_device,
        "library_params": dict(device_plan.library_params),
    }

    return {
        "booster": booster,
        "model": wrapper,
        "features": use_features,
        "trees_trained": best_iter,
        "early_stopped": best_iter < n_estimators,
        "training_meta": training_meta,
        "validation_loss_curve": _build_loss_curve(),
        "training_time_sec": training_time_sec,
        "xgb_runtime": "cpu_fallback" if gpu_fallback_reason else ("gpu" if used_gpu else "cpu"),
        "xgb_gpu_fallback_reason": gpu_fallback_reason,
        "executed_device": executed_device,
    }


def train_xgb_binary_classifier(**kwargs: Any) -> dict[str, Any]:
    """Binary Hit classifier — same pipeline as regressor with logistic objective."""
    parameters = dict(kwargs.get("parameters") or {})
    parameters["prediction_type"] = "binary"
    kwargs["parameters"] = parameters
    result = train_xgb_regressor(**kwargs)
    result["prediction_type"] = "binary"
    return result
