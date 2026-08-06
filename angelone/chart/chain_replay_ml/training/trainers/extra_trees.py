"""Extra Trees (Extremely Randomized Trees) — sklearn CPU; cuML GPU when available."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from ..algorithm_runtime import (
    build_algorithm_runtime,
    detect_gpu_name,
    merge_runtime_into_training_meta,
)
from .base import (
    ALGORITHM_EXTRA_TREES,
    ModelTrainer,
    register_trainer,
)
from .random_forest import _RfModelWrapper, _resolve_task


def _resolve_et_backend(
    parameters: dict[str, Any],
    *,
    task: str = "regression",
) -> tuple[str, Any, str | None]:
    from ..model_device import resolve_training_device

    plan = resolve_training_device("extra_trees", parameters)
    classification = task in ("binary", "multiclass")
    if plan.use_gpu:
        try:
            if classification:
                from cuml.ensemble import ExtraTreesClassifier as CuET  # type: ignore
            else:
                from cuml.ensemble import ExtraTreesRegressor as CuET  # type: ignore
            return "cuml", CuET, None
        except Exception as exc:
            plan_fallback = (
                f"cuML ExtraTrees {'Classifier' if classification else 'Regressor'} unavailable "
                f"({exc.__class__.__name__}); using scikit-learn CPU"
            )
    else:
        plan_fallback = plan.fallback_reason or "CPU requested"

    if classification:
        from sklearn.ensemble import ExtraTreesClassifier as SkET
    else:
        from sklearn.ensemble import ExtraTreesRegressor as SkET

    return "sklearn", SkET, plan_fallback


@register_trainer
class ExtraTreesTrainer(ModelTrainer):
    algorithm_id = ALGORITHM_EXTRA_TREES
    algorithm_label = "Extra Trees"

    def train(
        self,
        *,
        train_X: pd.DataFrame,
        train_y: pd.Series,
        val_X: pd.DataFrame,
        val_y: pd.Series,
        features: list[str],
        parameters: dict[str, Any],
        cancel_check: Callable[[], bool] | None = None,
        on_iteration: Callable[[dict[str, Any]], None] | None = None,
        prediction_type: str = "regression",
    ) -> dict[str, Any]:
        params = dict(parameters)
        params["prediction_type"] = prediction_type
        return train_extra_trees_regressor(
            train_X=train_X,
            train_y=train_y,
            val_X=val_X,
            val_y=val_y,
            features=features,
            parameters=params,
            cancel_check=cancel_check,
            on_iteration=on_iteration,
        )


def train_extra_trees_regressor(
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
    from ..xgb_trainer import TrainingCancelled, select_feature_columns

    if cancel_check and cancel_check():
        raise TrainingCancelled("Training cancelled")

    train_X, use_features = select_feature_columns(train_X, features)
    val_X, _ = select_feature_columns(val_X, features)

    task, classes = _resolve_task(parameters, train_y, val_y)
    train_y_fit = train_y
    val_y_fit = val_y
    if task == "binary":
        if set(classes).issubset({0, 1}):
            train_y_fit = pd.to_numeric(train_y, errors="coerce").fillna(0).astype(int)
            val_y_fit = pd.to_numeric(val_y, errors="coerce").fillna(0).astype(int)
        else:
            pos = max(classes) if classes else 1
            train_y_fit = (pd.to_numeric(train_y, errors="coerce") == float(pos)).astype(int)
            val_y_fit = (pd.to_numeric(val_y, errors="coerce") == float(pos)).astype(int)
    elif task == "multiclass":
        class_map = {c: i for i, c in enumerate(classes)}
        train_y_fit = pd.to_numeric(train_y, errors="coerce").map(
            lambda v: class_map.get(int(round(float(v))), 0)
        ).astype(int)
        val_y_fit = pd.to_numeric(val_y, errors="coerce").map(
            lambda v: class_map.get(int(round(float(v))), 0)
        ).astype(int)

    n_estimators = int(parameters.get("n_estimators", 300))
    max_depth = int(parameters.get("max_depth", 12))
    min_samples_leaf = max(1, int(parameters.get("min_child_weight", 1)))
    max_features = float(parameters.get("colsample_bytree", 0.8))
    if max_features <= 0 or max_features > 1:
        max_features = "sqrt"  # type: ignore[assignment]
    bootstrap = bool(parameters.get("bootstrap", False))  # Extra Trees default: False
    random_seed = int(parameters.get("random_seed", 42))
    n_jobs = int(parameters.get("n_jobs", -1))

    backend, Est, fallback_reason = _resolve_et_backend(parameters, task=task)
    train_start = time.monotonic()
    gpu_name = detect_gpu_name() if backend == "cuml" else None
    classification = task in ("binary", "multiclass")

    if on_iteration:
        on_iteration({
            "current_tree": 0,
            "trees_total": n_estimators,
            "validation_rmse": None,
            "train_progress_pct": 0.0,
            "backend": backend,
        })

    def _make_sklearn_est() -> Any:
        from sklearn.ensemble import (
            ExtraTreesClassifier as SkClf,
            ExtraTreesRegressor as SkReg,
        )

        Sk = SkClf if classification else SkReg
        return Sk(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else None,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_seed,
            n_jobs=n_jobs,
        )

    if backend == "cuml":
        est = Est(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else 16,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features if isinstance(max_features, float) else 1.0,
            random_state=random_seed,
            n_streams=max(1, min(8, os.cpu_count() or 1)),
        )
        try:
            import cudf  # type: ignore

            X_cu = cudf.DataFrame(train_X)
            y_dtype = int if classification else float
            y_cu = cudf.Series(np.asarray(train_y_fit, dtype=y_dtype))
            est.fit(X_cu, y_cu)
        except Exception as exc:
            backend = "sklearn"
            fallback_reason = f"cuML fit failed ({exc.__class__.__name__}); fell back to scikit-learn"
            gpu_name = None
            est = _make_sklearn_est()
            est.fit(train_X, train_y_fit)
    else:
        est = Est(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth > 0 else None,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=random_seed,
            n_jobs=n_jobs,
        )
        est.fit(train_X, train_y_fit)

    if cancel_check and cancel_check():
        raise TrainingCancelled("Training cancelled")

    wrapper = _RfModelWrapper(est, use_features, backend=backend, task=task)
    val_pred = wrapper.predict(val_X)
    train_pred = wrapper.predict(train_X)
    val_y_cmp = np.asarray(val_y_fit, dtype=float)
    train_y_cmp = np.asarray(train_y_fit, dtype=float)
    if task == "multiclass" and getattr(val_pred, "ndim", 1) == 2:
        val_rmse = float(np.mean(np.argmax(val_pred, axis=1) != val_y_cmp.astype(int)))
        train_rmse = float(np.mean(np.argmax(train_pred, axis=1) != train_y_cmp.astype(int)))
    else:
        val_rmse = float(np.sqrt(np.mean((val_y_cmp - np.asarray(val_pred, dtype=float).reshape(-1)[: len(val_y_cmp)]) ** 2)))
        train_rmse = float(np.sqrt(np.mean((train_y_cmp - np.asarray(train_pred, dtype=float).reshape(-1)[: len(train_y_cmp)]) ** 2)))
    training_time_sec = round(time.monotonic() - train_start, 2)

    if on_iteration:
        on_iteration({
            "current_tree": n_estimators,
            "trees_total": n_estimators,
            "validation_rmse": val_rmse,
            "train_progress_pct": 100.0,
            "backend": backend,
        })

    algo_params: dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth if max_depth > 0 else None,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "bootstrap": bootstrap,
        "random_state": random_seed,
        "prediction_type": task if task != "regression" else "regression",
    }
    if backend == "sklearn":
        algo_params["n_jobs"] = n_jobs

    kind = "Classifier" if classification else "Regressor"
    implementation = (
        f"cuML ExtraTrees{kind}"
        if backend == "cuml"
        else f"scikit-learn ExtraTrees{kind}"
    )
    runtime = build_algorithm_runtime(
        algorithm=ALGORITHM_EXTRA_TREES,
        implementation=implementation,
        device="cuda" if backend == "cuml" else "cpu",
        algorithm_parameters=algo_params,
        gpu_name=gpu_name,
        fallback_reason=fallback_reason,
        training_time_sec=training_time_sec,
    )
    training_meta = merge_runtime_into_training_meta(
        {
            "best_iteration": n_estimators,
            "early_stopping_rounds": 0,
            "best_validation_rmse": round(val_rmse, 6),
            "train_rmse": round(train_rmse, 6),
            "et_backend": backend,
        },
        runtime,
    )

    return {
        "booster": est,
        "model": wrapper,
        "features": use_features,
        "trees_trained": n_estimators,
        "early_stopped": False,
        "training_meta": training_meta,
        "validation_loss_curve": [
            {"iteration": n_estimators, "train_rmse": train_rmse, "validation_rmse": val_rmse},
        ],
        "training_time_sec": training_time_sec,
        "algorithm": ALGORITHM_EXTRA_TREES,
        "prediction_type": task if task != "regression" else "regression",
    }
