"""Random Forest trainer — GPU (cuML) when available, else scikit-learn CPU.

Backend is selected at train time; callers always get the same return shape as
boosting trainers (``model`` with predict/save_model). Explicitly logs which
implementation ran so GPU fallback is never ambiguous.
"""

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
    ALGORITHM_RANDOM_FOREST,
    ModelTrainer,
    register_trainer,
)


def _resolve_task(parameters: dict[str, Any], train_y: pd.Series, val_y: pd.Series) -> tuple[str, list[int]]:
    pred = str(parameters.get("prediction_type") or "regression").strip().lower()
    if pred not in ("binary", "classification", "multiclass"):
        return "regression", []
    y_all = pd.concat([train_y, val_y], ignore_index=True)
    classes = sorted({int(round(float(v))) for v in y_all.dropna().tolist()})
    if pred == "binary" or len(classes) <= 2:
        return "binary", classes
    return "multiclass", classes


def _resolve_rf_backend(
    parameters: dict[str, Any],
    *,
    task: str = "regression",
) -> tuple[str, Any, str | None]:
    """Return (backend_name, estimator_class, fallback_reason_if_cpu)."""
    from ..model_device import resolve_training_device

    plan = resolve_training_device("random_forest", parameters)
    classification = task in ("binary", "multiclass")
    if plan.use_gpu:
        try:
            if classification:
                from cuml.ensemble import RandomForestClassifier as CuRF  # type: ignore
            else:
                from cuml.ensemble import RandomForestRegressor as CuRF  # type: ignore
            return "cuml", CuRF, None
        except Exception as exc:
            plan_fallback = (
                f"cuML RF {'Classifier' if classification else 'Regressor'} unavailable "
                f"({exc.__class__.__name__}); using scikit-learn CPU"
            )
    else:
        plan_fallback = plan.fallback_reason or "CPU requested"

    if classification:
        from sklearn.ensemble import RandomForestClassifier as SkRF
    else:
        from sklearn.ensemble import RandomForestRegressor as SkRF

    return "sklearn", SkRF, plan_fallback


def _rf_parameters_snapshot(
    *,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    max_features: Any,
    bootstrap: bool,
    random_seed: int,
    n_jobs: int,
    backend: str,
) -> dict[str, Any]:
    mf = max_features
    if isinstance(mf, float) and 0 < mf <= 1:
        mf_disp: Any = mf
    elif mf in ("sqrt", "log2", None) or isinstance(mf, str):
        mf_disp = mf if mf is not None else "sqrt"
    else:
        mf_disp = mf
    out: dict[str, Any] = {
        "n_estimators": n_estimators,
        "max_depth": max_depth if max_depth > 0 else None,
        "min_samples_leaf": min_samples_leaf,
        "max_features": mf_disp,
        "bootstrap": bootstrap,
        "random_state": random_seed,
    }
    if backend == "sklearn":
        out["n_jobs"] = n_jobs
    return out


class _RfModelWrapper:
    """Uniform save/predict surface for sklearn or cuML RF / Extra Trees."""

    def __init__(
        self,
        estimator: Any,
        features: list[str],
        *,
        backend: str,
        task: str = "regression",
    ) -> None:
        self._est = estimator
        self._backend = backend
        self._task = task
        self.feature_names_in_ = np.array(list(features))
        imp = getattr(estimator, "feature_importances_", None)
        if imp is None:
            self.feature_importances_ = np.zeros(len(features), dtype=float)
        else:
            self.feature_importances_ = np.asarray(imp, dtype=float)

    def _predict_raw(self, X: pd.DataFrame) -> np.ndarray:
        if self._backend == "cuml":
            try:
                import cudf  # type: ignore

                pred = self._est.predict(cudf.DataFrame(X))
                return np.asarray(pred.to_pandas() if hasattr(pred, "to_pandas") else pred, dtype=float)
            except Exception:
                pass
        return np.asarray(self._est.predict(X), dtype=float)

    def _predict_proba(self, X: pd.DataFrame) -> np.ndarray | None:
        if not hasattr(self._est, "predict_proba"):
            return None
        if self._backend == "cuml":
            try:
                import cudf  # type: ignore

                proba = self._est.predict_proba(cudf.DataFrame(X))
                return np.asarray(proba.to_pandas() if hasattr(proba, "to_pandas") else proba, dtype=float)
            except Exception:
                pass
        return np.asarray(self._est.predict_proba(X), dtype=float)

    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        cols = list(self.feature_names_in_)
        X = X_df.loc[:, cols]
        if self._task == "binary":
            proba = self._predict_proba(X)
            if proba is not None:
                if proba.ndim == 2 and proba.shape[1] >= 2:
                    return proba[:, 1]
                return proba.reshape(-1)
        if self._task == "multiclass":
            proba = self._predict_proba(X)
            if proba is not None:
                return proba
        return self._predict_raw(X)

    def save_model(self, path: str) -> None:
        import joblib

        joblib.dump(
            {
                "backend": self._backend,
                "features": list(self.feature_names_in_),
                "estimator": self._est,
                "task": self._task,
            },
            path,
        )


def load_rf_model(path: str) -> _RfModelWrapper:
    import joblib

    payload = joblib.load(path)
    if isinstance(payload, dict) and "estimator" in payload:
        return _RfModelWrapper(
            payload["estimator"],
            list(payload.get("features") or []),
            backend=str(payload.get("backend") or "sklearn"),
            task=str(payload.get("task") or "regression"),
        )
    feats = list(getattr(payload, "feature_names_in_", []) or [])
    return _RfModelWrapper(payload, [str(f) for f in feats], backend="sklearn")


def train_rf_regressor(
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
    bootstrap = bool(parameters.get("bootstrap", True))
    random_seed = int(parameters.get("random_seed", 42))
    n_jobs = int(parameters.get("n_jobs", -1))

    backend, Est, fallback_reason = _resolve_rf_backend(parameters, task=task)
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
            RandomForestClassifier as SkClf,
            RandomForestRegressor as SkReg,
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
    # For binary, predict returns probabilities — compare to labels for a cheap loss proxy.
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

    algo_params = _rf_parameters_snapshot(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=bootstrap,
        random_seed=random_seed,
        n_jobs=n_jobs,
        backend=backend,
    )
    algo_params["prediction_type"] = task if task != "regression" else "regression"
    kind = "Classifier" if classification else "Regressor"
    implementation = (
        f"cuML RandomForest{kind}"
        if backend == "cuml"
        else f"scikit-learn RandomForest{kind}"
    )
    runtime = build_algorithm_runtime(
        algorithm=ALGORITHM_RANDOM_FOREST,
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
            "rf_backend": backend,
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
        "algorithm": ALGORITHM_RANDOM_FOREST,
        "prediction_type": task if task != "regression" else "regression",
    }


@register_trainer
class RandomForestTrainer(ModelTrainer):
    algorithm_id = ALGORITHM_RANDOM_FOREST
    algorithm_label = "Random Forest"

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
        return train_rf_regressor(
            train_X=train_X,
            train_y=train_y,
            val_X=val_X,
            val_y=val_y,
            features=features,
            parameters=params,
            cancel_check=cancel_check,
            on_iteration=on_iteration,
        )
