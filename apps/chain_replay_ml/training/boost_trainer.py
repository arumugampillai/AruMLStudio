"""Unified model trainer entry point — dispatches via ModelTrainer registry."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from .trainers import get_trainer, normalize_algorithm_id
from .xgb_trainer import (
    TrainingCancelled,
    feature_importance_df,
    select_feature_columns,
)

__all__ = [
    "TrainingCancelled",
    "feature_importance_df",
    "select_feature_columns",
    "train_regressor",
]


def train_regressor(
    *,
    algorithm: str,
    train_X: pd.DataFrame,
    train_y: pd.Series,
    val_X: pd.DataFrame,
    val_y: pd.Series,
    features: list[str],
    parameters: dict[str, Any],
    cancel_check: Callable[[], bool] | None = None,
    on_iteration: Callable[[dict[str, Any]], None] | None = None,
    prediction_type: str = "regression",
    log_fn: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Train with the registered trainer for ``algorithm`` (pipeline-agnostic).

    Device selection goes through ``model_device.resolve_training_device`` so GPU
    policy is consistent for Optuna, walk-forward, production fit, and retrain.
    """
    from .algorithm_runtime import attach_prediction_latency
    from .model_device import (
        announce_device_plan,
        emit_startup_diagnostics_once,
        resolve_training_device,
    )

    algo = normalize_algorithm_id(algorithm)
    from .algorithm_capabilities import assert_algorithm_supports_prediction_type

    assert_algorithm_supports_prediction_type(algo, prediction_type)
    emit_startup_diagnostics_once(log_fn=log_fn)
    plan = resolve_training_device(algo, parameters)
    announce_device_plan(plan, log_fn=log_fn)

    trainer = get_trainer(algo)
    params = dict(parameters)
    params.setdefault("prediction_type", prediction_type)
    result = trainer.train(
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y,
        features=features,
        parameters=params,
        cancel_check=cancel_check,
        on_iteration=on_iteration,
        prediction_type=prediction_type,
    )
    return attach_prediction_latency(result, val_X=val_X, features=features)
