"""XGBoost trainer adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from .base import (
    ALGORITHM_XGBOOST,
    ModelTrainer,
    register_trainer,
)


@register_trainer
class XGBoostTrainer(ModelTrainer):
    algorithm_id = ALGORITHM_XGBOOST
    algorithm_label = "XGBoost"

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
        from ..algorithm_capabilities import assert_algorithm_supports_prediction_type
        from ..xgb_trainer import train_xgb_binary_classifier, train_xgb_regressor

        assert_algorithm_supports_prediction_type(self.algorithm_id, prediction_type)
        kwargs = dict(
            train_X=train_X,
            train_y=train_y,
            val_X=val_X,
            val_y=val_y,
            features=features,
            parameters=parameters,
            cancel_check=cancel_check,
            on_iteration=on_iteration,
        )
        pred = str(prediction_type or "regression").strip().lower()
        if pred in ("binary", "classification", "multiclass"):
            result = train_xgb_binary_classifier(**kwargs)
        else:
            result = train_xgb_regressor(**kwargs)
        result.setdefault("algorithm", self.algorithm_id)
        return result
