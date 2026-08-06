"""CatBoost trainer adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from .base import (
    ALGORITHM_CATBOOST,
    ModelTrainer,
    register_trainer,
)


@register_trainer
class CatBoostTrainer(ModelTrainer):
    algorithm_id = ALGORITHM_CATBOOST
    algorithm_label = "CatBoost"

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
        from ..catboost_trainer import train_catboost_regressor

        assert_algorithm_supports_prediction_type(self.algorithm_id, prediction_type)
        params = dict(parameters)
        params["prediction_type"] = prediction_type
        result = train_catboost_regressor(
            train_X=train_X,
            train_y=train_y,
            val_X=val_X,
            val_y=val_y,
            features=features,
            parameters=params,
            cancel_check=cancel_check,
            on_iteration=on_iteration,
        )
        result.setdefault("algorithm", self.algorithm_id)
        return result
