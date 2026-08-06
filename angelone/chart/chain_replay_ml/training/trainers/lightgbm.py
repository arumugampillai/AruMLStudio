"""LightGBM trainer adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd

from .base import (
    ALGORITHM_LIGHTGBM,
    ModelTrainer,
    register_trainer,
)


@register_trainer
class LightGBMTrainer(ModelTrainer):
    algorithm_id = ALGORITHM_LIGHTGBM
    algorithm_label = "LightGBM"

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
        from ..lgb_trainer import train_lgb_regressor

        assert_algorithm_supports_prediction_type(self.algorithm_id, prediction_type)
        params = dict(parameters)
        params["prediction_type"] = prediction_type
        result = train_lgb_regressor(
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
