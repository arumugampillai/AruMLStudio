"""Trainer registry core (no auto-imports — avoids circular import)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import pandas as pd

ALGORITHM_XGBOOST = "xgboost"
ALGORITHM_LIGHTGBM = "lightgbm"
ALGORITHM_CATBOOST = "catboost"
ALGORITHM_RANDOM_FOREST = "random_forest"
ALGORITHM_EXTRA_TREES = "extra_trees"

ALGORITHM_LABELS: dict[str, str] = {
    ALGORITHM_XGBOOST: "XGBoost",
    ALGORITHM_LIGHTGBM: "LightGBM",
    ALGORITHM_CATBOOST: "CatBoost",
    ALGORITHM_RANDOM_FOREST: "Random Forest",
    ALGORITHM_EXTRA_TREES: "Extra Trees",
}

NATIVE_EXTENSIONS: dict[str, str] = {
    ALGORITHM_XGBOOST: "ubj",
    ALGORITHM_LIGHTGBM: "lgb",
    ALGORITHM_CATBOOST: "cbm",
    ALGORITHM_RANDOM_FOREST: "joblib",
    ALGORITHM_EXTRA_TREES: "joblib",
}

_ALIASES: dict[str, str] = {
    "xgb": ALGORITHM_XGBOOST,
    "lgb": ALGORITHM_LIGHTGBM,
    "light_gbm": ALGORITHM_LIGHTGBM,
    "cat": ALGORITHM_CATBOOST,
    "rf": ALGORITHM_RANDOM_FOREST,
    "randomforest": ALGORITHM_RANDOM_FOREST,
    "random-forest": ALGORITHM_RANDOM_FOREST,
    "et": ALGORITHM_EXTRA_TREES,
    "extratrees": ALGORITHM_EXTRA_TREES,
    "extra-trees": ALGORITHM_EXTRA_TREES,
    "extra_tree": ALGORITHM_EXTRA_TREES,
}

_TRAINERS: dict[str, type] = {}
_BUILTINS_LOADED = False


def normalize_algorithm_id(algorithm: str | None) -> str:
    key = str(algorithm or ALGORITHM_XGBOOST).strip().lower().replace(" ", "_")
    key = _ALIASES.get(key, key)
    if key in NATIVE_EXTENSIONS:
        return key
    return ALGORITHM_XGBOOST


def algorithm_display_label(algorithm: str | None) -> str:
    key = normalize_algorithm_id(algorithm)
    return ALGORITHM_LABELS.get(key, str(algorithm or "Unknown").title())


class ModelTrainer(ABC):
    """Common training interface for all supported algorithms."""

    algorithm_id: str
    algorithm_label: str

    @abstractmethod
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
        """Train; return dict with ``model``, ``features``, ``algorithm``, etc."""

    def supports_binary_classification(self) -> bool:
        from ..algorithm_capabilities import get_algorithm_capabilities

        return bool(get_algorithm_capabilities(self.algorithm_id).supports_binary_classification)

    def supports_prediction_type(self, prediction_type: str | None) -> bool:
        from ..algorithm_capabilities import algorithm_supports_prediction_type

        return algorithm_supports_prediction_type(self.algorithm_id, prediction_type)

    def feature_importance(self, model: Any, features: list[str] | None = None) -> Any:
        from ..xgb_trainer import feature_importance_df

        return feature_importance_df(model, features)


def register_trainer(cls: type) -> type:
    aid = str(getattr(cls, "algorithm_id", "") or "").strip().lower()
    if not aid:
        raise ValueError(f"Trainer {cls!r} missing algorithm_id")
    _TRAINERS[aid] = cls
    return cls


def _ensure_builtin_trainers() -> None:
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    from . import catboost as _catboost  # noqa: F401
    from . import extra_trees as _extra_trees  # noqa: F401
    from . import lightgbm as _lightgbm  # noqa: F401
    from . import random_forest as _random_forest  # noqa: F401
    from . import xgboost as _xgboost  # noqa: F401

    _BUILTINS_LOADED = True


def get_trainer(algorithm: str | None) -> ModelTrainer:
    _ensure_builtin_trainers()
    algo = normalize_algorithm_id(algorithm)
    cls = _TRAINERS.get(algo)
    if cls is None:
        raise ValueError(f"No trainer registered for algorithm={algorithm!r}")
    return cls()  # type: ignore[call-arg]


def supported_algorithms() -> list[tuple[str, str]]:
    _ensure_builtin_trainers()
    order = (
        ALGORITHM_XGBOOST,
        ALGORITHM_LIGHTGBM,
        ALGORITHM_CATBOOST,
        ALGORITHM_RANDOM_FOREST,
        ALGORITHM_EXTRA_TREES,
    )
    out: list[tuple[str, str]] = []
    for aid in order:
        if aid in _TRAINERS:
            out.append((aid, ALGORITHM_LABELS.get(aid, aid)))
    for aid, cls in sorted(_TRAINERS.items()):
        if aid not in {a for a, _ in out}:
            out.append((aid, str(getattr(cls, "algorithm_label", aid))))
    return out
