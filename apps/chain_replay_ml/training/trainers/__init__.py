"""Algorithm-agnostic trainers — use ``get_trainer(algorithm)``."""

from .base import (
    ALGORITHM_CATBOOST,
    ALGORITHM_EXTRA_TREES,
    ALGORITHM_LABELS,
    ALGORITHM_LIGHTGBM,
    ALGORITHM_RANDOM_FOREST,
    ALGORITHM_XGBOOST,
    NATIVE_EXTENSIONS,
    ModelTrainer,
    algorithm_display_label,
    get_trainer,
    normalize_algorithm_id,
    register_trainer,
    supported_algorithms,
)

__all__ = [
    "ALGORITHM_CATBOOST",
    "ALGORITHM_EXTRA_TREES",
    "ALGORITHM_LABELS",
    "ALGORITHM_LIGHTGBM",
    "ALGORITHM_RANDOM_FOREST",
    "ALGORITHM_XGBOOST",
    "NATIVE_EXTENSIONS",
    "ModelTrainer",
    "algorithm_display_label",
    "get_trainer",
    "normalize_algorithm_id",
    "register_trainer",
    "supported_algorithms",
]
