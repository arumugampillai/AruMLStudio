"""ML model training pipeline — config-driven, time-series aware."""

from .classifier_registration import register_research_candidate_as_classifier
from .config import TrainingConfig, normalize_training_config
from .config_validator import validate_training_config
from .orchestrator import train_model

__all__ = [
    "TrainingConfig",
    "normalize_training_config",
    "validate_training_config",
    "train_model",
    "register_research_candidate_as_classifier",
]

