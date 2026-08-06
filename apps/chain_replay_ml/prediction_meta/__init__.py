"""Prediction Meta Dataset — ensemble + per-model predictions from master SQLite rows."""

from .builder import build_prediction_meta_dataset, resolve_prediction_meta_db_path
from .model_registry import read_model_registry, read_prediction_versions
from .projects import (
    clone_project_config,
    create_project,
    db_filename_from_display_name,
    delete_project,
    get_project,
    list_available_models,
    list_master_datasets,
    list_projects,
    slugify_project_name,
)
from .status import read_prediction_meta_dashboard, read_prediction_meta_status

__all__ = [
    "build_prediction_meta_dataset",
    "resolve_prediction_meta_db_path",
    "read_prediction_meta_status",
    "read_prediction_meta_dashboard",
    "read_model_registry",
    "read_prediction_versions",
    "list_projects",
    "create_project",
    "clone_project_config",
    "delete_project",
    "get_project",
    "list_master_datasets",
    "list_available_models",
    "slugify_project_name",
    "db_filename_from_display_name",
]
