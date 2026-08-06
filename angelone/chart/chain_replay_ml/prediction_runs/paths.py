"""Filesystem paths for the Prediction Run registry."""

from __future__ import annotations

import os


def prediction_runs_dir(data_dir: str) -> str:
    path = os.path.join(data_dir, "prediction_runs")
    os.makedirs(path, exist_ok=True)
    return path


def prediction_runs_db_path(data_dir: str) -> str:
    return os.path.join(prediction_runs_dir(data_dir), "registry.db")
