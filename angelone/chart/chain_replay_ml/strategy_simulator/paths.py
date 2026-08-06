"""Filesystem paths for strategy simulation runs."""

from __future__ import annotations

import os


def strategy_runs_dir(data_dir: str) -> str:
    path = os.path.join(data_dir, "strategy_runs")
    os.makedirs(path, exist_ok=True)
    return path


def strategy_runs_db_path(data_dir: str) -> str:
    return os.path.join(strategy_runs_dir(data_dir), "registry.db")
