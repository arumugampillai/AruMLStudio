"""Filesystem paths for the Strategy Registry."""

from __future__ import annotations

import os


def strategies_dir(data_dir: str) -> str:
    path = os.path.join(data_dir, "strategies")
    os.makedirs(path, exist_ok=True)
    return path


def strategy_registry_db_path(data_dir: str) -> str:
    return os.path.join(strategies_dir(data_dir), "registry.db")


def strategy_package_dir(data_dir: str, strategy_id: str, version_label: str) -> str:
    safe = str(strategy_id).replace("/", "_")
    return os.path.join(strategies_dir(data_dir), safe, version_label)
