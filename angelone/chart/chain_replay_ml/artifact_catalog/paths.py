"""Filesystem paths for the Artifact Catalog."""

from __future__ import annotations

import os


def catalog_dir(data_dir: str) -> str:
    path = os.path.join(data_dir, "artifact_catalog")
    os.makedirs(path, exist_ok=True)
    return path


def catalog_db_path(data_dir: str) -> str:
    return os.path.join(catalog_dir(data_dir), "catalog.db")


def experiments_dir(data_dir: str) -> str:
    path = os.path.join(catalog_dir(data_dir), "experiments")
    os.makedirs(path, exist_ok=True)
    return path
