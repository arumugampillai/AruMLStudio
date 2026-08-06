"""Filesystem paths for Research Lab."""

from __future__ import annotations

import os


def research_lab_dir(data_dir: str) -> str:
    path = os.path.join(data_dir, "research_lab")
    os.makedirs(path, exist_ok=True)
    return path


def research_sessions_db_path(data_dir: str) -> str:
    return os.path.join(research_lab_dir(data_dir), "sessions.db")
