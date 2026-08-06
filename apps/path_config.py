"""Repo path anchors — ML Studio packages, static assets, and bundled data under ``apps/``."""

from __future__ import annotations

import os

_APPS_DIR = os.path.dirname(os.path.abspath(__file__))
APPS_DIR = _APPS_DIR
REPO_ROOT = os.path.dirname(_APPS_DIR)
STATIC_DIR = os.path.join(APPS_DIR, "static")
# Default project folder (``data/``, models, datasets) for bundled / dev layout.
CHART_DATA_ROOT = APPS_DIR


def ensure_ml_studio_paths() -> None:
    """Insert ``apps/`` and repo root on ``sys.path`` when needed."""
    import sys

    for path in (APPS_DIR, REPO_ROOT):
        if path not in sys.path:
            sys.path.insert(0, path)
