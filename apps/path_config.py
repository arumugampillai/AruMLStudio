"""Repo path anchors — ML Studio packages, static assets, and bundled data under ``apps/``."""

from __future__ import annotations

import os

_APPS_DIR = os.path.dirname(os.path.abspath(__file__))
APPS_DIR = _APPS_DIR
REPO_ROOT = os.path.dirname(_APPS_DIR)
STATIC_DIR = os.path.join(APPS_DIR, "static")
# Legacy non-authoritative fallback constant maintained for backward compatibility.
# Authoritative application data MUST resolve via chain_replay_ml.core.data_root.DataRootService.
CHART_DATA_ROOT = APPS_DIR


def canonical_data_root() -> str:
    """Return the authoritative application Data Root."""
    from chain_replay_ml.core.data_root import resolve_data_root
    return resolve_data_root()


def ensure_ml_studio_paths() -> None:
    """Insert ``apps/`` and repo root on ``sys.path`` when needed."""
    import sys

    for path in (REPO_ROOT, APPS_DIR):
        while path in sys.path:
            sys.path.remove(path)
        sys.path.insert(0, path)
