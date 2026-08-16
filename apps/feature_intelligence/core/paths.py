"""Path helpers for Feature Intelligence Core."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PACKAGE_ROOT / "config"
MIGRATIONS_DIR = PACKAGE_ROOT / "migrations"
MIGRATION_VERSIONS_DIR = MIGRATIONS_DIR / "versions"


def default_data_dir() -> Path:
    """Return the default on-disk data directory for FIC artifacts."""
    appdata = os.environ.get("APPDATA")
    if appdata:
        target_dir = Path(appdata) / "AruMLStudio" / "feature_intelligence"
        legacy_dir = Path(appdata) / "AruNeo" / "feature_intelligence"
    else:
        target_dir = Path.home() / ".arumlstudio" / "feature_intelligence"
        legacy_dir = Path.home() / ".aruneo" / "feature_intelligence"

    if not target_dir.exists() and legacy_dir.exists():
        try:
            import shutil
            shutil.copytree(legacy_dir, target_dir, dirs_exist_ok=True)
        except Exception:
            pass
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def default_db_path() -> Path:
    """Default SQLite path for ``feature_intelligence.db``."""
    return default_data_dir() / "feature_intelligence.db"
