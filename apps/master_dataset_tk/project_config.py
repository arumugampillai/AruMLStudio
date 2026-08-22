"""Persist and resolve ML Research Studio project (chart) folder."""

from __future__ import annotations

import json
import os
from typing import Any

_CONFIG_FILE = "ml_research_studio.json"


def bundled_chart_dir() -> str:
    from path_config import CHART_DATA_ROOT

    return CHART_DATA_ROOT


def config_path() -> str:
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    folder = os.path.join(base, "AruMLStudio") if os.environ.get("APPDATA") else os.path.join(base, ".arumlstudio")
    os.makedirs(folder, exist_ok=True)
    target_path = os.path.join(folder, _CONFIG_FILE)
    if not os.path.isfile(target_path):
        legacy_folder = os.path.join(base, "AruNeo") if os.environ.get("APPDATA") else os.path.join(base, ".aruneo")
        legacy_path = os.path.join(legacy_folder, _CONFIG_FILE)
        if os.path.isfile(legacy_path):
            try:
                import shutil
                shutil.copy2(legacy_path, target_path)
            except Exception:
                pass
    return target_path


def normalize_chart_dir(path: str) -> str:
    return os.path.abspath(os.path.normpath(str(path or "").strip()))


def resolve_chart_dir_from_selection(path: str) -> str:
    """Accept project data dir, repo root (``apps/``), or legacy ``angelone/chart``."""
    path = normalize_chart_dir(path)
    if os.path.isdir(os.path.join(path, "data")):
        return path
    nested_apps = os.path.join(path, "apps")
    if os.path.isdir(os.path.join(nested_apps, "data")) or os.path.isdir(
        os.path.join(nested_apps, "static")
    ):
        return nested_apps
    nested = os.path.join(path, "angelone", "chart")
    if os.path.isdir(os.path.join(nested, "data")):
        return nested
    return path


def validate_chart_dir(path: str) -> tuple[bool, str]:
    chart_dir = normalize_chart_dir(path)
    if not os.path.isdir(chart_dir):
        return False, f"Folder does not exist:\n{chart_dir}"
    return True, ""


def ensure_project_data_dir(chart_dir: str) -> str:
    data_dir = os.path.join(chart_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    return data_dir


def load_project_config() -> dict[str, Any]:
    path = config_path()
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_project_config(chart_dir: str) -> None:
    doc = load_project_config()
    doc["chart_dir"] = normalize_chart_dir(chart_dir)
    path = config_path()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)


def save_tick_data_dir(tick_data_dir: str) -> None:
    doc = load_project_config()
    doc["tick_data_dir"] = normalize_chart_dir(tick_data_dir)
    path = config_path()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)


DEFAULT_DATA_ROOT = r"D:\data"
DEFAULT_MASTER_DATA_DIR = r"D:\data\master_dataset"


def resolve_data_root(custom_root: str | None = None) -> str:
    from chain_replay_ml.core.data_root import resolve_data_root as _resolve
    return _resolve(custom_root)


def save_data_root(data_root: str) -> None:
    from chain_replay_ml.core.data_root import save_data_root as _save
    _save(data_root)


def get_data_root_service(data_root: str | None = None):
    from chain_replay_ml.core.data_root import get_data_root_service as _get_svc
    return _get_svc(data_root)


def save_master_data_dir(master_data_dir: str) -> None:
    doc = load_project_config()
    doc["master_data_dir"] = normalize_chart_dir(master_data_dir)
    path = config_path()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)


def resolve_master_data_dir(chart_dir: str | None = None) -> str:
    """Directory that holds master_dataset_*.db and related master JSON/prefs.

    Preference order:
    1. ``ARUMLSTUDIO_MASTER_DATA_DIR`` env (fallback: ``ARUNEO_MASTER_DATA_DIR``)
    2. ``master_data_dir`` in ml_research_studio.json
    3. Canonical ``D:\\data\\datasets\\master`` via DataRootService
    4. ``{chart_dir}/data/datasets`` (legacy default)
    """
    env = str(
        os.environ.get("ARUMLSTUDIO_MASTER_DATA_DIR")
        or os.environ.get("ARUNEO_MASTER_DATA_DIR")
        or ""
    ).strip()
    if env:
        path = normalize_chart_dir(env)
        os.makedirs(path, exist_ok=True)
        return path
    saved = str(load_project_config().get("master_data_dir") or "").strip()
    if saved:
        path = normalize_chart_dir(saved)
        os.makedirs(path, exist_ok=True)
        return path
    from chain_replay_ml.core.data_root import get_data_root_service
    canonical = get_data_root_service().get_datasets_dir("master")
    if os.path.isdir(canonical):
        return canonical
    legacy = r"D:\data\master_dataset"
    if os.path.isdir(legacy):
        return legacy
    base = chart_dir or bundled_chart_dir()
    path = os.path.join(base, "data", "datasets")
    os.makedirs(path, exist_ok=True)
    return path


def resolve_tick_data_dir(chart_dir: str | None = None) -> str:
    from tick_data_paths import resolve_tick_data_dir as _resolve

    return _resolve(chart_dir)


def resolve_chart_dir(*, cli_chart_dir: str | None = None) -> str:
    if cli_chart_dir:
        return resolve_chart_dir_from_selection(cli_chart_dir)
    saved = str(load_project_config().get("chart_dir") or "").strip()
    if saved:
        resolved = resolve_chart_dir_from_selection(saved)
        ok, _ = validate_chart_dir(resolved)
        if ok:
            return resolved
    return bundled_chart_dir()
