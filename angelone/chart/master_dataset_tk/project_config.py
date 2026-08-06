"""Persist and resolve ML Research Studio project (chart) folder."""

from __future__ import annotations

import json
import os
from typing import Any

_CONFIG_FILE = "ml_research_studio.json"


def bundled_chart_dir() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def config_path() -> str:
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    folder = os.path.join(base, "AruNeo")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, _CONFIG_FILE)


def normalize_chart_dir(path: str) -> str:
    return os.path.abspath(os.path.normpath(str(path or "").strip()))


def resolve_chart_dir_from_selection(path: str) -> str:
    """Accept chart dir directly, repo root, or nested angelone/chart."""
    path = normalize_chart_dir(path)
    if os.path.isdir(os.path.join(path, "data")):
        return path
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


DEFAULT_MASTER_DATA_DIR = r"D:\data\master_dataset"


def save_master_data_dir(master_data_dir: str) -> None:
    doc = load_project_config()
    doc["master_data_dir"] = normalize_chart_dir(master_data_dir)
    path = config_path()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)


def resolve_master_data_dir(chart_dir: str | None = None) -> str:
    """Directory that holds master_dataset_*.db and related master JSON/prefs.

    Preference order:
    1. ``ARUNEO_MASTER_DATA_DIR`` env
    2. ``master_data_dir`` in ml_research_studio.json
    3. ``{chart_dir}/data/datasets`` (legacy default)
    """
    env = str(os.environ.get("ARUNEO_MASTER_DATA_DIR") or "").strip()
    if env:
        path = normalize_chart_dir(env)
        os.makedirs(path, exist_ok=True)
        return path
    saved = str(load_project_config().get("master_data_dir") or "").strip()
    if saved:
        path = normalize_chart_dir(saved)
        os.makedirs(path, exist_ok=True)
        return path
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
