"""Central resolution for angel_market_YYYY-MM-DD.db tick database locations."""

from __future__ import annotations

import json
import os

DEFAULT_TICK_DATA_DIR = r"D:\data\ticks"


def tick_db_filename(day: str) -> str:
    return f"angel_market_{day}.db"


def _ml_research_studio_config_path() -> str:
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    return os.path.join(base, "AruNeo", "ml_research_studio.json")


def _load_config_tick_data_dir() -> str:
    path = _ml_research_studio_config_path()
    if not os.path.isfile(path):
        return ""
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        if not isinstance(doc, dict):
            return ""
        return str(doc.get("tick_data_dir") or "").strip()
    except (OSError, json.JSONDecodeError):
        return ""


def resolve_tick_data_dir(chart_dir: str | None = None) -> str:
    """Resolve the primary tick DB directory (env → config → default)."""
    del chart_dir  # reserved for future chart-relative overrides
    env = str(os.environ.get("ARUNEO_TICK_DATA_DIR") or "").strip()
    if env:
        tick_dir = os.path.abspath(os.path.normpath(env))
    else:
        saved = _load_config_tick_data_dir()
        if saved:
            tick_dir = os.path.abspath(os.path.normpath(saved))
        else:
            tick_dir = os.path.abspath(DEFAULT_TICK_DATA_DIR)
    os.makedirs(tick_dir, exist_ok=True)
    return tick_dir


def tick_search_dirs(chart_dir: str) -> list[str]:
    """Ordered directories to scan for angel_market_*.db files."""
    chart_dir = os.path.abspath(os.path.normpath(chart_dir))
    tick_dir = resolve_tick_data_dir(chart_dir)
    dirs: list[str] = [tick_dir]
    tick_old = os.path.join(tick_dir, "old")
    if os.path.isdir(tick_old):
        dirs.append(tick_old)

    legacy_root = os.path.join(chart_dir, "data")
    if legacy_root not in dirs:
        dirs.append(legacy_root)
    legacy_old = os.path.join(legacy_root, "old")
    if os.path.isdir(legacy_old) and legacy_old not in dirs:
        dirs.append(legacy_old)
    return dirs


def replay_db_path(chart_dir: str, day: str) -> str | None:
    """Return the first non-empty angel_market_<day>.db under tick_search_dirs."""
    filename = tick_db_filename(day)
    for search_dir in tick_search_dirs(chart_dir):
        if not os.path.isdir(search_dir):
            continue
        path = os.path.join(search_dir, filename)
        if not os.path.isfile(path):
            continue
        try:
            if os.path.getsize(path) > 0:
                return path
        except OSError:
            continue
    return None
