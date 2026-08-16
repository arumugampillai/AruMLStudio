"""Master dataset file naming — {market}_master_{interval}s_atm{band}.db."""

from __future__ import annotations

import json
import os
import re


def normalize_market_slug(market: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "", str(market or "nifty").lower())
    return slug or "nifty"


def master_dataset_slug(*, market: str, sampling_interval_sec: int) -> str:
    m = normalize_market_slug(market)
    return f"master_dataset_{m}_{int(sampling_interval_sec)}s"


def master_db_filename(
    *,
    market: str,
    sampling_interval_sec: int,
    atm_band: int | None = None,
) -> str:
    del atm_band  # legacy param — filename is interval + market only
    return f"{master_dataset_slug(market=market, sampling_interval_sec=sampling_interval_sec)}.db"


def _project_config_path() -> str:
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    folder = os.path.join(base, "AruMLStudio") if os.environ.get("APPDATA") else os.path.join(base, ".arumlstudio")
    target_path = os.path.join(folder, "ml_research_studio.json")
    if not os.path.isfile(target_path):
        legacy_path = os.path.join(base, "AruNeo", "ml_research_studio.json")
        if os.path.isfile(legacy_path):
            try:
                import shutil
                os.makedirs(folder, exist_ok=True)
                shutil.copy2(legacy_path, target_path)
            except Exception:
                return legacy_path
    return target_path


def _configured_master_data_dir() -> str | None:
    env = str(
        os.environ.get("ARUMLSTUDIO_MASTER_DATA_DIR")
        or os.environ.get("ARUNEO_MASTER_DATA_DIR")
        or ""
    ).strip()
    if env:
        return os.path.abspath(os.path.normpath(env))
    path = _project_config_path()
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(doc, dict):
        return None
    saved = str(doc.get("master_data_dir") or "").strip()
    if not saved:
        return None
    return os.path.abspath(os.path.normpath(saved))


def resolve_master_datasets_dir(data_dir: str | None = None) -> str:
    """Root folder for master_dataset_*.db files.

    Uses ``ARUNEO_MASTER_DATA_DIR`` / project ``master_data_dir`` when set;
    otherwise ``{data_dir}/datasets``.
    """
    configured = _configured_master_data_dir()
    if configured:
        os.makedirs(configured, exist_ok=True)
        return configured
    base = str(data_dir or "").strip() or "."
    datasets = os.path.join(base, "datasets")
    os.makedirs(datasets, exist_ok=True)
    return datasets


def path_relative_to_data_dir(path: str, data_dir: str) -> str:
    """Return a path relative to *data_dir* when possible; else absolute.

    On Windows, ``os.path.relpath`` raises ValueError when *path* and *data_dir*
    are on different drives (e.g. master DB on ``D:``, chart data on ``C:``).
    """
    abs_path = os.path.abspath(path)
    abs_base = os.path.abspath(data_dir or ".")
    try:
        return os.path.relpath(abs_path, abs_base).replace("\\", "/")
    except ValueError:
        return abs_path.replace("\\", "/")


def resolve_master_db_path(
    data_dir: str,
    *,
    market: str,
    sampling_interval_sec: int,
    atm_band: int | None = None,
    filename: str | None = None,
) -> str:
    del atm_band
    datasets = resolve_master_datasets_dir(data_dir)
    name = filename or master_db_filename(
        market=market,
        sampling_interval_sec=sampling_interval_sec,
    )
    return os.path.join(datasets, name)
