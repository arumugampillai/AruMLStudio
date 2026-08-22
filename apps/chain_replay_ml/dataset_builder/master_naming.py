from __future__ import annotations

import json
import os
import re
from typing import Any

# Authoritative supported sampling intervals across Master Dataset, Research Leaderboard, and Models
MASTER_DATASET_INTERVALS_SEC: tuple[int, ...] = (3, 6, 9, 10, 15, 30, 60)
MASTER_DATASET_INTERVAL_LABELS: tuple[str, ...] = ("3s", "6s", "9s", "10s", "15s", "30s", "1m")
MASTER_DATASET_INTERVAL_LABEL_TO_SEC: dict[str, int] = {
    "3s": 3,
    "6s": 6,
    "9s": 9,
    "10s": 10,
    "15s": 15,
    "30s": 30,
    "1m": 60,
    "60s": 60,
}
MASTER_DATASET_INTERVAL_SEC_TO_LABEL: dict[int, str] = {
    3: "3s",
    6: "6s",
    9: "9s",
    10: "10s",
    15: "15s",
    30: "30s",
    60: "1m",
}


def parse_sampling_interval_sec(v: Any, default: int = 6) -> int:
    """Parse sampling interval from int, str ('3s', '6s', '1m', 60) to seconds."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        iv = int(v)
        return iv if iv in MASTER_DATASET_INTERVAL_SEC_TO_LABEL else iv
    s = str(v).strip().lower()
    if s in MASTER_DATASET_INTERVAL_LABEL_TO_SEC:
        return MASTER_DATASET_INTERVAL_LABEL_TO_SEC[s]
    if s.endswith("s"):
        try:
            return int(s[:-1])
        except ValueError:
            pass
    if s.endswith("m"):
        try:
            return int(s[:-1]) * 60
        except ValueError:
            pass
    try:
        return int(s)
    except ValueError:
        return default


def format_sampling_interval_label(sec: int) -> str:
    """Format sampling interval in seconds to standard label ('3s', '6s', ..., '1m')."""
    return MASTER_DATASET_INTERVAL_SEC_TO_LABEL.get(int(sec), f"{int(sec)}s")


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
    otherwise canonical ``datasets/master`` via DataRootService, falling back to ``{data_dir}/datasets``.
    """
    configured = _configured_master_data_dir()
    if configured:
        os.makedirs(configured, exist_ok=True)
        return configured
    from chain_replay_ml.core.data_root import get_data_root_service
    canonical = get_data_root_service().get_datasets_dir("master")
    if os.path.isdir(canonical):
        return canonical
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
