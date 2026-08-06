"""Filesystem layout for Label Run artifacts."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone


def label_runs_dir(data_dir: str) -> str:
    """``{data_dir}/label_runs`` — never under datasets/."""
    return os.path.join(str(data_dir), "label_runs")


def mint_label_run_id(strategy: str, *, suffix: str | None = None) -> str:
    """Mint an immutable Label Run id (filesystem-safe)."""
    strat = re.sub(r"[^a-zA-Z0-9]+", "_", str(strategy or "label").strip().lower()).strip("_") or "label"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if suffix:
        safe = re.sub(r"[^a-zA-Z0-9]+", "_", str(suffix).strip()).strip("_")
        return f"{strat}_run_{stamp}_{safe}"
    return f"{strat}_run_{stamp}"


def label_run_parquet_path(data_dir: str, run_id: str) -> str:
    return os.path.join(label_runs_dir(data_dir), f"{run_id}.parquet")


def label_run_meta_path(data_dir: str, run_id: str) -> str:
    return os.path.join(label_runs_dir(data_dir), f"{run_id}_meta.json")
