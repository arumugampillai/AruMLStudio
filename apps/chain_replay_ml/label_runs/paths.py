"""Filesystem layout for Label Run artifacts."""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone


def label_runs_dir(data_dir: str | None = None) -> str:
    r"""Canonical application labels location: D:\data\datasets\labels."""
    if data_dir is None:
        from chain_replay_ml.core.data_root import get_data_root_service
        return get_data_root_service().get_datasets_dir("labels")
    d_str = str(data_dir).strip()
    sub_labels = os.path.join(d_str, "datasets", "labels")
    if os.path.isdir(sub_labels):
        return sub_labels
    if os.path.basename(os.path.normpath(d_str)).lower() in ("labels", "label_runs"):
        return os.path.abspath(d_str)
    legacy = os.path.join(d_str, "label_runs")
    if os.path.isdir(legacy):
        return legacy
    from chain_replay_ml.core.data_root import get_data_root_service
    return get_data_root_service().get_datasets_dir("labels")


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
