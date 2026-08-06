"""On-disk cache for replay feature day frames (survives server restarts)."""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any

import pandas as pd

from chain_replay_ml.dataset_builder.writer import _safe_filename


def _cache_root(data_dir: str, version: int) -> str:
    root = os.path.join(os.path.abspath(data_dir), "replay_cache", f"v{version}")
    os.makedirs(root, exist_ok=True)
    return root


def _key_digest(cache_key: tuple[Any, ...]) -> str:
    raw = json.dumps(cache_key, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def day_frame_disk_paths(data_dir: str, cache_key: tuple[Any, ...]) -> tuple[str, str]:
    """Return (parquet_path, meta_path) for a replay day-frame cache key."""
    version = int(cache_key[0]) if cache_key else 0
    model_name = str(cache_key[2]) if len(cache_key) > 2 else "model"
    date_str = str(cache_key[3]) if len(cache_key) > 3 else "day"
    safe_model = _safe_filename(model_name) or "model"
    subdir = os.path.join(_cache_root(data_dir, version), safe_model)
    os.makedirs(subdir, exist_ok=True)
    digest = _key_digest(cache_key)
    base = f"{date_str}_{digest}"
    return (
        os.path.join(subdir, f"{base}.parquet"),
        os.path.join(subdir, f"{base}.meta.json"),
    )


def load_day_frame_disk(data_dir: str, cache_key: tuple[Any, ...]) -> pd.DataFrame | None:
    parquet_path, meta_path = day_frame_disk_paths(data_dir, cache_key)
    if not os.path.isfile(parquet_path) or not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, encoding="utf-8") as fh:
            meta = json.load(fh)
        if meta.get("cache_key_digest") != _key_digest(cache_key):
            return None
        df = pd.read_parquet(parquet_path)
        return df if not df.empty else None
    except Exception:
        return None


def save_day_frame_disk(data_dir: str, cache_key: tuple[Any, ...], df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    parquet_path, meta_path = day_frame_disk_paths(data_dir, cache_key)
    try:
        df.to_parquet(parquet_path, index=False)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump({
                "cache_key_digest": _key_digest(cache_key),
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
            }, fh)
    except Exception:
        pass
