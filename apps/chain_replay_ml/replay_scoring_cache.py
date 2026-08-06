"""In-process cache for replay feature generation and model scoring."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from chain_replay_ml.ticks import TickTimeline

_CACHE_VERSION = 15  # disk day-frame cache + win32 thread parallel
_day_frame_cache: dict[tuple[Any, ...], pd.DataFrame] = {}
_scored_frame_cache: dict[tuple[Any, ...], pd.DataFrame] = {}
_token_timelines_cache: dict[tuple[Any, ...], dict[str, TickTimeline]] = {}


def replay_cache_key(
    data_dir: str,
    model_name: str,
    date_str: str,
    expiry_hint: str | None,
) -> tuple[Any, ...]:
    return (
        _CACHE_VERSION,
        os.path.abspath(data_dir),
        str(model_name),
        str(date_str),
        str(expiry_hint or ""),
    )


def replay_timeline_cache_key(
    data_dir: str,
    date_str: str,
    expiry_hint: str | None,
) -> tuple[Any, ...]:
    return (
        _CACHE_VERSION,
        os.path.abspath(data_dir),
        str(date_str),
        str(expiry_hint or ""),
    )


def inference_snapshot_cache_key(
    data_dir: str,
    date_str: str,
    expiry_hint: str | None,
    aligned_ts: float,
    feature_sig: str,
) -> tuple[Any, ...]:
    return (
        _CACHE_VERSION,
        "inference_snapshot",
        os.path.abspath(data_dir),
        str(date_str),
        str(expiry_hint or ""),
        round(float(aligned_ts), 3),
        str(feature_sig),
    )


_inference_snapshot_cache: dict[tuple[Any, ...], pd.DataFrame] = {}


def get_cached_inference_snapshot(key: tuple[Any, ...]) -> pd.DataFrame | None:
    df = _inference_snapshot_cache.get(key)
    if df is None or df.empty:
        return None
    return df.copy()


def set_cached_inference_snapshot(key: tuple[Any, ...], df: pd.DataFrame) -> None:
    if df is not None and not df.empty:
        _inference_snapshot_cache[key] = df.copy()


def inference_row_cache_key(
    data_dir: str,
    model_name: str,
    date_str: str,
    expiry_hint: str | None,
    grid_ts: float,
    token: str,
) -> tuple[Any, ...]:
    return (
        _CACHE_VERSION,
        "inference_row",
        os.path.abspath(data_dir),
        str(model_name),
        str(date_str),
        str(expiry_hint or ""),
        round(float(grid_ts), 3),
        str(token),
    )


_inference_row_cache: dict[tuple[Any, ...], pd.Series] = {}
_shared_inference_cache: dict[tuple[Any, ...], dict[str, Any]] = {}


def shared_inference_cache_key(
    data_dir: str,
    date_str: str,
    expiry_hint: str | None,
    grid_ts: float,
    token: str,
    feature_sig: str,
) -> tuple[Any, ...]:
    return (
        _CACHE_VERSION,
        "shared_inference",
        os.path.abspath(data_dir),
        str(date_str),
        str(expiry_hint or ""),
        round(float(grid_ts), 3),
        str(token),
        str(feature_sig),
    )


def get_cached_shared_features(key: tuple[Any, ...]) -> dict[str, Any] | None:
    hit = _shared_inference_cache.get(key)
    return dict(hit) if hit else None


def set_cached_shared_features(key: tuple[Any, ...], features: dict[str, Any]) -> None:
    if features:
        _shared_inference_cache[key] = dict(features)


def get_cached_inference_row(key: tuple[Any, ...]) -> pd.Series | None:
    row = _inference_row_cache.get(key)
    if row is None:
        return None
    return row.copy()


def set_cached_inference_row(key: tuple[Any, ...], row: pd.Series) -> None:
    if row is not None and not row.empty:
        _inference_row_cache[key] = row.copy()


def clear_replay_scoring_cache() -> None:
    _day_frame_cache.clear()
    _scored_frame_cache.clear()
    _token_timelines_cache.clear()
    _inference_snapshot_cache.clear()
    _inference_row_cache.clear()
    _shared_inference_cache.clear()


def get_cached_day_frame(key: tuple[Any, ...]) -> pd.DataFrame | None:
    hit = _day_frame_cache.get(key)
    if hit is not None:
        return hit.copy()
    if len(key) >= 2:
        from .replay_disk_cache import load_day_frame_disk

        data_dir = str(key[1])
        disk_hit = load_day_frame_disk(data_dir, key)
        if disk_hit is not None and not disk_hit.empty:
            _day_frame_cache[key] = disk_hit
            return disk_hit.copy()
    return None


def set_cached_day_frame(key: tuple[Any, ...], df: pd.DataFrame) -> None:
    if df is not None and not df.empty:
        _day_frame_cache[key] = df
        if len(key) >= 2:
            from .replay_disk_cache import save_day_frame_disk

            save_day_frame_disk(str(key[1]), key, df)


def get_cached_scored_frame(key: tuple[Any, ...]) -> pd.DataFrame | None:
    hit = _scored_frame_cache.get(key)
    return hit.copy() if hit is not None else None


def set_cached_scored_frame(key: tuple[Any, ...], df: pd.DataFrame) -> None:
    if df is not None and not df.empty:
        _scored_frame_cache[key] = df


def get_cached_token_timelines(key: tuple[Any, ...]) -> dict[str, TickTimeline] | None:
    hit = _token_timelines_cache.get(key)
    return hit if hit is not None else None


def set_cached_token_timelines(key: tuple[Any, ...], timelines: dict[str, TickTimeline]) -> None:
    if timelines:
        _token_timelines_cache[key] = timelines
