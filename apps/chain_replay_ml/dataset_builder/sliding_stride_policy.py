"""Sliding stride — decouple feature window (sampling interval) from sample step."""

from __future__ import annotations

from typing import Any


def resolve_feature_window_sec(sampling: dict[str, Any] | None) -> int:
    """Feature aggregation window duration (``trainingIntervalSec``)."""
    samp = sampling or {}
    raw = samp.get("trainingIntervalSec") or samp.get("interval_sec") or samp.get("sampling_interval_sec")
    return max(int(raw or 10), 1)


def resolve_sliding_stride_sec(sampling: dict[str, Any] | None) -> int:
    """Seconds between consecutive sample timestamps (defaults to feature window)."""
    window = resolve_feature_window_sec(sampling)
    samp = sampling or {}
    raw = samp.get("slidingStrideSec")
    if raw is None:
        raw = samp.get("sliding_stride_sec")
    if raw is None:
        return window
    return max(int(raw), 1)


def validate_sliding_stride(interval_sec: int, stride_sec: int) -> str | None:
    """Return a user-facing error message, or None when valid."""
    interval = max(int(interval_sec), 1)
    stride = max(int(stride_sec), 1)
    if stride <= 0:
        return "Sliding stride must be greater than 0."
    if stride > interval:
        return f"Sliding stride ({stride}s) cannot exceed sampling interval ({interval}s)."
    if interval % stride != 0:
        return (
            f"Sampling interval ({interval}s) must be evenly divisible by "
            f"sliding stride ({stride}s)."
        )
    return None


def sampling_stride_fields(sampling: dict[str, Any] | None) -> dict[str, int]:
    """Resolved window + stride for metadata and dataset configuration."""
    window = resolve_feature_window_sec(sampling)
    stride = resolve_sliding_stride_sec(sampling)
    return {
        "sampling_interval_sec": window,
        "feature_window_sec": window,
        "sliding_stride_sec": stride,
        "feature_grid_step_sec": stride,
    }
