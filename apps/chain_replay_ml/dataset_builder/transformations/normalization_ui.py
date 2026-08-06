"""Normalization transform UI helpers."""

from __future__ import annotations

from typing import Any

from .normalization import (
    NORMALIZATION_METHODS,
    normalization_column_name,
    normalize_normalization_method,
)

DEFAULT_NORMALIZATION_WINDOWS: tuple[int, ...] = (20, 50, 100)
DEFAULT_NORMALIZATION_METHODS: tuple[str, ...] = ("zscore_rolling", "zscore_expanding")

METHOD_DISPLAY_LABELS: dict[str, str] = {
    "zscore_rolling": "Rolling Z-Score",
    "zscore_expanding": "Expanding Z-Score",
    "robust": "Robust Scaling",
    "minmax": "Min-Max Scaling",
    "percentile_rank": "Percentile Rank",
    "quantile_rank": "Quantile Rank",
}

_WINDOWED = frozenset({
    "zscore_rolling", "robust", "minmax", "percentile_rank", "quantile_rank",
})


def build_normalization_transformation_config(
    *,
    enabled: bool,
    features: list[str],
    methods: list[str],
    windows: list[int],
    partition_by: list[str] | None = None,
    sample_interval_sec: float | int | None = None,
    ddof: int = 0,
    order: int = 75,
) -> dict[str, Any]:
    feats = [str(f).strip() for f in features if str(f).strip()]
    meths: list[str] = []
    for m in methods:
        try:
            n = normalize_normalization_method(m)
        except Exception:
            continue
        if n not in meths:
            meths.append(n)
    wins = sorted({int(w) for w in windows if int(w) > 0})
    needs_win = any(m in _WINDOWED for m in meths)
    entry: dict[str, Any] = {
        "id": "normalization",
        "enabled": bool(enabled) and bool(feats) and bool(meths) and (bool(wins) or not needs_win),
        "order": int(order),
        "name": "Normalization",
        "params": {
            "features": feats,
            "methods": meths,
            "windows": wins,
            "ddof": int(ddof),
            "partition_by": list(partition_by or ["trading_day", "token"]),
        },
    }
    if sample_interval_sec is not None:
        try:
            entry["params"]["sample_interval_sec"] = float(sample_interval_sec)
        except (TypeError, ValueError):
            pass
    return {
        "transformation_pipeline_version": 1,
        "transformations": [entry] if entry["enabled"] else [],
    }


def merge_normalization_into_config(
    base: dict[str, Any] | None,
    *,
    enabled: bool,
    features: list[str],
    methods: list[str],
    windows: list[int],
    partition_by: list[str] | None = None,
    sample_interval_sec: float | int | None = None,
    ddof: int = 0,
) -> dict[str, Any]:
    from .config import normalize_transformation_config

    cfg = normalize_transformation_config(base)
    transforms = [
        t for t in (cfg.get("transformations") or [])
        if isinstance(t, dict) and str(t.get("id") or "") != "normalization"
    ]
    norm_cfg = build_normalization_transformation_config(
        enabled=enabled,
        features=features,
        methods=methods,
        windows=windows,
        partition_by=partition_by,
        sample_interval_sec=sample_interval_sec,
        ddof=ddof,
    )
    for entry in norm_cfg.get("transformations") or []:
        transforms.append(entry)
    cfg["transformations"] = transforms
    return cfg


def normalization_column_count(
    *,
    enabled: bool,
    features: list[str],
    methods: list[str],
    windows: list[int],
) -> int:
    if not enabled:
        return 0
    n_f = len([f for f in features if str(f).strip()])
    wins = [int(w) for w in windows if int(w) > 0]
    total = 0
    for m in methods:
        try:
            meth = normalize_normalization_method(m)
        except Exception:
            continue
        if meth in _WINDOWED:
            total += n_f * len(wins)
        else:
            total += n_f
    return total


def preview_normalization_columns(
    *,
    features: list[str],
    methods: list[str],
    windows: list[int],
    limit: int = 12,
) -> list[str]:
    cols: list[str] = []
    wins = [int(w) for w in windows if int(w) > 0]
    for feat in features:
        f = str(feat).strip()
        if not f:
            continue
        for m in methods:
            try:
                meth = normalize_normalization_method(m)
            except Exception:
                continue
            if meth in _WINDOWED:
                for w in wins:
                    cols.append(normalization_column_name(f, meth, window=w))
                    if len(cols) >= limit:
                        return cols
            else:
                cols.append(normalization_column_name(f, meth))
                if len(cols) >= limit:
                    return cols
    return cols


def validate_normalization_settings(
    *,
    enabled: bool,
    features: list[str],
    methods: list[str],
    windows: list[int],
    available_features: list[str] | None = None,
) -> str | None:
    if not enabled:
        return None
    feats = [str(f).strip() for f in features if str(f).strip()]
    if not feats:
        return "Normalization is enabled but no features are selected."
    meths: list[str] = []
    for m in methods:
        try:
            meths.append(normalize_normalization_method(m))
        except Exception:
            return f"Normalization method {m!r} is not supported."
    if not meths:
        return "Normalization is enabled but no methods are selected."
    wins = [int(w) for w in windows if int(w) > 0]
    if any(m in _WINDOWED for m in meths) and not wins:
        return "Normalization needs at least one window for the selected methods."
    if available_features is not None:
        avail = {str(a) for a in available_features}
        missing = [f for f in feats if f not in avail]
        if missing:
            return "Normalization features not in Master:\n" + "\n".join(missing[:8])
    return None


def format_normalization_preview_text(
    *,
    enabled: bool,
    feature_count: int,
    method_count: int,
    window_count: int,
    columns_to_add: int,
    sample_names: list[str] | None = None,
) -> str:
    if not enabled:
        return "Normalization disabled — no extra columns."
    lines = [
        f"Features    : {feature_count}",
        f"Methods     : {method_count}",
        f"Windows     : {window_count}",
        f"New Columns : {columns_to_add}",
    ]
    if sample_names:
        lines.append("Examples    : " + ", ".join(sample_names[:4]))
    return "\n".join(lines)


__all__ = [
    "DEFAULT_NORMALIZATION_METHODS",
    "DEFAULT_NORMALIZATION_WINDOWS",
    "METHOD_DISPLAY_LABELS",
    "NORMALIZATION_METHODS",
    "build_normalization_transformation_config",
    "format_normalization_preview_text",
    "merge_normalization_into_config",
    "normalization_column_count",
    "preview_normalization_columns",
    "validate_normalization_settings",
]
