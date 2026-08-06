"""Regime / Bucket transform UI helpers."""

from __future__ import annotations

from typing import Any

from .regime import REGIME_METHODS, normalize_regime_method, regime_column_name

DEFAULT_REGIME_WINDOWS: tuple[int, ...] = (20, 50)
DEFAULT_REGIME_METHODS: tuple[str, ...] = ("binary_threshold", "ternary_state")

METHOD_DISPLAY_LABELS: dict[str, str] = {
    "threshold_bucket": "Threshold Bucket",
    "quantile_bucket": "Quantile Bucket",
    "equal_width": "Equal Width Bucket",
    "binary_threshold": "Binary Threshold",
    "ternary_state": "Ternary State",
}

_WINDOWED = frozenset({"quantile_bucket", "equal_width"})


def build_regime_transformation_config(
    *,
    enabled: bool,
    features: list[str],
    methods: list[str],
    windows: list[int],
    n_bins: int = 5,
    threshold: float = 0.0,
    low: float = -1.0,
    high: float = 1.0,
    edges: list[float] | None = None,
    partition_by: list[str] | None = None,
    sample_interval_sec: float | int | None = None,
    order: int = 80,
) -> dict[str, Any]:
    feats = [str(f).strip() for f in features if str(f).strip()]
    meths: list[str] = []
    for m in methods:
        try:
            n = normalize_regime_method(m)
        except Exception:
            continue
        if n not in meths:
            meths.append(n)
    wins = sorted({int(w) for w in windows if int(w) > 0})
    needs_win = any(m in _WINDOWED for m in meths)
    entry: dict[str, Any] = {
        "id": "regime",
        "enabled": bool(enabled) and bool(feats) and bool(meths) and (bool(wins) or not needs_win),
        "order": int(order),
        "name": "Regime / Bucket",
        "params": {
            "features": feats,
            "methods": meths,
            "windows": wins,
            "n_bins": int(n_bins),
            "threshold": float(threshold),
            "low": float(low),
            "high": float(high),
            "edges": list(edges if edges is not None else [-1.0, 0.0, 1.0]),
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


def merge_regime_into_config(
    base: dict[str, Any] | None,
    *,
    enabled: bool,
    features: list[str],
    methods: list[str],
    windows: list[int],
    n_bins: int = 5,
    threshold: float = 0.0,
    low: float = -1.0,
    high: float = 1.0,
    edges: list[float] | None = None,
    partition_by: list[str] | None = None,
    sample_interval_sec: float | int | None = None,
) -> dict[str, Any]:
    from .config import normalize_transformation_config

    cfg = normalize_transformation_config(base)
    transforms = [
        t for t in (cfg.get("transformations") or [])
        if isinstance(t, dict) and str(t.get("id") or "") != "regime"
    ]
    regime_cfg = build_regime_transformation_config(
        enabled=enabled,
        features=features,
        methods=methods,
        windows=windows,
        n_bins=n_bins,
        threshold=threshold,
        low=low,
        high=high,
        edges=edges,
        partition_by=partition_by,
        sample_interval_sec=sample_interval_sec,
    )
    for entry in regime_cfg.get("transformations") or []:
        transforms.append(entry)
    cfg["transformations"] = transforms
    return cfg


def regime_column_count(
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
            meth = normalize_regime_method(m)
        except Exception:
            continue
        if meth in _WINDOWED:
            total += n_f * len(wins)
        else:
            total += n_f
    return total


def preview_regime_columns(
    *,
    features: list[str],
    methods: list[str],
    windows: list[int],
    n_bins: int = 5,
    threshold: float = 0.0,
    low: float = -1.0,
    high: float = 1.0,
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
                meth = normalize_regime_method(m)
            except Exception:
                continue
            if meth in _WINDOWED:
                for w in wins:
                    cols.append(
                        regime_column_name(f, meth, window=w, n_bins=n_bins)
                    )
                    if len(cols) >= limit:
                        return cols
            else:
                cols.append(
                    regime_column_name(
                        f, meth, threshold=threshold, low=low, high=high
                    )
                )
                if len(cols) >= limit:
                    return cols
    return cols


def validate_regime_settings(
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
        return "Regime is enabled but no features are selected."
    meths: list[str] = []
    for m in methods:
        try:
            meths.append(normalize_regime_method(m))
        except Exception:
            return f"Regime method {m!r} is not supported."
    if not meths:
        return "Regime is enabled but no methods are selected."
    wins = [int(w) for w in windows if int(w) > 0]
    if any(m in _WINDOWED for m in meths) and not wins:
        return "Regime needs at least one window for the selected methods."
    if available_features is not None:
        avail = {str(a) for a in available_features}
        missing = [f for f in feats if f not in avail]
        if missing:
            return "Regime features not in Master:\n" + "\n".join(missing[:8])
    return None


def format_regime_preview_text(
    *,
    enabled: bool,
    feature_count: int,
    method_count: int,
    window_count: int,
    columns_to_add: int,
    sample_names: list[str] | None = None,
) -> str:
    if not enabled:
        return "Regime disabled — no extra columns."
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
    "DEFAULT_REGIME_METHODS",
    "DEFAULT_REGIME_WINDOWS",
    "METHOD_DISPLAY_LABELS",
    "REGIME_METHODS",
    "build_regime_transformation_config",
    "format_regime_preview_text",
    "merge_regime_into_config",
    "preview_regime_columns",
    "regime_column_count",
    "validate_regime_settings",
]
