"""Exponential Rolling UI helpers — periods, ops, config, preview, validation."""

from __future__ import annotations

from typing import Any

from .exponential_rolling import (
    EXPONENTIAL_ROLLING_OPS,
    exponential_rolling_column_name,
    normalize_exponential_rolling_op,
)

DEFAULT_EMA_PERIODS: tuple[int, ...] = (5, 10, 20, 50, 100, 200)
# Backward-compatible default when operations omitted from config/prefs.
DEFAULT_EXPONENTIAL_OPS: tuple[str, ...] = ("ema",)

OP_DISPLAY_LABELS: dict[str, str] = {
    "ema": "EMA",
    "ewm_mean": "EWM Mean",
    "ewm_std": "EWM Std",
}


def display_op_label(op: str) -> str:
    try:
        return OP_DISPLAY_LABELS[normalize_exponential_rolling_op(op)]
    except Exception:
        return str(op or "")


def build_exponential_rolling_config(
    *,
    enabled: bool,
    features: list[str],
    periods: list[int],
    operations: list[str] | None = None,
    partition_by: list[str] | None = None,
    sample_interval_sec: float | int | None = None,
    order: int = 42,
) -> dict[str, Any]:
    feats = [str(f).strip() for f in features if str(f).strip()]
    pers = sorted({int(p) for p in periods if int(p) > 0})
    ops: list[str] = []
    for op in operations or list(DEFAULT_EXPONENTIAL_OPS):
        try:
            n = normalize_exponential_rolling_op(op)
        except Exception:
            continue
        if n not in ops:
            ops.append(n)
    if not ops:
        ops = ["ema"]
    entry: dict[str, Any] = {
        "id": "exponential_rolling",
        "enabled": bool(enabled) and bool(feats) and bool(pers) and bool(ops),
        "order": int(order),
        "name": "Exponential Rolling",
        "params": {
            "features": feats,
            "periods": pers,
            "operations": ops,
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


def merge_exponential_rolling_into_config(
    base: dict[str, Any] | None,
    *,
    enabled: bool,
    features: list[str],
    periods: list[int],
    operations: list[str] | None = None,
    partition_by: list[str] | None = None,
    sample_interval_sec: float | int | None = None,
) -> dict[str, Any]:
    from .config import normalize_transformation_config

    cfg = normalize_transformation_config(base)
    transforms = [
        t
        for t in (cfg.get("transformations") or [])
        if isinstance(t, dict) and str(t.get("id") or "") != "exponential_rolling"
    ]
    exp_cfg = build_exponential_rolling_config(
        enabled=enabled,
        features=features,
        periods=periods,
        operations=operations,
        partition_by=partition_by,
        sample_interval_sec=sample_interval_sec,
    )
    for entry in exp_cfg.get("transformations") or []:
        transforms.append(entry)
    cfg["transformations"] = transforms
    return cfg


def exponential_rolling_column_count(
    *,
    enabled: bool,
    features: list[str],
    periods: list[int],
    operations: list[str] | None = None,
) -> int:
    if not enabled:
        return 0
    n_f = len([f for f in features if str(f).strip()])
    n_p = len([int(p) for p in periods if int(p) > 0])
    ops = operations or list(DEFAULT_EXPONENTIAL_OPS)
    n_o = 0
    for op in ops:
        try:
            normalize_exponential_rolling_op(op)
            n_o += 1
        except Exception:
            continue
    return n_f * n_p * max(n_o, 0)


def preview_exponential_rolling_columns(
    *,
    features: list[str],
    periods: list[int],
    operations: list[str] | None = None,
    limit: int = 12,
) -> list[str]:
    ops = operations or list(DEFAULT_EXPONENTIAL_OPS)
    cols: list[str] = []
    for feat in features:
        f = str(feat).strip()
        if not f:
            continue
        for per in periods:
            try:
                p = int(per)
            except (TypeError, ValueError):
                continue
            if p <= 0:
                continue
            for op in ops:
                try:
                    cols.append(exponential_rolling_column_name(f, op, p))
                except Exception:
                    continue
                if len(cols) >= limit:
                    return cols
    return cols


def validate_exponential_rolling_settings(
    *,
    enabled: bool,
    features: list[str],
    periods: list[int],
    operations: list[str] | None = None,
    available_features: list[str] | None = None,
) -> str | None:
    if not enabled:
        return None
    feats = [str(f).strip() for f in features if str(f).strip()]
    if not feats:
        return "Exponential Rolling is enabled but no features are selected."
    pers: list[int] = []
    for p in periods:
        try:
            pi = int(p)
        except (TypeError, ValueError):
            return f"Exponential Rolling period {p!r} is invalid."
        if pi <= 0:
            return f"Exponential Rolling period {pi} must be a positive integer."
        pers.append(pi)
    if not pers:
        return "Exponential Rolling is enabled but no periods are selected."
    ops_in = operations if operations is not None else list(DEFAULT_EXPONENTIAL_OPS)
    ops: list[str] = []
    for op in ops_in:
        try:
            ops.append(normalize_exponential_rolling_op(op))
        except Exception as exc:
            return str(exc).split("\n")[-1] if str(exc) else f"Invalid operation {op!r}."
    if not ops:
        return "Exponential Rolling is enabled but no operations are selected."
    seen: set[str] = set()
    for f in feats:
        for p in pers:
            for op in ops:
                name = exponential_rolling_column_name(f, op, p)
                if name in seen:
                    return f"Duplicate Exponential Rolling output name: {name}"
                seen.add(name)
    if available_features is not None:
        avail = {str(a) for a in available_features}
        missing = [f for f in feats if f not in avail]
        if missing:
            return "Exponential Rolling features not in Master:\n" + "\n".join(missing[:8])
    return None


def format_exponential_rolling_preview_text(
    *,
    enabled: bool,
    feature_count: int,
    period_count: int,
    operation_count: int,
    columns_to_add: int,
    sample_names: list[str] | None = None,
) -> str:
    if not enabled:
        return "Exponential Rolling disabled — no extra columns."
    lines = [
        f"Features    : {feature_count}",
        f"Periods     : {period_count}",
        f"Operations  : {operation_count}",
        f"New Columns : {columns_to_add}",
    ]
    if sample_names:
        lines.append("Examples    : " + ", ".join(sample_names[:4]))
    return "\n".join(lines)
