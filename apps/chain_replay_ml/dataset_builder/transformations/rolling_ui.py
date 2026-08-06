"""Rolling transform UI helpers — windows, operations, config, preview."""

from __future__ import annotations

from typing import Any

from .rolling import ROLLING_OPS, normalize_rolling_op, rolling_column_name

DEFAULT_ROLLING_WINDOWS: tuple[int, ...] = (5, 10, 20, 50)
DEFAULT_ROLLING_OPS: tuple[str, ...] = ("mean", "std")

OP_DISPLAY_LABELS: dict[str, str] = {
    "mean": "Mean",
    "std": "Standard Deviation",
    "min": "Minimum",
    "max": "Maximum",
    "median": "Median",
}


def display_op_label(op: str) -> str:
    try:
        return OP_DISPLAY_LABELS[normalize_rolling_op(op)]
    except Exception:
        return str(op or "")


def build_rolling_transformation_config(
    *,
    enabled: bool,
    features: list[str],
    windows: list[int],
    operations: list[str],
    partition_by: list[str] | None = None,
    sample_interval_sec: float | int | None = None,
    ddof: int = 0,
    order: int = 40,
) -> dict[str, Any]:
    """Build a pipeline config fragment for the Rolling Statistics transform."""
    feats = [str(f).strip() for f in features if str(f).strip()]
    wins = sorted({int(w) for w in windows if int(w) > 0})
    ops = []
    for op in operations:
        try:
            n = normalize_rolling_op(op)
        except Exception:
            continue
        if n not in ops:
            ops.append(n)
    entry: dict[str, Any] = {
        "id": "rolling",
        "enabled": bool(enabled) and bool(feats) and bool(wins) and bool(ops),
        "order": int(order),
        "name": "Rolling Statistics",
        "params": {
            "features": feats,
            "windows": wins,
            "operations": ops,
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


def merge_rolling_into_config(
    base: dict[str, Any] | None,
    *,
    enabled: bool,
    features: list[str],
    windows: list[int],
    operations: list[str],
    partition_by: list[str] | None = None,
    sample_interval_sec: float | int | None = None,
    ddof: int = 0,
) -> dict[str, Any]:
    """Merge Rolling Statistics into an existing pipeline config (replaces any prior rolling entry)."""
    from .config import normalize_transformation_config

    cfg = normalize_transformation_config(base)
    transforms = [
        t for t in (cfg.get("transformations") or [])
        if isinstance(t, dict) and str(t.get("id") or "") != "rolling"
    ]
    rolling_cfg = build_rolling_transformation_config(
        enabled=enabled,
        features=features,
        windows=windows,
        operations=operations,
        partition_by=partition_by,
        sample_interval_sec=sample_interval_sec,
        ddof=ddof,
    )
    for entry in rolling_cfg.get("transformations") or []:
        transforms.append(entry)
    cfg["transformations"] = transforms
    return cfg


def rolling_column_count(
    *,
    enabled: bool,
    features: list[str],
    windows: list[int],
    operations: list[str],
) -> int:
    if not enabled:
        return 0
    n_f = len([f for f in features if str(f).strip()])
    n_w = len([int(w) for w in windows if int(w) > 0])
    n_o = 0
    for op in operations:
        try:
            normalize_rolling_op(op)
            n_o += 1
        except Exception:
            continue
    return n_f * n_w * n_o


def preview_rolling_columns(
    *,
    features: list[str],
    windows: list[int],
    operations: list[str],
    limit: int = 12,
) -> list[str]:
    cols: list[str] = []
    for feat in features:
        f = str(feat).strip()
        if not f:
            continue
        for win in windows:
            try:
                w = int(win)
            except (TypeError, ValueError):
                continue
            if w <= 0:
                continue
            for op in operations:
                try:
                    cols.append(rolling_column_name(f, op, w))
                except Exception:
                    continue
                if len(cols) >= limit:
                    return cols
    return cols


def validate_rolling_settings(
    *,
    enabled: bool,
    features: list[str],
    windows: list[int],
    operations: list[str],
    available_features: list[str] | None = None,
) -> str | None:
    if not enabled:
        return None
    feats = [str(f).strip() for f in features if str(f).strip()]
    if not feats:
        return "Rolling Statistics is enabled but no features are selected."
    wins = []
    for w in windows:
        try:
            wi = int(w)
        except (TypeError, ValueError):
            return f"Rolling Statistics window {w!r} is invalid."
        if wi <= 0:
            return f"Rolling Statistics window {wi} must be a positive integer."
        wins.append(wi)
    if not wins:
        return "Rolling Statistics is enabled but no windows are selected."
    ops: list[str] = []
    for op in operations:
        try:
            ops.append(normalize_rolling_op(op))
        except Exception:
            return f"Rolling Statistics operation {op!r} is not supported."
    if not ops:
        return "Rolling Statistics is enabled but no operations are selected."
    if available_features is not None:
        avail = {str(a) for a in available_features}
        missing = [f for f in feats if f not in avail]
        if missing:
            return "Rolling Statistics features not in Master:\n" + "\n".join(missing[:8])
    return None


def format_rolling_preview_text(
    *,
    enabled: bool,
    feature_count: int,
    window_count: int,
    operation_count: int,
    columns_to_add: int,
    sample_names: list[str] | None = None,
) -> str:
    if not enabled:
        return "Rolling Statistics disabled — no extra columns."
    lines = [
        f"Features    : {feature_count}",
        f"Windows     : {window_count}",
        f"Operations  : {operation_count}",
        f"New Columns : {columns_to_add}",
    ]
    if sample_names:
        lines.append("Examples    : " + ", ".join(sample_names[:4]))
    return "\n".join(lines)
