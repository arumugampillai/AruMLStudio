"""Math (Unary) transform UI helpers."""

from __future__ import annotations

from typing import Any

from .math_transform import MATH_OPS, math_column_name, normalize_math_op

DEFAULT_MATH_OPS: tuple[str, ...] = ("abs", "log", "clip")

OP_DISPLAY_LABELS: dict[str, str] = {
    "abs": "Abs",
    "log": "Log",
    "sqrt": "Sqrt",
    "square": "Square",
    "cube": "Cube",
    "clip": "Clip",
    "sign": "Sign",
    "negate": "Negate",
}


def build_math_transformation_config(
    *,
    enabled: bool,
    features: list[str],
    operations: list[str],
    clip_min: float = 0.0,
    clip_max: float | None = None,
    partition_by: list[str] | None = None,
    sample_interval_sec: float | int | None = None,
    order: int = 70,
) -> dict[str, Any]:
    feats = [str(f).strip() for f in features if str(f).strip()]
    ops: list[str] = []
    for op in operations:
        try:
            n = normalize_math_op(op)
        except Exception:
            continue
        if n not in ops:
            ops.append(n)
    entry: dict[str, Any] = {
        "id": "math",
        "enabled": bool(enabled) and bool(feats) and bool(ops),
        "order": int(order),
        "name": "Math (Unary)",
        "params": {
            "features": feats,
            "operations": ops,
            "clip_min": float(clip_min),
            "clip_max": clip_max,
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


def merge_math_into_config(
    base: dict[str, Any] | None,
    *,
    enabled: bool,
    features: list[str],
    operations: list[str],
    clip_min: float = 0.0,
    clip_max: float | None = None,
    partition_by: list[str] | None = None,
    sample_interval_sec: float | int | None = None,
) -> dict[str, Any]:
    from .config import normalize_transformation_config

    cfg = normalize_transformation_config(base)
    transforms = [
        t for t in (cfg.get("transformations") or [])
        if isinstance(t, dict) and str(t.get("id") or "") != "math"
    ]
    math_cfg = build_math_transformation_config(
        enabled=enabled,
        features=features,
        operations=operations,
        clip_min=clip_min,
        clip_max=clip_max,
        partition_by=partition_by,
        sample_interval_sec=sample_interval_sec,
    )
    for entry in math_cfg.get("transformations") or []:
        transforms.append(entry)
    cfg["transformations"] = transforms
    return cfg


def math_column_count(*, enabled: bool, features: list[str], operations: list[str]) -> int:
    if not enabled:
        return 0
    n_f = len([f for f in features if str(f).strip()])
    n_o = 0
    for op in operations:
        try:
            normalize_math_op(op)
            n_o += 1
        except Exception:
            continue
    return n_f * n_o


def preview_math_columns(
    *,
    features: list[str],
    operations: list[str],
    clip_min: float = 0.0,
    clip_max: float | None = None,
    limit: int = 12,
) -> list[str]:
    cols: list[str] = []
    for feat in features:
        f = str(feat).strip()
        if not f:
            continue
        for op in operations:
            try:
                cols.append(
                    math_column_name(f, op, clip_min=clip_min, clip_max=clip_max)
                )
            except Exception:
                continue
            if len(cols) >= limit:
                return cols
    return cols


def validate_math_settings(
    *,
    enabled: bool,
    features: list[str],
    operations: list[str],
    available_features: list[str] | None = None,
) -> str | None:
    if not enabled:
        return None
    feats = [str(f).strip() for f in features if str(f).strip()]
    if not feats:
        return "Math is enabled but no features are selected."
    ops: list[str] = []
    for op in operations:
        try:
            ops.append(normalize_math_op(op))
        except Exception:
            return f"Math operation {op!r} is not supported."
    if not ops:
        return "Math is enabled but no operations are selected."
    if available_features is not None:
        avail = {str(a) for a in available_features}
        missing = [f for f in feats if f not in avail]
        if missing:
            return "Math features not in Master:\n" + "\n".join(missing[:8])
    return None


def format_math_preview_text(
    *,
    enabled: bool,
    feature_count: int,
    operation_count: int,
    columns_to_add: int,
    sample_names: list[str] | None = None,
) -> str:
    if not enabled:
        return "Math disabled — no extra columns."
    lines = [
        f"Features    : {feature_count}",
        f"Operations  : {operation_count}",
        f"New Columns : {columns_to_add}",
    ]
    if sample_names:
        lines.append("Examples    : " + ", ".join(sample_names[:4]))
    return "\n".join(lines)


__all__ = [
    "DEFAULT_MATH_OPS",
    "MATH_OPS",
    "OP_DISPLAY_LABELS",
    "build_math_transformation_config",
    "format_math_preview_text",
    "math_column_count",
    "merge_math_into_config",
    "preview_math_columns",
    "validate_math_settings",
]
