"""Shared plan-time describe helpers for time-shift transforms."""

from __future__ import annotations

from typing import Any, Callable

from .describe import MASTER_STAGE_ID, OutputDescriptor, make_stage_descriptor
from .time_shift import LagConfigError, parse_features_and_horizons


def describe_time_shift_stage(
    transform,
    params: dict[str, Any] | None,
    *,
    sample_interval_sec: float | int | None,
    enabled: bool | None,
    kind: str,
    name_fn: Callable[..., str],
) -> Any:
    """Build StageDescriptor for Lag / Difference / Return-style transforms."""
    is_enabled = bool(transform.enabled if enabled is None else enabled)
    params = dict(params or {})
    interval = float(sample_interval_sec) if sample_interval_sec is not None else float(
        params.get("sample_interval_sec") or 3.0
    )
    outputs: list[OutputDescriptor] = []
    try:
        features, offsets = parse_features_and_horizons(
            transform_name=str(transform.name or transform.id),
            params=params,
            sample_interval_sec=interval,
        )
        for feat in features:
            for sec, _rows, suffix, column in offsets:
                if kind == "difference":
                    name = name_fn(feat, sec, suffix=suffix, column=column)
                elif kind == "return":
                    name = name_fn(feat, sec, suffix=suffix, column=column)
                elif kind == "difference_clip":
                    name = name_fn(
                        feat,
                        sec,
                        suffix=suffix,
                        column=column,
                        clip_min=float(params.get("clip_min", 0.0) or 0.0),
                    )
                else:
                    name = name_fn(feat, sec, suffix=suffix)
                outputs.append(
                    OutputDescriptor(
                        name=str(name),
                        kind=kind,
                        source_feature=str(feat),
                        op=kind,
                        meta={"seconds": float(sec)},
                    )
                )
    except (LagConfigError, Exception):
        outputs = []

    return make_stage_descriptor(
        transform,
        enabled=is_enabled,
        outputs=outputs,
        input_sources=[MASTER_STAGE_ID],
        notes=f"Planned {kind} columns from features × horizons.",
    )
