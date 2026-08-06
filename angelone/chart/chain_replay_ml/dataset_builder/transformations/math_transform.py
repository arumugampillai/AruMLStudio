"""Math (Unary) transformation — pointwise one-input → one-output.

No windows. No pairwise interactions. Pipeline-only; disabled by default.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from .base import FeatureTransformation, TransformationResult, TransformContext
from .registry import register_transformation
from .time_shift import (
    LagConfigError,
    partition_columns,
    persist_sample_interval,
    resolve_sample_interval,
    resolve_transform_params,
)

MATH_OPS: tuple[str, ...] = (
    "abs",
    "log",
    "sqrt",
    "square",
    "cube",
    "clip",
    "sign",
    "negate",
)
_VALID_OPS = frozenset(MATH_OPS)

_OP_ALIASES: dict[str, str] = {
    "abs": "abs",
    "absolute": "abs",
    "log": "log",
    "ln": "log",
    "log_e": "log",
    "sqrt": "sqrt",
    "square": "square",
    "sq": "square",
    "cube": "cube",
    "clip": "clip",
    "sign": "sign",
    "sgn": "sign",
    "negate": "negate",
    "neg": "negate",
    "negative": "negate",
}


def normalize_math_op(op: str) -> str:
    key = str(op or "").strip().lower()
    if key not in _OP_ALIASES:
        raise LagConfigError(
            "Math Transformation\n"
            f"Invalid operation={op!r}. Expected one of {list(MATH_OPS)}."
        )
    return _OP_ALIASES[key]


def math_column_name(
    feature: str,
    op: str,
    *,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> str:
    feat = str(feature or "").strip()
    operation = normalize_math_op(op)
    if operation == "clip":
        lo = 0.0 if clip_min is None else float(clip_min)
        hi = clip_max
        if hi is None:
            return f"{feat}_clip_{lo:g}"
        return f"{feat}_clip_{lo:g}_{float(hi):g}"
    return f"{feat}_{operation}"


def _parse_features(params: dict[str, Any]) -> list[str]:
    raw = params.get("features")
    if raw is None:
        raw = params.get("feature")
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    out: list[str] = []
    for item in raw:
        name = str(item or "").strip()
        if name and name not in out:
            out.append(name)
    return out


def _parse_operations(params: dict[str, Any]) -> list[str]:
    raw = params.get("operations")
    if raw is None:
        raw = params.get("ops")
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    out: list[str] = []
    for item in raw:
        op = normalize_math_op(str(item))
        if op not in out:
            out.append(op)
    return out


def _apply_op(
    series: pd.Series,
    op: str,
    *,
    clip_min: float,
    clip_max: float | None,
) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    if op == "abs":
        return values.abs()
    if op == "log":
        return np.log(values.where(values > 0.0))
    if op == "sqrt":
        return np.sqrt(values.where(values >= 0.0))
    if op == "square":
        return values * values
    if op == "cube":
        return values * values * values
    if op == "clip":
        return values.clip(lower=clip_min, upper=clip_max)
    if op == "sign":
        return np.sign(values)
    if op == "negate":
        return -values
    raise LagConfigError(f"Math Transformation\nUnsupported op={op!r}")


@register_transformation
class MathTransformation(FeatureTransformation):
    """Pointwise unary math on Master / prior transform columns."""

    id = "math"
    name = "Math (Unary)"
    order = 70  # After Interaction (50); before Normalization (75)
    enabled = False
    depends_on: list[str] = []
    params: dict[str, Any] = {}

    def describe(
        self,
        params: dict[str, Any] | None = None,
        *,
        upstream=None,
        master_features: list[str] | None = None,
        sample_interval_sec: float | int | None = None,
        enabled: bool | None = None,
    ):
        from .describe import OutputDescriptor, make_stage_descriptor

        del upstream, master_features, sample_interval_sec
        is_enabled = bool(self.enabled if enabled is None else enabled)
        params = dict(params or {})
        outputs: list[OutputDescriptor] = []
        try:
            features = _parse_features(params)
            operations = _parse_operations(params)
            clip_min = float(params.get("clip_min", 0.0) or 0.0)
            clip_max_raw = params.get("clip_max", None)
            clip_max = None if clip_max_raw is None else float(clip_max_raw)
            for feat in features:
                for op in operations:
                    outputs.append(
                        OutputDescriptor(
                            name=math_column_name(
                                feat, op, clip_min=clip_min, clip_max=clip_max
                            ),
                            kind="math",
                        )
                    )
        except Exception:
            outputs = []
        return make_stage_descriptor(
            self,
            enabled=is_enabled,
            outputs=outputs,
            notes="Pointwise unary transforms (no windows).",
        )

    def transform(self, df: pd.DataFrame, context: TransformContext) -> TransformationResult:
        t0 = time.perf_counter()
        params = resolve_transform_params(self.id, getattr(self, "params", None), context.config)
        interval = resolve_sample_interval(context, params)
        features = _parse_features(params)
        operations = _parse_operations(params)
        if not features or not operations:
            raise LagConfigError(
                "Math Transformation\n"
                "Requires non-empty features and operations."
            )
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise LagConfigError(
                "Math Transformation\nFeature not found\n" + "\n".join(missing)
            )
        try:
            clip_min = float(params.get("clip_min", 0.0) or 0.0)
        except (TypeError, ValueError) as exc:
            raise LagConfigError(
                f"Math Transformation\nInvalid clip_min: {params.get('clip_min')!r}"
            ) from exc
        clip_max_raw = params.get("clip_max", None)
        clip_max: float | None
        if clip_max_raw is None:
            clip_max = None
        else:
            try:
                clip_max = float(clip_max_raw)
            except (TypeError, ValueError) as exc:
                raise LagConfigError(
                    f"Math Transformation\nInvalid clip_max: {clip_max_raw!r}"
                ) from exc

        persist_sample_interval(context, self.id, interval)
        # Unary math is pointwise; partition_by is unused but accepted for uniformity.
        _ = partition_columns(params, df)
        if context.cancelled():
            return TransformationResult.passthrough(
                df,
                transformation_id=self.id,
                transformation_name=self.name,
            )

        specs: list[tuple[str, str, str]] = []
        created: list[str] = []
        total = len(features) * len(operations)
        for feat in features:
            for op in operations:
                col = math_column_name(feat, op, clip_min=clip_min, clip_max=clip_max)
                specs.append((feat, op, col))
                created.append(col)

        def _pandas_fallback(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            for feat, op, col in specs:
                out[col] = _apply_op(out[feat], op, clip_min=clip_min, clip_max=clip_max)
            return out

        from .polars_ops import apply_math_ops_via_polars

        out = apply_math_ops_via_polars(
            df,
            specs=specs,
            clip_min=clip_min,
            clip_max=clip_max,
            pandas_fallback=_pandas_fallback,
        )
        context.report_progress(f"Math: {total}/{total} columns", total, total)

        elapsed = round(max(time.perf_counter() - t0, 0.0), 4)
        context.log("Math (Unary) Transformation")
        context.log(f"Features Selected : {len(features)}")
        context.log(f"Columns Created : {len(created)}")
        context.log(f"Elapsed : {elapsed:.2f} s")
        return TransformationResult(
            frame=out,
            created_columns=created,
            elapsed_sec=elapsed,
            rows_processed=int(len(out)),
            transformation_id=self.id,
            transformation_name=self.name,
            messages=[
                f"Features Selected : {len(features)}",
                f"operations={operations}",
                f"Columns Created : {len(created)}",
                "frame_backend=polars_math",
            ],
        )


__all__ = [
    "MATH_OPS",
    "MathTransformation",
    "math_column_name",
    "normalize_math_op",
]
