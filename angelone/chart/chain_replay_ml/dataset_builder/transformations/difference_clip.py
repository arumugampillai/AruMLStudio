"""Difference-clip transformation — clipped diffs (e.g. volume flow)."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from .base import FeatureTransformation, TransformationResult, TransformContext
from .registry import register_transformation
from .time_shift import (
    LagConfigError,
    parse_features_and_horizons,
    partition_columns,
    persist_sample_interval,
    resolve_sample_interval,
    resolve_transform_params,
    shift_feature_columns,
)


def difference_clip_column_name(
    feature: str,
    lag_seconds: int | float,
    *,
    suffix: str | None = None,
    column: str | None = None,
    clip_min: float = 0.0,
) -> str:
    if column is not None and str(column).strip():
        return str(column).strip()
    if abs(float(clip_min) - 0.0) < 1e-12:
        if suffix is not None and str(suffix).strip():
            return f"{feature}_flow_{str(suffix).strip()}"
        sec = int(lag_seconds) if float(lag_seconds).is_integer() else lag_seconds
        return f"{feature}_flow_{sec}s"
    if suffix is not None and str(suffix).strip():
        return f"{feature}_diffclip_{str(suffix).strip()}"
    sec = int(lag_seconds) if float(lag_seconds).is_integer() else lag_seconds
    return f"{feature}_diffclip_{sec}s"


@register_transformation
class DifferenceClipTransformation(FeatureTransformation):
    """``clip(x − x.shift(rows), lower, upper)`` on the sample grid."""

    id = "difference_clip"
    name = "Difference Clip"
    order = 25
    enabled = False
    depends_on: list[str] = []
    params: dict[str, Any] = {}

    def transform(self, df: pd.DataFrame, context: TransformContext) -> TransformationResult:
        t0 = time.perf_counter()
        params = resolve_transform_params(self.id, getattr(self, "params", None), context.config)
        interval = resolve_sample_interval(context, params)
        features, offsets = parse_features_and_horizons(
            transform_name="Difference Clip Transformation",
            params=params,
            sample_interval_sec=interval,
        )
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise LagConfigError(
                "Difference Clip Transformation\n"
                "Feature not found\n"
                + "\n".join(missing)
            )
        try:
            clip_min = float(params.get("clip_min", 0.0))
        except (TypeError, ValueError) as exc:
            raise LagConfigError(
                "Difference Clip Transformation\n"
                f"Invalid clip_min: {params.get('clip_min')!r}"
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
                    "Difference Clip Transformation\n"
                    f"Invalid clip_max: {clip_max_raw!r}"
                ) from exc
        persist_sample_interval(context, self.id, interval)
        partition_by = partition_columns(params, df)
        if context.cancelled():
            return TransformationResult.passthrough(
                df,
                transformation_id=self.id,
                transformation_name=self.name,
            )

        specs: list[tuple[str, int, str]] = []
        created: list[str] = []
        total = len(features) * len(offsets)
        for feat in features:
            for sec, rows, suffix, column in offsets:
                col = difference_clip_column_name(
                    feat,
                    sec,
                    suffix=suffix,
                    column=column,
                    clip_min=clip_min,
                )
                specs.append((feat, int(rows), col))
                created.append(col)

        def _pandas_fallback(frame: pd.DataFrame) -> pd.DataFrame:
            local = frame.copy()
            for feat, rows, col in specs:
                lagged = shift_feature_columns(
                    local, feature=feat, rows=rows, partition_by=partition_by
                )
                local[col] = (local[feat] - lagged).clip(lower=clip_min, upper=clip_max)
            return local

        from .polars_ops import apply_diff_clip_ops_via_polars

        out = apply_diff_clip_ops_via_polars(
            df,
            specs=specs,
            partition_by=partition_by,
            clip_min=clip_min,
            clip_max=clip_max,
            pandas_fallback=_pandas_fallback,
        )
        context.report_progress(f"Difference Clip: {total}/{total} columns", total, total)

        elapsed = round(max(time.perf_counter() - t0, 0.0), 4)
        lag_secs = [int(s) if float(s).is_integer() else s for s, _, _, _ in offsets]
        context.log("Difference Clip Transformation")
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
                f"Lag Seconds : {lag_secs}",
                f"Columns Created : {len(created)}",
                f"clip_min={clip_min}",
                f"clip_max={clip_max}",
                f"sample_interval_sec={interval}",
                f"partition_by={partition_by or []}",
                "frame_backend=polars_diff_clip",
            ],
        )
