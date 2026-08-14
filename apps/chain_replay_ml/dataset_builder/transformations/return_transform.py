"""Return transformation — percentage change vs lagged value over time horizons."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
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


def return_column_name(
    feature: str,
    lag_seconds: int | float,
    *,
    suffix: str | None = None,
    column: str | None = None,
) -> str:
    if column is not None and str(column).strip():
        return str(column).strip()
    if suffix is not None and str(suffix).strip():
        return f"{feature}_return_{str(suffix).strip()}"
    sec = int(lag_seconds) if float(lag_seconds).is_integer() else lag_seconds
    return f"{feature}_return_{sec}s"


@register_transformation
class ReturnTransformation(FeatureTransformation):
    """``feature_return_Ns = (feature − lag) / lag`` (row-shift on sample grid).

    Default: division by zero / missing lag → NaN (not zero).
    With ``params.denom_eps > 0``: denom = lagged + denom_eps (zero lag allowed).
    """

    id = "return"
    name = "Return"
    order = 30
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
        from .describe_helpers import describe_time_shift_stage

        del upstream, master_features
        return describe_time_shift_stage(
            self,
            params,
            sample_interval_sec=sample_interval_sec,
            enabled=enabled,
            kind="return",
            name_fn=return_column_name,
        )

    def transform(self, df: pd.DataFrame, context: TransformContext) -> TransformationResult:
        t0 = time.perf_counter()
        params = resolve_transform_params(self.id, getattr(self, "params", None), context.config)
        interval = resolve_sample_interval(context, params)
        features, offsets = parse_features_and_horizons(
            transform_name="Return Transformation",
            params=params,
            sample_interval_sec=interval,
        )
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise LagConfigError(
                "Return Transformation\n"
                "Feature not found\n"
                + "\n".join(missing)
            )
        persist_sample_interval(context, self.id, interval)
        partition_by = partition_columns(params, df)
        try:
            scale = float(params.get("scale", 1.0))
        except (TypeError, ValueError):
            scale = 1.0
        try:
            denom_eps = float(params.get("denom_eps", 0.0))
        except (TypeError, ValueError):
            denom_eps = 0.0
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
                col = return_column_name(feat, sec, suffix=suffix, column=column)
                specs.append((feat, int(rows), col))
                created.append(col)

        def _pandas_fallback(frame: pd.DataFrame) -> pd.DataFrame:
            local = frame.copy()
            new_cols: dict[str, pd.Series] = {}
            for feat, rows, col in specs:
                lagged = shift_feature_columns(
                    local, feature=feat, rows=rows, partition_by=partition_by
                )
                if denom_eps > 0:
                    series = (local[feat] - lagged) / (lagged + denom_eps)
                else:
                    denom = lagged.replace(0, np.nan)
                    series = (local[feat] - lagged) / denom
                if scale != 1.0:
                    series = series * scale
                new_cols[col] = series
            if not new_cols:
                return local
            return pd.concat(
                [local, pd.DataFrame(new_cols, index=local.index)],
                axis=1,
            )

        from .polars_ops import apply_return_ops_via_polars

        out = apply_return_ops_via_polars(
            df,
            specs=specs,
            partition_by=partition_by,
            scale=scale,
            denom_eps=denom_eps,
            pandas_fallback=_pandas_fallback,
        )
        context.report_progress(f"Return: {total}/{total} columns", total, total)

        elapsed = round(max(time.perf_counter() - t0, 0.0), 4)
        lag_secs = [int(s) if float(s).is_integer() else s for s, _, _, _ in offsets]
        context.log("Return Transformation")
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
                f"sample_interval_sec={interval}",
                f"partition_by={partition_by or []}",
                "frame_backend=polars_return",
            ],
        )
