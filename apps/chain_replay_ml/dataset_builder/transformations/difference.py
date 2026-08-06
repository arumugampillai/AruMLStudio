"""Difference transformation — current − lagged value over time horizons."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from .base import FeatureTransformation, TransformationResult, TransformContext
from .registry import register_transformation
from .time_shift import (
    LagConfigError,
    add_shifted_columns_via_polars,
    parse_features_and_horizons,
    partition_columns,
    persist_sample_interval,
    resolve_sample_interval,
    resolve_transform_params,
)


def difference_column_name(
    feature: str,
    lag_seconds: int | float,
    *,
    suffix: str | None = None,
    column: str | None = None,
) -> str:
    if column is not None and str(column).strip():
        return str(column).strip()
    if suffix is not None and str(suffix).strip():
        return f"{feature}_change_{str(suffix).strip()}"
    sec = int(lag_seconds) if float(lag_seconds).is_integer() else lag_seconds
    return f"{feature}_diff_{sec}s"


@register_transformation
class DifferenceTransformation(FeatureTransformation):
    """``feature_diff_Ns = feature − feature_lag_Ns`` (row-shift on sample grid)."""

    id = "difference"
    name = "Difference"
    order = 20
    enabled = False
    # Independent of Lag output columns — computes its own shifts.
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
            kind="difference",
            name_fn=difference_column_name,
        )

    def transform(self, df: pd.DataFrame, context: TransformContext) -> TransformationResult:
        t0 = time.perf_counter()
        params = resolve_transform_params(self.id, getattr(self, "params", None), context.config)
        interval = resolve_sample_interval(context, params)
        features, offsets = parse_features_and_horizons(
            transform_name="Difference Transformation",
            params=params,
            sample_interval_sec=interval,
        )
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise LagConfigError(
                "Difference Transformation\n"
                "Feature not found\n"
                + "\n".join(missing)
            )
        persist_sample_interval(context, self.id, interval)
        partition_by = partition_columns(params, df)
        if context.cancelled():
            return TransformationResult.passthrough(
                df,
                transformation_id=self.id,
                transformation_name=self.name,
            )

        # Build lagged columns via Polars, then difference in a second Polars pass.
        lag_specs: list[tuple[str, int, str]] = []
        diff_plan: list[tuple[str, str, str]] = []  # feat, lag_tmp, out_col
        created: list[str] = []
        total = len(features) * len(offsets)
        tmp_i = 0
        for feat in features:
            for sec, rows, suffix, column in offsets:
                tmp = f"__p2_lag_{tmp_i}"
                tmp_i += 1
                out_col = difference_column_name(feat, sec, suffix=suffix, column=column)
                lag_specs.append((feat, int(rows), tmp))
                diff_plan.append((feat, tmp, out_col))
                created.append(out_col)

        if context.cancelled():
            return TransformationResult.passthrough(
                df,
                transformation_id=self.id,
                transformation_name=self.name,
            )

        lagged = add_shifted_columns_via_polars(
            df, specs=lag_specs, partition_by=partition_by
        )
        # Prefer Polars for current − lagged; fall back to pandas.
        try:
            from chain_replay_ml.frame_backend import (
                arrow_table_to_polars,
                polars_to_pandas,
                require_polars,
            )
            import pyarrow as pa

            pl = require_polars()
            pl_df = arrow_table_to_polars(pa.Table.from_pandas(lagged, preserve_index=False))
            exprs = [
                (pl.col(feat) - pl.col(tmp)).alias(out_col)
                for feat, tmp, out_col in diff_plan
            ]
            drop_tmp = [tmp for _, tmp, _ in diff_plan]
            pl_df = pl_df.with_columns(exprs).drop(drop_tmp)
            out = polars_to_pandas(pl_df)
        except Exception:
            out = lagged.copy()
            for feat, tmp, out_col in diff_plan:
                out[out_col] = out[feat] - out[tmp]
            out = out.drop(columns=[tmp for _, tmp, _ in diff_plan], errors="ignore")

        context.report_progress(f"Difference: {total}/{total} columns", total, total)

        elapsed = round(max(time.perf_counter() - t0, 0.0), 4)
        lag_secs = [int(s) if float(s).is_integer() else s for s, _, _, _ in offsets]
        context.log("Difference Transformation")
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
                "frame_backend=polars_shift",
            ],
        )
