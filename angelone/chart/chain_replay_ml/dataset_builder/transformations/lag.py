"""Lag transformation — row-based lags on the sample grid.

Horizons must be exact multiples of ``sample_interval_sec``. Calendar-time
lookups are intentionally not supported.

Config example (under transformations[]):
{
  "id": "lag",
  "enabled": true,
  "params": {
    "features": ["ltp_to_spot_ratio"],
    "horizons": [
      {"seconds": 30, "suffix": "30s"},
      {"seconds": 60, "suffix": "1m"}
    ],
    "sample_interval_sec": 3,
    "partition_by": ["trading_day", "token"]
  }
}

Or with ``lag_seconds: [30, 60]`` → columns ``{feature}_lag_30s``, ``{feature}_lag_60s``.

When ``lag_seconds`` / ``horizons`` are omitted, horizons are derived from the
shared ``horizon_policy.json`` for ``sample_interval_sec``.

Row offsets are ``lag_seconds / sample_interval_sec`` (exact multiples only).
"""

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
    resolve_lag_row_offsets,
    resolve_sample_interval,
    resolve_transform_params,
)

# Re-export for existing imports.
__all__ = [
    "LagConfigError",
    "LagTransformation",
    "lag_column_name",
    "resolve_lag_row_offsets",
]


def lag_column_name(
    feature: str,
    lag_seconds: int | float,
    *,
    suffix: str | None = None,
) -> str:
    """Lag column name.

    Default: ``{feature}_lag_{Ns}`` (e.g. ``ltp_lag_60s``).
    With ``suffix``: Master-compatible name (e.g. ``ltp_to_spot_ratio_lag_1m``).
    """
    if suffix is not None and str(suffix).strip():
        return f"{feature}_lag_{str(suffix).strip()}"
    sec = int(lag_seconds) if float(lag_seconds).is_integer() else lag_seconds
    return f"{feature}_lag_{sec}s"


@register_transformation
class LagTransformation(FeatureTransformation):
    id = "lag"
    name = "Lag"
    order = 10
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
            kind="lag",
            name_fn=lag_column_name,
        )

    def transform(self, df: pd.DataFrame, context: TransformContext) -> TransformationResult:
        t0 = time.perf_counter()
        params = resolve_transform_params(self.id, getattr(self, "params", None), context.config)
        interval = resolve_sample_interval(context, params)
        features, offsets = parse_features_and_horizons(
            transform_name="Lag Transformation",
            params=params,
            sample_interval_sec=interval,
        )
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise LagConfigError(
                "Lag Transformation\n"
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

        created: list[str] = []
        total = len(features) * len(offsets)
        specs: list[tuple[str, int, str]] = []
        for feat in features:
            for sec, rows, suffix, _column in offsets:
                col = lag_column_name(feat, sec, suffix=suffix)
                specs.append((feat, int(rows), col))
                created.append(col)

        out = add_shifted_columns_via_polars(
            df, specs=specs, partition_by=partition_by
        )
        context.report_progress(f"Lag: {total}/{total} columns", total, total)

        elapsed = round(max(time.perf_counter() - t0, 0.0), 4)
        lag_secs = [int(s) if float(s).is_integer() else s for s, _, _, _ in offsets]
        context.log("Lag Transformation")
        context.log(f"Features Selected : {len(features)}")
        context.log("Lag Seconds")
        for sec in lag_secs:
            context.log(str(sec))
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
