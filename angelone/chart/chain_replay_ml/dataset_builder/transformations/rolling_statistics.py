"""Rolling statistics transformation — mean / std / z-score over row windows."""

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
)

_VALID_STATS = frozenset({"zscore", "mean", "std"})


def rolling_stat_column_name(feature: str, suffix: str, stat: str) -> str:
    st = str(stat or "zscore").strip().lower() or "zscore"
    suf = str(suffix or "").strip()
    return f"{feature}_{st}_{suf}" if suf else f"{feature}_{st}"


def _parse_windows(
    *,
    transform_name: str,
    params: dict[str, Any],
    sample_interval_sec: float,
) -> tuple[list[str], list[tuple[float, int, str | None, str | None]]]:
    """Reuse horizon parser with windows / window_seconds aliases."""
    adapted = dict(params)
    if adapted.get("horizons") is None and adapted.get("windows") is not None:
        adapted["horizons"] = adapted.get("windows")
    if (
        adapted.get("horizons") is None
        and adapted.get("lag_seconds") is None
        and adapted.get("window_seconds") is not None
    ):
        adapted["lag_seconds"] = adapted.get("window_seconds")
    return parse_features_and_horizons(
        transform_name=transform_name,
        params=adapted,
        sample_interval_sec=sample_interval_sec,
    )


@register_transformation
class RollingStatisticsTransformation(FeatureTransformation):
    """Rolling mean / std / z-score on the sample grid (exact row windows)."""

    id = "rolling_statistics"
    name = "Rolling Statistics"
    order = 40
    enabled = False
    depends_on: list[str] = []
    params: dict[str, Any] = {}

    def transform(self, df: pd.DataFrame, context: TransformContext) -> TransformationResult:
        t0 = time.perf_counter()
        params = resolve_transform_params(self.id, getattr(self, "params", None), context.config)
        interval = resolve_sample_interval(context, params)
        features, offsets = _parse_windows(
            transform_name="Rolling Statistics Transformation",
            params=params,
            sample_interval_sec=interval,
        )
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise LagConfigError(
                "Rolling Statistics Transformation\n"
                "Feature not found\n"
                + "\n".join(missing)
            )
        stat = str(params.get("stat") or "zscore").strip().lower() or "zscore"
        if stat not in _VALID_STATS:
            raise LagConfigError(
                "Rolling Statistics Transformation\n"
                f"Invalid stat={stat!r}. Expected one of {sorted(_VALID_STATS)}."
            )
        try:
            ddof = int(params.get("ddof", 0))
        except (TypeError, ValueError) as exc:
            raise LagConfigError(
                "Rolling Statistics Transformation\n"
                f"Invalid ddof: {params.get('ddof')!r}"
            ) from exc
        persist_sample_interval(context, self.id, interval)
        partition_by = partition_columns(params, df)
        if context.cancelled():
            return TransformationResult.passthrough(
                df,
                transformation_id=self.id,
                transformation_name=self.name,
            )

        specs: list[tuple[str, str, int, str]] = []
        created: list[str] = []
        for feat in features:
            for sec, rows, suffix, column in offsets:
                if column is not None and str(column).strip():
                    col = str(column).strip()
                else:
                    suf = (
                        str(suffix).strip()
                        if suffix is not None and str(suffix).strip()
                        else (
                            f"{int(sec)}s"
                            if float(sec).is_integer()
                            else f"{sec}s"
                        )
                    )
                    col = rolling_stat_column_name(feat, suf, stat)
                specs.append((feat, stat, int(rows), col))
                created.append(col)

        total = len(specs)

        def _pandas_fallback(frame: pd.DataFrame) -> pd.DataFrame:
            local = frame.copy()

            def _rolling(series: pd.Series, rows: int):
                return series.rolling(window=rows, min_periods=rows)

            for feat, st, rows, col in specs:
                if partition_by:
                    grouped = local.groupby(partition_by, sort=False, group_keys=False)[feat]
                    if st == "mean":
                        series = grouped.transform(lambda s, n=rows: _rolling(s, n).mean())
                    elif st == "std":
                        series = grouped.transform(
                            lambda s, n=rows, d=ddof: _rolling(s, n).std(ddof=d)
                        )
                    else:
                        mean = grouped.transform(lambda s, n=rows: _rolling(s, n).mean())
                        std = grouped.transform(
                            lambda s, n=rows, d=ddof: _rolling(s, n).std(ddof=d)
                        )
                        series = (local[feat] - mean) / std.replace(0, np.nan)
                else:
                    roll = _rolling(local[feat], rows)
                    if st == "mean":
                        series = roll.mean()
                    elif st == "std":
                        series = roll.std(ddof=ddof)
                    else:
                        mean = roll.mean()
                        std = roll.std(ddof=ddof)
                        series = (local[feat] - mean) / std.replace(0, np.nan)
                local[col] = series
            return local

        from .polars_ops import apply_rolling_stats_via_polars

        out = apply_rolling_stats_via_polars(
            df,
            specs=specs,
            ddof=ddof,
            partition_by=partition_by,
            pandas_fallback=_pandas_fallback,
        )
        context.report_progress(
            f"Rolling Statistics: {total}/{total} columns", total, total
        )

        elapsed = round(max(time.perf_counter() - t0, 0.0), 4)
        context.log("Rolling Statistics Transformation")
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
                f"stat={stat}",
                f"Columns Created : {len(created)}",
                f"sample_interval_sec={interval}",
                f"partition_by={partition_by or []}",
                "frame_backend=polars_rolling_stats",
            ],
        )
