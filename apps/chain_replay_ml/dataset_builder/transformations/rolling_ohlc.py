"""Rolling OHLC-style transforms over row windows on a sampled series."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from .base import FeatureTransformation, TransformationResult, TransformContext
from .registry import register_transformation
from .rolling_statistics import _parse_windows
from .time_shift import (
    LagConfigError,
    partition_columns,
    persist_sample_interval,
    resolve_sample_interval,
    resolve_transform_params,
    shift_feature_columns,
)

_ALL_OUTPUTS = (
    "body_pct",
    "range_pct",
    "dist_high_pct",
    "dist_low_pct",
    "range_pos",
)
_VALID_OUTPUTS = frozenset(_ALL_OUTPUTS)


def rolling_ohlc_column_name(feature: str, output: str, suffix: str) -> str:
    suf = str(suffix or "").strip()
    out = str(output or "").strip()
    return f"{feature}_{out}_{suf}" if suf else f"{feature}_{out}"


def _resolve_outputs(params: dict[str, Any]) -> list[str]:
    raw = params.get("outputs")
    if raw is None:
        return list(_ALL_OUTPUTS)
    outs = [str(o).strip() for o in raw if str(o).strip()]
    if not outs:
        return list(_ALL_OUTPUTS)
    bad = [o for o in outs if o not in _VALID_OUTPUTS]
    if bad:
        raise LagConfigError(
            "Rolling OHLC Transformation\n"
            f"Invalid outputs: {bad}. Expected subset of {sorted(_VALID_OUTPUTS)}."
        )
    return outs


def _column_for(
    *,
    feature: str,
    output: str,
    suffix: str | None,
    sec: float,
    column_override: Any,
    single_output: bool = False,
) -> str:
    if isinstance(column_override, dict):
        mapped = column_override.get(output)
        if mapped is not None and str(mapped).strip():
            return str(mapped).strip()
    elif (
        column_override is not None
        and not isinstance(column_override, dict)
        and str(column_override).strip()
        and single_output
    ):
        return str(column_override).strip()
    suf = (
        str(suffix).strip()
        if suffix is not None and str(suffix).strip()
        else (f"{int(sec)}s" if float(sec).is_integer() else f"{sec}s")
    )
    return rolling_ohlc_column_name(feature, output, suf)


@register_transformation
class RollingOhlcTransformation(FeatureTransformation):
    """Body / range / distance / range_pos over rolling windows of the series."""

    id = "rolling_ohlc"
    name = "Rolling OHLC"
    order = 45
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
        from .describe import MASTER_STAGE_ID, OutputDescriptor, make_stage_descriptor
        from .rolling_statistics import _parse_windows

        del upstream, master_features
        is_enabled = bool(self.enabled if enabled is None else enabled)
        params = dict(params or {})
        interval = float(sample_interval_sec) if sample_interval_sec is not None else float(
            params.get("sample_interval_sec") or 3.0
        )
        outputs: list[OutputDescriptor] = []
        try:
            features, offsets = _parse_windows(
                transform_name="Rolling OHLC Transformation",
                params=params,
                sample_interval_sec=interval,
            )
            output_names = _resolve_outputs(params)
            top_column_map = params.get("column_map")
            if not isinstance(top_column_map, dict):
                top_column_map = None
            single_output = len(output_names) == 1
            for feat in features:
                for sec, _rows, suffix, column in offsets:
                    override: Any = column
                    if top_column_map is not None:
                        if isinstance(column, dict):
                            override = {**top_column_map, **column}
                        elif column is None:
                            override = top_column_map
                    for output_name in output_names:
                        col = _column_for(
                            feature=feat,
                            output=output_name,
                            suffix=suffix,
                            sec=sec,
                            column_override=override,
                            single_output=single_output,
                        )
                        outputs.append(
                            OutputDescriptor(
                                name=str(col),
                                kind="rolling_ohlc",
                                source_feature=str(feat),
                                op=output_name,
                                meta={"seconds": float(sec)},
                            )
                        )
        except (LagConfigError, Exception):
            outputs = []

        return make_stage_descriptor(
            self,
            enabled=is_enabled,
            outputs=outputs,
            input_sources=[MASTER_STAGE_ID],
            notes="Planned rolling OHLC columns from features × windows × outputs.",
        )

    def transform(self, df: pd.DataFrame, context: TransformContext) -> TransformationResult:
        t0 = time.perf_counter()
        params = resolve_transform_params(self.id, getattr(self, "params", None), context.config)
        interval = resolve_sample_interval(context, params)
        features, offsets = _parse_windows(
            transform_name="Rolling OHLC Transformation",
            params=params,
            sample_interval_sec=interval,
        )
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise LagConfigError(
                "Rolling OHLC Transformation\n"
                "Feature not found\n"
                + "\n".join(missing)
            )
        outputs = _resolve_outputs(params)
        try:
            range_eps = float(params.get("range_eps", 1e-9))
        except (TypeError, ValueError):
            range_eps = 1e-9
        # Optional top-level output→name map (merged under per-window column dict).
        top_column_map = params.get("column_map")
        if not isinstance(top_column_map, dict):
            top_column_map = None
        persist_sample_interval(context, self.id, interval)
        partition_by = partition_columns(params, df)
        if context.cancelled():
            return TransformationResult.passthrough(
                df,
                transformation_id=self.id,
                transformation_name=self.name,
            )

        jobs: list[dict[str, Any]] = []
        created: list[str] = []
        single_output = len(outputs) == 1
        for feat in features:
            for sec, rows, suffix, column in offsets:
                override: Any = column
                if top_column_map is not None:
                    if isinstance(column, dict):
                        override = {**top_column_map, **column}
                    elif column is None:
                        override = top_column_map
                out_map: dict[str, str] = {}
                for output_name in outputs:
                    col = _column_for(
                        feature=feat,
                        output=output_name,
                        suffix=suffix,
                        sec=sec,
                        column_override=override,
                        single_output=single_output,
                    )
                    out_map[output_name] = col
                    created.append(col)
                jobs.append({"feature": feat, "rows": int(rows), "outputs": out_map})

        total = len(created)

        def _pandas_fallback(frame: pd.DataFrame) -> pd.DataFrame:
            local = frame.copy()
            for job in jobs:
                feat = job["feature"]
                rows = int(job["rows"])
                close = local[feat]
                open_ = shift_feature_columns(
                    local, feature=feat, rows=max(rows - 1, 0), partition_by=partition_by
                )
                if partition_by:
                    grouped = local.groupby(partition_by, sort=False, group_keys=False)[feat]
                    high = grouped.transform(
                        lambda s, n=rows: s.rolling(window=n, min_periods=n).max()
                    )
                    low = grouped.transform(
                        lambda s, n=rows: s.rolling(window=n, min_periods=n).min()
                    )
                else:
                    roll = local[feat].rolling(window=rows, min_periods=rows)
                    high = roll.max()
                    low = roll.min()
                open_safe = open_.replace(0, np.nan)
                high_safe = high.replace(0, np.nan)
                low_safe = low.replace(0, np.nan)
                range_span = high - low + range_eps
                computed = {
                    "body_pct": (close - open_) / open_safe * 100.0,
                    "range_pct": (high - low) / open_safe * 100.0,
                    "dist_high_pct": (close - high) / high_safe * 100.0,
                    "dist_low_pct": (close - low) / low_safe * 100.0,
                    "range_pos": (close - low) / range_span,
                }
                for out_name, col in job["outputs"].items():
                    local[col] = computed[out_name]
            return local

        from .polars_ops import apply_rolling_ohlc_via_polars

        out = apply_rolling_ohlc_via_polars(
            df,
            jobs=jobs,
            partition_by=partition_by,
            range_eps=range_eps,
            pandas_fallback=_pandas_fallback,
        )
        context.report_progress(f"Rolling OHLC: {total}/{total} columns", total, total)

        elapsed = round(max(time.perf_counter() - t0, 0.0), 4)
        context.log("Rolling OHLC Transformation")
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
                f"outputs={outputs}",
                f"Columns Created : {len(created)}",
                f"sample_interval_sec={interval}",
                f"partition_by={partition_by or []}",
                "frame_backend=polars_rolling_ohlc",
            ],
        )
