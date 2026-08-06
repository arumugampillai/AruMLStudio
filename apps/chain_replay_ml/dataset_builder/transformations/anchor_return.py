"""Anchor-return transformation — percent change vs a partition anchor."""

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


def anchor_return_column_name(
    feature: str,
    *,
    column: str | None = None,
    column_suffix: str | None = None,
) -> str:
    if column is not None and str(column).strip():
        return str(column).strip()
    if column_suffix is not None and str(column_suffix).strip():
        return f"{feature}_{str(column_suffix).strip()}"
    return f"{feature}_pct_change_from_open"


def _first_valid(series: pd.Series) -> float:
    for val in series:
        if pd.notna(val):
            return float(val)
    return float("nan")


@register_transformation
class AnchorReturnTransformation(FeatureTransformation):
    """``scale * (x − anchor) / anchor`` vs first valid value per partition."""

    id = "anchor_return"
    name = "Anchor Return"
    order = 35
    enabled = False
    depends_on: list[str] = []
    params: dict[str, Any] = {}

    def transform(self, df: pd.DataFrame, context: TransformContext) -> TransformationResult:
        t0 = time.perf_counter()
        params = resolve_transform_params(self.id, getattr(self, "params", None), context.config)
        interval = resolve_sample_interval(context, params)
        anchor_mode = str(params.get("anchor") or "first").strip().lower() or "first"
        if anchor_mode != "first":
            raise LagConfigError(
                "Anchor Return Transformation\n"
                f"Unsupported anchor={anchor_mode!r}. Only 'first' is implemented."
            )
        try:
            scale = float(params.get("scale", 100.0))
        except (TypeError, ValueError):
            scale = 100.0

        outputs_raw = params.get("outputs")
        specs: list[tuple[str, str]] = []
        if isinstance(outputs_raw, list) and outputs_raw:
            for item in outputs_raw:
                if not isinstance(item, dict):
                    raise LagConfigError(
                        "Anchor Return Transformation\n"
                        f"Invalid outputs entry: {item!r}"
                    )
                feature = str(item.get("feature") or "").strip()
                column = str(item.get("column") or "").strip()
                if not feature:
                    raise LagConfigError(
                        "Anchor Return Transformation\n"
                        "outputs[].feature is required."
                    )
                if not column:
                    column = anchor_return_column_name(feature)
                specs.append((feature, column))
        else:
            features = [str(f).strip() for f in (params.get("features") or []) if str(f).strip()]
            if not features:
                raise LagConfigError(
                    "Anchor Return Transformation\n"
                    "params.features / params.outputs is empty."
                )
            column_suffix = params.get("column_suffix")
            for feature in features:
                specs.append((
                    feature,
                    anchor_return_column_name(feature, column_suffix=column_suffix),
                ))

        missing = sorted({f for f, _ in specs if f not in df.columns})
        if missing:
            raise LagConfigError(
                "Anchor Return Transformation\n"
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

        created = [col for _, col in specs]
        total = len(specs)

        def _pandas_fallback(frame: pd.DataFrame) -> pd.DataFrame:
            local = frame.copy()
            for feat, col in specs:
                if partition_by:
                    anchors = local.groupby(partition_by, sort=False, group_keys=False)[feat].transform(
                        _first_valid
                    )
                else:
                    anchor_val = _first_valid(local[feat])
                    anchors = pd.Series(anchor_val, index=local.index, dtype=float)
                denom = anchors.where(anchors > 0, np.nan)
                local[col] = scale * (local[feat] - anchors) / denom
            return local

        from .polars_ops import apply_anchor_return_via_polars

        out = apply_anchor_return_via_polars(
            df,
            specs=specs,
            partition_by=partition_by,
            scale=scale,
            pandas_fallback=_pandas_fallback,
        )
        context.report_progress(f"Anchor Return: {total}/{total} columns", total, total)

        elapsed = round(max(time.perf_counter() - t0, 0.0), 4)
        context.log("Anchor Return Transformation")
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
                f"Outputs : {len(created)}",
                f"Columns Created : {len(created)}",
                f"scale={scale}",
                f"sample_interval_sec={interval}",
                f"partition_by={partition_by or []}",
                "frame_backend=polars_anchor",
            ],
        )
