"""Derived transformation — weighted-lag linear combinations (exact row algebra)."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd

from .base import FeatureTransformation, TransformationResult, TransformContext
from .registry import register_transformation
from .time_shift import (
    LagConfigError,
    partition_columns,
    persist_sample_interval,
    resolve_sample_interval,
    resolve_transform_params,
    shift_feature_columns,
)


def _resolve_term_rows(seconds: float, sample_interval_sec: float) -> int:
    try:
        interval = float(sample_interval_sec)
    except (TypeError, ValueError) as exc:
        raise LagConfigError(
            "Derived Transformation\n"
            "sample_interval_sec is missing or invalid.\n"
            f"Got: {sample_interval_sec!r}"
        ) from exc
    if interval <= 0:
        raise LagConfigError(
            "Derived Transformation\n"
            "sample_interval_sec must be > 0.\n"
            f"Got: {interval}"
        )
    try:
        sec = float(seconds)
    except (TypeError, ValueError) as exc:
        raise LagConfigError(
            "Derived Transformation\n"
            f"Invalid term.seconds: {seconds!r}"
        ) from exc
    if sec < 0:
        raise LagConfigError(
            "Derived Transformation\n"
            f"term.seconds must be >= 0.\nGot: {sec}"
        )
    if sec == 0:
        return 0
    rows = sec / interval
    rows_i = int(round(rows))
    if abs(rows - rows_i) > 1e-9 or rows_i < 1:
        raise LagConfigError(
            "Derived Transformation\n"
            "Invalid term.seconds (not divisible by sample interval).\n"
            f"seconds            : {sec}\n"
            f"sample_interval_sec: {interval}\n"
            f"rows               : {rows}\n"
            "Choose seconds that are exact multiples of the sample interval."
        )
    return rows_i


@register_transformation
class DerivedTransformation(FeatureTransformation):
    """``output = sum(coeff * feature.shift(seconds/interval))`` within partition."""

    id = "derived"
    name = "Derived"
    order = 60
    enabled = False
    depends_on: list[str] = []
    params: dict[str, Any] = {}

    def transform(self, df: pd.DataFrame, context: TransformContext) -> TransformationResult:
        t0 = time.perf_counter()
        params = resolve_transform_params(self.id, getattr(self, "params", None), context.config)
        interval = resolve_sample_interval(context, params)
        outputs_raw = params.get("outputs") or []
        if not isinstance(outputs_raw, list) or not outputs_raw:
            raise LagConfigError(
                "Derived Transformation\n"
                "params.outputs is empty.\n"
                "Provide [{feature, column, terms:[{seconds, coeff}]}]."
            )
        persist_sample_interval(context, self.id, interval)
        partition_by = partition_columns(params, df)
        if context.cancelled():
            return TransformationResult.passthrough(
                df,
                transformation_id=self.id,
                transformation_name=self.name,
            )

        # Validate + plan first, then compute via Polars.
        planned: list[tuple[str, list[tuple[str, int, float]]]] = []
        created: list[str] = []
        for spec in outputs_raw:
            if not isinstance(spec, dict):
                raise LagConfigError(
                    "Derived Transformation\n"
                    f"Invalid outputs entry: {spec!r}"
                )
            feature = str(spec.get("feature") or "").strip()
            column = str(spec.get("column") or "").strip()
            terms = spec.get("terms") or []
            if not column:
                raise LagConfigError(
                    "Derived Transformation\n"
                    "outputs[].column is required."
                )
            if not isinstance(terms, list) or not terms:
                raise LagConfigError(
                    "Derived Transformation\n"
                    f"outputs[].terms is empty for column={column!r}."
                )
            term_plan: list[tuple[str, int, float]] = []
            term_features: set[str] = set()
            for term in terms:
                if not isinstance(term, dict):
                    raise LagConfigError(
                        "Derived Transformation\n"
                        f"Invalid term: {term!r}"
                    )
                term_feat = str(term.get("feature") or feature or "").strip()
                if not term_feat:
                    raise LagConfigError(
                        "Derived Transformation\n"
                        f"outputs[].feature or term.feature is required for column={column!r}."
                    )
                try:
                    coeff = float(term.get("coeff"))
                except (TypeError, ValueError) as exc:
                    raise LagConfigError(
                        "Derived Transformation\n"
                        f"Invalid term.coeff: {term.get('coeff')!r}"
                    ) from exc
                rows = _resolve_term_rows(term.get("seconds"), interval)
                term_plan.append((term_feat, rows, coeff))
                term_features.add(term_feat)
            missing_feats = [f for f in term_features if f not in df.columns]
            if missing_feats:
                raise LagConfigError(
                    "Derived Transformation\n"
                    "Feature not found\n"
                    + "\n".join(missing_feats)
                )
            planned.append((column, term_plan))
            created.append(column)

        total = len(planned)

        def _pandas_fallback(frame: pd.DataFrame) -> pd.DataFrame:
            local = frame.copy()
            for column, term_plan in planned:
                series: pd.Series | None = None
                for term_feat, rows, coeff in term_plan:
                    shifted = shift_feature_columns(
                        local, feature=term_feat, rows=rows, partition_by=partition_by
                    )
                    piece = shifted * coeff
                    series = piece if series is None else series + piece
                local[column] = series
            return local

        from .polars_ops import apply_derived_ops_via_polars

        out = apply_derived_ops_via_polars(
            df,
            outputs=planned,
            partition_by=partition_by,
            pandas_fallback=_pandas_fallback,
        )
        context.report_progress(f"Derived: {total}/{total} columns", total, total)

        elapsed = round(max(time.perf_counter() - t0, 0.0), 4)
        context.log("Derived Transformation")
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
                f"sample_interval_sec={interval}",
                f"partition_by={partition_by or []}",
                "frame_backend=polars_derived",
            ],
        )
