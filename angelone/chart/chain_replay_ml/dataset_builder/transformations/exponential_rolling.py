"""Exponential Rolling — EMA / EWM stats on arbitrary columns.

Pipeline-only. Does not replace controller EMA Computed Base features
(``ltp_ema20``, ``iv_ema50``, …). Transform outputs use underscore-before-period
names (``feature_ema_20``, ``feature_ewm_std_20``) to stay distinct.
"""

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
)

# Supported now. Architecture accepts additional exponential statistics later.
EXPONENTIAL_ROLLING_OPS: tuple[str, ...] = ("ema", "ewm_mean", "ewm_std")
_FUTURE_OPS: frozenset[str] = frozenset({"ewm_var"})
_OP_ALIASES: dict[str, str] = {
    "ema": "ema",
    "ewm": "ema",
    "ewm_mean": "ewm_mean",
    "ewm_std": "ewm_std",
}

# EMA uses classic adjust=False; other EWM ops use pandas default adjust=True.
_OP_ADJUST: dict[str, bool] = {
    "ema": False,
    "ewm_mean": True,
    "ewm_std": True,
}


def normalize_exponential_rolling_op(op: str) -> str:
    key = str(op or "").strip().lower()
    if key in _FUTURE_OPS:
        raise LagConfigError(
            "Exponential Rolling Transformation\n"
            f"Operation {op!r} is reserved for a future release. "
            f"Supported: {list(EXPONENTIAL_ROLLING_OPS)}."
        )
    if key not in _OP_ALIASES:
        raise LagConfigError(
            "Exponential Rolling Transformation\n"
            f"Invalid operation={op!r}. Supported: {list(EXPONENTIAL_ROLLING_OPS)}."
        )
    return _OP_ALIASES[key]


def exponential_rolling_column_name(feature: str, op: str, period: int) -> str:
    """Deterministic name: ``<feature>_<op>_<period>`` (ema → ``_ema_``)."""
    feat = str(feature or "").strip()
    operation = normalize_exponential_rolling_op(op)
    per = int(period)
    if per <= 0:
        raise LagConfigError(
            "Exponential Rolling Transformation\n"
            f"Invalid period={period!r}; must be a positive integer."
        )
    return f"{feat}_{operation}_{per}"


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


def _parse_periods(params: dict[str, Any]) -> list[int]:
    raw = params.get("periods")
    if raw is None:
        raw = params.get("period")
    if raw is None:
        raw = params.get("spans")
    if raw is None:
        return []
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raw = [raw]
    out: list[int] = []
    for item in raw:
        try:
            per = int(item)
        except (TypeError, ValueError) as exc:
            raise LagConfigError(
                "Exponential Rolling Transformation\n"
                f"Invalid period={item!r}; expected positive integer."
            ) from exc
        if per <= 0:
            raise LagConfigError(
                "Exponential Rolling Transformation\n"
                f"Invalid period={per}; must be a positive integer."
            )
        if per not in out:
            out.append(per)
    return out


def _parse_operations(params: dict[str, Any]) -> list[str]:
    raw = params.get("operations")
    if raw is None:
        raw = params.get("ops")
    if raw is None:
        return ["ema"]  # backward-compatible default
    if isinstance(raw, str):
        raw = [raw]
    out: list[str] = []
    for item in raw:
        op = normalize_exponential_rolling_op(str(item))
        if op not in out:
            out.append(op)
    return out or ["ema"]


def _ewm_object(series: pd.Series, period: int, *, adjust: bool) -> Any:
    return series.astype(float).ewm(span=int(period), adjust=bool(adjust))


def _apply_op_on_ewm(ewm_obj: Any, op: str) -> pd.Series:
    if op == "ema" or op == "ewm_mean":
        return ewm_obj.mean()
    if op == "ewm_std":
        return ewm_obj.std()
    if op == "ewm_var":
        return ewm_obj.var()
    raise LagConfigError(
        "Exponential Rolling Transformation\n"
        f"Unsupported op={op!r}."
    )


def _apply_op(series: pd.Series, op: str, period: int) -> pd.Series:
    adjust = _OP_ADJUST[op]
    return _apply_op_on_ewm(_ewm_object(series, period, adjust=adjust), op)


@register_transformation
class ExponentialRollingTransformation(FeatureTransformation):
    """EMA / EWM stats on selected columns (pipeline engineering, not controller state)."""

    id = "exponential_rolling"
    name = "Exponential Rolling"
    order = 42  # After Rolling (40); before Interaction (50)
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

        del upstream, master_features, sample_interval_sec
        is_enabled = bool(self.enabled if enabled is None else enabled)
        params = dict(params or {})
        outputs: list[OutputDescriptor] = []
        try:
            features = _parse_features(params)
            periods = _parse_periods(params)
            operations = _parse_operations(params)
            for feat in features:
                for period in periods:
                    for op in operations:
                        outputs.append(
                            OutputDescriptor(
                                name=exponential_rolling_column_name(feat, op, period),
                                kind="exponential_rolling",
                                source_feature=feat,
                                op=op,
                                meta={"period": int(period)},
                            )
                        )
        except Exception:
            outputs = []
        return make_stage_descriptor(
            self,
            enabled=is_enabled,
            outputs=outputs,
            input_sources=[MASTER_STAGE_ID],
            notes="Planned exponential rolling columns from features × periods × ops.",
        )

    def transform(self, df: pd.DataFrame, context: TransformContext) -> TransformationResult:
        t0 = time.perf_counter()
        params = resolve_transform_params(
            self.id, getattr(self, "params", None), context.config
        )
        features = _parse_features(params)
        periods = _parse_periods(params)
        operations = _parse_operations(params)
        if not features or not periods or not operations:
            raise LagConfigError(
                "Exponential Rolling Transformation\n"
                "Requires non-empty features, periods, and operations."
            )
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise LagConfigError(
                "Exponential Rolling Transformation\n"
                "Feature not found\n"
                + "\n".join(missing)
            )

        planned: list[tuple[str, str, int, str]] = []
        seen_names: set[str] = set()
        for feat in features:
            for per in periods:
                for op in operations:
                    col = exponential_rolling_column_name(feat, op, per)
                    if col in seen_names:
                        raise LagConfigError(
                            "Exponential Rolling Transformation\n"
                            f"Duplicate output name: {col}"
                        )
                    if col in df.columns:
                        raise LagConfigError(
                            "Exponential Rolling Transformation\n"
                            f"Output column already exists: {col}"
                        )
                    seen_names.add(col)
                    planned.append((feat, op, per, col))

        interval = resolve_sample_interval(context, params)
        persist_sample_interval(context, self.id, interval)
        partition_by = partition_columns(params, df)
        if context.cancelled():
            return TransformationResult.passthrough(
                df,
                transformation_id=self.id,
                transformation_name=self.name,
            )

        created = [col for _, _, _, col in planned]
        total = len(planned)

        def _pandas_fallback(frame: pd.DataFrame) -> pd.DataFrame:
            local = frame.copy()
            ewm_cache: dict[tuple[str, int, bool], Any] = {}
            for feat, op, per, col in planned:
                adjust = _OP_ADJUST[op]
                cache_key = (feat, per, adjust)
                if partition_by:
                    grouped = local.groupby(partition_by, sort=False, group_keys=False)[feat]
                    series = grouped.transform(lambda s, p=per, o=op: _apply_op(s, o, p))
                else:
                    if cache_key not in ewm_cache:
                        ewm_cache[cache_key] = _ewm_object(
                            local[feat], per, adjust=adjust
                        )
                    series = _apply_op_on_ewm(ewm_cache[cache_key], op)
                local[col] = series
            return local

        from .polars_ops import apply_ewm_ops_via_polars

        out = apply_ewm_ops_via_polars(
            df,
            specs=planned,
            partition_by=partition_by,
            pandas_fallback=_pandas_fallback,
        )
        context.report_progress(
            f"Exponential Rolling: {total}/{total} columns",
            total,
            total,
        )

        elapsed = round(max(time.perf_counter() - t0, 0.0), 4)
        context.log("Exponential Rolling Transformation")
        context.log(f"Features Selected : {len(features)}")
        context.log(f"Periods           : {periods}")
        context.log(f"Operations        : {operations}")
        context.log(f"Columns Created   : {len(created)}")
        context.log(f"Elapsed           : {elapsed:.2f} s")
        return TransformationResult(
            frame=out,
            created_columns=created,
            elapsed_sec=elapsed,
            rows_processed=int(len(out)),
            transformation_id=self.id,
            transformation_name=self.name,
            messages=[
                f"Features Selected : {len(features)}",
                f"Periods : {periods}",
                f"Operations : {operations}",
                f"Columns Created : {len(created)}",
                f"sample_interval_sec={interval}",
                f"partition_by={partition_by or []}",
                "frame_backend=polars_ewm",
            ],
        )
