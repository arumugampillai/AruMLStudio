"""Rolling Statistics transformation — generic mean/std/min/max/median over row windows.

Pipeline-only feature engineering. Controllers must not emit these columns.
Disabled by default; columns appear only when the user enables Rolling Statistics.
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

ROLLING_OPS: tuple[str, ...] = ("mean", "std", "min", "max", "median")
_VALID_OPS = frozenset(ROLLING_OPS)

# Alias map for UI labels / legacy keys.
_OP_ALIASES: dict[str, str] = {
    "mean": "mean",
    "avg": "mean",
    "average": "mean",
    "std": "std",
    "stdev": "std",
    "standard_deviation": "std",
    "min": "min",
    "minimum": "min",
    "max": "max",
    "maximum": "max",
    "median": "median",
}


def normalize_rolling_op(op: str) -> str:
    key = str(op or "").strip().lower()
    if key not in _OP_ALIASES:
        raise LagConfigError(
            "Rolling Transformation\n"
            f"Invalid operation={op!r}. Expected one of {list(ROLLING_OPS)}."
        )
    return _OP_ALIASES[key]


def rolling_column_name(feature: str, op: str, window: int) -> str:
    """Deterministic name: ``<feature>_roll_<op>_<window>``."""
    feat = str(feature or "").strip()
    operation = normalize_rolling_op(op)
    win = int(window)
    if win <= 0:
        raise LagConfigError(
            "Rolling Transformation\n"
            f"Invalid window={window!r}; must be a positive integer."
        )
    return f"{feat}_roll_{operation}_{win}"


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


def _parse_windows(params: dict[str, Any]) -> list[int]:
    """Parse integer row windows from ``windows`` / ``window`` / ``window_rows``."""
    raw = params.get("windows")
    if raw is None:
        raw = params.get("window_rows")
    if raw is None:
        raw = params.get("window")
    if raw is None:
        return []
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raw = [raw]
    out: list[int] = []
    for item in raw:
        try:
            win = int(item)
        except (TypeError, ValueError) as exc:
            raise LagConfigError(
                "Rolling Transformation\n"
                f"Invalid window={item!r}; expected positive integer."
            ) from exc
        if win <= 0:
            raise LagConfigError(
                "Rolling Transformation\n"
                f"Invalid window={win}; must be a positive integer."
            )
        if win not in out:
            out.append(win)
    return out


def _parse_operations(params: dict[str, Any]) -> list[str]:
    raw = params.get("operations")
    if raw is None:
        raw = params.get("ops")
    if raw is None and params.get("stat") is not None:
        # Single-stat convenience (not zscore — that stays on rolling_statistics).
        raw = [params.get("stat")]
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    out: list[str] = []
    for item in raw:
        op = normalize_rolling_op(str(item))
        if op not in out:
            out.append(op)
    return out


def _apply_op(roller: Any, op: str, *, ddof: int) -> pd.Series:
    if op == "mean":
        return roller.mean()
    if op == "std":
        return roller.std(ddof=ddof)
    if op == "min":
        return roller.min()
    if op == "max":
        return roller.max()
    if op == "median":
        return roller.median()
    raise LagConfigError(f"Rolling Transformation\nUnsupported op={op!r}")


@register_transformation
class RollingTransformation(FeatureTransformation):
    """Generic rolling ops on Base / Computed Base (and prior transform) columns."""

    id = "rolling"
    name = "Rolling Statistics"
    order = 40  # After Lag/Diff/Return; before Interaction (50)
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
            windows = _parse_windows(params)
            operations = _parse_operations(params)
            for feat in features:
                for win in windows:
                    for op in operations:
                        outputs.append(
                            OutputDescriptor(
                                name=rolling_column_name(feat, op, win),
                                kind="rolling",
                                source_feature=feat,
                                op=op,
                                meta={"window": int(win)},
                            )
                        )
        except Exception:
            outputs = []
        return make_stage_descriptor(
            self,
            enabled=is_enabled,
            outputs=outputs,
            input_sources=[MASTER_STAGE_ID],
            notes="Planned rolling columns from features × windows × ops.",
        )

    def transform(self, df: pd.DataFrame, context: TransformContext) -> TransformationResult:
        t0 = time.perf_counter()
        params = resolve_transform_params(self.id, getattr(self, "params", None), context.config)
        features = _parse_features(params)
        windows = _parse_windows(params)
        operations = _parse_operations(params)
        if not features or not windows or not operations:
            raise LagConfigError(
                "Rolling Transformation\n"
                "Requires non-empty features, windows, and operations."
            )
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise LagConfigError(
                "Rolling Transformation\n"
                "Feature not found\n"
                + "\n".join(missing)
            )
        try:
            ddof = int(params.get("ddof", 0))
        except (TypeError, ValueError) as exc:
            raise LagConfigError(
                "Rolling Transformation\n"
                f"Invalid ddof: {params.get('ddof')!r}"
            ) from exc

        interval = resolve_sample_interval(context, params)
        persist_sample_interval(context, self.id, interval)
        partition_by = partition_columns(params, df)
        if context.cancelled():
            return TransformationResult.passthrough(
                df,
                transformation_id=self.id,
                transformation_name=self.name,
            )

        out = df.copy()
        created: list[str] = []
        total = len(features) * len(windows) * len(operations)
        specs: list[tuple[str, str, int, str]] = []
        for feat in features:
            for win in windows:
                for op in operations:
                    col = rolling_column_name(feat, op, win)
                    specs.append((feat, op, int(win), col))
                    created.append(col)

        def _pandas_fallback(frame: pd.DataFrame) -> pd.DataFrame:
            local = frame.copy()
            for feat in features:
                for win in windows:
                    if partition_by:
                        grouped = local.groupby(partition_by, sort=False, group_keys=False)[feat]
                        op_series = {
                            op: grouped.transform(
                                lambda s, o=op, n=win, d=ddof: _apply_op(
                                    s.rolling(window=n, min_periods=n), o, ddof=d
                                )
                            )
                            for op in operations
                        }
                    else:
                        roller = local[feat].rolling(window=win, min_periods=win)
                        op_series = {
                            op: _apply_op(roller, op, ddof=ddof) for op in operations
                        }
                    for op in operations:
                        local[rolling_column_name(feat, op, win)] = op_series[op]
            return local

        from .polars_ops import apply_rolling_ops_via_polars

        out = apply_rolling_ops_via_polars(
            df,
            specs=specs,
            ddof=ddof,
            partition_by=partition_by,
            pandas_fallback=_pandas_fallback,
        )
        context.report_progress(f"Rolling: {total}/{total} columns", total, total)

        elapsed = round(max(time.perf_counter() - t0, 0.0), 4)
        context.log("Rolling Transformation")
        context.log(f"Features Selected : {len(features)}")
        context.log(f"Windows           : {windows}")
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
                f"Windows : {windows}",
                f"Operations : {operations}",
                f"Columns Created : {len(created)}",
                f"sample_interval_sec={interval}",
                f"partition_by={partition_by or []}",
                "frame_backend=polars_rolling",
            ],
        )
