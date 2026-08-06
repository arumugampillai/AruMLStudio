"""Normalization transformation — rescale continuous features.

Answers \"how do I scale this feature?\" (not window aggregation).
Pipeline-only; disabled by default.
"""

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

NORMALIZATION_METHODS: tuple[str, ...] = (
    "zscore_rolling",
    "zscore_expanding",
    "robust",
    "minmax",
    "percentile_rank",
    "quantile_rank",
)
_VALID_METHODS = frozenset(NORMALIZATION_METHODS)

# Methods that need an integer row window.
_WINDOWED_METHODS = frozenset({
    "zscore_rolling",
    "robust",
    "minmax",
    "percentile_rank",
    "quantile_rank",
})

_METHOD_ALIASES: dict[str, str] = {
    "zscore_rolling": "zscore_rolling",
    "zscore": "zscore_rolling",
    "rolling_zscore": "zscore_rolling",
    "zscore_expanding": "zscore_expanding",
    "expanding_zscore": "zscore_expanding",
    "robust": "robust",
    "robust_scaling": "robust",
    "minmax": "minmax",
    "min_max": "minmax",
    "percentile_rank": "percentile_rank",
    "pctile": "percentile_rank",
    "percentile": "percentile_rank",
    "quantile_rank": "quantile_rank",
    "qrank": "quantile_rank",
}


def normalize_normalization_method(method: str) -> str:
    key = str(method or "").strip().lower()
    if key not in _METHOD_ALIASES:
        raise LagConfigError(
            "Normalization Transformation\n"
            f"Invalid method={method!r}. Expected one of {list(NORMALIZATION_METHODS)}."
        )
    return _METHOD_ALIASES[key]


def normalization_column_name(
    feature: str,
    method: str,
    *,
    window: int | None = None,
) -> str:
    feat = str(feature or "").strip()
    meth = normalize_normalization_method(method)
    if meth == "zscore_expanding":
        return f"{feat}_zscore_exp"
    if meth == "zscore_rolling":
        return f"{feat}_zscore_{int(window or 0)}"
    if meth == "robust":
        return f"{feat}_robust_{int(window or 0)}"
    if meth == "minmax":
        return f"{feat}_minmax_{int(window or 0)}"
    if meth == "percentile_rank":
        return f"{feat}_pctile_{int(window or 0)}"
    if meth == "quantile_rank":
        return f"{feat}_qrank_{int(window or 0)}"
    return f"{feat}_{meth}"


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
                "Normalization Transformation\n"
                f"Invalid window={item!r}; expected positive integer."
            ) from exc
        if win <= 0:
            raise LagConfigError(
                "Normalization Transformation\n"
                f"Invalid window={win}; must be a positive integer."
            )
        if win not in out:
            out.append(win)
    return out


def _parse_methods(params: dict[str, Any]) -> list[str]:
    raw = params.get("methods")
    if raw is None:
        raw = params.get("method")
    if raw is None:
        raw = params.get("operations")
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    out: list[str] = []
    for item in raw:
        meth = normalize_normalization_method(str(item))
        if meth not in out:
            out.append(meth)
    return out


def _rolling(series: pd.Series, rows: int):
    return series.rolling(window=rows, min_periods=rows)


def _apply_windowed(
    values: pd.Series,
    method: str,
    window: int,
    *,
    ddof: int,
) -> pd.Series:
    roll = _rolling(values, window)
    if method == "zscore_rolling":
        mean = roll.mean()
        std = roll.std(ddof=ddof)
        return (values - mean) / std.replace(0, np.nan)
    if method == "robust":
        med = roll.median()
        q1 = roll.quantile(0.25)
        q3 = roll.quantile(0.75)
        iqr = (q3 - q1).replace(0, np.nan)
        return (values - med) / iqr
    if method == "minmax":
        lo = roll.min()
        hi = roll.max()
        span = (hi - lo).replace(0, np.nan)
        return (values - lo) / span
    if method == "percentile_rank":
        # Average rank within the trailing window, scaled to [0, 1].
        def _pct(s: pd.Series) -> float:
            if s.isna().all():
                return float("nan")
            cur = s.iloc[-1]
            if pd.isna(cur):
                return float("nan")
            valid = s.dropna()
            if valid.empty:
                return float("nan")
            return float((valid <= cur).sum()) / float(len(valid))

        return roll.apply(_pct, raw=False)
    if method == "quantile_rank":
        # Discrete 0..n_bins-1 via empirical CDF in window (default 10 bins).
        n_bins = 10

        def _qrank(s: pd.Series) -> float:
            if s.isna().all():
                return float("nan")
            cur = s.iloc[-1]
            if pd.isna(cur):
                return float("nan")
            valid = s.dropna()
            if valid.empty:
                return float("nan")
            pct = float((valid <= cur).sum()) / float(len(valid))
            # Map (0,1] → 0..n_bins-1; exact 0 stays 0.
            idx = int(min(n_bins - 1, max(0, np.floor(pct * n_bins))))
            return float(idx)

        return roll.apply(_qrank, raw=False)
    raise LagConfigError(f"Normalization Transformation\nUnsupported method={method!r}")


def _apply_expanding_zscore(values: pd.Series, *, ddof: int) -> pd.Series:
    exp = values.expanding(min_periods=2)
    mean = exp.mean()
    std = exp.std(ddof=ddof)
    return (values - mean) / std.replace(0, np.nan)


@register_transformation
class NormalizationTransformation(FeatureTransformation):
    """Rescale continuous features (rolling / expanding modes)."""

    id = "normalization"
    name = "Normalization"
    order = 75  # After Math (70); before Regime (80)
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
        from .describe import OutputDescriptor, make_stage_descriptor

        del upstream, master_features, sample_interval_sec
        is_enabled = bool(self.enabled if enabled is None else enabled)
        params = dict(params or {})
        outputs: list[OutputDescriptor] = []
        try:
            features = _parse_features(params)
            methods = _parse_methods(params)
            windows = _parse_windows(params)
            for feat in features:
                for meth in methods:
                    if meth in _WINDOWED_METHODS:
                        for win in windows:
                            outputs.append(
                                OutputDescriptor(
                                    name=normalization_column_name(
                                        feat, meth, window=win
                                    ),
                                    kind="normalization",
                                )
                            )
                    else:
                        outputs.append(
                            OutputDescriptor(
                                name=normalization_column_name(feat, meth),
                                kind="normalization",
                            )
                        )
        except Exception:
            outputs = []
        return make_stage_descriptor(
            self,
            enabled=is_enabled,
            outputs=outputs,
            notes="Scale continuous features (z-score / robust / min-max / ranks).",
        )

    def transform(self, df: pd.DataFrame, context: TransformContext) -> TransformationResult:
        t0 = time.perf_counter()
        params = resolve_transform_params(self.id, getattr(self, "params", None), context.config)
        interval = resolve_sample_interval(context, params)
        features = _parse_features(params)
        methods = _parse_methods(params)
        windows = _parse_windows(params)
        if not features or not methods:
            raise LagConfigError(
                "Normalization Transformation\n"
                "Requires non-empty features and methods."
            )
        needs_window = any(m in _WINDOWED_METHODS for m in methods)
        if needs_window and not windows:
            raise LagConfigError(
                "Normalization Transformation\n"
                "Selected methods require at least one positive window."
            )
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise LagConfigError(
                "Normalization Transformation\nFeature not found\n" + "\n".join(missing)
            )
        try:
            ddof = int(params.get("ddof", 0))
        except (TypeError, ValueError) as exc:
            raise LagConfigError(
                f"Normalization Transformation\nInvalid ddof: {params.get('ddof')!r}"
            ) from exc

        persist_sample_interval(context, self.id, interval)
        partition_by = partition_columns(params, df)
        if context.cancelled():
            return TransformationResult.passthrough(
                df,
                transformation_id=self.id,
                transformation_name=self.name,
            )

        specs: list[tuple[str, str, int | None, str]] = []
        created: list[str] = []
        for feat in features:
            for meth in methods:
                if meth in _WINDOWED_METHODS:
                    for win in windows:
                        col = normalization_column_name(feat, meth, window=win)
                        specs.append((feat, meth, win, col))
                        created.append(col)
                else:
                    col = normalization_column_name(feat, meth)
                    specs.append((feat, meth, None, col))
                    created.append(col)
        total = len(specs)

        def _pandas_fallback(frame: pd.DataFrame) -> pd.DataFrame:
            local = frame.copy()
            for feat, meth, win, col in specs:
                if partition_by:
                    grouped = local.groupby(partition_by, sort=False, group_keys=False)[feat]
                    if meth == "zscore_expanding":
                        series = grouped.transform(
                            lambda s, d=ddof: _apply_expanding_zscore(s, ddof=d)
                        )
                    else:
                        series = grouped.transform(
                            lambda s, m=meth, n=int(win or 0), d=ddof: _apply_windowed(
                                s, m, n, ddof=d
                            )
                        )
                else:
                    values = pd.to_numeric(local[feat], errors="coerce")
                    if meth == "zscore_expanding":
                        series = _apply_expanding_zscore(values, ddof=ddof)
                    else:
                        series = _apply_windowed(values, meth, int(win or 0), ddof=ddof)
                local[col] = series
            return local

        from .polars_ops import apply_normalization_via_polars

        out = apply_normalization_via_polars(
            df,
            specs=specs,
            ddof=ddof,
            partition_by=partition_by,
            pandas_fallback=_pandas_fallback,
        )
        context.report_progress(f"Normalization: {total}/{total} columns", total, total)

        elapsed = round(max(time.perf_counter() - t0, 0.0), 4)
        context.log("Normalization Transformation")
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
                f"methods={methods}",
                f"Columns Created : {len(created)}",
                "frame_backend=polars_normalization",
            ],
        )


__all__ = [
    "NORMALIZATION_METHODS",
    "NormalizationTransformation",
    "normalization_column_name",
    "normalize_normalization_method",
]
