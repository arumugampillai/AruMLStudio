"""Regime / Bucket transformation — discretise continuous features.

Output is categorical integer codes, not continuous values.
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

REGIME_METHODS: tuple[str, ...] = (
    "threshold_bucket",
    "quantile_bucket",
    "equal_width",
    "binary_threshold",
    "ternary_state",
)
_VALID_METHODS = frozenset(REGIME_METHODS)

_WINDOWED_METHODS = frozenset({"quantile_bucket", "equal_width"})

_METHOD_ALIASES: dict[str, str] = {
    "threshold_bucket": "threshold_bucket",
    "thresholds": "threshold_bucket",
    "quantile_bucket": "quantile_bucket",
    "quantile": "quantile_bucket",
    "equal_width": "equal_width",
    "equalwidth": "equal_width",
    "binary_threshold": "binary_threshold",
    "binary": "binary_threshold",
    "ternary_state": "ternary_state",
    "ternary": "ternary_state",
}


def normalize_regime_method(method: str) -> str:
    key = str(method or "").strip().lower()
    if key not in _METHOD_ALIASES:
        raise LagConfigError(
            "Regime Transformation\n"
            f"Invalid method={method!r}. Expected one of {list(REGIME_METHODS)}."
        )
    return _METHOD_ALIASES[key]


def regime_column_name(
    feature: str,
    method: str,
    *,
    window: int | None = None,
    n_bins: int | None = None,
    threshold: float | None = None,
    low: float | None = None,
    high: float | None = None,
) -> str:
    feat = str(feature or "").strip()
    meth = normalize_regime_method(method)
    if meth == "binary_threshold":
        t = 0.0 if threshold is None else float(threshold)
        return f"{feat}_bin_{t:g}"
    if meth == "ternary_state":
        lo = -1.0 if low is None else float(low)
        hi = 1.0 if high is None else float(high)
        return f"{feat}_tern_{lo:g}_{hi:g}"
    if meth == "threshold_bucket":
        return f"{feat}_tbucket"
    if meth == "quantile_bucket":
        return f"{feat}_qbucket_{int(n_bins or 5)}_{int(window or 0)}"
    if meth == "equal_width":
        return f"{feat}_ewidth_{int(n_bins or 5)}_{int(window or 0)}"
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
                "Regime Transformation\n"
                f"Invalid window={item!r}; expected positive integer."
            ) from exc
        if win <= 0:
            raise LagConfigError(
                "Regime Transformation\n"
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
        meth = normalize_regime_method(str(item))
        if meth not in out:
            out.append(meth)
    return out


def _parse_edges(params: dict[str, Any]) -> list[float]:
    raw = params.get("edges")
    if raw is None:
        raw = params.get("thresholds")
    if raw is None:
        return [-1.0, 0.0, 1.0]
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        raw = [raw]
    out: list[float] = []
    for item in raw:
        try:
            out.append(float(item))
        except (TypeError, ValueError) as exc:
            raise LagConfigError(
                f"Regime Transformation\nInvalid edge={item!r}"
            ) from exc
    return sorted(out)


def _binary(values: pd.Series, threshold: float) -> pd.Series:
    return (pd.to_numeric(values, errors="coerce") >= threshold).astype(float)


def _ternary(values: pd.Series, low: float, high: float) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    out = pd.Series(np.nan, index=v.index, dtype=float)
    out = out.mask(v < low, -1.0)
    out = out.mask((v >= low) & (v <= high), 0.0)
    out = out.mask(v > high, 1.0)
    return out


def _threshold_bucket(values: pd.Series, edges: list[float]) -> pd.Series:
    v = pd.to_numeric(values, errors="coerce")
    # codes: 0 = below first edge, …, len(edges) = above last edge
    return pd.Series(np.digitize(v.to_numpy(), bins=np.asarray(edges, dtype=float), right=False), index=v.index, dtype=float).where(v.notna())


def _rolling_quantile_bucket(values: pd.Series, window: int, n_bins: int) -> pd.Series:
    roll = values.rolling(window=window, min_periods=window)

    def _code(s: pd.Series) -> float:
        if s.isna().all():
            return float("nan")
        cur = s.iloc[-1]
        if pd.isna(cur):
            return float("nan")
        valid = s.dropna()
        if valid.empty:
            return float("nan")
        pct = float((valid <= cur).sum()) / float(len(valid))
        return float(int(min(n_bins - 1, max(0, np.floor(pct * n_bins)))))

    return roll.apply(_code, raw=False)


def _rolling_equal_width(values: pd.Series, window: int, n_bins: int) -> pd.Series:
    roll = values.rolling(window=window, min_periods=window)

    def _code(s: pd.Series) -> float:
        if s.isna().all():
            return float("nan")
        cur = s.iloc[-1]
        if pd.isna(cur):
            return float("nan")
        lo = float(s.min())
        hi = float(s.max())
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return 0.0
        pct = (float(cur) - lo) / (hi - lo)
        return float(int(min(n_bins - 1, max(0, np.floor(pct * n_bins)))))

    return roll.apply(_code, raw=False)


@register_transformation
class RegimeTransformation(FeatureTransformation):
    """Discretise continuous features into categorical regime codes."""

    id = "regime"
    name = "Regime / Bucket"
    order = 80  # After Normalization (75)
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
            n_bins = int(params.get("n_bins", 5) or 5)
            threshold = float(params.get("threshold", 0.0) or 0.0)
            low = float(params.get("low", -1.0) if params.get("low") is not None else -1.0)
            high = float(params.get("high", 1.0) if params.get("high") is not None else 1.0)
            for feat in features:
                for meth in methods:
                    if meth in _WINDOWED_METHODS:
                        for win in windows:
                            outputs.append(
                                OutputDescriptor(
                                    name=regime_column_name(
                                        feat,
                                        meth,
                                        window=win,
                                        n_bins=n_bins,
                                    ),
                                    kind="regime",
                                )
                            )
                    else:
                        outputs.append(
                            OutputDescriptor(
                                name=regime_column_name(
                                    feat,
                                    meth,
                                    threshold=threshold,
                                    low=low,
                                    high=high,
                                ),
                                kind="regime",
                            )
                        )
        except Exception:
            outputs = []
        return make_stage_descriptor(
            self,
            enabled=is_enabled,
            outputs=outputs,
            notes="Discretise continuous features into categorical codes.",
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
                "Regime Transformation\nRequires non-empty features and methods."
            )
        needs_window = any(m in _WINDOWED_METHODS for m in methods)
        if needs_window and not windows:
            raise LagConfigError(
                "Regime Transformation\n"
                "Selected methods require at least one positive window."
            )
        missing = [f for f in features if f not in df.columns]
        if missing:
            raise LagConfigError(
                "Regime Transformation\nFeature not found\n" + "\n".join(missing)
            )
        try:
            n_bins = int(params.get("n_bins", 5) or 5)
        except (TypeError, ValueError) as exc:
            raise LagConfigError(
                f"Regime Transformation\nInvalid n_bins: {params.get('n_bins')!r}"
            ) from exc
        if n_bins < 2:
            raise LagConfigError("Regime Transformation\nn_bins must be >= 2.")
        try:
            threshold = float(params.get("threshold", 0.0) or 0.0)
        except (TypeError, ValueError) as exc:
            raise LagConfigError(
                f"Regime Transformation\nInvalid threshold: {params.get('threshold')!r}"
            ) from exc
        try:
            low = float(params.get("low", -1.0) if params.get("low") is not None else -1.0)
            high = float(params.get("high", 1.0) if params.get("high") is not None else 1.0)
        except (TypeError, ValueError) as exc:
            raise LagConfigError(
                "Regime Transformation\nInvalid low/high thresholds."
            ) from exc
        if low > high:
            raise LagConfigError("Regime Transformation\nlow must be <= high.")
        edges = _parse_edges(params)

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
                        col = regime_column_name(
                            feat, meth, window=win, n_bins=n_bins
                        )
                        specs.append((feat, meth, win, col))
                        created.append(col)
                else:
                    col = regime_column_name(
                        feat,
                        meth,
                        threshold=threshold,
                        low=low,
                        high=high,
                    )
                    specs.append((feat, meth, None, col))
                    created.append(col)
        total = len(specs)

        def _pandas_fallback(frame: pd.DataFrame) -> pd.DataFrame:
            local = frame.copy()
            for feat, meth, win, col in specs:

                def _compute(series: pd.Series, m=meth, w=win) -> pd.Series:
                    if m == "binary_threshold":
                        return _binary(series, threshold)
                    if m == "ternary_state":
                        return _ternary(series, low, high)
                    if m == "threshold_bucket":
                        return _threshold_bucket(series, edges)
                    if m == "quantile_bucket":
                        return _rolling_quantile_bucket(series, int(w or 0), n_bins)
                    if m == "equal_width":
                        return _rolling_equal_width(series, int(w or 0), n_bins)
                    raise LagConfigError(
                        f"Regime Transformation\nUnsupported method={m!r}"
                    )

                if partition_by:
                    series = local.groupby(partition_by, sort=False, group_keys=False)[
                        feat
                    ].transform(_compute)
                else:
                    series = _compute(local[feat])
                local[col] = series
            return local

        from .polars_ops import apply_regime_via_polars

        out = apply_regime_via_polars(
            df,
            specs=specs,
            threshold=threshold,
            low=low,
            high=high,
            edges=edges,
            n_bins=n_bins,
            partition_by=partition_by,
            pandas_fallback=_pandas_fallback,
        )
        context.report_progress(f"Regime: {total}/{total} columns", total, total)

        elapsed = round(max(time.perf_counter() - t0, 0.0), 4)
        context.log("Regime / Bucket Transformation")
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
                "frame_backend=polars_regime",
            ],
        )


__all__ = [
    "REGIME_METHODS",
    "RegimeTransformation",
    "normalize_regime_method",
    "regime_column_name",
]
