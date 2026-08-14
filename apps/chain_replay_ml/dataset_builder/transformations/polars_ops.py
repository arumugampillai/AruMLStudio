"""Polars-backed transform kernels (Phase P2). Pandas in/out at the boundary."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd


def _to_polars(df: pd.DataFrame) -> Any:
    from chain_replay_ml.frame_backend import arrow_table_to_polars, require_polars

    pl = require_polars()
    try:
        import pyarrow as pa

        return arrow_table_to_polars(pa.Table.from_pandas(df, preserve_index=False))
    except Exception:
        return pl.from_pandas(df)


def _from_polars(pl_df: Any) -> pd.DataFrame:
    from chain_replay_ml.frame_backend import polars_to_pandas

    return polars_to_pandas(pl_df)


def _try_polars(fn: Callable[[], pd.DataFrame], pandas_fallback: Callable[[pd.DataFrame], pd.DataFrame], df: pd.DataFrame) -> pd.DataFrame:
    try:
        return fn()
    except Exception:
        return pandas_fallback(df)


def _math_expr(pl: Any, feature: str, op: str, *, clip_min: float, clip_max: float | None) -> Any:
    col = pl.col(feature).cast(pl.Float64, strict=False)
    if op == "abs":
        return col.abs()
    if op == "log":
        return pl.when(col > 0.0).then(col.log()).otherwise(None)
    if op == "sqrt":
        return pl.when(col >= 0.0).then(col.sqrt()).otherwise(None)
    if op == "square":
        return col * col
    if op == "cube":
        return col * col * col
    if op == "clip":
        if clip_max is None:
            return col.clip(lower_bound=clip_min)
        return col.clip(lower_bound=clip_min, upper_bound=clip_max)
    if op == "sign":
        return col.sign()
    if op == "negate":
        return -col
    raise ValueError(f"Unsupported math op={op!r}")


def apply_math_ops_via_polars(
    df: pd.DataFrame,
    *,
    specs: list[tuple[str, str, str]],
    clip_min: float,
    clip_max: float | None,
    pandas_fallback,
) -> pd.DataFrame:
    """``specs`` = ``(feature, op, output_column)``. ``pandas_fallback(df)->df`` on failure."""
    if not specs:
        return df

    def _run() -> pd.DataFrame:
        from chain_replay_ml.frame_backend import require_polars

        pl = require_polars()
        pl_df = _to_polars(df)
        exprs = [
            _math_expr(pl, feat, op, clip_min=clip_min, clip_max=clip_max).alias(out_col)
            for feat, op, out_col in specs
        ]
        return _from_polars(pl_df.with_columns(exprs))

    return _try_polars(_run, pandas_fallback, df)


def _rolling_expr(
    pl: Any,
    feature: str,
    op: str,
    window: int,
    *,
    ddof: int,
    partition_by: list[str],
) -> Any:
    col = pl.col(feature).cast(pl.Float64, strict=False)
    win = int(window)
    if op == "mean":
        expr = col.rolling_mean(window_size=win, min_samples=win)
    elif op == "std":
        expr = col.rolling_std(window_size=win, min_samples=win, ddof=int(ddof))
    elif op == "min":
        expr = col.rolling_min(window_size=win, min_samples=win)
    elif op == "max":
        expr = col.rolling_max(window_size=win, min_samples=win)
    elif op == "median":
        expr = col.rolling_median(window_size=win, min_samples=win)
    else:
        raise ValueError(f"Unsupported rolling op={op!r}")
    if partition_by:
        expr = expr.over(partition_by)
    return expr


def apply_rolling_ops_via_polars(
    df: pd.DataFrame,
    *,
    specs: list[tuple[str, str, int, str]],
    ddof: int,
    partition_by: list[str],
    pandas_fallback,
) -> pd.DataFrame:
    """``specs`` = ``(feature, op, window, output_column)``."""
    if not specs:
        return df

    def _run() -> pd.DataFrame:
        from chain_replay_ml.frame_backend import require_polars

        pl = require_polars()
        pl_df = _to_polars(df)
        exprs = [
            _rolling_expr(
                pl, feat, op, win, ddof=ddof, partition_by=partition_by
            ).alias(out_col)
            for feat, op, win, out_col in specs
        ]
        return _from_polars(pl_df.with_columns(exprs))

    return _try_polars(_run, pandas_fallback, df)


def _ewm_expr(
    pl: Any,
    feature: str,
    op: str,
    period: int,
    *,
    partition_by: list[str],
) -> Any:
    col = pl.col(feature).cast(pl.Float64, strict=False)
    span = int(period)
    if op == "ema":
        expr = col.ewm_mean(span=span, adjust=False)
    elif op == "ewm_mean":
        expr = col.ewm_mean(span=span, adjust=True)
    elif op == "ewm_std":
        expr = col.ewm_std(span=span, adjust=True)
    else:
        raise ValueError(f"Unsupported ewm op={op!r}")
    if partition_by:
        expr = expr.over(partition_by)
    return expr


def apply_ewm_ops_via_polars(
    df: pd.DataFrame,
    *,
    specs: list[tuple[str, str, int, str]],
    partition_by: list[str],
    pandas_fallback,
) -> pd.DataFrame:
    """``specs`` = ``(feature, op, period, output_column)``."""
    if not specs:
        return df

    def _run() -> pd.DataFrame:
        from chain_replay_ml.frame_backend import require_polars

        pl = require_polars()
        pl_df = _to_polars(df)
        exprs = [
            _ewm_expr(pl, feat, op, per, partition_by=partition_by).alias(out_col)
            for feat, op, per, out_col in specs
        ]
        return _from_polars(pl_df.with_columns(exprs))

    return _try_polars(_run, pandas_fallback, df)


def apply_return_ops_via_polars(
    df: pd.DataFrame,
    *,
    specs: list[tuple[str, int, str]],
    partition_by: list[str],
    scale: float,
    denom_eps: float,
    pandas_fallback,
) -> pd.DataFrame:
    """``specs`` = ``(feature, rows, output_column)`` → ``scale * (x - lag) / lag``."""
    if not specs:
        return df

    def _run() -> pd.DataFrame:
        from chain_replay_ml.frame_backend import require_polars

        pl = require_polars()
        pl_df = _to_polars(df)
        exprs = []
        for feat, rows, out_col in specs:
            cur = pl.col(feat).cast(pl.Float64, strict=False)
            lag = cur.shift(int(rows))
            if partition_by:
                lag = lag.over(partition_by)
            if denom_eps > 0:
                expr = (cur - lag) / (lag + float(denom_eps))
            else:
                denom = pl.when(lag == 0.0).then(None).otherwise(lag)
                expr = (cur - lag) / denom
            if scale != 1.0:
                expr = expr * float(scale)
            exprs.append(expr.alias(out_col))
        return _from_polars(pl_df.with_columns(exprs))

    return _try_polars(_run, pandas_fallback, df)


def apply_diff_clip_ops_via_polars(
    df: pd.DataFrame,
    *,
    specs: list[tuple[str, int, str]],
    partition_by: list[str],
    clip_min: float,
    clip_max: float | None,
    pandas_fallback,
) -> pd.DataFrame:
    """``specs`` = ``(feature, rows, output_column)`` → ``clip(x - lag)``."""
    if not specs:
        return df

    def _run() -> pd.DataFrame:
        from chain_replay_ml.frame_backend import require_polars

        pl = require_polars()
        pl_df = _to_polars(df)
        exprs = []
        for feat, rows, out_col in specs:
            cur = pl.col(feat).cast(pl.Float64, strict=False)
            lag = cur.shift(int(rows))
            if partition_by:
                lag = lag.over(partition_by)
            diff = cur - lag
            if clip_max is None:
                expr = diff.clip(lower_bound=float(clip_min))
            else:
                expr = diff.clip(lower_bound=float(clip_min), upper_bound=float(clip_max))
            exprs.append(expr.alias(out_col))
        return _from_polars(pl_df.with_columns(exprs))

    return _try_polars(_run, pandas_fallback, df)


def apply_derived_ops_via_polars(
    df: pd.DataFrame,
    *,
    outputs: list[tuple[str, list[tuple[str, int, float]]]],
    partition_by: list[str],
    pandas_fallback,
) -> pd.DataFrame:
    """``outputs`` = ``(out_col, [(feature, rows, coeff), ...])``."""
    if not outputs:
        return df

    def _run() -> pd.DataFrame:
        from chain_replay_ml.frame_backend import require_polars

        pl = require_polars()
        pl_df = _to_polars(df)
        exprs = []
        for out_col, terms in outputs:
            pieces = []
            for feat, rows, coeff in terms:
                shifted = pl.col(feat).cast(pl.Float64, strict=False).shift(int(rows))
                if partition_by:
                    shifted = shifted.over(partition_by)
                pieces.append(shifted * float(coeff))
            expr = pieces[0]
            for piece in pieces[1:]:
                expr = expr + piece
            exprs.append(expr.alias(out_col))
        return _from_polars(pl_df.with_columns(exprs))

    return _try_polars(_run, pandas_fallback, df)


def apply_anchor_return_via_polars(
    df: pd.DataFrame,
    *,
    specs: list[tuple[str, str]],
    partition_by: list[str],
    scale: float,
    pandas_fallback,
) -> pd.DataFrame:
    """``specs`` = ``(feature, out_col)`` → ``scale * (x - first) / first`` (first>0)."""
    if not specs:
        return df

    def _run() -> pd.DataFrame:
        from chain_replay_ml.frame_backend import require_polars

        pl = require_polars()
        pl_df = _to_polars(df)
        exprs = []
        for feat, out_col in specs:
            cur = pl.col(feat).cast(pl.Float64, strict=False)
            # first non-null per partition (or whole frame)
            if partition_by:
                anchor = cur.drop_nulls().first().over(partition_by)
            else:
                anchor = cur.drop_nulls().first()
            denom = pl.when(anchor > 0.0).then(anchor).otherwise(None)
            exprs.append((float(scale) * (cur - anchor) / denom).alias(out_col))
        return _from_polars(pl_df.with_columns(exprs))

    return _try_polars(_run, pandas_fallback, df)


def _interaction_expr(
    pl: Any,
    left: str,
    right: str,
    op: str,
    *,
    scale: float,
    div_zero: str,
    eps: float,
) -> Any:
    l = pl.col(left).cast(pl.Float64, strict=False)
    r = pl.col(right).cast(pl.Float64, strict=False)
    if op == "multiply":
        expr = l * r
    elif op == "add":
        expr = l + r
    elif op == "subtract":
        expr = l - r
    elif op == "min":
        expr = pl.min_horizontal(l, r)
    elif op == "max":
        expr = pl.max_horizontal(l, r)
    elif op == "absolute_difference":
        expr = (l - r).abs()
    elif op == "divide":
        zero = r.abs() <= float(eps)
        if div_zero == "fail":
            # Evaluate later via pandas fallback if zeros present — signal here.
            safe = pl.when(zero).then(None).otherwise(r)
            expr = l / safe
        elif div_zero == "zero":
            expr = pl.when(zero).then(0.0).otherwise(l / r)
        else:
            safe = pl.when(zero).then(None).otherwise(r)
            expr = l / safe
    else:
        raise ValueError(f"Unsupported interaction op={op!r}")
    if scale != 1.0:
        expr = expr * float(scale)
    return expr


def apply_interaction_via_polars(
    df: pd.DataFrame,
    *,
    pairs: list[dict[str, Any]],
    pandas_fallback,
) -> pd.DataFrame:
    """Apply interaction pairs sequentially (supports intra-step chaining)."""
    if not pairs:
        return df

    def _run() -> pd.DataFrame:
        from chain_replay_ml.frame_backend import require_polars

        pl = require_polars()
        pl_df = _to_polars(df)
        # Fail-fast for div_zero=fail if zeros present.
        for pair in pairs:
            if pair["op"] == "divide" and str(pair.get("div_zero") or "null") == "fail":
                right = pair["right"]
                eps = float(pair.get("eps", 1e-12))
                # Check against current frame (including prior pair outputs).
                if right in pl_df.columns:
                    bad = pl_df.select(
                        (pl.col(right).cast(pl.Float64, strict=False).abs() <= eps).any()
                    ).item()
                    if bad:
                        raise ValueError("divide-by-zero (div_zero=fail)")
            expr = _interaction_expr(
                pl,
                pair["left"],
                pair["right"],
                pair["op"],
                scale=float(pair.get("scale", 1.0)),
                div_zero=str(pair.get("div_zero") or "null"),
                eps=float(pair.get("eps", 1e-12)),
            ).alias(pair["output"])
            pl_df = pl_df.with_columns(expr)
        return _from_polars(pl_df)

    return _try_polars(_run, pandas_fallback, df)


def apply_rolling_stats_via_polars(
    df: pd.DataFrame,
    *,
    specs: list[tuple[str, str, int, str]],
    ddof: int,
    partition_by: list[str],
    pandas_fallback,
) -> pd.DataFrame:
    """``specs`` = ``(feature, stat, window, out_col)`` for mean/std/zscore."""
    if not specs:
        return df

    def _run() -> pd.DataFrame:
        from chain_replay_ml.frame_backend import require_polars

        pl = require_polars()
        pl_df = _to_polars(df)
        exprs = []
        for feat, stat, win, out_col in specs:
            col = pl.col(feat).cast(pl.Float64, strict=False)
            w = int(win)
            mean = col.rolling_mean(window_size=w, min_samples=w)
            std = col.rolling_std(window_size=w, min_samples=w, ddof=int(ddof))
            if partition_by:
                mean = mean.over(partition_by)
                std = std.over(partition_by)
            if stat == "mean":
                expr = mean
            elif stat == "std":
                expr = std
            elif stat == "zscore":
                safe_std = pl.when(std == 0.0).then(None).otherwise(std)
                expr = (col - mean) / safe_std
            else:
                raise ValueError(f"Unsupported rolling_statistics stat={stat!r}")
            exprs.append(expr.alias(out_col))
        return _from_polars(pl_df.with_columns(exprs))

    return _try_polars(_run, pandas_fallback, df)


def apply_rolling_ohlc_via_polars(
    df: pd.DataFrame,
    *,
    jobs: list[dict[str, Any]],
    partition_by: list[str],
    range_eps: float,
    pandas_fallback,
) -> pd.DataFrame:
    """Each job: feature, rows, outputs dict out_name→column."""
    if not jobs:
        return df

    def _run() -> pd.DataFrame:
        from chain_replay_ml.frame_backend import require_polars

        pl = require_polars()
        pl_df = _to_polars(df)
        exprs = []
        for job in jobs:
            feat = job["feature"]
            rows = int(job["rows"])
            col = pl.col(feat).cast(pl.Float64, strict=False)
            open_ = col.shift(max(rows - 1, 0))
            high = col.rolling_max(window_size=rows, min_samples=rows)
            low = col.rolling_min(window_size=rows, min_samples=rows)
            if partition_by:
                open_ = open_.over(partition_by)
                high = high.over(partition_by)
                low = low.over(partition_by)
            close = col
            open_safe = pl.when(open_ == 0.0).then(None).otherwise(open_)
            high_safe = pl.when(high == 0.0).then(None).otherwise(high)
            low_safe = pl.when(low == 0.0).then(None).otherwise(low)
            range_span = high - low + float(range_eps)
            computed = {
                "body_pct": (close - open_) / open_safe * 100.0,
                "range_pct": (high - low) / open_safe * 100.0,
                "dist_high_pct": (close - high) / high_safe * 100.0,
                "dist_low_pct": (close - low) / low_safe * 100.0,
                "range_pos": (close - low) / range_span,
            }
            for out_name, out_col in job["outputs"].items():
                exprs.append(computed[out_name].alias(out_col))
        return _from_polars(pl_df.with_columns(exprs))

    return _try_polars(_run, pandas_fallback, df)


def apply_ohlc_history_via_polars(
    df: pd.DataFrame,
    *,
    plans: list[dict[str, Any]],
    partition_by: list[str],
    build_history_columns,
    pandas_fallback,
) -> pd.DataFrame:
    """OHLC candle history: NumPy kernel per partition, Polars frame assembly.

    Each plan: ``feature``, ``period_rows``, ``history_len``, ``fields``,
    ``name_fn(h, field) -> column``.
    """
    if not plans:
        return df

    def _run() -> pd.DataFrame:
        from chain_replay_ml.frame_backend import require_polars

        pl = require_polars()
        pl_df = _to_polars(df)
        # Pre-allocate output columns as nulls, then fill via numpy per group.
        out_names: list[str] = []
        for plan in plans:
            for h in range(1, int(plan["history_len"]) + 1):
                for fld in plan["fields"]:
                    out_names.append(plan["name_fn"](h, fld))
        if out_names:
            pl_df = pl_df.with_columns([pl.lit(None).cast(pl.Float64).alias(n) for n in out_names])

        if partition_by:
            # Work in pandas groups for index fidelity, assemble columns then to polars once.
            # Still uses NumPy candle kernel; partition loop is the remaining pandas edge.
            pdf = _from_polars(pl_df)
            grouped = pdf.groupby(partition_by, sort=False, group_keys=False)
            for plan in plans:
                feat = plan["feature"]
                for _, gdf in grouped:
                    hist_map = build_history_columns(
                        gdf[feat].to_numpy(dtype=float, copy=False),
                        period_rows=int(plan["period_rows"]),
                        history_len=int(plan["history_len"]),
                        fields=list(plan["fields"]),
                    )
                    positions = gdf.index
                    for h in range(1, int(plan["history_len"]) + 1):
                        for fld in plan["fields"]:
                            name = plan["name_fn"](h, fld)
                            pdf.loc[positions, name] = hist_map[(h, fld)]
            return pdf

        pdf = _from_polars(pl_df)
        for plan in plans:
            hist_map = build_history_columns(
                pdf[plan["feature"]].to_numpy(dtype=float, copy=False),
                period_rows=int(plan["period_rows"]),
                history_len=int(plan["history_len"]),
                fields=list(plan["fields"]),
            )
            for h in range(1, int(plan["history_len"]) + 1):
                for fld in plan["fields"]:
                    pdf[plan["name_fn"](h, fld)] = hist_map[(h, fld)]
        return pdf

    return _try_polars(_run, pandas_fallback, df)


def _norm_expr(
    pl: Any,
    feature: str,
    method: str,
    window: int | None,
    *,
    ddof: int,
    partition_by: list[str],
) -> Any:
    col = pl.col(feature).cast(pl.Float64, strict=False)
    if method == "zscore_expanding":
        # Expanding mean/std via cumulative moments; min_periods=2 → null at first row.
        ones = col.is_not_null().cast(pl.Float64)
        filled = col.fill_null(0.0)
        if partition_by:
            n = ones.cum_sum().over(partition_by)
            csum = filled.cum_sum().over(partition_by)
            csum2 = (filled ** 2).cum_sum().over(partition_by)
        else:
            n = ones.cum_sum()
            csum = filled.cum_sum()
            csum2 = (filled ** 2).cum_sum()
        mean = csum / n
        denom = (n - float(ddof)).clip(lower_bound=1e-15)
        var = (csum2 - (csum * csum) / n) / denom
        std = var.sqrt()
        safe = pl.when((n < 2.0) | (std == 0.0)).then(None).otherwise(std)
        return (col - mean) / safe

    win = int(window or 0)
    if method == "zscore_rolling":
        mean = col.rolling_mean(window_size=win, min_samples=win)
        std = col.rolling_std(window_size=win, min_samples=win, ddof=int(ddof))
        if partition_by:
            mean = mean.over(partition_by)
            std = std.over(partition_by)
        safe = pl.when(std == 0.0).then(None).otherwise(std)
        return (col - mean) / safe
    if method == "robust":
        med = col.rolling_median(window_size=win, min_samples=win)
        q1 = col.rolling_quantile(quantile=0.25, window_size=win, min_samples=win)
        q3 = col.rolling_quantile(quantile=0.75, window_size=win, min_samples=win)
        if partition_by:
            med = med.over(partition_by)
            q1 = q1.over(partition_by)
            q3 = q3.over(partition_by)
        iqr = q3 - q1
        safe = pl.when(iqr == 0.0).then(None).otherwise(iqr)
        return (col - med) / safe
    if method == "minmax":
        lo = col.rolling_min(window_size=win, min_samples=win)
        hi = col.rolling_max(window_size=win, min_samples=win)
        if partition_by:
            lo = lo.over(partition_by)
            hi = hi.over(partition_by)
        span = hi - lo
        safe = pl.when(span == 0.0).then(None).otherwise(span)
        return (col - lo) / safe
    if method in ("percentile_rank", "quantile_rank"):
        n_bins = 10

        def _pct(s: Any) -> float:
            arr = s.drop_nulls()
            if arr.len() == 0:
                return float("nan")
            cur = s[-1]
            if cur is None:
                return float("nan")
            pct = float((arr <= cur).sum()) / float(arr.len())
            if method == "percentile_rank":
                return pct
            return float(int(min(n_bins - 1, max(0, np.floor(pct * n_bins)))))

        expr = col.rolling_map(_pct, window_size=win, min_samples=win)
        if partition_by:
            expr = expr.over(partition_by)
        return expr
    raise ValueError(f"Unsupported normalization method={method!r}")


def apply_normalization_via_polars(
    df: pd.DataFrame,
    *,
    specs: list[tuple[str, str, int | None, str]],
    ddof: int,
    partition_by: list[str],
    pandas_fallback,
) -> pd.DataFrame:
    """``specs`` = ``(feature, method, window|None, out_col)``."""
    if not specs:
        return df

    def _run() -> pd.DataFrame:
        from chain_replay_ml.frame_backend import require_polars

        pl = require_polars()
        pl_df = _to_polars(df)
        exprs = [
            _norm_expr(
                pl, feat, meth, win, ddof=ddof, partition_by=partition_by
            ).alias(out_col)
            for feat, meth, win, out_col in specs
        ]
        return _from_polars(pl_df.with_columns(exprs))

    return _try_polars(_run, pandas_fallback, df)


def _regime_expr(
    pl: Any,
    feature: str,
    method: str,
    window: int | None,
    *,
    threshold: float,
    low: float,
    high: float,
    edges: list[float],
    n_bins: int,
    partition_by: list[str],
) -> Any:
    col = pl.col(feature).cast(pl.Float64, strict=False)
    if method == "binary_threshold":
        return pl.when(col.is_null()).then(None).otherwise(
            (col >= float(threshold)).cast(pl.Float64)
        )
    if method == "ternary_state":
        return (
            pl.when(col.is_null())
            .then(None)
            .when(col < float(low))
            .then(-1.0)
            .when(col > float(high))
            .then(1.0)
            .otherwise(0.0)
        )
    if method == "threshold_bucket":
        # digitize left-open/right: code = number of edges <= value? pandas right=False
        # np.digitize(x, bins, right=False): returns i such that bins[i-1] <= x < bins[i]
        expr = pl.lit(0.0)
        for i, edge in enumerate(edges):
            expr = pl.when(col >= float(edge)).then(float(i + 1)).otherwise(expr)
        return pl.when(col.is_null()).then(None).otherwise(expr)
    win = int(window or 0)
    if method == "quantile_bucket":

        def _code(s: Any) -> float:
            arr = s.drop_nulls()
            if arr.len() == 0:
                return float("nan")
            cur = s[-1]
            if cur is None:
                return float("nan")
            pct = float((arr <= cur).sum()) / float(arr.len())
            return float(int(min(n_bins - 1, max(0, np.floor(pct * n_bins)))))

        expr = col.rolling_map(_code, window_size=win, min_samples=win)
        if partition_by:
            expr = expr.over(partition_by)
        return expr
    if method == "equal_width":

        def _code(s: Any) -> float:
            arr = s.drop_nulls()
            if arr.len() == 0:
                return float("nan")
            cur = s[-1]
            if cur is None:
                return float("nan")
            lo = float(arr.min())
            hi = float(arr.max())
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                return 0.0
            pct = (float(cur) - lo) / (hi - lo)
            return float(int(min(n_bins - 1, max(0, np.floor(pct * n_bins)))))

        expr = col.rolling_map(_code, window_size=win, min_samples=win)
        if partition_by:
            expr = expr.over(partition_by)
        return expr
    raise ValueError(f"Unsupported regime method={method!r}")


def apply_regime_via_polars(
    df: pd.DataFrame,
    *,
    specs: list[tuple[str, str, int | None, str]],
    threshold: float,
    low: float,
    high: float,
    edges: list[float],
    n_bins: int,
    partition_by: list[str],
    pandas_fallback,
) -> pd.DataFrame:
    """``specs`` = ``(feature, method, window|None, out_col)``."""
    if not specs:
        return df

    def _run() -> pd.DataFrame:
        from chain_replay_ml.frame_backend import require_polars

        pl = require_polars()
        pl_df = _to_polars(df)
        exprs = [
            _regime_expr(
                pl,
                feat,
                meth,
                win,
                threshold=threshold,
                low=low,
                high=high,
                edges=edges,
                n_bins=n_bins,
                partition_by=partition_by,
            ).alias(out_col)
            for feat, meth, win, out_col in specs
        ]
        return _from_polars(pl_df.with_columns(exprs))

    return _try_polars(_run, pandas_fallback, df)
