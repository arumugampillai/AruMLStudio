"""Strike Prediction Dashboard — series prep helpers (no Tk).

Builds chart-ready series from prediction_dataset rows and optional
strategy_trades alignments.
"""

from __future__ import annotations

import bisect
from typing import Any, Sequence

import pandas as pd

from chain_replay_ml.training.prediction_packages import PROBABILITY_OUTPUT_COLUMNS

# Soft probability columns first (continuous), then TB, then binary confidence_*.
_CONFIDENCE_SOFT_PRIORITY: tuple[str, ...] = (
    "pred_prob_up_2pct_5m",
    "pred_prob_up_3pct_5m",
    "pred_prob_up_4pct_5m",
    "pred_prob_up_5pct_5m",
    "pred_prob_up_6pct_5m",
    "pred_prob_up_gt6pct_5m",
    "tb_pred_probability",
)

EMA_SPAN_CHOICES: tuple[int, ...] = (5, 10)
DEFAULT_EMA_SPAN = 5
POSITION_SIZE_MAX = 5


def resolve_confidence_column(
    available_columns: Sequence[str] | set[str],
    *,
    preferred: str | None = None,
) -> str | None:
    """Pick the best confidence / probability column present in the dataset.

    Preference order:
    1. Explicit ``preferred`` if present
    2. Soft ``pred_prob_*`` / ``tb_pred_probability`` (ladder order)
    3. Any other ``pred_prob_*`` column
    4. ``confidence_*_pred`` binary inference columns (``target_hit`` first)
    """
    avail = {str(c) for c in available_columns}
    if preferred and preferred in avail:
        return preferred
    for col in _CONFIDENCE_SOFT_PRIORITY:
        if col in avail:
            return col
    for col in PROBABILITY_OUTPUT_COLUMNS:
        if col in avail and col not in _CONFIDENCE_SOFT_PRIORITY:
            return col
    soft = sorted(c for c in avail if c.startswith("pred_prob_"))
    if soft:
        return soft[0]
    preferred_binary = (
        "confidence_target_hit_pred",
        "confidence_rr_1_1_pred",
        "confidence_trade_winner_pred",
    )
    for col in preferred_binary:
        if col in avail:
            return col
    binary = sorted(
        c for c in avail if c.startswith("confidence_") and c.endswith("_pred")
    )
    return binary[0] if binary else None


def apply_ema(values: Sequence[float | None], span: int) -> list[float | None]:
    """EMA overlay via ``ml_phase1.indicators.ema`` (ewm span, adjust=False)."""
    span_i = int(span) if int(span) in EMA_SPAN_CHOICES else DEFAULT_EMA_SPAN
    if not values:
        return []
    series = pd.Series(
        [float(v) if v is not None else float("nan") for v in values],
        dtype="float64",
    )
    from ml_phase1.indicators import ema

    out = ema(series, span_i)
    result: list[float | None] = []
    for v in out.tolist():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            result.append(None)
        else:
            result.append(float(v))
    return result


def prediction_error_series(
    predicted: Sequence[float | None],
    actual: Sequence[float | None],
) -> list[float | None]:
    """Error = Predicted LTP − Actual LTP (None when either side missing)."""
    n = min(len(predicted), len(actual))
    out: list[float | None] = []
    for i in range(n):
        p, a = predicted[i], actual[i]
        if p is None or a is None:
            out.append(None)
            continue
        try:
            out.append(float(p) - float(a))
        except (TypeError, ValueError):
            out.append(None)
    return out


ERROR_QUANTILE_PROBS: tuple[float, ...] = tuple(i / 100.0 for i in range(1, 100))
DEFAULT_RECENT_SAMPLES = 30
DEFAULT_DOWNSAMPLE_TARGET = 300
# Relative MAE change below this → "stable" magnitude trend.
_TREND_STABLE_REL = 0.10
# Absolute mean-bias change (₹) below this → "stable" bias trend.
_TREND_BIAS_STABLE_ABS = 0.25


def _finite_error_values(errors: Sequence[float | None]) -> list[float]:
    out: list[float] = []
    for v in errors:
        fv = _finite(v)
        if fv is not None:
            out.append(fv)
    return out


def prediction_error_mae(errors: Sequence[float | None]) -> float | None:
    vals = _finite_error_values(errors)
    if not vals:
        return None
    return sum(abs(v) for v in vals) / len(vals)


def prediction_error_rmse(errors: Sequence[float | None]) -> float | None:
    vals = _finite_error_values(errors)
    if not vals:
        return None
    return (sum(v * v for v in vals) / len(vals)) ** 0.5


def _error_quantile_key(prob: float) -> str:
    """Map probability in (0, 1] to a zero-padded key like ``p01``…``p99``."""
    return f"p{int(round(float(prob) * 100)):02d}"


def prediction_error_quantiles(
    errors: Sequence[float | None],
    *,
    probs: Sequence[float] = ERROR_QUANTILE_PROBS,
) -> dict[str, dict[str, float | int | None]]:
    """Empirical quantiles of signed error (P01…P99 by default).

    Each key maps to ``{"error": <₹ quantile or None>, "samples": <int>}``.
    ``samples`` is the count of finite errors in the half-open bin
    ``(P{i-1}, Pi]`` with ``P00 = -∞``. For large *n* each bin is ~*n*/100;
    values above P99 are outside these bins.
    """
    vals = sorted(_finite_error_values(errors))
    out: dict[str, dict[str, float | int | None]] = {
        _error_quantile_key(p): {"error": None, "samples": 0} for p in probs
    }
    if not vals:
        return out
    n = len(vals)
    # Compute quantile thresholds first, then bin counts on the sorted series.
    thresholds: list[tuple[str, float]] = []
    for p in probs:
        key = _error_quantile_key(p)
        if n == 1:
            qv = vals[0]
        else:
            # Linear interpolation between closest ranks (0-based).
            pos = float(p) * (n - 1)
            lo = int(pos)
            hi = min(lo + 1, n - 1)
            frac = pos - lo
            qv = vals[lo] * (1.0 - frac) + vals[hi] * frac
        thresholds.append((key, float(qv)))

    prev = float("-inf")
    for key, qv in thresholds:
        # Count vals in (prev, qv] via bisect on the sorted list.
        right = bisect.bisect_right(vals, qv)
        left = 0 if prev == float("-inf") else bisect.bisect_right(vals, prev)
        out[key] = {"error": qv, "samples": right - left}
        prev = qv
    return out


def prediction_error_trend_flags(
    errors: Sequence[float | None],
    *,
    recent_n: int | None = None,
) -> dict[str, Any]:
    """Compare earlier vs later errors for magnitude and bias trend flags.

    When ``recent_n`` is set and there are enough points, compare the last
    ``recent_n`` finite errors vs the preceding ``recent_n``. Otherwise split
    finite errors into first half vs second half.
    """
    vals = _finite_error_values(errors)
    empty = {
        "magnitude": "stable",
        "magnitude_label": "Error stable",
        "bias": "stable",
        "bias_label": "Bias stable",
        "earlier_mae": None,
        "later_mae": None,
        "earlier_mean": None,
        "later_mean": None,
        "mode": "empty",
        "n_earlier": 0,
        "n_later": 0,
    }
    if len(vals) < 4:
        return empty

    if recent_n is not None and int(recent_n) > 0 and len(vals) >= int(recent_n) * 2:
        rn = int(recent_n)
        earlier, later = vals[-(2 * rn) : -rn], vals[-rn:]
        mode = f"recent_{rn}"
    else:
        mid = len(vals) // 2
        earlier, later = vals[:mid], vals[mid:]
        mode = "halves"

    def _mae(xs: list[float]) -> float:
        return sum(abs(x) for x in xs) / len(xs)

    def _mean(xs: list[float]) -> float:
        return sum(xs) / len(xs)

    earlier_mae, later_mae = _mae(earlier), _mae(later)
    earlier_mean, later_mean = _mean(earlier), _mean(later)

    base = max(earlier_mae, 1e-9)
    rel = (later_mae - earlier_mae) / base
    if rel > _TREND_STABLE_REL:
        magnitude = "expanding"
        magnitude_label = "Error expanding"
    elif rel < -_TREND_STABLE_REL:
        magnitude = "contracting"
        magnitude_label = "Error contracting"
    else:
        magnitude = "stable"
        magnitude_label = "Error stable"

    bias_delta = later_mean - earlier_mean
    if bias_delta > _TREND_BIAS_STABLE_ABS:
        bias = "more_optimistic"
        bias_label = "Bias more optimistic recently"
    elif bias_delta < -_TREND_BIAS_STABLE_ABS:
        bias = "more_pessimistic"
        bias_label = "Bias more pessimistic recently"
    else:
        bias = "stable"
        bias_label = "Bias stable"

    return {
        "magnitude": magnitude,
        "magnitude_label": magnitude_label,
        "bias": bias,
        "bias_label": bias_label,
        "earlier_mae": earlier_mae,
        "later_mae": later_mae,
        "earlier_mean": earlier_mean,
        "later_mean": later_mean,
        "mode": mode,
        "n_earlier": len(earlier),
        "n_later": len(later),
    }


def series_ltp_stats(values: Sequence[float | None]) -> dict[str, float | int | None]:
    """Min / max / mean / median over finite series values."""
    vals = _finite_error_values(values)
    n = len(vals)
    if not n:
        return {"n": 0, "min": None, "max": None, "mean": None, "median": None}
    ordered = sorted(vals)
    mean = sum(vals) / n
    mid = n // 2
    if n % 2 == 1:
        median = ordered[mid]
    else:
        median = (ordered[mid - 1] + ordered[mid]) / 2.0
    return {
        "n": n,
        "min": ordered[0],
        "max": ordered[-1],
        "mean": mean,
        "median": median,
    }


def ltp_vs_prediction_summary(
    actual: Sequence[float | None],
    predicted: Sequence[float | None],
) -> dict[str, Any]:
    """Summary blocks for the Actual vs Prediction tab.

    Premium Range / Spread are derived from actual LTP (min → max, max − min).
    """
    actual_stats = series_ltp_stats(actual)
    predicted_stats = series_ltp_stats(predicted)
    prem_min = actual_stats["min"]
    prem_max = actual_stats["max"]
    spread: float | None = None
    if prem_min is not None and prem_max is not None:
        spread = float(prem_max) - float(prem_min)
    return {
        "actual": actual_stats,
        "predicted": predicted_stats,
        "premium_range": {
            "min": prem_min,
            "max": prem_max,
            "spread": spread,
            "n": actual_stats["n"],
        },
    }


def prediction_error_summary(
    errors: Sequence[float | None],
    error_ema: Sequence[float | None] | None = None,
    *,
    recent_n: int | None = None,
) -> dict[str, Any]:
    """Stats-first summary for the Prediction Error tab."""
    vals = _finite_error_values(errors)
    n = len(vals)
    mean_err = (sum(vals) / n) if n else None
    mae = prediction_error_mae(errors)
    rmse = prediction_error_rmse(errors)
    if n:
        optimistic = sum(1 for v in vals if v > 0)
        pessimistic = sum(1 for v in vals if v < 0)
        pct_optimistic = 100.0 * optimistic / n
        pct_pessimistic = 100.0 * pessimistic / n
        std_err = (sum((v - float(mean_err)) ** 2 for v in vals) / n) ** 0.5
    else:
        pct_optimistic = pct_pessimistic = std_err = None

    latest_error: float | None = None
    for v in reversed(list(errors)):
        fv = _finite(v)
        if fv is not None:
            latest_error = fv
            break

    latest_ema: float | None = None
    if error_ema is not None:
        for v in reversed(list(error_ema)):
            fv = _finite(v)
            if fv is not None:
                latest_ema = fv
                break

    quantiles = prediction_error_quantiles(errors)
    trends = prediction_error_trend_flags(errors, recent_n=recent_n)

    return {
        "n": n,
        "mean_error": mean_err,
        "mae": mae,
        "rmse": rmse,
        "pct_optimistic": pct_optimistic,
        "pct_pessimistic": pct_pessimistic,
        "std_error": std_err,
        "latest_error": latest_error,
        "latest_error_ema": latest_ema,
        "quantiles": quantiles,
        "trends": trends,
    }


def _series_delta(
    a: Sequence[float | None],
    b: Sequence[float | None],
) -> list[float | None]:
    """Element-wise ``a − b`` (None when either side missing)."""
    n = min(len(a), len(b))
    out: list[float | None] = []
    for i in range(n):
        left, right = a[i], b[i]
        if left is None or right is None:
            out.append(None)
            continue
        try:
            out.append(float(left) - float(right))
        except (TypeError, ValueError):
            out.append(None)
    return out


def resolve_future_actual_series(
    current_ltp: Sequence[float | None],
    actual_future_ltp: Sequence[float | None] | None,
) -> list[float | None]:
    """Future actual for error / recent samples.

    Prefer ``actual_future_ltp`` when any finite value exists; otherwise fall
    back to ``current_ltp`` (the series historically used for prediction error).
    """
    current = list(current_ltp)
    if not actual_future_ltp:
        return current
    future = [_finite(v) for v in actual_future_ltp]
    if len(future) < len(current):
        future.extend([None] * (len(current) - len(future)))
    elif len(future) > len(current):
        future = future[: len(current)]
    if any(v is not None for v in future):
        return future
    return current


def recent_prediction_error_rows(
    timestamps: Sequence[Any],
    current_ltp: Sequence[float | None],
    future_actual: Sequence[float | None],
    future_pred: Sequence[float | None],
    errors: Sequence[float | None],
    error_ema: Sequence[float | None],
    *,
    confidence: Sequence[float | None] | None = None,
    confidence_ema: Sequence[float | None] | None = None,
    n: int = DEFAULT_RECENT_SAMPLES,
) -> list[dict[str, Any]]:
    """Last ``n`` rows (newest last) for the recent-samples table.

    Columns: timestamp, current_ltp, future_actual, future_pred,
    actual_delta (future_actual − current), pred_delta (future_pred − current),
    error (future_pred − future_actual), error_ema, confidence, confidence_ema.
    """
    limit = max(1, int(n))
    length = min(
        len(timestamps),
        len(current_ltp),
        len(future_actual),
        len(future_pred),
        len(errors),
        len(error_ema),
    )
    if length <= 0:
        return []
    conf = list(confidence) if confidence is not None else []
    conf_ema = list(confidence_ema) if confidence_ema is not None else []
    start = max(0, length - limit)
    actual_delta = _series_delta(future_actual, current_ltp)
    pred_delta = _series_delta(future_pred, current_ltp)
    rows: list[dict[str, Any]] = []
    for i in range(start, length):
        cur = _finite(current_ltp[i]) if i < len(current_ltp) else None
        fact = _finite(future_actual[i]) if i < len(future_actual) else None
        fpred = _finite(future_pred[i]) if i < len(future_pred) else None
        rows.append(
            {
                "index": i,
                "timestamp": timestamps[i] if i < len(timestamps) else None,
                "current_ltp": cur,
                "future_actual": fact,
                "future_pred": fpred,
                # Backward-compatible aliases (Future Actual / Future Pred).
                "actual": fact,
                "predicted": fpred,
                "actual_delta": (
                    _finite(actual_delta[i]) if i < len(actual_delta) else None
                ),
                "pred_delta": (
                    _finite(pred_delta[i]) if i < len(pred_delta) else None
                ),
                "error": _finite(errors[i]) if i < len(errors) else None,
                "error_ema": _finite(error_ema[i]) if i < len(error_ema) else None,
                "confidence": (
                    _finite(conf[i]) if i < len(conf) else None
                ),
                "confidence_ema": (
                    _finite(conf_ema[i]) if i < len(conf_ema) else None
                ),
            }
        )
    return rows


def downsample_series(
    values: Sequence[float | None],
    *,
    target: int = DEFAULT_DOWNSAMPLE_TARGET,
    timestamps: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Downsample a series to ~``target`` points via equal-width bucket means.

    Returns ``values`` (bucket means or None), optional ``timestamps`` (bucket
    mid), and ``source_indices`` (representative original index per bucket) for
    crosshair mapping back to the full series.
    """
    n = len(values)
    tgt = max(2, int(target))
    if n <= tgt:
        idxs = list(range(n))
        return {
            "values": [(_finite(v)) for v in values],
            "timestamps": list(timestamps) if timestamps is not None else None,
            "source_indices": idxs,
            "n_source": n,
            "n_downsampled": n,
        }

    bucket = n / float(tgt)
    out_vals: list[float | None] = []
    out_ts: list[Any] | None = [] if timestamps is not None else None
    source_indices: list[int] = []
    for b in range(tgt):
        start = int(b * bucket)
        end = int((b + 1) * bucket)
        if end <= start:
            end = min(start + 1, n)
        end = min(end, n)
        chunk = [_finite(values[i]) for i in range(start, end)]
        finite = [v for v in chunk if v is not None]
        out_vals.append(sum(finite) / len(finite) if finite else None)
        mid = start + (end - start - 1) // 2
        source_indices.append(mid)
        if out_ts is not None and timestamps is not None:
            out_ts.append(timestamps[mid] if mid < len(timestamps) else None)
    return {
        "values": out_vals,
        "timestamps": out_ts,
        "source_indices": source_indices,
        "n_source": n,
        "n_downsampled": len(out_vals),
    }


def prediction_gap_series(
    predicted_ema: Sequence[float | None],
    actual: Sequence[float | None],
) -> list[float | None]:
    """Gap = Prediction EMA − Actual LTP (None when either side missing)."""
    n = min(len(predicted_ema), len(actual))
    out: list[float | None] = []
    for i in range(n):
        e, a = predicted_ema[i], actual[i]
        if e is None or a is None:
            out.append(None)
            continue
        try:
            out.append(float(e) - float(a))
        except (TypeError, ValueError):
            out.append(None)
    return out


def ema_slope_series(
    ema_values: Sequence[float | None],
    *,
    window: int = 1,
) -> list[float | None]:
    """Simple slope of an EMA: ``EMA[t] − EMA[t − window]`` (default window=1)."""
    win = max(1, int(window))
    out: list[float | None] = []
    for i, cur in enumerate(ema_values):
        if i < win:
            out.append(None)
            continue
        prev = ema_values[i - win]
        if cur is None or prev is None:
            out.append(None)
            continue
        try:
            out.append(float(cur) - float(prev))
        except (TypeError, ValueError):
            out.append(None)
    return out


def series_index_from_x(
    x: float,
    *,
    pad: float,
    inner_w: float,
    n_points: int,
) -> int | None:
    """Map canvas x-coordinate to nearest series index (0 … n_points−1)."""
    n = int(n_points)
    if n <= 0 or inner_w <= 0:
        return None
    if n == 1:
        return 0
    rel = (float(x) - float(pad)) / float(inner_w)
    idx = int(round(rel * (n - 1)))
    if idx < 0:
        return 0
    if idx >= n:
        return n - 1
    return idx


def index_for_timestamp(
    timestamps: Sequence[Any],
    cursor_ts: Any,
) -> int | None:
    """Nearest index for ``cursor_ts`` in a timestamp sequence (exact or closest)."""
    if not timestamps or cursor_ts is None:
        return None
    target = _finite(cursor_ts)
    if target is None:
        # Fall back to string equality for non-numeric stamps.
        needle = str(cursor_ts)
        for i, ts in enumerate(timestamps):
            if ts is not None and str(ts) == needle:
                return i
        return None
    best_i: int | None = None
    best_d = float("inf")
    for i, ts in enumerate(timestamps):
        tv = _finite(ts)
        if tv is None:
            continue
        d = abs(tv - target)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def crosshair_detail_at_index(
    bundle: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    """Snapshot of dashboard values at a shared crosshair index."""

    def _at(key: str) -> float | None:
        vals = bundle.get(key)
        if not isinstance(vals, (list, tuple)) or index < 0 or index >= len(vals):
            return None
        return _finite(vals[index])

    timestamps = bundle.get("timestamps") or []
    ts = timestamps[index] if 0 <= index < len(timestamps) else None
    return {
        "index": index,
        "timestamp": ts,
        "actual_ltp": _at("actual_ltp"),
        "predicted_ltp": _at("predicted_ltp"),
        "predicted_ema": _at("predicted_ema"),
        "confidence": _at("confidence"),
        "confidence_ema_5": _at("confidence_ema_5"),
        "confidence_ema_10": _at("confidence_ema_10"),
        "confidence_ema": _at("confidence_ema"),
        "error": _at("error"),
        "gap": _at("gap"),
        "regression_slope": _at("regression_ema_slope"),
    }


def _finite(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if pd.isna(x):
        return None
    return x


def rows_to_column_map(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
) -> dict[str, list[Any]]:
    """Transpose query_predictions result into column → values lists."""
    idx = {c: i for i, c in enumerate(columns)}
    out: dict[str, list[Any]] = {c: [] for c in columns}
    for row in rows:
        for c, i in idx.items():
            out[c].append(row[i] if i < len(row) else None)
    return out


def distinct_strikes(
    strike_values: Sequence[Any],
    option_types: Sequence[Any] | None = None,
) -> list[str]:
    """Sorted unique strike labels, optionally with option_type (e.g. ``24500 CE``)."""
    labels: set[str] = set()
    for i, s in enumerate(strike_values):
        if s is None or str(s).strip() == "":
            continue
        try:
            strike_txt = str(int(float(s))) if float(s) == int(float(s)) else str(float(s))
        except (TypeError, ValueError):
            strike_txt = str(s).strip()
        ot = ""
        if option_types is not None and i < len(option_types) and option_types[i] is not None:
            ot = str(option_types[i]).strip().upper()
        labels.add(f"{strike_txt} {ot}".strip() if ot else strike_txt)
    def _key(lab: str) -> tuple[float, str]:
        parts = lab.split()
        try:
            return (float(parts[0]), parts[1] if len(parts) > 1 else "")
        except (TypeError, ValueError, IndexError):
            return (0.0, lab)

    return sorted(labels, key=_key)


def parse_strike_label(label: str) -> tuple[float | None, str | None]:
    """Parse ``24500 CE`` → (24500.0, 'CE'); bare strike → (strike, None)."""
    text = str(label or "").strip()
    if not text:
        return None, None
    parts = text.split()
    try:
        strike = float(parts[0])
    except (TypeError, ValueError):
        return None, None
    ot = parts[1].upper() if len(parts) > 1 else None
    return strike, ot


def clamp_position_size(raw: Any) -> int:
    """Map trade lots/qty into discrete size 1–5 (0 = flat / unknown)."""
    try:
        n = int(float(raw))
    except (TypeError, ValueError):
        return 0
    if n <= 0:
        return 0
    return max(1, min(POSITION_SIZE_MAX, n))


def trade_position_size(trade: dict[str, Any]) -> int:
    """Derive 1–5 size from trade dict (prefer lots, else qty heuristic)."""
    meta = trade.get("meta") if isinstance(trade.get("meta"), dict) else {}
    for key in ("lots", "position_size", "size"):
        if trade.get(key) is not None:
            return clamp_position_size(trade.get(key))
        if meta.get(key) is not None:
            return clamp_position_size(meta.get(key))
    qty = trade.get("qty")
    if qty is not None:
        try:
            q = int(float(qty))
        except (TypeError, ValueError):
            return 0
        # Common lot size 65 — approximate lots then clamp.
        lots = max(1, round(q / 65.0)) if q >= 65 else (1 if q > 0 else 0)
        return clamp_position_size(lots)
    return 0


def align_trades_to_timestamps(
    timestamps: Sequence[Any],
    prediction_ids: Sequence[Any],
    trades: Sequence[dict[str, Any]],
    *,
    strike: float | None = None,
    option_type: str | None = None,
    trading_day: str | None = None,
) -> tuple[list[float], list[float | None], bool]:
    """Build position-size and cumulative P&L series aligned to prediction rows.

    Returns ``(position_sizes, cumulative_pnl_or_none, has_trade_data)``.

    - Without matching trades: sizes are all 0.0 and P&L is all ``None`` with
      ``has_trade_data=False`` (UI should show an empty / no-trade state).
    - With trades: size is 1–5 while a trade is open for the strike; P&L is
      cumulative realized ``net_pnl`` after each exit (forward-filled).
    """
    n = len(timestamps)
    sizes = [0.0] * n
    pnl: list[float | None] = [None] * n
    if n == 0:
        return sizes, pnl, False

    def _strike_match(t: dict[str, Any]) -> bool:
        if trading_day and str(t.get("trading_day") or "") != str(trading_day):
            return False
        if option_type:
            if str(t.get("option_type") or "").strip().upper() != option_type.upper():
                return False
        if strike is not None:
            ts = _finite(t.get("strike"))
            if ts is None or abs(ts - float(strike)) > 1e-6:
                return False
        return True

    matched = [t for t in trades if _strike_match(t)]
    if not matched:
        return sizes, pnl, False

    id_to_idx = {
        str(pid): i
        for i, pid in enumerate(prediction_ids)
        if pid is not None and str(pid)
    }
    ts_vals: list[float | None] = [_finite(t) for t in timestamps]

    for t in matched:
        size = float(trade_position_size(t))
        entry_id = str(t.get("entry_prediction_id") or "")
        entry_ts = _finite(t.get("entry_ts"))
        exit_ts = _finite(t.get("exit_ts"))
        start_i: int | None = id_to_idx.get(entry_id)
        if start_i is None and entry_ts is not None:
            for i, tv in enumerate(ts_vals):
                if tv is not None and tv >= entry_ts:
                    start_i = i
                    break
        if start_i is None:
            continue
        end_i = n - 1
        if exit_ts is not None:
            for i in range(start_i, n):
                tv = ts_vals[i]
                if tv is not None and tv > exit_ts:
                    end_i = max(start_i, i - 1)
                    break
        for i in range(start_i, end_i + 1):
            sizes[i] = max(sizes[i], size)

    # Cumulative realized P&L by exit order, forward-filled onto timeline.
    closed = sorted(
        (t for t in matched if _finite(t.get("exit_ts")) is not None),
        key=lambda t: float(t.get("exit_ts") or 0.0),
    )
    running = 0.0
    exit_events: list[tuple[float, float]] = []
    for t in closed:
        et = _finite(t.get("exit_ts"))
        npnl = _finite(t.get("net_pnl"))
        if et is None or npnl is None:
            continue
        running += npnl
        exit_events.append((et, running))

    if exit_events:
        ei = 0
        last: float | None = None
        for i, tv in enumerate(ts_vals):
            while ei < len(exit_events) and tv is not None and tv >= exit_events[ei][0]:
                last = exit_events[ei][1]
                ei += 1
            pnl[i] = last
        # If all exits are after the series, leave None until first exit.

    return sizes, pnl, True


def _ema_list(vals: Sequence[float | None], span: int) -> list[float | None]:
    return apply_ema(vals, span)


def _ema_chart_fill(ema_vals: Sequence[float | None]) -> list[float]:
    """Forward-fill EMA for canvas drawing (None → nan until first value)."""
    out: list[float] = []
    last = 0.0
    started = False
    for v in ema_vals:
        if v is not None:
            last = v
            started = True
            out.append(v)
        elif started:
            out.append(last)
        else:
            out.append(float("nan"))
    return out


def build_strike_chart_bundle(
    columns: Sequence[str],
    rows: Sequence[Sequence[Any]],
    *,
    ema_span: int = DEFAULT_EMA_SPAN,
    confidence_column: str | None = None,
    trades: Sequence[dict[str, Any]] | None = None,
    strike: float | None = None,
    option_type: str | None = None,
    trading_day: str | None = None,
) -> dict[str, Any]:
    """Assemble chart series for one strike selection (six dashboard tabs)."""
    colmap = rows_to_column_map(columns, rows)
    # Actual vs Prediction chart uses current LTP vs predicted future LTP.
    actual = [_finite(v) for v in colmap.get("current_ltp", [])]
    predicted = [_finite(v) for v in colmap.get("predicted_future_ltp", [])]
    actual_future_raw = [_finite(v) for v in colmap.get("actual_future_ltp", [])]
    future_actual = resolve_future_actual_series(actual, actual_future_raw)
    conf_col = resolve_confidence_column(
        columns, preferred=confidence_column
    )
    confidence: list[float | None] = []
    if conf_col and conf_col in colmap:
        confidence = [_finite(v) for v in colmap[conf_col]]
    else:
        confidence = [None] * len(actual)

    # Prediction Error = Future Pred − Future Actual (fallback: current LTP).
    error = prediction_error_series(predicted, future_actual)
    span = int(ema_span) if int(ema_span) in EMA_SPAN_CHOICES else DEFAULT_EMA_SPAN
    timestamps = colmap.get("timestamp", [])

    def _series_for_chart(vals: list[float | None]) -> list[float]:
        """Replace None with NaN for drawing (keep length)."""
        return [float(v) if v is not None else float("nan") for v in vals]

    predicted_ema_raw = _ema_list(predicted, span)
    confidence_ema_raw = _ema_list(confidence, span)
    confidence_ema_5_raw = _ema_list(confidence, 5)
    confidence_ema_10_raw = _ema_list(confidence, 10)
    error_ema_raw = _ema_list(error, span)
    gap = prediction_gap_series(predicted_ema_raw, actual)
    regression_ema_slope = ema_slope_series(predicted_ema_raw, window=1)

    sizes, pnl, has_trades = align_trades_to_timestamps(
        timestamps,
        colmap.get("prediction_id", []),
        list(trades or []),
        strike=strike,
        option_type=option_type,
        trading_day=trading_day,
    )

    ltp_summary = ltp_vs_prediction_summary(actual, predicted)
    error_summary = prediction_error_summary(error, error_ema_raw)
    recent_errors = recent_prediction_error_rows(
        timestamps,
        actual,
        future_actual,
        predicted,
        error,
        error_ema_raw,
        confidence=confidence,
        confidence_ema=confidence_ema_raw,
        n=DEFAULT_RECENT_SAMPLES,
    )
    error_downsampled = downsample_series(
        error,
        target=DEFAULT_DOWNSAMPLE_TARGET,
        timestamps=timestamps,
    )
    error_ema_downsampled = downsample_series(
        error_ema_raw,
        target=DEFAULT_DOWNSAMPLE_TARGET,
        timestamps=timestamps,
    )

    return {
        "actual_ltp": actual,
        "current_ltp": actual,
        "actual_future_ltp": list(actual_future_raw),
        "future_actual_ltp": list(future_actual),
        "predicted_ltp": predicted,
        "predicted_ema": _ema_chart_fill(predicted_ema_raw),
        "predicted_ema_raw": list(predicted_ema_raw),
        "confidence": confidence,
        "confidence_ema": _ema_chart_fill(confidence_ema_raw),
        "confidence_ema_5": _ema_chart_fill(confidence_ema_5_raw),
        "confidence_ema_10": _ema_chart_fill(confidence_ema_10_raw),
        "confidence_ema_5_raw": list(confidence_ema_5_raw),
        "confidence_ema_10_raw": list(confidence_ema_10_raw),
        "confidence_column": conf_col,
        "error": error,
        "error_ema": _ema_chart_fill(error_ema_raw),
        "error_ema_raw": list(error_ema_raw),
        "ltp_summary": ltp_summary,
        "error_summary": error_summary,
        "error_recent_rows": recent_errors,
        "error_downsampled": error_downsampled,
        "error_ema_downsampled": error_ema_downsampled,
        "gap": gap,
        "gap_chart": _series_for_chart(gap),
        "regression": predicted,
        "regression_ema": _ema_chart_fill(predicted_ema_raw),
        "regression_ema_slope": regression_ema_slope,
        "regression_slope_chart": _series_for_chart(regression_ema_slope),
        "position_size": sizes,
        "pnl": pnl,
        "has_trade_data": has_trades,
        "ema_span": span,
        "timestamps": timestamps,
        "row_count": len(rows),
        "actual_chart": _series_for_chart(actual),
        "predicted_chart": _series_for_chart(predicted),
        "confidence_chart": _series_for_chart(confidence),
        "error_chart": _series_for_chart(error),
        "pnl_chart": _series_for_chart(pnl),
    }


DASHBOARD_QUERY_COLUMNS: tuple[str, ...] = (
    "prediction_id",
    "trading_day",
    "timestamp",
    "token",
    "strike",
    "option_type",
    "current_ltp",
    "predicted_future_ltp",
    "actual_future_ltp",
    "prediction_error",
    "absolute_error",
    "tb_pred_probability",
    *PROBABILITY_OUTPUT_COLUMNS,
)
