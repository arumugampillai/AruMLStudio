"""Prediction quality analysis for a single fold."""

from __future__ import annotations

import math
from typing import Any


def _num(v: Any) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def analyze_prediction_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[float] = []
    biases: list[float] = []
    pred_returns: list[float] = []
    actual_returns: list[float] = []
    direction_hits = 0
    direction_total = 0
    confidences: list[float] = []

    for row in rows:
        err = _num(row.get("prediction_error"))
        if err is not None:
            errors.append(err)
            biases.append(err)
        ltp = _num(row.get("ltp"))
        pred = _num(row.get("predicted_ltp"))
        actual = _num(row.get("actual_ltp"))
        if ltp and ltp > 0 and pred is not None and actual is not None:
            pred_returns.append((pred - ltp) / ltp * 100.0)
            actual_returns.append((actual - ltp) / ltp * 100.0)
        dc = row.get("direction_correct")
        if dc is not None:
            direction_total += 1
            direction_hits += int(dc)
        conf = _num(row.get("confidence"))
        if conf is not None:
            confidences.append(conf)

    if not errors:
        return {
            "row_count": len(rows),
            "mae": None,
            "rmse": None,
            "bias": None,
            "bias_pct": None,
            "directional_accuracy_pct": None,
            "median_error": None,
            "p95_error": None,
            "max_error": None,
            "calibration_buckets": [],
            "confidence_histogram": [],
        }

    abs_errors = [abs(e) for e in errors]
    mae = sum(abs_errors) / len(abs_errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    bias = sum(biases) / len(biases)
    sorted_abs = sorted(abs_errors)
    p95_idx = min(len(sorted_abs) - 1, int(math.ceil(0.95 * len(sorted_abs))) - 1)
    p95_error = sorted_abs[p95_idx] if sorted_abs else None
    bias_pct_vals = []
    for row in rows:
        ltp = _num(row.get("ltp"))
        err = _num(row.get("prediction_error"))
        if ltp and ltp > 0 and err is not None:
            bias_pct_vals.append(err / ltp * 100.0)

    return {
        "row_count": len(rows),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "bias": round(bias, 4),
        "bias_pct": round(sum(bias_pct_vals) / len(bias_pct_vals), 4) if bias_pct_vals else None,
        "directional_accuracy_pct": round(direction_hits / direction_total * 100.0, 2) if direction_total else None,
        "median_error": round(sorted(abs_errors)[len(abs_errors) // 2], 4),
        "p95_error": round(p95_error, 4) if p95_error is not None else None,
        "max_error": round(max(abs_errors), 4),
        "calibration_buckets": _calibration_buckets(pred_returns, actual_returns),
        "confidence_histogram": _histogram(confidences, bins=10),
    }


def _calibration_buckets(
    pred_returns: list[float],
    actual_returns: list[float],
    *,
    n_bins: int = 5,
) -> list[dict[str, Any]]:
    if not pred_returns or len(pred_returns) != len(actual_returns):
        return []
    pairs = list(zip(pred_returns, actual_returns))
    pairs.sort(key=lambda p: p[0])
    size = max(1, len(pairs) // n_bins)
    buckets: list[dict[str, Any]] = []
    for i in range(0, len(pairs), size):
        chunk = pairs[i : i + size]
        if not chunk:
            continue
        preds = [p[0] for p in chunk]
        actuals = [p[1] for p in chunk]
        buckets.append({
            "bin": len(buckets) + 1,
            "count": len(chunk),
            "pred_return_avg_pct": round(sum(preds) / len(preds), 4),
            "actual_return_avg_pct": round(sum(actuals) / len(actuals), 4),
            "calibration_error_pct": round(
                sum(actuals) / len(actuals) - sum(preds) / len(preds), 4
            ),
        })
    return buckets


def _histogram(values: list[float], *, bins: int = 10) -> list[dict[str, Any]]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if lo == hi:
        return [{"bin": 1, "low": lo, "high": hi, "count": len(values)}]
    width = (hi - lo) / bins
    out: list[dict[str, Any]] = []
    for i in range(bins):
        b_lo = lo + i * width
        b_hi = lo + (i + 1) * width
        count = sum(1 for v in values if (b_lo <= v < b_hi) or (i == bins - 1 and v == b_hi))
        out.append({"bin": i + 1, "low": round(b_lo, 4), "high": round(b_hi, 4), "count": count})
    return out
