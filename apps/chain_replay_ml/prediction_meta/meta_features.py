"""Derived prediction meta features — confidence, ranking, ids, context."""

from __future__ import annotations

import statistics
from datetime import datetime, timedelta, timezone
from typing import Any

from chain_replay_ml.bs import expiry_close_ts

_IST = timezone(timedelta(hours=5, minutes=30))


def format_grid_time(timestamp: float) -> str:
    return datetime.fromtimestamp(float(timestamp), tz=_IST).strftime("%H:%M:%S")


def build_prediction_id(
    *,
    trading_day: str,
    timestamp: float,
    strike: Any,
    option_type: Any,
    token: str,
    prediction_version: int,
) -> str:
    """Stable, human-readable id; version suffix allows re-prediction with new model sets."""
    strike_s = str(strike or "").strip() or "NA"
    opt_s = str(option_type or "").strip().upper() or "NA"
    time_s = format_grid_time(timestamp)
    return f"{trading_day}|{time_s}|{strike_s}|{opt_s}|{token}|v{int(prediction_version)}"


def minutes_to_expiry_at(*, timestamp: float, expiry: str | None) -> float | None:
    if not expiry:
        return None
    try:
        return round(max(0.0, (expiry_close_ts(str(expiry)) - float(timestamp)) / 60.0), 2)
    except (TypeError, ValueError, OSError, OverflowError):
        return None


def resolve_minutes_to_expiry(row: dict[str, Any], *, timestamp: float) -> float | None:
    raw = row.get("minutes_to_expiry")
    if raw is not None:
        try:
            return round(float(raw), 2)
        except (TypeError, ValueError):
            pass
    return minutes_to_expiry_at(timestamp=timestamp, expiry=row.get("expiry"))


def confidence_features(
    preds: list[float],
    *,
    mean: float | None,
    median: float | None,
    entry_ltp: float | None,
) -> dict[str, float | None]:
    if not preds:
        return {
            "prediction_min": None,
            "prediction_max": None,
            "prediction_range_pct": None,
            "mean_minus_current_ltp": None,
            "median_minus_current_ltp": None,
        }
    pred_min = min(preds)
    pred_max = max(preds)
    range_pct = None
    if entry_ltp is not None and entry_ltp > 0:
        range_pct = round((pred_max - pred_min) / entry_ltp * 100.0, 4)
    mean_minus = round(mean - entry_ltp, 4) if mean is not None and entry_ltp is not None else None
    median_minus = round(median - entry_ltp, 4) if median is not None and entry_ltp is not None else None
    return {
        "prediction_min": round(pred_min, 4),
        "prediction_max": round(pred_max, 4),
        "prediction_range_pct": range_pct,
        "mean_minus_current_ltp": mean_minus,
        "median_minus_current_ltp": median_minus,
    }


def model_deltas_and_ranks(
    model_preds: list[float | None],
    *,
    ensemble_mean: float | None,
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Per-model delta from ensemble mean and rank (1 = lowest prediction)."""
    deltas: dict[str, float | None] = {}
    ranks: dict[str, float | None] = {}
    ok_pairs: list[tuple[int, float]] = []
    for i, pred in enumerate(model_preds, start=1):
        if pred is None or ensemble_mean is None:
            deltas[f"model_{i}_delta_from_mean"] = None
            ranks[f"model_{i}_rank"] = None
        else:
            deltas[f"model_{i}_delta_from_mean"] = round(pred - ensemble_mean, 4)
            ok_pairs.append((i, pred))

    if ok_pairs:
        sorted_pairs = sorted(ok_pairs, key=lambda x: x[1])
        for rank, (model_i, _pred) in enumerate(sorted_pairs, start=1):
            ranks[f"model_{model_i}_rank"] = float(rank)
        for i in range(1, len(model_preds) + 1):
            ranks.setdefault(f"model_{i}_rank", None)

    return deltas, ranks


def extend_ensemble_meta(
    preds: list[float],
    *,
    models_ok: int,
    models_failed: int,
    entry_ltp: float | None,
) -> dict[str, Any]:
    if not preds:
        base = {
            "ensemble_mean": None,
            "ensemble_median": None,
            "ensemble_std": None,
            "ensemble_spread": None,
            "agreement": 0.0,
            "models_ok": models_ok,
            "models_failed": models_failed,
        }
        base.update(confidence_features([], mean=None, median=None, entry_ltp=entry_ltp))
        return base

    mean = statistics.fmean(preds)
    median = float(statistics.median(preds))
    std = statistics.pstdev(preds) if len(preds) > 1 else 0.0
    spread = max(preds) - min(preds)
    agree_n = _agreement_count(preds, mean, std)
    base = {
        "ensemble_mean": round(mean, 4),
        "ensemble_median": round(median, 4),
        "ensemble_std": round(std, 4),
        "ensemble_spread": round(spread, 4),
        "agreement": round(agree_n / models_ok, 4) if models_ok else 0.0,
        "models_ok": models_ok,
        "models_failed": models_failed,
    }
    base.update(confidence_features(preds, mean=mean, median=median, entry_ltp=entry_ltp))
    return base


def _agreement_count(values: list[float], mean: float, std: float) -> int:
    if not values:
        return 0
    if std <= 0:
        return len(values)
    return sum(1 for v in values if abs(v - mean) <= std)
