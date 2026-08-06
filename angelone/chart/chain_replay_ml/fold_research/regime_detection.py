"""Market regime tagging and MAE-by-regime for fold research."""

from __future__ import annotations

import math
from typing import Any


def _num(v: Any) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _classify_row(row: dict[str, Any], prev_spot: float | None) -> str:
    spot = _num(row.get("spot"))
    ltp = _num(row.get("ltp"))
    regimes: list[str] = []

    if spot is not None and prev_spot is not None and prev_spot > 0:
        ret = (spot - prev_spot) / prev_spot * 100.0
        if ret > 0.15:
            regimes.append("Momentum")
        elif ret < -0.15:
            regimes.append("Reversal")
        if abs(ret) > 0.35:
            regimes.append("Trending")
        elif abs(ret) < 0.03:
            regimes.append("Range")

    if ltp is not None and ltp < 18:
        regimes.append("Low Premium")
    if ltp is not None and ltp > 40:
        regimes.append("High Premium")

    if not regimes:
        return "Neutral"
    return " + ".join(regimes[:2])


def analyze_regimes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"available": False, "regimes": [], "note": "No rows for regime analysis."}

    sorted_rows = sorted(rows, key=lambda r: (_num(r.get("timestamp")) or 0.0))
    prev_spot: float | None = None
    by_regime: dict[str, list[float]] = {}
    counts: dict[str, int] = {}

    for row in sorted_rows:
        regime = _classify_row(row, prev_spot)
        counts[regime] = counts.get(regime, 0) + 1
        err = _num(row.get("prediction_error"))
        if err is not None:
            by_regime.setdefault(regime, []).append(abs(err))
        spot = _num(row.get("spot"))
        if spot is not None:
            prev_spot = spot

    # Volatility expansion/compression from spot returns
    spot_returns: list[float] = []
    prev = None
    for row in sorted_rows:
        spot = _num(row.get("spot"))
        if spot is None:
            continue
        if prev is not None and prev > 0:
            spot_returns.append(abs((spot - prev) / prev * 100.0))
        prev = spot
    vol = sum(spot_returns) / len(spot_returns) if spot_returns else 0.0
    vol_label = "Volatility Expansion" if vol > 0.08 else ("Volatility Compression" if vol < 0.02 else "Normal Vol")

    regime_rows: list[dict[str, Any]] = []
    for regime, errors in sorted(by_regime.items(), key=lambda kv: -len(kv[1])):
        mae = round(sum(errors) / len(errors), 4) if errors else None
        regime_rows.append({
            "regime": regime,
            "row_count": counts.get(regime, 0),
            "mae": mae,
        })

    return {
        "available": True,
        "volatility_regime": vol_label,
        "volatility_proxy_pct": round(vol, 4),
        "regimes": regime_rows,
        "note": "Regimes derived from spot/ltp in prediction rows. IV/PCR labels require feature rehydration.",
    }
