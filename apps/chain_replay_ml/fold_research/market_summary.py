"""Market context summary derived from prediction rows in a fold."""

from __future__ import annotations

import math
from typing import Any


def _num(v: Any) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def summarize_market_context(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"row_count": 0, "available": False}

    spots = [_num(r.get("spot")) for r in rows]
    spots = [s for s in spots if s is not None]
    ltps = [_num(r.get("ltp")) for r in rows]
    ltps = [x for x in ltps if x is not None]
    timestamps = [_num(r.get("timestamp")) for r in rows]
    timestamps = [t for t in timestamps if t is not None]
    days = sorted({str(r.get("trading_day")) for r in rows if r.get("trading_day")})
    tokens = sorted({str(r.get("token")) for r in rows if r.get("token")})
    strikes = [_num(r.get("strike")) for r in rows]
    strikes = [s for s in strikes if s is not None]

    spot_returns: list[float] = []
    sorted_by_ts = sorted(
        [r for r in rows if _num(r.get("timestamp")) is not None],
        key=lambda r: _num(r.get("timestamp")) or 0.0,
    )
    prev_spot = None
    for row in sorted_by_ts:
        spot = _num(row.get("spot"))
        if spot is None:
            continue
        if prev_spot is not None and prev_spot > 0:
            spot_returns.append((spot - prev_spot) / prev_spot * 100.0)
        prev_spot = spot

    volatility_proxy = None
    if len(spot_returns) > 1:
        mean = sum(spot_returns) / len(spot_returns)
        variance = sum((x - mean) ** 2 for x in spot_returns) / len(spot_returns)
        volatility_proxy = round(math.sqrt(variance), 4)

    trend_pct = None
    if spots:
        if spots[0] != 0:
            trend_pct = round((spots[-1] - spots[0]) / spots[0] * 100.0, 4)

    atm_moves: list[float] = []
    for row in rows:
        spot = _num(row.get("spot"))
        strike = _num(row.get("strike"))
        if spot is not None and strike is not None:
            atm_moves.append(strike - spot)
    atm_move_avg = round(sum(atm_moves) / len(atm_moves), 2) if atm_moves else None

    return {
        "available": True,
        "row_count": len(rows),
        "trading_days": days,
        "trading_day_count": len(days),
        "token_count": len(tokens),
        "tokens": tokens[:20],
        "spot_min": round(min(spots), 2) if spots else None,
        "spot_max": round(max(spots), 2) if spots else None,
        "spot_start": round(spots[0], 2) if spots else None,
        "spot_end": round(spots[-1], 2) if spots else None,
        "spot_trend_pct": trend_pct,
        "ltp_min": round(min(ltps), 2) if ltps else None,
        "ltp_max": round(max(ltps), 2) if ltps else None,
        "timestamp_start": min(timestamps) if timestamps else None,
        "timestamp_end": max(timestamps) if timestamps else None,
        "timestamp_span_sec": round(max(timestamps) - min(timestamps), 1) if len(timestamps) >= 2 else 0,
        "volatility_proxy_pct": volatility_proxy,
        "atm_strike_minus_spot_avg": atm_move_avg,
        "strike_min": round(min(strikes), 0) if strikes else None,
        "strike_max": round(max(strikes), 0) if strikes else None,
        "regime_note": (
            "Derived from fold validation rows only (spot/ltp). "
            "IV, PCR, and gap require chain tick enrichment — not stored in prediction rows yet."
        ),
    }
