"""Attach evaluation prices to ORMP feature rows (stable dataset, no labels)."""

from __future__ import annotations

from typing import Any

from .feature_export import FORWARD_LTP_HORIZONS_MIN, PRICE_COLUMNS


def attach_forward_ltp_prices(
    rows: list[dict[str, Any]],
    *,
    horizons_min: tuple[int, ...] = FORWARD_LTP_HORIZONS_MIN,
) -> list[dict[str, Any]]:
    """Fill future_ltp_{h}m from same-day spot_ltp lookup.

    ``spot_ltp`` must already be set on each row (1-minute Close).
    Missing future bar → null. Does not cross trading-day boundaries.
    """
    by_ts: dict[float, float] = {
        float(r["timestamp"]): float(r["spot_ltp"])
        for r in rows
        if r.get("spot_ltp") is not None
    }
    for r in rows:
        ts = float(r["timestamp"])
        for h in horizons_min:
            key = f"future_ltp_{h}m"
            fut = by_ts.get(ts + float(h) * 60.0)
            r[key] = None if fut is None or float(fut) <= 0 else float(fut)
        for col in PRICE_COLUMNS:
            r.setdefault(col, None)
    return rows
