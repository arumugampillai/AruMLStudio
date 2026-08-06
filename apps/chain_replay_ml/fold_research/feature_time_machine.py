"""Feature time machine — per-tick values with heuristic contribution trail."""

from __future__ import annotations

import math
from typing import Any


def _num(v: Any) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


_BULLISH_UP = frozenset({"spot_ema20_to_ltp_ratio", "delta", "chain_pcr", "current_iv"})


def _contribution_pct(delta: float | None, value: float | None, fname: str, *, direction: str) -> float | None:
    if delta is None or value is None or value == 0:
        return None
    raw = delta / abs(value) * 100.0
    if fname in _BULLISH_UP:
        aligned = raw if direction == "long" else -raw
    elif fname == "theta":
        aligned = -raw if direction == "long" else raw
    else:
        aligned = raw
    return round(aligned, 1)


def build_feature_time_machine(
    feature_series: dict[str, list[dict[str, Any]]],
    *,
    direction: str = "long",
) -> dict[str, Any]:
    """Build entry→exit trails with contribution % per step."""
    trails: dict[str, list[dict[str, Any]]] = {}
    for fname, series in feature_series.items():
        if not series:
            continue
        trail: list[dict[str, Any]] = []
        for pt in series:
            val = _num(pt.get("value"))
            delta = _num(pt.get("delta"))
            contrib = _contribution_pct(delta, val, fname, direction=direction)
            trail.append({
                "rel_label": pt.get("rel_label") or pt.get("time_label"),
                "value": val,
                "delta": delta,
                "contribution_pct": contrib,
                "arrow": "↑" if (contrib or 0) > 0 else ("↓" if (contrib or 0) < 0 else "—"),
            })
        trails[fname] = trail
    return {"trails": trails, "feature_count": len(trails)}
