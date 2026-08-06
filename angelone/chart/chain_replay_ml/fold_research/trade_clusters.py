"""Auto cluster discovery for fold trades."""

from __future__ import annotations

import math
from typing import Any


def _num(v: Any) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _cluster_key(trade: dict[str, Any]) -> str:
    reason = str(trade.get("exit_reason") or "exit")
    pnl = _num(trade.get("net_pnl")) or 0.0
    hold = float(trade.get("holding_seconds") or 0)
    outcome = "Win" if pnl > 0 else ("Loss" if pnl < 0 else "Flat")
    if hold <= 20:
        hold_band = "Fast"
    elif hold <= 60:
        hold_band = "Medium"
    else:
        hold_band = "Slow"
    return f"{outcome} · {reason} · {hold_band}"


def discover_trade_clusters(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """Group trades into explainable clusters."""
    if not trades:
        return {"available": False, "clusters": []}

    buckets: dict[str, list[dict[str, Any]]] = {}
    for t in trades:
        key = _cluster_key(t)
        buckets.setdefault(key, []).append(t)

    clusters: list[dict[str, Any]] = []
    for i, (label, group) in enumerate(sorted(buckets.items(), key=lambda kv: -len(kv[1])), start=1):
        pnls = [_num(t.get("net_pnl")) for t in group]
        pnls = [p for p in pnls if p is not None]
        avg = round(sum(pnls) / len(pnls), 2) if pnls else None
        wins = sum(1 for p in pnls if p > 0)
        clusters.append({
            "cluster_id": i,
            "label": label,
            "trade_count": len(group),
            "average_pnl": avg,
            "win_rate_pct": round(wins / len(group) * 100, 1) if group else None,
            "tags": label.split(" · "),
        })

    return {
        "available": True,
        "cluster_count": len(clusters),
        "clusters": clusters[:12],
    }
