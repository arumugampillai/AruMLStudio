"""Composite research score for leaderboard ranking."""

from __future__ import annotations

from typing import Any


def composite_research_score(metrics: dict[str, Any] | None) -> float | None:
    if not metrics:
        return None
    trades = int(metrics.get("trade_count") or 0)
    if trades <= 0:
        return None
    profit = float(metrics.get("profit") or 0)
    pf = float(metrics.get("profit_factor") or 0)
    wr = float(metrics.get("win_rate_pct") or 0)
    dd = float(metrics.get("max_drawdown") or 0)
    return round(profit + pf * 25.0 + wr * 2.0 - dd * 0.5, 4)


def stability_score(fold_metrics: list[dict[str, Any]] | None) -> float | None:
    """Lower variance across folds = higher stability."""
    folds = fold_metrics or []
    profits = [float(f.get("profit") or 0) for f in folds if int(f.get("trade_count") or 0) > 0]
    if len(profits) < 2:
        return None
    mean = sum(profits) / len(profits)
    if mean == 0:
        return None
    variance = sum((p - mean) ** 2 for p in profits) / len(profits)
    std = variance ** 0.5
    return round(max(0.0, 100.0 - (std / abs(mean) * 100.0)), 4)
