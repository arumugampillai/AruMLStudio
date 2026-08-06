"""Leaderboard rankings for research lab."""

from __future__ import annotations

from typing import Any

from .matrix import build_research_matrix

LEADERBOARD_MODES = {
    "highest_profit": ("profit", True),
    "highest_win_rate": ("win_rate_pct", True),
    "highest_profit_factor": ("profit_factor", True),
    "lowest_drawdown": ("max_drawdown", False),
    "most_trades": ("trade_count", True),
    "best_composite": ("composite_score", True),
    "most_stable": ("stability_score", True),
}


def build_leaderboard(
    data_dir: str,
    *,
    mode: str = "best_composite",
    filters: dict[str, Any] | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    matrix = build_research_matrix(data_dir, filters=filters, limit=500)
    rows = list(matrix.get("rows") or [])
    sort_key, reverse = LEADERBOARD_MODES.get(mode, ("composite_score", True))

    def _sort_val(row: dict[str, Any]) -> float:
        val = row.get(sort_key)
        if val is None:
            return float("-inf") if reverse else float("inf")
        return float(val)

    rows.sort(key=_sort_val, reverse=reverse)
    top = rows[: max(1, min(limit, 100))]

    for i, row in enumerate(top, start=1):
        row["rank"] = i

    return {
        "ok": True,
        "mode": mode,
        "sort_key": sort_key,
        "reverse": reverse,
        "available_modes": list(LEADERBOARD_MODES.keys()),
        "filters": matrix.get("filters"),
        "leaderboard": top,
        "total_candidates": len(matrix.get("rows") or []),
    }


def build_research_summary(data_dir: str) -> dict[str, Any]:
    matrix = build_research_matrix(data_dir, limit=500)
    rows = matrix.get("rows") or []
    if not rows:
        return {
            "ok": True,
            "strategy_run_count": 0,
            "model_count": 0,
            "strategy_count": 0,
            "top_profit": None,
            "top_composite": None,
        }

    by_profit = sorted(rows, key=lambda r: float(r.get("profit") or -1e18), reverse=True)
    by_composite = sorted(rows, key=lambda r: float(r.get("composite_score") or -1e18), reverse=True)

    return {
        "ok": True,
        "strategy_run_count": len(rows),
        "model_count": len(matrix.get("models") or []),
        "strategy_count": len(matrix.get("strategies") or []),
        "top_profit": by_profit[0] if by_profit else None,
        "top_composite": by_composite[0] if by_composite else None,
        "models": matrix.get("models"),
        "strategies": matrix.get("strategies"),
    }
