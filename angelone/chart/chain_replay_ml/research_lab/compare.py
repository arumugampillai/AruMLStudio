"""Compare strategy runs and cross-run research views."""

from __future__ import annotations

from typing import Any

from chain_replay_ml.strategy_simulator.service import get_strategy_run_detail, get_strategy_run_trades


def compare_strategy_runs(data_dir: str, run_ids: list[str]) -> dict[str, Any]:
    ids = [str(r).strip() for r in run_ids if str(r).strip()]
    if len(ids) < 2:
        return {"ok": False, "error": "at least two strategy_run_id values required"}

    runs: list[dict[str, Any]] = []
    for rid in ids:
        doc = get_strategy_run_detail(data_dir, rid)
        if not doc or not doc.get("run"):
            return {"ok": False, "error": f"strategy run not found: {rid}"}
        runs.append(doc["run"])

    metrics_keys = (
        "trade_count", "profit", "win_rate_pct", "profit_factor",
        "max_drawdown", "avg_return_pct", "avg_holding_sec",
    )
    comparison: list[dict[str, Any]] = []
    for key in metrics_keys:
        comparison.append({
            "metric": key,
            "values": {r["strategy_run_id"]: (r.get("metrics") or {}).get(key) for r in runs},
        })

    return {
        "ok": True,
        "runs": runs,
        "metric_comparison": comparison,
    }


def compare_strategy_run_trades(
    data_dir: str,
    run_a: str,
    run_b: str,
    *,
    limit: int = 200,
) -> dict[str, Any]:
    a = get_strategy_run_trades(data_dir, run_a, limit=limit)
    b = get_strategy_run_trades(data_dir, run_b, limit=limit)
    if not a.get("ok") or not b.get("ok"):
        return {"ok": False, "error": "one or both strategy runs not found"}
    return {
        "ok": True,
        "run_a": a.get("run"),
        "run_b": b.get("run"),
        "trades_a": a.get("trades"),
        "trades_b": b.get("trades"),
        "total_a": a.get("total"),
        "total_b": b.get("total"),
    }
