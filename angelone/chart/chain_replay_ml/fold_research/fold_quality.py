"""Fold-level research quality score."""

from __future__ import annotations

from typing import Any


def compute_fold_quality(
    *,
    prediction_quality: dict[str, Any] | None,
    trading_metrics: dict[str, Any] | None,
    regime_analysis: dict[str, Any] | None,
    trade_count: int = 0,
) -> dict[str, Any]:
    """Score an entire fold for research prioritization."""
    pq = prediction_quality or {}
    tm = trading_metrics or {}
    reg = regime_analysis or {}

    mae = pq.get("mae")
    dir_acc = pq.get("directional_accuracy_pct")
    pred_score = 70
    if mae is not None:
        pred_score = max(40, min(98, int(95 - float(mae) * 2)))
    if dir_acc is not None:
        pred_score = int(pred_score * 0.5 + float(dir_acc) * 0.5)

    profit = tm.get("profit")
    win_rate = tm.get("win_rate_pct")
    pf = tm.get("profit_factor")
    strat_score = 60
    if win_rate is not None:
        strat_score = max(35, min(95, int(float(win_rate))))
    if pf is not None and pf > 0:
        strat_score = int(strat_score * 0.6 + min(float(pf), 3.0) / 3.0 * 40)

    exec_score = 75
    if trade_count > 0 and profit is not None:
        exec_score = 88 if float(profit) > 0 else 62
    if tm.get("max_drawdown") is not None:
        dd = abs(float(tm["max_drawdown"]))
        exec_score = max(40, exec_score - int(min(dd / 100, 25)))

    regimes = reg.get("regimes") or []
    regime_score = 80
    if regimes:
        counts = [int(r.get("row_count") or 0) for r in regimes]
        total = sum(counts) or 1
        dominant = max(counts) / total
        regime_score = int(70 + (1.0 - dominant) * 30)

    dimensions = [
        {"label": "Prediction", "score": pred_score, "max": 100},
        {"label": "Strategy", "score": strat_score, "max": 100},
        {"label": "Execution", "score": exec_score, "max": 100},
        {"label": "Regime Stability", "score": regime_score, "max": 100},
    ]
    total = round(sum(d["score"] for d in dimensions) / len(dimensions))

    note = None
    if total >= 85:
        note = "Strong fold — prediction and execution align."
    elif total < 65:
        note = "Worth deeper investigation — weak strategy or unstable regime mix."

    return {
        "total": total,
        "max": 100,
        "dimensions": dimensions,
        "note": note,
    }
