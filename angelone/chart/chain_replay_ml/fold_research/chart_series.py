"""Chart-ready series for fold research dashboards."""

from __future__ import annotations

from typing import Any


def build_chart_series(
    rows: list[dict[str, Any]],
    trades: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    errors: list[float] = []
    for row in rows:
        try:
            err = float(row.get("prediction_error"))
            if err == err:  # finite
                errors.append(abs(err))
        except (TypeError, ValueError):
            pass

    trade_pnls: list[float] = []
    equity: list[dict[str, Any]] = []
    drawdown: list[dict[str, Any]] = []
    cum = 0.0
    peak = 0.0
    for trade in sorted(trades or [], key=lambda t: (t.get("exit_ts") or 0, t.get("trade_id") or "")):
        try:
            pnl = float(trade.get("net_pnl") or 0)
        except (TypeError, ValueError):
            pnl = 0.0
        trade_pnls.append(pnl)
        cum += pnl
        peak = max(peak, cum)
        ts = trade.get("exit_ts")
        equity.append({"ts": ts, "value": round(cum, 2)})
        drawdown.append({"ts": ts, "value": round(peak - cum, 2)})

    return {
        "prediction_errors": errors,
        "trade_pnls": trade_pnls,
        "equity_curve": equity,
        "drawdown_curve": drawdown,
    }
