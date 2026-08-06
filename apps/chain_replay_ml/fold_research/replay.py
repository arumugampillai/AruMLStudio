"""Second-by-second style replay timeline for a fold."""

from __future__ import annotations

from typing import Any

from chain_replay_ml.strategy_simulator.metrics import compute_trade_metrics


def build_replay_timeline(
    rows: list[dict[str, Any]],
    *,
    trades: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Merge prediction rows and strategy trades into a chronological event stream."""
    events: list[dict[str, Any]] = []

    for row in rows:
        ts = row.get("timestamp")
        if ts is None:
            continue
        events.append({
            "event_type": "prediction",
            "timestamp": float(ts),
            "trading_day": row.get("trading_day"),
            "token": row.get("token"),
            "strike": row.get("strike"),
            "option_type": row.get("option_type"),
            "spot": row.get("spot"),
            "ltp": row.get("ltp"),
            "predicted_ltp": row.get("predicted_ltp"),
            "actual_ltp": row.get("actual_ltp"),
            "prediction_error": row.get("prediction_error"),
            "direction_correct": row.get("direction_correct"),
            "confidence": row.get("confidence"),
            "prediction_id": row.get("prediction_id"),
            "label": _prediction_label(row),
            "display_type": _display_type("prediction"),
        })

    for trade in trades or []:
        entry_ts = trade.get("entry_ts")
        exit_ts = trade.get("exit_ts")
        if entry_ts is not None:
            events.append({
                "event_type": "trade_entry",
                "timestamp": float(entry_ts),
                "trading_day": trade.get("trading_day"),
                "token": trade.get("token"),
                "strike": trade.get("strike"),
                "option_type": trade.get("option_type"),
                "price": trade.get("entry_price"),
                "trade_id": trade.get("trade_id"),
                "label": f"BUY {trade.get('token')} @ {trade.get('entry_price')}",
                "display_type": _display_type("trade_entry"),
            })
        if exit_ts is not None:
            events.append({
                "event_type": "trade_exit",
                "timestamp": float(exit_ts),
                "trading_day": trade.get("trading_day"),
                "token": trade.get("token"),
                "strike": trade.get("strike"),
                "option_type": trade.get("option_type"),
                "price": trade.get("exit_price"),
                "net_pnl": trade.get("net_pnl"),
                "return_pct": trade.get("return_pct"),
                "exit_reason": trade.get("exit_reason"),
                "trade_id": trade.get("trade_id"),
                "label": (
                    f"EXIT {trade.get('token')} @ {trade.get('exit_price')} "
                    f"({trade.get('exit_reason')}, PnL {trade.get('net_pnl')})"
                ),
                "display_type": _display_type("trade_exit"),
            })

    events.sort(key=lambda e: (e.get("timestamp") or 0, _event_order(e.get("event_type"))))
    for i, ev in enumerate(events):
        ev["sequence"] = i + 1
    return events


_EVENT_LABELS = {
    "prediction": "Prediction",
    "trade_entry": "Entry",
    "trade_exit": "Exit",
    "signal": "Signal",
    "stop_moved": "Stop moved",
    "target_hit": "Target",
}


def _display_type(event_type: str | None) -> str:
    return _EVENT_LABELS.get(str(event_type or ""), str(event_type or "Event").replace("_", " ").title())


def _event_order(event_type: str | None) -> int:
    return {"trade_exit": 0, "prediction": 1, "trade_entry": 2}.get(str(event_type or ""), 3)


def _prediction_label(row: dict[str, Any]) -> str:
    token = row.get("token") or "?"
    pred = row.get("predicted_ltp")
    actual = row.get("actual_ltp")
    return f"PRED {token} pred={pred} actual={actual}"
