"""Counterfactual strategy replay — what-if scenarios on historical premium path."""

from __future__ import annotations

import math
from typing import Any


def _num(v: Any) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _pnl_at(entry: float, exit_premium: float, direction: str) -> float:
    if direction == "long":
        return round(exit_premium - entry, 4)
    return round(entry - exit_premium, 4)


def simulate_scenario(
    premium_path: list[dict[str, Any]],
    *,
    entry_price: float,
    entry_ts: float,
    direction: str = "long",
    stop_pct: float | None = None,
    target_pct: float | None = None,
    max_hold_sec: float | None = None,
    ignore_stop: bool = False,
) -> dict[str, Any]:
    """Walk premium path with alternate exit rules."""
    if not premium_path or entry_price <= 0:
        return {"profit": None, "exit_reason": "no_data"}

    stop_price = target_price = None
    if stop_pct is not None and not ignore_stop:
        if direction == "long":
            stop_price = entry_price * (1.0 - stop_pct / 100.0)
        else:
            stop_price = entry_price * (1.0 + stop_pct / 100.0)
    if target_pct is not None:
        if direction == "long":
            target_price = entry_price * (1.0 + target_pct / 100.0)
        else:
            target_price = entry_price * (1.0 - target_pct / 100.0)

    exit_premium = entry_price
    exit_ts = entry_ts
    exit_reason = "end_of_path"

    for pt in premium_path:
        ts = _num(pt.get("timestamp"))
        prem = _num(pt.get("value"))
        if ts is None or prem is None:
            continue
        if ts < entry_ts:
            continue
        hold = ts - entry_ts
        exit_premium = prem
        exit_ts = ts

        if max_hold_sec is not None and hold >= max_hold_sec:
            exit_reason = "max_hold"
            break

        if direction == "long":
            if stop_price is not None and prem <= stop_price:
                exit_reason = "stop"
                break
            if target_price is not None and prem >= target_price:
                exit_reason = "target"
                break
        else:
            if stop_price is not None and prem >= stop_price:
                exit_reason = "stop"
                break
            if target_price is not None and prem <= target_price:
                exit_reason = "target"
                break

    profit = _pnl_at(entry_price, exit_premium, direction)
    return {
        "profit": profit,
        "exit_premium": exit_premium,
        "exit_ts": exit_ts,
        "held_seconds": round(exit_ts - entry_ts, 1) if exit_ts else None,
        "exit_reason": exit_reason,
    }


def build_counterfactuals(
    trade: dict[str, Any],
    premium_path: list[dict[str, Any]],
    *,
    cfg: dict[str, Any] | None = None,
    direction: str = "long",
) -> dict[str, Any]:
    """Build what-if scenarios for a completed trade."""
    entry_price = _num(trade.get("entry_price"))
    entry_ts = _num(trade.get("entry_ts"))
    if entry_price is None or entry_ts is None:
        return {"available": False, "scenarios": []}

    entry = cfg or {}
    target_cfg = entry.get("target") or {}
    stop_cfg = entry.get("stop") or {}
    hold_cfg = entry.get("hold_time") or entry.get("max_hold") or {}

    current_stop = float(stop_cfg.get("stop_loss_pct") or 5.0)
    current_target = float(target_cfg.get("target_profit_pct") or 8.0)
    current_hold = float(
        hold_cfg.get("max_hold_sec")
        or hold_cfg.get("max_hold_seconds")
        or trade.get("holding_seconds")
        or 30
    )

    actual_profit = _num(trade.get("net_pnl"))

    specs: list[tuple[str, dict[str, Any]]] = [
        ("Actual", {}),
        (f"Stop = {current_stop:.0f}%", {"stop_pct": current_stop, "target_pct": current_target, "max_hold_sec": current_hold}),
        ("Stop = 7%", {"stop_pct": 7.0, "target_pct": current_target, "max_hold_sec": current_hold}),
        ("Hold = 60 sec", {"stop_pct": current_stop, "target_pct": current_target, "max_hold_sec": 60.0}),
        ("No Stop", {"stop_pct": current_stop, "target_pct": current_target, "max_hold_sec": current_hold, "ignore_stop": True}),
        ("Target = 10%", {"stop_pct": current_stop, "target_pct": 10.0, "max_hold_sec": current_hold}),
    ]

    scenarios: list[dict[str, Any]] = []
    for label, kwargs in specs:
        if label == "Actual":
            scenarios.append({
                "label": label,
                "profit": actual_profit,
                "exit_reason": trade.get("exit_reason"),
                "is_actual": True,
            })
            continue
        ignore_stop = bool(kwargs.pop("ignore_stop", False))
        sim = simulate_scenario(
            premium_path,
            entry_price=entry_price,
            entry_ts=entry_ts,
            direction=direction,
            ignore_stop=ignore_stop,
            **kwargs,
        )
        scenarios.append({
            "label": label,
            "profit": sim.get("profit"),
            "exit_reason": sim.get("exit_reason"),
            "held_seconds": sim.get("held_seconds"),
            "is_actual": False,
        })

    best = max(
        (s for s in scenarios if s.get("profit") is not None),
        key=lambda s: float(s["profit"]),
        default=None,
    )
    return {
        "available": True,
        "scenarios": scenarios,
        "best_label": best.get("label") if best else None,
        "scenario_count": len([s for s in scenarios if not s.get("is_actual")]),
    }
