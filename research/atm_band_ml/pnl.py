"""Net P&L helpers for ML paper trades (1 lot, same charges as backtest)."""
from __future__ import annotations

from typing import Any, Mapping

from config import manipulate_prefs as mp
from shared.data.data_api_utils import calculate_charges

_DEFAULT_QTY = 65


def ml_qty_for_symbol(symbol: str | None) -> int:
    q = mp.lot_size_for_symbol(symbol)
    return int(q) if q else _DEFAULT_QTY


def ml_trade_pnl_rs(
    entry: Mapping[str, Any],
    *,
    mark_ltp: float | None = None,
) -> dict[str, float]:
    """Gross / charges / net for a closed trade or open position marked to *mark_ltp*."""
    try:
        entry_ltp = float(entry.get("entry_ltp") or 0.0)
    except (TypeError, ValueError):
        entry_ltp = 0.0
    if entry_ltp <= 0:
        return {"gross_pnl": 0.0, "charges": 0.0, "net_pnl": 0.0}

    try:
        qty = int(entry.get("qty") or ml_qty_for_symbol(str(entry.get("symbol") or "")))
    except (TypeError, ValueError):
        qty = ml_qty_for_symbol(str(entry.get("symbol") or ""))

    exit_ltp: float | None = mark_ltp
    if exit_ltp is None:
        raw_exit = entry.get("exit_ltp")
        if raw_exit is not None:
            try:
                exit_ltp = float(raw_exit)
            except (TypeError, ValueError):
                exit_ltp = None
    if exit_ltp is None:
        return {"gross_pnl": 0.0, "charges": 0.0, "net_pnl": 0.0}

    v_buy = entry_ltp * qty
    v_sell = float(exit_ltp) * qty
    gross = v_sell - v_buy
    charges = float(calculate_charges(v_buy, v_sell))
    net = gross - charges
    return {
        "gross_pnl": round(gross, 2),
        "charges": round(charges, 2),
        "net_pnl": round(net, 2),
    }


def aggregate_ml_pnl(
    entries: list[dict[str, Any]],
    ltp_ref: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    realized = 0.0
    unrealized = 0.0
    ltps = ltp_ref or {}
    for entry in entries or []:
        status = str(entry.get("status") or "").upper()
        if status == "CLOSED":
            if entry.get("net_pnl") is not None:
                try:
                    realized += float(entry["net_pnl"])
                    continue
                except (TypeError, ValueError):
                    pass
            realized += ml_trade_pnl_rs(entry)["net_pnl"]
        elif status == "OPEN":
            tok = str(entry.get("token") or "").strip()
            raw_ltp = ltps.get(tok)
            if raw_ltp is None:
                continue
            try:
                mark = float(raw_ltp)
            except (TypeError, ValueError):
                continue
            if mark > 0:
                unrealized += ml_trade_pnl_rs(entry, mark_ltp=mark)["net_pnl"]
    total = realized + unrealized
    return {
        "net_pnl_realized": round(realized, 2),
        "net_pnl_unrealized": round(unrealized, 2),
        "net_pnl_total": round(total, 2),
    }


def fmt_rs_pnl(value: float | None) -> str:
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"₹{v:+,.2f}"
