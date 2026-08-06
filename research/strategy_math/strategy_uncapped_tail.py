"""Detect uncapped loss tails (naked short CE/PE) for chain-style P&L views."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

_LOSS_TOL_RUPEES = -0.5


def leg_lots_from_leg(leg: Mapping[str, Any], *, lot_size: int | None = None) -> int:
    try:
        lots = int(leg.get("lots") or 0)
    except (TypeError, ValueError):
        lots = 0
    if lots > 0:
        return lots
    badge = str(leg.get("badge") or "")
    if "-" in badge:
        try:
            return max(1, int(badge.rsplit("-", 1)[-1]))
        except ValueError:
            pass
    try:
        qty = int(leg.get("quantity") or 0)
    except (TypeError, ValueError):
        qty = 0
    if qty <= 0:
        return 1
    lot_sz = max(1, int(lot_size or 0))
    if lot_sz <= 0:
        return max(1, qty)
    lots_f = qty / float(lot_sz)
    if abs(lots_f - round(lots_f)) < 1e-6:
        return max(1, int(round(lots_f)))
    return max(1, int(round(lots_f)))


def _leg_side(leg: Mapping[str, Any]) -> str:
    side = str(leg.get("side") or leg.get("transaction_type") or "B").upper()
    return "B" if side in ("B", "BUY") else "S"


def _leg_option_type(leg: Mapping[str, Any]) -> str:
    return str(leg.get("option_type") or "").strip().upper()


def compute_uncapped_tail_cache(
    legs: Sequence[Mapping[str, Any]],
    pnl_by_strike: dict[float, float],
    *,
    lot_size: int | None = None,
) -> dict[str, Any]:
    """Naked short CE (upside) / PE (downside) tails from merged leg exposure."""
    if not pnl_by_strike or not legs:
        return {}
    ce_long = ce_short = pe_long = pe_short = 0
    for leg in legs:
        opt = _leg_option_type(leg)
        side = _leg_side(leg)
        lots = leg_lots_from_leg(leg, lot_size=lot_size)
        if opt == "CE":
            if side == "B":
                ce_long += lots
            else:
                ce_short += lots
        elif opt == "PE":
            if side == "B":
                pe_long += lots
            else:
                pe_short += lots

    peak_strike = max(pnl_by_strike, key=lambda k: pnl_by_strike[k])
    tails: list[dict[str, Any]] = []

    if ce_short > ce_long:
        unprotected = ce_short - ce_long
        loss_candidates = sorted(
            s for s, pnl in pnl_by_strike.items() if pnl < _LOSS_TOL_RUPEES and s >= peak_strike
        )
        if loss_candidates:
            tails.append(
                {
                    "direction": "up",
                    "start_strike": loss_candidates[0],
                    "lots": unprotected,
                }
            )

    if pe_short > pe_long:
        unprotected = pe_short - pe_long
        loss_candidates = sorted(
            (s for s, pnl in pnl_by_strike.items() if pnl < _LOSS_TOL_RUPEES and s <= peak_strike),
            reverse=True,
        )
        if loss_candidates:
            tails.append(
                {
                    "direction": "down",
                    "start_strike": loss_candidates[0],
                    "lots": unprotected,
                }
            )

    return {"tails": tails} if tails else {}


def uncapped_tails_from_cache(cache: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not cache:
        return []
    tails = cache.get("tails")
    if isinstance(tails, list) and tails:
        return [t for t in tails if isinstance(t, dict)]
    if cache.get("direction"):
        return [dict(cache)]
    return []


def uncapped_lots_at_strike(
    strike: float,
    tail_cache: Mapping[str, Any] | None,
    pnl: float | None,
    *,
    loss_tol: float = _LOSS_TOL_RUPEES,
) -> int | None:
    if not tail_cache or pnl is None or float(pnl) >= loss_tol:
        return None
    try:
        strike_f = float(strike)
    except (TypeError, ValueError):
        return None
    for tail in uncapped_tails_from_cache(tail_cache):
        direction = tail.get("direction")
        try:
            start = float(tail.get("start_strike"))
            lots = int(tail.get("lots") or 0) or None
        except (TypeError, ValueError):
            continue
        if lots is None:
            continue
        if direction == "up" and strike_f >= start:
            return lots
        if direction == "down" and strike_f <= start:
            return lots
    return None
