"""Per-leg lot multipliers and side overrides for option strategy chain menus."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

LEG_LOT_MULTIPLIER_MAX = 6
LEG_EMPTY_STRIKE_MENU_MAX = 3


def clamp_leg_lot_multiple(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = 1
    return max(1, min(LEG_LOT_MULTIPLIER_MAX, n))


def default_leg_lot_multiples(strategy: str) -> dict[int, int]:
    """Default lot multiples per leg index when a strategy is selected."""
    key = str(strategy or "").strip().lower()
    base = {1: 1, 2: 1, 3: 1, 4: 1}
    if key in ("call_ratio_backspread", "put_ratio_backspread"):
        base[2] = 2
    elif key in ("call_ratio_spread", "put_ratio_spread"):
        base[2] = 2
    elif key in ("call_butterfly", "put_butterfly"):
        base[2] = 2
    return base


def normalize_leg_lot_multiples(
    multiples: Mapping[int, Any] | None,
    *,
    strategy: str | None = None,
) -> dict[int, int]:
    key = str(strategy or "").strip().lower()
    defaults = default_leg_lot_multiples(key)
    out = dict(defaults)
    if multiples:
        for idx in (1, 2, 3, 4):
            if idx in multiples:
                out[idx] = clamp_leg_lot_multiple(multiples[idx])
    required_leg2 = defaults.get(2, 1)
    if required_leg2 > 1 and out.get(2, 1) < required_leg2:
        out[2] = required_leg2
    return out


def apply_leg_lot_multiples(
    legs: Sequence[Mapping[str, Any]],
    base_qty: int,
    multiples: Mapping[int, Any],
    *,
    strategy: str | None = None,
) -> list[dict[str, Any]]:
    """Scale each built leg qty to base_qty × leg multiple."""
    qty_base = max(1, int(base_qty))
    mult = normalize_leg_lot_multiples(multiples, strategy=strategy)
    out: list[dict[str, Any]] = []
    for i, leg in enumerate(legs[:4], start=1):
        row = dict(leg)
        row["quantity"] = qty_base * mult.get(i, 1)
        out.append(row)
    return out


def normalize_leg_side_overrides(
    overrides: Mapping[int, Any] | None,
) -> dict[int, str]:
    out: dict[int, str] = {}
    if not overrides:
        return out
    for idx in (1, 2, 3, 4):
        if idx not in overrides:
            continue
        side = str(overrides[idx] or "").upper()
        if side in ("B", "BUY"):
            out[idx] = "B"
        elif side in ("S", "SELL"):
            out[idx] = "S"
    return out


def apply_leg_side_overrides(
    legs: Sequence[Mapping[str, Any]],
    overrides: Mapping[int, Any],
) -> list[dict[str, Any]]:
    """Flip transaction_type per leg when user picked Buy/Sell on an empty strike."""
    side_map = normalize_leg_side_overrides(overrides)
    if not side_map:
        return [dict(leg) for leg in legs]
    out: list[dict[str, Any]] = []
    for i, leg in enumerate(legs[:4], start=1):
        row = dict(leg)
        side = side_map.get(i)
        if side:
            row["transaction_type"] = side
        out.append(row)
    return out


def leg_menu_side_label(side: str) -> str:
    return "Buy" if str(side or "").upper() in ("B", "BUY") else "Sell"


_LEG1_SIDE: dict[str, str] = {
    "long_call": "B",
    "long_put": "B",
    "short_call": "S",
    "short_put": "S",
    "bull_call": "B",
    "bear_put": "B",
    "credit_call": "S",
    "credit_put": "S",
    "iron_condor": "S",
    "reverse_iron_condor": "B",
    "long_straddle": "B",
    "short_straddle": "S",
    "long_strangle": "B",
    "short_strangle": "S",
    "iron_butterfly": "S",
    "long_iron_butterfly": "B",
    "broken_wing_butterfly": "S",
    "long_broken_wing_butterfly": "B",
    "jade_lizard": "S",
    "reverse_jade_lizard": "S",
    "call_ratio_backspread": "S",
    "put_ratio_backspread": "S",
    "call_ratio_spread": "B",
    "put_ratio_spread": "B",
    "call_butterfly": "B",
    "put_butterfly": "B",
    "synthetic_long": "B",
    "synthetic_short": "S",
}

_LEG2_SIDE: dict[str, str] = {
    "bull_call": "S",
    "bear_put": "S",
    "credit_call": "B",
    "credit_put": "B",
    "iron_condor": "S",
    "reverse_iron_condor": "B",
    "long_strangle": "B",
    "short_strangle": "S",
    "jade_lizard": "S",
    "reverse_jade_lizard": "S",
}


def expected_leg_side(strategy: str, leg_index: int) -> str | None:
    key = str(strategy or "").strip().lower()
    if leg_index == 1:
        return _LEG1_SIDE.get(key)
    if leg_index == 2:
        return _LEG2_SIDE.get(key)
    return None


def legs_have_directional_uncapped_risk(legs: Sequence[Mapping[str, Any]]) -> bool:
    """True when short CE or short PE qty exceeds long qty (unbounded wing risk)."""
    for opt in ("CE", "PE"):
        long_qty = 0.0
        short_qty = 0.0
        for leg in legs:
            if str(leg.get("option_type") or "").upper() != opt:
                continue
            tt = str(leg.get("transaction_type") or "").upper()
            try:
                q = float(leg.get("quantity") or 0)
            except (TypeError, ValueError):
                continue
            if q <= 0:
                continue
            if tt in ("B", "BUY"):
                long_qty += q
            elif tt in ("S", "SELL"):
                short_qty += q
        if short_qty > long_qty > 0:
            return True
    return False
