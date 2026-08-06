"""Expiry P&L payoff curve for multi-leg option strategies."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from api.multi_leg_margin import strike_step_for_index


def _leg_option_type(leg: Mapping[str, Any]) -> str:
    opt = str(leg.get("option_type") or leg.get("type") or "").strip().upper()
    if opt in ("CE", "PE"):
        return opt
    sym = str(leg.get("trading_symbol") or "").strip().upper()
    if sym.endswith("CE"):
        return "CE"
    if sym.endswith("PE"):
        return "PE"
    return ""


def _leg_transaction_side(leg: Mapping[str, Any]) -> str:
    tt = str(leg.get("transaction_type") or "").strip().upper()
    if tt in ("B", "BUY"):
        return "B"
    if tt in ("S", "SELL"):
        return "S"
    return ""


def _leg_payoff_quantity(leg: Mapping[str, Any], lot_size: int | None) -> float:
    """
    Contract quantity for payoff (shares/units).

    Leg builders normally store broker quantity (lots × lot size). If quantity
    looks like a lot count (≤10) and lot_size is known, scale up.
    """
    try:
        qty = float(leg.get("quantity") or leg.get("qt") or leg.get("qty") or 0)
    except (TypeError, ValueError):
        return 0.0
    if qty <= 0:
        return 0.0
    if lot_size is not None and lot_size > 1 and qty <= 10 and qty == int(qty):
        return qty * float(lot_size)
    return qty


def prepare_legs_for_payoff(
    legs: Sequence[Mapping[str, Any]],
    *,
    lot_size: int | None = None,
) -> list[dict[str, Any]]:
    """Normalize leg dicts for generic expiry payoff (shared by graph + table)."""
    out: list[dict[str, Any]] = []
    for leg in legs:
        row = dict(leg)
        opt = _leg_option_type(row)
        if opt:
            row["option_type"] = opt
        qty = _leg_payoff_quantity(row, lot_size)
        if qty > 0:
            row["quantity"] = qty
        out.append(row)
    return out


def _leg_payoff_rupees(
    leg: Mapping[str, Any],
    spot: float,
    *,
    lot_size: int | None = None,
) -> float:
    try:
        qty = _leg_payoff_quantity(leg, lot_size)
        px = float(leg.get("price") or 0)
        strike = float(leg.get("strike"))
    except (TypeError, ValueError):
        return 0.0
    if qty <= 0:
        return 0.0

    opt = _leg_option_type(leg)
    side = _leg_transaction_side(leg)
    if not opt or not side:
        return 0.0

    expiry_spot = float(spot)
    if opt == "CE":
        intrinsic = max(expiry_spot - strike, 0.0)
    elif opt == "PE":
        intrinsic = max(strike - expiry_spot, 0.0)
    else:
        return 0.0

    if side == "B":
        return qty * (intrinsic - px)
    return qty * (px - intrinsic)


def strategy_payoff_rupees(
    legs: Sequence[Mapping[str, Any]],
    spot: float,
    *,
    lot_size: int | None = None,
) -> float:
    """Sum expiry payoffs for all legs at ``spot`` (generic, strategy-agnostic)."""
    prepared = prepare_legs_for_payoff(legs, lot_size=lot_size)
    return sum(
        _leg_payoff_rupees(leg, float(spot), lot_size=lot_size) for leg in prepared
    )


def payoff_spot_range(
    legs: Sequence[Mapping[str, Any]],
    *,
    index: str | None = None,
    spot: float | None = None,
    pad_pts: float | None = None,
    strategy_key: str | None = None,
) -> tuple[float, float]:
    strikes: list[float] = []
    for leg in legs:
        try:
            strikes.append(float(leg.get("strike")))
        except (TypeError, ValueError):
            continue
    step = strike_step_for_index(index)
    pad = float(pad_pts) if pad_pts is not None else max(step * 4, 200)
    if not strikes:
        s = float(spot or 24000)
        return s - pad, s + pad

    lo = min(strikes) - pad
    hi = max(strikes) + pad
    if spot is not None:
        try:
            sp = float(spot)
            lo = min(lo, sp - pad)
            hi = max(hi, sp + pad)
        except (TypeError, ValueError):
            pass
    key = str(strategy_key or "").strip().lower()
    if key in ("call_ratio_backspread", "call_ratio_spread", "synthetic_long"):
        hi = max(hi, max(strikes) + step * 50)
    if key in ("put_ratio_backspread", "put_ratio_spread", "synthetic_short"):
        lo = min(lo, min(strikes) - step * 50)
    if key in ("call_ratio_backspread", "put_ratio_backspread", "call_ratio_spread", "put_ratio_spread"):
        ext = max(step * 30, 1500)
        lo = min(lo, min(strikes) - ext)
        hi = max(hi, max(strikes) + ext)
    if hi <= lo:
        hi = lo + step * 8
    return lo, hi


def _sample_spot_prices(
    legs: Sequence[Mapping[str, Any]],
    *,
    index: str | None = None,
    spot: float | None = None,
    steps: int = 100,
    pad_pts: float | None = None,
    strategy_key: str | None = None,
) -> list[float]:
    lo, hi = payoff_spot_range(
        legs,
        index=index,
        spot=spot,
        pad_pts=pad_pts,
        strategy_key=strategy_key,
    )
    n = max(20, int(steps))
    prices: set[float] = set()
    if n == 1:
        prices.add(lo)
    else:
        for i in range(n):
            prices.add(lo + (hi - lo) * (i / (n - 1)))
    for leg in legs:
        try:
            prices.add(float(leg.get("strike")))
        except (TypeError, ValueError):
            continue
    if spot is not None:
        try:
            prices.add(float(spot))
        except (TypeError, ValueError):
            pass
    return sorted(prices)


def sample_payoff_curve(
    legs: Sequence[Mapping[str, Any]],
    *,
    index: str | None = None,
    spot: float | None = None,
    steps: int = 100,
    lot_size: int | None = None,
    pad_pts: float | None = None,
    strategy_key: str | None = None,
) -> list[tuple[float, float]]:
    prices = _sample_spot_prices(
        legs,
        index=index,
        spot=spot,
        steps=steps,
        pad_pts=pad_pts,
        strategy_key=strategy_key,
    )
    return [
        (price, strategy_payoff_rupees(legs, price, lot_size=lot_size))
        for price in prices
    ]


def max_loss_from_payoff_scan(
    legs: Sequence[Mapping[str, Any]],
    *,
    index: str | None = None,
    spot: float | None = None,
    steps: int = 120,
    lot_size: int | None = None,
) -> float | None:
    """Worst expiry P&L (rupees) by scanning the payoff curve."""
    if not legs:
        return None
    curve = sample_payoff_curve(
        legs, index=index, spot=spot, steps=steps, lot_size=lot_size
    )
    if not curve:
        return None
    min_pnl = min(pnl for _, pnl in curve)
    if min_pnl >= 0:
        return 0.0
    return abs(float(min_pnl))


def max_profit_from_payoff_scan(
    legs: Sequence[Mapping[str, Any]],
    *,
    index: str | None = None,
    spot: float | None = None,
    steps: int = 120,
    lot_size: int | None = None,
) -> float | None:
    """Best expiry P&L (rupees) by scanning the payoff curve."""
    if not legs:
        return None
    curve = sample_payoff_curve(
        legs, index=index, spot=spot, steps=steps, lot_size=lot_size
    )
    if not curve:
        return None
    max_pnl = max(pnl for _, pnl in curve)
    if max_pnl <= 0:
        return 0.0
    return float(max_pnl)


def breakevens_from_payoff_scan(
    legs: Sequence[Mapping[str, Any]],
    *,
    index: str | None = None,
    spot: float | None = None,
    steps: int = 300,
    lot_size: int | None = None,
    strategy_key: str | None = None,
) -> tuple[float | None, float | None]:
    """Approximate lower / upper breakevens from payoff zero crossings."""
    if not legs:
        return None, None
    curve = sample_payoff_curve(
        legs,
        index=index,
        spot=spot,
        steps=steps,
        lot_size=lot_size,
        strategy_key=strategy_key,
    )
    if len(curve) < 2:
        return None, None
    zeros: list[float] = []
    for i in range(len(curve) - 1):
        s0, p0 = curve[i]
        s1, p1 = curve[i + 1]
        if p0 == 0:
            zeros.append(float(s0))
        if p0 * p1 < 0:
            t = abs(p0) / (abs(p0) + abs(p1)) if (abs(p0) + abs(p1)) > 0 else 0.5
            zeros.append(float(s0 + (s1 - s0) * t))
    if not zeros:
        return None, None
    zeros.sort()
    if len(zeros) == 1:
        return zeros[0], None
    return zeros[0], zeros[-1]


_ADAPTIVE_FINE_STEP = 20.0
_ADAPTIVE_SPOT_RADIUS = 30.0
_ADAPTIVE_CENTER_RADIUS = 25.0
_ADAPTIVE_BE_RADIUS = 40.0
_ADAPTIVE_MAX_ROWS = 100


def spot_inside_profit_zone(
    spot: float,
    lower_be: float | None,
    upper_be: float | None,
    *,
    strategy_key: str | None = None,
) -> bool:
    """True when rounded spot is in the expiry profit region defined by breakevens."""
    try:
        sp = int(round(float(spot)))
    except (TypeError, ValueError):
        return False
    lbe = int(round(float(lower_be))) if lower_be is not None else None
    ube = int(round(float(upper_be))) if upper_be is not None else None
    key = str(strategy_key or "").strip().lower()
    if lbe is not None and ube is not None:
        return lbe <= sp <= ube
    if lbe is not None:
        if key == "bear_put":
            return sp <= lbe
        return sp >= lbe
    if ube is not None:
        if key == "credit_call":
            return sp <= ube
        return sp >= ube
    return False


def _projection_spot_range(
    legs: Sequence[Mapping[str, Any]],
    *,
    spot: float,
    half_range: float,
) -> tuple[float, float]:
    half = max(0.0, float(half_range))
    lo = float(spot) - half
    hi = float(spot) + half
    for leg in legs:
        try:
            strike = float(leg.get("strike"))
        except (TypeError, ValueError):
            continue
        lo = min(lo, strike - half)
        hi = max(hi, strike + half)
    return lo, hi


_CHAIN_PROJECTION_STRIKES_EACH_SIDE = 30


def chain_projection_strikes(
    spot: float,
    *,
    index: str | None = None,
    lower_be: float | None = None,
    upper_be: float | None = None,
    available_strikes: Sequence[float] | None = None,
    strikes_each_side: int = _CHAIN_PROJECTION_STRIKES_EACH_SIDE,
) -> list[float]:
    """
    Strike levels for combine P&L projection chain view.

    ``strikes_each_side`` rows below and above ATM (50/100 pt grid), plus spot
    and breakeven levels when they fall off the grid.
    """
    from api.multi_leg_margin import strike_step_for_index

    step = max(1.0, float(strike_step_for_index(index)))
    n_each = max(1, int(strikes_each_side))
    atm_grid = round(float(spot) / step) * step
    atm_strike = _index_strike_price(atm_grid)
    if available_strikes:
        avail = sorted({_index_strike_price(s) for s in available_strikes})
        if avail:
            atm_strike = min(avail, key=lambda s: abs(s - atm_grid))

    core = (
        [_index_strike_price(atm_strike - i * step) for i in range(n_each, 0, -1)]
        + [atm_strike]
        + [_index_strike_price(atm_strike + i * step) for i in range(1, n_each + 1)]
    )

    strike_set = set(core)
    strike_set.add(_index_strike_price(spot))
    for be in (lower_be, upper_be):
        if be is not None:
            strike_set.add(_index_strike_price(float(be)))
    return sorted(strike_set)


def _index_strike_price(value: float) -> float:
    return float(int(round(float(value))))


def _coarse_grid_prices(lo: float, hi: float, stride: float) -> set[float]:
    stride = max(1.0, float(stride))
    start = math.floor(lo / stride) * stride
    end = math.ceil(hi / stride) * stride
    if end < start:
        end = start
    prices: set[float] = set()
    k = start
    while k <= end + 1e-6:
        prices.add(_index_strike_price(k))
        k += stride
    return prices


def _fine_band_prices(
    anchor: float,
    radius: float,
    step: float,
    *,
    lo: float,
    hi: float,
) -> set[float]:
    step_i = max(1, int(round(float(step))))
    center = int(round(float(anchor)))
    radius_i = max(step_i, int(round(float(radius))))
    prices: set[float] = set()
    for k in range(center - radius_i, center + radius_i + 1, step_i):
        if lo - 1e-6 <= k <= hi + 1e-6:
            prices.add(float(k))
    return prices


def _cap_projection_prices(
    prices: set[float],
    *,
    max_rows: int,
    protected: set[float],
    coarse_stride: float,
) -> set[float]:
    if len(prices) <= max_rows:
        return prices
    stride = max(1.0, float(coarse_stride))
    protected_ints = {int(round(p)) for p in protected}
    removable = sorted(
        p
        for p in prices
        if int(round(p)) not in protected_ints
        and abs(float(p) - round(float(p) / stride) * stride) < 0.01
    )
    out = set(prices)
    for p in removable:
        if len(out) <= max_rows:
            break
        out.discard(p)
    return out


def adaptive_projection_prices(
    *,
    lo: float,
    hi: float,
    coarse_stride: float,
    spot: float,
    lower_be: float | None,
    upper_be: float | None,
    center_strike: float | None,
    legs: Sequence[Mapping[str, Any]],
    spot_fine_band: bool = False,
) -> set[float]:
    """Coarse backbone plus fine bands near center/breakevens; spot band optional."""
    prices = _coarse_grid_prices(lo, hi, coarse_stride)
    anchors: list[tuple[float, float, float]] = []
    if spot_fine_band:
        anchors.append((float(spot), _ADAPTIVE_SPOT_RADIUS, _ADAPTIVE_FINE_STEP))
    if center_strike is not None:
        anchors.append(
            (float(center_strike), _ADAPTIVE_CENTER_RADIUS, _ADAPTIVE_FINE_STEP)
        )
    for be in (lower_be, upper_be):
        if be is not None:
            anchors.append((float(be), _ADAPTIVE_BE_RADIUS, _ADAPTIVE_FINE_STEP))
    for anchor, radius, fine_step in anchors:
        prices |= _fine_band_prices(anchor, radius, fine_step, lo=lo, hi=hi)
    for leg in legs:
        try:
            prices.add(_index_strike_price(float(leg.get("strike"))))
        except (TypeError, ValueError):
            continue
    prices.add(_index_strike_price(float(spot)))
    protected = {_index_strike_price(float(spot))}
    for be in (lower_be, upper_be):
        if be is not None:
            protected.add(_index_strike_price(float(be)))
    if center_strike is not None:
        protected.add(_index_strike_price(float(center_strike)))
    return _cap_projection_prices(
        prices,
        max_rows=_ADAPTIVE_MAX_ROWS,
        protected=protected,
        coarse_stride=coarse_stride,
    )


def _projection_use_adaptive_key_levels(
    *,
    lower_be: float | None,
    upper_be: float | None,
    center_strike: float | None,
) -> bool:
    return (
        center_strike is not None
        or lower_be is not None
        or upper_be is not None
    )


def expiry_payoff_projection_table(
    legs: Sequence[Mapping[str, Any]],
    *,
    spot: float,
    index: str | None = None,
    half_range: float = 500,
    step: float | None = None,
    lot_size: int | None = None,
    lower_be: float | None = None,
    upper_be: float | None = None,
    center_strike: float | None = None,
    adaptive_in_profit_zone: bool = False,
) -> list[tuple[float, float]]:
    """
    Expiry P&L grid for the projection popup.

    Range = union of (spot ± half_range) and (each leg strike ± half_range),
    stepped on the index grid — same leg engine as the risk graph.

    When key levels exist, overlays 20 pt rows near center and breakevens.
    Spot 20 pt band is added only when ``adaptive_in_profit_zone`` is True
    (spot inside the profit region).
    """
    if not legs:
        return []
    try:
        center = float(spot)
    except (TypeError, ValueError):
        return []

    lo, hi = _projection_spot_range(legs, spot=center, half_range=half_range)
    stride = float(step) if step is not None else float(strike_step_for_index(index))
    stride = max(1.0, stride)

    use_adaptive = _projection_use_adaptive_key_levels(
        lower_be=lower_be,
        upper_be=upper_be,
        center_strike=center_strike,
    )
    if use_adaptive:
        prices = adaptive_projection_prices(
            lo=lo,
            hi=hi,
            coarse_stride=stride,
            spot=center,
            lower_be=lower_be,
            upper_be=upper_be,
            center_strike=center_strike,
            legs=legs,
            spot_fine_band=adaptive_in_profit_zone,
        )
    else:
        prices = _coarse_grid_prices(lo, hi, stride)
        for leg in legs:
            try:
                prices.add(_index_strike_price(float(leg.get("strike"))))
            except (TypeError, ValueError):
                continue
        prices.add(_index_strike_price(center))

    rows: list[tuple[float, float]] = []
    for expiry_spot in sorted(prices):
        pnl = strategy_payoff_rupees(legs, expiry_spot, lot_size=lot_size)
        rows.append((expiry_spot, pnl))
    return rows
