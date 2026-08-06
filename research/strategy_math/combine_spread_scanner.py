"""Scan option chain for bull call / bear put spreads by width and min ROR."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

SPREAD_WIDTH_CHOICES = (50, 100, 150, 200, 250, 300)
MIN_SCAN_ENTRY_NET = 1.0
SCAN_QTY_INCREMENT_ENTRY_LT = 20.0
SCAN_QTY_INCREMENT_ENTRY_TIER_2_LT = 2.5
SCAN_QTY_INCREMENT_ENTRY_TIER_3_LT = 7.0
SCAN_QTY_INCREMENT_MAX_MULT = 19


def chain_strikes_for_expiry(
    options_df: Any,
    *,
    index: str,
    expiry_raw: str,
) -> list[float]:
    """Sorted unique strikes on the loaded chain for index + expiry."""
    from api.multi_leg_margin import _expiry_date_key

    if options_df is None or getattr(options_df, "empty", True):
        return []
    expiry_key = _expiry_date_key(expiry_raw)
    idx = str(index or "").strip().upper()
    if not expiry_key or not idx:
        return []
    mask = (options_df["pSymbolName"].str.upper() == idx) & (
        options_df["ExpiryDate"].map(_expiry_date_key) == expiry_key
    )
    sub = options_df.loc[mask, "dStrikePrice;"]
    out: list[float] = []
    for raw in sub.dropna().unique():
        try:
            out.append(float(raw))
        except (TypeError, ValueError):
            continue
    out.sort()
    return out


@dataclass
class SpreadScanHit:
    strategy_key: str
    strategy_label: str
    legs: list[dict[str, Any]]
    qty: int
    index: str
    expiry_display: str
    expiry_raw: str
    exchange: str
    wing_steps: int
    width_pts: int
    entry_net: float
    max_profit: float | None
    max_risk: float | None
    ror_pct: float | None
    spot_at_add: float | None


def wing_steps_for_width(width_pts: int, *, index: str) -> int:
    from api.multi_leg_margin import strike_step_for_index

    step = max(1, int(strike_step_for_index(index)))
    wings = int(round(float(width_pts) / float(step)))
    return max(1, wings)


def _strike_set(strikes: Sequence[float]) -> set[float]:
    return {float(s) for s in strikes}


def entry_magnitude_qty_multiplier(entry_net: float, *, qty_increment: bool) -> int:
    """Map |entry| bands to qty multipliers; same band for bull and bear."""
    if not qty_increment:
        return 1
    try:
        entry = abs(float(entry_net))
    except (TypeError, ValueError):
        return 1
    if entry >= SCAN_QTY_INCREMENT_ENTRY_LT:
        return 1
    if entry < SCAN_QTY_INCREMENT_ENTRY_TIER_2_LT:
        return 2
    if entry < SCAN_QTY_INCREMENT_ENTRY_TIER_3_LT:
        return 3
    return min(SCAN_QTY_INCREMENT_MAX_MULT, 4)


def hit_eligible_for_qty_increment(hit: SpreadScanHit) -> bool:
    try:
        entry = abs(float(hit.entry_net))
    except (TypeError, ValueError):
        return False
    return entry < SCAN_QTY_INCREMENT_ENTRY_LT


def rescaled_scan_hit(hit: SpreadScanHit, new_broker_qty: int) -> SpreadScanHit:
    """Scale leg quantities and refresh spread metrics for a scan row."""
    from api.multi_leg_margin import (
        estimate_strategy_max_profit_rupees,
        estimate_strategy_risk_rupees,
        estimate_strategy_rom_pct,
    )
    from research.strategy_math.strategy_tracker import entry_net_per_unit

    old_qty = max(1, int(hit.qty))
    new_qty = max(1, int(new_broker_qty))
    if new_qty == old_qty:
        return hit
    factor = float(new_qty) / float(old_qty)
    new_legs: list[dict[str, Any]] = []
    for leg in hit.legs or []:
        row = dict(leg)
        try:
            leg_q = int(leg.get("quantity") or old_qty)
        except (TypeError, ValueError):
            leg_q = old_qty
        row["quantity"] = max(1, int(round(leg_q * factor)))
        new_legs.append(row)

    risk = estimate_strategy_risk_rupees(
        hit.strategy_key, new_legs, index=hit.index, wing_steps=hit.wing_steps
    )
    profit, profit_uncapped = estimate_strategy_max_profit_rupees(
        hit.strategy_key, new_legs, index=hit.index, wing_steps=hit.wing_steps
    )
    ror: float | None = None
    if not profit_uncapped and profit is not None and risk is not None and risk > 0:
        ror = estimate_strategy_rom_pct(
            hit.strategy_key,
            max_risk=risk,
            max_profit=profit,
            profit_uncapped=profit_uncapped,
            risk_uncapped=False,
        )
    try:
        entry_net = float(entry_net_per_unit(new_legs))
    except (TypeError, ValueError):
        entry_net = hit.entry_net

    return SpreadScanHit(
        strategy_key=hit.strategy_key,
        strategy_label=hit.strategy_label,
        legs=new_legs,
        qty=new_qty,
        index=hit.index,
        expiry_display=hit.expiry_display,
        expiry_raw=hit.expiry_raw,
        exchange=hit.exchange,
        wing_steps=hit.wing_steps,
        width_pts=hit.width_pts,
        entry_net=entry_net,
        max_profit=float(profit) if profit is not None else None,
        max_risk=float(risk) if risk is not None else None,
        ror_pct=float(ror) if ror is not None else None,
        spot_at_add=hit.spot_at_add,
    )


def apply_scan_qty_schedule(
    hits: list[SpreadScanHit],
    base_broker_qty: int,
    *,
    qty_increment: bool,
) -> list[SpreadScanHit]:
    if not hits or not qty_increment:
        return list(hits)
    base = max(1, int(base_broker_qty))
    out: list[SpreadScanHit] = []
    for hit in hits:
        if hit_eligible_for_qty_increment(hit):
            mult = entry_magnitude_qty_multiplier(hit.entry_net, qty_increment=True)
            out.append(rescaled_scan_hit(hit, base * mult))
        else:
            out.append(hit)
    return out


def scan_vertical_spreads(
    *,
    options_df: Any,
    index: str,
    expiry_raw: str,
    expiry_display: str,
    exchange: str,
    width_pts: int,
    min_ror_pct: float,
    quantity: int,
    product: str,
    ltp_tracker: Mapping[str, Any] | None = None,
    spot: float | None = None,
    include_bull_call: bool = True,
    include_bear_put: bool = True,
    qty_increment: bool = False,
) -> list[SpreadScanHit]:
    from api.multi_leg_margin import (
        build_bear_put_spread_legs,
        build_bull_call_spread_legs,
        estimate_strategy_max_profit_rupees,
        estimate_strategy_risk_rupees,
        estimate_strategy_rom_pct,
    )
    from research.strategy_math.strategy_tracker import entry_net_per_unit

    strikes = chain_strikes_for_expiry(
        options_df, index=index, expiry_raw=expiry_raw
    )
    if not strikes:
        return []

    wing_steps = wing_steps_for_width(width_pts, index=index)
    width = float(width_pts)
    ce_set = _strike_set(strikes)
    pe_set = ce_set
    qty_i = max(1, int(quantity))
    min_ror = float(min_ror_pct)
    hits: list[SpreadScanHit] = []

    def _maybe_add(
        strategy_key: str,
        strategy_label: str,
        legs: list[dict[str, Any]] | None,
    ) -> None:
        if not legs:
            return
        risk = estimate_strategy_risk_rupees(
            strategy_key, legs, index=index, wing_steps=wing_steps
        )
        profit, profit_uncapped = estimate_strategy_max_profit_rupees(
            strategy_key, legs, index=index, wing_steps=wing_steps
        )
        if profit_uncapped or profit is None or risk is None or risk <= 0:
            return
        ror = estimate_strategy_rom_pct(
            strategy_key,
            max_risk=risk,
            max_profit=profit,
            profit_uncapped=profit_uncapped,
            risk_uncapped=False,
        )
        if ror is None or ror < min_ror:
            return
        try:
            entry_net = float(entry_net_per_unit(legs))
        except (TypeError, ValueError):
            entry_net = 0.0
        if abs(entry_net) <= MIN_SCAN_ENTRY_NET:
            return
        hits.append(
            SpreadScanHit(
                strategy_key=strategy_key,
                strategy_label=strategy_label,
                legs=[dict(leg) for leg in legs],
                qty=qty_i,
                index=str(index).upper(),
                expiry_display=str(expiry_display or ""),
                expiry_raw=str(expiry_raw or ""),
                exchange=str(exchange or "nse_fo"),
                wing_steps=wing_steps,
                width_pts=int(width_pts),
                entry_net=entry_net,
                max_profit=float(profit),
                max_risk=float(risk),
                ror_pct=float(ror),
                spot_at_add=float(spot) if spot is not None else None,
            )
        )

    if include_bull_call:
        for buy_k in strikes:
            sell_k = float(buy_k) + width
            if sell_k not in ce_set:
                continue
            legs = build_bull_call_spread_legs(
                options_df=options_df,
                index=index,
                expiry_raw=expiry_raw,
                selected_strike=float(buy_k),
                long_token=None,
                long_price=None,
                exchange=exchange,
                quantity=qty_i,
                product=product,
                ltp_tracker=ltp_tracker,
                wing_steps=wing_steps,
                second_strike=sell_k,
            )
            _maybe_add("bull_call", "Bull Call Spread", legs)

    if include_bear_put:
        for sell_k in strikes:
            buy_k = float(sell_k) + width
            if buy_k not in pe_set:
                continue
            legs = build_bear_put_spread_legs(
                options_df=options_df,
                index=index,
                expiry_raw=expiry_raw,
                selected_strike=float(buy_k),
                long_token=None,
                long_price=None,
                exchange=exchange,
                quantity=qty_i,
                product=product,
                ltp_tracker=ltp_tracker,
                wing_steps=wing_steps,
                second_strike=float(sell_k),
            )
            _maybe_add("bear_put", "Bear Put Spread", legs)

    hits.sort(key=lambda h: (h.ror_pct or 0.0, h.max_profit or 0.0), reverse=True)
    return apply_scan_qty_schedule(hits, qty_i, qty_increment=bool(qty_increment))


def spread_scan_hit_to_combine_entry(
    hit: SpreadScanHit,
    *,
    margin_est: float | None = None,
) -> dict[str, Any]:
    from research.strategy_math.combine_strategy_tracker import new_combine_entry

    return new_combine_entry(
        strategy_key=hit.strategy_key,
        strategy_label=hit.strategy_label,
        legs=hit.legs,
        qty=hit.qty,
        index=hit.index,
        wing_steps=hit.wing_steps,
        spot_at_add=hit.spot_at_add,
        expiry_display=hit.expiry_display,
        expiry_raw=hit.expiry_raw,
        margin_est=margin_est,
        max_profit=hit.max_profit,
        max_risk=hit.max_risk,
        max_profit_uncapped=False,
        max_risk_uncapped=False,
    )


def scanner_chain_context(order_panel: Any) -> dict[str, Any]:
    """Resolve option chain context from the live order panel."""
    op = order_panel
    if op is None:
        raise ValueError("order panel not ready")
    tm = getattr(op, "top_menu", None)
    if tm is None:
        raise ValueError("option chain not ready")
    options_df = getattr(tm, "options_df", None)
    if options_df is None or getattr(options_df, "empty", True):
        raise ValueError("load option chain first")
    index = str(tm.index_var.get() or "NIFTY").upper()
    expiry_display = str(tm.expiry_var.get() or "").strip()
    try:
        expiry_raw = str(tm.display_to_expiry(expiry_display))
    except Exception as exc:
        raise ValueError("select expiry on chain") from exc
    from app.constants import INDEX_MASTER_FILES

    exchange = INDEX_MASTER_FILES.get(index, {}).get("exchange", "nse_fo")
    from research.strategy_math.combine_strategy_tracker import resolve_index_spot

    spot = resolve_index_spot(op, index)
    product = "MIS"
    try:
        product = str(op.product_var.get() or "MIS").upper()
    except Exception:
        pass
    return {
        "index": index,
        "expiry_display": expiry_display,
        "expiry_raw": expiry_raw,
        "options_df": options_df,
        "order_panel": op,
        "exchange": exchange,
        "spot": spot,
        "product": product,
        "ltp_tracker": getattr(op, "ltp_tracker", None),
        "client": getattr(op, "client", None),
    }


def scanner_order_qty(*, order_panel: Any, index: str) -> int:
    from config import manipulate_prefs as mp

    lot_size = mp.lot_size_for_index(index)
    if order_panel is not None and hasattr(order_panel, "_manual_order_qty"):
        try:
            return max(lot_size, int(order_panel._manual_order_qty()))
        except (TypeError, ValueError):
            pass
    return lot_size


def scanner_strategy_flags_from_prefs(prefs: dict[str, Any] | None = None) -> tuple[bool, bool]:
    from config import manipulate_prefs as mp

    choice = mp.combine_spread_scanner_strategy(prefs)
    if choice == "bull_call":
        return True, False
    if choice == "bear_put":
        return False, True
    return True, True


def scanner_strategy_label_from_prefs(prefs: dict[str, Any] | None = None) -> str:
    from config import manipulate_prefs as mp

    choice = mp.combine_spread_scanner_strategy(prefs)
    if choice == "bull_call":
        return "Bull Call"
    if choice == "bear_put":
        return "Bear Put"
    return "Both"


def margin_rupees_for_hit(
    client: Any,
    hit: SpreadScanHit,
    *,
    ltp_tracker: Mapping[str, Any] | None = None,
    options_df: Any | None = None,
) -> tuple[float | None, str | None]:
    from api.multi_leg_margin import (
        calculate_multi_leg_margin,
        extract_margin_rupees,
        margin_error_from_payload,
    )
    from angelone.angel_margin import extract_angel_margin_rupees

    if client is None:
        return None, "login required"
    try:
        result = calculate_multi_leg_margin(
            client,
            hit.legs,
            ltp_tracker=ltp_tracker,
            strategy=hit.strategy_key,
            index=hit.index,
            wing_steps=hit.wing_steps,
            options_df=options_df,
            expiry_raw=hit.expiry_raw,
        )
    except Exception as exc:
        return None, str(exc)
    amount = None
    if str(result.get("_source") or "") == "angel" and not result.get("_angel_failed"):
        amount = extract_angel_margin_rupees(result)
    if amount is None:
        amount = extract_margin_rupees(
            result,
            leg_count=len(hit.legs or []),
            strategy=hit.strategy_key,
        )
    if amount is None:
        err = (
            result.get("error")
            or margin_error_from_payload(result)
            or "margin unavailable"
        )
        return None, str(err)
    return float(amount), None


def auto_add_bull_bear_spreads(
    *,
    order_panel: Any,
    target_manager: Any,
) -> dict[str, Any]:
    """Scan with saved scanner prefs (ROR, width, strategy), fetch margin, add to combine."""
    from config import manipulate_prefs as mp
    from research.strategy_math.strategy_tracker import compact_legs_summary

    if target_manager is None or not hasattr(target_manager, "record_combine_entry"):
        raise ValueError("trade tracker not ready")

    prefs = mp.load()
    min_ror = mp.combine_spread_scanner_min_ror(prefs)
    width_pts = mp.combine_spread_scanner_width_pts(prefs)
    include_bull, include_bear = scanner_strategy_flags_from_prefs(prefs)
    strategy_label = scanner_strategy_label_from_prefs(prefs)
    qty_increment = mp.combine_spread_scanner_qty_increment(prefs)
    ctx = scanner_chain_context(order_panel)
    index = str(ctx["index"])
    qty = scanner_order_qty(order_panel=order_panel, index=index)
    options_df = ctx["options_df"]
    odf = options_df.copy() if hasattr(options_df, "copy") else options_df

    hits = scan_vertical_spreads(
        options_df=options_df,
        index=index,
        expiry_raw=str(ctx["expiry_raw"]),
        expiry_display=str(ctx["expiry_display"]),
        exchange=str(ctx["exchange"]),
        width_pts=width_pts,
        min_ror_pct=min_ror,
        quantity=qty,
        product=str(ctx["product"]),
        ltp_tracker=ctx.get("ltp_tracker"),
        spot=ctx.get("spot"),
        include_bull_call=include_bull,
        include_bear_put=include_bear,
        qty_increment=qty_increment,
    )
    if not hits:
        return {
            "added": 0,
            "scanned": 0,
            "margin_ok": 0,
            "errors": [],
            "min_ror": min_ror,
            "width_pts": width_pts,
            "strategy_label": strategy_label,
            "qty_increment": qty_increment,
        }

    client = ctx.get("client")
    if client is None:
        raise ValueError("Complete 2FA login before auto add")

    added = 0
    margin_ok = 0
    errors: list[str] = []
    for hit in hits:
        margin, err = margin_rupees_for_hit(
            client,
            hit,
            ltp_tracker=ctx.get("ltp_tracker"),
            options_df=odf,
        )
        if margin is not None:
            margin_ok += 1
        elif err:
            errors.append(f"{hit.strategy_label} {compact_legs_summary(hit.legs)}: {err}")
        entry = spread_scan_hit_to_combine_entry(hit, margin_est=margin)
        if target_manager.record_combine_entry(entry):
            added += 1

    return {
        "added": added,
        "scanned": len(hits),
        "margin_ok": margin_ok,
        "errors": errors,
        "min_ror": min_ror,
        "width_pts": width_pts,
        "strategy_label": strategy_label,
        "qty_increment": qty_increment,
    }
