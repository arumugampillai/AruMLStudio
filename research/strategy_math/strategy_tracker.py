"""Option strategy tracker helpers (test entries + live leg P&L)."""
from __future__ import annotations

import datetime as dt
import time
from typing import Any, Iterable, Mapping, Sequence

from shared.data.data_api_utils import calculate_charges


def parse_expiry_display_date(value: Any) -> dt.date | None:
    """Parse UI expiry display (``YYYY-MM-DD``) to a calendar date."""
    s = str(value or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%d-%b-%Y", "%d-%m-%Y"):
        try:
            return dt.datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def entry_expiry_date(entry: Mapping[str, Any]) -> dt.date | None:
    cached = entry.get("expiry_date")
    if cached:
        try:
            return dt.date.fromisoformat(str(cached))
        except ValueError:
            pass
    return parse_expiry_display_date(entry.get("expiry_display"))


def purge_expired_strategy_entries(
    entries: list[dict[str, Any]],
    *,
    today: dt.date | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Drop rows whose expiry date is before today (permanent delete from JSON)."""
    ref = today or dt.date.today()
    kept: list[dict[str, Any]] = []
    removed = 0
    for entry in entries:
        exp = entry_expiry_date(entry)
        if exp is not None and exp < ref:
            removed += 1
            continue
        kept.append(entry)
    return kept, removed


def strategy_entry_visible(
    entry: Mapping[str, Any],
    *,
    current_expiry_display: str,
    today: dt.date | None = None,
) -> bool:
    """
    Show current selected expiry and future expiries only.

    - expiry > today  → visible
    - expiry == today → visible only if it matches current chain expiry
    - expiry < today  → not in JSON (purged)
    """
    ref = today or dt.date.today()
    exp = entry_expiry_date(entry)
    if exp is None:
        return True
    if exp < ref:
        return False
    if exp > ref:
        return True
    cur = str(current_expiry_display or "").strip()
    return bool(cur) and str(entry.get("expiry_display") or "").strip() == cur


def strategy_label_for_key(key: str, labels: Mapping[str, str] | None = None) -> str:
    if labels and key in labels:
        return str(labels[key])
    return str(key or "?").replace("_", " ").title()


def _strategy_labels_map() -> dict[str, str]:
    try:
        from research.strategy_math.option_strategy_side_tab import _STRATEGY_LABELS

        return dict(_STRATEGY_LABELS)
    except ImportError:
        return {}


def infer_strategy_key_from_legs(
    legs: Sequence[Mapping[str, Any]],
    base_key: str,
) -> str:
    """
    Infer effective strategy from built leg sides/qty when UI key is a spread
    family but leg multipliers change the structure (e.g. bull_call 1:2 → call_ratio_spread).
    """
    base = str(base_key or "").strip().lower()
    if len(legs) != 2:
        return base

    opt = str(legs[0].get("option_type") or "").strip().upper()
    if str(legs[1].get("option_type") or "").strip().upper() != opt:
        return base
    if opt not in ("CE", "PE"):
        return base

    parsed: list[tuple[str, float, int]] = []
    for leg in legs:
        side = str(leg.get("transaction_type") or "").upper()
        side_ch = "B" if side in ("B", "BUY") else "S"
        try:
            strike = float(leg.get("strike"))
        except (TypeError, ValueError):
            return base
        try:
            qty = max(1, int(leg.get("quantity") or 1))
        except (TypeError, ValueError):
            qty = 1
        parsed.append((side_ch, strike, qty))

    buys = [p for p in parsed if p[0] == "B"]
    sells = [p for p in parsed if p[0] == "S"]
    if len(buys) != 1 or len(sells) != 1:
        return base

    _, buy_k, buy_q = buys[0]
    _, sell_k, sell_q = sells[0]
    from math import gcd

    g = gcd(buy_q, sell_q) or 1
    br, sr = buy_q // g, sell_q // g

    if opt == "CE":
        if buy_k < sell_k:
            if br == 1 and sr == 1:
                return "bull_call"
            if br == 1 and sr == 2:
                return "call_ratio_spread"
            if br == 2 and sr == 1:
                return "call_ratio_backspread"
        elif sell_k < buy_k:
            if sr == 1 and br == 1:
                return "credit_call"
            if sr == 1 and br == 2:
                return "call_ratio_backspread"
            if sr == 2 and br == 1:
                return "call_ratio_spread"
    elif opt == "PE":
        if buy_k > sell_k:
            if br == 1 and sr == 1:
                return "bear_put"
            if br == 1 and sr == 2:
                return "put_ratio_spread"
            if br == 2 and sr == 1:
                return "put_ratio_backspread"
        elif sell_k > buy_k:
            if sr == 1 and br == 1:
                return "credit_put"
            if sr == 1 and br == 2:
                return "put_ratio_backspread"
            if sr == 2 and br == 1:
                return "put_ratio_spread"

    return base


def resolve_strategy_key_and_label(
    legs: Sequence[Mapping[str, Any]],
    base_key: str,
    base_label: str = "",
) -> tuple[str, str]:
    key = infer_strategy_key_from_legs(legs, base_key)
    labels = _strategy_labels_map()
    if key == str(base_key or "").strip().lower() and str(base_label or "").strip():
        return key, str(base_label).strip()
    return key, strategy_label_for_key(key, labels)


def combine_entry_effective_strategy(entry: Mapping[str, Any]) -> tuple[str, str]:
    """Display/save key+label from stored legs (fixes legacy bull_call 1:2 rows)."""
    legs = entry.get("legs") or []
    base_key = str(entry.get("strategy_key") or "")
    base_label = str(entry.get("strategy_label") or "")
    return resolve_strategy_key_and_label(legs, base_key, base_label)


def compact_legs_summary(legs: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for leg in legs:
        side = str(leg.get("transaction_type") or "").upper()
        side_txt = "S" if side in ("S", "SELL") else "B"
        strike = leg.get("strike")
        opt = str(leg.get("option_type") or "").strip().upper()
        if strike is not None and opt in ("CE", "PE"):
            parts.append(f"{side_txt}{int(round(float(strike)))}{opt}")
        else:
            sym = str(leg.get("trading_symbol") or "").strip()
            parts.append(f"{side_txt}{sym[-8:]}" if sym else f"{side_txt}?")
    return " / ".join(parts) if parts else "—"


def format_leg_copy_token(
    leg: Mapping[str, Any],
    *,
    lot_size: int | None = None,
    qty_lots: int = 1,
) -> str:
    """Clipboard leg token: S23600CE @ 45.00 ×1 lot."""
    side = str(leg.get("transaction_type") or "").upper()
    side_txt = "S" if side in ("S", "SELL") else "B"
    strike = leg.get("strike")
    opt = str(leg.get("option_type") or "").strip().upper()
    if strike is not None and opt in ("CE", "PE"):
        token = f"{side_txt}{int(round(float(strike)))}{opt}"
    else:
        sym = str(leg.get("trading_symbol") or "").strip()
        token = f"{side_txt}{sym[-8:]}" if sym else f"{side_txt}?"
    try:
        px = float(leg.get("price") or 0)
    except (TypeError, ValueError):
        px = 0.0
    px_s = f" @ {px:.2f}" if px > 0 else ""
    lot_s = _leg_lot_label(leg, qty_lots=qty_lots, lot_size=lot_size)
    if lot_s:
        return f"{token}{px_s} {lot_s}"
    return f"{token}{px_s}".strip()


def format_legs_copy_summary(
    legs: Sequence[Mapping[str, Any]],
    *,
    lot_size: int | None = None,
    qty_lots: int = 1,
) -> str:
    parts = [
        format_leg_copy_token(leg, lot_size=lot_size, qty_lots=qty_lots) for leg in legs
    ]
    return " / ".join(parts) if parts else "—"


def format_legs_copy_block(
    legs: Sequence[Mapping[str, Any]],
    *,
    lot_size: int | None = None,
    qty_lots: int = 1,
) -> str:
    if not legs:
        return "Leg 1: —"
    return "\n".join(
        f"Leg {i}: {format_leg_copy_token(leg, lot_size=lot_size, qty_lots=qty_lots)}"
        for i, leg in enumerate(legs, start=1)
    )


def format_entry_legs_copy(entry: Mapping[str, Any]) -> str:
    from config import manipulate_prefs as mp

    index = str(entry.get("index") or "NIFTY").upper()
    lot_size = mp.lot_size_for_index(index)
    qty_lots = entry_qty_lots(entry, lot_size=lot_size)
    legs = resolve_entry_legs(entry, lot_size=lot_size)
    return format_legs_copy_block(legs, lot_size=lot_size, qty_lots=qty_lots)


def _leg_lot_label(
    leg: Mapping[str, Any],
    *,
    qty_lots: int,
    lot_size: int | None,
) -> str:
    try:
        q = int(leg.get("quantity") or 0)
    except (TypeError, ValueError):
        q = 0
    if q > 0 and lot_size is not None and lot_size > 0:
        lots = q / float(lot_size)
        if abs(lots - round(lots)) < 1e-6:
            n = int(round(lots))
            return f"×{n} lot" if n == 1 else f"×{n} lots"
    return f"×{max(1, int(qty_lots))} lot"


def format_legs_detail(
    legs: Sequence[Mapping[str, Any]],
    *,
    qty_lots: int = 1,
    lot_size: int | None = None,
    ltp_by_angel_token: Mapping[str, float] | None = None,
) -> str:
    lines: list[str] = []
    for i, leg in enumerate(legs, 1):
        side = "Sell" if _side_is_sell(dict(leg)) else "Buy"
        try:
            px = float(leg.get("price") or 0)
        except (TypeError, ValueError):
            px = 0.0
        lot_txt = _leg_lot_label(leg, qty_lots=qty_lots, lot_size=lot_size)
        ltp_txt = ""
        if ltp_by_angel_token is not None:
            atok = leg_angel_token(leg)
            if atok:
                try:
                    live = float(ltp_by_angel_token.get(atok, 0) or 0)
                    if live > 0:
                        ltp_txt = f"  LTP {live:.2f}"
                except (TypeError, ValueError):
                    pass
        strike = leg.get("strike")
        opt = str(leg.get("option_type") or "").strip().upper()
        if strike is not None and opt in ("CE", "PE"):
            lines.append(
                f"{i}. {side} {int(round(float(strike)))} {opt} @ {px:.2f}{ltp_txt}  ({lot_txt})"
            )
        else:
            sym = str(leg.get("angel_symbol") or leg.get("trading_symbol") or "?").strip()
            lines.append(f"{i}. {side} {sym} @ {px:.2f}{ltp_txt}  ({lot_txt})")
    return "\n".join(lines) if lines else "—"


def format_entry_legs_detail(
    entry: Mapping[str, Any],
    *,
    include_live_ltp: bool = False,
) -> str:
    from config import manipulate_prefs as mp

    index = str(entry.get("index") or "NIFTY").upper()
    lot_size = mp.lot_size_for_index(index)
    qty_lots = entry_qty_lots(entry, lot_size=lot_size)
    legs = legs_for_projection(entry, lot_size=lot_size)
    ltp_cache = None
    if include_live_ltp and str(entry.get("status") or "").upper() == "OPEN":
        try:
            from research.strategy_math.strategy_ltp_feed import get_strategy_ltp_cache

            ltp_cache = get_strategy_ltp_cache()
        except Exception:
            ltp_cache = None
    return format_legs_detail(
        legs,
        qty_lots=qty_lots,
        lot_size=lot_size,
        ltp_by_angel_token=ltp_cache,
    )


def entry_broker_qty(entry: Mapping[str, Any], *, lot_size: int) -> int:
    """
    Broker quantity for strategy entry legs.

    ``entry["qty"]`` is stored as broker units (lots × lot size) from
    ``_order_qty()``, not raw lot count.
    """
    try:
        qty = max(1, int(entry.get("qty") or 1))
    except (TypeError, ValueError):
        qty = 1
    lot = max(1, int(lot_size))
    if qty >= lot:
        return qty
    return qty * lot


def entry_qty_lots(entry: Mapping[str, Any], *, lot_size: int) -> int:
    """Lot count for display (inverse of entry_broker_qty)."""
    try:
        qty = max(1, int(entry.get("qty") or 1))
    except (TypeError, ValueError):
        return 1
    lot = max(1, int(lot_size))
    if qty >= lot:
        return max(1, qty // lot)
    return qty


def _infer_leg_broker_qty(
    strategy_key: str,
    leg: Mapping[str, Any],
    *,
    fallback_broker_qty: int,
) -> int:
    """Rebuild per-leg broker qty for stored entries missing ``quantity``."""
    key = str(strategy_key or "").strip().lower()
    tt = str(leg.get("transaction_type") or "").upper()
    side = "S" if tt in ("S", "SELL") else "B"
    base = max(1, int(fallback_broker_qty))
    if key in ("call_butterfly", "put_butterfly") and side == "S":
        return base * 2
    if key in ("call_ratio_backspread", "put_ratio_backspread") and side == "B":
        return base * 2
    if key in ("call_ratio_spread", "put_ratio_spread") and side == "S":
        return base * 2
    return base


def _entry_leg_broker_qty(
    leg: Mapping[str, Any],
    *,
    strategy_key: str,
    fallback_broker_qty: int,
) -> int:
    try:
        stored = leg.get("quantity")
        if stored is not None:
            qty = int(stored)
            if qty > 0:
                return qty
    except (TypeError, ValueError):
        pass
    return _infer_leg_broker_qty(
        strategy_key, leg, fallback_broker_qty=fallback_broker_qty
    )


def legs_for_projection(
    entry: Mapping[str, Any],
    *,
    lot_size: int,
) -> list[dict[str, Any]]:
    broker_qty = entry_broker_qty(entry, lot_size=lot_size)
    strategy_key = str(entry.get("strategy_key") or "")
    out: list[dict[str, Any]] = []
    for leg in entry.get("legs") or []:
        row = dict(leg)
        try:
            row["price"] = float(leg.get("price") or 0)
        except (TypeError, ValueError):
            row["price"] = 0.0
        row["quantity"] = _entry_leg_broker_qty(
            leg,
            strategy_key=strategy_key,
            fallback_broker_qty=broker_qty,
        )
        row["token"] = leg_angel_token(leg) or leg_neo_token(leg)
        row["neo_token"] = leg_neo_token(leg)
        row["angel_token"] = leg_angel_token(leg)
        row["angel_symbol"] = str(leg.get("angel_symbol") or "").strip()
        if leg.get("strike") is not None:
            try:
                row["strike"] = float(leg.get("strike"))
            except (TypeError, ValueError):
                pass
        out.append(row)
    return out


def resolve_entry_legs(
    entry: Mapping[str, Any],
    *,
    lot_size: int | None = None,
) -> list[dict[str, Any]]:
    """Entry legs with broker ``quantity`` on each row (inference for older rows)."""
    from config import manipulate_prefs as mp

    index = str(entry.get("index") or "NIFTY").upper()
    lot = max(1, int(lot_size or mp.lot_size_for_index(index)))
    return legs_for_projection(entry, lot_size=lot)


def entry_net_per_unit_for_entry(entry: Mapping[str, Any]) -> float:
    legs = resolve_entry_legs(entry)
    return entry_net_per_unit(legs)


def format_entry_pnl_summary(
    entry: Mapping[str, Any],
    *,
    unrealized: float | None = None,
) -> str:
    col_lines: list[str] = []
    margin = entry.get("margin_est")
    if margin is not None:
        col_lines.append(f"Margin ₹{float(margin):,.0f}")
    try:
        entry_net = entry_net_per_unit_for_entry(entry)
        col_lines.append(f"Entry net {entry_net:+.2f}/unit")
    except (TypeError, ValueError):
        pass
    spot = entry.get("spot_at_entry")
    if spot is not None:
        col_lines.append(f"Entry spot {float(spot):,.2f}")
    lines: list[str] = []
    if col_lines:
        lines.append("\n".join(col_lines))
    status = str(entry.get("status") or "").upper()
    tail: list[str] = []
    if status == "OPEN" and unrealized is not None:
        tail.append(f"Unrealized P&L ₹{unrealized:,.0f}")
    elif status == "COMPLETED":
        net = entry.get("net_pnl")
        if net is not None:
            tail.append(f"Net P&L ₹{float(net):,.0f}")
    mode = "Test" if entry.get("test_mode") else "Live"
    tail.append(mode)
    if tail:
        lines.append("  ·  ".join(tail))
    return "\n".join(lines)


def apply_live_ltp_to_legs(
    legs: list[dict[str, Any]],
    ltp_by_token: Mapping[str, float],
) -> None:
    for leg in legs:
        tok = leg_angel_token(leg)
        if not tok:
            continue
        raw = ltp_by_token.get(tok)
        if raw is None:
            continue
        try:
            ltp = float(raw)
            if ltp > 0:
                leg["live_ltp"] = ltp
        except (TypeError, ValueError):
            pass


def leg_neo_token(leg: Mapping[str, Any]) -> str:
    return str(
        leg.get("neo_token") or leg.get("instrument_token") or leg.get("token") or ""
    ).strip()


def leg_angel_token(leg: Mapping[str, Any]) -> str:
    return str(leg.get("angel_token") or "").strip()


def resolve_leg_angel_identity(
    leg: Mapping[str, Any],
    *,
    expiry_date: dt.date | None = None,
) -> tuple[str, str]:
    """Resolve (angel_symbol, angel_token) for a strategy leg."""
    sym = str(leg.get("angel_symbol") or "").strip()
    tok = leg_angel_token(leg)
    if sym and tok:
        return sym, tok
    try:
        from angelone.angel_margin import _resolve_angel_token, load_angel_instruments_df
        from angelone.symbol_bridge import compact_kotak_index_option_to_angel

        df = load_angel_instruments_df()
        leg_copy = dict(leg)
        if expiry_date is not None and not leg_copy.get("expiry_date"):
            leg_copy["expiry_date"] = expiry_date
        resolved_tok = _resolve_angel_token(leg_copy, df, options_df=None)
        if not resolved_tok:
            return sym, tok
        if not sym:
            opt = str(leg.get("option_type") or "").strip().upper()
            ts = str(leg.get("trading_symbol") or "").strip()
            built = compact_kotak_index_option_to_angel(ts, opt)
            sym = built or ts
        return sym, str(resolved_tok).strip()
    except Exception:
        return sym, tok


def enrich_stored_leg(
    leg: dict[str, Any],
    *,
    expiry_date: dt.date | None = None,
) -> dict[str, Any]:
    """Ensure neo + angel identity fields exist on a persisted leg row."""
    row = dict(leg)
    neo = leg_neo_token(row)
    if neo:
        row["neo_token"] = neo
        row["instrument_token"] = neo
    sym, tok = resolve_leg_angel_identity(row, expiry_date=expiry_date)
    if sym:
        row["angel_symbol"] = sym
    if tok:
        row["angel_token"] = tok
    return row


def enrich_strategy_entry(entry: dict[str, Any]) -> bool:
    """Fill missing angel tokens on stored legs; return True if any leg changed."""
    exp = entry_expiry_date(entry)
    legs = entry.get("legs") or []
    changed = False
    new_legs: list[dict[str, Any]] = []
    for leg in legs:
        if not isinstance(leg, dict):
            new_legs.append(leg)
            continue
        before_tok = leg_angel_token(leg)
        before_sym = str(leg.get("angel_symbol") or "").strip()
        enriched = enrich_stored_leg(dict(leg), expiry_date=exp)
        if leg_angel_token(enriched) != before_tok or str(
            enriched.get("angel_symbol") or ""
        ).strip() != before_sym:
            changed = True
        new_legs.append(enriched)
    entry["legs"] = new_legs
    return changed


def enrich_strategy_entries(entries: list[dict[str, Any]]) -> bool:
    changed = False
    for entry in entries:
        if enrich_strategy_entry(entry):
            changed = True
    return changed


def open_angel_tokens_from_entries(
    entries: Iterable[Mapping[str, Any]],
) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        if str(entry.get("status") or "").upper() != "OPEN":
            continue
        for leg in entry.get("legs") or []:
            tok = leg_angel_token(leg)
            if tok and tok not in seen:
                seen.add(tok)
                tokens.append(tok)
    return tokens


def _side_is_sell(leg: dict[str, Any]) -> bool:
    return str(leg.get("transaction_type") or "").upper() in ("S", "SELL")


def _leg_broker_qty(leg: Mapping[str, Any]) -> float:
    try:
        qty = float(leg.get("quantity") or leg.get("qt") or leg.get("qty") or 0)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, qty)


def _structure_base_qty(legs: Sequence[Mapping[str, Any]]) -> float:
    from api.multi_leg_margin import _structure_base_qty as base_qty

    return float(base_qty(legs))


def entry_net_rupees(legs: Sequence[Mapping[str, Any]]) -> float:
    """Signed entry cash in rupees: credit positive, debit negative."""
    total = 0.0
    for leg in legs:
        try:
            px = float(leg.get("price") or 0)
        except (TypeError, ValueError):
            px = 0.0
        qty = _leg_broker_qty(leg)
        if qty <= 0:
            qty = 1.0
        cash = px * qty
        if _side_is_sell(dict(leg)):
            total += cash
        else:
            total -= cash
    return total


def entry_net_per_unit(legs: list[dict[str, Any]]) -> float:
    """Signed entry cash per index point (per share): credit positive, debit negative."""
    base = _structure_base_qty(legs)
    if base <= 0:
        return entry_net_rupees(legs)
    return entry_net_rupees(legs) / base


def _leg_close_ltp(
    leg: Mapping[str, Any],
    ltp_by_token: Mapping[str, float],
) -> float | None:
    tok = leg_angel_token(leg)
    if not tok:
        return None
    ltp = float(ltp_by_token.get(tok, 0) or 0)
    if ltp <= 0:
        return None
    return ltp


def mark_to_market_rupees(
    entry: dict[str, Any],
    ltp_by_token: Mapping[str, float],
) -> float | None:
    """Current unrealized P&L in rupees from live LTPs (no charges)."""
    legs = resolve_entry_legs(entry)
    if not legs:
        return None
    entry_cash = entry_net_rupees(legs)
    close_cash = 0.0
    for leg in legs:
        ltp = _leg_close_ltp(leg, ltp_by_token)
        if ltp is None:
            return None
        qty = _leg_broker_qty(leg)
        if qty <= 0:
            qty = 1.0
        cash = ltp * qty
        if _side_is_sell(dict(leg)):
            close_cash -= cash
        else:
            close_cash += cash
    return entry_cash + close_cash


def finalize_net_pnl_rupees(entry: dict[str, Any], ltp_by_token: Mapping[str, float]) -> float | None:
    """Net P&L with estimated charges when closing at current LTPs."""
    gross = mark_to_market_rupees(entry, ltp_by_token)
    if gross is None:
        return None
    legs = resolve_entry_legs(entry)
    buy_val = 0.0
    sell_val = 0.0
    for leg in legs:
        ltp = _leg_close_ltp(leg, ltp_by_token)
        if ltp is None:
            return gross
        qty = _leg_broker_qty(leg)
        if qty <= 0:
            qty = 1.0
        val = ltp * qty
        if _side_is_sell(dict(leg)):
            sell_val += val
        else:
            buy_val += val
    charges = calculate_charges(buy_val, sell_val)
    return gross - charges


def new_strategy_entry(
    *,
    strategy_key: str,
    strategy_label: str,
    legs: list[dict[str, Any]],
    qty: int,
    index: str,
    wing_steps: int,
    test_mode: bool,
    spot_at_entry: float | None = None,
    expiry_display: str = "",
    expiry_raw: str = "",
    margin_est: float | None = None,
    max_profit: float | None = None,
    max_risk: float | None = None,
    order_ids: list[str] | None = None,
) -> dict[str, Any]:
    now = time.time()
    exp_disp = str(expiry_display or "").strip()
    exp_date = parse_expiry_display_date(exp_disp)
    stored_legs: list[dict[str, Any]] = []
    for leg in legs:
        neo_tok = str(leg.get("instrument_token") or leg.get("token") or "").strip()
        row = {
            "trading_symbol": str(leg.get("trading_symbol") or "").strip(),
            "instrument_token": neo_tok,
            "neo_token": neo_tok,
            "exchange_segment": str(leg.get("exchange_segment") or "nse_fo"),
            "transaction_type": str(leg.get("transaction_type") or "B").upper()[:1],
            "price": float(leg.get("price") or 0),
            "quantity": max(1, int(leg.get("quantity") or qty)),
            "strike": leg.get("strike"),
            "option_type": str(leg.get("option_type") or "").strip().upper(),
            "product": str(leg.get("product") or "MIS").upper(),
            "order_id": None,
        }
        stored_legs.append(enrich_stored_leg(row, expiry_date=exp_date))
    oids = [str(o) for o in (order_ids or []) if str(o).strip()]
    for i, oid in enumerate(oids):
        if i < len(stored_legs):
            stored_legs[i]["order_id"] = oid
    return {
        "cycle_id": f"strat_{int(now * 1000)}",
        "strategy_key": strategy_key,
        "strategy_label": strategy_label,
        "opened_at_epoch": now,
        "closed_at_epoch": None,
        "status": "OPEN",
        "test_mode": bool(test_mode),
        "index": str(index or "NIFTY").upper(),
        "spot_at_entry": float(spot_at_entry) if spot_at_entry is not None else None,
        "expiry_display": exp_disp,
        "expiry_raw": str(expiry_raw or "").strip(),
        "expiry_date": exp_date.isoformat() if exp_date else None,
        "qty": max(1, int(qty)),
        "wing_steps": int(wing_steps),
        "legs": stored_legs,
        "entry_net_per_unit": entry_net_per_unit(
            [dict(leg) for leg in stored_legs]
        ),
        "margin_est": float(margin_est) if margin_est is not None else None,
        "max_profit": float(max_profit) if max_profit is not None else None,
        "max_risk": float(max_risk) if max_risk is not None else None,
        "net_pnl": None,
        "order_ids": oids,
    }
