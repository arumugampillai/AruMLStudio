"""Option-token Base tape / book levels for Registry (Tier 1–2 freeze)."""

from __future__ import annotations

from typing import Any


def emit_option_tape_features(
    raw: dict[str, Any],
    *,
    ts: float,
    option_timeline: Any | None,
) -> dict[str, Any]:
    """Emit option Base market-state from the option ``TickTimeline``.

    Registry names (distinct from legacy extractor keys ``oi`` / ``volume``):
    option_oi, option_day_volume, ltq, total_buy_qty, total_sell_qty,
    option_bid, option_ask.
    """
    out = dict(raw)
    keys = (
        "option_oi",
        "option_day_volume",
        "ltq",
        "total_buy_qty",
        "total_sell_qty",
        "option_bid",
        "option_ask",
    )
    tl = option_timeline
    if tl is None or not getattr(tl, "timestamps", None):
        for key in keys:
            out[key] = None
        return out

    oi = tl.oi_at(ts) if hasattr(tl, "oi_at") else None
    vol = tl.volume_at(ts) if hasattr(tl, "volume_at") else None
    ltq = tl.ltq_at(ts) if hasattr(tl, "ltq_at") else None
    buy = tl.total_buy_at(ts) if hasattr(tl, "total_buy_at") else None
    sell = tl.total_sell_at(ts) if hasattr(tl, "total_sell_at") else None

    out["option_oi"] = float(oi) if oi is not None else None
    out["option_day_volume"] = float(vol) if vol is not None else None
    out["ltq"] = float(ltq) if ltq is not None else None
    out["total_buy_qty"] = float(buy) if buy is not None else None
    out["total_sell_qty"] = float(sell) if sell is not None else None

    bid = ask = None
    book = tl.book_at(ts) if hasattr(tl, "book_at") else None
    if book is not None and getattr(book, "has_l1", False):
        bid_paise = int(book.bid_prices_paise[0] or 0)
        ask_paise = int(book.ask_prices_paise[0] or 0)
        if bid_paise > 0:
            bid = bid_paise / 100.0
        if ask_paise > 0:
            ask = ask_paise / 100.0
    out["option_bid"] = bid
    out["option_ask"] = ask
    return out


__all__ = ["emit_option_tape_features"]
