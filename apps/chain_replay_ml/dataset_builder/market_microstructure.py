"""Market Microstructure Controller — option book levels (Immediate).

Canonical levels from Angel SNAP_QUOTE L1–L5 book already stored on ticks.
``bid_ask_spread`` remains RAW Base from ``enrich_dataset_features`` (not
re-emitted here). Packaging (e.g. spread-normalized LTP step) belongs in the
Transformation Pipeline.
"""

from __future__ import annotations

from typing import Any, Iterable

from chain_replay_ml.ticks import BookSnapshot, TickTimeline

# Feature Registry names owned by controller ``token.book``.
MARKET_MICROSTRUCTURE_FEATURES: tuple[str, ...] = (
    "mid_price",
    "microprice",
    "microprice_bias",
    "book_imbalance_l1",
    "book_imbalance_l1_5",
    "bid_depth_l1_5",
    "ask_depth_l1_5",
    "book_depth_slope_bid",
    "book_depth_slope_ask",
)
MARKET_MICROSTRUCTURE_FEATURE_SET: frozenset[str] = frozenset(
    MARKET_MICROSTRUCTURE_FEATURES
)

# Pipeline-only packaging (Difference(ltp) / bid_ask_spread).
SPREAD_NORMALIZED_LTP_STEP = "ltp_step_div_bid_ask_spread"


def active_market_microstructure_features(
    active: Iterable[str] | None,
) -> frozenset[str]:
    if active is None:
        return MARKET_MICROSTRUCTURE_FEATURE_SET
    return frozenset(
        str(f) for f in active if str(f) in MARKET_MICROSTRUCTURE_FEATURE_SET
    )


def needs_market_microstructure(active: Iterable[str] | None) -> bool:
    if active is None:
        return True
    return any(str(f) in MARKET_MICROSTRUCTURE_FEATURE_SET for f in active)


def _depth_slope(quantities: tuple[int, ...]) -> float | None:
    """OLS slope of qty vs level index (0 = L1 …). Needs ≥2 levels with qty ≥ 0."""
    xs: list[float] = []
    ys: list[float] = []
    for i, q in enumerate(quantities):
        qty = float(int(q or 0))
        if qty < 0:
            continue
        xs.append(float(i))
        ys.append(qty)
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x <= 1e-12:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return float(cov / var_x)


def compute_microstructure_levels(book: BookSnapshot) -> dict[str, float | None]:
    """Pure levels from a book snapshot (rupees / dimensionless / lots)."""
    out: dict[str, float | None] = {name: None for name in MARKET_MICROSTRUCTURE_FEATURES}
    if not book.has_l1:
        return out

    bb = book.bid_prices_paise[0] / 100.0
    ba = book.ask_prices_paise[0] / 100.0
    bq1 = float(book.bid_quantities[0] or 0)
    aq1 = float(book.ask_quantities[0] or 0)

    out["mid_price"] = (bb + ba) / 2.0
    spread = max(0.0, ba - bb)
    # Prefer explicit book spread when present (matches bid_ask_spread paise path).
    if book.spread_paise and book.spread_paise > 0:
        spread = float(book.spread_paise) / 100.0

    denom_l1 = bq1 + aq1
    if denom_l1 > 0:
        # Size-weighted: more bid size → price closer to ask (buying pressure).
        out["microprice"] = (ba * bq1 + bb * aq1) / denom_l1
        out["book_imbalance_l1"] = (bq1 - aq1) / denom_l1
    else:
        out["microprice"] = out["mid_price"]
        out["book_imbalance_l1"] = None

    # Dimensionless location of microprice in the spread (≈ ±0.5 at extremes).
    mid = out["mid_price"]
    micro = out["microprice"]
    if mid is not None and micro is not None and spread > 1e-12:
        out["microprice_bias"] = (float(micro) - float(mid)) / spread

    bid_depth = float(sum(int(q or 0) for q in book.bid_quantities))
    ask_depth = float(sum(int(q or 0) for q in book.ask_quantities))
    out["bid_depth_l1_5"] = bid_depth
    out["ask_depth_l1_5"] = ask_depth
    denom_5 = bid_depth + ask_depth
    if denom_5 > 0:
        out["book_imbalance_l1_5"] = (bid_depth - ask_depth) / denom_5

    out["book_depth_slope_bid"] = _depth_slope(book.bid_quantities)
    out["book_depth_slope_ask"] = _depth_slope(book.ask_quantities)
    return out


def enrich_market_microstructure_features(
    raw: dict[str, Any],
    *,
    ts: float,
    option_timeline: TickTimeline | None = None,
    active_features: frozenset[str] | None = None,
) -> dict[str, Any]:
    wanted = active_market_microstructure_features(active_features)
    if not wanted:
        return raw
    out = dict(raw)
    if option_timeline is None:
        for name in wanted:
            out.setdefault(name, None)
        return out
    book = option_timeline.book_at(float(ts))
    if book is None:
        for name in wanted:
            out.setdefault(name, None)
        return out
    levels = compute_microstructure_levels(book)
    for name in wanted:
        out[name] = levels.get(name)
    return out
