"""PrimitiveMappingProvider — isolated legacy→PR_* binding (Sprint 2)."""

from __future__ import annotations

from typing import Any, Callable

# Explicit map: highest priority. Extend via freeze/ADR as catalog grows.
# Only map when the feature clearly depends on a seed primitive — do not invent.
EXPLICIT_NAME_MAP: dict[str, tuple[str, ...]] = {
    "spot": ("PR_SPOT",),
    "volume": ("PR_VOLUME",),
    "oi": ("PR_OI",),
    "iv": ("PR_IV",),
    "delta": ("PR_DELTA",),
    "gamma": ("PR_GAMMA",),
    "theta": ("PR_THETA",),
    "vega": ("PR_VEGA",),
    "rho": ("PR_RHO",),
    "strike": ("PR_STRIKE",),
    "expiry": ("PR_EXPIRY",),
    "bid": ("PR_BID",),
    "ask": ("PR_ASK",),
    "bid_ask_spread": ("PR_BID", "PR_ASK"),
    "spot_change_1m": ("PR_SPOT",),
    "spot_change_5m": ("PR_SPOT",),
    "spot_change_15m": ("PR_SPOT",),
    # IV
    "atm_iv_ce": ("PR_IV",),
    "atm_iv_pe": ("PR_IV",),
    "current_iv": ("PR_IV",),
    "roll_iv": ("PR_IV",),
    # OI / PCR / walls
    "atm_pcr": ("PR_OI",),
    "chain_pcr": ("PR_OI",),
    "total_call_oi": ("PR_OI",),
    "total_put_oi": ("PR_OI",),
    "option_oi": ("PR_OI",),
    "max_call_oi_pct": ("PR_OI",),
    "max_put_oi_pct": ("PR_OI",),
    "distance_to_call_build_wall": ("PR_OI",),
    "distance_to_put_build_wall": ("PR_OI",),
    "distance_to_max_call_oi_strikes": ("PR_OI", "PR_STRIKE"),
    "distance_to_max_put_oi_strikes": ("PR_OI", "PR_STRIKE"),
    # GEX → gamma
    "call_gex": ("PR_GAMMA",),
    "put_gex": ("PR_GAMMA",),
    "chain_gex": ("PR_GAMMA",),
    "net_gex": ("PR_GAMMA",),
    # Higher-order greeks (clear dependency)
    "charm": ("PR_DELTA",),
    "vanna": ("PR_DELTA", "PR_VEGA"),
    "volga": ("PR_VEGA",),
    # Time / expiry
    "days_to_expiry": ("PR_EXPIRY",),
    "minutes_to_expiry": ("PR_EXPIRY",),
    "is_expiry_day": ("PR_EXPIRY",),
    "is_first_hour": ("PR_TIME",),
    "is_last_hour": ("PR_TIME",),
    "minute_of_day": ("PR_TIME",),
    "minutes_since_open": ("PR_TIME",),
    "minutes_to_close": ("PR_TIME",),
    "time_since_cross_min": ("PR_TIME",),
    "roll_age_min": ("PR_TIME",),
    "rows_since_roll": ("PR_TIME",),
    # Book / quotes
    "book_depth_slope_ask": ("PR_ASK",),
    "book_depth_slope_bid": ("PR_BID",),
    "book_imbalance_l1": ("PR_BID", "PR_ASK"),
    "book_imbalance_l1_5": ("PR_BID", "PR_ASK"),
    "option_bid": ("PR_BID",),
    "option_ask": ("PR_ASK",),
    "microprice": ("PR_BID", "PR_ASK"),
    "microprice_bias": ("PR_BID", "PR_ASK"),
    "mid_price": ("PR_BID", "PR_ASK"),
    # Futures / spot-like
    "futures_bid": ("PR_BID",),
    "futures_ask": ("PR_ASK",),
    "futures_oi": ("PR_OI",),
    "futures_day_volume": ("PR_VOLUME",),
    "futures_ltp": ("PR_SPOT",),
    "futures_vwap": ("PR_SPOT",),
    "futures_spread": ("PR_BID", "PR_ASK"),
    "synthetic_forward_spot": ("PR_SPOT",),
    "distance_from_spot_pct": ("PR_SPOT",),
    "weighted_spot_ema": ("PR_SPOT",),
    "weighted_spot_close_ema": ("PR_SPOT",),
    "weighted_spot_high_ema": ("PR_SPOT",),
    "weighted_spot_low_ema": ("PR_SPOT",),
    # Volume
    "ltq": ("PR_VOLUME",),
    "option_day_volume": ("PR_VOLUME",),
    "otm_ce_volume": ("PR_VOLUME",),
    "otm_pe_volume": ("PR_VOLUME",),
    "otm_pcr_volume": ("PR_VOLUME",),
    "total_ce_volume": ("PR_VOLUME",),
    "total_pe_volume": ("PR_VOLUME",),
    "total_buy_qty": ("PR_VOLUME",),
    "total_sell_qty": ("PR_VOLUME",),
}

# Prefix / token rules (applied in order).
_PREFIX_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("spot_", ("PR_SPOT",)),
    ("volume_", ("PR_VOLUME",)),
    ("oi_", ("PR_OI",)),
    ("iv_", ("PR_IV",)),
    ("atm_iv", ("PR_IV",)),
    ("delta_", ("PR_DELTA",)),
    ("gamma_", ("PR_GAMMA",)),
    ("theta_", ("PR_THETA",)),
    ("vega_", ("PR_VEGA",)),
    ("rho_", ("PR_RHO",)),
    ("strike_", ("PR_STRIKE",)),
    ("expiry_", ("PR_EXPIRY",)),
    ("bid_", ("PR_BID",)),
    ("ask_", ("PR_ASK",)),
    ("option_bid", ("PR_BID",)),
    ("option_ask", ("PR_ASK",)),
    ("futures_bid", ("PR_BID",)),
    ("futures_ask", ("PR_ASK",)),
    ("futures_oi", ("PR_OI",)),
    ("futures_day_volume", ("PR_VOLUME",)),
    ("futures_", ("PR_SPOT",)),
    ("book_", ("PR_BID", "PR_ASK")),
    ("weighted_spot_", ("PR_SPOT",)),
    ("distance_to_max_", ("PR_OI", "PR_STRIKE")),
    ("distance_to_", ("PR_OI",)),
    ("total_call_oi", ("PR_OI",)),
    ("total_put_oi", ("PR_OI",)),
    ("minutes_", ("PR_TIME",)),
]

_TOKEN_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("bid_ask", ("PR_BID", "PR_ASK")),
    ("open_interest", ("PR_OI",)),
    ("implied_vol", ("PR_IV",)),
    ("_pcr", ("PR_OI",)),
    ("_gex", ("PR_GAMMA",)),
    ("_oi", ("PR_OI",)),
    ("_volume", ("PR_VOLUME",)),
]


RuleFn = Callable[[dict[str, Any]], list[str] | None]


class PrimitiveMappingProvider:
    """Resolve legacy feature dict → list of primitive_ids."""

    def __init__(
        self,
        *,
        explicit_map: dict[str, tuple[str, ...]] | None = None,
        extra_rules: list[RuleFn] | None = None,
    ) -> None:
        self._explicit = dict(explicit_map or EXPLICIT_NAME_MAP)
        self._extra_rules = list(extra_rules or [])

    def resolve(self, legacy_feature: dict[str, Any]) -> list[str]:
        name = str(legacy_feature.get("name") or "").strip()
        if not name:
            return []

        # 1. Explicit map
        if name in self._explicit:
            return list(self._explicit[name])

        # 2. Rule engine
        for rule in self._extra_rules:
            hit = rule(legacy_feature)
            if hit:
                return list(hit)

        for prefix, ids in _PREFIX_RULES:
            if name.startswith(prefix):
                return list(ids)

        for token, ids in _TOKEN_RULES:
            if token in name:
                return list(ids)

        # Domain hint (conservative — only when domain clearly maps to a seed PR)
        domain = str(legacy_feature.get("primary_domain") or "").lower()
        if domain == "greeks":
            for g, pid in (
                ("delta", "PR_DELTA"),
                ("gamma", "PR_GAMMA"),
                ("theta", "PR_THETA"),
                ("vega", "PR_VEGA"),
                ("rho", "PR_RHO"),
                ("charm", "PR_DELTA"),
                ("vanna", "PR_DELTA"),
                ("volga", "PR_VEGA"),
            ):
                if g in name:
                    return [pid]
        if domain == "implied_volatility":
            return ["PR_IV"]
        if domain == "open_interest":
            return ["PR_OI"]
        if domain == "time_session":
            return ["PR_TIME"]
        if domain == "volume_liquidity":
            if "bid" in name and "ask" in name:
                return ["PR_BID", "PR_ASK"]
            if "bid" in name:
                return ["PR_BID"]
            if "ask" in name:
                return ["PR_ASK"]
            return ["PR_VOLUME"]
        if domain == "spot_futures":
            if "bid" in name:
                return ["PR_BID"]
            if "ask" in name:
                return ["PR_ASK"]
            if name.endswith("_oi") or "_oi" in name:
                return ["PR_OI"]
            if "volume" in name:
                return ["PR_VOLUME"]
            return ["PR_SPOT"]

        # 3. Fallback — empty (synchronizer reports UNMAPPED_PRIMITIVES)
        return []
