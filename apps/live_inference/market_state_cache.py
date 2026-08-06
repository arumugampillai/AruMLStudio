"""Replay market-state session cache — one LiveMarketState per replay day/expiry."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from .market_state import LiveMarketState

_live_market_state_cache: dict[tuple[Any, ...], LiveMarketState] = {}
_scoring_expiry_cache: dict[tuple[Any, ...], dict[str, Any]] = {}


@dataclass
class MarketStateAcquireMeta:
    cache_hit: bool = False
    resolve_expiry_ms: float = 0.0
    day_context_load_ms: float = 0.0
    wrap_ms: float = 0.0
    total_ms: float = 0.0
    cache_key: str = ""
    resolved_expiry: str | None = None
    day_context_cache_hit: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "cache_hit": self.cache_hit,
            "resolve_expiry_ms": round(self.resolve_expiry_ms, 3),
            "day_context_load_ms": round(self.day_context_load_ms, 3),
            "wrap_ms": round(self.wrap_ms, 3),
            "total_ms": round(self.total_ms, 3),
            "cache_key": self.cache_key,
            "resolved_expiry": self.resolved_expiry,
            "day_context_cache_hit": self.day_context_cache_hit,
        }


def _config_fingerprint(replay_config: dict[str, Any]) -> str:
    return str(replay_config.get("market") or "NIFTY")


def market_state_cache_key(
    data_dir: str,
    *,
    date_str: str,
    expiry_hint: str | None,
    underlying: str,
    replay_config: dict[str, Any],
    resolved_expiry: str | None = None,
) -> tuple[Any, ...]:
    exp = str(resolved_expiry or expiry_hint or "").strip()
    return (
        str(data_dir),
        str(date_str),
        str(underlying or "NIFTY"),
        exp,
        _config_fingerprint(replay_config),
    )


def cached_scoring_expiry(
    chart_dir: str,
    date_str: str,
    expiry_hint: str | None,
    *,
    underlying: str = "NIFTY",
) -> dict[str, Any]:
    from chain_replay_ml.replay_feature_scoring import resolve_scoring_expiry

    key = (chart_dir, date_str, str(expiry_hint or ""), str(underlying))
    hit = _scoring_expiry_cache.get(key)
    if hit is not None:
        return hit
    resolved = resolve_scoring_expiry(chart_dir, date_str, expiry_hint, underlying=underlying)
    _scoring_expiry_cache[key] = resolved
    return resolved


def clear_market_state_caches() -> None:
    _live_market_state_cache.clear()
    _scoring_expiry_cache.clear()
    from .tick_source import clear_day_context_cache

    clear_day_context_cache()


def market_state_cache_stats() -> dict[str, Any]:
    from .tick_source import day_context_cache_stats

    return {
        "live_market_states": len(_live_market_state_cache),
        "scoring_expiry_entries": len(_scoring_expiry_cache),
        **day_context_cache_stats(),
    }


def acquire_replay_market_state(
    data_dir: str,
    *,
    date_str: str,
    expiry_hint: str | None,
    replay_config: dict[str, Any],
    underlying: str = "NIFTY",
) -> tuple[LiveMarketState, MarketStateAcquireMeta]:
    """Return cached LiveMarketState for replay session or load once."""
    from chain_replay_ml.replay_feature_scoring import chart_dir_from_data_dir
    from .tick_source import day_context_cache_stats, load_replay_tick_source

    t_total0 = time.perf_counter()
    meta = MarketStateAcquireMeta()
    chart_dir = chart_dir_from_data_dir(data_dir)

    hinted = str(expiry_hint or "").strip()
    if hinted:
        key = market_state_cache_key(
            data_dir,
            date_str=date_str,
            expiry_hint=expiry_hint,
            underlying=underlying,
            replay_config=replay_config,
            resolved_expiry=hinted,
        )
        cached_state = _live_market_state_cache.get(key)
        if cached_state is not None:
            meta.cache_hit = True
            meta.resolved_expiry = hinted
            meta.cache_key = "|".join(str(k) for k in key)
            meta.total_ms = round((time.perf_counter() - t_total0) * 1000.0, 3)
            return cached_state, meta

    t_res0 = time.perf_counter()
    expiry_resolution = cached_scoring_expiry(
        chart_dir, date_str, expiry_hint, underlying=underlying,
    )
    meta.resolve_expiry_ms = round((time.perf_counter() - t_res0) * 1000.0, 3)
    resolved_expiry = str(expiry_resolution.get("resolved_expiry") or expiry_hint or "").strip()
    meta.resolved_expiry = resolved_expiry or None

    key = market_state_cache_key(
        data_dir,
        date_str=date_str,
        expiry_hint=expiry_hint,
        underlying=underlying,
        replay_config=replay_config,
        resolved_expiry=resolved_expiry,
    )
    meta.cache_key = "|".join(str(k) for k in key)

    cached_state = _live_market_state_cache.get(key)
    if cached_state is not None:
        meta.cache_hit = True
        meta.total_ms = round((time.perf_counter() - t_total0) * 1000.0, 3)
        return cached_state, meta

    dc_before = day_context_cache_stats().get("cached_contexts", 0)
    t_load0 = time.perf_counter()
    source = load_replay_tick_source(
        data_dir,
        date_str=date_str,
        expiry_hint=resolved_expiry or expiry_hint,
        replay_config=replay_config,
        underlying=underlying,
        expiry_resolution=expiry_resolution,
    )
    state = LiveMarketState.from_tick_source(source)
    meta.day_context_load_ms = round((time.perf_counter() - t_load0) * 1000.0, 3)
    dc_after = day_context_cache_stats().get("cached_contexts", 0)
    meta.day_context_cache_hit = dc_after <= dc_before and dc_before > 0

    t_wrap0 = time.perf_counter()
    _live_market_state_cache[key] = state
    meta.wrap_ms = round((time.perf_counter() - t_wrap0) * 1000.0, 3)
    meta.total_ms = round((time.perf_counter() - t_total0) * 1000.0, 3)
    return state, meta
