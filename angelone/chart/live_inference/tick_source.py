"""Tick source adapters — Layer 1 input (replay SQLite today, live ring later)."""

from __future__ import annotations

import os
from typing import Any, Protocol

from chain_replay_ml.dataset_builder.day_context import DayContext, SourceSpec, load_day_context
from chain_replay_ml.replay_feature_scoring import (
    chart_dir_from_data_dir,
    resolve_replay_source_spec,
)


class TickSource(Protocol):
    """Abstract tick feed — replay bulk load or live incremental."""

    @property
    def source_kind(self) -> str: ...

    @property
    def trading_day(self) -> str: ...

    @property
    def expiry(self) -> str: ...

    @property
    def market(self) -> str: ...

    def day_context(self) -> DayContext: ...


_market_context_cache: dict[tuple[Any, ...], DayContext] = {}


def clear_day_context_cache() -> None:
    _market_context_cache.clear()


def day_context_cache_stats() -> dict[str, Any]:
    return {
        "cached_contexts": len(_market_context_cache),
        "context_keys": ["|".join(str(x) for x in k) for k in list(_market_context_cache.keys())[:8]],
    }


def _cache_key(chart_dir: str, source: SourceSpec) -> tuple[Any, ...]:
    return (
        os.path.abspath(chart_dir),
        str(source.source_id),
        str(source.trading_day),
        str(source.market),
        str(source.expiry),
    )


class ReplaySqliteTickSource:
    """Bulk-load replay ticks from angel_market SQLite (Layer 1 replay path)."""

    def __init__(
        self,
        *,
        chart_dir: str,
        source: SourceSpec,
        ctx: DayContext | None = None,
    ) -> None:
        self._chart_dir = chart_dir
        self._source = source
        self._ctx = ctx

    @classmethod
    def from_replay_request(
        cls,
        data_dir: str,
        *,
        date_str: str,
        expiry_hint: str | None,
        replay_config: dict[str, Any],
        underlying: str = "NIFTY",
    ) -> ReplaySqliteTickSource:
        return load_replay_tick_source(
            data_dir,
            date_str=date_str,
            expiry_hint=expiry_hint,
            replay_config=replay_config,
            underlying=underlying,
        )

    @property
    def source_kind(self) -> str:
        return "replay_sqlite"

    @property
    def trading_day(self) -> str:
        return str(self._source.trading_day)

    @property
    def expiry(self) -> str:
        return str(self._source.expiry)

    @property
    def market(self) -> str:
        return str(self._source.market)

    def day_context(self) -> DayContext:
        if self._ctx is None:
            key = _cache_key(self._chart_dir, self._source)
            ctx = _market_context_cache.get(key)
            if ctx is None:
                ctx = load_day_context(self._chart_dir, self._source)
                _market_context_cache[key] = ctx
            self._ctx = ctx
        return self._ctx


def load_replay_tick_source(
    data_dir: str,
    *,
    date_str: str,
    expiry_hint: str | None,
    replay_config: dict[str, Any],
    underlying: str = "NIFTY",
    expiry_resolution: dict[str, Any] | None = None,
) -> ReplaySqliteTickSource:
    """Load tick source; reuse cached DayContext when available."""
    chart_dir = chart_dir_from_data_dir(data_dir)
    market = str(replay_config.get("market") or underlying or "NIFTY")
    if expiry_resolution is None:
        from .market_state_cache import cached_scoring_expiry

        expiry_resolution = cached_scoring_expiry(chart_dir, date_str, expiry_hint, underlying=market)
    resolved_expiry = str(expiry_resolution.get("resolved_expiry") or expiry_hint or "").strip()
    source_info = (
        resolve_replay_source_spec(replay_config, date_str, resolved_expiry)
        if resolved_expiry
        else None
    )
    if not source_info:
        raise ValueError("could_not_resolve_replay_source")
    source = SourceSpec(
        source_id=source_info["source_id"],
        trading_day=source_info["trading_day"],
        market=source_info["market"],
        expiry=source_info["expiry"],
    )
    key = _cache_key(chart_dir, source)
    ctx = _market_context_cache.get(key)
    if ctx is None:
        ctx = load_day_context(chart_dir, source)
        _market_context_cache[key] = ctx
    return ReplaySqliteTickSource(chart_dir=chart_dir, source=source, ctx=ctx)
