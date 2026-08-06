"""Layer 1 — live market state (no ML)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from chain_replay_ml.dataset_builder.day_context import DayContext
from chain_replay_ml.dataset_builder.tick_coverage import clipped_grid_bounds
from chain_replay_ml.ticks import TickTimeline

from .tick_source import TickSource
from .versions import market_state_version


@dataclass
class LiveMarketState:
    """Read-only market state view — models must never touch this layer."""

    ctx: DayContext
    source_kind: str
    version: str = market_state_version()

    @classmethod
    def from_tick_source(cls, source: TickSource) -> LiveMarketState:
        return cls(ctx=source.day_context(), source_kind=source.source_kind)

    @classmethod
    def from_replay(
        cls,
        data_dir: str,
        *,
        date_str: str,
        expiry_hint: str | None,
        replay_config: dict[str, Any],
        underlying: str = "NIFTY",
    ) -> LiveMarketState:
        from .market_state_cache import acquire_replay_market_state

        state, _meta = acquire_replay_market_state(
            data_dir,
            date_str=date_str,
            expiry_hint=expiry_hint,
            replay_config=replay_config,
            underlying=underlying,
        )
        return state

    @property
    def trading_day(self) -> str:
        return str(self.ctx.source.trading_day)

    @property
    def expiry(self) -> str:
        return str(self.ctx.source.expiry)

    @property
    def market(self) -> str:
        return str(self.ctx.source.market)

    @property
    def index_timeline(self) -> TickTimeline:
        return self.ctx.index_tl

    @property
    def strike_mapping(self) -> dict[tuple[float, str], tuple[str, str, TickTimeline]]:
        return self.ctx.strike_mapping

    def grid_bounds(self, *, max_horizon_sec: int = 0) -> tuple[float, float] | None:
        return clipped_grid_bounds(self.ctx, max_horizon_sec=max_horizon_sec)

    def session_open_ts(self) -> float:
        return float(self.ctx.open_ts)

    def session_close_ts(self) -> float:
        return float(self.ctx.close_ts)
