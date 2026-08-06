"""Spot high/low/close EMA controllers — session-continuous spot.hl subgroup.

Gap policy (frozen for migration):
- Same as Spot EMA: effective session start at first market data; session-continuous.
- Per-token row gaps do NOT reset spot.hl controllers.
- Emit NULL until each EmaController reaches its warmup period (sample == period).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .rolling_controllers import ControllerSample, EmaController

SPOT_HL_PERIODS: tuple[int, ...] = (20, 50, 100, 200, 300)
SPOT_HL_CLOSE_PERIODS: tuple[int, ...] = (20, 50, 200, 300)


def hl_bar_bounds(origin_ts: float, ts: float, step_sec: float) -> tuple[float, float]:
    """Grid bar (start, end] bounds aligned with ``TickTimeline.high_low_rupees_series_on_grid``."""
    step = max(float(step_sec), 1.0)
    origin = float(origin_ts)
    idx = int((float(ts) - origin) / step)
    idx = max(0, idx)
    bar_end = origin + idx * step
    bar_start = origin - step if idx == 0 else origin + (idx - 1) * step
    return float(bar_start), float(bar_end)


@dataclass
class SpotHlSideControllers:
    """EMA stack for one spot HL side (high or low) at periods 20–300."""

    ema20: EmaController = field(default_factory=lambda: EmaController(20))
    ema50: EmaController = field(default_factory=lambda: EmaController(50))
    ema100: EmaController = field(default_factory=lambda: EmaController(100))
    ema200: EmaController = field(default_factory=lambda: EmaController(200))
    ema300: EmaController = field(default_factory=lambda: EmaController(300))

    def controller(self, period: int) -> EmaController:
        attr = f"ema{int(period)}"
        if not hasattr(self, attr):
            raise KeyError(f"unsupported spot.hl period: {period}")
        return getattr(self, attr)

    def update(self, price: float, *, ts: float) -> None:
        sample = ControllerSample.ltp(float(price), ts=ts)
        for period in SPOT_HL_PERIODS:
            self.controller(period).update(sample)

    def reset(self, ts: float | None = None) -> None:
        for period in SPOT_HL_PERIODS:
            self.controller(period).reset(ts)


@dataclass
class SpotHlCloseControllers:
    """EMA stack on grid close (spot LTP) — periods 20/50/200/300 only (no 100)."""

    ema20: EmaController = field(default_factory=lambda: EmaController(20))
    ema50: EmaController = field(default_factory=lambda: EmaController(50))
    ema200: EmaController = field(default_factory=lambda: EmaController(200))
    ema300: EmaController = field(default_factory=lambda: EmaController(300))

    def controller(self, period: int) -> EmaController:
        attr = f"ema{int(period)}"
        if not hasattr(self, attr):
            raise KeyError(f"unsupported spot.hl close period: {period}")
        return getattr(self, attr)

    def update(self, price: float, *, ts: float) -> None:
        sample = ControllerSample.ltp(float(price), ts=ts)
        for period in SPOT_HL_CLOSE_PERIODS:
            self.controller(period).update(sample)

    def reset(self, ts: float | None = None) -> None:
        for period in SPOT_HL_CLOSE_PERIODS:
            self.controller(period).reset(ts)


@dataclass
class SpotHlControllers:
    """Session-wide spot.hl controller subgroup under ``SpotControllers``."""

    high: SpotHlSideControllers = field(default_factory=SpotHlSideControllers)
    low: SpotHlSideControllers = field(default_factory=SpotHlSideControllers)
    close: SpotHlCloseControllers = field(default_factory=SpotHlCloseControllers)
    grid_step_sec: float = 3.0
    _last_stream_ts: float | None = field(default=None, repr=False)

    def reset(self, ts: float | None = None) -> None:
        self.high.reset(ts)
        self.low.reset(ts)
        self.close.reset(ts)
        self._last_stream_ts = None

    def update_bar(
        self,
        *,
        index_tl: Any,
        ts: float,
        close: float,
        grid_step_sec: float,
        grid_origin_ts: float,
    ) -> None:
        """Ingest one feature-grid bar — at most once per timestamp."""
        if close <= 0:
            return
        ts_f = float(ts)
        if self._last_stream_ts is not None and ts_f == float(self._last_stream_ts):
            return
        step = max(float(grid_step_sec), 1.0)
        self.grid_step_sec = step
        bar_start, bar_end = hl_bar_bounds(grid_origin_ts, ts_f, step)
        high, low = index_tl._range_rupees_between(bar_start, bar_end)
        if high <= 0 or low <= 0:
            return
        self.high.update(high, ts=ts_f)
        self.low.update(low, ts=ts_f)
        self.close.update(float(close), ts=ts_f)
        self._last_stream_ts = ts_f
