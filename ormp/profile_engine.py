"""Incremental Open Relative Market Profile engine (no look-ahead).

All durations are stored in **seconds** (integer for candle intervals;
continuous splits use integer second allocation with remainder so the
sum is exact). Export converts to minutes when desired.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

PathMode = Literal["snapshot", "continuous"]
DEFAULT_CANDLE_INTERVAL_SEC = 60


@dataclass
class BandState:
    index: int
    lower_price: float
    upper_price: float
    total_time: float = 0.0  # seconds
    visit_count: int = 0
    current_visit_start: float | None = None
    current_visit_duration: float = 0.0  # seconds; ongoing visit
    last_visit_duration: float = 0.0  # seconds; most recently completed visit
    longest_visit_duration: float = 0.0  # seconds
    last_enter_timestamp: float | None = None

    @property
    def average_visit_duration(self) -> float | None:
        """Average stay in seconds (includes in-progress time in total_time)."""
        if self.visit_count <= 0:
            return None
        return float(self.total_time) / float(self.visit_count)


def entered_bands(previous_band: int, current_band: int) -> list[int]:
    """Bands entered after leaving previous, through current inclusive."""
    if previous_band == current_band:
        return []
    step = 1 if current_band > previous_band else -1
    return list(range(previous_band + step, current_band + step, step))


def split_duration_seconds(total_sec: int, n: int) -> list[int]:
    """Split total_sec across n bands; remainder distributed to the first bands."""
    if n <= 0:
        raise ValueError("n must be > 0")
    if total_sec <= 0:
        raise ValueError("total_sec must be > 0")
    base, rem = divmod(int(total_sec), int(n))
    return [base + (1 if i < rem else 0) for i in range(n)]


@dataclass
class OrmpProfile:
    """One trading day's ORMP state. Destroy and recreate each day."""

    spot_open: float
    band_size_pct: float
    band_size_points: float
    path_mode: PathMode = "snapshot"
    candle_interval_sec: int = DEFAULT_CANDLE_INTERVAL_SEC
    bands: dict[int, BandState] = field(default_factory=dict)
    current_band: int | None = None
    highest_band: int | None = None
    lowest_band: int | None = None
    total_band_transitions: int = 0
    upward_transitions: int = 0
    downward_transitions: int = 0
    return_to_open_count: int = 0
    time_above_open: float = 0.0  # seconds
    time_below_open: float = 0.0  # seconds
    candles_processed: int = 0
    # Most recent classified move (endpoints only; not intermediate continuous steps)
    last_from_band: int | None = None
    last_to_band: int | None = None
    last_move_duration_sec: float = 0.0

    @classmethod
    def create(
        cls,
        spot_open: float,
        band_size_pct: float,
        *,
        path_mode: PathMode = "snapshot",
        candle_interval_sec: int = DEFAULT_CANDLE_INTERVAL_SEC,
    ) -> OrmpProfile:
        if spot_open <= 0:
            raise ValueError("spot_open must be > 0")
        if band_size_pct <= 0:
            raise ValueError("band_size_pct must be > 0")
        if path_mode not in ("snapshot", "continuous"):
            raise ValueError(f"unsupported ormp_path_mode: {path_mode}")
        if int(candle_interval_sec) <= 0:
            raise ValueError("candle_interval_sec must be > 0")
        w_pts = float(spot_open) * float(band_size_pct) / 100.0
        if w_pts <= 0:
            raise ValueError("band_size_points must be > 0")
        return cls(
            spot_open=float(spot_open),
            band_size_pct=float(band_size_pct),
            band_size_points=w_pts,
            path_mode=path_mode,
            candle_interval_sec=int(candle_interval_sec),
        )

    def band_index_for_price(self, price: float) -> int:
        """Signed integer band via floor division (correct for negatives)."""
        distance = float(price) - self.spot_open
        return int(math.floor(distance / self.band_size_points))

    def band_bounds(self, index: int) -> tuple[float, float]:
        lower = self.spot_open + index * self.band_size_points
        upper = self.spot_open + (index + 1) * self.band_size_points
        return lower, upper

    def _ensure_band(self, index: int) -> BandState:
        band = self.bands.get(index)
        if band is None:
            lower, upper = self.band_bounds(index)
            band = BandState(index=index, lower_price=lower, upper_price=upper)
            self.bands[index] = band
        return band

    def _touch_extrema(self, idx: int) -> None:
        if self.highest_band is None or idx > self.highest_band:
            self.highest_band = idx
        if self.lowest_band is None or idx < self.lowest_band:
            self.lowest_band = idx

    def _add_location_time(self, idx: int, duration_sec: float) -> None:
        if idx > 0:
            self.time_above_open += duration_sec
        elif idx < 0:
            self.time_below_open += duration_sec

    def _finish_visit(self, band: BandState) -> None:
        """Close the ongoing visit → becomes last_visit_duration."""
        band.last_visit_duration = band.current_visit_duration
        if band.current_visit_duration > band.longest_visit_duration:
            band.longest_visit_duration = band.current_visit_duration
        band.current_visit_duration = 0.0
        band.current_visit_start = None

    def _start_visit(self, band: BandState, timestamp: float, duration_sec: float) -> None:
        """Open a new visit; current_visit_duration starts at duration_sec."""
        band.visit_count += 1
        band.current_visit_start = timestamp
        band.current_visit_duration = float(duration_sec)
        band.total_time += float(duration_sec)
        band.last_enter_timestamp = timestamp
        if band.current_visit_duration > band.longest_visit_duration:
            band.longest_visit_duration = band.current_visit_duration

    def _record_step_transition(self, from_band: int, to_band: int) -> None:
        self.total_band_transitions += 1
        if to_band > from_band:
            self.upward_transitions += 1
        elif to_band < from_band:
            self.downward_transitions += 1
        if to_band == 0 and from_band != 0:
            self.return_to_open_count += 1

    def _record_last_move(
        self,
        from_band: int | None,
        to_band: int,
        duration_sec: float,
    ) -> None:
        """Record classified previous→current endpoints for export features."""
        self.last_from_band = from_band
        self.last_to_band = int(to_band)
        self.last_move_duration_sec = float(duration_sec)

    def last_transition_distance(self) -> int:
        if self.last_from_band is None or self.last_to_band is None:
            return 0
        return abs(int(self.last_to_band) - int(self.last_from_band))

    def last_transition_direction(self) -> int:
        if self.last_from_band is None or self.last_to_band is None:
            return 0
        delta = int(self.last_to_band) - int(self.last_from_band)
        if delta > 0:
            return 1
        if delta < 0:
            return -1
        return 0

    def update(
        self,
        price: float,
        timestamp: float,
        *,
        duration_sec: float | None = None,
        minutes: float | None = None,
    ) -> None:
        """Ingest one candle assignment price. Durations in seconds. No look-ahead.

        ``minutes`` is accepted as a deprecated alias (converted to seconds).
        """
        if duration_sec is None and minutes is None:
            duration_sec = float(self.candle_interval_sec)
        elif duration_sec is None:
            duration_sec = float(minutes) * 60.0
        if duration_sec <= 0:
            raise ValueError("duration_sec must be > 0")

        idx = self.band_index_for_price(price)
        from_band = self.current_band

        if self.current_band is None:
            band = self._ensure_band(idx)
            band.visit_count = 1
            band.current_visit_start = timestamp
            band.current_visit_duration = float(duration_sec)
            band.total_time = float(duration_sec)
            band.last_enter_timestamp = timestamp
            band.longest_visit_duration = float(duration_sec)
            self.current_band = idx
            self.highest_band = idx
            self.lowest_band = idx
            self._add_location_time(idx, float(duration_sec))
            self._record_last_move(None, idx, float(duration_sec))
            self.candles_processed += 1
            return

        if idx == self.current_band:
            band = self.bands[idx]
            band.current_visit_duration += float(duration_sec)
            band.total_time += float(duration_sec)
            if band.current_visit_duration > band.longest_visit_duration:
                band.longest_visit_duration = band.current_visit_duration
            self._add_location_time(idx, float(duration_sec))
            self._record_last_move(from_band, idx, float(duration_sec))
            self.candles_processed += 1
            return

        if self.path_mode == "continuous":
            self._update_continuous(idx, timestamp, float(duration_sec))
        else:
            self._update_snapshot(idx, timestamp, float(duration_sec))
        self._record_last_move(from_band, idx, float(duration_sec))
        self.candles_processed += 1

    def _update_snapshot(self, idx: int, timestamp: float, duration_sec: float) -> None:
        assert self.current_band is not None
        prev = self.bands[self.current_band]
        self._finish_visit(prev)
        self._record_step_transition(self.current_band, idx)

        band = self._ensure_band(idx)
        self._start_visit(band, timestamp, duration_sec)
        self.current_band = idx
        self._touch_extrema(idx)
        self._add_location_time(idx, duration_sec)

    def _update_continuous(self, idx: int, timestamp: float, duration_sec: float) -> None:
        """Minimum continuous path; split candle seconds across entered bands.

        Intermediate entered bands finalize immediately
        (``last_visit_duration`` = slice). The destination band keeps an
        open ``current_visit_duration`` until a later transition leaves it.
        """
        assert self.current_band is not None
        prev_idx = self.current_band
        path = entered_bands(prev_idx, idx)
        n = abs(idx - prev_idx)
        if n == 0 or len(path) != n:
            raise RuntimeError(
                f"continuous path error: prev={prev_idx} curr={idx} path={path}"
            )

        total_sec = int(round(float(duration_sec)))
        slices = split_duration_seconds(total_sec, n)

        # Leave the starting band without a new entry credit.
        self._finish_visit(self.bands[prev_idx])

        from_b = prev_idx
        for i, to_b in enumerate(path):
            slice_sec = float(slices[i])
            self._record_step_transition(from_b, to_b)
            band = self._ensure_band(to_b)
            self._start_visit(band, timestamp, slice_sec)
            self._touch_extrema(to_b)
            self._add_location_time(to_b, slice_sec)
            # Pass-through bands complete immediately; destination stays open.
            if i < len(path) - 1:
                self._finish_visit(band)
            from_b = to_b

        self.current_band = idx

    def total_band_time(self) -> float:
        """Total band time in seconds."""
        return float(sum(b.total_time for b in self.bands.values()))

    def unique_band_count(self) -> int:
        return len(self.bands)

    def validate_time_accounting(self) -> dict[str, Any]:
        """Require sum(band times sec) == candles_processed * candle_interval_sec."""
        total = self.total_band_time()
        expected = float(self.candles_processed * self.candle_interval_sec)
        ok = abs(total - expected) < 1e-9
        return {
            "ok": ok,
            "total_band_time_sec": total,
            "trading_seconds": expected,
            "trading_minutes": expected / 60.0,
            # Back-compat keys used by dataset builder / reports
            "total_band_time": expected / 60.0 if ok else total / 60.0,
            "delta": total - expected,
        }
