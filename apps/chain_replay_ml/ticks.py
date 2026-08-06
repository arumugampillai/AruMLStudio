"""Tick timeline index for point-in-time LTP lookup."""

from __future__ import annotations

import bisect
import json
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Sequence

EMA_BAR_INTERVAL_SEC = 10
BOOK_LEVELS = 5
_EMPTY_BOOK: tuple[int, ...] = (0,) * BOOK_LEVELS


@dataclass(frozen=True)
class BookSnapshot:
    """L1–L5 book at a point in time (prices in paise, quantities in lots)."""

    bid_prices_paise: tuple[int, ...]
    ask_prices_paise: tuple[int, ...]
    bid_quantities: tuple[int, ...]
    ask_quantities: tuple[int, ...]
    spread_paise: int = 0

    @property
    def has_l1(self) -> bool:
        return bool(self.bid_prices_paise and self.ask_prices_paise
                    and self.bid_prices_paise[0] > 0 and self.ask_prices_paise[0] > 0)


def _pad_levels(values: Sequence[int] | None, *, levels: int = BOOK_LEVELS) -> tuple[int, ...]:
    out: list[int] = []
    if values:
        for v in list(values)[:levels]:
            try:
                out.append(int(v) if v is not None else 0)
            except (TypeError, ValueError):
                out.append(0)
    while len(out) < levels:
        out.append(0)
    return tuple(out)


def _parse_book_json(raw: Any, *, levels: int = BOOK_LEVELS) -> tuple[int, ...]:
    if raw is None or raw == "":
        return _EMPTY_BOOK
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
    except (TypeError, ValueError, json.JSONDecodeError):
        return _EMPTY_BOOK
    if not isinstance(data, list):
        return _EMPTY_BOOK
    out: list[int] = []
    for item in data[:levels]:
        if item is None or item == "":
            out.append(0)
            continue
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            out.append(0)
    while len(out) < levels:
        out.append(0)
    return tuple(out)


@dataclass
class TickTimeline:
    timestamps: list[float] = field(default_factory=list)
    ltps_paise: list[int] = field(default_factory=list)
    volumes: list[int] = field(default_factory=list)
    oi_list: list[int] = field(default_factory=list)
    spreads_paise: list[int] = field(default_factory=list)
    # Exchange day average traded price (session VWAP equivalent), paise.
    atps_paise: list[int] = field(default_factory=list)
    ltqs: list[int] = field(default_factory=list)
    total_buys: list[int] = field(default_factory=list)
    total_sells: list[int] = field(default_factory=list)
    bid_prices_paise: list[tuple[int, ...]] = field(default_factory=list)
    ask_prices_paise: list[tuple[int, ...]] = field(default_factory=list)
    bid_quantities: list[tuple[int, ...]] = field(default_factory=list)
    ask_quantities: list[tuple[int, ...]] = field(default_factory=list)
    # Cached path-outcome arrays (built once per day / timeline).
    _path_ts: Any = field(default=None, init=False, repr=False, compare=False)
    _path_ltp: Any = field(default=None, init=False, repr=False, compare=False)

    def append(
        self,
        ts: float,
        ltp_paise: int,
        volume: int = 0,
        oi: int = 0,
        spread_paise: int = 0,
        *,
        atp_paise: int = 0,
        ltq: int = 0,
        total_buy: int = 0,
        total_sell: int = 0,
        bid_prices_paise: Sequence[int] | None = None,
        ask_prices_paise: Sequence[int] | None = None,
        bid_quantities: Sequence[int] | None = None,
        ask_quantities: Sequence[int] | None = None,
    ) -> None:
        if self.timestamps and ts < self.timestamps[-1]:
            return
        self.timestamps.append(ts)
        self.ltps_paise.append(ltp_paise)
        self.volumes.append(volume)
        self.oi_list.append(oi)
        self.spreads_paise.append(spread_paise)
        self.atps_paise.append(int(atp_paise or 0))
        self.ltqs.append(int(ltq or 0))
        self.total_buys.append(int(total_buy or 0))
        self.total_sells.append(int(total_sell or 0))
        self.bid_prices_paise.append(_pad_levels(bid_prices_paise))
        self.ask_prices_paise.append(_pad_levels(ask_prices_paise))
        self.bid_quantities.append(_pad_levels(bid_quantities))
        self.ask_quantities.append(_pad_levels(ask_quantities))
        self._path_ts = None
        self._path_ltp = None

    def invalidate_path_arrays(self) -> None:
        self._path_ts = None
        self._path_ltp = None

    def ensure_path_arrays(self) -> tuple[Any, Any]:
        """
        Return (timestamps_np, ltps_rupees_np) for path-outcome hot loops.

        Built once; reused for every prediction on this timeline.
        """
        import numpy as np

        n = len(self.timestamps)
        if (
            self._path_ts is not None
            and self._path_ltp is not None
            and int(getattr(self._path_ts, "shape", [0])[0]) == n
            and int(getattr(self._path_ltp, "shape", [0])[0]) == n
        ):
            return self._path_ts, self._path_ltp
        self._path_ts = np.asarray(self.timestamps, dtype=np.float64)
        # Single vectorized paise → rupee conversion for the whole day/token.
        self._path_ltp = np.asarray(self.ltps_paise, dtype=np.float64) * 0.01
        return self._path_ts, self._path_ltp

    def ltp_paise_at(self, target_ts: float) -> int | None:
        if not self.timestamps:
            return None
        idx = bisect.bisect_right(self.timestamps, target_ts) - 1
        if idx < 0:
            return None
        return self.ltps_paise[idx]

    def ltp_rupees_at(self, target_ts: float) -> float | None:
        paise = self.ltp_paise_at(target_ts)
        return None if paise is None or paise <= 0 else paise / 100.0

    def tick_age_sec_at(self, target_ts: float) -> float | None:
        """Seconds since the last tick at or before target_ts."""
        if not self.timestamps:
            return None
        idx = bisect.bisect_right(self.timestamps, target_ts) - 1
        if idx < 0:
            return None
        return float(target_ts - self.timestamps[idx])

    def is_fresh_at(self, target_ts: float, max_stale_sec: float = 10.0) -> bool:
        """True if last tick is within max_stale_sec and LTP is valid (Rule 2)."""
        age = self.tick_age_sec_at(target_ts)
        if age is None or age > max_stale_sec:
            return False
        paise = self.ltp_paise_at(target_ts)
        return paise is not None and paise > 0

    def volume_at(self, target_ts: float) -> int | None:
        if not self.timestamps:
            return None
        idx = bisect.bisect_right(self.timestamps, target_ts) - 1
        if idx < 0:
            return None
        return self.volumes[idx]

    def oi_at(self, target_ts: float) -> int | None:
        if not self.timestamps:
            return None
        idx = bisect.bisect_right(self.timestamps, target_ts) - 1
        if idx < 0:
            return None
        return self.oi_list[idx]

    def _scalar_at(self, values: list[int], target_ts: float, *, positive_only: bool = False) -> int | None:
        if not self.timestamps:
            return None
        idx = bisect.bisect_right(self.timestamps, target_ts) - 1
        if idx < 0:
            return None
        if idx >= len(values):
            return None
        v = int(values[idx] or 0)
        if positive_only and v <= 0:
            return None
        return v

    def ltq_at(self, target_ts: float) -> int | None:
        return self._scalar_at(self.ltqs, target_ts, positive_only=True)

    def total_buy_at(self, target_ts: float) -> int | None:
        return self._scalar_at(self.total_buys, target_ts, positive_only=True)

    def total_sell_at(self, target_ts: float) -> int | None:
        return self._scalar_at(self.total_sells, target_ts, positive_only=True)

    def spread_paise_at(self, target_ts: float) -> int | None:
        if not self.timestamps:
            return None
        idx = bisect.bisect_right(self.timestamps, target_ts) - 1
        if idx < 0:
            return None
        return self.spreads_paise[idx]

    def atp_paise_at(self, target_ts: float) -> int | None:
        """As-of exchange average traded price (paise); None if missing/zero."""
        if not self.timestamps:
            return None
        idx = bisect.bisect_right(self.timestamps, target_ts) - 1
        if idx < 0:
            return None
        # Back-compat for timelines built before atp arrays existed.
        if idx >= len(self.atps_paise):
            return None
        paise = int(self.atps_paise[idx] or 0)
        return paise if paise > 0 else None

    def atp_rupees_at(self, target_ts: float) -> float | None:
        paise = self.atp_paise_at(target_ts)
        return None if paise is None else paise / 100.0

    def book_at(self, target_ts: float) -> BookSnapshot | None:
        """As-of L1–L5 book snapshot (backward-only)."""
        if not self.timestamps:
            return None
        idx = bisect.bisect_right(self.timestamps, target_ts) - 1
        if idx < 0:
            return None
        # Back-compat for timelines built before book arrays existed.
        if idx >= len(self.bid_prices_paise):
            return BookSnapshot(
                bid_prices_paise=_EMPTY_BOOK,
                ask_prices_paise=_EMPTY_BOOK,
                bid_quantities=_EMPTY_BOOK,
                ask_quantities=_EMPTY_BOOK,
                spread_paise=int(self.spreads_paise[idx] or 0),
            )
        return BookSnapshot(
            bid_prices_paise=self.bid_prices_paise[idx],
            ask_prices_paise=self.ask_prices_paise[idx],
            bid_quantities=self.bid_quantities[idx],
            ask_quantities=self.ask_quantities[idx],
            spread_paise=int(self.spreads_paise[idx] or 0),
        )

    def minute_ltp_rupees_series(self, open_ts: float, close_ts: float) -> list[float]:
        """LTP at each 1-minute grid point (forward-filled) — fast path for EMA precompute."""
        grid = minute_grid(open_ts, close_ts)
        if not grid:
            return []
        if not self.timestamps:
            return [0.0] * len(grid)
        prices: list[float] = []
        last = float(self.ltps_paise[0] / 100.0) if self.ltps_paise else 0.0
        for m in grid:
            p = self.ltp_rupees_at(float(m))
            if p is not None and p > 0:
                last = float(p)
            prices.append(last)
        return prices

    def ltp_rupees_series_on_grid(
        self,
        open_ts: float,
        close_ts: float,
        step_sec: float = EMA_BAR_INTERVAL_SEC,
    ) -> list[float]:
        """LTP forward-filled on a fixed step grid (default 10s) — used for EMA features."""
        grid = uniform_grid(open_ts, close_ts, step_sec)
        if not grid:
            return []
        if not self.timestamps:
            return [0.0] * len(grid)
        try:
            import numpy as np

            ts_arr = np.asarray(self.timestamps, dtype=float)
            paise_arr = np.asarray(self.ltps_paise, dtype=float)
            grid_arr = np.asarray(grid, dtype=float)
            idx = np.searchsorted(ts_arr, grid_arr, side="right") - 1
            prices = np.zeros(len(grid_arr), dtype=float)
            valid = idx >= 0
            if valid.any():
                prices[valid] = paise_arr[idx[valid]] / 100.0
            seed = float(paise_arr[0] / 100.0) if len(paise_arr) else 0.0
            for i in range(len(prices)):
                if prices[i] > 0:
                    seed = float(prices[i])
                else:
                    prices[i] = seed
            return prices.tolist()
        except Exception:
            prices: list[float] = []
            last = float(self.ltps_paise[0] / 100.0) if self.ltps_paise else 0.0
            for pt in grid:
                p = self.ltp_rupees_at(float(pt))
                if p is not None and p > 0:
                    last = float(p)
                prices.append(last)
            return prices

    def last_tick_ts_series_on_grid(
        self,
        open_ts: float,
        close_ts: float,
        step_sec: float = EMA_BAR_INTERVAL_SEC,
    ) -> list[float]:
        """Last tick timestamp at or before each grid point (for gap-aware rolling features)."""
        from chain_replay_ml.dataset_builder.gap_policy_instrumentation import gap_policy_profile_block

        with gap_policy_profile_block("TickTimeline.last_tick_ts_series_on_grid"):
            grid = uniform_grid(open_ts, close_ts, step_sec)
            if not grid or not self.timestamps:
                return []
            try:
                import numpy as np

                ts_arr = np.asarray(self.timestamps, dtype=float)
                grid_arr = np.asarray(grid, dtype=float)
                idx = np.searchsorted(ts_arr, grid_arr, side="right") - 1
                out = np.empty(len(grid_arr), dtype=float)
                valid = idx >= 0
                out[valid] = ts_arr[idx[valid]]
                out[~valid] = grid_arr[~valid]
                return out.tolist()
            except Exception:
                out: list[float] = []
                for pt in grid:
                    idx = bisect.bisect_right(self.timestamps, float(pt)) - 1
                    if idx < 0:
                        out.append(float(pt))
                    else:
                        out.append(float(self.timestamps[idx]))
                return out

    def _range_rupees_between(self, start_ts: float, end_ts: float) -> tuple[float, float]:
        """High/low rupees for ticks in (start_ts, end_ts]; falls back to LTP at end_ts."""
        close = self.ltp_rupees_at(end_ts)
        fallback = float(close) if close is not None and close > 0 else 0.0
        if not self.timestamps:
            return fallback, fallback
        left = bisect.bisect_right(self.timestamps, start_ts)
        right = bisect.bisect_right(self.timestamps, end_ts)
        if left >= right:
            return fallback, fallback
        segment = self.ltps_paise[left:right]
        return float(max(segment) / 100.0), float(min(segment) / 100.0)

    def high_low_rupees_series_on_grid(
        self,
        open_ts: float,
        close_ts: float,
        step_sec: float = EMA_BAR_INTERVAL_SEC,
    ) -> tuple[list[float], list[float]]:
        """Per-grid-bar spot high/low from ticks in each step interval."""
        grid = uniform_grid(open_ts, close_ts, step_sec)
        if not grid:
            return [], []
        step = max(float(step_sec), 1.0)
        highs: list[float] = []
        lows: list[float] = []
        for i, pt in enumerate(grid):
            start = float(pt) - step if i == 0 else float(grid[i - 1])
            high, low = self._range_rupees_between(start, float(pt))
            highs.append(high)
            lows.append(low)
        return highs, lows

    def analyze_future_trajectory(
        self,
        start_ts: float,
        duration_sec: float,
    ) -> dict[str, object]:
        if not self.timestamps:
            return {}

        baseline = self.ltp_paise_at(start_ts)
        if baseline is None or baseline <= 0:
            return {}

        start_idx = bisect.bisect_left(self.timestamps, start_ts)
        end_idx = bisect.bisect_right(self.timestamps, start_ts + duration_sec)

        high_paise = baseline
        low_paise = baseline
        high_ts = start_ts
        low_ts = start_ts

        if start_idx < end_idx:
            for idx in range(start_idx, end_idx):
                ts = self.timestamps[idx]
                ltp = self.ltps_paise[idx]
                if ltp > high_paise:
                    high_paise = ltp
                    high_ts = ts
                if ltp < low_paise:
                    low_paise = ltp
                    low_ts = ts

        return {
            "future_high_paise": high_paise,
            "future_low_paise": low_paise,
            "time_to_high_sec": max(0.0, high_ts - start_ts),
            "time_to_low_sec": max(0.0, low_ts - start_ts),
            "high_first": 1 if high_ts < low_ts else 0,
            "low_first": 1 if low_ts < high_ts else 0,
            "baseline_paise": baseline,
        }

    def check_scalp_outcome_seconds(
        self,
        start_ts: float,
        seconds: float,
        up_pct: float,
        down_pct: float,
    ) -> int:
        if not self.timestamps:
            return 0

        baseline = self.ltp_paise_at(start_ts)
        if baseline is None or baseline <= 0:
            return 0

        entry_idx = bisect.bisect_right(self.timestamps, start_ts) - 1
        end_idx = bisect.bisect_right(self.timestamps, start_ts + seconds)

        up_threshold = baseline * (1.0 + up_pct / 100.0)
        down_threshold = baseline * (1.0 - down_pct / 100.0)

        for idx in range(entry_idx + 1, end_idx):
            ltp = self.ltps_paise[idx]
            if ltp >= up_threshold:
                return 1
            if ltp <= down_threshold:
                return -1

        return 0


def load_tick_timelines(
    conn: sqlite3.Connection,
    tokens: list[str],
    open_ts: float,
    close_ts: float,
) -> dict[str, TickTimeline]:
    if not tokens:
        return {}

    cols = {row[1] for row in conn.execute("PRAGMA table_info(ticks)")}
    has_v1 = "bid_prices" in cols
    has_qty = "bid_quantities" in cols and "ask_quantities" in cols
    has_atp = "atp" in cols
    has_ltq = "ltq" in cols
    has_total_buy = "total_buy" in cols
    has_total_sell = "total_sell" in cols
    has_legacy_book = (not has_v1) and "best_bid" in cols and "best_ask" in cols

    select_cols = ["token", "ts", "ltp", "day_volume", "oi"]
    if has_atp:
        select_cols.append("atp")
    if has_ltq:
        select_cols.append("ltq")
    if has_total_buy:
        select_cols.append("total_buy")
    if has_total_sell:
        select_cols.append("total_sell")
    if has_v1:
        select_cols.extend(["bid_prices", "ask_prices"])
        if has_qty:
            select_cols.extend(["bid_quantities", "ask_quantities"])
    elif has_legacy_book:
        select_cols.extend(["best_bid", "best_ask"])

    placeholders = ",".join("?" for _ in tokens)
    order = "ORDER BY ts ASC"
    if "sequence_number" in cols:
        order = "ORDER BY ts ASC, sequence_number ASC"
    sql = f"""
        SELECT {", ".join(select_cols)}
        FROM ticks
        WHERE token IN ({placeholders})
          AND ts >= ?
          AND ts <= ?
          AND ltp IS NOT NULL AND ltp > 0
        {order}
    """
    col_index = {name: i for i, name in enumerate(select_cols)}

    def _cell(row: Any, name: str) -> Any:
        idx = col_index.get(name)
        return None if idx is None else row[idx]

    params: list[object] = [*tokens, open_ts, close_ts]
    out: dict[str, TickTimeline] = {tok: TickTimeline() for tok in tokens}

    for row in conn.execute(sql, params):
        tok = str(row[0])
        if tok not in out:
            continue
        try:
            ts = float(row[1])
            ltp_paise = int(row[2])
            volume = int(row[3] or 0)
            oi = int(row[4] or 0)

            atp_paise = 0
            ltq = 0
            total_buy = 0
            total_sell = 0
            spread_paise = 0
            bid_px = _EMPTY_BOOK
            ask_px = _EMPTY_BOOK
            bid_qty = _EMPTY_BOOK
            ask_qty = _EMPTY_BOOK

            if has_atp:
                atp_raw = _cell(row, "atp")
                if atp_raw is not None:
                    if has_v1:
                        atp_paise = int(atp_raw or 0)
                    else:
                        atp_paise = int(round(float(atp_raw) * 100))
            if has_ltq:
                ltq = int(_cell(row, "ltq") or 0)
            if has_total_buy:
                total_buy = int(_cell(row, "total_buy") or 0)
            if has_total_sell:
                total_sell = int(_cell(row, "total_sell") or 0)

            if has_v1:
                bid_px = _parse_book_json(_cell(row, "bid_prices"))
                ask_px = _parse_book_json(_cell(row, "ask_prices"))
                if has_qty:
                    bid_qty = _parse_book_json(_cell(row, "bid_quantities"))
                    ask_qty = _parse_book_json(_cell(row, "ask_quantities"))
                if bid_px[0] > 0 and ask_px[0] > 0:
                    spread_paise = max(0, ask_px[0] - bid_px[0])
            elif has_legacy_book:
                best_bid_val = _cell(row, "best_bid")
                best_ask_val = _cell(row, "best_ask")
                if best_bid_val is not None and best_ask_val is not None:
                    best_bid_paise = int(round(float(best_bid_val) * 100))
                    best_ask_paise = int(round(float(best_ask_val) * 100))
                    spread_paise = max(0, best_ask_paise - best_bid_paise)
                    bid_px = _pad_levels([best_bid_paise])
                    ask_px = _pad_levels([best_ask_paise])

            out[tok].append(
                ts,
                ltp_paise,
                volume,
                oi,
                spread_paise,
                atp_paise=atp_paise,
                ltq=ltq,
                total_buy=total_buy,
                total_sell=total_sell,
                bid_prices_paise=bid_px,
                ask_prices_paise=ask_px,
                bid_quantities=bid_qty,
                ask_quantities=ask_qty,
            )
        except (TypeError, ValueError):
            continue
    return out


def uniform_grid(open_ts: float, close_ts: float, step_sec: float = EMA_BAR_INTERVAL_SEC) -> list[float]:
    """Fixed-interval grid from session open through close (inclusive)."""
    step = float(step_sec)
    if close_ts <= open_ts or step <= 0:
        return []
    start = int(open_ts)
    rem = start % int(step)
    if rem:
        start += int(step) - rem
    if start < open_ts:
        start += int(step)
    rows: list[float] = []
    t = float(start)
    while t <= close_ts + 0.001:
        rows.append(t)
        t += step
    return rows


def minute_grid(open_ts: float, close_ts: float) -> list[float]:
    """One row per minute from session open through close (inclusive)."""
    if close_ts <= open_ts:
        return []
    start = int(open_ts)
    start = start - (start % 60)
    if start < open_ts:
        start += 60
    rows: list[float] = []
    t = float(start)
    while t <= close_ts + 0.001:
        rows.append(t)
        t += 60.0
    return rows
