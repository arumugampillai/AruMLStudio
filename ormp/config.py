"""ORMP configuration (research sandbox only)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

PriceSource = Literal["close", "hlc3", "ohlc4", "typical_price"]
PathMode = Literal["snapshot", "continuous"]

DEFAULT_BAND_SIZE_PCT = 0.05
DEFAULT_PRICE_SOURCE: PriceSource = "close"
DEFAULT_PATH_MODE: PathMode = "snapshot"
DEFAULT_NIFTY_TOKEN = "99926000"
DEFAULT_INTERVAL_SEC = 60

# Market Context (multi-TF EMA ratios) — extend timeframes without schema rewrites.
DEFAULT_MARKET_CONTEXT_TIMEFRAMES: tuple[str, ...] = ("1m", "5m", "15m")
DEFAULT_EMA_PERIODS: tuple[int, ...] = (9, 20, 50, 100, 200)

# NSE cash session 09:15–15:29 inclusive → 375 one-minute bars on a full day.
FULL_SESSION_MINUTES = 375


def default_candle_db_path() -> str:
    """Read-only Angel historic OHLC store (1m NIFTY lives here)."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    return os.path.join(repo, "apps", "data", "angel_historic_bars.db")


def default_output_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "outputs")


@dataclass(frozen=True)
class OrmpConfig:
    """Build / engine configuration."""

    candle_db_path: str = ""
    output_dir: str = ""
    nifty_token: str = DEFAULT_NIFTY_TOKEN
    interval_sec: int = DEFAULT_INTERVAL_SEC
    band_size_pct: float = DEFAULT_BAND_SIZE_PCT
    price_source: PriceSource = DEFAULT_PRICE_SOURCE
    path_mode: PathMode = DEFAULT_PATH_MODE
    from_date: str | None = None  # YYYY-MM-DD inclusive
    to_date: str | None = None  # YYYY-MM-DD inclusive
    # Optional suffix for immutable versioning, e.g. "v20260723_022800".
    # Empty = classic tag only (ormp_dataset_bs0p05_close_snapshot.db).
    artifact_suffix: str = ""
    # Market Context: configurable TFs / EMA periods (ratios only for now).
    market_context_timeframes: tuple[str, ...] = field(
        default_factory=lambda: DEFAULT_MARKET_CONTEXT_TIMEFRAMES
    )
    ema_periods: tuple[int, ...] = field(default_factory=lambda: DEFAULT_EMA_PERIODS)

    def __post_init__(self) -> None:
        if not self.candle_db_path:
            object.__setattr__(self, "candle_db_path", default_candle_db_path())
        if not self.output_dir:
            object.__setattr__(self, "output_dir", default_output_dir())
        if self.band_size_pct <= 0:
            raise ValueError("ormp_band_size_pct must be > 0")
        if self.price_source not in ("close", "hlc3", "ohlc4", "typical_price"):
            raise ValueError(f"unsupported ormp_price_source: {self.price_source}")
        if self.path_mode not in ("snapshot", "continuous"):
            raise ValueError(f"unsupported ormp_path_mode: {self.path_mode}")
        tfs = tuple(str(t).strip().lower() for t in self.market_context_timeframes)
        periods = tuple(int(p) for p in self.ema_periods)
        if not tfs:
            raise ValueError("market_context_timeframes must be non-empty")
        if not periods or any(p <= 0 for p in periods):
            raise ValueError("ema_periods must be positive integers")
        object.__setattr__(self, "market_context_timeframes", tfs)
        object.__setattr__(self, "ema_periods", periods)
