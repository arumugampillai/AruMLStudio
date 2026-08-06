"""Technical indicators for Phase 1 spot features."""

from __future__ import annotations

import pandas as pd

from .constants import ATR_PERIOD, EMA_PERIODS


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=int(span), adjust=False).mean()


def add_emas(df: pd.DataFrame, prefix: str = "") -> pd.DataFrame:
    out = df.copy()
    close = out["close"]
    pfx = f"{prefix}_" if prefix else ""
    for period in EMA_PERIODS:
        out[f"{pfx}EMA{period}"] = ema(close, period)
    return out


def wilder_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def pct_spread(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denom = denominator.replace(0, pd.NA)
    return (numerator / denom) * 100.0


def pct_change(num: pd.Series, ref: pd.Series) -> pd.Series:
    ref_safe = ref.replace(0, pd.NA)
    return ((num - ref) / ref_safe) * 100.0
