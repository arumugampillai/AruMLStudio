"""Spot HL formula helpers — features emit via spot_hl_registry + SpotControllers.hl."""

from __future__ import annotations

import numpy as np

SPOT_HL_PERIODS: tuple[int, ...] = (20, 50, 100, 200, 300)
CHANNEL_EPS = 1e-6


def _ema_series_from_prices(prices: np.ndarray, period: int) -> np.ndarray:
    ema = np.zeros_like(prices)
    if len(prices) == 0:
        return ema
    ema[0] = prices[0]
    alpha = 2.0 / (float(period) + 1.0)
    for idx in range(1, len(prices)):
        ema[idx] = prices[idx] * alpha + ema[idx - 1] * (1.0 - alpha)
    return ema


def _weighted_blend(
    ema20: float | None,
    ema50: float | None,
    ema200: float | None,
    ema300: float | None,
) -> float | None:
    if ema20 is None or ema50 is None or ema200 is None or ema300 is None:
        return None
    return float(ema20) * 4.0 + float(ema50) * 3.0 + float(ema200) * 2.0 + float(ema300)


def _ratio_to_ltp(numerator: float | None, ltp: float) -> float:
    if numerator is None or ltp <= 0:
        return 0.0
    return float(numerator) / float(ltp)


def _weighted_to_ltp_ratio(
    ema20: float | None,
    ema50: float | None,
    ema200: float | None,
    ema300: float | None,
    ltp: float,
) -> float:
    blend = _weighted_blend(ema20, ema50, ema200, ema300)
    if blend is None or ltp <= 0:
        return 0.0
    return float(blend) / (10.0 * float(ltp))


def _safe_ratio(numerator: float | None, denominator: float | None) -> float:
    if numerator is None or denominator is None:
        return 0.0
    den = float(denominator)
    if den == 0:
        return 0.0
    return float(numerator) / den


def _ltp_to_channel_width(ltp: float, high_ema: float | None, low_ema: float | None) -> float:
    if ltp <= 0 or high_ema is None or low_ema is None:
        return 0.0
    width = float(high_ema) - float(low_ema)
    return float(ltp) / (abs(width) + CHANNEL_EPS)
