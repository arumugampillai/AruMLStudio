"""Hybrid re-anchor state machine (IV + Spot + Time)."""

from __future__ import annotations

from dataclasses import dataclass, field

from . import bs
from .constants import (
    DEFAULT_IV_THRESHOLD_PCT,
    DEFAULT_MAX_ROLL_AGE_MIN,
    DEFAULT_SPOT_THRESHOLD_PCT,
    RISK_FREE_RATE,
)


@dataclass
class ReanchorThresholds:
    iv_pct: float = DEFAULT_IV_THRESHOLD_PCT
    spot_pct: float = DEFAULT_SPOT_THRESHOLD_PCT
    max_age_min: float = DEFAULT_MAX_ROLL_AGE_MIN


@dataclass
class RollState:
    roll_iv: float | None = None
    roll_anchor_ts: float = 0.0
    roll_spot: float | None = None
    roll_ltp: float | None = None
    roll_greeks: dict[str, float] = field(default_factory=dict)
    roll_count: int = 0


def evaluate_triggers(
    *,
    actual_iv: float | None,
    actual_spot: float | None,
    roll: RollState,
    row_ts: float,
    thresholds: ReanchorThresholds,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if actual_iv is not None and roll.roll_iv and roll.roll_iv > 0:
        iv_drift = abs(actual_iv - roll.roll_iv) / roll.roll_iv * 100.0
        if iv_drift >= thresholds.iv_pct:
            reasons.append("IV")
    if actual_spot is not None and roll.roll_spot and roll.roll_spot > 0:
        spot_drift = abs(actual_spot - roll.roll_spot) / roll.roll_spot * 100.0
        if spot_drift >= thresholds.spot_pct:
            reasons.append("Spot")
    age_min = (row_ts - roll.roll_anchor_ts) / 60.0
    if age_min >= thresholds.max_age_min:
        reasons.append("Time")
    return (len(reasons) > 0, reasons)


def apply_roll(
    roll: RollState,
    *,
    actual_iv: float | None,
    actual_spot: float | None,
    actual_ltp: float | None,
    row_ts: float,
    option_type: str,
    strike_rupees: float,
    expiry_ts: float,
) -> None:
    if actual_iv is not None:
        roll.roll_iv = actual_iv
    roll.roll_anchor_ts = row_ts
    if actual_spot is not None:
        roll.roll_spot = actual_spot
    if actual_ltp is not None:
        roll.roll_ltp = actual_ltp
    t_row = bs.time_to_expiry_years(expiry_ts, row_ts)
    iv_for_g = roll.roll_iv
    spot_for_g = roll.roll_spot
    if spot_for_g and iv_for_g and iv_for_g > 0 and t_row > 0:
        roll.roll_greeks = bs.greeks(option_type, spot_for_g, strike_rupees, RISK_FREE_RATE, t_row, iv_for_g)
    roll.roll_count += 1


def iv_drift_from_roll_pct(actual_iv: float | None, roll_iv: float | None) -> float | None:
    if actual_iv is None or roll_iv is None or roll_iv <= 0:
        return None
    return (actual_iv - roll_iv) / roll_iv * 100.0


def spot_drift_from_roll_pct(actual_spot: float | None, roll_spot: float | None) -> float | None:
    if actual_spot is None or roll_spot is None or roll_spot <= 0:
        return None
    return (actual_spot - roll_spot) / roll_spot * 100.0


# Bit flags: IV=1, Spot=2, Time=4 (matches "+" join order from evaluate_triggers)
_ROLL_REASON_CODES: dict[str, int] = {
    "no": 0,
    "IV": 1,
    "Spot": 2,
    "Time": 3,
    "IV+Spot": 4,
    "IV+Time": 5,
    "Spot+Time": 6,
    "IV+Spot+Time": 7,
}


def encode_roll_reason(reason: str) -> int:
    """Map roll_reason string to numeric code for tree models."""
    return _ROLL_REASON_CODES.get(reason, 0)
