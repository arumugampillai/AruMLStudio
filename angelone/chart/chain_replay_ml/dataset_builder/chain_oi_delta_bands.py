"""OI by abs(delta) band for controller ``token.chain`` (v30).

Snapshot aggregates — cannot be rebuilt from Master without these buckets.

Bands (half-open except the last):
  [0.0, 0.2)  [0.2, 0.4)  [0.4, 0.6)  [0.6, 0.8)  [0.8, 1.0]

Per band: CE OI and PE OI separately (10 Computed Base features).
Delta from BS IV on the loaded ATM-band chain (same path as GEX).
"""

from __future__ import annotations

from typing import Any, Iterable

from chain_replay_ml import bs
from chain_replay_ml.constants import RISK_FREE_RATE
from chain_replay_ml.ticks import TickTimeline

# (suffix, lo, hi_exclusive_or_inclusive_last)
_BAND_SPECS: tuple[tuple[str, float, float, bool], ...] = (
    ("0_20", 0.0, 0.2, False),
    ("20_40", 0.2, 0.4, False),
    ("40_60", 0.4, 0.6, False),
    ("60_80", 0.6, 0.8, False),
    ("80_100", 0.8, 1.0, True),
)

OI_ABS_DELTA_BAND_FEATURES: tuple[str, ...] = tuple(
    f"oi_abs_delta_{suffix}_{side}"
    for suffix, _, _, _ in _BAND_SPECS
    for side in ("ce", "pe")
)
OI_ABS_DELTA_BAND_FEATURE_SET: frozenset[str] = frozenset(OI_ABS_DELTA_BAND_FEATURES)


def active_oi_abs_delta_band_features(active: Iterable[str] | None) -> frozenset[str]:
    if active is None:
        return OI_ABS_DELTA_BAND_FEATURE_SET
    return frozenset(str(f) for f in active if str(f) in OI_ABS_DELTA_BAND_FEATURE_SET)


def needs_oi_abs_delta_bands(active: Iterable[str] | None) -> bool:
    if active is None:
        return True
    return any(str(f) in OI_ABS_DELTA_BAND_FEATURE_SET for f in active)


def _band_suffix(abs_delta: float) -> str | None:
    if abs_delta < 0.0:
        return None
    for suffix, lo, hi, inclusive_hi in _BAND_SPECS:
        if abs_delta < lo:
            continue
        if inclusive_hi:
            if abs_delta <= hi + 1e-12:
                return suffix
        elif abs_delta < hi:
            return suffix
    # Numerical |Δ| slightly above 1 → clamp into last band.
    if abs_delta > 1.0:
        return "80_100"
    return None


def compute_oi_abs_delta_bands_at(
    strike_mapping: dict[tuple[float, str], tuple[str, str, TickTimeline]],
    *,
    index_tl: TickTimeline,
    ts: float,
    expiry_ts: float,
) -> dict[str, float | None]:
    """Sum OI into abs(delta) × CE/PE buckets at one timestamp."""
    out: dict[str, float | None] = {name: None for name in OI_ABS_DELTA_BAND_FEATURES}
    if expiry_ts <= ts:
        return out
    spot = index_tl.ltp_rupees_at(ts)
    if spot is None or spot <= 0:
        return out
    t_exp = bs.time_to_expiry_years(expiry_ts, ts)
    if t_exp is None or t_exp <= 0:
        return out

    sums: dict[str, float] = {name: 0.0 for name in OI_ABS_DELTA_BAND_FEATURES}

    for (strike, opt_type), (_tok, _meta, tl) in strike_mapping.items():
        side = str(opt_type or "").upper()
        if side not in ("CE", "PE"):
            continue
        oi = tl.oi_at(ts)
        if oi is None or oi <= 0:
            continue
        ltp = tl.ltp_rupees_at(ts)
        if ltp is None or ltp <= 0:
            continue
        iv = bs.implied_volatility(
            side, ltp, float(spot), float(strike), RISK_FREE_RATE, t_exp
        )
        if iv is None or iv <= 0:
            continue
        delta = float(
            bs.greeks(side, float(spot), float(strike), RISK_FREE_RATE, t_exp, float(iv)).get(
                "delta", 0.0
            )
        )
        suffix = _band_suffix(abs(delta))
        if suffix is None:
            continue
        key = f"oi_abs_delta_{suffix}_{side.lower()}"
        if key in sums:
            sums[key] += float(oi)

    for name, val in sums.items():
        out[name] = float(val)
    return out


__all__ = [
    "OI_ABS_DELTA_BAND_FEATURES",
    "OI_ABS_DELTA_BAND_FEATURE_SET",
    "active_oi_abs_delta_band_features",
    "compute_oi_abs_delta_bands_at",
    "needs_oi_abs_delta_bands",
]
