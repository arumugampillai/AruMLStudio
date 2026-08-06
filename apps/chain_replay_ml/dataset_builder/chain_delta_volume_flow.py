"""Chain delta-weighted volume flow for controller ``token.chain`` (Wave B).

Canonical level:
  delta_w_volume_flow_{1m|5m} = Σ_i (delta_i × Δday_volume_i)

over the loaded expiry chain. Δvolume = volume(ts) − volume(ts − horizon).
Lag / difference / rolling → Transformation Pipeline only.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

from chain_replay_ml.features_atm_band import compute_delta_at_ts
from chain_replay_ml.ticks import TickTimeline

DELTA_W_VOLUME_FLOW_HORIZONS: tuple[tuple[str, float], ...] = (
    ("1m", 60.0),
    ("5m", 300.0),
)

DELTA_W_VOLUME_FLOW_FEATURES: tuple[str, ...] = tuple(
    f"delta_w_volume_flow_{suffix}" for suffix, _ in DELTA_W_VOLUME_FLOW_HORIZONS
)
DELTA_W_VOLUME_FLOW_FEATURE_SET: frozenset[str] = frozenset(DELTA_W_VOLUME_FLOW_FEATURES)


def active_delta_w_volume_flow_features(active: Iterable[str] | None) -> frozenset[str]:
    if active is None:
        return DELTA_W_VOLUME_FLOW_FEATURE_SET
    return frozenset(str(f) for f in active if str(f) in DELTA_W_VOLUME_FLOW_FEATURE_SET)


def needs_delta_w_volume_flow(active: Iterable[str] | None) -> bool:
    if active is None:
        return True
    return any(str(f) in DELTA_W_VOLUME_FLOW_FEATURE_SET for f in active)


def _volume_delta(tl: TickTimeline, ts: float, lookback_sec: float) -> float | None:
    cur = tl.volume_at(ts)
    past = tl.volume_at(ts - float(lookback_sec))
    if cur is None or past is None:
        return None
    return float(cur) - float(past)


def compute_delta_w_volume_flow_at(
    strike_mapping: dict[tuple[float, str], tuple[str, str, TickTimeline]],
    *,
    index_tl: TickTimeline,
    ts: float,
    expiry_ts: float,
    lookback_sec: float,
) -> float | None:
    """Σ (BS delta × traded volume over lookback) across monitored chain."""
    if lookback_sec <= 0 or expiry_ts <= ts:
        return None
    total = 0.0
    n_ok = 0
    for (strike_r, opt_type), (_tok, _sym, tl) in strike_mapping.items():
        dvol = _volume_delta(tl, ts, lookback_sec)
        if dvol is None or dvol == 0.0:
            continue
        delta = compute_delta_at_ts(
            ts=ts,
            index_timeline=index_tl,
            option_timeline=tl,
            option_type=str(opt_type),
            strike_rupees=float(strike_r),
            expiry_ts=expiry_ts,
        )
        if delta is None:
            continue
        total += float(delta) * float(dvol)
        n_ok += 1
    if n_ok <= 0:
        return None
    return float(total)


def compute_all_delta_w_volume_flows_at(
    strike_mapping: dict[tuple[float, str], tuple[str, str, TickTimeline]],
    *,
    index_tl: TickTimeline,
    ts: float,
    expiry_ts: float,
    horizons: Sequence[tuple[str, float]] = DELTA_W_VOLUME_FLOW_HORIZONS,
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for suffix, lookback in horizons:
        out[f"delta_w_volume_flow_{suffix}"] = compute_delta_w_volume_flow_at(
            strike_mapping,
            index_tl=index_tl,
            ts=ts,
            expiry_ts=expiry_ts,
            lookback_sec=float(lookback),
        )
    return out
