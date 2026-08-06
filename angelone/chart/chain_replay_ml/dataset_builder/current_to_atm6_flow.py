"""Current-strike to ATM+6 wing flow feature — volume/OI participation toward ATM."""

from __future__ import annotations

from typing import Any, Iterable

from chain_replay_ml.features_atm_band import compute_pct_change
from chain_replay_ml.ticks import TickTimeline

LOOKBACK_SEC = 60.0
NUM_STRIKES = 7

CURRENT_TO_ATM6_FLOW_FEATURE = "current_to_atm6_flow_delta_ltp_to_spot_ratio"

CURRENT_TO_ATM6_FLOW_FEATURES: frozenset[str] = frozenset({CURRENT_TO_ATM6_FLOW_FEATURE})


def active_current_to_atm6_flow_features(active: Iterable[str] | None) -> frozenset[str]:
    if not active:
        return CURRENT_TO_ATM6_FLOW_FEATURES
    wanted = frozenset(active)
    return CURRENT_TO_ATM6_FLOW_FEATURES & wanted


def needs_current_to_atm6_flow(active: Iterable[str] | None) -> bool:
    return bool(active_current_to_atm6_flow_features(active))


def strikes_toward_atm(
    current_strike: float,
    *,
    step: int,
    option_type: str,
) -> list[float]:
    """Seven strikes from current toward ATM: CE → lower, PE → higher."""
    if step <= 0:
        return []
    direction = -1 if str(option_type).upper() == "CE" else 1
    return [float(current_strike) + direction * i * float(step) for i in range(NUM_STRIKES)]


def _pct_change_at(
    timeline: TickTimeline,
    ts: float,
    *,
    attr: str,
    lookback_sec: float,
) -> float | None:
    if attr == "volume":
        cur = timeline.volume_at(ts)
        past = timeline.volume_at(ts - lookback_sec)
    elif attr == "oi":
        cur = timeline.oi_at(ts)
        past = timeline.oi_at(ts - lookback_sec)
    else:
        return None
    return compute_pct_change(
        float(cur) if cur is not None else None,
        float(past) if past is not None else None,
    )


def _avg_change_pct_across_strikes(
    strike_mapping: dict[tuple[float, str], tuple[str, str, TickTimeline]],
    ts: float,
    *,
    current_strike: float,
    step: int,
    option_type: str,
    attr: str,
    lookback_sec: float = LOOKBACK_SEC,
) -> float | None:
    values: list[float] = []
    opt = str(option_type).upper()
    for strike in strikes_toward_atm(current_strike, step=step, option_type=opt):
        entry = strike_mapping.get((strike, opt))
        if not entry:
            return None
        _, _, tl = entry
        pct = _pct_change_at(tl, ts, attr=attr, lookback_sec=lookback_sec)
        if pct is None:
            return None
        values.append(float(pct))
    if len(values) != NUM_STRIKES:
        return None
    return sum(values) / float(len(values))


def compute_current_to_atm6_flow_delta_ltp_to_spot_ratio(
    *,
    strike_mapping: dict[tuple[float, str], tuple[str, str, TickTimeline]],
    ts: float,
    current_strike: float,
    step: int,
    option_type: str,
    delta: float | None,
    ltp: float | None,
    spot: float | None,
    lookback_sec: float = LOOKBACK_SEC,
) -> float | None:
    """Flow strength (avg vol% + avg OI% over 7 strikes toward ATM) × |delta| × LTP / spot."""
    if step <= 0 or delta is None or ltp is None or ltp <= 0 or spot is None or spot <= 0:
        return None

    delta_abs = abs(float(delta))

    vol_avg = _avg_change_pct_across_strikes(
        strike_mapping,
        ts,
        current_strike=current_strike,
        step=step,
        option_type=option_type,
        attr="volume",
        lookback_sec=lookback_sec,
    )
    oi_avg = _avg_change_pct_across_strikes(
        strike_mapping,
        ts,
        current_strike=current_strike,
        step=step,
        option_type=option_type,
        attr="oi",
        lookback_sec=lookback_sec,
    )
    if vol_avg is None or oi_avg is None:
        return None

    flow_strength = (vol_avg + oi_avg) / 2.0
    return float(flow_strength * delta_abs * float(ltp) / float(spot))


def enrich_current_to_atm6_flow_features(
    raw: dict[str, Any],
    *,
    ts: float,
    strike_mapping: dict[tuple[float, str], tuple[str, str, TickTimeline]],
    strike_rupees: float,
    strike_step: int,
    option_type: str,
    active_features: frozenset[str] | None = None,
) -> dict[str, Any]:
    wanted = active_current_to_atm6_flow_features(active_features)
    out = dict(raw)
    if CURRENT_TO_ATM6_FLOW_FEATURE not in wanted:
        return out

    out[CURRENT_TO_ATM6_FLOW_FEATURE] = compute_current_to_atm6_flow_delta_ltp_to_spot_ratio(
        strike_mapping=strike_mapping,
        ts=ts,
        current_strike=float(strike_rupees),
        step=int(strike_step),
        option_type=str(option_type).upper(),
        delta=out.get("delta"),
        ltp=out.get("ltp"),
        spot=out.get("spot"),
    )
    return out
