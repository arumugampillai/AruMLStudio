"""Chain Gamma Exposure (GEX) for controller ``token.chain`` (Wave B).

Convention (SpotGamma-style 1% move, lot_size = 1 until wired from meta):

  contrib_i = gamma_i × OI_i × spot² × 0.01

  call_gex  = Σ contrib over CE
  put_gex   = Σ contrib over PE          (positive magnitude)
  net_gex   = call_gex − put_gex         (puts subtract; dealer-short / RR convention)
  chain_gex = call_gex + put_gex         (total unsigned gamma×OI mass)

Proxy over the loaded nearest-weekly ATM band chain — not full-street GEX.
RoC / lag / rolling → Transformation Pipeline only.
"""

from __future__ import annotations

from typing import Any, Iterable

from chain_replay_ml import bs
from chain_replay_ml.constants import RISK_FREE_RATE
from chain_replay_ml.ticks import TickTimeline

# 1% spot move scaling (common public GEX convention).
GEX_SPOT_MOVE = 0.01

CHAIN_GEX_FEATURES: tuple[str, ...] = (
    "call_gex",
    "put_gex",
    "net_gex",
    "chain_gex",
    "gamma_flip_spot",
    "gamma_flip_distance",
)
CHAIN_GEX_FEATURE_SET: frozenset[str] = frozenset(CHAIN_GEX_FEATURES)


def active_chain_gex_features(active: Iterable[str] | None) -> frozenset[str]:
    if active is None:
        return CHAIN_GEX_FEATURE_SET
    return frozenset(str(f) for f in active if str(f) in CHAIN_GEX_FEATURE_SET)


def needs_chain_gex(active: Iterable[str] | None) -> bool:
    if active is None:
        return True
    return any(str(f) in CHAIN_GEX_FEATURE_SET for f in active)


def _gamma_oi_contrib(
    *,
    option_type: str,
    strike: float,
    ts: float,
    spot: float,
    t_exp: float,
    tl: TickTimeline,
) -> float | None:
    oi = tl.oi_at(ts)
    if oi is None or oi <= 0:
        return None
    ltp = tl.ltp_rupees_at(ts)
    if ltp is None or ltp <= 0:
        return None
    iv = bs.implied_volatility(
        option_type, ltp, spot, float(strike), RISK_FREE_RATE, t_exp
    )
    if iv is None or iv <= 0:
        return None
    gamma = float(
        bs.greeks(option_type, spot, float(strike), RISK_FREE_RATE, t_exp, float(iv)).get(
            "gamma", 0.0
        )
    )
    if gamma <= 0:
        return None
    return float(gamma) * float(oi) * float(spot) * float(spot) * GEX_SPOT_MOVE


def compute_chain_gex_at(
    strike_mapping: dict[tuple[float, str], tuple[str, str, TickTimeline]],
    *,
    index_tl: TickTimeline,
    ts: float,
    expiry_ts: float,
) -> dict[str, float | None]:
    """Compute call/put/net/chain GEX and gamma-flip levels at one timestamp."""
    out: dict[str, float | None] = {name: None for name in CHAIN_GEX_FEATURES}
    if expiry_ts <= ts:
        return out
    spot = index_tl.ltp_rupees_at(ts)
    if spot is None or spot <= 0:
        return out
    t_exp = bs.time_to_expiry_years(expiry_ts, ts)
    if t_exp is None or t_exp <= 0:
        return out

    call_sum = 0.0
    put_sum = 0.0
    n_call = 0
    n_put = 0
    spot_f = float(spot)
    net_by_strike: dict[float, float] = {}
    for (strike_r, opt_type), (_tok, _sym, tl) in strike_mapping.items():
        opt = str(opt_type).upper()
        contrib = _gamma_oi_contrib(
            option_type=opt,
            strike=float(strike_r),
            ts=ts,
            spot=spot_f,
            t_exp=float(t_exp),
            tl=tl,
        )
        if contrib is None:
            continue
        strike_f = float(strike_r)
        if opt == "CE":
            call_sum += contrib
            n_call += 1
            net_by_strike[strike_f] = net_by_strike.get(strike_f, 0.0) + float(contrib)
        elif opt == "PE":
            put_sum += contrib
            n_put += 1
            net_by_strike[strike_f] = net_by_strike.get(strike_f, 0.0) - float(contrib)

    if n_call <= 0 and n_put <= 0:
        return out
    out["call_gex"] = float(call_sum) if n_call > 0 else 0.0
    out["put_gex"] = float(put_sum) if n_put > 0 else 0.0
    out["net_gex"] = float(call_sum - put_sum)
    out["chain_gex"] = float(call_sum + put_sum)

    flip = _gamma_flip_spot_from_net_by_strike(net_by_strike)
    if flip is not None and flip > 0:
        out["gamma_flip_spot"] = float(flip)
        out["gamma_flip_distance"] = float((spot_f - flip) / spot_f)
    return out


def _gamma_flip_spot_from_net_by_strike(net_by_strike: dict[float, float]) -> float | None:
    """Zero-crossing of cumulative net GEX across strikes (low → high).

    Linear interpolation between the two strikes where cumulative signed GEX
    changes sign. Returns None when fewer than two strikes or no crossing.
    """
    strikes = sorted(net_by_strike)
    if len(strikes) < 2:
        return None
    cum = 0.0
    prev_strike: float | None = None
    prev_cum = 0.0
    for strike in strikes:
        prev_cum = cum
        cum += float(net_by_strike[strike])
        if prev_strike is not None and prev_cum * cum < 0.0:
            denom = abs(prev_cum) + abs(cum)
            if denom <= 0:
                return float(strike)
            w = abs(prev_cum) / denom
            return float(prev_strike + w * (strike - prev_strike))
        prev_strike = float(strike)
    return None


__all__ = [
    "CHAIN_GEX_FEATURES",
    "CHAIN_GEX_FEATURE_SET",
    "GEX_SPOT_MOVE",
    "active_chain_gex_features",
    "compute_chain_gex_at",
    "needs_chain_gex",
]
