"""Chain IV skew levels for controller ``token.chain`` (Wave B).

Canonical surface metrics from BS IV on the loaded expiry chain.
Changes / lags / rolling → Transformation Pipeline only.
"""

from __future__ import annotations

from typing import Any, Iterable

from chain_replay_ml import bs
from chain_replay_ml.constants import RISK_FREE_RATE
from chain_replay_ml.ticks import TickTimeline

# Fixed OTM wing distance in strike steps (mirrors ATM±5 wing used by ATM6).
IV_SKEW_ATM_STEPS = 5

# Acceptable delta band around ±0.25 for 25Δ risk-reversal.
_DELTA_25_LO = 0.15
_DELTA_25_HI = 0.35

CHAIN_IV_SKEW_FEATURES: tuple[str, ...] = (
    "iv_skew_atm",
    "iv_call_put_skew",
    "iv_skew_25d",
    "iv_butterfly_25d",
)
CHAIN_IV_SKEW_FEATURE_SET: frozenset[str] = frozenset(CHAIN_IV_SKEW_FEATURES)

CHAIN_ATM_IV_FEATURES: tuple[str, ...] = ("atm_iv_ce", "atm_iv_pe")
CHAIN_ATM_IV_FEATURE_SET: frozenset[str] = frozenset(CHAIN_ATM_IV_FEATURES)


def active_chain_iv_skew_features(active: Iterable[str] | None) -> frozenset[str]:
    if active is None:
        return CHAIN_IV_SKEW_FEATURE_SET
    return frozenset(str(f) for f in active if str(f) in CHAIN_IV_SKEW_FEATURE_SET)


def needs_chain_iv_skew(active: Iterable[str] | None) -> bool:
    if active is None:
        return True
    return any(str(f) in CHAIN_IV_SKEW_FEATURE_SET for f in active)


def needs_atm_iv(active: Iterable[str] | None) -> bool:
    if active is None:
        return True
    return any(str(f) in CHAIN_ATM_IV_FEATURE_SET for f in active)


def _option_iv_at(
    strike_mapping: dict[tuple[float, str], tuple[str, str, TickTimeline]],
    *,
    strike: float,
    option_type: str,
    ts: float,
    spot: float,
    t_exp: float,
) -> float | None:
    entry = strike_mapping.get((float(strike), option_type))
    if not entry:
        return None
    _, _, tl = entry
    ltp = tl.ltp_rupees_at(ts)
    if ltp is None or ltp <= 0:
        return None
    iv = bs.implied_volatility(
        option_type, ltp, spot, float(strike), RISK_FREE_RATE, t_exp
    )
    if iv is None or iv <= 0:
        return None
    return float(iv)


def _option_iv_delta_at(
    strike_mapping: dict[tuple[float, str], tuple[str, str, TickTimeline]],
    *,
    strike: float,
    option_type: str,
    ts: float,
    spot: float,
    t_exp: float,
) -> tuple[float | None, float | None]:
    entry = strike_mapping.get((float(strike), option_type))
    if not entry:
        return None, None
    _, _, tl = entry
    ltp = tl.ltp_rupees_at(ts)
    if ltp is None or ltp <= 0:
        return None, None
    iv = bs.implied_volatility(
        option_type, ltp, spot, float(strike), RISK_FREE_RATE, t_exp
    )
    if iv is None or iv <= 0:
        return None, None
    g = bs.greeks(option_type, spot, float(strike), RISK_FREE_RATE, t_exp, float(iv))
    return float(iv), float(g.get("delta", 0.0))


def compute_chain_iv_skew_at(
    strike_mapping: dict[tuple[float, str], tuple[str, str, TickTimeline]],
    *,
    ts: float,
    spot: float,
    atm_strike: float,
    strike_step: int,
    expiry_ts: float,
    wing_steps: int = IV_SKEW_ATM_STEPS,
) -> dict[str, float | None]:
    """Compute IV skew levels at one timestamp (IV units = decimal σ)."""
    out: dict[str, float | None] = {name: None for name in CHAIN_IV_SKEW_FEATURES}
    out["atm_iv_ce"] = None
    out["atm_iv_pe"] = None
    if spot <= 0 or strike_step <= 0 or expiry_ts <= ts:
        return out

    t_exp = bs.time_to_expiry_years(expiry_ts, ts)
    if t_exp is None or t_exp <= 0:
        return out

    atm = float(atm_strike)
    step = int(strike_step)
    k = max(1, int(wing_steps))

    iv_ce_atm = _option_iv_at(
        strike_mapping,
        strike=atm,
        option_type="CE",
        ts=ts,
        spot=spot,
        t_exp=t_exp,
    )
    iv_pe_atm = _option_iv_at(
        strike_mapping,
        strike=atm,
        option_type="PE",
        ts=ts,
        spot=spot,
        t_exp=t_exp,
    )
    out["atm_iv_ce"] = iv_ce_atm
    out["atm_iv_pe"] = iv_pe_atm
    if iv_ce_atm is not None and iv_pe_atm is not None:
        out["iv_call_put_skew"] = float(iv_ce_atm - iv_pe_atm)

    iv_pe_otm = _option_iv_at(
        strike_mapping,
        strike=atm - k * step,
        option_type="PE",
        ts=ts,
        spot=spot,
        t_exp=t_exp,
    )
    iv_ce_otm = _option_iv_at(
        strike_mapping,
        strike=atm + k * step,
        option_type="CE",
        ts=ts,
        spot=spot,
        t_exp=t_exp,
    )
    if iv_pe_otm is not None and iv_ce_otm is not None:
        # Positive ⇒ OTM puts richer than OTM calls (typical equity put skew).
        out["iv_skew_atm"] = float(iv_pe_otm - iv_ce_otm)

    # 25Δ risk reversal on the loaded chain (prefer deltas near ±0.25).
    best_ce: tuple[float, float] | None = None  # (abs err, iv)
    best_pe: tuple[float, float] | None = None
    for (strike_r, opt_type), _entry in strike_mapping.items():
        if opt_type == "CE":
            iv, delta = _option_iv_delta_at(
                strike_mapping,
                strike=strike_r,
                option_type="CE",
                ts=ts,
                spot=spot,
                t_exp=t_exp,
            )
            if iv is None or delta is None:
                continue
            if not (_DELTA_25_LO <= delta <= _DELTA_25_HI):
                continue
            err = abs(delta - 0.25)
            if best_ce is None or err < best_ce[0]:
                best_ce = (err, iv)
        elif opt_type == "PE":
            iv, delta = _option_iv_delta_at(
                strike_mapping,
                strike=strike_r,
                option_type="PE",
                ts=ts,
                spot=spot,
                t_exp=t_exp,
            )
            if iv is None or delta is None:
                continue
            if not (-_DELTA_25_HI <= delta <= -_DELTA_25_LO):
                continue
            err = abs(delta + 0.25)
            if best_pe is None or err < best_pe[0]:
                best_pe = (err, iv)

    if best_ce is not None and best_pe is not None:
        out["iv_skew_25d"] = float(best_pe[1] - best_ce[1])
        # ATM IV for butterfly: mean of ATM CE/PE when both present, else either side.
        atm_iv: float | None = None
        if iv_ce_atm is not None and iv_pe_atm is not None:
            atm_iv = 0.5 * (float(iv_ce_atm) + float(iv_pe_atm))
        elif iv_ce_atm is not None:
            atm_iv = float(iv_ce_atm)
        elif iv_pe_atm is not None:
            atm_iv = float(iv_pe_atm)
        if atm_iv is not None:
            out["iv_butterfly_25d"] = float(
                0.5 * (float(best_ce[1]) + float(best_pe[1])) - atm_iv
            )
    return out
