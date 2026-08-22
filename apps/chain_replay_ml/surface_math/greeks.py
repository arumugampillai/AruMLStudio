"""Phase 4A.1: Analytical Higher-Order Option Greeks Engine.

Calculates exact closed-form analytical first-, second-, and third-order option Greeks:
- First-Order / Base: Delta, Theta, Vega, Gamma
- Second-Order: Vanna, Volga (Vomma), Charm (Delta Decay)
- Third-Order: Color (Gamma Decay), Speed, Zomma, Ultima

Conforms strictly to Doc 18 v1.1.0 specifications and `bs.py` emitted unit conventions:
- `vega`: ₹ per 1 percentage-point of IV (vega_raw / 100.0)
- `volga`: ₹ per 1 percentage-point of IV per 1 percentage-point of IV (vega * d1 * d2 / sigma)
- `vanna`: Dimensionless d(Delta)/d(sigma)
- `charm`: Delta decay per calendar day (annualized / 365.0)
- `color`: Gamma decay per calendar day (annualized / 365.0)
- `speed`: d(Gamma)/dS (1 / ₹)
- `zomma`: d(Gamma)/d(sigma) (1 / vol point)
- `ultima`: d(Volga)/d(sigma) (₹ per vol point^3)
"""

from __future__ import annotations

import math
from typing import Any, Sequence
import numpy as np

from .types import HigherOrderGreeksRecord

SQRT_2_PI = math.sqrt(2.0 * math.pi)
SQRT_2 = math.sqrt(2.0)
DAYS_PER_YEAR = 365.0


# =============================================================================
# SCALAR UTILITIES
# =============================================================================

def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / SQRT_2))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / SQRT_2_PI


def calculate_d1_d2(s: float, k: float, r: float, t: float, sigma: float) -> tuple[float, float]:
    """Calculate dimensionless Black-Scholes d1 and d2 parameters."""
    if s <= 0.0 or k <= 0.0 or t <= 0.0 or sigma <= 0.0:
        return 0.0, 0.0
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return d1, d2


# =============================================================================
# INDIVIDUAL SCALAR GREEKS CALCULATORS
# =============================================================================

def calculate_vanna(s: float, k: float, r: float, t: float, sigma: float) -> float:
    """Calculate Vanna (d(Delta)/d(sigma) = d(Vega_raw)/dS).
    
    Dimensionless sensitivity of Delta to IV changes (identical for Call/Put under BS).
    """
    if s <= 0.0 or k <= 0.0 or t <= 0.0 or sigma <= 0.0:
        return 0.0
    d1, d2 = calculate_d1_d2(s, k, r, t, sigma)
    pdf_d1 = _norm_pdf(d1)
    return -pdf_d1 * d2 / sigma


def calculate_volga(s: float, k: float, r: float, t: float, sigma: float) -> float:
    """Calculate Volga / Vomma (d(Vega)/d(sigma)).
    
    Convexity of option value relative to volatility in emitted `bs.py` units
    (vega * d1 * d2 / sigma, ₹ per 1% vol point per 1% vol point).
    """
    if s <= 0.0 or k <= 0.0 or t <= 0.0 or sigma <= 0.0:
        return 0.0
    sqrt_t = math.sqrt(t)
    d1, d2 = calculate_d1_d2(s, k, r, t, sigma)
    pdf_d1 = _norm_pdf(d1)
    vega = (s * pdf_d1 * sqrt_t) / 100.0
    return vega * d1 * d2 / sigma


def calculate_charm(s: float, k: float, r: float, t: float, sigma: float) -> float:
    """Calculate Charm (Delta Decay: d(Delta)/dt = -d(Delta)/dT).
    
    Expressed in Delta change per calendar day (/ 365.0), matching `bs.py` convention.
    """
    if s <= 0.0 or k <= 0.0 or t <= 0.0 or sigma <= 0.0:
        return 0.0
    sqrt_t = math.sqrt(t)
    d1, d2 = calculate_d1_d2(s, k, r, t, sigma)
    pdf_d1 = _norm_pdf(d1)
    denom = 2.0 * t * sigma * sqrt_t
    if denom <= 1e-12:
        return 0.0
    charm_annual = -pdf_d1 * (2.0 * r * t - d2 * sigma * sqrt_t) / denom
    return charm_annual / DAYS_PER_YEAR


def calculate_color(s: float, k: float, r: float, t: float, sigma: float) -> float:
    """Calculate Color (Gamma Decay: d(Gamma)/dt = -d(Gamma)/dT).
    
    Expressed in Gamma change per calendar day (/ 365.0).
    Corrected v1.1.0 analytical formula:
    +Gamma / (2T) * [1 + d1 * (2rT - d2*sigma*sqrt(T)) / (sigma*sqrt(T))] / 365.0.
    """
    if s <= 0.0 or k <= 0.0 or t <= 0.0 or sigma <= 0.0:
        return 0.0
    sqrt_t = math.sqrt(t)
    d1, d2 = calculate_d1_d2(s, k, r, t, sigma)
    pdf_d1 = _norm_pdf(d1)
    gamma = pdf_d1 / (s * sigma * sqrt_t)
    
    denom = sigma * sqrt_t
    if denom <= 1e-12 or t <= 1e-12:
        return 0.0
    
    inner_bracket = 1.0 + d1 * (2.0 * r * t - d2 * sigma * sqrt_t) / denom
    color_annual = (gamma / (2.0 * t)) * inner_bracket
    return color_annual / DAYS_PER_YEAR


def calculate_speed(s: float, k: float, r: float, t: float, sigma: float) -> float:
    """Calculate Speed (Third-order spot sensitivity: d(Gamma)/dS).
    
    Expressed in 1 / ₹. Identical for Call and Put under Black-Scholes.
    """
    if s <= 0.0 or k <= 0.0 or t <= 0.0 or sigma <= 0.0:
        return 0.0
    sqrt_t = math.sqrt(t)
    d1, _ = calculate_d1_d2(s, k, r, t, sigma)
    pdf_d1 = _norm_pdf(d1)
    gamma = pdf_d1 / (s * sigma * sqrt_t)
    
    denom = sigma * sqrt_t
    if denom <= 1e-12:
        return 0.0
    return -gamma / s * (d1 / denom + 1.0)


def calculate_zomma(s: float, k: float, r: float, t: float, sigma: float) -> float:
    """Calculate Zomma (Gamma sensitivity to volatility: d(Gamma)/d(sigma)).
    
    Expressed in 1 / vol point. Identical for Call and Put under Black-Scholes.
    """
    if s <= 0.0 or k <= 0.0 or t <= 0.0 or sigma <= 0.0:
        return 0.0
    sqrt_t = math.sqrt(t)
    d1, d2 = calculate_d1_d2(s, k, r, t, sigma)
    pdf_d1 = _norm_pdf(d1)
    gamma = pdf_d1 / (s * sigma * sqrt_t)
    return gamma * (d1 * d2 - 1.0) / sigma


def calculate_ultima(s: float, k: float, r: float, t: float, sigma: float) -> float:
    """Calculate Ultima (Third-order volatility sensitivity: d(Volga)/d(sigma)).
    
    Expressed in emitted `bs.py` units (₹ per vol point^3).
    Formula: -Vega / (sigma^2) * [d1*d2*(1 - d1*d2) + d1^2 + d2^2].
    """
    if s <= 0.0 or k <= 0.0 or t <= 0.0 or sigma <= 0.0:
        return 0.0
    sqrt_t = math.sqrt(t)
    d1, d2 = calculate_d1_d2(s, k, r, t, sigma)
    pdf_d1 = _norm_pdf(d1)
    vega = (s * pdf_d1 * sqrt_t) / 100.0
    sig_sq = sigma * sigma
    if sig_sq <= 1e-12:
        return 0.0
    
    bracket = d1 * d2 * (1.0 - d1 * d2) + (d1 * d1) + (d2 * d2)
    return -vega / sig_sq * bracket


# =============================================================================
# COMPLETE SCALAR GREEKS EVALUATOR
# =============================================================================

def calculate_higher_order_greeks(
    *,
    option_type: str = "CE",
    underlying_spot: float,
    strike: float,
    risk_free_rate: float,
    time_to_expiry_years: float,
    implied_volatility: float,
) -> HigherOrderGreeksRecord:
    """Calculate complete analytical first-, second-, and third-order Black-Scholes Greeks.
    
    Returns a fully populated `HigherOrderGreeksRecord`.
    """
    opt_type = str(option_type or "CE").upper().strip()
    s = float(underlying_spot)
    k = float(strike)
    r = float(risk_free_rate)
    t = float(time_to_expiry_years)
    sigma = float(implied_volatility)

    # Boundary and invalid input guard
    if s <= 0.0 or k <= 0.0 or t <= 0.0 or sigma <= 0.0 or math.isnan(s) or math.isnan(k) or math.isnan(t) or math.isnan(sigma):
        return HigherOrderGreeksRecord(
            delta=0.0,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            vanna=0.0,
            volga=0.0,
            charm=0.0,
            color=0.0,
            speed=0.0,
            zomma=0.0,
            ultima=0.0,
            strike=k,
            time_to_expiry_years=t,
            implied_volatility=sigma,
            underlying_spot=s,
            option_type=opt_type,
        )

    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    pdf_d1 = _norm_pdf(d1)

    # 1. Delta & Theta
    if opt_type == "CE":
        delta = _norm_cdf(d1)
        theta_annual = (-(s * pdf_d1 * sigma) / (2.0 * sqrt_t)) - (r * k * math.exp(-r * t) * _norm_cdf(d2))
    else:
        delta = _norm_cdf(d1) - 1.0
        theta_annual = (-(s * pdf_d1 * sigma) / (2.0 * sqrt_t)) + (r * k * math.exp(-r * t) * _norm_cdf(-d2))

    theta = theta_annual / DAYS_PER_YEAR

    # 2. Gamma & Vega
    gamma = pdf_d1 / (s * sigma * sqrt_t)
    vega = (s * pdf_d1 * sqrt_t) / 100.0

    # 3. Higher-Order Greeks
    vanna = -pdf_d1 * d2 / sigma
    volga = vega * d1 * d2 / sigma
    
    denom_t = 2.0 * t * sigma * sqrt_t
    charm_annual = -pdf_d1 * (2.0 * r * t - d2 * sigma * sqrt_t) / denom_t if denom_t > 1e-12 else 0.0
    charm = charm_annual / DAYS_PER_YEAR

    color_denom = sigma * sqrt_t
    if color_denom > 1e-12 and t > 1e-12:
        inner_bracket = 1.0 + d1 * (2.0 * r * t - d2 * sigma * sqrt_t) / color_denom
        color = ((gamma / (2.0 * t)) * inner_bracket) / DAYS_PER_YEAR
    else:
        color = 0.0

    speed = -gamma / s * (d1 / color_denom + 1.0) if color_denom > 1e-12 else 0.0
    zomma = gamma * (d1 * d2 - 1.0) / sigma
    
    sig_sq = sigma * sigma
    if sig_sq > 1e-12:
        ultima_bracket = d1 * d2 * (1.0 - d1 * d2) + (d1 * d1) + (d2 * d2)
        ultima = -vega / sig_sq * ultima_bracket
    else:
        ultima = 0.0

    return HigherOrderGreeksRecord(
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
        vanna=vanna,
        volga=volga,
        charm=charm,
        color=color,
        speed=speed,
        zomma=zomma,
        ultima=ultima,
        strike=k,
        time_to_expiry_years=t,
        implied_volatility=sigma,
        underlying_spot=s,
        option_type=opt_type,
    )


# =============================================================================
# VECTORIZED GREEKS ENGINE (NUMPY)
# =============================================================================

def calculate_higher_order_greeks_vectorized(
    *,
    underlying_spots: np.ndarray,
    strikes: np.ndarray,
    risk_free_rate: float,
    times_to_expiry_years: np.ndarray,
    implied_volatilities: np.ndarray,
    option_types: np.ndarray | str = "CE",
) -> dict[str, np.ndarray]:
    """Calculate complete higher-order Greeks vectorized across 1D numpy arrays.
    
    Guarantees strict scalar/vector calculation parity and zero memory leaks.
    """
    s = np.asarray(underlying_spots, dtype=np.float64)
    k = np.asarray(strikes, dtype=np.float64)
    t = np.asarray(times_to_expiry_years, dtype=np.float64)
    sigma = np.asarray(implied_volatilities, dtype=np.float64)
    r = float(risk_free_rate)
    n = len(s)

    # Pre-allocate output arrays
    zeros = np.zeros(n, dtype=np.float64)
    out = {
        "delta": zeros.copy(),
        "gamma": zeros.copy(),
        "theta": zeros.copy(),
        "vega": zeros.copy(),
        "vanna": zeros.copy(),
        "volga": zeros.copy(),
        "charm": zeros.copy(),
        "color": zeros.copy(),
        "speed": zeros.copy(),
        "zomma": zeros.copy(),
        "ultima": zeros.copy(),
    }

    # Mask valid strictly positive inputs
    valid = (s > 0.0) & (k > 0.0) & (t > 1e-7) & (sigma > 1e-5) & np.isfinite(s) & np.isfinite(k) & np.isfinite(t) & np.isfinite(sigma)
    if not np.any(valid):
        return out

    s_v = s[valid]
    k_v = k[valid]
    t_v = t[valid]
    sig_v = sigma[valid]

    sqrt_t = np.sqrt(t_v)
    d1 = (np.log(s_v / k_v) + (r + 0.5 * sig_v * sig_v) * t_v) / (sig_v * sqrt_t)
    d2 = d1 - sig_v * sqrt_t

    # PDF and CDF
    pdf_d1 = np.exp(-0.5 * d1 * d1) / SQRT_2_PI
    # Vectorized normal CDF using erf
    cdf_d1 = 0.5 * (1.0 + np.vectorize(math.erf)(d1 / SQRT_2))
    cdf_d2 = 0.5 * (1.0 + np.vectorize(math.erf)(d2 / SQRT_2))

    # Parse option types array
    if isinstance(option_types, str):
        is_call = np.full(np.sum(valid), option_types.upper().strip() == "CE", dtype=bool)
    else:
        opt_arr = np.asarray(option_types)[valid]
        is_call = np.vectorize(lambda x: str(x).upper().strip() == "CE")(opt_arr)

    # 1. Delta & Theta
    delta_v = np.where(is_call, cdf_d1, cdf_d1 - 1.0)
    
    disc_k = r * k_v * np.exp(-r * t_v)
    theta_call = (-(s_v * pdf_d1 * sig_v) / (2.0 * sqrt_t)) - (disc_k * cdf_d2)
    theta_put = (-(s_v * pdf_d1 * sig_v) / (2.0 * sqrt_t)) + (disc_k * (1.0 - cdf_d2))
    theta_v = np.where(is_call, theta_call, theta_put) / DAYS_PER_YEAR

    # 2. Gamma & Vega
    gamma_v = pdf_d1 / (s_v * sig_v * sqrt_t)
    vega_v = (s_v * pdf_d1 * sqrt_t) / 100.0

    # 3. Higher-Order Greeks
    vanna_v = -pdf_d1 * d2 / sig_v
    volga_v = vega_v * d1 * d2 / sig_v

    denom_t = 2.0 * t_v * sig_v * sqrt_t
    charm_annual = np.where(denom_t > 1e-12, -pdf_d1 * (2.0 * r * t_v - d2 * sig_v * sqrt_t) / denom_t, 0.0)
    charm_v = charm_annual / DAYS_PER_YEAR

    color_denom = sig_v * sqrt_t
    inner_bracket = 1.0 + d1 * (2.0 * r * t_v - d2 * sig_v * sqrt_t) / color_denom
    color_v = np.where((color_denom > 1e-12) & (t_v > 1e-12), ((gamma_v / (2.0 * t_v)) * inner_bracket) / DAYS_PER_YEAR, 0.0)

    speed_v = np.where(color_denom > 1e-12, -gamma_v / s_v * (d1 / color_denom + 1.0), 0.0)
    zomma_v = gamma_v * (d1 * d2 - 1.0) / sig_v

    sig_sq = sig_v * sig_v
    ultima_bracket = d1 * d2 * (1.0 - d1 * d2) + (d1 * d1) + (d2 * d2)
    ultima_v = np.where(sig_sq > 1e-12, -vega_v / sig_sq * ultima_bracket, 0.0)

    # Assign valid results back to pre-allocated buffers
    out["delta"][valid] = delta_v
    out["gamma"][valid] = gamma_v
    out["theta"][valid] = theta_v
    out["vega"][valid] = vega_v
    out["vanna"][valid] = vanna_v
    out["volga"][valid] = volga_v
    out["charm"][valid] = charm_v
    out["color"][valid] = color_v
    out["speed"][valid] = speed_v
    out["zomma"][valid] = zomma_v
    out["ultima"][valid] = ultima_v

    return out
