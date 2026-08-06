"""Black-Scholes pricing, IV, and Greeks (aligned with ui/option_greeks_panel.py)."""

from __future__ import annotations

import math
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from .constants import EPS_T, RISK_FREE_RATE, SECONDS_IN_YEAR

IST = ZoneInfo("Asia/Kolkata")


def _parse_day_text(raw: str | date | datetime) -> date:
    """Accept common expiry/day formats used by UI and exports."""
    if isinstance(raw, datetime):
        return raw.date()
    if isinstance(raw, date):
        return raw
    text = str(raw or "").strip()
    for fmt in ("%Y-%m-%d", "%d%b%Y", "%d-%b-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    # Keep a strict failure so bad inputs are visible to callers.
    raise ValueError(f"Unsupported date format: {text!r}")


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def expiry_close_ts(expiry: str) -> float:
    day = _parse_day_text(expiry)
    dt = datetime(day.year, day.month, day.day, 15, 30, 0, tzinfo=IST)
    return dt.timestamp()


def time_to_expiry_years(expiry_ts: float, as_of_ts: float) -> float:
    return max((expiry_ts - as_of_ts) / SECONDS_IN_YEAR, EPS_T)


def bs_price(option_type: str, s: float, k: float, r: float, t: float, sigma: float) -> float:
    if s <= 0 or k <= 0 or t <= 0 or sigma <= 0:
        return 0.0
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    if option_type == "CE":
        return s * norm_cdf(d1) - k * math.exp(-r * t) * norm_cdf(d2)
    return k * math.exp(-r * t) * norm_cdf(-d2) - s * norm_cdf(-d1)


def bs_vega_raw(s: float, k: float, r: float, t: float, sigma: float) -> float:
    if s <= 0 or k <= 0 or t <= 0 or sigma <= 0:
        return 0.0
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    return s * norm_pdf(d1) * sqrt_t


def option_price_bounds(option_type: str, s: float, k: float, r: float, t: float) -> tuple[float, float]:
    disc_k = k * math.exp(-r * t)
    if option_type == "CE":
        return max(0.0, s - disc_k), s
    return max(0.0, disc_k - s), disc_k


def implied_volatility(
    option_type: str,
    market_price: float,
    s: float,
    k: float,
    r: float,
    t: float,
) -> float | None:
    if market_price <= 0 or s <= 0 or k <= 0 or t <= 0:
        return None
    lower, upper = option_price_bounds(option_type, s, k, r, t)
    if market_price < lower - 1e-6 or market_price > upper + 1e-6:
        return None

    sigma = 0.30
    min_sigma, max_sigma = 1e-4, 5.0
    for _ in range(80):
        price = bs_price(option_type, s, k, r, t, sigma)
        diff = price - market_price
        if abs(diff) < 1e-5:
            return sigma
        vega = bs_vega_raw(s, k, r, t, sigma)
        if abs(vega) < 1e-8:
            break
        sigma -= diff / vega
        sigma = min(max(sigma, min_sigma), max_sigma)

    low, high = min_sigma, max_sigma
    low_diff = bs_price(option_type, s, k, r, t, low) - market_price
    high_diff = bs_price(option_type, s, k, r, t, high) - market_price
    if low_diff * high_diff > 0:
        return None

    for _ in range(120):
        mid = 0.5 * (low + high)
        mid_diff = bs_price(option_type, s, k, r, t, mid) - market_price
        if abs(mid_diff) < 1e-5:
            return mid
        if low_diff * mid_diff <= 0:
            high, high_diff = mid, mid_diff
        else:
            low, low_diff = mid, mid_diff
    return 0.5 * (low + high)


def greeks(option_type: str, s: float, k: float, r: float, t: float, sigma: float) -> dict[str, float]:
    """Black-Scholes first- and second-order greeks.

    ``vega`` is per one percentage-point of IV (``vega_raw / 100``).
    ``vanna`` is ``∂Δ/∂σ`` with ``σ`` in decimal (same for CE/PE).
    ``volga`` is ``∂vega/∂σ`` using the emitted (per-vol-pt) vega convention:
    ``volga = vega * d1 * d2 / σ``. A +0.01 move in ``σ`` changes delta by
    ``≈ vanna * 0.01`` and vega by ``≈ volga * 0.01``.
    ``charm`` is ``∂Δ/∂T`` (T in years) then scaled to **per calendar day**
    (``/ 365``), matching ``theta``. With ``q=0``, CE and PE share charm.
    ``speed`` is ``∂Γ/∂S`` (same for CE/PE under BS).
    """
    if s <= 0 or k <= 0 or t <= 0 or sigma <= 0:
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
            "vanna": 0.0,
            "volga": 0.0,
            "charm": 0.0,
            "speed": 0.0,
        }
    sqrt_t = math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    pdf_d1 = norm_pdf(d1)
    if option_type == "CE":
        delta = norm_cdf(d1)
        theta_annual = (-(s * pdf_d1 * sigma) / (2 * sqrt_t)) - (r * k * math.exp(-r * t) * norm_cdf(d2))
    else:
        delta = norm_cdf(d1) - 1.0
        theta_annual = (-(s * pdf_d1 * sigma) / (2 * sqrt_t)) + (r * k * math.exp(-r * t) * norm_cdf(-d2))
    gamma = pdf_d1 / (s * sigma * sqrt_t)
    vega = bs_vega_raw(s, k, r, t, sigma) / 100.0
    # ∂Δ/∂σ (σ decimal); identical for calls and puts under BS.
    vanna = -pdf_d1 * d2 / sigma
    # ∂vega/∂σ with vega already scaled to ₹ per vol point.
    volga = vega * d1 * d2 / sigma
    # ∂Δ/∂T (years), q=0 — identical for CE/PE; emit per calendar day.
    charm_annual = -pdf_d1 * (2.0 * r * t - d2 * sigma * sqrt_t) / (2.0 * t * sigma * sqrt_t)
    charm = charm_annual / 365.0
    # ∂Γ/∂S
    speed = -gamma / s * (d1 / (sigma * sqrt_t) + 1.0)
    return {
        "delta": delta,
        "gamma": gamma,
        "theta": theta_annual / 365.0,
        "vega": vega,
        "vanna": vanna,
        "volga": volga,
        "charm": charm,
        "speed": speed,
    }


def greek_predicted_ltp(
    anchor_ltp: float,
    g: dict[str, float],
    spot_change_points: float,
    fwd_min: float,
    iv_change_pct: float = 0.0,
) -> float:
    delta_eff = g["delta"] * spot_change_points
    gamma_eff = 0.5 * g["gamma"] * spot_change_points * spot_change_points
    theta_eff = g["theta"] * fwd_min / 1440.0
    vega_eff = g["vega"] * iv_change_pct
    return max(0.0, anchor_ltp + delta_eff + gamma_eff + theta_eff + vega_eff)


def parse_trading_date(day: str) -> date:
    return _parse_day_text(day)


def days_to_expiry(trading_day: str, expiry: str) -> int:
    return (parse_trading_date(expiry) - parse_trading_date(trading_day)).days


def minute_of_day_ist(ts: float) -> int:
    dt = datetime.fromtimestamp(ts, tz=IST)
    return dt.hour * 60 + dt.minute


def format_time_hhmm(ts: float) -> str:
    dt = datetime.fromtimestamp(ts, tz=IST)
    return dt.strftime("%H:%M")


def normalize_strike_rupees(strike_raw: float | int | None) -> float:
    if strike_raw is None:
        return 0.0
    s = float(strike_raw)
    return s / 100.0 if s >= 10000 else s
