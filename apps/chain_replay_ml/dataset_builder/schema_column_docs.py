"""Rich documentation for ml_schema_registry.json column entries.

Curated entries override auto-generated descriptions. Expand group-by-group over time.
"""

from __future__ import annotations

from typing import Any

# Keep in sync with sharp_momentum.py (DECAY_AT_3S, REF_STEP_SEC, COUNT_EPS).
_SHARP_REF_STEP_SEC = 3.0
_SHARP_COUNT_EPS = 1e-6
_SHARP_DECAY_AT_3S: dict[str, float] = {
    "1m": 0.900,
    "3m": 0.962,
    "5m": 0.977,
    "10m": 0.989,
}

# Keys: display_name, description, interpretation, example, expected_range,
# formula_ref, formula_doc, unit, nullable, expected_null_reason, learning_level, used_by

RICH_COLUMN_DOCS: dict[str, dict[str, Any]] = {
    # ── Greeks (12) ──────────────────────────────────────────────────────────
    "delta": {
        "display_name": "Delta",
        "description": "Black-Scholes Delta — sensitivity of option premium to a ₹1 move in the underlying spot.",
        "interpretation": "Calls typically have positive Delta (0 to 1); puts negative (−1 to 0). Higher absolute Delta means the option behaves more like the underlying.",
        "example": "0.62",
        "expected_range": "−1 to +1",
        "formula_ref": "bs_delta",
        "formula_doc": "Black-Scholes Delta from spot, strike, IV, time to expiry, and risk-free rate.",
        "unit": "ratio",
        "learning_level": "Beginner",
        "used_by": ["training", "audit", "prediction"],
    },
    "abs_delta": {
        "display_name": "Absolute Delta",
        "description": "Absolute value of Delta, ignoring call/put sign.",
        "interpretation": "Useful for comparing moneyness across CE and PE rows. Values near 0.5 often indicate ATM options.",
        "example": "0.62",
        "expected_range": "0 to 1",
        "formula_ref": "abs_delta",
        "formula_doc": "abs(delta)",
        "unit": "ratio",
        "learning_level": "Beginner",
    },
    "gamma": {
        "display_name": "Gamma",
        "description": "Rate of change of Delta per ₹1 move in the underlying spot.",
        "interpretation": "Highest near ATM and close to expiry. Rising Gamma means Delta can change quickly with small spot moves.",
        "example": "0.0018",
        "expected_range": "0 to ~0.05",
        "formula_ref": "bs_gamma",
        "formula_doc": "Black-Scholes Gamma.",
        "unit": "ratio",
        "learning_level": "Intermediate",
    },
    "theta": {
        "display_name": "Theta",
        "description": "Expected option premium decay from one calendar day of time passing, holding other inputs constant.",
        "interpretation": "Usually negative for long options. More negative Theta means faster time decay.",
        "example": "−8.45",
        "expected_range": "typically negative; magnitude rises near expiry",
        "formula_ref": "bs_theta",
        "formula_doc": "Black-Scholes Theta (per day).",
        "unit": "₹/day",
        "learning_level": "Beginner",
        "used_by": ["training", "audit", "prediction"],
    },
    "vega": {
        "display_name": "Vega",
        "description": "Sensitivity of option premium to a one percentage-point change in implied volatility.",
        "interpretation": "Higher Vega means the option price is more sensitive to IV shifts. Largest for ATM options with more time to expiry.",
        "example": "12.30",
        "expected_range": "0 to positive (per vol point)",
        "formula_ref": "bs_vega",
        "formula_doc": "Black-Scholes Vega.",
        "unit": "₹/vol-pt",
        "learning_level": "Intermediate",
    },
    "vanna": {
        "display_name": "Vanna",
        "description": "Cross sensitivity of Delta to implied volatility: ∂Δ/∂σ with σ in decimal (Black-Scholes).",
        "interpretation": "Positive Vanna means Delta rises when IV rises. For a +1 vol-point move (σ += 0.01), ΔDelta ≈ Vanna × 0.01. Same for CE and PE under BS.",
        "example": "0.85",
        "expected_range": "signed; largest near ATM with moderate T",
        "formula_ref": "bs_vanna",
        "formula_doc": "vanna = −n(d1)·d2 / σ",
        "unit": "Δ / σ",
        "learning_level": "Advanced",
        "used_by": ["training", "audit", "prediction"],
    },
    "volga": {
        "display_name": "Volga",
        "description": "Convexity of Vega to implied volatility: ∂vega/∂σ using the registry Vega (₹ per vol-point).",
        "interpretation": "Positive Volga means Vega rises when IV rises. For a +1 vol-point move (σ += 0.01), ΔVega ≈ Volga × 0.01. Also called vomma.",
        "example": "2.40",
        "expected_range": "typically positive away from very short T",
        "formula_ref": "bs_volga",
        "formula_doc": "volga = vega · d1 · d2 / σ (vega already /100)",
        "unit": "(₹/vol-pt) / σ",
        "learning_level": "Advanced",
        "used_by": ["training", "audit", "prediction"],
    },
    "charm": {
        "display_name": "Charm",
        "description": "Rate of change of delta with respect to calendar time (per day).",
        "interpretation": "How much delta decays as one day of time passes, holding other BS inputs fixed. Same for CE/PE when q=0.",
        "example": "-0.012",
        "expected_range": "typically small negative near ATM",
        "formula_ref": "bs_charm",
        "formula_doc": "charm = [−n(d1)·(2rT − d2·σ·√T) / (2T·σ·√T)] / 365",
        "unit": "Δ / day",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "IV unavailable or BS inputs invalid.",
        "used_by": ["training", "audit", "prediction"],
    },
    "speed": {
        "display_name": "Speed",
        "description": "Rate of change of gamma with respect to spot (∂Γ/∂S).",
        "interpretation": "How gamma changes as the underlying moves. Same for CE/PE under BS.",
        "example": "-1.2e-8",
        "expected_range": "near zero away from ATM; larger magnitude near ATM",
        "formula_ref": "bs_speed",
        "formula_doc": "speed = −Γ/S · (d1/(σ√T) + 1)",
        "unit": "Γ / ₹",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "IV unavailable or BS inputs invalid.",
        "used_by": ["training", "audit", "prediction"],
    },
    "color": {
        "display_name": "Color",
        "description": "Gamma decay with respect to calendar time (∂Γ/∂t = -∂Γ/∂T).",
        "interpretation": "Rate of change of Gamma per calendar day. Positive for ATM options.",
        "example": "0.000056",
        "expected_range": "near zero away from ATM; positive peak near ATM",
        "formula_ref": "bs_color",
        "formula_doc": "color = +[Γ / (2T)] · [1 + d1 · (2rT − d2·σ·√T) / (σ·√T)] / 365",
        "unit": "Γ / day",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "IV unavailable or BS inputs invalid.",
        "used_by": ["training", "audit", "prediction"],
    },
    "zomma": {
        "display_name": "Zomma",
        "description": "Sensitivity of Gamma to implied volatility: ∂Γ/∂σ.",
        "interpretation": "Measures how option Gamma responds to shifts in implied volatility.",
        "example": "-0.0052",
        "expected_range": "signed; typically negative near ATM",
        "formula_ref": "bs_zomma",
        "formula_doc": "zomma = Γ · (d1·d2 − 1) / σ",
        "unit": "1 / vol-pt",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "IV unavailable or BS inputs invalid.",
        "used_by": ["training", "audit", "prediction"],
    },
    "ultima": {
        "display_name": "Ultima",
        "description": "Third-order volatility sensitivity: ∂Volga/∂σ (convexity of Volga).",
        "interpretation": "Measures the acceleration of Volga with respect to implied volatility.",
        "example": "-750.0",
        "expected_range": "signed; negative peak near ATM",
        "formula_ref": "bs_ultima",
        "formula_doc": "ultima = −(Vega / σ²) · [d1·d2·(1 − d1·d2) + d1² + d2²]",
        "unit": "₹ / vol-pt³",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "IV unavailable or BS inputs invalid.",
        "used_by": ["training", "audit", "prediction"],
    },
    "svi_param_a": {
        "display_name": "SVI a (Level)",
        "description": "Gatheral Raw SVI vertical variance level parameter.",
        "interpretation": "Baseline total variance of the calibrated volatility smile.",
        "example": "0.0025",
        "expected_range": "typically small positive; bounded by non-negativity",
        "formula_ref": "svi_a",
        "formula_doc": "w(k) = a + b·[ρ·(k−m) + √((k−m)² + σ²)]",
        "unit": "variance",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "SVI calibration failed or insufficient strikes.",
        "used_by": ["training", "audit", "prediction"],
    },
    "svi_param_b": {
        "display_name": "SVI b (Slope/Angle)",
        "description": "Gatheral Raw SVI asymptote slope parameter.",
        "interpretation": "Controls wing spread and smile angle across moneyness.",
        "example": "0.08",
        "expected_range": "b >= 0; b*(1+|rho|) < 4/T",
        "formula_ref": "svi_b",
        "formula_doc": "b >= 0",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "SVI calibration failed or insufficient strikes.",
        "used_by": ["training", "audit", "prediction"],
    },
    "svi_param_rho": {
        "display_name": "SVI rho (Skew)",
        "description": "Gatheral Raw SVI smile orientation / skew rotation parameter.",
        "interpretation": "Negative for equity indices, reflecting downside put skew.",
        "example": "-0.45",
        "expected_range": "-1 to 1",
        "formula_ref": "svi_rho",
        "formula_doc": "rho in (-1, 1)",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "SVI calibration failed or insufficient strikes.",
        "used_by": ["training", "audit", "prediction"],
    },
    "svi_param_m": {
        "display_name": "SVI m (Vertex Shift)",
        "description": "Gatheral Raw SVI vertex horizontal translation parameter.",
        "interpretation": "Location of the minimum variance point in log-moneyness.",
        "example": "-0.015",
        "expected_range": "-0.3 to 0.3",
        "formula_ref": "svi_m",
        "formula_doc": "m = argmin(w(k))",
        "unit": "log-moneyness",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "SVI calibration failed or insufficient strikes.",
        "used_by": ["training", "audit", "prediction"],
    },
    "svi_param_sigma": {
        "display_name": "SVI sigma (Curvature)",
        "description": "Gatheral Raw SVI vertex curvature / smoothness parameter.",
        "interpretation": "Controls ATM smile width and curvature smoothness.",
        "example": "0.06",
        "expected_range": "sigma > 0",
        "formula_ref": "svi_sigma",
        "formula_doc": "sigma > 0",
        "unit": "width",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "SVI calibration failed or insufficient strikes.",
        "used_by": ["training", "audit", "prediction"],
    },
    "svi_calibration_rmse": {
        "display_name": "SVI Calibration RMSE",
        "description": "Root mean squared implied volatility error of the Raw SVI surface calibration.",
        "interpretation": "Calibration goodness-of-fit. Tier 1 <= 0.03, Tier 2 <= 0.06.",
        "example": "0.015",
        "expected_range": "0 to 0.10",
        "formula_ref": "svi_rmse",
        "formula_doc": "RMSE = sqrt(mean((iv_obs - iv_svi)^2))",
        "unit": "vol error",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "SVI calibration failed or insufficient strikes.",
        "used_by": ["training", "audit", "prediction"],
    },
    "sabr_param_alpha": {
        "display_name": "SABR alpha (Vol Scale)",
        "description": "Hagan SABR volatility scale parameter inverted from ATM market volatility.",
        "interpretation": "Overall initial volatility level under SABR dynamics.",
        "example": "23.5",
        "expected_range": "alpha > 0",
        "formula_ref": "sabr_alpha",
        "formula_doc": "alpha inverted from ATM cubic expansion",
        "unit": "vol scale",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "SABR calibration failed or insufficient strikes.",
        "used_by": ["training", "audit", "prediction"],
    },
    "sabr_param_rho": {
        "display_name": "SABR rho (Correlation)",
        "description": "Hagan SABR asset-volatility correlation parameter controlling skew asymmetry.",
        "interpretation": "Negative for equity indices, generating asymmetric volatility smile skew.",
        "example": "-0.35",
        "expected_range": "-1 to 1",
        "formula_ref": "sabr_rho",
        "formula_doc": "rho in (-1, 1)",
        "unit": "correlation",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "SABR calibration failed or insufficient strikes.",
        "used_by": ["training", "audit", "prediction"],
    },
    "sabr_param_nu": {
        "display_name": "SABR nu (Vol-of-Vol)",
        "description": "Hagan SABR volatility of volatility parameter controlling smile curvature.",
        "interpretation": "Higher nu produces more pronounced smile wings.",
        "example": "0.75",
        "expected_range": "nu > 0",
        "formula_ref": "sabr_nu",
        "formula_doc": "nu > 0",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "SABR calibration failed or insufficient strikes.",
        "used_by": ["training", "audit", "prediction"],
    },
    "sabr_calibration_rmse": {
        "display_name": "SABR Calibration RMSE",
        "description": "Root mean squared implied volatility error of the SABR surface calibration.",
        "interpretation": "Calibration goodness-of-fit. Tier 1 <= 0.03, Tier 2 <= 0.06.",
        "example": "0.018",
        "expected_range": "0 to 0.10",
        "formula_ref": "sabr_rmse",
        "formula_doc": "RMSE = sqrt(mean((iv_obs - iv_sabr)^2))",
        "unit": "vol error",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "SABR calibration failed or insufficient strikes.",
        "used_by": ["training", "audit", "prediction"],
    },
    "iv_skew_10d": {
        "display_name": "10-Delta IV Skew",
        "description": "Extreme tail risk asymmetry: 10-Delta Put IV minus 10-Delta Call IV.",
        "interpretation": "Captures institutional crash hedging demand in the extreme wings.",
        "example": "0.052",
        "expected_range": "-0.10 to 0.30",
        "formula_ref": "skew_10d",
        "formula_doc": "skew_10d = iv_put(10d) - iv_call(10d)",
        "unit": "vol spread",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Insufficient strike depth to bracket 10-Delta.",
        "used_by": ["training", "audit", "prediction"],
    },
    "iv_curvature_25d": {
        "display_name": "25-Delta Smile Curvature",
        "description": "Smile butterfly / tail convexity: average of 25-Delta wings minus ATM IV.",
        "interpretation": "Market expectation of jump risk / fat tails.",
        "example": "0.015",
        "expected_range": "-0.05 to 0.15",
        "formula_ref": "curv_25d",
        "formula_doc": "curv_25d = 0.5*(iv_call(25d) + iv_put(25d)) - iv_atm",
        "unit": "vol spread",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Insufficient strike depth to bracket 25-Delta.",
        "used_by": ["training", "audit", "prediction"],
    },
    "iv_term_slope_near_next": {
        "display_name": "Term Structure Slope (Near/Next)",
        "description": "Term-structure slope between near and next expiry slices.",
        "interpretation": "Positive indicates contango (rising term structure); negative indicates backwardation.",
        "example": "0.0085",
        "expected_range": "-0.50 to 0.50",
        "formula_ref": "term_slope",
        "formula_doc": "slope = (iv_next - iv_near) / (sqrt(T2) - sqrt(T1))",
        "unit": "vol / sqrt(year)",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Next expiry slice unavailable in dataset.",
        "used_by": ["training", "audit", "prediction"],
    },
    "surface_displacement_5m": {
        "display_name": "Surface Displacement (5m)",
        "description": "Backward-looking 5-minute volatility shock displacement.",
        "interpretation": "Short-term volatility repricing velocity.",
        "example": "0.004",
        "expected_range": "-0.10 to 0.10",
        "formula_ref": "disp_5m",
        "formula_doc": "disp_5m = iv_atm(t) - iv_atm(t - 5m)",
        "unit": "vol change",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Insufficient historical snapshots.",
        "used_by": ["training", "audit", "prediction"],
    },
    "surface_displacement_15m": {
        "display_name": "Surface Displacement (15m)",
        "description": "Backward-looking 15-minute volatility shock displacement.",
        "interpretation": "Medium-term volatility repricing velocity.",
        "example": "0.009",
        "expected_range": "-0.15 to 0.15",
        "formula_ref": "disp_15m",
        "formula_doc": "disp_15m = iv_atm(t) - iv_atm(t - 15m)",
        "unit": "vol change",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Insufficient historical snapshots.",
        "used_by": ["training", "audit", "prediction"],
    },
    "surface_acceleration_15m": {
        "display_name": "Surface Acceleration (15m)",
        "description": "Backward-looking second-order volatility acceleration / convexity.",
        "interpretation": "Detects accelerating volatility expansions or contractions.",
        "example": "0.0015",
        "expected_range": "-0.10 to 0.10",
        "formula_ref": "accel_15m",
        "formula_doc": "accel_15m = iv(t) - 2*iv(t - 15m) + iv(t - 30m)",
        "unit": "vol accel",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Insufficient historical snapshots.",
        "used_by": ["training", "audit", "prediction"],
    },
    "vrp_proxy_30m": {
        "display_name": "Variance Risk Premium Proxy (30m)",
        "description": "Annualized Variance Risk Premium proxy: Implied Variance minus 30-minute Realized Variance.",
        "interpretation": "Spread between forward option variance and recent realized volatility.",
        "example": "0.0022",
        "expected_range": "-0.05 to 0.10",
        "formula_ref": "vrp_30m",
        "formula_doc": "vrp_30m = iv_atm(t)^2 - RealizedVariance_30m",
        "unit": "variance spread",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Insufficient spot history to calculate realized variance.",
        "used_by": ["training", "audit", "prediction"],
    },
    "abs_delta": {
        "display_name": "Absolute Delta",
        "description": "Absolute value of Delta, ignoring call/put sign.",
        "interpretation": "Useful for comparing moneyness across CE and PE rows. Values near 0.5 often indicate ATM options.",
        "example": "0.62",
        "expected_range": "0 to 1",
        "formula_ref": "abs_delta",
        "formula_doc": "abs(delta)",
        "unit": "ratio",
        "learning_level": "Beginner",
    },
    "delta_x_spot": {
        "display_name": "Delta × Spot",
        "description": "Delta scaled by the current underlying spot price.",
        "interpretation": "Approximates the rupee exposure of the option's Delta per unit move in spot. Useful for comparing exposure across strikes.",
        "example": "14250",
        "expected_range": "depends on spot level",
        "formula_ref": "delta_x_spot",
        "formula_doc": "Delta × spot",
        "unit": "₹",
        "learning_level": "Advanced",
    },
    "gamma_x_spot": {
        "display_name": "Gamma × Spot",
        "description": "Gamma scaled by the current underlying spot price.",
        "interpretation": "Expresses Gamma exposure in spot-scaled terms for cross-strike comparison.",
        "example": "41.2",
        "expected_range": "depends on spot level",
        "formula_ref": "gamma_x_spot",
        "formula_doc": "Gamma × spot",
        "unit": "₹",
        "learning_level": "Advanced",
    },
    "delta_ltp_to_spot_ratio": {
        "display_name": "Delta LTP / Spot",
        "description": "Delta exposure normalized by spot, scaled by current option premium.",
        "interpretation": "Combines directional sensitivity (delta) with premium size relative to index level.",
        "example": "0.012",
        "expected_range": "signed, varies by moneyness",
        "formula_ref": "delta_ltp_to_spot_ratio",
        "formula_doc": "(delta × current_ltp) / spot",
        "unit": "ratio",
        "learning_level": "Advanced",
    },
    "gamma_ltp_to_spot_ratio": {
        "display_name": "Gamma LTP / Spot",
        "description": "Gamma convexity normalized by spot, scaled by current option premium.",
        "interpretation": "Premium-weighted gamma exposure relative to index — higher near ATM with meaningful premium.",
        "example": "0.0008",
        "expected_range": "non-negative for long options",
        "formula_ref": "gamma_ltp_to_spot_ratio",
        "formula_doc": "(gamma × current_ltp) / spot",
        "unit": "ratio",
        "learning_level": "Advanced",
    },
    "theta_per_min": {
        "display_name": "Theta (Per Minute)",
        "description": "Converts per-day Theta to expected premium decay per minute.",
        "interpretation": "More negative values indicate faster intraday time decay. Helpful for short-horizon sampling intervals.",
        "example": "−0.35",
        "expected_range": "typically ≤ 0",
        "formula_ref": "theta_per_min",
        "formula_doc": "theta / 1440 (per-day Theta → ₹/min; 1440 minutes per calendar day).",
        "unit": "₹/min",
        "learning_level": "Intermediate",
    },
    "vega_per_ivpt": {
        "display_name": "Vega per IV Point",
        "description": "Vega expressed per one implied-volatility point (1%).",
        "interpretation": "Shows how much premium changes for a 1% IV move — aligned with how traders quote IV.",
        "example": "12.30",
        "expected_range": "0 to positive",
        "formula_ref": "vega_per_ivpt",
        "formula_doc": "Vega per 1% IV change",
        "unit": "₹",
        "learning_level": "Advanced",
    },
    "delta_change_5m": {
        "display_name": "Delta Change (5 Minutes)",
        "description": "Change in Delta over the previous five minutes.",
        "interpretation": "Positive change on calls often means the option moved closer to ATM or spot rallied. Large swings can signal rapid repricing.",
        "example": "0.04",
        "expected_range": "−2 to +2 (typical)",
        "formula_ref": "delta_change_5m",
        "formula_doc": "Delta(now) − Delta(now − 5m)",
        "unit": "ratio",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Requires a Greek snapshot at least five minutes earlier in the replay.",
    },
    "gamma_change_5m": {
        "display_name": "Gamma Change (5 Minutes)",
        "description": "Change in Gamma over the previous five minutes.",
        "interpretation": "Rising Gamma near expiry can signal unstable Delta. Sudden spikes often occur around ATM as spot crosses strikes.",
        "example": "0.0003",
        "expected_range": "small values near 0",
        "formula_ref": "gamma_change_5m",
        "formula_doc": "Gamma(now) − Gamma(now − 5m)",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Requires a Greek snapshot at least five minutes earlier in the replay.",
    },
    "theta_change_5m": {
        "display_name": "Theta Change (5 Minutes)",
        "description": "Change in Theta over the previous five minutes.",
        "interpretation": "Theta becomes more negative as expiry approaches. A sharp change can reflect moneyness or IV repricing.",
        "example": "−1.20",
        "expected_range": "unbounded; usually negative",
        "formula_ref": "theta_change_5m",
        "formula_doc": "Theta(now) − Theta(now − 5m)",
        "unit": "₹/day",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Requires a Greek snapshot at least five minutes earlier in the replay.",
    },
    # ── Open Interest (15) ───────────────────────────────────────────────────
    "oi_change_1m": {
        "display_name": "OI Change (1 Minute)",
        "description": "Change in open interest over the previous one minute.",
        "interpretation": "Positive values indicate new positions being opened or rebuilt. Large spikes may signal fresh institutional activity.",
        "example": "12500",
        "expected_range": "any integer",
        "formula_ref": "oi_change",
        "formula_doc": "OI(now) − OI(now − 1m)",
        "unit": "contracts",
        "learning_level": "Beginner",
    },
    "oi_change_5m": {
        "display_name": "OI Change (5 Minutes)",
        "description": "Change in open interest over the previous five minutes.",
        "interpretation": "Sustained positive OI build-up with price rise may indicate long buildup; with price fall, short buildup.",
        "example": "48200",
        "expected_range": "any integer",
        "formula_ref": "oi_change",
        "formula_doc": "OI(now) − OI(now − 5m)",
        "unit": "contracts",
        "learning_level": "Beginner",
    },
    "oi_change_15m": {
        "display_name": "OI Change (15 Minutes)",
        "description": "Change in open interest over the previous fifteen minutes.",
        "interpretation": "Smoother than 1m/5m changes; useful for detecting sustained positioning trends across the chain.",
        "example": "120000",
        "expected_range": "any integer",
        "formula_ref": "oi_change",
        "formula_doc": "OI(now) − OI(now − 15m)",
        "unit": "contracts",
        "learning_level": "Intermediate",
    },
    "oi_change_pct_1m": {
        "display_name": "OI Change % (1 Minute)",
        "description": "Percentage change in open interest over the previous one minute.",
        "interpretation": "Normalizes OI change for strikes with different base OI. Values above +5% on low-OI strikes can be noisy.",
        "example": "2.4",
        "expected_range": "−100% to +∞ (unbounded above)",
        "formula_ref": "oi_change_pct",
        "formula_doc": "(OI(now) − OI(now − 1m)) / OI(now − 1m) × 100",
        "unit": "%",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Undefined when prior OI is zero.",
    },
    "oi_change_pct_5m": {
        "display_name": "OI Change % (5 Minutes)",
        "description": "Percentage change in open interest over the previous five minutes.",
        "interpretation": "High positive % on a strike may indicate aggressive new positioning at that strike.",
        "example": "8.1",
        "expected_range": "−100% to +∞",
        "formula_ref": "oi_change_pct",
        "formula_doc": "(OI(now) − OI(now − 5m)) / OI(now − 5m) × 100",
        "unit": "%",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Undefined when prior OI is zero.",
    },
    "oi_change_pct_15m": {
        "display_name": "OI Change % (15 Minutes)",
        "description": "Percentage change in open interest over the previous fifteen minutes.",
        "interpretation": "Captures slower OI trends; compare across strikes to find where positioning is concentrating.",
        "example": "15.6",
        "expected_range": "−100% to +∞",
        "formula_ref": "oi_change_pct",
        "formula_doc": "(OI(now) − OI(now − 15m)) / OI(now − 15m) × 100",
        "unit": "%",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Undefined when prior OI is zero.",
    },
    "distance_to_max_call_oi_strikes": {
        "display_name": "Distance to Max Call OI Strike",
        "description": "Strike distance (in strike steps) from the current strike to the call strike with highest open interest on the chain.",
        "interpretation": "Negative values mean max call OI is below current strike; positive above. Often acts as a magnetic strike level.",
        "example": "2",
        "expected_range": "integer strike steps",
        "formula_ref": "distance_to_max_oi_strike",
        "formula_doc": "strike − argmax_call_OI_strike (in steps)",
        "unit": "strikes",
        "learning_level": "Advanced",
    },
    "distance_to_max_put_oi_strikes": {
        "display_name": "Distance to Max Put OI Strike",
        "description": "Strike distance from the current strike to the put strike with highest open interest.",
        "interpretation": "Put OI walls often act as support zones. Distance helps locate the row relative to that wall.",
        "example": "−1",
        "expected_range": "integer strike steps",
        "formula_ref": "distance_to_max_oi_strike",
        "formula_doc": "strike − argmax_put_OI_strike (in steps)",
        "unit": "strikes",
        "learning_level": "Advanced",
    },
    "max_call_oi_pct": {
        "display_name": "Max Call OI %",
        "description": "This strike's call OI as a percentage of total call OI on the chain.",
        "interpretation": "High values mean this strike dominates call positioning — potential resistance if spot approaches from below.",
        "example": "12.5",
        "expected_range": "0 to 100",
        "formula_ref": "max_oi_pct",
        "formula_doc": "call_OI(strike) / sum(call_OI) × 100",
        "unit": "%",
        "learning_level": "Intermediate",
    },
    "max_put_oi_pct": {
        "display_name": "Max Put OI %",
        "description": "This strike's put OI as a percentage of total put OI on the chain.",
        "interpretation": "High values indicate dominant put positioning at this strike — often watched as support.",
        "example": "11.8",
        "expected_range": "0 to 100",
        "formula_ref": "max_oi_pct",
        "formula_doc": "put_OI(strike) / sum(put_OI) × 100",
        "unit": "%",
        "learning_level": "Intermediate",
    },
    "distance_to_call_build_wall": {
        "display_name": "Distance to Call Build Wall",
        "description": "Strike distance to the nearest call strike with significant recent OI build-up (call wall).",
        "interpretation": "Approaching a call build wall from below may slow rallies as writers defend the level.",
        "example": "3",
        "expected_range": "integer strike steps",
        "formula_ref": "oi_build_wall_distance",
        "formula_doc": "Distance to nearest call OI build-up strike",
        "unit": "strikes",
        "learning_level": "Advanced",
    },
    "distance_to_put_build_wall": {
        "display_name": "Distance to Put Build Wall",
        "description": "Strike distance to the nearest put strike with significant recent OI build-up (put wall).",
        "interpretation": "Put build walls below spot are often monitored as intraday support.",
        "example": "−2",
        "expected_range": "integer strike steps",
        "formula_ref": "oi_build_wall_distance",
        "formula_doc": "Distance to nearest put OI build-up strike",
        "unit": "strikes",
        "learning_level": "Advanced",
    },
    "oi_wall_bias": {
        "display_name": "OI Wall Bias",
        "description": "Signed measure of whether call or put OI walls are closer to the current strike.",
        "interpretation": "Positive bias suggests call-side positioning dominates nearby; negative suggests put-side dominance.",
        "example": "0.35",
        "expected_range": "−1 to +1",
        "formula_ref": "oi_wall_bias",
        "formula_doc": "Normalized call vs put wall proximity",
        "unit": "ratio",
        "learning_level": "Advanced",
    },
    "pinning_pressure": {
        "display_name": "Pinning Pressure",
        "description": "Estimated tendency for spot to gravitate toward high-OI strikes near expiry.",
        "interpretation": "Higher values near expiry on ATM rows may indicate pin risk — spot hugging max pain / max OI.",
        "example": "0.72",
        "expected_range": "0 to 1",
        "formula_ref": "pinning_pressure",
        "formula_doc": "OI-weighted distance to spot vs max-OI strikes",
        "unit": "ratio",
        "learning_level": "Advanced",
    },
    "oi_velocity_1m": {
        "display_name": "OI Velocity (1 Minute)",
        "description": "Rate of open-interest change per minute over the last minute.",
        "interpretation": "High positive velocity signals rapid new positioning. Compare with price direction for buildup vs unwinding context.",
        "example": "12500",
        "expected_range": "contracts per minute",
        "formula_ref": "oi_velocity",
        "formula_doc": "OI_change_1m / 1 minute",
        "unit": "contracts/min",
        "learning_level": "Intermediate",
    },
    # ── Volume (10) ────────────────────────────────────────────────────────
    "volume_change_5s": {
        "display_name": "Volume Change (5 Seconds)",
        "description": "Difference in traded option volume during the previous five seconds.",
        "interpretation": "Positive values indicate increasing trading activity. Spikes often precede short-term premium moves.",
        "example": "420",
        "expected_range": "0 to large positive integers",
        "formula_ref": "volume_change",
        "formula_doc": "volume(now) − volume(now − 5s)",
        "unit": "contracts",
        "learning_level": "Beginner",
    },
    "volume_change_15s": {
        "display_name": "Volume Change (15 Seconds)",
        "description": "Difference in traded option volume during the previous fifteen seconds.",
        "interpretation": "Smoother than 5s; useful for detecting sustained bursts of activity without tick noise.",
        "example": "1180",
        "expected_range": "0 to large positive integers",
        "formula_ref": "volume_change",
        "formula_doc": "volume(now) − volume(now − 15s)",
        "unit": "contracts",
        "learning_level": "Beginner",
    },
    "volume_change_30s": {
        "display_name": "Volume Change (30 Seconds)",
        "description": "Difference in traded option volume during the previous thirty seconds.",
        "interpretation": "Half-minute activity window aligned with common sampling intervals.",
        "example": "2450",
        "expected_range": "0 to large positive integers",
        "formula_ref": "volume_change",
        "formula_doc": "volume(now) − volume(now − 30s)",
        "unit": "contracts",
        "learning_level": "Beginner",
    },
    "volume_change_1m": {
        "display_name": "Volume Change (1 Minute)",
        "description": "Difference in traded option volume during the previous one minute.",
        "interpretation": "Rising volume with rising premium may confirm momentum; rising volume with falling premium may signal distribution.",
        "example": "5200",
        "expected_range": "0 to large positive integers",
        "formula_ref": "volume_change",
        "formula_doc": "volume(now) − volume(now − 1m)",
        "unit": "contracts",
        "learning_level": "Beginner",
    },
    "volume_change_5m": {
        "display_name": "Volume Change (5 Minutes)",
        "description": "Difference in traded option volume during the previous five minutes.",
        "interpretation": "Captures medium-term participation. Compare across strikes to find where flow is concentrating.",
        "example": "18500",
        "expected_range": "0 to large positive integers",
        "formula_ref": "volume_change",
        "formula_doc": "volume(now) − volume(now − 5m)",
        "unit": "contracts",
        "learning_level": "Intermediate",
    },
    "volume_change_15m": {
        "display_name": "Volume Change (15 Minutes)",
        "description": "Difference in traded option volume during the previous fifteen minutes.",
        "interpretation": "Longer window for trend participation; less sensitive to single large prints.",
        "example": "42000",
        "expected_range": "0 to large positive integers",
        "formula_ref": "volume_change",
        "formula_doc": "volume(now) − volume(now − 15m)",
        "unit": "contracts",
        "learning_level": "Intermediate",
    },
    "opt_volume_flow_5s": {
        "display_name": "Option Volume Flow (5 Seconds)",
        "description": "Net option volume transacted in the last five seconds (buy vs sell initiated flow where available).",
        "interpretation": "Positive flow suggests aggressive buying; negative suggests aggressive selling.",
        "example": "180",
        "expected_range": "positive or negative integers",
        "formula_ref": "opt_volume_flow",
        "formula_doc": "Signed volume flow over 5s window",
        "unit": "contracts",
        "learning_level": "Advanced",
    },
    "opt_volume_flow_15s": {
        "display_name": "Option Volume Flow (15 Seconds)",
        "description": "Net option volume flow over the previous fifteen seconds.",
        "interpretation": "Less noisy than 5s flow; useful for confirming direction of short-term pressure.",
        "example": "−320",
        "expected_range": "positive or negative integers",
        "formula_ref": "opt_volume_flow",
        "formula_doc": "Signed volume flow over 15s window",
        "unit": "contracts",
        "learning_level": "Advanced",
    },
    "opt_volume_flow_30s": {
        "display_name": "Option Volume Flow (30 Seconds)",
        "description": "Net option volume flow over the previous thirty seconds.",
        "interpretation": "Aligns with 30s sampling; sustained positive flow can support premium expansion.",
        "example": "950",
        "expected_range": "positive or negative integers",
        "formula_ref": "opt_volume_flow",
        "formula_doc": "Signed volume flow over 30s window",
        "unit": "contracts",
        "learning_level": "Advanced",
    },
    "opt_volume_flow_1m": {
        "display_name": "Option Volume Flow (1 Minute)",
        "description": "Net option volume flow over the previous one minute.",
        "interpretation": "Minute-level flow balance — compare with OI change to distinguish new positions vs churn.",
        "example": "2100",
        "expected_range": "positive or negative integers",
        "formula_ref": "opt_volume_flow",
        "formula_doc": "Signed volume flow over 1m window",
        "unit": "contracts",
        "learning_level": "Advanced",
    },
    # ── Price & Returns (remaining) ──────────────────────────────────────────
    "bid_ask_spread": {
        "display_name": "Bid-Ask Spread",
        "description": "Difference between best ask and best bid for the option, in rupees.",
        "interpretation": "Wider spreads imply higher transaction cost and less liquidity. Spikes often coincide with fast markets or illiquid strikes.",
        "example": "0.85",
        "expected_range": "0 to several rupees",
        "formula_ref": "bid_ask_spread",
        "formula_doc": "(ask − bid) in rupees from option tick",
        "unit": "₹",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Bid/ask not available on the option tick at sample time.",
    },
    "option_vwap": {
        "display_name": "Option VWAP",
        "description": "Session option VWAP from exchange day average traded price (ATP) on the option token.",
        "interpretation": "Anchor for whether current premium is rich or cheap vs the volume-weighted session average. Distances to LTP belong in the Transformation Pipeline (Interaction), not as separate Registry features.",
        "example": "142.55",
        "expected_range": "positive; near option LTP when trading is balanced",
        "formula_ref": "ticks.atp",
        "formula_doc": "option_vwap = average_traded_price (rupees) as-of sample ts",
        "unit": "₹",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "ATP missing or zero on the option tick at sample time.",
        "used_by": ["training", "audit", "prediction"],
    },
    "spot_open": {
        "display_name": "Spot Open",
        "description": "Index session open from exchange SNAP_QUOTE (token_day_meta.day_open).",
        "interpretation": "Day open level for the underlying. Distances and returns vs spot belong in the Transformation Pipeline.",
        "example": "24500.00",
        "expected_range": "near spot",
        "formula_ref": "token_day_meta.day_open",
        "formula_doc": "spot_open = index day_open (paise) / 100",
        "unit": "₹",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "token_day_meta missing for the index token on the trading day.",
        "used_by": ["training", "audit", "prediction"],
    },
    "spot_high": {
        "display_name": "Spot High",
        "description": "Index session high from exchange SNAP_QUOTE (token_day_meta.day_high).",
        "interpretation": "Running day high for the underlying as of last meta write. Distances vs spot are Pipeline only.",
        "example": "24580.00",
        "expected_range": "≥ spot_open; near spot",
        "formula_ref": "token_day_meta.day_high",
        "formula_doc": "spot_high = index day_high (paise) / 100",
        "unit": "₹",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "token_day_meta missing for the index token on the trading day.",
        "used_by": ["training", "audit", "prediction"],
    },
    "spot_low": {
        "display_name": "Spot Low",
        "description": "Index session low from exchange SNAP_QUOTE (token_day_meta.day_low).",
        "interpretation": "Running day low for the underlying as of last meta write. Distances vs spot are Pipeline only.",
        "example": "24420.00",
        "expected_range": "≤ spot_open; near spot",
        "formula_ref": "token_day_meta.day_low",
        "formula_doc": "spot_low = index day_low (paise) / 100",
        "unit": "₹",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "token_day_meta missing for the index token on the trading day.",
        "used_by": ["training", "audit", "prediction"],
    },
    "spot_prev_close": {
        "display_name": "Spot Previous Close",
        "description": "Index previous close from exchange SNAP_QUOTE (token_day_meta.prev_close).",
        "interpretation": "Prior session settlement / close reference. Gap and return packaging vs spot are Pipeline only.",
        "example": "24485.00",
        "expected_range": "near spot",
        "formula_ref": "token_day_meta.prev_close",
        "formula_doc": "spot_prev_close = index prev_close (paise) / 100",
        "unit": "₹",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "token_day_meta missing for the index token on the trading day.",
        "used_by": ["training", "audit", "prediction"],
    },
    "option_open": {
        "display_name": "Option Open",
        "description": "Option session open from exchange SNAP_QUOTE (token_day_meta.day_open).",
        "interpretation": "Day open premium for this option token. Distances vs ltp belong in the Transformation Pipeline.",
        "example": "142.00",
        "expected_range": "near option LTP",
        "formula_ref": "token_day_meta.day_open",
        "formula_doc": "option_open = option day_open (paise) / 100",
        "unit": "₹",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "token_day_meta missing for the option token on the trading day.",
        "used_by": ["training", "audit", "prediction"],
    },
    "option_high": {
        "display_name": "Option High",
        "description": "Option session high from exchange SNAP_QUOTE (token_day_meta.day_high).",
        "interpretation": "Running day high premium. Distances vs ltp are Pipeline only.",
        "example": "168.50",
        "expected_range": "≥ option_open; near option LTP",
        "formula_ref": "token_day_meta.day_high",
        "formula_doc": "option_high = option day_high (paise) / 100",
        "unit": "₹",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "token_day_meta missing for the option token on the trading day.",
        "used_by": ["training", "audit", "prediction"],
    },
    "option_low": {
        "display_name": "Option Low",
        "description": "Option session low from exchange SNAP_QUOTE (token_day_meta.day_low).",
        "interpretation": "Running day low premium. Distances vs ltp are Pipeline only.",
        "example": "118.00",
        "expected_range": "≤ option_open; near option LTP",
        "formula_ref": "token_day_meta.day_low",
        "formula_doc": "option_low = option day_low (paise) / 100",
        "unit": "₹",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "token_day_meta missing for the option token on the trading day.",
        "used_by": ["training", "audit", "prediction"],
    },
    "option_prev_close": {
        "display_name": "Option Previous Close",
        "description": "Option previous close from exchange SNAP_QUOTE (token_day_meta.prev_close).",
        "interpretation": "Prior session close premium. Gap and return packaging vs ltp are Pipeline only.",
        "example": "135.25",
        "expected_range": "near option LTP",
        "formula_ref": "token_day_meta.prev_close",
        "formula_doc": "option_prev_close = option prev_close (paise) / 100",
        "unit": "₹",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "token_day_meta missing for the option token on the trading day.",
        "used_by": ["training", "audit", "prediction"],
    },
    "futures_ltp": {
        "display_name": "Futures LTP",
        "description": "Last traded price of the current-month (front-month) NIFTY index futures contract.",
        "interpretation": "Primary futures price level. Basis and premium-to-VWAP relationships vs spot/futures_vwap belong in the Transformation Pipeline.",
        "example": "24512.50",
        "expected_range": "near spot; typically within a few dozen points",
        "formula_ref": "futures_tl.ltp",
        "formula_doc": "futures_ltp = front-month FUTIDX LTP (rupees) as-of sample ts",
        "unit": "₹",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Futures timeline unavailable for the day or no tick at/before sample ts.",
        "used_by": ["training", "audit", "prediction"],
    },
    "futures_vwap": {
        "display_name": "Futures VWAP",
        "description": "Session VWAP of the front-month NIFTY futures from exchange day average traded price (ATP).",
        "interpretation": "Session volume-weighted futures anchor. Distances and basis packaging vs spot/futures_ltp are Pipeline Interaction only.",
        "example": "24508.20",
        "expected_range": "near futures_ltp",
        "formula_ref": "futures_tl.atp",
        "formula_doc": "futures_vwap = average_traded_price (rupees) as-of sample ts on FUTIDX",
        "unit": "₹",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Futures timeline unavailable, or ATP missing/zero on the futures tick.",
        "used_by": ["training", "audit", "prediction"],
    },
    "futures_day_volume": {
        "display_name": "Futures Day Volume",
        "description": "Cumulative traded volume for the day on the front-month futures contract (as-of sample time).",
        "interpretation": "Session activity on futures. Incremental flow = Δ futures_day_volume via Pipeline Difference.",
        "example": "185420",
        "expected_range": "non-decreasing within session",
        "formula_ref": "futures_tl.day_volume",
        "formula_doc": "futures_day_volume = volume_trade_for_the_day as-of ts",
        "unit": "contracts",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Futures timeline unavailable or volume missing on the futures tick.",
        "used_by": ["training", "audit", "prediction"],
    },
    "futures_bid": {
        "display_name": "Futures Bid",
        "description": "Best bid (L1) on the front-month futures contract, in rupees.",
        "interpretation": "Top-of-book buy interest. Pair with futures_ask for spread and microprice via Pipeline / later controllers.",
        "example": "24511.00",
        "expected_range": "near futures_ltp",
        "formula_ref": "futures_tl.book L1 bid",
        "formula_doc": "futures_bid = bid_prices[0] / 100",
        "unit": "₹",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Futures timeline unavailable or L1 bid missing/zero.",
        "used_by": ["training", "audit", "prediction"],
    },
    "futures_ask": {
        "display_name": "Futures Ask",
        "description": "Best ask (L1) on the front-month futures contract, in rupees.",
        "interpretation": "Top-of-book sell interest. Pair with futures_bid for spread and microprice packaging.",
        "example": "24512.00",
        "expected_range": "near futures_ltp; typically ≥ futures_bid",
        "formula_ref": "futures_tl.book L1 ask",
        "formula_doc": "futures_ask = ask_prices[0] / 100",
        "unit": "₹",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Futures timeline unavailable or L1 ask missing/zero.",
        "used_by": ["training", "audit", "prediction"],
    },
    "futures_spread": {
        "display_name": "Futures Spread",
        "description": "L1 bid–ask spread on the front-month futures contract, in rupees.",
        "interpretation": "Futures liquidity / transaction-cost proxy. Spread-normalized futures steps belong in the Transformation Pipeline.",
        "example": "1.00",
        "expected_range": "typically a few ticks for NIFTY futures",
        "formula_ref": "futures_ask − futures_bid",
        "formula_doc": "futures_spread = max(0, futures_ask − futures_bid)",
        "unit": "₹",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Futures timeline unavailable or L1 book incomplete.",
        "used_by": ["training", "audit", "prediction"],
    },
    "futures_oi": {
        "display_name": "Futures OI",
        "description": "Open interest on the front-month NIFTY futures contract as-of sample time.",
        "interpretation": "Futures positioning stock. Changes → Pipeline Difference.",
        "example": "12543000",
        "expected_range": "positive integers",
        "formula_ref": "futures_tl.oi",
        "formula_doc": "futures_oi = open_interest as-of ts",
        "unit": "contracts",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Futures timeline unavailable or OI missing.",
    },
    "option_oi": {
        "display_name": "Option OI",
        "description": "Open interest on the current option token as-of sample time.",
        "interpretation": "Strike-level positioning. Fundamental market state for options.",
        "example": "482100",
        "expected_range": "non-negative",
        "formula_ref": "option_tl.oi",
        "formula_doc": "option_oi = open_interest as-of ts",
        "unit": "contracts",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Option timeline missing or OI absent.",
    },
    "option_day_volume": {
        "display_name": "Option Day Volume",
        "description": "Cumulative traded volume for the day on the current option token.",
        "interpretation": "Session activity at the strike. Completes symmetry with futures_day_volume.",
        "example": "15240",
        "expected_range": "non-decreasing within session",
        "formula_ref": "option_tl.day_volume",
        "formula_doc": "option_day_volume = volume_trade_for_the_day as-of ts",
        "unit": "contracts",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Option timeline missing or volume absent.",
    },
    "option_bid": {
        "display_name": "Option Bid",
        "description": "Best bid (L1) on the current option token, in rupees.",
        "interpretation": "Raw top-of-book buy. Completes symmetry with futures_bid.",
        "example": "142.50",
        "expected_range": "near option LTP",
        "formula_ref": "option_tl.book L1 bid",
        "formula_doc": "option_bid = bid_prices[0] / 100",
        "unit": "₹",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "L1 bid missing/zero.",
    },
    "option_ask": {
        "display_name": "Option Ask",
        "description": "Best ask (L1) on the current option token, in rupees.",
        "interpretation": "Raw top-of-book sell. Completes symmetry with futures_ask.",
        "example": "143.00",
        "expected_range": "near option LTP; typically ≥ option_bid",
        "formula_ref": "option_tl.book L1 ask",
        "formula_doc": "option_ask = ask_prices[0] / 100",
        "unit": "₹",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "L1 ask missing/zero.",
    },
    "ltq": {
        "display_name": "Last Traded Quantity",
        "description": "Quantity of the most recent trade print on the option token.",
        "interpretation": "Tape intensity / burst size for the option.",
        "example": "65",
        "expected_range": "positive integers when a trade printed",
        "formula_ref": "ticks.ltq",
        "formula_doc": "ltq = last_traded_quantity as-of ts",
        "unit": "contracts",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "No LTQ on timeline or zero.",
    },
    "total_buy_qty": {
        "display_name": "Total Buy Quantity",
        "description": "Exchange full-book total buy quantity (not limited to L1–L5).",
        "interpretation": "Broad buy-side depth pressure beyond top-of-book.",
        "example": "421135",
        "expected_range": "positive",
        "formula_ref": "ticks.total_buy",
        "formula_doc": "total_buy_qty = total_buy_quantity as-of ts",
        "unit": "contracts",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Field missing or zero on tick.",
    },
    "total_sell_qty": {
        "display_name": "Total Sell Quantity",
        "description": "Exchange full-book total sell quantity (not limited to L1–L5).",
        "interpretation": "Broad sell-side depth pressure beyond top-of-book.",
        "example": "398220",
        "expected_range": "positive",
        "formula_ref": "ticks.total_sell",
        "formula_doc": "total_sell_qty = total_sell_quantity as-of ts",
        "unit": "contracts",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Field missing or zero on tick.",
    },
    "atm_iv_ce": {
        "display_name": "ATM Call IV",
        "description": "Black-Scholes implied volatility of the ATM call on the loaded expiry.",
        "interpretation": "Current ATM call vol state. Computed Base chain level.",
        "example": "0.142",
        "expected_range": "typically 0.05–0.50 for NIFTY weeklies",
        "formula_ref": "bs.implied_volatility ATM CE",
        "formula_doc": "IV from ATM CE LTP vs spot",
        "unit": "decimal IV",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "ATM CE LTP/IV unavailable.",
    },
    "atm_iv_pe": {
        "display_name": "ATM Put IV",
        "description": "Black-Scholes implied volatility of the ATM put on the loaded expiry.",
        "interpretation": "Current ATM put vol state. Computed Base chain level.",
        "example": "0.148",
        "expected_range": "typically 0.05–0.50 for NIFTY weeklies",
        "formula_ref": "bs.implied_volatility ATM PE",
        "formula_doc": "IV from ATM PE LTP vs spot",
        "unit": "decimal IV",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "ATM PE LTP/IV unavailable.",
    },
    "total_call_oi": {
        "display_name": "Total Call OI",
        "description": "Sum of open interest across all loaded CE strikes for the expiry.",
        "interpretation": "Chain-wide call positioning stock.",
        "example": "18500000",
        "expected_range": "positive",
        "formula_ref": "Σ CE oi",
        "formula_doc": "total_call_oi = sum(oi_ce) over chain",
        "unit": "contracts",
        "learning_level": "Intermediate",
        "nullable": True,
    },
    "total_put_oi": {
        "display_name": "Total Put OI",
        "description": "Sum of open interest across all loaded PE strikes for the expiry.",
        "interpretation": "Chain-wide put positioning stock.",
        "example": "19200000",
        "expected_range": "positive",
        "formula_ref": "Σ PE oi",
        "formula_doc": "total_put_oi = sum(oi_pe) over chain",
        "unit": "contracts",
        "learning_level": "Intermediate",
        "nullable": True,
    },
    "oi_abs_delta_0_20_ce": {
        "display_name": "OI |Δ| 0–0.2 CE",
        "description": "Sum of CE open interest on loaded strikes with abs(BS delta) in [0.0, 0.2).",
        "interpretation": "Deep OTM / low-delta call positioning on the ATM-band chain.",
        "example": "1200000",
        "expected_range": "≥ 0",
        "formula_ref": "oi_abs_delta_0_20_ce",
        "formula_doc": "sum(oi_ce) where 0.0 ≤ |Δ| < 0.2",
        "unit": "contracts",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Expiry/spot unavailable or no priced CE in band.",
        "used_by": ["training", "audit", "prediction"],
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_oi_delta_bands.py",
            "function": "compute_oi_abs_delta_bands_at()",
        },
    },
    "oi_abs_delta_0_20_pe": {
        "display_name": "OI |Δ| 0–0.2 PE",
        "description": "Sum of PE open interest on loaded strikes with abs(BS delta) in [0.0, 0.2).",
        "interpretation": "Deep OTM / low-delta put positioning.",
        "example": "1800000",
        "expected_range": "≥ 0",
        "formula_ref": "oi_abs_delta_0_20_pe",
        "formula_doc": "sum(oi_pe) where 0.0 ≤ |Δ| < 0.2",
        "unit": "contracts",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Expiry/spot unavailable or no priced PE in band.",
        "used_by": ["training", "audit", "prediction"],
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_oi_delta_bands.py",
            "function": "compute_oi_abs_delta_bands_at()",
        },
    },
    "oi_abs_delta_20_40_ce": {
        "display_name": "OI |Δ| 0.2–0.4 CE",
        "description": "Sum of CE OI with abs(BS delta) in [0.2, 0.4).",
        "interpretation": "OTM call positioning.",
        "example": "900000",
        "expected_range": "≥ 0",
        "formula_ref": "oi_abs_delta_20_40_ce",
        "formula_doc": "sum(oi_ce) where 0.2 ≤ |Δ| < 0.4",
        "unit": "contracts",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Expiry/spot unavailable or no priced CE in band.",
        "used_by": ["training", "audit", "prediction"],
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_oi_delta_bands.py",
            "function": "compute_oi_abs_delta_bands_at()",
        },
    },
    "oi_abs_delta_20_40_pe": {
        "display_name": "OI |Δ| 0.2–0.4 PE",
        "description": "Sum of PE OI with abs(BS delta) in [0.2, 0.4).",
        "interpretation": "OTM put positioning.",
        "example": "1100000",
        "expected_range": "≥ 0",
        "formula_ref": "oi_abs_delta_20_40_pe",
        "formula_doc": "sum(oi_pe) where 0.2 ≤ |Δ| < 0.4",
        "unit": "contracts",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Expiry/spot unavailable or no priced PE in band.",
        "used_by": ["training", "audit", "prediction"],
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_oi_delta_bands.py",
            "function": "compute_oi_abs_delta_bands_at()",
        },
    },
    "oi_abs_delta_40_60_ce": {
        "display_name": "OI |Δ| 0.4–0.6 CE",
        "description": "Sum of CE OI with abs(BS delta) in [0.4, 0.6).",
        "interpretation": "Near-ATM call positioning.",
        "example": "1500000",
        "expected_range": "≥ 0",
        "formula_ref": "oi_abs_delta_40_60_ce",
        "formula_doc": "sum(oi_ce) where 0.4 ≤ |Δ| < 0.6",
        "unit": "contracts",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Expiry/spot unavailable or no priced CE in band.",
        "used_by": ["training", "audit", "prediction"],
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_oi_delta_bands.py",
            "function": "compute_oi_abs_delta_bands_at()",
        },
    },
    "oi_abs_delta_40_60_pe": {
        "display_name": "OI |Δ| 0.4–0.6 PE",
        "description": "Sum of PE OI with abs(BS delta) in [0.4, 0.6).",
        "interpretation": "Near-ATM put positioning.",
        "example": "1400000",
        "expected_range": "≥ 0",
        "formula_ref": "oi_abs_delta_40_60_pe",
        "formula_doc": "sum(oi_pe) where 0.4 ≤ |Δ| < 0.6",
        "unit": "contracts",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Expiry/spot unavailable or no priced PE in band.",
        "used_by": ["training", "audit", "prediction"],
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_oi_delta_bands.py",
            "function": "compute_oi_abs_delta_bands_at()",
        },
    },
    "oi_abs_delta_60_80_ce": {
        "display_name": "OI |Δ| 0.6–0.8 CE",
        "description": "Sum of CE OI with abs(BS delta) in [0.6, 0.8).",
        "interpretation": "ITM-leaning call positioning.",
        "example": "700000",
        "expected_range": "≥ 0",
        "formula_ref": "oi_abs_delta_60_80_ce",
        "formula_doc": "sum(oi_ce) where 0.6 ≤ |Δ| < 0.8",
        "unit": "contracts",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Expiry/spot unavailable or no priced CE in band.",
        "used_by": ["training", "audit", "prediction"],
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_oi_delta_bands.py",
            "function": "compute_oi_abs_delta_bands_at()",
        },
    },
    "oi_abs_delta_60_80_pe": {
        "display_name": "OI |Δ| 0.6–0.8 PE",
        "description": "Sum of PE OI with abs(BS delta) in [0.6, 0.8).",
        "interpretation": "ITM-leaning put positioning.",
        "example": "650000",
        "expected_range": "≥ 0",
        "formula_ref": "oi_abs_delta_60_80_pe",
        "formula_doc": "sum(oi_pe) where 0.6 ≤ |Δ| < 0.8",
        "unit": "contracts",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Expiry/spot unavailable or no priced PE in band.",
        "used_by": ["training", "audit", "prediction"],
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_oi_delta_bands.py",
            "function": "compute_oi_abs_delta_bands_at()",
        },
    },
    "oi_abs_delta_80_100_ce": {
        "display_name": "OI |Δ| 0.8–1.0 CE",
        "description": "Sum of CE OI with abs(BS delta) in [0.8, 1.0] (clamps |Δ|>1 into this band).",
        "interpretation": "Deep ITM call positioning; often thin on ATM-band chains.",
        "example": "200000",
        "expected_range": "≥ 0",
        "formula_ref": "oi_abs_delta_80_100_ce",
        "formula_doc": "sum(oi_ce) where 0.8 ≤ |Δ| ≤ 1.0",
        "unit": "contracts",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Expiry/spot unavailable or no priced CE in band.",
        "used_by": ["training", "audit", "prediction"],
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_oi_delta_bands.py",
            "function": "compute_oi_abs_delta_bands_at()",
        },
    },
    "oi_abs_delta_80_100_pe": {
        "display_name": "OI |Δ| 0.8–1.0 PE",
        "description": "Sum of PE OI with abs(BS delta) in [0.8, 1.0] (clamps |Δ|>1 into this band).",
        "interpretation": "Deep ITM put positioning; often thin on ATM-band chains.",
        "example": "180000",
        "expected_range": "≥ 0",
        "formula_ref": "oi_abs_delta_80_100_pe",
        "formula_doc": "sum(oi_pe) where 0.8 ≤ |Δ| ≤ 1.0",
        "unit": "contracts",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Expiry/spot unavailable or no priced PE in band.",
        "used_by": ["training", "audit", "prediction"],
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_oi_delta_bands.py",
            "function": "compute_oi_abs_delta_bands_at()",
        },
    },
    "total_ce_volume": {
        "display_name": "Total CE Day Volume",
        "description": "Sum of day cumulative volume across loaded CE strikes.",
        "interpretation": "Chain-wide call session activity.",
        "example": "4200000",
        "expected_range": "non-decreasing within session",
        "formula_ref": "Σ CE day_volume",
        "formula_doc": "total_ce_volume = sum(volume_ce) over chain",
        "unit": "contracts",
        "learning_level": "Intermediate",
        "nullable": True,
    },
    "total_pe_volume": {
        "display_name": "Total PE Day Volume",
        "description": "Sum of day cumulative volume across loaded PE strikes.",
        "interpretation": "Chain-wide put session activity.",
        "example": "3900000",
        "expected_range": "non-decreasing within session",
        "formula_ref": "Σ PE day_volume",
        "formula_doc": "total_pe_volume = sum(volume_pe) over chain",
        "unit": "contracts",
        "learning_level": "Intermediate",
        "nullable": True,
    },
    "otm_ce_volume": {
        "display_name": "OTM CE Day Volume",
        "description": (
            "Sum of day cumulative volume on CE strikes strictly above spot "
            "(OTM calls; ATM excluded)."
        ),
        "interpretation": "Upside-call session activity on the loaded chain.",
        "example": "2100000",
        "expected_range": "≥ 0; ≤ total_ce_volume",
        "formula_ref": "otm_ce_volume",
        "formula_doc": "sum(volume_ce) for strikes with K > spot",
        "unit": "contracts",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "No loaded CE strikes or spot unavailable.",
        "used_by": ["training", "audit", "prediction"],
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_maps.py",
            "function": "build_chain_maps()",
        },
    },
    "otm_pe_volume": {
        "display_name": "OTM PE Day Volume",
        "description": (
            "Sum of day cumulative volume on PE strikes strictly below spot "
            "(OTM puts; ATM excluded)."
        ),
        "interpretation": "Downside-put session activity on the loaded chain.",
        "example": "2500000",
        "expected_range": "≥ 0; ≤ total_pe_volume",
        "formula_ref": "otm_pe_volume",
        "formula_doc": "sum(volume_pe) for strikes with K < spot",
        "unit": "contracts",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "No loaded PE strikes or spot unavailable.",
        "used_by": ["training", "audit", "prediction"],
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_maps.py",
            "function": "build_chain_maps()",
        },
    },
    "otm_pcr_volume": {
        "display_name": "OTM PCR (Volume)",
        "description": "OTM put volume / OTM call volume: otm_pe_volume / otm_ce_volume.",
        "interpretation": (
            "Volume-side put/call pressure away from ATM. "
            ">1 ⇒ more OTM put activity than OTM call on the loaded chain."
        ),
        "example": "1.19",
        "expected_range": "typically 0.2–5",
        "formula_ref": "otm_pcr_volume",
        "formula_doc": "otm_pe_volume / otm_ce_volume",
        "unit": "ratio",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "otm_ce_volume is 0 or unavailable.",
        "used_by": ["training", "audit", "prediction"],
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_maps.py",
            "function": "build_chain_maps()",
        },
    },
    "bs_reiv_pred": {
        "display_name": "BS Roll-IV Predicted Price",
        "description": "Black-Scholes theoretical premium using the session roll IV anchor and current spot.",
        "interpretation": "Baseline fair-value estimate from the re-anchor model. Compare to actual LTP to see mispricing vs the roll anchor.",
        "example": "142.30",
        "expected_range": "0 to spot-level",
        "formula_ref": "bs_reiv_pred",
        "formula_doc": "BS price(option_type, spot, strike, r, T, roll_iv)",
        "unit": "₹",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Roll IV anchor not yet initialized or inputs missing.",
    },
    "dgt_reiv_pred": {
        "display_name": "Greek Taylor Predicted Price",
        "description": "Roll-anchor LTP adjusted by Delta/Gamma/Theta for spot move and time elapsed since last roll.",
        "interpretation": "Short-horizon fair value from local Greek expansion. Useful when spot moves quickly between rolls.",
        "example": "143.10",
        "expected_range": "0 to spot-level",
        "formula_ref": "dgt_reiv_pred",
        "formula_doc": "roll_ltp + Δ·Δspot + ½Γ·Δspot² + Θ·Δt",
        "inspection_dependencies": ["bs_reiv_pred", "delta", "gamma", "theta", "spot_change"],
        "inspection_verify_formula": (
            "max(0, bs_reiv_pred + delta * spot_change + 0.5 * gamma * spot_change * spot_change "
            "+ theta * roll_age_min / 1440.0)"
        ),
        "unit": "₹",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Roll Greeks or anchor LTP unavailable at sample time.",
    },
    "ltp_return_5s": {
        "display_name": "LTP Return (5 Seconds)",
        "description": "Percentage return of option LTP over the previous five seconds.",
        "interpretation": "Captures very short-term premium momentum. Large positive values may signal aggressive buying.",
        "example": "1.25",
        "expected_range": "unbounded %",
        "formula_ref": "ltp_return",
        "formula_doc": "(LTP(now) − LTP(now − 5s)) / LTP(now − 5s) × 100",
        "unit": "%",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Prior LTP missing or zero in the lookback window.",
    },
    "ltp_return_15s": {
        "display_name": "LTP Return (15 Seconds)",
        "description": "Percentage return of option LTP over the previous fifteen seconds.",
        "interpretation": "Slightly smoother than 5s; aligns with common sampling cadence.",
        "example": "2.10",
        "expected_range": "unbounded %",
        "formula_ref": "ltp_return",
        "formula_doc": "(LTP(now) − LTP(now − 15s)) / LTP(now − 15s) × 100",
        "unit": "%",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Prior LTP missing or zero in the lookback window.",
    },
    "ltp_return_30s": {
        "display_name": "LTP Return (30 Seconds)",
        "description": "Percentage return of option LTP over the previous thirty seconds.",
        "interpretation": "Half-minute premium momentum; less tick-noisy than 5s/15s windows.",
        "example": "−0.85",
        "expected_range": "unbounded %",
        "formula_ref": "ltp_return",
        "formula_doc": "(LTP(now) − LTP(now − 30s)) / LTP(now − 30s) × 100",
        "unit": "%",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Prior LTP missing or zero in the lookback window.",
    },
    "ltp_return_1m": {
        "display_name": "LTP Return (1 Minute)",
        "description": "Percentage return of option LTP over the previous one minute.",
        "interpretation": "Minute-scale premium trend; often correlated with volume bursts and IV repricing.",
        "example": "3.40",
        "expected_range": "unbounded %",
        "formula_ref": "ltp_return",
        "formula_doc": "(LTP(now) − LTP(now − 1m)) / LTP(now − 1m) × 100",
        "unit": "%",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Prior LTP missing or zero in the lookback window.",
    },
    "spot_change_5s": {
        "display_name": "Spot Change % (5 Seconds)",
        "description": "Percentage change in underlying spot over the previous five seconds.",
        "interpretation": "Very short-term index move driving Delta and premium repricing across the chain.",
        "example": "0.04",
        "expected_range": "small % moves intraday",
        "formula_ref": "spot_change",
        "formula_doc": "(spot(now) − spot(now − 5s)) / spot(now − 5s) × 100",
        "unit": "%",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Prior spot tick missing in the lookback window.",
    },
    "spot_change_15s": {
        "display_name": "Spot Change % (15 Seconds)",
        "description": "Percentage change in underlying spot over the previous fifteen seconds.",
        "interpretation": "Short spot impulse; compare with LTP return to see beta/leverage of the option row.",
        "example": "0.08",
        "expected_range": "small % moves intraday",
        "formula_ref": "spot_change",
        "formula_doc": "(spot(now) − spot(now − 15s)) / spot(now − 15s) × 100",
        "unit": "%",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Prior spot tick missing in the lookback window.",
    },
    "spot_change_30s": {
        "display_name": "Spot Change % (30 Seconds)",
        "description": "Percentage change in underlying spot over the previous thirty seconds.",
        "interpretation": "Half-minute index trend context for the option row.",
        "example": "−0.12",
        "expected_range": "small % moves intraday",
        "formula_ref": "spot_change",
        "formula_doc": "(spot(now) − spot(now − 30s)) / spot(now − 30s) × 100",
        "unit": "%",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Prior spot tick missing in the lookback window.",
    },
    "spot_change_1m": {
        "display_name": "Spot Change % (1 Minute)",
        "description": "Percentage change in underlying spot over the previous one minute.",
        "interpretation": "Minute-level index direction; positive spot change often lifts call premiums and pressures puts.",
        "example": "0.22",
        "expected_range": "small % moves intraday",
        "formula_ref": "spot_change",
        "formula_doc": "(spot(now) − spot(now − 1m)) / spot(now − 1m) × 100",
        "unit": "%",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Prior spot tick missing in the lookback window.",
    },
    "ltp_change_1m": {
        "display_name": "LTP Change (1 Minute)",
        "description": "Absolute change in option LTP in rupees over the previous one minute.",
        "interpretation": "Rupee move in premium; scale depends on moneyness and IV. Easier to compare across time than percentage on deep OTM.",
        "example": "4.50",
        "expected_range": "unbounded ₹",
        "formula_ref": "ltp_change",
        "formula_doc": "LTP(now) − LTP(now − 1m)",
        "unit": "₹",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Prior LTP missing in the lookback window.",
    },
    "ltp_change_5m": {
        "display_name": "LTP Change (5 Minutes)",
        "description": "Absolute change in option LTP in rupees over the previous five minutes.",
        "interpretation": "Medium-term premium drift; useful for detecting sustained trends on a strike.",
        "example": "12.80",
        "expected_range": "unbounded ₹",
        "formula_ref": "ltp_change",
        "formula_doc": "LTP(now) − LTP(now − 5m)",
        "unit": "₹",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Prior LTP missing in the lookback window.",
    },
    "ltp_change_15m": {
        "display_name": "LTP Change (15 Minutes)",
        "description": "Absolute change in option LTP in rupees over the previous fifteen minutes.",
        "interpretation": "Longer premium trend window; smoother than 1m/5m changes.",
        "example": "−6.20",
        "expected_range": "unbounded ₹",
        "formula_ref": "ltp_change",
        "formula_doc": "LTP(now) − LTP(now − 15m)",
        "unit": "₹",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Prior LTP missing in the lookback window.",
    },
    "spot_body_pct_10s": {
        "display_name": "Spot Candle Body % (10s)",
        "description": "Percentage body of the underlying 10-second OHLC candle ending at the sample time.",
        "interpretation": "Positive body means spot closed above the 10s open (bullish micro-bar). Large bodies signal impulsive index moves.",
        "example": "0.06",
        "expected_range": "typically small %",
        "formula_ref": "spot_body_pct_10s",
        "formula_doc": "(close − open) / open × 100 over 10s OHLC",
        "unit": "%",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Insufficient spot ticks to form a 10-second OHLC bar.",
    },
    "opt_body_pct_10s": {
        "display_name": "Option Candle Body % (10s)",
        "description": "Percentage body of the option 10-second OHLC candle ending at the sample time.",
        "interpretation": "Micro-structure of premium movement. Strong positive body on rising volume may confirm short-term demand.",
        "example": "1.80",
        "expected_range": "unbounded % on thin premiums",
        "formula_ref": "opt_body_pct_10s",
        "formula_doc": "(close − open) / open × 100 over 10s OHLC",
        "unit": "%",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Insufficient option ticks to form a 10-second OHLC bar.",
    },
    # ── Implied Volatility (remaining) ───────────────────────────────────────
    "roll_iv": {
        "display_name": "Roll IV",
        "description": "Implied volatility anchor from the last re-anchor (roll) event for this contract.",
        "interpretation": "Stable reference IV between rolls. Compare current_iv to roll_iv via iv_drift_from_roll to see repricing.",
        "example": "17.25",
        "expected_range": "0 to 300+",
        "formula_ref": "roll_iv",
        "formula_doc": "IV stored at last roll / session init",
        "unit": "%",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Roll state not initialized for this option token yet.",
    },
    "iv_change_1m": {
        "display_name": "IV Change (1 Minute)",
        "description": "Absolute change in implied volatility (percentage points) over the previous one minute.",
        "interpretation": "Rising IV often lifts option premiums even if spot is flat. Sharp 1m spikes may precede event risk.",
        "example": "0.45",
        "expected_range": "unbounded vol points",
        "formula_ref": "iv_change",
        "formula_doc": "IV(now) − IV(now − 1m)",
        "unit": "%",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "IV could not be solved at current or lookback timestamp.",
    },
    "iv_change_5m": {
        "display_name": "IV Change (5 Minutes)",
        "description": "Absolute change in implied volatility (percentage points) over the previous five minutes.",
        "interpretation": "Medium-term vol trend on the strike. Sustained IV rise with flat spot suggests vol buying.",
        "example": "1.20",
        "expected_range": "unbounded vol points",
        "formula_ref": "iv_change",
        "formula_doc": "IV(now) − IV(now − 5m)",
        "unit": "%",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "IV could not be solved at current or lookback timestamp.",
    },
    "iv_change_15m": {
        "display_name": "IV Change (15 Minutes)",
        "description": "Absolute change in implied volatility (percentage points) over the previous fifteen minutes.",
        "interpretation": "Smoother vol regime signal than 1m/5m; useful for detecting session-wide vol expansion or compression.",
        "example": "−0.80",
        "expected_range": "unbounded vol points",
        "formula_ref": "iv_change",
        "formula_doc": "IV(now) − IV(now − 15m)",
        "unit": "%",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "IV could not be solved at current or lookback timestamp.",
    },
    "iv_drift_from_roll": {
        "display_name": "IV Drift from Roll",
        "description": "Percentage drift of current IV relative to the roll IV anchor.",
        "interpretation": "Positive drift means IV has risen since the last roll; negative means compression vs anchor.",
        "example": "3.5",
        "expected_range": "unbounded %",
        "formula_ref": "iv_drift_from_roll",
        "formula_doc": "(actual_iv − roll_iv) / roll_iv × 100",
        "unit": "%",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Roll IV or current IV unavailable.",
    },
    "iv_pct_change_1m": {
        "display_name": "IV % Change (1 Minute)",
        "description": "Percentage change in implied volatility over the previous one minute.",
        "interpretation": "Relative vol move; +10% on a 20 IV strike means IV moved to ~22.",
        "example": "2.8",
        "expected_range": "unbounded %",
        "formula_ref": "iv_pct_change",
        "formula_doc": "(IV(now) − IV(now − 1m)) / IV(now − 1m) × 100",
        "unit": "%",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Prior IV is zero or IV unavailable at lookback.",
    },
    "iv_zscore_30m": {
        "display_name": "IV Z-Score (30 Minutes)",
        "description": "Z-score of current IV vs the rolling 30-minute IV distribution for this contract.",
        "interpretation": "Values above +2 suggest unusually high IV vs recent history; below −2 unusually low.",
        "example": "1.45",
        "expected_range": "typically −3 to +3",
        "formula_ref": "iv_zscore_30m",
        "formula_doc": "(IV − mean_30m) / std_30m",
        "unit": "z-score",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Fewer than 600 IV samples since last token gap (30-minute calendar window at 3s grid).",
    },
    "iv_zscore_1m": {
        "display_name": "IV Z-Score (1 Minute)",
        "description": "Z-score of current IV vs the rolling 1-minute IV distribution for this contract.",
        "formula_ref": "iv_zscore_1m",
        "formula_doc": "(IV − mean_1m) / std_1m",
        "unit": "z-score",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Fewer than 20 IV samples since last token gap (1-minute calendar window at 3s grid).",
    },
    "iv_zscore_5m": {
        "display_name": "IV Z-Score (5 Minutes)",
        "description": "Z-score of current IV vs the rolling 5-minute IV distribution for this contract.",
        "formula_ref": "iv_zscore_5m",
        "formula_doc": "(IV − mean_5m) / std_5m",
        "unit": "z-score",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Fewer than 100 IV samples since last token gap (5-minute calendar window at 3s grid).",
    },
    "iv_zscore_15m": {
        "display_name": "IV Z-Score (15 Minutes)",
        "description": "Z-score of current IV vs the rolling 15-minute IV distribution for this contract.",
        "formula_ref": "iv_zscore_15m",
        "formula_doc": "(IV − mean_15m) / std_15m",
        "unit": "z-score",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Fewer than 300 IV samples since last token gap (15-minute calendar window at 3s grid).",
    },
    "iv_vs_atm": {
        "display_name": "IV vs ATM",
        "description": "Percentage premium or discount of this strike's IV vs ATM implied volatility.",
        "interpretation": "Positive means richer vol than ATM (common on wings); negative means cheaper vol than ATM.",
        "example": "4.2",
        "expected_range": "unbounded %",
        "formula_ref": "iv_vs_atm",
        "formula_doc": "(IV_strike − IV_atm) / IV_atm × 100",
        "unit": "%",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "ATM IV could not be computed from ATM call/put at sample time.",
    },
    # ── Trend & Momentum (14) ────────────────────────────────────────────────
    "spot_vs_ema20_pct": {
        "display_name": "Spot vs EMA20 %",
        "description": "Percentage distance of spot from the 20-period EMA on the index.",
        "interpretation": "Positive values mean spot is above EMA20 (bullish trend context); negative below (bearish).",
        "example": "0.35",
        "expected_range": "typically −2% to +2% intraday",
        "formula_ref": "spot_vs_ema20_pct",
        "formula_doc": "(spot − EMA20) / EMA20 × 100",
        "unit": "%",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "EMA20 not yet available (warmup period).",
    },
    "ema_spread_pct": {
        "display_name": "EMA9–EMA20 Spread %",
        "description": "Percentage spread between EMA9 and EMA20 on the index.",
        "interpretation": "Positive spread (EMA9 > EMA20) indicates short-term bullish alignment; negative indicates bearish.",
        "example": "0.12",
        "expected_range": "small %",
        "formula_ref": "ema_spread_pct",
        "formula_doc": "(EMA9 − EMA20) / EMA20 × 100",
        "unit": "%",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "EMA9 or EMA20 not yet available.",
    },
    "ema9_slope": {
        "display_name": "EMA9 Slope",
        "description": "Percentage change in EMA9 over the previous one minute.",
        "interpretation": "Rising slope confirms strengthening short-term trend; falling slope suggests momentum loss.",
        "example": "0.05",
        "expected_range": "small %",
        "formula_ref": "ema9_slope",
        "formula_doc": "(EMA9(now) − EMA9(now − 1m)) / EMA9(now − 1m) × 100",
        "unit": "%",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "EMA9 history insufficient for 1-minute slope.",
    },
    "ema9_gt_ema20": {
        "display_name": "EMA9 > EMA20",
        "description": "Binary flag: 1 when EMA9 is above EMA20, else 0.",
        "interpretation": "Simple trend filter aligned with golden-cross style logic on the index.",
        "example": "1",
        "expected_range": "0 or 1",
        "formula_ref": "ema9_gt_ema20",
        "formula_doc": "1 if EMA9 > EMA20 else 0",
        "unit": "flag",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "EMA9 or EMA20 not yet available.",
    },
    "ema_spread_vs_spot_pct": {
        "display_name": "EMA Spread vs Spot %",
        "description": "EMA9–EMA20 spread expressed as a percentage of spot.",
        "interpretation": "Normalizes trend strength to index level; larger magnitude means stronger short vs medium trend separation.",
        "example": "0.08",
        "expected_range": "small %",
        "formula_ref": "ema_spread_vs_spot_pct",
        "formula_doc": "(EMA9 − EMA20) / spot × 100",
        "unit": "%",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "EMA or spot unavailable.",
    },
    "time_since_cross_min": {
        "display_name": "Time Since EMA Cross (Minutes)",
        "description": "Minutes elapsed since the last EMA9/EMA20 cross on the index.",
        "interpretation": "Fresh crosses (low minutes) often coincide with trend shifts; stale crosses mean established regime.",
        "example": "18.5",
        "expected_range": "0 to session length",
        "formula_ref": "time_since_cross_min",
        "formula_doc": "Minutes since last EMA9/EMA20 crossover",
        "unit": "minutes",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "No EMA cross detected yet in the session.",
    },
    "cross_age_decay": {
        "display_name": "Cross Age Decay",
        "description": "Exponential decay weight based on minutes since the last EMA cross.",
        "interpretation": "Near 1.0 right after a cross; decays toward 0 as the cross ages. Emphasizes recent regime changes.",
        "example": "0.55",
        "expected_range": "0 to 1",
        "formula_ref": "cross_age_decay",
        "formula_doc": "exp(−time_since_cross_min / 30)",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "No EMA cross timestamp available.",
    },
    "price_dist_from_cross_pct": {
        "display_name": "Price Distance from Cross %",
        "description": "Percentage move in spot since the last EMA9/EMA20 cross.",
        "interpretation": "Measures how far price has traveled in the current trend leg since the cross event.",
        "example": "0.42",
        "expected_range": "unbounded %",
        "formula_ref": "price_dist_from_cross_pct",
        "formula_doc": "Spot move % since last EMA cross",
        "unit": "%",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Cross price reference unavailable.",
    },
    "spot_rv_5m": {
        "display_name": "Spot Realized Vol (5m)",
        "description": "Realized volatility of spot returns over the previous five minutes.",
        "interpretation": "Higher RV means the index has been moving more; compare to IV for vol risk premium context.",
        "example": "0.18",
        "expected_range": "0 to positive",
        "formula_ref": "spot_rv_5m",
        "formula_doc": "Std dev of 10s spot returns over 5m window",
        "unit": "%",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Fewer than five spot samples in the 5-minute window.",
    },
    "spot_rv_10m": {
        "display_name": "Spot Realized Vol (10m)",
        "description": "Realized volatility of spot returns over the previous ten minutes.",
        "interpretation": "Smoother realized vol baseline than 5m; useful for detecting vol regime shifts.",
        "example": "0.15",
        "expected_range": "0 to positive",
        "formula_ref": "spot_rv_10m",
        "formula_doc": "Std dev of 10s spot returns over 10m window",
        "unit": "%",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Fewer than five spot samples in the 10-minute window.",
    },
    "spot_rv_ratio": {
        "display_name": "Spot RV Ratio (5m/10m)",
        "description": "Ratio of 5-minute to 10-minute spot realized volatility.",
        "interpretation": "Values above 1 mean vol is accelerating recently; below 1 mean vol is compressing.",
        "example": "1.20",
        "expected_range": "0 to positive",
        "formula_ref": "spot_rv_ratio",
        "formula_doc": "spot_rv_5m / spot_rv_10m",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Either RV window not ready.",
    },
    "iv_rv_spread_5m": {
        "display_name": "IV−RV Spread (5m)",
        "description": (
            "current_iv (percent) minus spot_rv_5m. IV decimal values are scaled ×100; "
            "values already >3 are treated as percent. spot_rv is rolling std of "
            "per-sample % spot returns (not annualized)."
        ),
        "interpretation": (
            "Positive ⇒ implied vol above recent realized move intensity. "
            "Useful short-horizon richness / regime filter; not a pure vol-risk-premium."
        ),
        "example": "12.4",
        "expected_range": "signed; scale depends on IV% vs sample-return std",
        "formula_ref": "iv_rv_spread_5m",
        "formula_doc": "iv_as_percent(current_iv) − spot_rv_5m",
        "unit": "percent points (mixed scale — see formula_doc)",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "IV missing or spot_rv_5m not ready.",
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/iv_rv_spread_features.py",
            "function": "enrich_iv_rv_spread_features()",
        },
    },
    "iv_rv_spread_10m": {
        "display_name": "IV−RV Spread (10m)",
        "description": (
            "current_iv (percent) minus spot_rv_10m. Same unit convention as iv_rv_spread_5m."
        ),
        "interpretation": "Smoother IV−RV gap than the 5m window.",
        "example": "13.1",
        "expected_range": "signed; see iv_rv_spread_5m",
        "formula_ref": "iv_rv_spread_10m",
        "formula_doc": "iv_as_percent(current_iv) − spot_rv_10m",
        "unit": "percent points (mixed scale — see formula_doc)",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "IV missing or spot_rv_10m not ready.",
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/iv_rv_spread_features.py",
            "function": "enrich_iv_rv_spread_features()",
        },
    },
    "opt_rv_5m": {
        "display_name": "Option Realized Vol (5m)",
        "description": "Realized volatility of option LTP returns over the previous five minutes.",
        "interpretation": "Premium realized vol; often exceeds spot RV due to leverage and IV changes.",
        "example": "2.45",
        "expected_range": "0 to positive",
        "formula_ref": "opt_rv_5m",
        "formula_doc": "Std dev of 10s LTP returns over 5m window",
        "unit": "%",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Fewer than five LTP samples in the 5-minute window.",
    },
    "opt_rv_10m": {
        "display_name": "Option Realized Vol (10m)",
        "description": "Realized volatility of option LTP returns over the previous ten minutes.",
        "interpretation": "Longer premium vol baseline for the contract row.",
        "example": "2.10",
        "expected_range": "0 to positive",
        "formula_ref": "opt_rv_10m",
        "formula_doc": "Std dev of 10s LTP returns over 10m window",
        "unit": "%",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Fewer than five LTP samples in the 10-minute window.",
    },
    "opt_rv_ratio": {
        "display_name": "Option RV Ratio (5m/10m)",
        "description": "Ratio of 5-minute to 10-minute option realized volatility.",
        "interpretation": "Premium vol acceleration signal; spikes may precede breakout moves on the strike.",
        "example": "1.17",
        "expected_range": "0 to positive",
        "formula_ref": "opt_rv_ratio",
        "formula_doc": "opt_rv_5m / opt_rv_10m",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Either option RV window could not be computed.",
    },
    # ── Time & Expiry (8) ────────────────────────────────────────────────────
    "minutes_to_expiry": {
        "display_name": "Minutes to Expiry",
        "description": "Calendar minutes remaining until option expiry from the sample timestamp.",
        "interpretation": "Drives Theta/Gamma intensity. Values near 0 mean expiry-day or final-hour effects dominate.",
        "example": "375",
        "expected_range": "0 to weeks (in minutes)",
        "formula_ref": "minutes_to_expiry",
        "formula_doc": "(expiry_ts − sample_ts) / 60",
        "unit": "minutes",
        "learning_level": "Beginner",
    },
    "minutes_since_open": {
        "display_name": "Minutes Since Open",
        "description": "Minutes elapsed since the cash market open for the session.",
        "interpretation": "Early session (low values) often has wider spreads and vol discovery; midday stabilizes.",
        "example": "45",
        "expected_range": "0 to ~375",
        "formula_ref": "minutes_since_open",
        "formula_doc": "(sample_ts − open_ts) / 60",
        "unit": "minutes",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Session open timestamp not configured.",
    },
    "minutes_to_close": {
        "display_name": "Minutes to Close",
        "description": "Minutes remaining until the cash market close.",
        "interpretation": "Last-hour rows (low values) often see gamma/theta extremes and pinning effects.",
        "example": "120",
        "expected_range": "0 to ~375",
        "formula_ref": "minutes_to_close",
        "formula_doc": "(close_ts − sample_ts) / 60",
        "unit": "minutes",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Session close timestamp not configured.",
    },
    "is_first_hour": {
        "display_name": "Is First Hour",
        "description": "Binary flag: 1 during the first 60 minutes after market open.",
        "interpretation": "Captures opening-range behavior — often higher vol and trend establishment.",
        "example": "1",
        "expected_range": "0 or 1",
        "formula_ref": "is_first_hour",
        "formula_doc": "1 if minutes_since_open ≤ 60",
        "unit": "flag",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Session open timestamp not configured.",
    },
    "is_last_hour": {
        "display_name": "Is Last Hour",
        "description": "Binary flag: 1 during the final 60 minutes before market close.",
        "interpretation": "Expiry-week last hour often shows pinning, theta decay, and position squaring.",
        "example": "0",
        "expected_range": "0 or 1",
        "formula_ref": "is_last_hour",
        "formula_doc": "1 if minutes_to_close ≤ 60",
        "unit": "flag",
        "learning_level": "Beginner",
        "nullable": True,
        "expected_null_reason": "Session close timestamp not configured.",
    },
    "minute_of_day": {
        "display_name": "Minute of Day",
        "description": "IST minute-of-day index for the sample timestamp (e.g. 555 = 09:15).",
        "interpretation": "Cyclical time feature for intraday seasonality — open, lunch, and close patterns.",
        "example": "555",
        "expected_range": "375–930 typical NSE cash hours",
        "formula_ref": "minute_of_day",
        "formula_doc": "Hour×60 + minute in IST",
        "unit": "minute index",
        "learning_level": "Beginner",
    },
    "days_to_expiry": {
        "display_name": "Days to Expiry",
        "description": "Calendar days from trading day to expiry (0 on expiry day).",
        "interpretation": "Weekly vs monthly context; 0 triggers expiry-day microstructure effects.",
        "example": "2",
        "expected_range": "0 to ~30+",
        "formula_ref": "days_to_expiry",
        "formula_doc": "Calendar days between trading_day and expiry",
        "unit": "days",
        "learning_level": "Beginner",
    },
    "is_expiry_day": {
        "display_name": "Is Expiry Day",
        "description": "Binary flag: 1 when the trading day equals the contract expiry date.",
        "interpretation": "Expiry-day rows have accelerated theta, gamma spikes, and pinning — model separately if needed.",
        "example": "0",
        "expected_range": "0 or 1",
        "formula_ref": "is_expiry_day",
        "formula_doc": "1 if days_to_expiry == 0",
        "unit": "flag",
        "learning_level": "Beginner",
    },
    # ── Strike Position / Moneyness (remaining) ────────────────────────────────
    "distance_from_spot_pct": {
        "display_name": "Distance from Spot %",
        "description": "Percentage distance of strike from current spot.",
        "interpretation": "Positive means strike above spot (OTM for calls, ITM for puts); negative below spot.",
        "example": "1.25",
        "expected_range": "unbounded %",
        "formula_ref": "distance_from_spot_pct",
        "formula_doc": "(strike − spot) / spot × 100",
        "unit": "%",
        "learning_level": "Beginner",
    },
    "distance_from_atm_pct": {
        "display_name": "Distance from ATM %",
        "description": "Percentage distance of strike from the ATM strike.",
        "interpretation": "Near 0 means ATM row. CE rows are typically ≥ 0; PE rows ≤ 0 in the band layout.",
        "example": "0.55",
        "expected_range": "band-dependent",
        "formula_ref": "distance_from_atm_pct",
        "formula_doc": "(strike − ATM) / ATM × 100",
        "unit": "%",
        "learning_level": "Beginner",
    },
    "distance_from_atm_points": {
        "display_name": "Spot Distance from ATM (Points)",
        "description": "Absolute point distance of spot from the ATM strike (spot − ATM).",
        "interpretation": "Signed spot displacement from ATM; positive means spot above ATM strike.",
        "example": "12.50",
        "expected_range": "± several strike steps",
        "formula_ref": "distance_from_atm_points",
        "formula_doc": "spot − ATM_strike",
        "unit": "points",
        "learning_level": "Beginner",
    },
    "strike_distance_from_atm": {
        "display_name": "Strike Distance from ATM (Steps)",
        "description": "Number of strike steps between this row's strike and ATM.",
        "interpretation": "Integer moneyness grid position: 0 = ATM, +2 = two steps OTM for calls, etc.",
        "example": "2",
        "expected_range": "integers in band",
        "formula_ref": "strike_distance_from_atm",
        "formula_doc": "(strike − ATM) / strike_step",
        "unit": "strikes",
        "learning_level": "Beginner",
    },
    "moneyness": {
        "display_name": "Moneyness",
        "description": "Ratio of spot to strike (spot / strike).",
        "interpretation": "Above 1 means call ITM / put OTM; below 1 call OTM / put ITM. Strong cross-sectional driver of Delta.",
        "example": "0.998",
        "expected_range": "0.9–1.1 typical in ATM band",
        "formula_ref": "moneyness",
        "formula_doc": "spot / strike",
        "unit": "ratio",
        "learning_level": "Beginner",
    },
    "is_call": {
        "display_name": "Is Call",
        "description": "Binary flag: 1 for call (CE), 0 for put (PE).",
        "interpretation": "Separates call vs put rows for models that pool both sides; affects Delta sign and payoff shape.",
        "example": "1",
        "expected_range": "0 or 1",
        "formula_ref": "is_call",
        "formula_doc": "1 if option_type == CE else 0",
        "unit": "flag",
        "learning_level": "Beginner",
    },
    "strike_to_spot_ratio": {
        "display_name": "Strike / Spot Ratio",
        "description": "Strike divided by current spot.",
        "interpretation": "Inverse-style moneyness; values above 1 mean strike above spot.",
        "example": "1.011",
        "expected_range": "near 1.0 in ATM band",
        "formula_ref": "strike_to_spot_ratio",
        "formula_doc": "strike / spot",
        "unit": "ratio",
        "learning_level": "Beginner",
    },
    "ltp_to_spot_ratio": {
        "display_name": "LTP / Spot Ratio",
        "description": "Option premium as a fraction of spot at the current timestamp.",
        "interpretation": "Normalized premium scale (LTP ÷ spot). Higher on ATM and high-IV regimes. Base for lag and change features in this family.",
        "example": "0.0063",
        "expected_range": "0 to small fraction",
        "formula_ref": "ltp_to_spot_ratio",
        "formula_doc": "current_ltp / current_spot",
        "unit": "ratio",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "LTP missing or zero.",
    },
    "moneyness_delta_ltp_to_spot_ratio": {
        "display_name": "Moneyness × |Delta| × LTP / Spot",
        "description": "Moneyness and absolute delta combined with premium, normalized by spot.",
        "interpretation": "Captures how deep ITM/OTM positioning interacts with delta-weighted premium load on the index.",
        "example": "0.008",
        "expected_range": "non-negative",
        "formula_ref": "moneyness_delta_ltp_to_spot_ratio",
        "formula_doc": "(moneyness × abs(delta) × current_ltp) / spot",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Missing moneyness, delta, LTP, or spot.",
    },
    "weighted_spot_ema_to_ltp_ratio_x_moneyness": {
        "display_name": "Weighted Spot EMA/LTP × Moneyness",
        "description": "Weighted spot EMA-to-LTP ratio scaled by strike moneyness (spot/strike).",
        "formula_ref": "weighted_spot_ema_to_ltp_ratio_x_moneyness",
        "formula_doc": "weighted_spot_ema_to_ltp_ratio × moneyness",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/advanced_composite_features.py",
            "function": "enrich_advanced_composite_features()",
        },
    },
    "weighted_spot_ema_to_ltp_ratio_x_moneyness_x_delta": {
        "display_name": "Weighted Spot EMA/LTP × Moneyness × Delta",
        "description": "Weighted spot EMA-to-LTP ratio scaled by moneyness and delta.",
        "formula_ref": "weighted_spot_ema_to_ltp_ratio_x_moneyness_x_delta",
        "formula_doc": "weighted_spot_ema_to_ltp_ratio × moneyness × delta",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/advanced_composite_features.py",
            "function": "enrich_advanced_composite_features()",
        },
    },
    # ── ATM Straddle (11) ──────────────────────────────────────────────────────
    "atm_straddle": {
        "display_name": "ATM Straddle Price",
        "description": "Sum of ATM call and put premiums at the sample timestamp.",
        "interpretation": "Market's priced move magnitude to expiry. Rising straddle = vol expansion or event premium.",
        "example": "285.50",
        "expected_range": "0 to large ₹",
        "formula_ref": "atm_straddle",
        "formula_doc": "LTP(ATM CE) + LTP(ATM PE)",
        "unit": "₹",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "ATM call or put LTP unavailable.",
    },
    "atm_straddle_change_1m": {
        "display_name": "ATM Straddle Change (1m)",
        "description": "Absolute change in ATM straddle price over one minute.",
        "interpretation": "Fast straddle expansion often signals vol buying or impending move.",
        "example": "4.20",
        "expected_range": "unbounded ₹",
        "formula_ref": "atm_straddle_change",
        "formula_doc": "straddle(now) − straddle(now − 1m)",
        "unit": "₹",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Straddle history insufficient for 1-minute lookback.",
    },
    "atm_straddle_change_5m": {
        "display_name": "ATM Straddle Change (5m)",
        "description": "Absolute change in ATM straddle price over five minutes.",
        "interpretation": "Medium-term vol repricing on the index; compare with spot range for vol/range mismatch.",
        "example": "12.50",
        "expected_range": "unbounded ₹",
        "formula_ref": "atm_straddle_change",
        "formula_doc": "straddle(now) − straddle(now − 5m)",
        "unit": "₹",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Straddle history insufficient for 5-minute lookback.",
    },
    "atm_straddle_change_pct_1m": {
        "display_name": "ATM Straddle % Change (1m)",
        "description": "Percentage change in ATM straddle over one minute.",
        "interpretation": "Relative vol move; +5% on a 280 straddle ≈ ₹14 expansion.",
        "example": "1.5",
        "expected_range": "unbounded %",
        "formula_ref": "atm_straddle_change_pct",
        "formula_doc": "(straddle(now) − straddle(now − 1m)) / straddle(now − 1m) × 100",
        "unit": "%",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Prior straddle zero or unavailable.",
    },
    "atm_straddle_change_pct_5m": {
        "display_name": "ATM Straddle % Change (5m)",
        "description": "Percentage change in ATM straddle over five minutes.",
        "interpretation": "Session vol trend gauge independent of individual strike row.",
        "example": "3.8",
        "expected_range": "unbounded %",
        "formula_ref": "atm_straddle_change_pct",
        "formula_doc": "(straddle(now) − straddle(now − 5m)) / straddle(now − 5m) × 100",
        "unit": "%",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Prior straddle zero or unavailable.",
    },
    "atm_straddle_zscore_30m": {
        "display_name": "ATM Straddle Z-Score (30m)",
        "description": "Z-score of ATM straddle vs its 30-minute rolling distribution.",
        "interpretation": "High z-score means straddle is expensive vs recent session — vol rich regime.",
        "example": "1.8",
        "expected_range": "typically −3 to +3",
        "formula_ref": "atm_straddle_zscore_30m",
        "formula_doc": "(straddle − mean_30m) / std_30m",
        "unit": "z-score",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Insufficient straddle history in 30-minute window.",
    },
    "atm_straddle_zscore_change_5m": {
        "display_name": "ATM Straddle Z-Score Change (5m)",
        "description": "Change in ATM straddle z-score over five minutes.",
        "interpretation": "Rising z-score change means vol is becoming unusually rich vs recent baseline.",
        "example": "0.45",
        "expected_range": "unbounded",
        "formula_ref": "atm_straddle_zscore_change_5m",
        "formula_doc": "zscore(now) − zscore(now − 5m)",
        "unit": "z-score",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Z-score unavailable at current or lookback time.",
    },
    "atm_straddle_change_accel": {
        "display_name": "ATM Straddle Change Acceleration",
        "description": "Difference between 1-minute straddle change and the per-minute average of the 5-minute change.",
        "interpretation": "Positive acceleration means straddle is expanding faster than the recent 5m trend.",
        "example": "2.10",
        "expected_range": "unbounded ₹",
        "formula_ref": "atm_straddle_change_accel",
        "formula_doc": "Δ1m − (Δ5m / 5)",
        "unit": "₹",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Both 1m and 5m straddle changes required.",
    },
    "atm_straddle_slope_5m": {
        "display_name": "ATM Straddle Slope (5m)",
        "description": "Average per-minute change in ATM straddle over the last five minutes.",
        "interpretation": "Linearized vol trend; positive slope = steady straddle expansion.",
        "example": "2.50",
        "expected_range": "₹/minute",
        "formula_ref": "atm_straddle_slope_5m",
        "formula_doc": "(straddle(now) − straddle(now − 5m)) / 5",
        "unit": "₹/min",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Straddle history insufficient for 5-minute slope.",
    },
    "atm_straddle_slope_15m": {
        "display_name": "ATM Straddle Slope (15m)",
        "description": "Average per-minute change in ATM straddle over the last fifteen minutes.",
        "interpretation": "Smoother vol trend than 5m slope; detects sustained vol regimes.",
        "example": "1.20",
        "expected_range": "₹/minute",
        "formula_ref": "atm_straddle_slope_15m",
        "formula_doc": "(straddle(now) − straddle(now − 15m)) / 15",
        "unit": "₹/min",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Straddle history insufficient for 15-minute slope.",
    },
    "atm_straddle_pct_change_from_open": {
        "display_name": "ATM Straddle % Change from Open",
        "description": "Percentage change in ATM straddle since the session open straddle print.",
        "interpretation": "Session vol drift — large positive values mean vol expanded materially since open.",
        "example": "8.5",
        "expected_range": "unbounded %",
        "formula_ref": "atm_straddle_pct_change_from_open",
        "formula_doc": "(straddle(now) − straddle_open) / straddle_open × 100",
        "unit": "%",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Session-open straddle reference unavailable.",
    },
    "ce_atm6_ltp_sum": {
        "display_name": "CE ATM+5 OTM LTP Sum",
        "description": (
            "Sum of call LTP at ATM and the next five OTM call strikes "
            "(ATM, +1 OTM, …, +5 OTM)."
        ),
        "interpretation": (
            "Aggregate call-side premium on the ATM wing — captures how much call premium "
            "is stacked above spot in rupee terms."
        ),
        "example": "425.50",
        "expected_range": "positive ₹",
        "formula_ref": "ce_atm6_ltp_sum",
        "formula_doc": "CE_ATM_LTP + CE_1_OTM + … + CE_5_OTM",
        "unit": "₹",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Any of the six CE strikes missing LTP at sample time.",
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_maps.py",
            "function": "precompute_chain_maps()",
        },
    },
    "pe_atm6_ltp_sum": {
        "display_name": "PE ATM+5 OTM LTP Sum",
        "description": (
            "Sum of put LTP at ATM and the next five OTM put strikes "
            "(ATM, −1 OTM, …, −5 OTM)."
        ),
        "interpretation": (
            "Aggregate put-side premium on the ATM wing — mirror of ce_atm6_ltp_sum for puts."
        ),
        "example": "380.25",
        "expected_range": "positive ₹",
        "formula_ref": "pe_atm6_ltp_sum",
        "formula_doc": "PE_ATM_LTP + PE_1_OTM + … + PE_5_OTM",
        "unit": "₹",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Any of the six PE strikes missing LTP at sample time.",
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_maps.py",
            "function": "precompute_chain_maps()",
        },
    },
    "ce_atm6_ltp_to_spot_ratio": {
        "display_name": "CE ATM+6 LTP / Spot Ratio",
        "description": "ce_atm6_ltp_sum divided by current spot — normalized call-wing premium.",
        "interpretation": (
            "Scale-invariant call premium load relative to index level. "
            "Comparable across sessions and spot levels."
        ),
        "example": "0.0185",
        "expected_range": "small positive fraction",
        "formula_ref": "ce_atm6_ltp_to_spot_ratio",
        "formula_doc": "ce_atm6_ltp_sum / current_spot",
        "unit": "ratio",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "ce_atm6_ltp_sum or spot missing or non-positive.",
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_maps.py",
            "function": "chain_features_at()",
        },
    },
    "pe_atm6_ltp_to_spot_ratio": {
        "display_name": "PE ATM+6 LTP / Spot Ratio",
        "description": "pe_atm6_ltp_sum divided by current spot — normalized put-wing premium.",
        "interpretation": (
            "Scale-invariant put premium load relative to index. "
            "Pair with ce_atm6_ltp_to_spot_ratio for wing skew."
        ),
        "example": "0.0165",
        "expected_range": "small positive fraction",
        "formula_ref": "pe_atm6_ltp_to_spot_ratio",
        "formula_doc": "pe_atm6_ltp_sum / current_spot",
        "unit": "ratio",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "pe_atm6_ltp_sum or spot missing or non-positive.",
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_maps.py",
            "function": "chain_features_at()",
        },
    },
    "ce_pe_atm6_ltp_ratio": {
        "display_name": "CE/PE ATM+6 LTP Ratio",
        "description": "Ratio of call-wing to put-wing ATM+6 premium sums.",
        "interpretation": (
            "Values above 1 = more call premium stacked on the ATM wing than puts; below 1 = put-heavy. "
            "Summarizes relative premium balance between sides."
        ),
        "example": "1.12",
        "expected_range": "positive ratio",
        "formula_ref": "ce_pe_atm6_ltp_ratio",
        "formula_doc": "ce_atm6_ltp_sum / (pe_atm6_ltp_sum + ε)",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Either side sum unavailable.",
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_maps.py",
            "function": "chain_features_at()",
        },
    },
    "ce_minus_pe_atm6_ltp": {
        "display_name": "CE − PE ATM+6 LTP",
        "description": "Absolute difference between call-wing and put-wing ATM+6 premium sums.",
        "interpretation": (
            "Signed premium imbalance in rupees — positive when calls dominate, negative when puts dominate. "
            "Complements ce_pe_atm6_ltp_ratio with absolute magnitude."
        ),
        "example": "45.25",
        "expected_range": "signed ₹",
        "formula_ref": "ce_minus_pe_atm6_ltp",
        "formula_doc": "ce_atm6_ltp_sum − pe_atm6_ltp_sum",
        "unit": "₹",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Either side sum unavailable.",
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_maps.py",
            "function": "chain_features_at()",
        },
    },
    "ce_pe_atm6_ltp_diff_pct": {
        "display_name": "CE/PE ATM+6 LTP Diff %",
        "description": (
            "Normalized call-vs-put ATM+6 premium imbalance: "
            "difference divided by total wing premium."
        ),
        "interpretation": (
            "Bounded skew measure in (−1, +1): positive = call-heavy wing, negative = put-heavy. "
            "Scale-invariant alternative to ce_minus_pe_atm6_ltp and ce_pe_atm6_ltp_ratio."
        ),
        "example": "0.087",
        "expected_range": "−1 to +1 typical",
        "formula_ref": "ce_pe_atm6_ltp_diff_pct",
        "formula_doc": "(ce_atm6_ltp_sum − pe_atm6_ltp_sum) / (ce_atm6_ltp_sum + pe_atm6_ltp_sum + ε)",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Either side sum unavailable.",
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_maps.py",
            "function": "chain_features_at()",
        },
    },
    # ── Option Chain (7) ─────────────────────────────────────────────────────
    "chain_pcr": {
        "display_name": "Chain Put-Call Ratio (OI)",
        "description": "Total put open interest divided by total call open interest on the chain.",
        "interpretation": "PCR above 1 means more put OI than call OI — often interpreted as defensive positioning.",
        "example": "1.15",
        "expected_range": "0.5 to 2.0 typical",
        "formula_ref": "chain_pcr",
        "formula_doc": "sum(put_OI) / sum(call_OI)",
        "unit": "ratio",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Chain OI aggregation unavailable at sample time.",
    },
    "atm_pcr": {
        "display_name": "ATM Put-Call Ratio (OI)",
        "description": "Put OI divided by call OI at ATM and adjacent strikes.",
        "interpretation": "Localized sentiment near the money; less diluted than full-chain PCR.",
        "example": "1.05",
        "expected_range": "0.5 to 2.0 typical",
        "formula_ref": "atm_pcr",
        "formula_doc": "ATM-band put_OI / call_OI",
        "unit": "ratio",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "ATM OI aggregation unavailable.",
    },
    "iv_skew_atm": {
        "display_name": "IV Skew (ATM Wing)",
        "description": (
            "OTM put IV minus OTM call IV at ATM ± 5 strike steps "
            "(IV_PE(ATM−5ΔK) − IV_CE(ATM+5ΔK)). Decimal Black–Scholes σ."
        ),
        "interpretation": (
            "Positive ⇒ puts richer than calls (typical equity put skew). "
            "Chain-wide level from token.chain; lag/diff via Pipeline."
        ),
        "example": "0.025",
        "expected_range": "often slightly positive for index options",
        "formula_ref": "iv_skew_atm",
        "formula_doc": "IV_PE(ATM − 5·step) − IV_CE(ATM + 5·step)",
        "unit": "IV (decimal)",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Missing wing LTP/IV on either side.",
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_iv_skew.py",
            "function": "compute_chain_iv_skew_at()",
        },
    },
    "iv_call_put_skew": {
        "display_name": "IV Call−Put Skew (ATM)",
        "description": "ATM CE IV minus ATM PE IV (decimal σ).",
        "interpretation": "ATM side gap; complements wing skew.",
        "example": "-0.005",
        "expected_range": "near zero typical",
        "formula_ref": "iv_call_put_skew",
        "formula_doc": "IV_CE(ATM) − IV_PE(ATM)",
        "unit": "IV (decimal)",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Missing ATM CE or PE IV.",
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_iv_skew.py",
            "function": "compute_chain_iv_skew_at()",
        },
    },
    "iv_skew_25d": {
        "display_name": "IV Skew 25Δ",
        "description": (
            "25Δ put IV minus 25Δ call IV, picking strikes whose BS delta is "
            "closest to ±0.25 within [0.15, 0.35]."
        ),
        "interpretation": "Classic risk-reversal skew; null when band lacks liquid 25Δ strikes.",
        "example": "0.03",
        "expected_range": "often positive for index options",
        "formula_ref": "iv_skew_25d",
        "formula_doc": "IV(PE≈−0.25Δ) − IV(CE≈+0.25Δ)",
        "unit": "IV (decimal)",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "No CE/PE within 25Δ acceptance band or IV invert failed.",
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_iv_skew.py",
            "function": "compute_chain_iv_skew_at()",
        },
    },
    "iv_butterfly_25d": {
        "display_name": "IV Butterfly 25Δ",
        "description": (
            "Smile butterfly: average of nearest 25Δ call and put IVs minus ATM IV "
            "(mean of ATM CE/PE when both available)."
        ),
        "interpretation": (
            "Positive ⇒ wings richer than ATM (smile); negative ⇒ flatter/inverted wings. "
            "Same 25Δ strike selection as iv_skew_25d."
        ),
        "example": "0.015",
        "expected_range": "typically small positive for index smiles",
        "formula_ref": "iv_butterfly_25d",
        "formula_doc": "0.5 · (IV_25Δ_CE + IV_25Δ_PE) − ATM_IV",
        "unit": "IV (decimal)",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Missing 25Δ CE/PE IVs or ATM IV on the chain.",
        "used_by": ["training", "audit", "prediction"],
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_iv_skew.py",
            "function": "compute_chain_iv_skew_at()",
        },
    },
    "delta_w_volume_flow_1m": {
        "display_name": "Delta-Weighted Volume Flow (1m)",
        "description": (
            "Chain sum of BS delta × traded volume over the last 1 minute "
            "(Δday_volume = volume_now − volume_1m_ago) across the loaded expiry chain."
        ),
        "interpretation": (
            "Positive ⇒ net call-delta volume (bullish dealer/buyer pressure on calls); "
            "negative ⇒ put-delta volume dominance. Generalizes beyond ATM+6 path flow."
        ),
        "example": "12500",
        "expected_range": "signed; scales with chain activity",
        "formula_ref": "delta_w_volume_flow_1m",
        "formula_doc": "Σ_i delta_i × (volume_i(t) − volume_i(t−60s))",
        "unit": "delta × lots",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "No strike with both valid Δvolume and BS delta.",
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_delta_volume_flow.py",
            "function": "compute_delta_w_volume_flow_at()",
        },
    },
    "delta_w_volume_flow_5m": {
        "display_name": "Delta-Weighted Volume Flow (5m)",
        "description": (
            "Same as delta_w_volume_flow_1m over a 5-minute volume lookback."
        ),
        "interpretation": "Smoother chain delta-flow than the 1m window.",
        "example": "48000",
        "expected_range": "signed; scales with chain activity",
        "formula_ref": "delta_w_volume_flow_5m",
        "formula_doc": "Σ_i delta_i × (volume_i(t) − volume_i(t−300s))",
        "unit": "delta × lots",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "No strike with both valid Δvolume and BS delta.",
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_delta_volume_flow.py",
            "function": "compute_delta_w_volume_flow_at()",
        },
    },
    "call_gex": {
        "display_name": "Call GEX",
        "description": (
            "Sum of CE gamma×OI×spot²×0.01 over the loaded chain (1% spot-move scale, lot=1)."
        ),
        "interpretation": "Call-side gamma exposure mass. Proxy within subscribed ATM band.",
        "example": "1.2e9",
        "expected_range": "≥ 0",
        "formula_ref": "call_gex",
        "formula_doc": "Σ_CE gamma × OI × spot² × 0.01",
        "unit": "gamma×OI×spot²×0.01",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "No CE with valid OI/LTP/IV/gamma.",
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_gex.py",
            "function": "compute_chain_gex_at()",
        },
    },
    "put_gex": {
        "display_name": "Put GEX",
        "description": (
            "Sum of PE gamma×OI×spot²×0.01 over the loaded chain (positive magnitude)."
        ),
        "interpretation": "Put-side gamma exposure mass before netting.",
        "example": "1.5e9",
        "expected_range": "≥ 0",
        "formula_ref": "put_gex",
        "formula_doc": "Σ_PE gamma × OI × spot² × 0.01",
        "unit": "gamma×OI×spot²×0.01",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "No PE with valid OI/LTP/IV/gamma.",
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_gex.py",
            "function": "compute_chain_gex_at()",
        },
    },
    "net_gex": {
        "display_name": "Net GEX",
        "description": "call_gex − put_gex (puts subtract; dealer-short / risk-reversal convention).",
        "interpretation": (
            "Positive ⇒ call gamma dominates; negative ⇒ put gamma dominates. "
            "RoC via Pipeline Difference — not a separate Registry feature."
        ),
        "example": "-3.0e8",
        "expected_range": "signed",
        "formula_ref": "net_gex",
        "formula_doc": "call_gex − put_gex",
        "unit": "gamma×OI×spot²×0.01",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "No valid CE/PE gamma contributions.",
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_gex.py",
            "function": "compute_chain_gex_at()",
        },
    },
    "chain_gex": {
        "display_name": "Chain GEX (Total)",
        "description": "call_gex + put_gex — total unsigned gamma×OI mass on the loaded chain.",
        "interpretation": "Overall gamma inventory intensity; pair with net_gex for direction.",
        "example": "2.7e9",
        "expected_range": "≥ 0",
        "formula_ref": "chain_gex",
        "formula_doc": "call_gex + put_gex",
        "unit": "gamma×OI×spot²×0.01",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "No valid CE/PE gamma contributions.",
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_gex.py",
            "function": "compute_chain_gex_at()",
        },
    },
    "gamma_flip_spot": {
        "display_name": "Gamma Flip Spot",
        "description": (
            "Underlying level where cumulative net GEX (call − put contrib by strike) "
            "crosses zero when walking strikes low→high."
        ),
        "interpretation": (
            "Canonical dealer gamma flip level on the loaded chain. "
            "Distances vs spot are also Registry Computed Base (gamma_flip_distance)."
        ),
        "example": "24450.0",
        "expected_range": "near spot / ATM band",
        "formula_ref": "gamma_flip_spot",
        "formula_doc": "interpolate strike where cum(net_gex_by_strike) changes sign",
        "unit": "₹",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Fewer than two strikes with GEX, or no cumulative sign change.",
        "used_by": ["training", "audit", "prediction"],
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_gex.py",
            "function": "compute_chain_gex_at()",
        },
    },
    "gamma_flip_distance": {
        "display_name": "Gamma Flip Distance",
        "description": "Relative distance of spot from gamma_flip_spot: (spot − flip) / spot.",
        "interpretation": (
            "Positive ⇒ spot above flip level; negative ⇒ below. "
            "Current market-state metric, not a Pipeline packaging column."
        ),
        "example": "0.0024",
        "expected_range": "typically within a few percent",
        "formula_ref": "gamma_flip_distance",
        "formula_doc": "(spot − gamma_flip_spot) / spot",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "gamma_flip_spot unavailable or spot ≤ 0.",
        "used_by": ["training", "audit", "prediction"],
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_gex.py",
            "function": "compute_chain_gex_at()",
        },
    },
    "synthetic_forward_spot": {
        "display_name": "Synthetic Forward Spot",
        "description": (
            "Put–call parity forward at nearest ATM strike: K + (C − P) using CE/PE LTPs."
        ),
        "interpretation": (
            "Implied forward from the ATM straddle pair. Compare to cash spot / futures "
            "via Pipeline Diff or Interaction; Registry stores the level only."
        ),
        "example": "24512.5",
        "expected_range": "near spot / futures",
        "formula_ref": "synthetic_forward_spot",
        "formula_doc": "K_atm + LTP_CE(K) − LTP_PE(K)",
        "unit": "₹",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Missing ATM CE/PE LTP or non-positive premiums.",
        "used_by": ["training", "audit", "prediction"],
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/chain_maps.py",
            "function": "build_chain_maps() / chain_features_at()",
        },
    },
    "chain_pcr_change_5m": {
        "display_name": "Chain PCR Change (5m)",
        "description": "Absolute change in chain PCR over five minutes.",
        "interpretation": "Rising PCR may indicate put buildup or call unwinding across the chain.",
        "example": "0.04",
        "expected_range": "small absolute changes",
        "formula_ref": "chain_pcr_change_5m",
        "formula_doc": "PCR(now) − PCR(now − 5m)",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "PCR unavailable at current or 5m lookback.",
    },
    "atm_pcr_change_5m": {
        "display_name": "ATM PCR Change (5m)",
        "description": "Absolute change in ATM PCR over five minutes.",
        "interpretation": "Near-money positioning shift; more sensitive to tactical hedging than chain-wide PCR.",
        "example": "0.02",
        "expected_range": "small absolute changes",
        "formula_ref": "atm_pcr_change_5m",
        "formula_doc": "ATM_PCR(now) − ATM_PCR(now − 5m)",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "ATM PCR unavailable at current or 5m lookback.",
    },
    "spot_dist_high_5m_pct": {
        "display_name": "Spot Distance from 5m High %",
        "description": "Percentage distance of current spot from the 5-minute high.",
        "interpretation": "Near 0% means spot is at the 5m high (breakout context); large negative means pullback from high.",
        "example": "−0.15",
        "expected_range": "≤ 0 %",
        "formula_ref": "spot_dist_high_5m_pct",
        "formula_doc": "(spot − high_5m) / high_5m × 100",
        "unit": "%",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Insufficient spot ticks for 5-minute OHLC.",
    },
    "spot_dist_low_5m_pct": {
        "display_name": "Spot Distance from 5m Low %",
        "description": "Percentage distance of current spot from the 5-minute low.",
        "interpretation": "Near 0% means spot is at the 5m low; positive values mean recovery from the low.",
        "example": "0.22",
        "expected_range": "≥ 0 %",
        "formula_ref": "spot_dist_low_5m_pct",
        "formula_doc": "(spot − low_5m) / low_5m × 100",
        "unit": "%",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Insufficient spot ticks for 5-minute OHLC.",
    },
    "spot_range_pos_5m": {
        "display_name": "Spot Position in 5m Range",
        "description": "Normalized position of spot within the 5-minute high-low range (0 = low, 1 = high).",
        "interpretation": "Values near 1 mean spot is pressing the top of the recent range; near 0 at the bottom.",
        "example": "0.72",
        "expected_range": "0 to 1",
        "formula_ref": "spot_range_pos_5m",
        "formula_doc": "(spot − low_5m) / (high_5m − low_5m)",
        "unit": "ratio",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Insufficient spot ticks for 5-minute OHLC.",
    },
    # ── Previous Candles (6) ──────────────────────────────────────────────────
    "spot_body_pct_prev1": {
        "display_name": "Spot Body % (Previous 10s Bar)",
        "description": "Body percentage of the spot 10-second candle immediately before the current bar.",
        "interpretation": "Prior bar momentum context; consecutive same-sign bodies suggest micro-trend continuation.",
        "example": "0.04",
        "expected_range": "small %",
        "formula_ref": "spot_body_pct_prev1",
        "formula_doc": "Body % of spot 10s bar at t−10s to t",
        "unit": "%",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Prior 10-second spot OHLC bar unavailable.",
    },
    "spot_body_pct_prev2": {
        "display_name": "Spot Body % (2 Bars Ago)",
        "description": "Body percentage of the spot 10-second candle two bars before the current sample.",
        "interpretation": "Second-lag micro-structure for sequence patterns.",
        "example": "−0.02",
        "expected_range": "small %",
        "formula_ref": "spot_body_pct_prev2",
        "formula_doc": "Body % of spot 10s bar at t−20s to t−10s",
        "unit": "%",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "OHLC bar two steps back unavailable.",
    },
    "spot_body_pct_prev3": {
        "display_name": "Spot Body % (3 Bars Ago)",
        "description": "Body percentage of the spot 10-second candle three bars before the current sample.",
        "interpretation": "Third-lag spot candle context for short memory features.",
        "example": "0.01",
        "expected_range": "small %",
        "formula_ref": "spot_body_pct_prev3",
        "formula_doc": "Body % of spot 10s bar at t−30s to t−20s",
        "unit": "%",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "OHLC bar three steps back unavailable.",
    },
    "spot_range_pct_prev1": {
        "display_name": "Spot Range % (Previous 10s Bar)",
        "description": "High-low range as percentage of open for the previous spot 10-second candle.",
        "interpretation": "Prior bar volatility; wide range may mean continuation or mean-reversion depending on regime.",
        "example": "0.08",
        "expected_range": "small positive %",
        "formula_ref": "spot_range_pct_prev1",
        "formula_doc": "(high − low) / open × 100 for prior 10s bar",
        "unit": "%",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Prior 10-second spot OHLC bar unavailable.",
    },
    "opt_body_pct_prev1": {
        "display_name": "Option Body % (Previous 10s Bar)",
        "description": "Body percentage of the option 10-second candle immediately before the current bar.",
        "interpretation": "Prior premium impulse on this strike; pairs with spot_body_pct_prev1 for divergence signals.",
        "example": "1.20",
        "expected_range": "unbounded % on thin premiums",
        "formula_ref": "opt_body_pct_prev1",
        "formula_doc": "Body % of option 10s bar at t−10s to t",
        "unit": "%",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Prior 10-second option OHLC bar unavailable.",
    },
    "opt_range_pct_prev1": {
        "display_name": "Option Range % (Previous 10s Bar)",
        "description": "High-low range as percentage of open for the previous option 10-second candle.",
        "interpretation": "Prior bar premium volatility; spikes often follow news or spot breaks.",
        "example": "2.50",
        "expected_range": "positive %",
        "formula_ref": "opt_range_pct_prev1",
        "formula_doc": "(high − low) / open × 100 for prior 10s bar",
        "unit": "%",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Prior 10-second option OHLC bar unavailable.",
    },
    # ── Advanced (4) ─────────────────────────────────────────────────────────
    "roll_age_min": {
        "display_name": "Roll Age (Minutes)",
        "description": "Minutes elapsed since the last IV re-anchor (roll) for this contract.",
        "interpretation": "Young rolls (low age) have fresher Greek anchors; stale rolls may drift from market.",
        "example": "12.5",
        "expected_range": "0 to session length",
        "formula_ref": "roll_age_min",
        "formula_doc": "(sample_ts − roll_anchor_ts) / 60",
        "unit": "minutes",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Roll state not initialized for this option.",
    },
    "rows_since_roll": {
        "display_name": "Rows Since Roll",
        "description": "Number of dataset rows sampled since the last re-anchor for this contract.",
        "interpretation": "Discrete counterpart to roll_age_min at the build sampling interval.",
        "example": "75",
        "expected_range": "0 to large integers",
        "formula_ref": "rows_since_roll",
        "formula_doc": "Row counter since last roll event",
        "unit": "rows",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Roll state not initialized for this option.",
    },
    "opt_volume_acc_5s_1m": {
        "display_name": "Option Volume Acceleration (5s/1m)",
        "description": "Ratio of 5-second volume increment to 1-minute volume increment.",
        "interpretation": "Values above 1 mean recent volume is front-loaded in the last 5s vs the minute — burst activity.",
        "example": "1.35",
        "expected_range": "0 to positive",
        "formula_ref": "opt_volume_acc_5s_1m",
        "formula_doc": "volume_flow_5s / (volume_flow_1m + ε)",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Volume lookback data unavailable.",
    },
    "spot_vol_ratio_10s_1m": {
        "display_name": "Spot Volatility Ratio (10s/1m)",
        "description": "Ratio of the current 10-second spot range to the average 1-minute range over recent bars.",
        "interpretation": "Above 1 means the latest 10s bar is wider than typical — micro vol expansion on the index.",
        "example": "1.42",
        "expected_range": "0 to positive",
        "formula_ref": "spot_vol_ratio_10s_1m",
        "formula_doc": "range_10s / avg(range over last 6×10s bars)",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Insufficient spot OHLC history for range ratio.",
    },
    # ── Key columns outside priority groups (kept from earlier overrides) ──
    "spot": {
        "display_name": "Spot",
        "description": "Underlying index spot price at the sample timestamp.",
        "interpretation": "Underlying index price used to compute option Greeks, moneyness, and chain-wide context.",
        "example": "22950.25",
        "expected_range": "market-dependent (e.g. 18000–30000 for NIFTY)",
        "formula_ref": "spot_price",
        "formula_doc": "Index spot from replay tick",
        "unit": "₹",
        "learning_level": "Beginner",
        "used_by": ["training", "audit", "prediction"],
        "tags": ["price", "underlying"],
        "depends_on": [],
        "compute_cost": "low",
    },
    "ltp": {
        "display_name": "LTP",
        "description": "Option last traded price at the sample timestamp.",
        "interpretation": "Primary premium input for Greeks, IV, and prediction targets.",
        "example": "145.60",
        "expected_range": "0 to spot-level",
        "formula_ref": "option_ltp",
        "formula_doc": "Last traded price from option tick",
        "unit": "₹",
        "learning_level": "Beginner",
        "used_by": ["training", "audit", "prediction"],
    },
    "current_iv": {
        "display_name": "Implied Volatility",
        "description": "Implied volatility backed out from the current option premium.",
        "interpretation": "Higher IV means the market prices larger expected moves. Compare to session rank for relative richness.",
        "example": "18.5",
        "expected_range": "0 to 300+",
        "formula_ref": "implied_vol",
        "formula_doc": "IV solved from Black-Scholes using LTP",
        "unit": "%",
        "learning_level": "Beginner",
    },
    "iv_rank_session": {
        "display_name": "IV Rank (Session)",
        "description": "Current IV's percentile rank between today's session minimum and maximum IV.",
        "interpretation": "Near 100% means IV is at session highs (expensive options); near 0% at session lows.",
        "example": "72",
        "expected_range": "0 to 100",
        "formula_ref": "iv_rank_session",
        "formula_doc": "(IV − session_min) / (session_max − session_min) × 100",
        "unit": "%",
        "nullable": True,
        "expected_null_reason": "IV unavailable at this timestamp (no valid option quote for BS solve).",
        "learning_level": "Intermediate",
    },
    "strike": {
        "display_name": "Strike",
        "description": "Option strike price for this contract row.",
        "interpretation": "Used with spot to determine moneyness and distance from ATM.",
        "example": "23000",
        "expected_range": "exchange-listed strikes",
        "formula_ref": "strike_price",
        "formula_doc": "Contract strike from chain metadata",
        "unit": "₹",
        "learning_level": "Beginner",
        "used_by": ["training", "audit"],
    },
}

# LTP/spot ratio lag (Pipeline Owned) + absolute-change (still Master) family
_LTP_TO_SPOT_DOC_HORIZONS: tuple[tuple[str, str, str], ...] = (
    ("10s", "10 Seconds", "Immediate momentum — how normalized premium moved in the last 10 seconds."),
    ("30s", "30 Seconds", "Short-term momentum in premium vs spot."),
    ("1m", "1 Minute", "Very short trend in LTP/spot relationship."),
    ("3m", "3 Minutes", "Intraday micro-trend in normalized premium."),
    ("5m", "5 Minutes", "Strong short-term trend in premium scaling."),
    ("15m", "15 Minutes", "Broader market context for option premium relative to spot."),
)

# Lag docs for pipeline horizons only (10s retired — invalid at default 3s).
_LTP_TO_SPOT_PIPELINE_LAG_SUFFIXES = frozenset({"30s", "1m", "3m", "5m", "15m"})

for _suffix, _label, _change_interp in _LTP_TO_SPOT_DOC_HORIZONS:
    if _suffix in _LTP_TO_SPOT_PIPELINE_LAG_SUFFIXES:
        RICH_COLUMN_DOCS[f"ltp_to_spot_ratio_lag_{_suffix}"] = {
            "display_name": f"LTP/Spot Ratio Lag ({_label})",
            "description": (
                f"LTP/spot ratio {_label.lower()} ago — Pipeline Owned (Lag transform). "
                "Not a Master Dataset column."
            ),
            "interpretation": (
                f"Row-shift of ltp_to_spot_ratio by {_suffix} on the sample grid. "
                "Use with the matching change feature to see how fast premium is scaling vs spot."
            ),
            "example": "0.0061",
            "expected_range": "0 to small fraction",
            "formula_ref": f"ltp_to_spot_ratio_lag_{_suffix}",
            "formula_doc": f"ltp_to_spot_ratio.shift({_suffix} / sample_interval)",
            "unit": "ratio",
            "learning_level": "Intermediate",
            "nullable": True,
            "expected_null_reason": "Warmup rows before the lag offset, or missing base ratio.",
        }
    RICH_COLUMN_DOCS[f"ltp_to_spot_ratio_change_{_suffix}"] = {
        "display_name": f"LTP/Spot Ratio Change ({_label})",
        "description": (
            f"Absolute change in LTP/spot ratio vs {_label.lower()} ago "
            f"(current ratio minus lagged ratio)."
        ),
        "interpretation": _change_interp + " Positive = premium scaling up vs spot; negative = compressing.",
        "example": "0.0002",
        "expected_range": "small signed delta",
        "formula_ref": f"ltp_to_spot_ratio_change_{_suffix}",
        "formula_doc": f"ltp_to_spot_ratio − ltp_to_spot_ratio(at now − {_suffix})",
        "unit": "ratio",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "Current or lagged LTP/spot ratio unavailable.",
    }

# LTP × fractional volume change family (volume group)
_LTP_X_VOLUME_DOC_HORIZONS: tuple[tuple[str, str, str], ...] = (
    ("10s", "10 Seconds", "Immediate volume surge scaled by premium — tick-level activity burst."),
    ("30s", "30 Seconds", "Short-term volume momentum weighted by LTP."),
    ("1m", "1 Minute", "Very short participation trend in premium terms."),
    ("3m", "3 Minutes", "Intraday micro-trend in volume acceleration × premium."),
    ("5m", "5 Minutes", "Strong short-term flow signal scaled by option price."),
    ("15m", "15 Minutes", "Broader participation context weighted by current premium."),
)

for _suffix, _label, _interp in _LTP_X_VOLUME_DOC_HORIZONS:
    RICH_COLUMN_DOCS[f"ltp_x_volume_change_pct_{_suffix}"] = {
        "display_name": f"LTP × Volume Change % ({_label})",
        "description": (
            f"Current option LTP multiplied by fractional volume change over {_label.lower()}. "
            "Combines premium level with relative volume acceleration."
        ),
        "interpretation": _interp + " Positive = rising volume with meaningful premium; magnifies flow on higher-priced options.",
        "example": "12.5",
        "expected_range": "signed, scales with LTP and volume delta",
        "formula_ref": f"ltp_x_volume_change_pct_{_suffix}",
        "formula_doc": (
            f"current_ltp × (current_volume − volume_{_suffix}_ago) / (volume_{_suffix}_ago + ε)"
        ),
        "unit": "₹·ratio",
        "learning_level": "Intermediate",
        "nullable": True,
        "expected_null_reason": "LTP or volume unavailable at current or lag timestamp.",
    }

# Greek Taylor predicted price lag / change / error family (price group)
_DGT_DOC_HORIZONS: tuple[tuple[str, str, str], ...] = (
    ("10s", "10 Seconds", "Immediate drift in Greek fair value since the last roll anchor."),
    ("30s", "30 Seconds", "Short-term move in Taylor-expanded fair value."),
    ("1m", "1 Minute", "Very short trend in roll-anchor Greek prediction."),
)

for _suffix, _label, _change_interp in _DGT_DOC_HORIZONS:
    RICH_COLUMN_DOCS[f"dgt_reiv_pred_lag_{_suffix}"] = {
        "display_name": f"Greek Taylor Pred Lag ({_label})",
        "description": (
            f"dgt_reiv_pred computed {_label.lower()} ago — Greek fair value at an earlier sample."
        ),
        "interpretation": (
            f"Snapshot of dgt_reiv_pred at (now − {_suffix}). "
            "Pairs with change features to measure how fast fair value is moving."
        ),
        "example": "142.50",
        "expected_range": "0 to spot-level",
        "formula_ref": f"dgt_reiv_pred_lag_{_suffix}",
        "formula_doc": f"dgt_reiv_pred at (now − {_suffix}) from per-token roll history",
        "unit": "₹",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Insufficient per-token history before the lag window.",
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/extended_features.py",
            "function": "enrich_dataset_features()",
        },
    }
    RICH_COLUMN_DOCS[f"dgt_reiv_pred_change_{_suffix}"] = {
        "display_name": f"Greek Taylor Pred Change ({_label})",
        "description": (
            f"Absolute change in dgt_reiv_pred vs {_label.lower()} ago "
            f"(current prediction minus lagged prediction)."
        ),
        "interpretation": _change_interp + " Positive = fair value rising; negative = compressing.",
        "example": "0.85",
        "expected_range": "signed ₹ delta",
        "formula_ref": f"dgt_reiv_pred_change_{_suffix}",
        "formula_doc": f"dgt_reiv_pred − dgt_reiv_pred_lag_{_suffix}",
        "unit": "₹",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Current or lagged dgt_reiv_pred unavailable.",
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/extended_features.py",
            "function": "enrich_dataset_features()",
        },
    }

for _suffix, _label in (("10s", "10 Seconds"), ("30s", "30 Seconds")):
    RICH_COLUMN_DOCS[f"dgt_prediction_error_lag_{_suffix}"] = {
        "display_name": f"Greek Prediction Error Lag ({_label})",
        "description": (
            f"Market vs Greek fair-value gap {_label.lower()} ago: "
            f"LTP at lag minus dgt_reiv_pred at the same lag."
        ),
        "interpretation": (
            "Historical mispricing of the Taylor model — useful for mean-reversion and momentum signals."
        ),
        "example": "1.20",
        "expected_range": "signed ₹",
        "formula_ref": f"dgt_prediction_error_lag_{_suffix}",
        "formula_doc": f"ltp_{_suffix}_ago − dgt_reiv_pred_lag_{_suffix}",
        "unit": "₹",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "LTP or dgt_reiv_pred missing at the lag timestamp.",
        "implementation": {
            "module": "chain_replay_ml/dataset_builder/extended_features.py",
            "function": "enrich_dataset_features()",
        },
    }

RICH_COLUMN_DOCS["ltp_to_dgt_reiv_ratio"] = {
    "display_name": "LTP / Greek Taylor Ratio",
    "description": "Current option LTP divided by Greek Taylor predicted fair value (dgt_reiv_pred).",
    "interpretation": (
        "Values above 1.0 = market trading above model fair value; below 1.0 = below. "
        "Scale-invariant alternative to dgt_prediction_error when comparing across strikes."
    ),
    "example": "1.012",
    "expected_range": "typically near 1.0",
    "formula_ref": "ltp_to_dgt_reiv_ratio",
    "formula_doc": "current_ltp / dgt_reiv_pred",
    "inspection_dependencies": ["ltp", "dgt_reiv_pred"],
    "unit": "ratio",
    "learning_level": "Advanced",
    "nullable": True,
    "expected_null_reason": "LTP or dgt_reiv_pred missing or non-positive.",
    "implementation": {
        "module": "chain_replay_ml/dataset_builder/extended_features.py",
        "function": "enrich_dataset_features()",
    },
}

RICH_COLUMN_DOCS["ltp_to_bs_reiv_ratio"] = {
    "display_name": "LTP / BS Roll-IV Ratio",
    "description": "Current option LTP divided by Black-Scholes roll-IV predicted price (bs_reiv_pred).",
    "interpretation": (
        "Values above 1.0 = market trading above BS fair value; below 1.0 = below. "
        "Compare with ltp_to_dgt_reiv_ratio to see BS vs Greek-Taylor mispricing."
    ),
    "example": "1.008",
    "expected_range": "typically near 1.0",
    "formula_ref": "ltp_to_bs_reiv_ratio",
    "formula_doc": "current_ltp / bs_reiv_pred",
    "unit": "ratio",
    "learning_level": "Advanced",
    "nullable": True,
    "expected_null_reason": "LTP or bs_reiv_pred missing or non-positive.",
    "implementation": {
        "module": "chain_replay_ml/dataset_builder/extended_features.py",
        "function": "enrich_dataset_features()",
    },
}

RICH_COLUMN_DOCS["dgt_reiv_to_ltp_ratio"] = {
    "display_name": "Greek Taylor / LTP Ratio",
    "description": "Greek Taylor predicted fair value (dgt_reiv_pred) divided by current option LTP.",
    "interpretation": (
        "Inverse of ltp_to_dgt_reiv_ratio. Values below 1.0 = market above model fair value; "
        "above 1.0 = market below fair value. Scale-invariant fair-value anchor vs market LTP."
    ),
    "example": "0.988",
    "expected_range": "typically near 1.0",
    "formula_ref": "dgt_reiv_to_ltp_ratio",
    "formula_doc": "dgt_reiv_pred / current_ltp",
    "unit": "ratio",
    "learning_level": "Advanced",
    "nullable": True,
    "expected_null_reason": "dgt_reiv_pred or LTP missing or non-positive.",
    "implementation": {
        "module": "chain_replay_ml/dataset_builder/extended_features.py",
        "function": "enrich_dataset_features()",
    },
}

RICH_COLUMN_DOCS["bs_reiv_to_ltp_ratio"] = {
    "display_name": "BS Roll-IV / LTP Ratio",
    "description": "Black-Scholes roll-IV predicted price (bs_reiv_pred) divided by current option LTP.",
    "interpretation": (
        "Inverse of ltp_to_bs_reiv_ratio. Values below 1.0 = market above BS fair value; "
        "above 1.0 = market below BS fair value."
    ),
    "example": "0.992",
    "expected_range": "typically near 1.0",
    "formula_ref": "bs_reiv_to_ltp_ratio",
    "formula_doc": "bs_reiv_pred / current_ltp",
    "unit": "ratio",
    "learning_level": "Advanced",
    "nullable": True,
    "expected_null_reason": "bs_reiv_pred or LTP missing or non-positive.",
    "implementation": {
        "module": "chain_replay_ml/dataset_builder/extended_features.py",
        "function": "enrich_dataset_features()",
    },
}

RICH_COLUMN_DOCS["side_to_ltp_ratio"] = {
    "display_name": "Side ATM+6 Sum / LTP Ratio",
    "description": (
        "ATM+6 wing LTP sum on the contract's own side (CE or PE) divided by current option LTP."
    ),
    "interpretation": (
        "For CE rows: ce_atm6_ltp_sum / current_ltp. For PE rows: pe_atm6_ltp_sum / current_ltp. "
        "Measures how much aggregate wing premium on the same side exists relative to this contract's LTP."
    ),
    "example": "12.4",
    "expected_range": "positive, varies by strike and moneyness",
    "formula_ref": "side_to_ltp_ratio",
    "formula_doc": "ce_atm6_ltp_sum / current_ltp if CE else pe_atm6_ltp_sum / current_ltp",
    "unit": "ratio",
    "learning_level": "Advanced",
    "nullable": True,
    "expected_null_reason": "Side ATM+6 sum or LTP missing or non-positive.",
    "implementation": {
        "module": "chain_replay_ml/dataset_builder/extended_features.py",
        "function": "enrich_with_chain_maps()",
    },
}

RICH_COLUMN_DOCS["atm6_total_to_ltp_ratio"] = {
    "display_name": "ATM+6 Total Sum / LTP Ratio",
    "description": (
        "Sum of CE and PE ATM+6 wing LTP divided by current option LTP."
    ),
    "interpretation": (
        "(ce_atm6_ltp_sum + pe_atm6_ltp_sum) / current_ltp — total ATM wing premium load "
        "relative to this contract's market price."
    ),
    "example": "24.8",
    "expected_range": "positive",
    "formula_ref": "atm6_total_to_ltp_ratio",
    "formula_doc": "(ce_atm6_ltp_sum + pe_atm6_ltp_sum) / current_ltp",
    "unit": "ratio",
    "learning_level": "Advanced",
    "nullable": True,
    "expected_null_reason": "CE/PE ATM+6 sums or LTP missing or non-positive.",
    "implementation": {
        "module": "chain_replay_ml/dataset_builder/extended_features.py",
        "function": "enrich_with_chain_maps()",
    },
}

_EMA_TO_LTP_IMPL = {
    "module": "chain_replay_ml/dataset_builder/extended_features.py",
    "function": "enrich_with_chain_maps()",
}

_EMA_BAR_SEC = 10
_EMA_LOOKBACK_MIN = {
    9: 1.5,
    20: 3.3,
    50: 8.3,
    100: 16.7,
    200: 33.3,
    300: 50.0,
}

for _period in (9, 20, 50, 100, 200):
    _lookback = _EMA_LOOKBACK_MIN[_period]
    _bar_desc = f"{_EMA_BAR_SEC}-second bars (~{_lookback:g} min lookback)"
    RICH_COLUMN_DOCS[f"ltp_ema{_period}_to_ltp_ratio"] = {
        "display_name": f"Option LTP EMA{_period} / LTP Ratio",
        "description": (
            f"Option LTP {_period}-period EMA on {_bar_desc}, divided by current option LTP."
        ),
        "interpretation": (
            f"ltp_ema{_period} / current_ltp — short-horizon smoothed premium vs market price. "
            f"EMA{_period} on 10s bars ≈ {_lookback:g} minutes of history."
        ),
        "example": "0.98",
        "expected_range": "typically near 1.0",
        "formula_ref": f"ltp_ema{_period}_to_ltp_ratio",
        "formula_doc": f"ema{_period}(ltp, 10s bars) / current_ltp",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Option LTP EMA or current LTP missing or non-positive.",
        "implementation": dict(_EMA_TO_LTP_IMPL),
    }
    RICH_COLUMN_DOCS[f"spot_ema{_period}_to_ltp_ratio"] = {
        "display_name": f"Spot EMA{_period} / LTP Ratio",
        "description": (
            f"Underlying spot {_period}-period EMA on {_bar_desc}, divided by current option LTP."
        ),
        "interpretation": (
            f"spot_ema{_period} / current_ltp — index trend normalized to option premium. "
            f"EMA{_period} on 10s bars ≈ {_lookback:g} minutes."
        ),
        "example": "95.2",
        "expected_range": "positive, varies with spot and strike",
        "formula_ref": f"spot_ema{_period}_to_ltp_ratio",
        "formula_doc": f"ema{_period}(spot, 10s bars) / current_ltp",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Spot EMA or current LTP missing or non-positive.",
        "implementation": dict(_EMA_TO_LTP_IMPL),
    }
    RICH_COLUMN_DOCS[f"ltp_ema{_period}_to_spot_ratio"] = {
        "display_name": f"Option LTP EMA{_period} / Spot Ratio",
        "description": (
            f"Option LTP {_period}-period EMA on {_bar_desc}, divided by current spot."
        ),
        "interpretation": (
            f"ltp_ema{_period} / current_spot — option premium trend normalized to index level "
            f"(≈ {_lookback:g} min lookback)."
        ),
        "example": "0.0012",
        "expected_range": "small positive fraction for OTM options",
        "formula_ref": f"ltp_ema{_period}_to_spot_ratio",
        "formula_doc": f"ema{_period}(ltp, 10s bars) / current_spot",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Option LTP EMA or spot missing or non-positive.",
        "implementation": dict(_EMA_TO_LTP_IMPL),
    }

_IV_EMA_RATIO_IMPL = {
    "module": "chain_replay_ml/dataset_builder/iv_ema_ratio_features.py",
    "function": "enrich_iv_ema_ratio_features()",
}
_SPOT_RATIO_MONEYNESS_IMPL = {
    "module": "chain_replay_ml/dataset_builder/spot_ratio_moneyness_features.py",
    "function": "enrich_spot_ratio_moneyness_features()",
}
for _period in (9, 20, 50, 100, 200, 300):
    _lookback = _EMA_LOOKBACK_MIN[_period]
    _bar_desc = f"{_EMA_BAR_SEC}-second bars (~{_lookback:g} min lookback)"
    RICH_COLUMN_DOCS[f"ltp_ema{_period}_to_spot_ratio_x_iv_ema{_period}"] = {
        "display_name": f"LTP EMA{_period}/Spot × IV EMA{_period}",
        "description": (
            f"(Option LTP EMA{_period} / spot) × IV EMA{_period} on {_bar_desc}."
        ),
        "interpretation": (
            f"(ema{_period}(ltp)/spot) × ema{_period}(iv) — premium trend × smoothed IV."
        ),
        "example": "0.00018",
        "expected_range": "small positive",
        "formula_ref": f"ltp_ema{_period}_to_spot_ratio_x_iv_ema{_period}",
        "formula_doc": f"(ema{_period}(ltp) / spot) * ema{_period}(iv)",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "LTP EMA, IV EMA, or spot missing / non-positive.",
        "implementation": dict(_IV_EMA_RATIO_IMPL),
    }
    RICH_COLUMN_DOCS[f"spot_to_ltp_ratio_x_iv_ema{_period}"] = {
        "display_name": f"Spot/LTP × IV EMA{_period}",
        "description": f"(spot / ltp) × IV EMA{_period} on {_bar_desc}.",
        "interpretation": "Spot/LTP ratio scaled by smoothed implied volatility.",
        "example": "250.0",
        "expected_range": "positive",
        "formula_ref": f"spot_to_ltp_ratio_x_iv_ema{_period}",
        "formula_doc": f"(spot / ltp) * ema{_period}(iv)",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "IV EMA, spot, or LTP missing / non-positive.",
        "implementation": dict(_IV_EMA_RATIO_IMPL),
    }
    RICH_COLUMN_DOCS[f"iv_ema{_period}_to_ltp_ratio"] = {
        "display_name": f"IV EMA{_period} / LTP Ratio",
        "description": f"IV EMA{_period} divided by current option LTP ({_bar_desc}).",
        "interpretation": "Smoothed IV normalized by option premium.",
        "example": "0.0015",
        "expected_range": "positive",
        "formula_ref": f"iv_ema{_period}_to_ltp_ratio",
        "formula_doc": f"ema{_period}(iv) / ltp",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "IV EMA or LTP missing / non-positive.",
        "implementation": dict(_IV_EMA_RATIO_IMPL),
    }
    RICH_COLUMN_DOCS[f"iv_ema{_period}_to_spot_ratio"] = {
        "display_name": f"IV EMA{_period} / Spot Ratio",
        "description": f"IV EMA{_period} divided by current spot ({_bar_desc}).",
        "interpretation": "Smoothed IV normalized by index level.",
        "example": "6e-6",
        "expected_range": "small positive",
        "formula_ref": f"iv_ema{_period}_to_spot_ratio",
        "formula_doc": f"ema{_period}(iv) / spot",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "IV EMA or spot missing / non-positive.",
        "implementation": dict(_IV_EMA_RATIO_IMPL),
    }
    RICH_COLUMN_DOCS[f"spot_to_ltp_ratio_x_iv_ema{_period}_x_moneyness"] = {
        "display_name": f"Spot/LTP × IV EMA{_period} × Moneyness",
        "description": (
            f"(spot / ltp) × IV EMA{_period} × moneyness on {_bar_desc}."
        ),
        "interpretation": "Spot/LTP × smoothed IV further scaled by moneyness.",
        "example": "255.0",
        "expected_range": "positive",
        "formula_ref": f"spot_to_ltp_ratio_x_iv_ema{_period}_x_moneyness",
        "formula_doc": f"(spot / ltp) * ema{_period}(iv) * moneyness",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "IV EMA, spot, LTP, or moneyness missing / non-positive.",
        "implementation": dict(_IV_EMA_RATIO_IMPL),
    }

# Spot EMA300 / LTP + EMA×moneyness crosses (Spot and Other Ratio group)
RICH_COLUMN_DOCS["spot_ema300_to_ltp_ratio"] = {
    "display_name": "Spot EMA300 / LTP Ratio",
    "description": (
        f"Underlying spot 300-period EMA on {_EMA_BAR_SEC}-second bars "
        f"(~{_EMA_LOOKBACK_MIN[300]:g} min lookback), divided by current option LTP."
    ),
    "interpretation": (
        "spot_ema300 / current_ltp — longer-horizon index trend normalized to option premium."
    ),
    "example": "95.2",
    "expected_range": "positive, varies with spot and strike",
    "formula_ref": "spot_ema300_to_ltp_ratio",
    "formula_doc": "ema300(spot, 10s bars) / current_ltp",
    "unit": "ratio",
    "learning_level": "Advanced",
    "nullable": True,
    "expected_null_reason": "Spot EMA300 or current LTP missing or non-positive.",
    "implementation": dict(_SPOT_RATIO_MONEYNESS_IMPL),
}
for _period in (9, 20, 50, 100, 200, 300):
    _lookback = _EMA_LOOKBACK_MIN[_period]
    _bar_desc = f"{_EMA_BAR_SEC}-second bars (~{_lookback:g} min lookback)"
    RICH_COLUMN_DOCS[f"spot_ema{_period}_to_ltp_ratio_x_moneyness"] = {
        "display_name": f"Spot EMA{_period}/LTP × Moneyness",
        "description": f"(spot EMA{_period} / ltp) × moneyness on {_bar_desc}.",
        "interpretation": (
            f"(ema{_period}(spot) / ltp) × moneyness — index trend vs premium, scaled by moneyness."
        ),
        "example": "97.0",
        "expected_range": "positive",
        "formula_ref": f"spot_ema{_period}_to_ltp_ratio_x_moneyness",
        "formula_doc": f"(ema{_period}(spot) / ltp) * moneyness",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "Spot EMA, LTP, or moneyness missing / non-positive.",
        "implementation": dict(_SPOT_RATIO_MONEYNESS_IMPL),
    }
    RICH_COLUMN_DOCS[f"ltp_ema{_period}_to_spot_ratio_x_moneyness"] = {
        "display_name": f"LTP EMA{_period}/Spot × Moneyness",
        "description": f"(option LTP EMA{_period} / spot) × moneyness on {_bar_desc}.",
        "interpretation": (
            f"(ema{_period}(ltp) / spot) × moneyness — premium trend vs index, scaled by moneyness."
        ),
        "example": "0.00122",
        "expected_range": "small positive",
        "formula_ref": f"ltp_ema{_period}_to_spot_ratio_x_moneyness",
        "formula_doc": f"(ema{_period}(ltp) / spot) * moneyness",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "expected_null_reason": "LTP EMA, spot, or moneyness missing / non-positive.",
        "implementation": dict(_SPOT_RATIO_MONEYNESS_IMPL),
    }

RICH_COLUMN_DOCS["atm6_total_to_spot_ratio"] = {
    "display_name": "ATM+6 Total Sum / Spot Ratio",
    "description": "Sum of CE and PE ATM+6 wing LTP divided by current spot.",
    "interpretation": (
        "(ce_atm6_ltp_sum + pe_atm6_ltp_sum) / current_spot — total ATM wing premium load "
        "normalized to index level."
    ),
    "example": "0.024",
    "expected_range": "positive",
    "formula_ref": "atm6_total_to_spot_ratio",
    "formula_doc": "(ce_atm6_ltp_sum + pe_atm6_ltp_sum) / current_spot",
    "unit": "ratio",
    "learning_level": "Advanced",
    "nullable": True,
    "expected_null_reason": "CE/PE ATM+6 sums or spot missing or non-positive.",
    "implementation": dict(_EMA_TO_LTP_IMPL),
}

RICH_COLUMN_DOCS["ltp_std20_to_ltp_ratio"] = {
    "display_name": "LTP Std(20) / LTP Ratio",
    "description": "Standard deviation of the last 20 ten-second option LTP samples divided by current LTP.",
    "interpretation": (
        "std(last_20_ltp_on_10s_grid) / current_ltp — short-term option premium volatility "
        "relative to current price (~3.3 minutes of history)."
    ),
    "example": "0.04",
    "expected_range": "non-negative, typically small fraction",
    "formula_ref": "ltp_std20_to_ltp_ratio",
    "formula_doc": "std(last_20_ltp_on_10s_grid) / current_ltp",
    "unit": "ratio",
    "learning_level": "Advanced",
    "nullable": True,
    "expected_null_reason": "Fewer than 20 ten-second LTP samples or current LTP missing or non-positive.",
    "implementation": dict(_EMA_TO_LTP_IMPL),
}

RICH_COLUMN_DOCS["ltp_std20_to_spot_ratio"] = {
    "display_name": "LTP Std(20) / Spot Ratio",
    "description": "Standard deviation of the last 20 ten-second option LTP samples divided by spot.",
    "interpretation": (
        "std(last_20_ltp_on_10s_grid) / current_spot — short-term option premium volatility "
        "normalized to index level (~3.3 minutes of history)."
    ),
    "example": "0.0008",
    "expected_range": "non-negative",
    "formula_ref": "ltp_std20_to_spot_ratio",
    "formula_doc": "std(last_20_ltp_on_10s_grid) / current_spot",
    "unit": "ratio",
    "learning_level": "Advanced",
    "nullable": True,
    "expected_null_reason": "Fewer than 20 ten-second LTP samples or spot missing or non-positive.",
    "implementation": dict(_EMA_TO_LTP_IMPL),
}

RICH_COLUMN_DOCS["dgt_to_spot_ratio"] = {
    "display_name": "Greek Taylor / Spot Ratio",
    "description": "Greek Taylor predicted fair value (dgt_reiv_pred) divided by current spot.",
    "interpretation": (
        "Normalized Greek fair value relative to index level — comparable to ltp_to_spot_ratio "
        "but using model fair value instead of market LTP."
    ),
    "example": "0.0062",
    "expected_range": "small positive fraction",
    "formula_ref": "dgt_to_spot_ratio",
    "formula_doc": "dgt_reiv_pred / current_spot",
    "unit": "ratio",
    "learning_level": "Advanced",
    "nullable": True,
    "expected_null_reason": "dgt_reiv_pred or spot missing or non-positive.",
    "implementation": {
        "module": "chain_replay_ml/dataset_builder/extended_features.py",
        "function": "enrich_dataset_features()",
    },
}

RICH_COLUMN_DOCS["bs_to_spot_ratio"] = {
    "display_name": "BS Roll-IV / Spot Ratio",
    "description": "Black-Scholes roll-IV predicted price (bs_reiv_pred) divided by current spot.",
    "interpretation": (
        "Normalized BS fair value relative to index — baseline theoretical premium scaling. "
        "Pairs with dgt_to_spot_ratio to contrast BS vs Greek-Taylor normalization."
    ),
    "example": "0.0060",
    "expected_range": "small positive fraction",
    "formula_ref": "bs_to_spot_ratio",
    "formula_doc": "bs_reiv_pred / current_spot",
    "unit": "ratio",
    "learning_level": "Advanced",
    "nullable": True,
    "expected_null_reason": "bs_reiv_pred or spot missing or non-positive.",
    "implementation": {
        "module": "chain_replay_ml/dataset_builder/extended_features.py",
        "function": "enrich_dataset_features()",
    },
}

RICH_COLUMN_DOCS["dgt_prediction_error"] = {
    "display_name": "Greek Prediction Error",
    "description": "Current market LTP minus Greek Taylor predicted fair value (dgt_reiv_pred).",
    "interpretation": (
        "Positive = option trading above model fair value; negative = below. "
        "Direct signal for predicting actual LTP when the target is premium level."
    ),
    "example": "1.45",
    "expected_range": "signed ₹",
    "formula_ref": "dgt_prediction_error",
    "formula_doc": "current_ltp − dgt_reiv_pred",
    "unit": "₹",
    "learning_level": "Advanced",
    "nullable": True,
    "expected_null_reason": "LTP or dgt_reiv_pred unavailable at sample time.",
    "implementation": {
        "module": "chain_replay_ml/dataset_builder/extended_features.py",
        "function": "enrich_dataset_features()",
    },
}

RICH_COLUMN_DOCS["dgt_prediction_error_change_10s"] = {
    "display_name": "Greek Prediction Error Change (10 Seconds)",
    "description": "Change in Greek prediction error vs 10 seconds ago.",
    "interpretation": (
        "Rising error = market diverging from fair value; falling = converging. "
        "Captures short-term repricing relative to the Taylor anchor."
    ),
    "example": "0.35",
    "expected_range": "signed ₹ delta",
    "formula_ref": "dgt_prediction_error_change_10s",
    "formula_doc": "dgt_prediction_error − dgt_prediction_error_lag_10s",
    "unit": "₹",
    "learning_level": "Advanced",
    "nullable": True,
    "expected_null_reason": "Current or lagged prediction error unavailable.",
    "implementation": {
        "module": "chain_replay_ml/dataset_builder/extended_features.py",
        "function": "enrich_dataset_features()",
    },
}

_SHARP_MOMENTUM_IMPL = {
    "module": "chain_replay_ml/dataset_builder/sharp_momentum.py",
    "function": "enrich_sharp_momentum_features()",
}

_IV_ZSCORE_IMPL = {
    "module": "chain_replay_ml/dataset_builder/iv_zscore_features.py",
    "function": "enrich_iv_zscore_features()",
}

_IV_ZSCORE_BASE_IMPL = {
    "module": "chain_replay_ml/dataset_builder/extended_features.py",
    "function": "enrich_dataset_features()",
}


def _sharp_decay_table() -> str:
    return ", ".join(f"{h}={_SHARP_DECAY_AT_3S[h]}" for h in ("1m", "3m", "5m", "10m"))


def _sharp_score_description(side: str, horizon: str) -> str:
    decay = _SHARP_DECAY_AT_3S[horizon]
    side_label = "up" if side == "up" else "down"
    if side == "up":
        impulse = "add positive spot change (spot[t] − spot[t−1])"
        idle = "on spot fall the up score only decays"
    else:
        impulse = "add |spot[t] − spot[t−1]| when spot falls"
        idle = "on spot rise the down score only decays"
    return (
        f"Time-decayed cumulative spot {side_label} impulse ({horizon} memory) divided by option LTP. "
        f"State advances each feature-grid step: (1) decay — "
        f"score_{horizon} × {decay}^((dt_sec)/{_SHARP_REF_STEP_SEC:g}); "
        f"(2) on {side_label} move {impulse} ({idle}). "
        f"Decay@3s per horizon: {_sharp_decay_table()}. "
        f"Exported as spot_{side_label}_score_{horizon} / ltp."
    )


def _sharp_score_interpretation(horizon: str) -> str:
    decay = _SHARP_DECAY_AT_3S[horizon]
    return (
        "Shared spot momentum magnitude normalized per strike premium. "
        f"{horizon} uses decay@3s={decay} (higher = slower fade / longer memory)."
    )


def _sharp_count_description(side: str, horizon: str) -> str:
    decay = _SHARP_DECAY_AT_3S[horizon]
    side_label = "up" if side == "up" else "down"
    tick_rule = "+1 when spot rises" if side == "up" else "+1 when spot falls"
    return (
        f"Option LTP divided by decayed spot {side_label} tick count ({horizon} memory). "
        f"Count state each grid step: count_{horizon} × {decay}^((dt_sec)/{_SHARP_REF_STEP_SEC:g}); "
        f"{tick_rule}. Decay@3s per horizon: {_sharp_decay_table()}. "
        f"Exported as ltp / (spot_{side_label}_count_{horizon} + {_SHARP_COUNT_EPS:g})."
    )


def _sharp_count_interpretation(side: str, horizon: str) -> str:
    side_label = "up" if side == "up" else "down"
    return (
        f"Premium per unit of spot {side_label} activity; rises when few recent {side_label} ticks "
        f"({horizon} memory, decay@3s={_SHARP_DECAY_AT_3S[horizon]})."
    )


for _h in ("1m", "3m", "5m", "10m"):
    for _side in ("up", "down"):
        _name = f"spot_{_side}_score_{_h}_to_ltp_ratio"
        RICH_COLUMN_DOCS[_name] = {
            "display_name": f"Spot {_side.title()} Score {_h} / LTP",
            "description": _sharp_score_description(_side, _h),
            "interpretation": _sharp_score_interpretation(_h),
            "formula_ref": "sharp_momentum_score",
            "formula_doc": f"spot_{_side}_score_{_h} / ltp",
            "unit": "ratio",
            "learning_level": "Advanced",
            "nullable": False,
            "implementation": _SHARP_MOMENTUM_IMPL,
        }
        _cname = f"ltp_to_{_h}_spot_{_side}_sample_count_ratio"
        RICH_COLUMN_DOCS[_cname] = {
            "display_name": f"LTP / Spot {_side.title()} Count {_h}",
            "description": _sharp_count_description(_side, _h),
            "interpretation": _sharp_count_interpretation(_side, _h),
            "formula_ref": "sharp_momentum_count",
            "formula_doc": f"ltp / (spot_{_side}_count_{_h} + {_SHARP_COUNT_EPS:g})",
            "unit": "ratio",
            "learning_level": "Advanced",
            "nullable": False,
            "implementation": _SHARP_MOMENTUM_IMPL,
        }

RICH_COLUMN_DOCS["weighted_spot_ema_to_ltp_ratio"] = {
    "display_name": "Weighted Spot EMA / LTP",
    "description": "Weighted blend of spot EMA9/20/50/200 divided by option LTP.",
    "formula_ref": "weighted_spot_ema_to_ltp_ratio",
    "formula_doc": "(spot_ema9×4 + spot_ema20×3 + spot_ema50×2 + spot_ema200) / (10 × ltp)",
    "unit": "ratio",
    "learning_level": "Advanced",
    "nullable": False,
    "depends_on": [
        "ltp",
        "spot",
        "spot_ema9_to_ltp_ratio",
        "spot_ema20_to_ltp_ratio",
        "spot_ema50_to_ltp_ratio",
        "spot_ema200_to_ltp_ratio",
    ],
    "implementation": _SHARP_MOMENTUM_IMPL,
}
RICH_COLUMN_DOCS["weighted_ltp_ema_to_ltp_ratio"] = {
    "display_name": "Weighted LTP EMA / LTP",
    "description": "Weighted blend of LTP EMA9/20/50/200 divided by current LTP.",
    "formula_ref": "weighted_ltp_ema_to_ltp_ratio",
    "formula_doc": "(ltp_ema9×4 + ltp_ema20×3 + ltp_ema50×2 + ltp_ema200) / (10 × ltp)",
    "unit": "ratio",
    "learning_level": "Advanced",
    "nullable": False,
    "depends_on": [
        "ltp",
        "ltp_ema9_to_ltp_ratio",
        "ltp_ema20_to_ltp_ratio",
        "ltp_ema50_to_ltp_ratio",
        "ltp_ema200_to_ltp_ratio",
    ],
    "implementation": _SHARP_MOMENTUM_IMPL,
}

for _feat, _label, _formula in (
    (
        "weighted_iv_zscore_x_weighted_spot_ema_to_ltp_ratio",
        "Weighted IV Z-Score × Weighted Spot EMA/LTP",
        "((iv_zscore_1m×3 + iv_zscore_5m×2 + iv_zscore_15m) / 6) × weighted_spot_ema_to_ltp_ratio",
    ),
    (
        "weighted_spot_ema_to_ltp_ratio_x_delta",
        "Weighted Spot EMA/LTP × Delta",
        "weighted_spot_ema_to_ltp_ratio × delta",
    ),
    (
        "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_1m",
        "Weighted Spot EMA/LTP × IV Z-Score 1m",
        "weighted_spot_ema_to_ltp_ratio × iv_zscore_1m",
    ),
    (
        "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_5m",
        "Weighted Spot EMA/LTP × IV Z-Score 5m",
        "weighted_spot_ema_to_ltp_ratio × iv_zscore_5m",
    ),
    (
        "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m",
        "Weighted Spot EMA/LTP × IV Z-Score 15m",
        "weighted_spot_ema_to_ltp_ratio × iv_zscore_15m",
    ),
    (
        "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_1m_x_delta",
        "Weighted Spot EMA/LTP × IV Z-Score 1m × Delta",
        "weighted_spot_ema_to_ltp_ratio × iv_zscore_1m × delta",
    ),
    (
        "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_5m_x_delta",
        "Weighted Spot EMA/LTP × IV Z-Score 5m × Delta",
        "weighted_spot_ema_to_ltp_ratio × iv_zscore_5m × delta",
    ),
    (
        "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m_x_delta",
        "Weighted Spot EMA/LTP × IV Z-Score 15m × Delta",
        "weighted_spot_ema_to_ltp_ratio × iv_zscore_15m × delta",
    ),
):
    RICH_COLUMN_DOCS[_feat] = {
        "display_name": _label,
        "description": f"Cross feature: {_formula}.",
        "formula_ref": _feat,
        "formula_doc": _formula,
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": True,
        "implementation": _IV_ZSCORE_IMPL,
    }

_SPOT_HL_IMPL = {
    "module": "chain_replay_ml/dataset_builder/spot_hl_registry.py",
    "function": "enrich_spot_hl_ratio_registry_features() / enrich_spot_hl_composite_registry_features()",
}

_SPOT_HL_BAR_DESC = "10-second feature-grid bars (high/low from ticks in each step interval)"
_SPOT_HL_WEIGHTED_FORMULA = (
    "spot_{side}_ema20×4 + spot_{side}_ema50×3 + spot_{side}_ema200×2 + spot_{side}_ema300"
)
_SPOT_HL_CLOSE_WEIGHTED_FORMULA = (
    "spot_ema20×4 + spot_ema50×3 + spot_ema200×2 + spot_ema300"
)

for _period in (20, 50, 100, 200, 300):
    _hl_lookback = _EMA_LOOKBACK_MIN[_period]
    RICH_COLUMN_DOCS[f"spot_high_ema{_period}_to_ltp_ratio"] = {
        "display_name": f"Spot High EMA{_period} / LTP",
        "description": (
            f"EMA{_period} of per-bar spot highs on {_SPOT_HL_BAR_DESC}, divided by option LTP. "
            f"Each bar high = max tick LTP in the grid step; EMA{_period} ≈ {_hl_lookback:g} min memory."
        ),
        "interpretation": (
            f"spot_high_ema{_period} / ltp — upper envelope of index movement normalized to premium."
        ),
        "formula_ref": "spot_hl_high_ema",
        "formula_doc": f"ema{_period}(spot_high, 10s bars) / ltp",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": False,
        "implementation": _SPOT_HL_IMPL,
    }
    RICH_COLUMN_DOCS[f"spot_low_ema{_period}_to_ltp_ratio"] = {
        "display_name": f"Spot Low EMA{_period} / LTP",
        "description": (
            f"EMA{_period} of per-bar spot lows on {_SPOT_HL_BAR_DESC}, divided by option LTP. "
            f"Each bar low = min tick LTP in the grid step; EMA{_period} ≈ {_hl_lookback:g} min memory."
        ),
        "interpretation": (
            f"spot_low_ema{_period} / ltp — lower envelope of index movement normalized to premium."
        ),
        "formula_ref": "spot_hl_low_ema",
        "formula_doc": f"ema{_period}(spot_low, 10s bars) / ltp",
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": False,
        "implementation": _SPOT_HL_IMPL,
    }
    RICH_COLUMN_DOCS[f"spot_ema{_period}_channel_width"] = {
        "display_name": f"Spot EMA{_period} Channel Width",
        "description": (
            f"Spot high-low EMA{_period} channel width on {_SPOT_HL_BAR_DESC}: "
            f"spot_high_ema{_period} − spot_low_ema{_period}."
        ),
        "interpretation": (
            "Width of the spot high/low EMA envelope. Tight when range is compressed; "
            "wide when the high-low band expands."
        ),
        "formula_ref": "spot_hl_channel_width",
        "formula_doc": f"spot_high_ema{_period} − spot_low_ema{_period}",
        "unit": "points",
        "learning_level": "Advanced",
        "nullable": True,
        "implementation": _SPOT_HL_IMPL,
    }
    RICH_COLUMN_DOCS[f"ltp_to_spot_ema{_period}_channel_width_ratio"] = {
        "display_name": f"LTP / Spot EMA{_period} Channel Width",
        "description": (
            f"Option LTP divided by the spot high-low EMA{_period} channel width. "
            f"Pipeline Owned (Wave 5): recreate via Interaction "
            f"`ltp / spot_ema{_period}_channel_width` "
            f"(same {_SPOT_HL_BAR_DESC})."
        ),
        "interpretation": (
            "High when premium is large relative to a tight spot channel; low when the channel is wide. "
            "Captures high-low balance / volatility envelope vs option price."
        ),
        "formula_ref": "spot_hl_channel_width",
        "formula_doc": (
            f"ltp / (abs(spot_ema{_period}_channel_width) + 1e-6)  "
            f"[Master parity]; Interaction: ltp ÷ spot_ema{_period}_channel_width"
        ),
        "unit": "ratio",
        "learning_level": "Advanced",
        "nullable": False,
        "implementation": _SPOT_HL_IMPL,
    }

RICH_COLUMN_DOCS["weighted_spot_high_ema_to_ltp_ratio"] = {
    "display_name": "Weighted Spot High EMA / LTP",
    "description": (
        "Weighted blend of spot high EMA20/50/200/300 (per-bar tick highs) divided by option LTP."
    ),
    "interpretation": "Upper spot channel trend normalized per strike premium.",
    "formula_ref": "weighted_spot_high_ema_to_ltp_ratio",
    "formula_doc": (
        f"({_SPOT_HL_WEIGHTED_FORMULA.format(side='high')}) / (10 × ltp)"
    ),
    "unit": "ratio",
    "learning_level": "Advanced",
    "nullable": False,
    "implementation": _SPOT_HL_IMPL,
}
RICH_COLUMN_DOCS["weighted_spot_low_ema_to_ltp_ratio"] = {
    "display_name": "Weighted Spot Low EMA / LTP",
    "description": (
        "Weighted blend of spot low EMA20/50/200/300 (per-bar tick lows) divided by option LTP."
    ),
    "interpretation": "Lower spot channel trend normalized per strike premium.",
    "formula_ref": "weighted_spot_low_ema_to_ltp_ratio",
    "formula_doc": (
        f"({_SPOT_HL_WEIGHTED_FORMULA.format(side='low')}) / (10 × ltp)"
    ),
    "unit": "ratio",
    "learning_level": "Advanced",
    "nullable": False,
    "implementation": _SPOT_HL_IMPL,
}
RICH_COLUMN_DOCS["weighted_spot_high_ema_to_weighted_spot_low_ema"] = {
    "display_name": "Weighted High / Weighted Low Spot EMA",
    "description": (
        "Ratio of weighted spot-high EMA blend to weighted spot-low EMA blend "
        "(same 20/50/200/300 weights on high vs low bar series)."
    ),
    "interpretation": "High-low channel balance: >1 means upper envelope dominates.",
    "formula_ref": "weighted_spot_high_ema_to_weighted_spot_low_ema",
    "formula_doc": (
        f"({_SPOT_HL_WEIGHTED_FORMULA.format(side='high')}) / "
        f"({_SPOT_HL_WEIGHTED_FORMULA.format(side='low')})"
    ),
    "unit": "ratio",
    "learning_level": "Advanced",
    "nullable": False,
    "implementation": _SPOT_HL_IMPL,
}
RICH_COLUMN_DOCS["weighted_spot_ema_to_weighted_spot_low_ema"] = {
    "display_name": "Weighted Spot Close / Weighted Low EMA",
    "description": (
        "Weighted spot close EMA blend (20/50/200/300 on LTP close bars) divided by "
        "weighted spot-low EMA blend."
    ),
    "interpretation": "Spot trend vs lower channel — premium context for downside envelope.",
    "formula_ref": "weighted_spot_ema_to_weighted_spot_low_ema",
    "formula_doc": (
        f"({_SPOT_HL_CLOSE_WEIGHTED_FORMULA}) / ({_SPOT_HL_WEIGHTED_FORMULA.format(side='low')})"
    ),
    "unit": "ratio",
    "learning_level": "Advanced",
    "nullable": False,
    "implementation": _SPOT_HL_IMPL,
}
RICH_COLUMN_DOCS["weighted_spot_ema_to_weighted_spot_high_ema"] = {
    "display_name": "Weighted Spot Close / Weighted High EMA",
    "description": (
        "Weighted spot close EMA blend (20/50/200/300 on LTP close bars) divided by "
        "weighted spot-high EMA blend."
    ),
    "interpretation": "Spot trend vs upper channel — how close price sits under the high envelope.",
    "formula_ref": "weighted_spot_ema_to_weighted_spot_high_ema",
    "formula_doc": (
        f"({_SPOT_HL_CLOSE_WEIGHTED_FORMULA}) / ({_SPOT_HL_WEIGHTED_FORMULA.format(side='high')})"
    ),
    "unit": "ratio",
    "learning_level": "Advanced",
    "nullable": False,
    "implementation": _SPOT_HL_IMPL,
}

_CHAIN_FLOW_IMPL = {
    "module": "chain_replay_ml/dataset_builder/current_to_atm6_flow.py",
    "function": "enrich_current_to_atm6_flow_features()",
}

RICH_COLUMN_DOCS["current_to_atm6_flow_delta_ltp_to_spot_ratio"] = {
    "display_name": "Current→ATM6 Flow × Delta × LTP / Spot",
    "description": (
        "Average 1-minute volume and OI change % across seven strikes from the current "
        "contract toward ATM, blended 50/50, then scaled by delta, current LTP, and divided by spot."
    ),
    "interpretation": (
        "For OTM option-buying: rising participation on strikes between current and ATM, "
        "combined with higher delta sensitivity and premium, normalized to underlying scale. "
        "Uses |delta| so CE and PE contribute symmetrically. "
        "CE walks lower strikes toward ATM; PE walks higher strikes toward ATM."
    ),
    "example": "0.42",
    "expected_range": "signed, scales with flow %, delta, and LTP/spot",
    "formula_ref": "current_to_atm6_flow_delta_ltp_to_spot_ratio",
    "formula_doc": (
        "((avg(volume_change_pct_1m over 7 strikes toward ATM) + "
        "avg(oi_change_pct_1m over 7 strikes toward ATM)) / 2) × |delta| × ltp / spot"
    ),
    "unit": "ratio",
    "learning_level": "Advanced",
    "nullable": True,
    "expected_null_reason": (
        "Any of the seven strikes missing, volume/OI history unavailable, or delta/LTP/spot invalid."
    ),
    "implementation": _CHAIN_FLOW_IMPL,
}

_HISTORIC_SPOT_EMA_IMPL = {
    "module": "chain_replay_ml/dataset_builder/historic_spot_ema_context.py",
    "function": "enrich_historic_spot_ema_features()",
}

for _ivl in ("1m", "3m", "5m", "15m"):
    for _per in (9, 20, 50, 100, 200):
        _name = f"spot_{_ivl}_ema{_per}"
        RICH_COLUMN_DOCS[_name] = {
            "display_name": f"Spot {_ivl.upper()} EMA {_per}",
            "description": (
                f"NIFTY EMA{_per} on {_ivl} historic candles from angel_historic_bars.db. "
                "As-of join: most recent candle with bucket_start ≤ tick timestamp. "
                "Not computed from Trading Day ticks."
            ),
            "interpretation": (
                "Higher-timeframe market context for the underlying at each tick. "
                "Warm-up history comes from continuous prior sessions in the historic DB."
            ),
            "example": "24500.5",
            "expected_range": "positive NIFTY index level",
            "formula_ref": "historic_spot_ema_asof",
            "formula_doc": (
                f"EMA{_per}(close) on {_ivl} bars; lookup last bar with bucket_start ≤ ts"
            ),
            "unit": "index points",
            "learning_level": "Intermediate",
            "nullable": True,
            "expected_null_reason": (
                "No historic candle at or before the tick, or EMA still warming up."
            ),
            "implementation": _HISTORIC_SPOT_EMA_IMPL,
        }

_MICRO_IMPL = {
    "module": "chain_replay_ml/dataset_builder/market_microstructure.py",
    "function": "enrich_market_microstructure_features()",
}

RICH_COLUMN_DOCS["mid_price"] = {
    "display_name": "Fair Mid Price",
    "description": "Option book mid = (best_bid + best_ask) / 2 from L1 SNAP_QUOTE.",
    "interpretation": "Liquidity midpoint; ignore for crossed books (ask < bid).",
    "example": "125.05",
    "expected_range": "positive premium (₹)",
    "formula_ref": "book_mid_l1",
    "formula_doc": "(bid_l1 + ask_l1) / 2",
    "unit": "₹",
    "learning_level": "Beginner",
    "nullable": True,
    "expected_null_reason": "Missing L1 bid or ask.",
    "implementation": _MICRO_IMPL,
}

RICH_COLUMN_DOCS["microprice"] = {
    "display_name": "Micro Price",
    "description": "Size-weighted L1 mid: (ask·bid_qty + bid·ask_qty) / (bid_qty + ask_qty).",
    "interpretation": "Book pressure pulls the fair price toward the thinner side.",
    "example": "125.12",
    "expected_range": "between bid and ask when quantities > 0",
    "formula_ref": "book_microprice_l1",
    "formula_doc": "(ask×bq + bid×aq) / (bq + aq); falls back to mid if qty sum is 0",
    "unit": "₹",
    "learning_level": "Intermediate",
    "nullable": True,
    "expected_null_reason": "Missing L1 bid/ask.",
    "implementation": _MICRO_IMPL,
}

RICH_COLUMN_DOCS["microprice_bias"] = {
    "display_name": "Microprice Bias",
    "description": (
        "Dimensionless location of microprice in the spread: "
        "(microprice − mid_price) / bid_ask_spread."
    ),
    "interpretation": (
        "≈ +0.5 strong buy-side pressure; ≈ 0 balanced; ≈ −0.5 strong sell-side pressure. "
        "Independent of absolute spread width."
    ),
    "example": "0.25",
    "expected_range": "typically (−0.5, +0.5) for healthy books",
    "formula_ref": "microprice_bias",
    "formula_doc": "(microprice − mid_price) / spread",
    "unit": "ratio",
    "learning_level": "Intermediate",
    "nullable": True,
    "expected_null_reason": "Missing L1 book or zero spread.",
    "implementation": _MICRO_IMPL,
}

RICH_COLUMN_DOCS["book_imbalance_l1"] = {
    "display_name": "Order Book Imbalance L1",
    "description": "(bid_qty − ask_qty) / (bid_qty + ask_qty) at best level.",
    "interpretation": "+1 = all bid size; −1 = all ask size.",
    "example": "0.25",
    "expected_range": "[-1, 1]",
    "formula_ref": "book_imbalance_l1",
    "formula_doc": "(bq1 − aq1) / (bq1 + aq1)",
    "unit": "ratio",
    "learning_level": "Intermediate",
    "nullable": True,
    "expected_null_reason": "Missing L1 or zero total quantity.",
    "implementation": _MICRO_IMPL,
}

RICH_COLUMN_DOCS["book_imbalance_l1_5"] = {
    "display_name": "Order Book Imbalance L1–L5",
    "description": "Cumulative (Σbid_qty − Σask_qty) / (Σbid + Σask) over five levels.",
    "interpretation": "Deeper book pressure than L1 alone.",
    "example": "0.10",
    "expected_range": "[-1, 1]",
    "formula_ref": "book_imbalance_l1_5",
    "formula_doc": "(Σbq − Σaq) / (Σbq + Σaq) over L1–L5",
    "unit": "ratio",
    "learning_level": "Intermediate",
    "nullable": True,
    "expected_null_reason": "Missing book or zero total depth.",
    "implementation": _MICRO_IMPL,
}

RICH_COLUMN_DOCS["bid_depth_l1_5"] = {
    "display_name": "Bid Depth L1–L5",
    "description": "Sum of bid quantities across five book levels.",
    "interpretation": "Displayed buy-side liquidity on the option.",
    "example": "4200",
    "expected_range": "≥ 0 lots",
    "formula_ref": "bid_depth_l1_5",
    "formula_doc": "sum(bid_quantities[0:5])",
    "unit": "lots",
    "learning_level": "Beginner",
    "nullable": True,
    "expected_null_reason": "Missing book snapshot.",
    "implementation": _MICRO_IMPL,
}

RICH_COLUMN_DOCS["ask_depth_l1_5"] = {
    "display_name": "Ask Depth L1–L5",
    "description": "Sum of ask quantities across five book levels.",
    "interpretation": "Displayed sell-side liquidity on the option.",
    "example": "3800",
    "expected_range": "≥ 0 lots",
    "formula_ref": "ask_depth_l1_5",
    "formula_doc": "sum(ask_quantities[0:5])",
    "unit": "lots",
    "learning_level": "Beginner",
    "nullable": True,
    "expected_null_reason": "Missing book snapshot.",
    "implementation": _MICRO_IMPL,
}

RICH_COLUMN_DOCS["book_depth_slope_bid"] = {
    "display_name": "Book Depth Slope (Bid)",
    "description": (
        "OLS slope of bid quantity vs level index (0 = L1 … L5) on the option book."
    ),
    "interpretation": (
        "Negative slope ⇒ size thins away from touch (typical). "
        "Flatter / positive ⇒ deeper resting size further from L1."
    ),
    "example": "-20",
    "expected_range": "lots per level",
    "formula_ref": "book_depth_slope_bid",
    "formula_doc": "OLS(qty ~ level_index) on bid_quantities L1–L5",
    "unit": "lots/level",
    "learning_level": "Intermediate",
    "nullable": True,
    "expected_null_reason": "Fewer than two usable bid levels.",
    "implementation": _MICRO_IMPL,
}

RICH_COLUMN_DOCS["book_depth_slope_ask"] = {
    "display_name": "Book Depth Slope (Ask)",
    "description": (
        "OLS slope of ask quantity vs level index (0 = L1 … L5) on the option book."
    ),
    "interpretation": (
        "Same shape interpretation as bid slope on the sell side of the book."
    ),
    "example": "-10",
    "expected_range": "lots per level",
    "formula_ref": "book_depth_slope_ask",
    "formula_doc": "OLS(qty ~ level_index) on ask_quantities L1–L5",
    "unit": "lots/level",
    "learning_level": "Intermediate",
    "nullable": True,
    "expected_null_reason": "Fewer than two usable ask levels.",
    "implementation": _MICRO_IMPL,
}

