"""Phase 4A.3: SABR (Stochastic Alpha Beta Rho) Surface Calibrator.

Implements Hagan et al. (2002) closed-form lognormal volatility approximation:
    sigma_SABR(K, F_T; alpha, beta, rho, nu)
with:
- Fixed beta parameterization (default beta = 0.5, configurable to 1.0)
- Direct alpha root inversion from ATM implied volatility
- 2D bounded optimization over (rho, nu)
- Strict parameter bounds: alpha > 0, -1 < rho < 1, nu > 0
- Tiered quality classification and safe error containment.

Conforms strictly to Doc 18 v1.1.0 specifications and Phase 4A.0 contracts.
"""

from __future__ import annotations

import math
from typing import Any, Sequence
import numpy as np
from scipy.optimize import minimize

from .types import (
    CalibrationQualityTier,
    CalibrationStatus,
    DEFAULT_SURFACE_MATH_CONFIG,
    SabrBetaMode,
    SabrCalibrationResult,
    SabrParameters,
    SurfaceMathConfig,
)


def evaluate_sabr_volatility(
    strike: np.ndarray | float,
    forward_price: float,
    time_to_expiry_years: float,
    params: SabrParameters,
) -> np.ndarray | float:
    """Evaluate closed-form SABR implied volatility for scalar or array strikes."""
    if isinstance(strike, (int, float)):
        return params.implied_volatility(float(strike), forward_price, time_to_expiry_years)
    
    k_arr = np.asarray(strike, dtype=np.float64)
    out = np.zeros_like(k_arr)
    for i, k in enumerate(k_arr):
        out[i] = params.implied_volatility(float(k), forward_price, time_to_expiry_years)
    return out


def invert_sabr_alpha_from_atm(
    atm_vol: float,
    forward_price: float,
    time_to_expiry_years: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """Invert initial alpha from ATM implied volatility using cubic root solver or Newton step."""
    sigma_atm = max(1e-4, float(atm_vol))
    F = max(1e-4, float(forward_price))
    T = max(1e-6, float(time_to_expiry_years))
    f_pow = math.pow(F, 1.0 - beta)
    
    # First-order initial guess
    alpha = sigma_atm * f_pow

    # Newton-Raphson refinement on cubic equation
    for _ in range(20):
        # Cubic coefficients: c1 * alpha^3 + c2 * alpha^2 + (1 + c3) * alpha - sigma_atm * f_pow = 0
        c1 = ((1.0 - beta) ** 2 / 24.0) * (T / math.pow(F, 2.0 * (1.0 - beta)))
        c2 = 0.25 * rho * beta * nu * T / f_pow
        c3 = ((2.0 - 3.0 * rho * rho) / 24.0) * (nu * nu) * T

        f_val = c1 * (alpha ** 3) + c2 * (alpha ** 2) + (1.0 + c3) * alpha - sigma_atm * f_pow
        df_val = 3.0 * c1 * (alpha ** 2) + 2.0 * c2 * alpha + (1.0 + c3)

        if abs(df_val) < 1e-12:
            break
        step = f_val / df_val
        alpha -= step
        if abs(step) < 1e-8 or alpha <= 0.0:
            break

    return max(1e-4, alpha)


# =============================================================================
# SABR CALIBRATOR CLASS
# =============================================================================

class SabrCalibrator:
    """Hagan SABR parametric volatility surface calibrator with direct alpha inversion."""

    def __init__(self, config: SurfaceMathConfig | None = None) -> None:
        self.config = config or DEFAULT_SURFACE_MATH_CONFIG

    def calibrate_slice(
        self,
        *,
        strikes: Sequence[float] | np.ndarray,
        implied_volatilities: Sequence[float] | np.ndarray,
        underlying_spot: float,
        time_to_expiry_years: float,
        risk_free_rate: float = 0.07,
        forward_price: float | None = None,
        beta: float | None = None,
        weights: Sequence[float] | np.ndarray | None = None,
        as_of_timestamp: float = 0.0,
        expiry_date: str = "",
    ) -> SabrCalibrationResult:
        """Calibrate SABR (alpha, rho, nu) for a fixed beta on a single expiration slice."""
        k_raw = np.asarray(strikes, dtype=np.float64)
        iv_raw = np.asarray(implied_volatilities, dtype=np.float64)
        t = float(time_to_expiry_years)
        s = float(underlying_spot)
        r = float(risk_free_rate)
        beta_val = float(beta if beta is not None else self.config.sabr_beta)

        fwd = float(forward_price) if forward_price is not None and forward_price > 0 else s * math.exp(r * max(0.0, t))

        # 1. Validation and pre-filtering
        if s <= 0.0 or fwd <= 0.0 or t <= 0.0 or len(k_raw) == 0:
            return self._empty_result(
                status=CalibrationStatus.INSUFFICIENT_DATA,
                reason="Invalid spot, forward, time-to-expiry, or empty strikes array",
                as_of_timestamp=as_of_timestamp,
                expiry_date=expiry_date,
                time_to_expiry_years=t,
                forward_price=fwd,
                beta=beta_val,
            )

        valid_mask = (k_raw > 0.0) & (iv_raw > self.config.min_iv_bound) & (iv_raw < self.config.max_iv_bound) & np.isfinite(k_raw) & np.isfinite(iv_raw)
        k_clean = k_raw[valid_mask]
        iv_clean = iv_raw[valid_mask]

        # Deduplicate strikes
        if len(k_clean) > 0:
            unique_k, inv_idx = np.unique(k_clean, return_inverse=True)
            if len(unique_k) < len(k_clean):
                unique_iv = np.zeros_like(unique_k)
                counts = np.zeros_like(unique_k)
                for i, k_idx in enumerate(inv_idx):
                    unique_iv[k_idx] += iv_clean[i]
                    counts[k_idx] += 1
                k_clean = unique_k
                iv_clean = unique_iv / np.maximum(1, counts)

        n_strikes = len(k_clean)

        # 2. Path B: Minimum liquid strikes check
        if n_strikes < self.config.min_liquid_strikes:
            return self._empty_result(
                status=CalibrationStatus.INSUFFICIENT_DATA,
                reason=f"Insufficient liquid strikes ({n_strikes} < min {self.config.min_liquid_strikes})",
                as_of_timestamp=as_of_timestamp,
                expiry_date=expiry_date,
                time_to_expiry_years=t,
                forward_price=fwd,
                beta=beta_val,
                strikes_used=n_strikes,
            )

        # 3. ATM volatility identification (interpolated at forward price)
        if fwd <= k_clean[0]:
            atm_iv = float(iv_clean[0])
        elif fwd >= k_clean[-1]:
            atm_iv = float(iv_clean[-1])
        else:
            atm_iv = float(np.interp(fwd, k_clean, iv_clean))

        # Weights normalization
        if weights is not None and len(weights) == len(k_raw):
            w_weights = np.asarray(weights, dtype=np.float64)[valid_mask]
            w_weights = np.maximum(1e-4, w_weights)
            w_weights = w_weights / np.sum(w_weights)
        else:
            w_weights = np.full(n_strikes, 1.0 / n_strikes, dtype=np.float64)

        # 4. Optimization over (rho, nu) with direct alpha inversion
        best_params, opt_status, iterations, warnings = self._fit_sabr(
            strikes=k_clean,
            iv_obs=iv_clean,
            weights=w_weights,
            fwd=fwd,
            atm_iv=atm_iv,
            t=t,
            beta=beta_val,
        )

        # 5. Evaluate errors
        iv_pred = np.array([best_params.implied_volatility(float(k), fwd, t) for k in k_clean], dtype=np.float64)
        iv_errors = iv_clean - iv_pred
        rmse = float(np.sqrt(np.mean(iv_errors ** 2)))
        mae = float(np.mean(np.abs(iv_errors)))

        # 6. Quality tier governance
        if rmse <= self.config.tier1_rmse_threshold and opt_status == CalibrationStatus.CONVERGED:
            quality_tier = CalibrationQualityTier.TIER_1_HIGH_PRECISION
        elif rmse <= self.config.tier2_rmse_threshold:
            quality_tier = CalibrationQualityTier.TIER_2_ACCEPTABLE
            if opt_status == CalibrationStatus.CONVERGED:
                opt_status = CalibrationStatus.NOISY_FIT
        else:
            quality_tier = CalibrationQualityTier.TIER_3_FAILED
            opt_status = CalibrationStatus.CALIB_WARNING

        return SabrCalibrationResult(
            parameters=best_params,
            status=opt_status,
            quality_tier=quality_tier,
            rmse=rmse,
            mae=mae,
            strikes_used=n_strikes,
            forward_price=fwd,
            atm_implied_volatility=atm_iv,
            as_of_timestamp=as_of_timestamp,
            expiry_date=expiry_date,
            time_to_expiry_years=t,
            warnings=warnings,
        )

    def _fit_sabr(
        self,
        *,
        strikes: np.ndarray,
        iv_obs: np.ndarray,
        weights: np.ndarray,
        fwd: float,
        atm_iv: float,
        t: float,
        beta: float,
    ) -> tuple[SabrParameters, CalibrationStatus, int, list[str]]:
        """2D Nelder-Mead optimization for (rho, nu) with direct alpha inversion."""
        warnings: list[str] = []

        # Initial guess: rho = -0.30 (standard equity skew), nu = 0.50
        rho0 = -0.30
        nu0 = 0.50

        def objective(x: np.ndarray) -> float:
            rho_val = float(np.clip(x[0], -0.999, 0.999))
            nu_val = max(1e-4, float(x[1]))

            # Invert exact alpha for candidate (rho, nu)
            alpha_val = invert_sabr_alpha_from_atm(
                atm_vol=atm_iv,
                forward_price=fwd,
                time_to_expiry_years=t,
                beta=beta,
                rho=rho_val,
                nu=nu_val,
            )

            sabr = SabrParameters(alpha=alpha_val, beta=beta, rho=rho_val, nu=nu_val)
            pred = np.array([sabr.implied_volatility(float(k), fwd, t) for k in strikes], dtype=np.float64)
            diff = iv_obs - pred
            return float(np.sum(weights * (diff ** 2)))

        bounds = [(-0.999, 0.999), (1e-4, 8.0)]
        opt_res = minimize(
            objective,
            x0=np.array([rho0, nu0]),
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": 250, "xatol": 1e-5, "fatol": 1e-6},
        )

        rho_opt = float(np.clip(opt_res.x[0], -0.999, 0.999))
        nu_opt = max(1e-4, float(opt_res.x[1]))
        alpha_opt = invert_sabr_alpha_from_atm(
            atm_vol=atm_iv,
            forward_price=fwd,
            time_to_expiry_years=t,
            beta=beta,
            rho=rho_opt,
            nu=nu_opt,
        )

        status = CalibrationStatus.CONVERGED if opt_res.success else CalibrationStatus.ASYMPTOTIC_BOUND
        if not opt_res.success:
            warnings.append(f"OPTIMIZER_WARNING: {opt_res.message}")

        params = SabrParameters(
            alpha=float(alpha_opt),
            beta=float(beta),
            rho=float(rho_opt),
            nu=float(nu_opt),
        )

        return params, status, int(opt_res.nit), warnings

    def _empty_result(
        self,
        *,
        status: CalibrationStatus,
        reason: str,
        as_of_timestamp: float,
        expiry_date: str,
        time_to_expiry_years: float,
        forward_price: float,
        beta: float,
        strikes_used: int = 0,
    ) -> SabrCalibrationResult:
        """Construct a clean zeroed SabrCalibrationResult for invalid or sparse data slices."""
        return SabrCalibrationResult(
            parameters=SabrParameters(alpha=0.0, beta=beta, rho=0.0, nu=0.0),
            status=status,
            quality_tier=CalibrationQualityTier.TIER_3_FAILED,
            rmse=1.0,
            mae=1.0,
            strikes_used=strikes_used,
            forward_price=forward_price,
            atm_implied_volatility=0.0,
            as_of_timestamp=as_of_timestamp,
            expiry_date=expiry_date,
            time_to_expiry_years=time_to_expiry_years,
            warnings=[reason],
        )
