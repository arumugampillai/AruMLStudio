"""Phase 4A.2: Raw SVI (Stochastic Volatility Inspired) Surface Calibrator.

Implements Gatheral's Raw SVI 5-parameter total variance formulation:
    w(k; chi) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
where k = ln(K / F_T) is log-forward moneyness and chi = {a, b, rho, m, sigma}.

Uses Zeliade quasi-explicit linear-nonlinear decomposition:
- Inner loop: Bounded linear least-squares for linear parameters (a, b, c) where c = b * rho.
- Outer loop: 2D bounded Nelder-Mead / Powell search for nonlinear parameters (m, sigma).
- Strict parameter bounds and Lee wing-moment no-arbitrage enforcement.

Conforms strictly to Doc 18 v1.1.0 specifications and Phase 4A.0 contracts.
"""

from __future__ import annotations

import math
from typing import Any, Sequence
import numpy as np
from scipy.optimize import minimize, lsq_linear

from .types import (
    CalibrationQualityTier,
    CalibrationStatus,
    DEFAULT_SURFACE_MATH_CONFIG,
    SurfaceMathConfig,
    SviCalibrationResult,
    SviParameters,
)


def compute_log_moneyness(
    strikes: np.ndarray | Sequence[float],
    forward_price: float,
) -> np.ndarray:
    """Compute log-forward moneyness k = ln(K / F_T)."""
    k_arr = np.asarray(strikes, dtype=np.float64)
    f = float(forward_price)
    if f <= 0.0:
        raise ValueError(f"Forward price must be strictly positive, got {f}")
    return np.log(k_arr / f)


def evaluate_raw_svi(
    k: np.ndarray | float,
    params: SviParameters,
) -> np.ndarray | float:
    """Evaluate Raw SVI total implied variance w(k) for scalar or array moneyness."""
    if isinstance(k, (int, float)):
        return params.total_variance(float(k))
    
    k_arr = np.asarray(k, dtype=np.float64)
    diff = k_arr - params.m
    w = params.a + params.b * (params.rho * diff + np.sqrt(diff * diff + params.sigma * params.sigma))
    return np.maximum(0.0, w)


def evaluate_svi_implied_volatility(
    k: np.ndarray | float,
    params: SviParameters,
    time_to_expiry_years: float,
) -> np.ndarray | float:
    """Evaluate annualized implied volatility sigma(k, T) = sqrt(max(0, w(k)) / T)."""
    t = max(1e-6, float(time_to_expiry_years))
    w = evaluate_raw_svi(k, params)
    return np.sqrt(w / t)


# =============================================================================
# ZELIADE QUASI-EXPLICIT SVI CALIBRATOR
# =============================================================================

class SviCalibrator:
    """Quasi-explicit Raw SVI volatility surface calibrator with strict no-arbitrage constraints."""

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
        weights: Sequence[float] | np.ndarray | None = None,
        as_of_timestamp: float = 0.0,
        expiry_date: str = "",
    ) -> SviCalibrationResult:
        """Calibrate a single expiry slice using Zeliade quasi-explicit linear/nonlinear search.
        
        Guarantees strict parameter bounds, no-arbitrage verification, and tiered quality classification.
        """
        k_raw = np.asarray(strikes, dtype=np.float64)
        iv_raw = np.asarray(implied_volatilities, dtype=np.float64)
        t = float(time_to_expiry_years)
        s = float(underlying_spot)
        r = float(risk_free_rate)

        # 1. Forward price calculation (F_T = S * exp(r * T))
        fwd = float(forward_price) if forward_price is not None and forward_price > 0 else s * math.exp(r * max(0.0, t))

        # 2. Input validation & Pre-filtering
        if s <= 0.0 or fwd <= 0.0 or t <= 0.0 or len(k_raw) == 0:
            return self._empty_result(
                status=CalibrationStatus.INSUFFICIENT_DATA,
                reason="Invalid spot, forward, time-to-expiry, or empty strikes array",
                as_of_timestamp=as_of_timestamp,
                expiry_date=expiry_date,
                time_to_expiry_years=t,
                forward_price=fwd,
            )

        # Filter valid positive finite strikes and IVs
        valid_mask = (k_raw > 0.0) & (iv_raw > self.config.min_iv_bound) & (iv_raw < self.config.max_iv_bound) & np.isfinite(k_raw) & np.isfinite(iv_raw)
        k_clean = k_raw[valid_mask]
        iv_clean = iv_raw[valid_mask]

        # Deduplicate strikes if necessary (average IV for duplicate strikes)
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

        # 3. Path B Check: Minimum liquid strikes required
        if n_strikes < self.config.min_liquid_strikes:
            return self._empty_result(
                status=CalibrationStatus.INSUFFICIENT_DATA,
                reason=f"Insufficient liquid strikes ({n_strikes} < min {self.config.min_liquid_strikes})",
                as_of_timestamp=as_of_timestamp,
                expiry_date=expiry_date,
                time_to_expiry_years=t,
                forward_price=fwd,
                strikes_used=n_strikes,
            )

        # 4. Compute observed log-moneyness and total variance
        log_k = np.log(k_clean / fwd)
        w_obs = (iv_clean ** 2) * t

        # Weights normalization
        if weights is not None and len(weights) == len(k_raw):
            w_weights = np.asarray(weights, dtype=np.float64)[valid_mask]
            w_weights = np.maximum(1e-4, w_weights)
            w_weights = w_weights / np.sum(w_weights)
        else:
            w_weights = np.full(n_strikes, 1.0 / n_strikes, dtype=np.float64)

        # 5. Zeliade Calibration Execution
        best_params, opt_status, iterations, warnings = self._fit_zeliade(
            log_k=log_k,
            w_obs=w_obs,
            weights=w_weights,
            t=t,
        )

        # 6. Evaluate Model Errors & Goodness-of-Fit
        w_pred = evaluate_raw_svi(log_k, best_params)
        iv_pred = np.sqrt(np.maximum(0.0, w_pred) / t)

        iv_errors = iv_clean - iv_pred
        rmse = float(np.sqrt(np.mean(iv_errors ** 2)))
        mae = float(np.mean(np.abs(iv_errors)))
        max_error = float(np.max(np.abs(iv_errors)))

        # 7. No-Arbitrage Verification
        is_arb_free, arb_violations = best_params.verify_no_arbitrage(t)
        if not is_arb_free:
            warnings.extend(arb_violations)

        # 8. Tiered Quality Classification
        if rmse <= self.config.tier1_rmse_threshold and is_arb_free and opt_status == CalibrationStatus.CONVERGED:
            quality_tier = CalibrationQualityTier.TIER_1_HIGH_PRECISION
        elif rmse <= self.config.tier2_rmse_threshold:
            quality_tier = CalibrationQualityTier.TIER_2_ACCEPTABLE
            if opt_status == CalibrationStatus.CONVERGED:
                opt_status = CalibrationStatus.NOISY_FIT
        else:
            quality_tier = CalibrationQualityTier.TIER_3_FAILED
            opt_status = CalibrationStatus.CALIB_WARNING if is_arb_free else CalibrationStatus.CALIB_FAILED

        return SviCalibrationResult(
            parameters=best_params,
            status=opt_status,
            quality_tier=quality_tier,
            rmse=rmse,
            mae=mae,
            max_error=max_error,
            strikes_used=n_strikes,
            optimization_iterations=iterations,
            as_of_timestamp=as_of_timestamp,
            expiry_date=expiry_date,
            time_to_expiry_years=t,
            warnings=warnings,
        )

    def _fit_zeliade(
        self,
        *,
        log_k: np.ndarray,
        w_obs: np.ndarray,
        weights: np.ndarray,
        t: float,
    ) -> tuple[SviParameters, CalibrationStatus, int, list[str]]:
        """Zeliade 2D quasi-explicit solver."""
        warnings: list[str] = []
        sqrt_weights = np.sqrt(weights)
        lee_bound = 4.0 / max(1e-6, t)

        # Initial guess for (m, sigma)
        # m0: moneyness of minimum variance, clamped near 0.0
        min_idx = int(np.argmin(w_obs))
        m0 = float(np.clip(log_k[min_idx], -0.2, 0.2))
        sigma0 = 0.10

        # Objective function for outer 2D search over (m, sigma)
        def objective(x: np.ndarray) -> float:
            m_val = float(x[0])
            sigma_val = max(1e-4, float(x[1]))

            # Inner linear system: w(k) = a + b*y1 + c*y2
            # y1 = sqrt((k - m)^2 + sigma^2), y2 = k - m
            diff = log_k - m_val
            y1 = np.sqrt(diff * diff + sigma_val * sigma_val)
            y2 = diff

            # Design matrix A = [1, y1, y2]
            n = len(log_k)
            A = np.column_stack([np.ones(n), y1, y2])
            A_weighted = A * sqrt_weights[:, np.newaxis]
            b_weighted = w_obs * sqrt_weights

            # Linear bounds for [a, b, c]:
            # a >= -1.0, b >= 0, c in [-b, b] -> approximate bounds:
            # lb = [-1.0, 0.0, -lee_bound], ub = [10.0, lee_bound, lee_bound]
            lb = np.array([-0.5, 0.0, -lee_bound], dtype=np.float64)
            ub = np.array([5.0, lee_bound, lee_bound], dtype=np.float64)

            try:
                res_linear = lsq_linear(A_weighted, b_weighted, bounds=(lb, ub), method="bvls", max_iter=30)
                if not res_linear.success:
                    return 1e6
                a_fit, b_fit, c_fit = res_linear.x

                # Enforce |c| <= b (i.e. |rho| <= 1)
                if b_fit > 1e-8:
                    c_fit = np.clip(c_fit, -0.999 * b_fit, 0.999 * b_fit)
                else:
                    c_fit = 0.0

                # Non-negativity constraint penalty: a + b*sigma*sqrt(1 - (c/b)^2) >= 0
                rho_fit = c_fit / b_fit if b_fit > 1e-8 else 0.0
                min_w = a_fit + b_fit * sigma_val * math.sqrt(max(0.0, 1.0 - rho_fit * rho_fit))
                penalty = 0.0
                if min_w < 0.0:
                    penalty = 1000.0 * (min_w ** 2)

                # Weighted sum of squared errors
                w_pred = a_fit + b_fit * y1 + c_fit * y2
                sse = float(np.sum(weights * ((w_obs - w_pred) ** 2))) + penalty
                return sse
            except Exception:
                return 1e6

        # Outer 2D Bounded Search
        bounds = [
            (float(np.min(log_k)) - 0.3, float(np.max(log_k)) + 0.3),  # m bounds
            (1e-4, 1.5),                                              # sigma bounds
        ]

        opt_res = minimize(
            objective,
            x0=np.array([m0, sigma0]),
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": 200, "xatol": 1e-5, "fatol": 1e-6},
        )

        m_opt = float(opt_res.x[0])
        sigma_opt = max(1e-4, float(opt_res.x[1]))

        # Recompute optimal linear parameters (a, b, c) at optimal (m, sigma)
        diff = log_k - m_opt
        y1 = np.sqrt(diff * diff + sigma_opt * sigma_opt)
        y2 = diff
        n = len(log_k)
        A = np.column_stack([np.ones(n), y1, y2])
        A_weighted = A * sqrt_weights[:, np.newaxis]
        b_weighted = w_obs * sqrt_weights
        lb = np.array([-0.5, 0.0, -lee_bound], dtype=np.float64)
        ub = np.array([5.0, lee_bound, lee_bound], dtype=np.float64)

        res_linear = lsq_linear(A_weighted, b_weighted, bounds=(lb, ub), method="bvls")
        a_opt, b_opt, c_opt = res_linear.x

        if b_opt > 1e-8:
            rho_opt = float(np.clip(c_opt / b_opt, -0.999, 0.999))
        else:
            rho_opt = 0.0

        # Final non-negativity correction on a if needed
        min_w = a_opt + b_opt * sigma_opt * math.sqrt(max(0.0, 1.0 - rho_opt * rho_opt))
        if min_w < 0.0:
            a_opt -= min_w  # Shift a upwards to guarantee non-negativity
            warnings.append(f"NON_NEGATIVITY_LEVEL_SHIFT: Shifted a by {-min_w:.6f}")

        status = CalibrationStatus.CONVERGED if opt_res.success else CalibrationStatus.ASYMPTOTIC_BOUND
        if not opt_res.success:
            warnings.append(f"OPTIMIZER_WARNING: {opt_res.message}")

        params = SviParameters(
            a=float(a_opt),
            b=float(b_opt),
            rho=float(rho_opt),
            m=float(m_opt),
            sigma=float(sigma_opt),
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
        strikes_used: int = 0,
    ) -> SviCalibrationResult:
        """Construct a clean, zeroed SviCalibrationResult for unavailable / failed data slices."""
        return SviCalibrationResult(
            parameters=SviParameters(a=0.0, b=0.0, rho=0.0, m=0.0, sigma=0.0),
            status=status,
            quality_tier=CalibrationQualityTier.TIER_3_FAILED,
            rmse=1.0,
            mae=1.0,
            max_error=1.0,
            strikes_used=strikes_used,
            optimization_iterations=0,
            as_of_timestamp=as_of_timestamp,
            expiry_date=expiry_date,
            time_to_expiry_years=time_to_expiry_years,
            warnings=[reason],
        )
