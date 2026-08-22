"""Phase 4A.4: Option-Surface Topology & Dynamic Derivatives Engine.

Calculates cross-sectional volatility skew, smile curvature, term-structure slopes,
and backward-looking volatility dynamics (velocity, acceleration, VRP proxy).

Key Features:
- iv_skew_25d: 25-Delta Put IV minus 25-Delta Call IV (downside risk premium)
- iv_skew_10d: 10-Delta Put IV minus 10-Delta Call IV (tail risk asymmetry)
- iv_curvature_25d: Smile butterfly (average of 25d wings minus ATM IV)
- iv_term_slope_near_next: Term-structure slope between near and next expiry slices
- surface_displacement_5m / 15m: Backward-looking volatility shock velocity
- surface_acceleration_15m: Second-order backward-looking volatility convexity
- vrp_proxy_30m: Annualized Variance Risk Premium proxy (IV^2 - RV_30m^2)

Strict Quality Gating:
- Parametric SVI / SABR evaluations require Tier 1 or Tier 2 calibration.
- Tier 3 calibrations are rejected for parametric topology, falling back to direct empirical chain interpolation.
- Zero future look-ahead data leakage guaranteed.

Conforms strictly to Doc 18 v1.1.0 specifications and Phase 4A.0 contracts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Sequence
import numpy as np

from .types import (
    CalibrationQualityTier,
    CalibrationStatus,
    DEFAULT_SURFACE_MATH_CONFIG,
    SurfaceMathConfig,
    SurfaceTopologicalFeatures,
    SviCalibrationResult,
    SviParameters,
)
from .svi import evaluate_raw_svi, evaluate_svi_implied_volatility

# Annualization factors for intraday Indian equity markets
MINUTES_PER_TRADING_DAY = 375.0
TRADING_DAYS_PER_YEAR = 252.0
ANNUAL_FACTOR_MINUTES = MINUTES_PER_TRADING_DAY * TRADING_DAYS_PER_YEAR


class TopologySource(str, Enum):
    """Source method used to evaluate surface topology."""
    PARAMETRIC_SVI = "PARAMETRIC_SVI"
    PARAMETRIC_SABR = "PARAMETRIC_SABR"
    EMPIRICAL_CHAIN = "EMPIRICAL_CHAIN"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class SurfaceTopologyEvaluation:
    """Full surface topological evaluation dossier with quality flags and source provenance."""
    features: SurfaceTopologicalFeatures
    source: TopologySource
    quality_tier: CalibrationQualityTier
    is_valid: bool
    quality_flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["features"] = self.features.to_dict()
        d["source"] = self.source.value
        d["quality_tier"] = self.quality_tier.value
        return d


# =============================================================================
# DELTA-BASED STRIKE LOCATOR (EMPIRICAL & ANALYTICAL)
# =============================================================================

def find_strike_for_delta(
    *,
    target_delta: float,
    is_call: bool,
    underlying_spot: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    strikes: np.ndarray,
    implied_vols: np.ndarray,
) -> tuple[float | None, float | None]:
    """Find strike and interpolated implied volatility for a target Black-Scholes Delta.
    
    Returns (strike, implied_vol) or (None, None) if target delta is outside the chain.
    """
    s = float(underlying_spot)
    t = float(time_to_expiry_years)
    r = float(risk_free_rate)
    target_d = float(target_delta)

    if s <= 0 or t <= 0 or len(strikes) < 3:
        return None, None

    sqrt_t = math.sqrt(t)
    deltas = np.zeros_like(strikes, dtype=np.float64)

    for i, (k, iv) in enumerate(zip(strikes, implied_vols)):
        if k <= 0 or iv <= 0:
            deltas[i] = np.nan
            continue
        d1 = (math.log(s / k) + (r + 0.5 * iv * iv) * t) / (iv * sqrt_t)
        norm_cdf = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
        deltas[i] = norm_cdf if is_call else norm_cdf - 1.0

    # Filter out NaNs and sort by strike
    valid = np.isfinite(deltas)
    if np.sum(valid) < 3:
        return None, None

    k_valid = strikes[valid]
    iv_valid = implied_vols[valid]
    d_valid = deltas[valid]

    # Sort monotonically
    sort_idx = np.argsort(k_valid)
    k_sorted = k_valid[sort_idx]
    iv_sorted = iv_valid[sort_idx]
    d_sorted = d_valid[sort_idx]

    # Check if target delta is bracketed
    min_d = np.min(d_sorted)
    max_d = np.max(d_sorted)

    if target_d < min_d or target_d > max_d:
        return None, None

    # Interpolate strike and IV corresponding to target delta
    # Since call delta is monotonically decreasing with strike, sort by delta for interp
    d_sort_idx = np.argsort(d_sorted)
    target_k = float(np.interp(target_d, d_sorted[d_sort_idx], k_sorted[d_sort_idx]))
    target_iv = float(np.interp(target_d, d_sorted[d_sort_idx], iv_sorted[d_sort_idx]))

    return target_k, target_iv


# =============================================================================
# SURFACE TOPOLOGY EVALUATOR
# =============================================================================

class SurfaceTopologyEngine:
    """Calculates cross-sectional volatility skew, smile curvature, and term-structure slope."""

    def __init__(self, config: SurfaceMathConfig | None = None) -> None:
        self.config = config or DEFAULT_SURFACE_MATH_CONFIG

    def evaluate_cross_sectional_topology(
        self,
        *,
        underlying_spot: float,
        time_to_expiry_years: float,
        risk_free_rate: float = 0.07,
        forward_price: float | None = None,
        strikes: Sequence[float] | np.ndarray | None = None,
        implied_volatilities: Sequence[float] | np.ndarray | None = None,
        svi_result: SviCalibrationResult | None = None,
        as_of_timestamp: float = 0.0,
    ) -> SurfaceTopologyEvaluation:
        """Evaluate 25d/10d skew and smile curvature with strict calibration quality gating."""
        s = float(underlying_spot)
        t = float(time_to_expiry_years)
        r = float(risk_free_rate)
        fwd = float(forward_price) if forward_price is not None and forward_price > 0 else s * math.exp(r * max(0.0, t))

        flags: list[str] = []

        if s <= 0 or fwd <= 0 or t <= 0:
            return self._empty_evaluation(
                reason="Invalid spot, forward, or time-to-expiry",
                as_of_timestamp=as_of_timestamp,
                forward_price=fwd,
            )

        # 1. Evaluate ATM IV
        atm_iv: float = 0.15  # Fallback default
        if strikes is not None and implied_volatilities is not None and len(strikes) > 0:
            k_arr = np.asarray(strikes, dtype=np.float64)
            iv_arr = np.asarray(implied_volatilities, dtype=np.float64)
            valid = (k_arr > 0) & (iv_arr > 0) & np.isfinite(k_arr) & np.isfinite(iv_arr)
            if np.any(valid):
                k_v = k_arr[valid]
                iv_v = iv_arr[valid]
                atm_idx = int(np.argmin(np.abs(k_v - fwd)))
                atm_iv = float(iv_v[atm_idx])

        # 2. Check SVI Parametric Path (Gated by Quality Tier)
        use_parametric_svi = False
        if svi_result is not None:
            if svi_result.quality_tier in (CalibrationQualityTier.TIER_1_HIGH_PRECISION, CalibrationQualityTier.TIER_2_ACCEPTABLE):
                use_parametric_svi = True
            else:
                flags.append(f"SVI_CALIBRATION_REJECTED: Quality tier is {svi_result.quality_tier.value}")

        skew_25d: float | None = None
        skew_10d: float | None = None
        curvature_25d: float | None = None
        source = TopologySource.UNAVAILABLE
        tier = CalibrationQualityTier.TIER_3_FAILED

        # 3. Path A: Evaluate via Parametric SVI
        if use_parametric_svi and svi_result is not None:
            p = svi_result.parameters
            # Synthetic evaluation of IV across moneyness
            atm_iv_svi = p.implied_volatility(0.0, t)
            atm_iv = atm_iv_svi

            # Solve for 25d and 10d strikes using parametric SVI
            # Fine grid in log-moneyness [-0.5, 0.5]
            k_grid_moneyness = np.linspace(-0.5, 0.5, 201)
            k_grid_strikes = fwd * np.exp(k_grid_moneyness)
            iv_grid = np.array([p.implied_volatility(float(km), t) for km in k_grid_moneyness])

            _, iv_25d_call = find_strike_for_delta(
                target_delta=0.25, is_call=True, underlying_spot=s, time_to_expiry_years=t, risk_free_rate=r,
                strikes=k_grid_strikes, implied_vols=iv_grid
            )
            _, iv_25d_put = find_strike_for_delta(
                target_delta=-0.25, is_call=False, underlying_spot=s, time_to_expiry_years=t, risk_free_rate=r,
                strikes=k_grid_strikes, implied_vols=iv_grid
            )
            _, iv_10d_call = find_strike_for_delta(
                target_delta=0.10, is_call=True, underlying_spot=s, time_to_expiry_years=t, risk_free_rate=r,
                strikes=k_grid_strikes, implied_vols=iv_grid
            )
            _, iv_10d_put = find_strike_for_delta(
                target_delta=-0.10, is_call=False, underlying_spot=s, time_to_expiry_years=t, risk_free_rate=r,
                strikes=k_grid_strikes, implied_vols=iv_grid
            )

            if iv_25d_put is not None and iv_25d_call is not None:
                skew_25d = iv_25d_put - iv_25d_call
                curvature_25d = 0.5 * (iv_25d_call + iv_25d_put) - atm_iv
            if iv_10d_put is not None and iv_10d_call is not None:
                skew_10d = iv_10d_put - iv_10d_call

            source = TopologySource.PARAMETRIC_SVI
            tier = svi_result.quality_tier

        # 4. Path B: Direct Empirical Chain Interpolation Fallback
        elif strikes is not None and implied_volatilities is not None and len(strikes) >= self.config.min_liquid_strikes:
            k_clean = np.asarray(strikes, dtype=np.float64)
            iv_clean = np.asarray(implied_volatilities, dtype=np.float64)
            valid = (k_clean > 0) & (iv_clean > 0) & np.isfinite(k_clean) & np.isfinite(iv_clean)

            if np.sum(valid) >= self.config.min_liquid_strikes:
                k_v = k_clean[valid]
                iv_v = iv_clean[valid]

                _, iv_25d_call = find_strike_for_delta(
                    target_delta=0.25, is_call=True, underlying_spot=s, time_to_expiry_years=t, risk_free_rate=r,
                    strikes=k_v, implied_vols=iv_v
                )
                _, iv_25d_put = find_strike_for_delta(
                    target_delta=-0.25, is_call=False, underlying_spot=s, time_to_expiry_years=t, risk_free_rate=r,
                    strikes=k_v, implied_vols=iv_v
                )
                _, iv_10d_call = find_strike_for_delta(
                    target_delta=0.10, is_call=True, underlying_spot=s, time_to_expiry_years=t, risk_free_rate=r,
                    strikes=k_v, implied_vols=iv_v
                )
                _, iv_10d_put = find_strike_for_delta(
                    target_delta=-0.10, is_call=False, underlying_spot=s, time_to_expiry_years=t, risk_free_rate=r,
                    strikes=k_v, implied_vols=iv_v
                )

                if iv_25d_put is not None and iv_25d_call is not None:
                    skew_25d = iv_25d_put - iv_25d_call
                    curvature_25d = 0.5 * (iv_25d_call + iv_25d_put) - atm_iv
                if iv_10d_put is not None and iv_10d_call is not None:
                    skew_10d = iv_10d_put - iv_10d_call

                source = TopologySource.EMPIRICAL_CHAIN
                tier = CalibrationQualityTier.TIER_2_ACCEPTABLE
                flags.append("EMPIRICAL_CHAIN_EVALUATION")

        if skew_25d is None:
            flags.append("INSUFFICIENT_DELTA_POINTS: Could not bracket 25d wings in strike spectrum")

        features = SurfaceTopologicalFeatures(
            iv_skew_25d=skew_25d,
            iv_skew_10d=skew_10d,
            iv_curvature_25d=curvature_25d,
            iv_term_slope_near_next=None,
            surface_displacement_5m=None,
            surface_displacement_15m=None,
            surface_acceleration_15m=None,
            vrp_proxy_30m=None,
            atm_iv=atm_iv,
            forward_price=fwd,
            as_of_timestamp=as_of_timestamp,
        )

        return SurfaceTopologyEvaluation(
            features=features,
            source=source,
            quality_tier=tier,
            is_valid=(skew_25d is not None),
            quality_flags=flags,
        )

    def calculate_term_structure_slope(
        self,
        *,
        near_expiry_iv: float,
        near_expiry_years: float,
        next_expiry_iv: float,
        next_expiry_years: float,
    ) -> float | None:
        """Calculate term-structure slope: (IV_next - IV_near) / (sqrt(T_next) - sqrt(T_near))."""
        iv1 = float(near_expiry_iv)
        t1 = float(near_expiry_years)
        iv2 = float(next_expiry_iv)
        t2 = float(next_expiry_years)

        if iv1 <= 0 or iv2 <= 0 or t1 <= 1e-6 or t2 <= t1 + 1e-5:
            return None

        diff_sqrt_t = math.sqrt(t2) - math.sqrt(t1)
        if diff_sqrt_t <= 1e-6:
            return None

        return (iv2 - iv1) / diff_sqrt_t

    def _empty_evaluation(
        self,
        reason: str,
        as_of_timestamp: float,
        forward_price: float,
    ) -> SurfaceTopologyEvaluation:
        feats = SurfaceTopologicalFeatures(
            iv_skew_25d=None,
            iv_skew_10d=None,
            iv_curvature_25d=None,
            iv_term_slope_near_next=None,
            surface_displacement_5m=None,
            surface_displacement_15m=None,
            surface_acceleration_15m=None,
            vrp_proxy_30m=None,
            atm_iv=0.0,
            forward_price=forward_price,
            as_of_timestamp=as_of_timestamp,
        )
        return SurfaceTopologyEvaluation(
            features=feats,
            source=TopologySource.UNAVAILABLE,
            quality_tier=CalibrationQualityTier.TIER_3_FAILED,
            is_valid=False,
            quality_flags=[reason],
        )


# =============================================================================
# SURFACE DYNAMICS & VRP ENGINE (STRICTLY BACKWARD-LOOKING)
# =============================================================================

class SurfaceDynamicsEngine:
    """Computes backward-looking volatility velocity, acceleration, and Variance Risk Premium."""

    def __init__(
        self,
        max_lookback_gap_seconds: float = 120.0,
    ) -> None:
        self.max_lookback_gap_seconds = max_lookback_gap_seconds

    def compute_surface_displacement(
        self,
        *,
        current_timestamp: float,
        current_atm_iv: float,
        history: Sequence[tuple[float, float]],  # List of (timestamp, atm_iv)
        target_lag_seconds: float,
    ) -> float | None:
        """Compute backward-looking volatility displacement: IV(t) - IV(t - lag).
        
        Zero look-ahead leakage: only timestamps <= current_timestamp are queried.
        """
        now_ts = float(current_timestamp)
        now_iv = float(current_atm_iv)
        target_ts = now_ts - float(target_lag_seconds)

        if not history:
            return None

        # Filter strictly past history <= now_ts
        past_hist = [(ts, iv) for ts, iv in history if ts <= now_ts]
        if not past_hist:
            return None

        # Find closest timestamp to target_ts
        best_ts, best_iv = min(past_hist, key=lambda item: abs(item[0] - target_ts))
        time_gap = abs(best_ts - target_ts)

        # If closest available timestamp exceeds maximum lookback gap tolerance, return None
        if time_gap > self.max_lookback_gap_seconds:
            return None

        return now_iv - best_iv

    def compute_surface_acceleration(
        self,
        *,
        current_timestamp: float,
        current_atm_iv: float,
        history: Sequence[tuple[float, float]],
        lag_seconds: float = 900.0,  # 15 minutes
    ) -> float | None:
        """Compute backward-looking volatility acceleration: IV(t) - 2*IV(t - lag) + IV(t - 2*lag)."""
        now_ts = float(current_timestamp)
        now_iv = float(current_atm_iv)

        lag1 = self.compute_surface_displacement(
            current_timestamp=now_ts, current_atm_iv=now_iv, history=history, target_lag_seconds=lag_seconds
        )
        lag2 = self.compute_surface_displacement(
            current_timestamp=now_ts, current_atm_iv=now_iv, history=history, target_lag_seconds=2.0 * lag_seconds
        )

        if lag1 is None or lag2 is None:
            return None

        # Accel = (IV_now - IV_lag1) - (IV_lag1 - IV_lag2) = IV_now - 2*IV_lag1 + IV_lag2
        # Since lag1 = IV_now - IV_lag1  => IV_lag1 = IV_now - lag1
        # Since lag2 = IV_now - IV_lag2  => IV_lag2 = IV_now - lag2
        # Accel = IV_now - 2*(IV_now - lag1) + (IV_now - lag2) = 2*lag1 - lag2
        return 2.0 * lag1 - lag2

    def compute_vrp_proxy(
        self,
        *,
        current_atm_iv: float,
        spot_price_history_30m: Sequence[tuple[float, float]],  # List of (timestamp, spot_price)
        current_timestamp: float,
    ) -> float | None:
        """Compute backward-looking Variance Risk Premium proxy: VRP = IV^2 - RealizedVariance_30m.
        
        Realized variance is calculated from 1-minute spot log-returns over the past 30 minutes.
        Zero future data access guaranteed.
        """
        now_ts = float(current_timestamp)
        now_iv = float(current_atm_iv)

        if now_iv <= 0:
            return None

        # Filter strictly past spot prices in window [now_ts - 1800, now_ts]
        window_start = now_ts - 1800.0
        past_spots = [p for ts, p in spot_price_history_30m if window_start <= ts <= now_ts and p > 0]

        if len(past_spots) < 10:
            return None  # Insufficient return samples to estimate realized variance

        prices = np.asarray(past_spots, dtype=np.float64)
        log_returns = np.diff(np.log(prices))

        if len(log_returns) == 0:
            return None

        # Annualized realized variance: sum(r^2) * (ANNUAL_MINUTES / window_minutes)
        window_minutes = max(1.0, (now_ts - window_start) / 60.0)
        rv_annualized = float(np.sum(log_returns ** 2)) * (ANNUAL_FACTOR_MINUTES / window_minutes)

        iv_variance = now_iv * now_iv
        return iv_variance - rv_annualized
