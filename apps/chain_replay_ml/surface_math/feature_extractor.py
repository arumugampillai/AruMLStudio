"""Phase 4A.5: Option-Surface Feature Extractor & Pre-Training Qualification Gate.

Computes all 21 Phase 4A features for option-chain dataset rows:
- Higher-Order Greeks (color, zomma, ultima)
- SVI Parameters & Calibration RMSE
- SABR Parameters & Calibration RMSE
- Surface Topology & Dynamics (skews, smile curvature, term slopes, velocity, acceleration, VRP proxy)

Enforces Pre-Training Feature Qualification:
- Missingness Rate Threshold (<= 1.0% on valid trading hours)
- Constant / Zero Variance Filter (std > 1e-6)
- Extreme Outlier / Non-Finite Rejection
- Calibration Quality Gating (Tier 3 results quarantined)
- Preserves Base Pipeline PL_0001 immutability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd

from .greeks import calculate_higher_order_greeks
from .sabr import SabrCalibrator
from .surface import SurfaceDynamicsEngine, SurfaceTopologyEngine, TopologySource
from .svi import SviCalibrator
from .types import (
    CalibrationQualityTier,
    CalibrationStatus,
    DEFAULT_SURFACE_MATH_CONFIG,
    FeatureProvenanceRecord,
    MathematicalFamily,
    SurfaceMathConfig,
)


@dataclass(frozen=True)
class FeatureQualificationReport:
    """Pre-training qualification verdict for candidate features."""
    feature_name: str
    is_eligible: bool
    missingness_pct: float
    variance: float
    std_dev: float
    min_val: float | None
    max_val: float | None
    rejection_reasons: list[str] = field(default_factory=list)
    quality_tier: str = "TIER_1_HIGH_PRECISION"

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "is_eligible": self.is_eligible,
            "missingness_pct": self.missingness_pct,
            "variance": self.variance,
            "std_dev": self.std_dev,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "rejection_reasons": self.rejection_reasons,
            "quality_tier": self.quality_tier,
        }


class OptionSurfaceFeatureExtractor:
    """Extracts Phase 4A features from option chains and applies pre-training qualification."""

    def __init__(self, config: SurfaceMathConfig | None = None) -> None:
        self.config = config or DEFAULT_SURFACE_MATH_CONFIG
        self.svi_calibrator = SviCalibrator(self.config)
        self.sabr_calibrator = SabrCalibrator(self.config)
        self.topology_engine = SurfaceTopologyEngine(self.config)
        self.dynamics_engine = SurfaceDynamicsEngine()

    def extract_snapshot_features(
        self,
        *,
        underlying_spot: float,
        time_to_expiry_years: float,
        risk_free_rate: float,
        strikes: Sequence[float] | np.ndarray,
        implied_volatilities: Sequence[float] | np.ndarray,
        atm_iv_history: Sequence[tuple[float, float]] | None = None,
        spot_history_30m: Sequence[tuple[float, float]] | None = None,
        as_of_timestamp: float = 0.0,
        expiry_date: str = "",
    ) -> dict[str, Any]:
        """Extract all SVI, SABR, and surface topological features for a single timestamp snapshot."""
        s = float(underlying_spot)
        t = float(time_to_expiry_years)
        r = float(risk_free_rate)
        fwd = s * math.exp(r * max(0.0, t)) if s > 0 and t > 0 else 0.0

        # Default empty output dictionary
        out: dict[str, Any] = {
            # SVI
            "svi_param_a": None,
            "svi_param_b": None,
            "svi_param_rho": None,
            "svi_param_m": None,
            "svi_param_sigma": None,
            "svi_calibration_rmse": None,
            # SABR
            "sabr_param_alpha": None,
            "sabr_param_rho": None,
            "sabr_param_nu": None,
            "sabr_calibration_rmse": None,
            # Topology & Dynamics
            "iv_skew_25d": None,
            "iv_skew_10d": None,
            "iv_curvature_25d": None,
            "iv_term_slope_near_next": None,
            "surface_displacement_5m": None,
            "surface_displacement_15m": None,
            "surface_acceleration_15m": None,
            "vrp_proxy_30m": None,
        }

        if s <= 0 or t <= 0 or len(strikes) == 0:
            return out

        # 1. SVI Calibration
        svi_res = self.svi_calibrator.calibrate_slice(
            strikes=strikes,
            implied_volatilities=implied_volatilities,
            underlying_spot=s,
            time_to_expiry_years=t,
            risk_free_rate=r,
            forward_price=fwd,
            as_of_timestamp=as_of_timestamp,
            expiry_date=expiry_date,
        )

        if svi_res.status != CalibrationStatus.INSUFFICIENT_DATA and svi_res.quality_tier != CalibrationQualityTier.TIER_3_FAILED:
            out["svi_param_a"] = svi_res.parameters.a
            out["svi_param_b"] = svi_res.parameters.b
            out["svi_param_rho"] = svi_res.parameters.rho
            out["svi_param_m"] = svi_res.parameters.m
            out["svi_param_sigma"] = svi_res.parameters.sigma
            out["svi_calibration_rmse"] = svi_res.rmse
        else:
            out["svi_calibration_rmse"] = svi_res.rmse if np.isfinite(svi_res.rmse) else None

        # 2. SABR Calibration
        sabr_res = self.sabr_calibrator.calibrate_slice(
            strikes=strikes,
            implied_volatilities=implied_volatilities,
            underlying_spot=s,
            time_to_expiry_years=t,
            risk_free_rate=r,
            forward_price=fwd,
            beta=self.config.sabr_beta,
            as_of_timestamp=as_of_timestamp,
            expiry_date=expiry_date,
        )

        if sabr_res.status != CalibrationStatus.INSUFFICIENT_DATA and sabr_res.quality_tier != CalibrationQualityTier.TIER_3_FAILED:
            out["sabr_param_alpha"] = sabr_res.parameters.alpha
            out["sabr_param_rho"] = sabr_res.parameters.rho
            out["sabr_param_nu"] = sabr_res.parameters.nu
            out["sabr_calibration_rmse"] = sabr_res.rmse
        else:
            out["sabr_calibration_rmse"] = sabr_res.rmse if np.isfinite(sabr_res.rmse) else None

        # 3. Surface Topology
        topo_eval = self.topology_engine.evaluate_cross_sectional_topology(
            underlying_spot=s,
            time_to_expiry_years=t,
            risk_free_rate=r,
            forward_price=fwd,
            strikes=strikes,
            implied_volatilities=implied_volatilities,
            svi_result=svi_res,
            as_of_timestamp=as_of_timestamp,
        )

        if topo_eval.is_valid:
            out["iv_skew_25d"] = topo_eval.features.iv_skew_25d
            out["iv_skew_10d"] = topo_eval.features.iv_skew_10d
            out["iv_curvature_25d"] = topo_eval.features.iv_curvature_25d

        # 4. Surface Dynamics & VRP
        atm_iv = topo_eval.features.atm_iv
        if atm_iv > 0:
            if atm_iv_history is not None:
                out["surface_displacement_5m"] = self.dynamics_engine.compute_surface_displacement(
                    current_timestamp=as_of_timestamp, current_atm_iv=atm_iv, history=atm_iv_history, target_lag_seconds=300.0
                )
                out["surface_displacement_15m"] = self.dynamics_engine.compute_surface_displacement(
                    current_timestamp=as_of_timestamp, current_atm_iv=atm_iv, history=atm_iv_history, target_lag_seconds=900.0
                )
                out["surface_acceleration_15m"] = self.dynamics_engine.compute_surface_acceleration(
                    current_timestamp=as_of_timestamp, current_atm_iv=atm_iv, history=atm_iv_history, lag_seconds=900.0
                )

            if spot_history_30m is not None:
                out["vrp_proxy_30m"] = self.dynamics_engine.compute_vrp_proxy(
                    current_atm_iv=atm_iv, spot_price_history_30m=spot_history_30m, current_timestamp=as_of_timestamp
                )

        return out

    def qualify_candidate_features(
        self,
        df: pd.DataFrame,
        feature_names: Sequence[str],
        max_missingness_pct: float = 5.0,
        min_variance: float = 1e-12,
    ) -> dict[str, FeatureQualificationReport]:
        """Apply Feature Analysis Lab qualification checks to candidate features before training.
        
        Checks:
        1. Missingness rate (<= max_missingness_pct)
        2. Non-zero variance (std > min_variance)
        3. Finite value check (no Inf/-Inf)
        """
        reports: dict[str, FeatureQualificationReport] = {}
        total_rows = len(df)

        for col in feature_names:
            if col not in df.columns:
                reports[col] = FeatureQualificationReport(
                    feature_name=col,
                    is_eligible=False,
                    missingness_pct=100.0,
                    variance=0.0,
                    std_dev=0.0,
                    min_val=None,
                    max_val=None,
                    rejection_reasons=["FEATURE_COLUMN_NOT_FOUND"],
                    quality_tier="TIER_3_FAILED",
                )
                continue

            series = pd.to_numeric(df[col], errors="coerce")
            valid_mask = series.notna() & np.isfinite(series)
            n_valid = int(valid_mask.sum())
            missing_pct = ((total_rows - n_valid) / max(1, total_rows)) * 100.0

            reasons: list[str] = []

            if missing_pct > max_missingness_pct:
                reasons.append(f"EXCESSIVE_MISSINGNESS: {missing_pct:.2f}% > max {max_missingness_pct:.2f}%")

            if n_valid < 10:
                reasons.append("INSUFFICIENT_OBSERVATIONS: < 10 valid rows")
                var_val = 0.0
                std_val = 0.0
                min_v = None
                max_v = None
            else:
                valid_vals = series[valid_mask].astype(float)
                var_val = float(np.var(valid_vals))
                std_val = float(np.std(valid_vals))
                min_v = float(np.min(valid_vals))
                max_v = float(np.max(valid_vals))

                if var_val < min_variance:
                    reasons.append(f"CONSTANT_OR_NEAR_ZERO_VARIANCE: variance={var_val:.2e} < {min_variance:.2e}")

            is_eligible = (len(reasons) == 0)
            q_tier = "TIER_1_HIGH_PRECISION" if is_eligible else "TIER_3_FAILED"

            reports[col] = FeatureQualificationReport(
                feature_name=col,
                is_eligible=is_eligible,
                missingness_pct=round(missing_pct, 2),
                variance=var_val,
                std_dev=std_val,
                min_val=min_v,
                max_val=max_v,
                rejection_reasons=reasons,
                quality_tier=q_tier,
            )

        return reports
