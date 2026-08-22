"""Phase 4A: Advanced Option-Surface Mathematics Data Contracts & Types.

Defines authoritative data contracts, calibration containers, quality governance enums,
provenance schemas, and configuration for higher-order Greeks, SVI, and SABR surface models.
Adheres strictly to Doc 18 v1.1.0 specification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
from typing import Any, Sequence


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# =============================================================================
# ENUMS
# =============================================================================

class MathematicalFamily(str, Enum):
    """Categorical mathematical family for Phase 4A feature primitives."""
    HIGHER_ORDER_GREEKS = "HIGHER_ORDER_GREEKS"
    SVI_SURFACE = "SVI_SURFACE"
    SABR_SURFACE = "SABR_SURFACE"
    SURFACE_TOPOLOGY = "SURFACE_TOPOLOGY"
    SURFACE_DYNAMICS = "SURFACE_DYNAMICS"


class CalibrationStatus(str, Enum):
    """Convergence and numerical status of parametric surface calibration."""
    CONVERGED = "CONVERGED"
    ASYMPTOTIC_BOUND = "ASYMPTOTIC_BOUND"
    FALLBACK_LINEAR = "FALLBACK_LINEAR"
    NOISY_FIT = "NOISY_FIT"
    CALIB_WARNING = "CALIB_WARNING"
    CALIB_FAILED = "CALIB_FAILED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class CalibrationQualityTier(str, Enum):
    """Tiered governance for calibration goodness-of-fit error (RMSE)."""
    TIER_1_HIGH_PRECISION = "TIER_1_HIGH_PRECISION"  # RMSE <= 0.03 (<= 3.0% vol error)
    TIER_2_ACCEPTABLE = "TIER_2_ACCEPTABLE"          # 0.03 < RMSE <= 0.06 (3.0% - 6.0% vol error)
    TIER_3_FAILED = "TIER_3_FAILED"                  # RMSE > 0.06 or non-converged


class SabrBetaMode(str, Enum):
    """Elasticity beta parameter mode for SABR model."""
    CIR_SQUARE_ROOT = "CIR_SQUARE_ROOT"  # beta = 0.5 (Default for equity index options)
    LOGNORMAL = "LOGNORMAL"              # beta = 1.0 (Pure lognormal diffusion)
    CUSTOM = "CUSTOM"                    # Custom user-defined beta


# =============================================================================
# HIGHER-ORDER GREEKS DATA CONTRACT
# =============================================================================

@dataclass(frozen=True)
class HigherOrderGreeksRecord:
    """Analytical first-, second-, and third-order option Greeks container.
    
    Emitted units align with `bs.py` conventions:
    - `vega`: ₹ per 1% IV point (vega_raw / 100.0)
    - `volga`: ₹ per 1% IV point per 1% IV point (vega * d1 * d2 / sigma)
    - `vanna`: Dimensionless d(Delta)/d(sigma)
    - `charm`: Delta decay per calendar day (annualized / 365.0)
    - `color`: Gamma decay per calendar day (annualized / 365.0)
    - `speed`: d(Gamma)/dS (1 / ₹)
    - `zomma`: d(Gamma)/d(sigma) (1 / vol point)
    - `ultima`: d(Volga)/d(sigma) (₹ per vol point^3)
    """
    delta: float
    gamma: float
    theta: float
    vega: float
    vanna: float
    volga: float
    charm: float
    color: float
    speed: float
    zomma: float
    ultima: float
    strike: float
    time_to_expiry_years: float
    implied_volatility: float
    underlying_spot: float
    option_type: str = "CE"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HigherOrderGreeksRecord:
        return cls(**d)


# =============================================================================
# RAW SVI (STOCHASTIC VOLATILITY INSPIRED) DATA CONTRACTS
# =============================================================================

@dataclass(frozen=True)
class SviParameters:
    """Gatheral Raw SVI 5-parameter total variance container.
    
    Formula: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))
    where k = ln(K / F_T) is log-moneyness.
    """
    a: float        # Overall variance level (vertical translation)
    b: float        # Asymptote angle / slope (b >= 0)
    rho: float      # Smile orientation / skew (-1 < rho < 1)
    m: float        # Vertex horizontal shift (moneyness translation)
    sigma: float    # Vertex curvature / smoothness (sigma > 0)

    def total_variance(self, k: float) -> float:
        """Compute total implied variance w(k)."""
        diff = k - self.m
        val = self.a + self.b * (self.rho * diff + math.sqrt(diff * diff + self.sigma * self.sigma))
        return max(0.0, val)

    def implied_volatility(self, k: float, time_to_expiry_years: float) -> float:
        """Compute annualized implied volatility sigma(k, T) from total variance."""
        t = max(1e-6, float(time_to_expiry_years))
        w = self.total_variance(k)
        return math.sqrt(w / t)

    def verify_no_arbitrage(self, time_to_expiry_years: float) -> tuple[bool, list[str]]:
        """Verify strict geometric no-arbitrage bounds."""
        violations: list[str] = []
        if self.b < 0:
            violations.append(f"NEGATIVE_B_SLOPE: b={self.b} < 0")
        if abs(self.rho) >= 1.0:
            violations.append(f"INVALID_RHO_RANGE: |rho|={abs(self.rho)} >= 1")
        if self.sigma <= 0:
            violations.append(f"NON_POSITIVE_SIGMA: sigma={self.sigma} <= 0")

        # Non-negativity condition: min(w(k)) = a + b * sigma * sqrt(1 - rho^2) >= 0
        if abs(self.rho) < 1.0 and self.sigma > 0:
            min_w = self.a + self.b * self.sigma * math.sqrt(1.0 - self.rho * self.rho)
            if min_w < -1e-7:
                violations.append(f"NEGATIVE_TOTAL_VARIANCE_VERTEX: min_w={min_w:.6f} < 0")

        # Lee's moment boundary (wing slope condition)
        t = max(1e-6, float(time_to_expiry_years))
        lee_bound = 4.0 / t
        wing_slope = self.b * (1.0 + abs(self.rho))
        if wing_slope >= lee_bound:
            violations.append(f"LEE_WING_SLOPE_VIOLATION: b*(1+|rho|)={wing_slope:.4f} >= 4/T={lee_bound:.4f}")

        is_valid = (len(violations) == 0)
        return is_valid, violations

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SviParameters:
        return cls(**d)


@dataclass(frozen=True)
class SviCalibrationResult:
    """Full calibration dossier and diagnostic container for SVI surface fit."""
    parameters: SviParameters
    status: CalibrationStatus
    quality_tier: CalibrationQualityTier
    rmse: float
    mae: float
    max_error: float
    strikes_used: int
    optimization_iterations: int
    as_of_timestamp: float
    expiry_date: str
    time_to_expiry_years: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["parameters"] = self.parameters.to_dict()
        d["status"] = self.status.value
        d["quality_tier"] = self.quality_tier.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SviCalibrationResult:
        p_dict = d.get("parameters", {})
        params = SviParameters.from_dict(p_dict)
        return cls(
            parameters=params,
            status=CalibrationStatus(d.get("status", CalibrationStatus.CONVERGED.value)),
            quality_tier=CalibrationQualityTier(d.get("quality_tier", CalibrationQualityTier.TIER_1_HIGH_PRECISION.value)),
            rmse=float(d.get("rmse", 0.0)),
            mae=float(d.get("mae", 0.0)),
            max_error=float(d.get("max_error", 0.0)),
            strikes_used=int(d.get("strikes_used", 0)),
            optimization_iterations=int(d.get("optimization_iterations", 0)),
            as_of_timestamp=float(d.get("as_of_timestamp", 0.0)),
            expiry_date=str(d.get("expiry_date", "")),
            time_to_expiry_years=float(d.get("time_to_expiry_years", 0.0)),
            warnings=list(d.get("warnings", [])),
        )


# =============================================================================
# SABR VOLATILITY MODEL DATA CONTRACTS
# =============================================================================

@dataclass(frozen=True)
class SabrParameters:
    """Hagan et al. SABR stochastic volatility parameter container."""
    alpha: float    # Initial volatility / ATM scale (alpha > 0)
    beta: float     # Elasticity parameter (beta in [0, 1])
    rho: float      # Asset-vol correlation / skew (-1 < rho < 1)
    nu: float       # Volatility of volatility / smile curvature (nu > 0)

    def implied_volatility(self, strike: float, forward: float, time_to_expiry_years: float) -> float:
        """Evaluate closed-form SABR implied lognormal volatility sigma_SABR(K, F)."""
        K = max(1e-4, float(strike))
        F = max(1e-4, float(forward))
        T = max(1e-6, float(time_to_expiry_years))
        alpha = max(1e-6, self.alpha)
        beta = self.beta
        rho = max(-0.999, min(0.999, self.rho))
        nu = max(1e-6, self.nu)

        # ATM special case (K == F)
        if abs(K - F) < 1e-5:
            f_pow = math.pow(F, 1.0 - beta)
            term1 = alpha / f_pow
            term2 = (
                ((1.0 - beta) ** 2 / 24.0) * (alpha ** 2 / math.pow(F, 2.0 * (1.0 - beta)))
                + (0.25 * rho * beta * nu * alpha / f_pow)
                + ((2.0 - 3.0 * rho ** 2) / 24.0) * (nu ** 2)
            ) * T
            return max(1e-4, term1 * (1.0 + term2))

        # General OTM / ITM expansion
        log_fk = math.log(F / K)
        fk_mid = math.pow(F * K, (1.0 - beta) / 2.0)
        denom_pow = 1.0 + ((1.0 - beta) ** 2 / 24.0) * (log_fk ** 2) + ((1.0 - beta) ** 4 / 1920.0) * (log_fk ** 4)
        z = (nu / alpha) * fk_mid * log_fk

        if abs(z) < 1e-5:
            z_over_chi = 1.0
        else:
            sqrt_term = math.sqrt(max(0.0, 1.0 - 2.0 * rho * z + z * z))
            chi_z = math.log(max(1e-12, (sqrt_term + z - rho) / (1.0 - rho)))
            z_over_chi = z / chi_z if abs(chi_z) > 1e-12 else 1.0

        higher_order = (
            ((1.0 - beta) ** 2 / 24.0) * (alpha ** 2 / math.pow(F * K, 1.0 - beta))
            + (0.25 * rho * beta * nu * alpha / fk_mid)
            + ((2.0 - 3.0 * rho ** 2) / 24.0) * (nu ** 2)
        ) * T

        sigma_sabr = (alpha / (fk_mid * denom_pow)) * z_over_chi * (1.0 + higher_order)
        return max(1e-4, sigma_sabr)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SabrParameters:
        return cls(**d)


@dataclass(frozen=True)
class SabrCalibrationResult:
    """Full calibration dossier and diagnostic container for SABR model fit."""
    parameters: SabrParameters
    status: CalibrationStatus
    quality_tier: CalibrationQualityTier
    rmse: float
    mae: float
    strikes_used: int
    forward_price: float
    atm_implied_volatility: float
    as_of_timestamp: float
    expiry_date: str
    time_to_expiry_years: float
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["parameters"] = self.parameters.to_dict()
        d["status"] = self.status.value
        d["quality_tier"] = self.quality_tier.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SabrCalibrationResult:
        p_dict = d.get("parameters", {})
        params = SabrParameters.from_dict(p_dict)
        return cls(
            parameters=params,
            status=CalibrationStatus(d.get("status", CalibrationStatus.CONVERGED.value)),
            quality_tier=CalibrationQualityTier(d.get("quality_tier", CalibrationQualityTier.TIER_1_HIGH_PRECISION.value)),
            rmse=float(d.get("rmse", 0.0)),
            mae=float(d.get("mae", 0.0)),
            strikes_used=int(d.get("strikes_used", 0)),
            forward_price=float(d.get("forward_price", 0.0)),
            atm_implied_volatility=float(d.get("atm_implied_volatility", 0.0)),
            as_of_timestamp=float(d.get("as_of_timestamp", 0.0)),
            expiry_date=str(d.get("expiry_date", "")),
            time_to_expiry_years=float(d.get("time_to_expiry_years", 0.0)),
            warnings=list(d.get("warnings", [])),
        )


# =============================================================================
# SURFACE-DERIVED TOPOLOGICAL FEATURES CONTAINER
# =============================================================================

@dataclass(frozen=True)
class SurfaceTopologicalFeatures:
    """Evaluated cross-sectional skew, curvature, term structure, and volatility dynamics."""
    iv_skew_25d: float | None
    iv_skew_10d: float | None
    iv_curvature_25d: float | None
    iv_term_slope_near_next: float | None
    surface_displacement_5m: float | None
    surface_displacement_15m: float | None
    surface_acceleration_15m: float | None
    vrp_proxy_30m: float | None
    atm_iv: float
    forward_price: float
    as_of_timestamp: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SurfaceTopologicalFeatures:
        return cls(**d)


# =============================================================================
# PROVENANCE & LINEAGE CONTAINER
# =============================================================================

@dataclass(frozen=True)
class FeatureProvenanceRecord:
    """Cryptographic provenance and mathematical specification record for a Phase 4A feature."""
    feature_name: str
    mathematical_family: MathematicalFamily
    formula_expression: str
    source_fields: list[str]
    source_snapshot_hash: str
    calibration_rmse: float | None = None
    calibration_status: CalibrationStatus | None = None
    calibration_tier: CalibrationQualityTier | None = None
    units: str = ""
    implementation_version: str = "1.1.0"
    created_at: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mathematical_family"] = self.mathematical_family.value
        d["calibration_status"] = self.calibration_status.value if self.calibration_status else None
        d["calibration_tier"] = self.calibration_tier.value if self.calibration_tier else None
        return d

    def compute_provenance_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> FeatureProvenanceRecord:
        fam = MathematicalFamily(d["mathematical_family"])
        st = CalibrationStatus(d["calibration_status"]) if d.get("calibration_status") else None
        tier = CalibrationQualityTier(d["calibration_tier"]) if d.get("calibration_tier") else None
        return cls(
            feature_name=d["feature_name"],
            mathematical_family=fam,
            formula_expression=d["formula_expression"],
            source_fields=list(d.get("source_fields", [])),
            source_snapshot_hash=d.get("source_snapshot_hash", ""),
            calibration_rmse=d.get("calibration_rmse"),
            calibration_status=st,
            calibration_tier=tier,
            units=d.get("units", ""),
            implementation_version=d.get("implementation_version", "1.1.0"),
            created_at=d.get("created_at", _utc_now_iso()),
        )


# =============================================================================
# SURFACE MATH CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class SurfaceMathConfig:
    """Configurable baseline hypothesis parameters for Phase 4A option surface engine.
    
    Default configuration:
    - SABR beta = 0.5 (CIR square-root model, standard for equity index options)
    - 5-minute calibration grid for SVI and SABR surface models
    - Tier 1 RMSE threshold = 0.03, Tier 2 RMSE threshold = 0.06
    - Maximum 4 parallel CPU workers (16 GB workstation safety)
    """
    sabr_beta: float = 0.5
    sabr_beta_mode: SabrBetaMode = SabrBetaMode.CIR_SQUARE_ROOT
    calibration_interval_minutes: int = 5
    min_liquid_strikes: int = 5
    max_bid_ask_spread_pct: float = 20.0
    min_strike_volume: int = 1
    tier1_rmse_threshold: float = 0.03
    tier2_rmse_threshold: float = 0.06
    max_iv_bound: float = 5.0
    min_iv_bound: float = 0.0001
    min_time_to_expiry_years: float = 1e-6
    vrp_realized_vol_lookback_minutes: int = 30
    enable_linear_fallback: bool = True
    max_cpu_workers: int = 4
    config_version: str = "4A.0.v1.1.0"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["sabr_beta_mode"] = self.sabr_beta_mode.value
        return d

    def compute_config_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> SurfaceMathConfig:
        if not isinstance(d, dict):
            return cls()
        beta_mode_val = d.get("sabr_beta_mode", SabrBetaMode.CIR_SQUARE_ROOT.value)
        beta_mode = SabrBetaMode(beta_mode_val) if isinstance(beta_mode_val, str) else beta_mode_val
        valid_fields = {
            "sabr_beta",
            "calibration_interval_minutes",
            "min_liquid_strikes",
            "max_bid_ask_spread_pct",
            "min_strike_volume",
            "tier1_rmse_threshold",
            "tier2_rmse_threshold",
            "max_iv_bound",
            "min_iv_bound",
            "min_time_to_expiry_years",
            "vrp_realized_vol_lookback_minutes",
            "enable_linear_fallback",
            "max_cpu_workers",
            "config_version",
        }
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(sabr_beta_mode=beta_mode, **filtered)


# Canonical Default Configuration Instance
DEFAULT_SURFACE_MATH_CONFIG = SurfaceMathConfig()
