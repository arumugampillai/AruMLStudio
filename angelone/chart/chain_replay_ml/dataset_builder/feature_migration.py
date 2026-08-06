"""Feature-family migration tracker (Master → Transformation Pipeline).

Migration is family-based and evidence-based:

  Current Master Column → Pipeline generate → Parity test → Pass?
    No  → keep Master emitter
    Yes → mark Pipeline Owned → stop Master → remove Registry entry

See ``docs/master-dataset/FEATURE_OWNERSHIP.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ParityStatus = Literal["pending", "passed", "failed", "partial", "blocked"]

# ---------------------------------------------------------------------------
# Horizons (row-based; exact multiples of sample_interval, default 3s)
# ---------------------------------------------------------------------------

LTP_TO_SPOT_RATIO_LAG_HORIZONS: tuple[tuple[str, float], ...] = (
    ("30s", 30.0),
    ("1m", 60.0),
    ("3m", 180.0),
    ("5m", 300.0),
    ("15m", 900.0),
)

LTP_TO_SPOT_RATIO_CHANGE_HORIZONS: tuple[tuple[str, float], ...] = (
    ("30s", 30.0),
    ("1m", 60.0),
    ("3m", 180.0),
    ("5m", 300.0),
    ("15m", 900.0),
)

DGT_REIV_PRED_LAG_HORIZONS: tuple[tuple[str, float], ...] = (
    ("30s", 30.0),
    ("1m", 60.0),
)

DGT_REIV_PRED_CHANGE_HORIZONS: tuple[tuple[str, float], ...] = (
    ("30s", 30.0),
    ("1m", 60.0),
)

DGT_PREDICTION_ERROR_LAG_HORIZONS: tuple[tuple[str, float], ...] = (
    ("30s", 30.0),
)

# Features permanently removed (not Master, not Registry, not Pipeline).
# Non-divisible at default sample_interval_sec=3 (5∤3, 10∤3) + tick-OHLC candle family.
RETIRED_FEATURES: frozenset[str] = frozenset({
    "ltp_to_spot_ratio_lag_10s",
    "ltp_to_spot_ratio_change_10s",
    "dgt_reiv_pred_lag_10s",
    "dgt_reiv_pred_change_10s",
    "dgt_prediction_error_lag_10s",
    "dgt_prediction_error_change_10s",
    "ltp_return_5s",
    "spot_change_5s",
    "volume_change_5s",
    "opt_volume_flow_5s",  # 5∤3 at default sample_interval_sec=3
    # Phase 3 — 5s/10s tick-OHLC / candle family (∤3 at default interval)
    "opt_volume_acc_5s_1m",
    "spot_body_pct_10s",
    "opt_body_pct_10s",
    "spot_vol_ratio_10s_1m",
    "ltp_x_volume_change_pct_10s",
    "spot_body_pct_prev1",
    "spot_body_pct_prev2",
    "spot_body_pct_prev3",
    "spot_range_pct_prev1",
    "opt_body_pct_prev1",
    "opt_range_pct_prev1",
})


@dataclass
class MigrationFamily:
    """One removable historical feature family."""

    family_id: str
    base_feature: str
    transform: str
    features: tuple[str, ...]
    horizons: tuple[tuple[str, float], ...] = ()
    parity: ParityStatus = "pending"
    removed_from_master: bool = False
    removed_from_registry: bool = False
    pipeline_owned: bool = False
    notes: str = ""
    parity_detail: dict[str, Any] = field(default_factory=dict)
    generator: str = ""  # override for PIPELINE_OWNED_GENERATORS (default: first transform token)

    def master_column(self, suffix: str) -> str:
        return f"{self.base_feature}_lag_{suffix}"

    @property
    def generator_id(self) -> str:
        if self.generator:
            return self.generator
        return self.transform.split("/")[0]

    def pipeline_config(
        self,
        *,
        sample_interval_sec: float,
        partition_by: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build Lag pipeline config that emits Master-compatible column names."""
        parts = partition_by or ["trading_day", "token"]
        horizons = [
            {"seconds": float(sec), "suffix": str(suffix)}
            for suffix, sec in self.horizons
        ]
        return {
            "transformation_pipeline_version": 1,
            "transformations": [{
                "id": "lag",
                "enabled": True,
                "params": {
                    "features": [self.base_feature],
                    "horizons": horizons,
                    "partition_by": parts,
                    "sample_interval_sec": sample_interval_sec,
                },
            }],
        }


def _passed_family(
    *,
    family_id: str,
    base_feature: str,
    transform: str,
    features: tuple[str, ...],
    horizons: tuple[tuple[str, float], ...] = (),
    notes: str = "",
    generator: str = "",
) -> MigrationFamily:
    return MigrationFamily(
        family_id=family_id,
        base_feature=base_feature,
        transform=transform,
        features=features,
        horizons=horizons,
        parity="passed",
        removed_from_master=True,
        removed_from_registry=True,
        pipeline_owned=True,
        notes=notes,
        generator=generator,
    )


# Mutable status registry (updated by parity harness / migration steps).
_MIGRATION_FAMILIES: dict[str, MigrationFamily] = {
    "ltp_to_spot_ratio": _passed_family(
        family_id="ltp_to_spot_ratio",
        base_feature="ltp_to_spot_ratio",
        transform="lag",
        features=tuple(
            f"ltp_to_spot_ratio_lag_{suffix}"
            for suffix, _ in LTP_TO_SPOT_RATIO_LAG_HORIZONS
        ),
        horizons=LTP_TO_SPOT_RATIO_LAG_HORIZONS,
        notes=(
            "Pipeline Owned. *_lag_10s retired (10∤3). "
            "Lags via Lag transform only."
        ),
    ),
    "ltp_to_spot_ratio_change": _passed_family(
        family_id="ltp_to_spot_ratio_change",
        base_feature="ltp_to_spot_ratio",
        transform="difference",
        features=tuple(
            f"ltp_to_spot_ratio_change_{suffix}"
            for suffix, _ in LTP_TO_SPOT_RATIO_CHANGE_HORIZONS
        ),
        horizons=LTP_TO_SPOT_RATIO_CHANGE_HORIZONS,
        notes=(
            "Pipeline Owned abs difference. *_change_10s retired (10∤3). "
            "Via Difference transform with suffix/column."
        ),
        generator="difference",
    ),
    "dgt_reiv_pred_lag": _passed_family(
        family_id="dgt_reiv_pred_lag",
        base_feature="dgt_reiv_pred",
        transform="lag",
        features=tuple(
            f"dgt_reiv_pred_lag_{suffix}"
            for suffix, _ in DGT_REIV_PRED_LAG_HORIZONS
        ),
        horizons=DGT_REIV_PRED_LAG_HORIZONS,
        notes="Pipeline Owned. *_lag_10s retired. DGT controller no longer emits lags.",
    ),
    "dgt_reiv_pred_change": _passed_family(
        family_id="dgt_reiv_pred_change",
        base_feature="dgt_reiv_pred",
        transform="difference",
        features=tuple(
            f"dgt_reiv_pred_change_{suffix}"
            for suffix, _ in DGT_REIV_PRED_CHANGE_HORIZONS
        ),
        horizons=DGT_REIV_PRED_CHANGE_HORIZONS,
        notes="Pipeline Owned abs difference. *_change_10s retired.",
        generator="difference",
    ),
    "dgt_prediction_error_lag": _passed_family(
        family_id="dgt_prediction_error_lag",
        base_feature="dgt_prediction_error",
        transform="lag",
        features=tuple(
            f"dgt_prediction_error_lag_{suffix}"
            for suffix, _ in DGT_PREDICTION_ERROR_LAG_HORIZONS
        ),
        horizons=DGT_PREDICTION_ERROR_LAG_HORIZONS,
        notes="Pipeline Owned. *_lag_10s and *_change_10s retired.",
    ),
    "ltp_change": _passed_family(
        family_id="ltp_change",
        base_feature="ltp",
        transform="difference",
        features=("ltp_change_1m", "ltp_change_5m", "ltp_change_15m"),
        horizons=(("1m", 60.0), ("5m", 300.0), ("15m", 900.0)),
        notes="Pipeline Owned abs difference via Difference + column=.",
        generator="difference",
    ),
    "greeks_change": _passed_family(
        family_id="greeks_change",
        base_feature="delta",
        transform="difference",
        features=("delta_change_5m", "gamma_change_5m", "theta_change_5m"),
        horizons=(("5m", 300.0),),
        notes="Pipeline Owned abs greek differences (delta/gamma/theta @ 5m).",
        generator="difference",
    ),
    "oi": _passed_family(
        family_id="oi",
        base_feature="oi",
        transform="difference/return",
        features=(
            "oi_change_5m",
            "oi_change_15m",
            "oi_velocity_1m",
            "oi_change_1m",
            "oi_change_pct_1m",
            "oi_change_pct_5m",
            "oi_change_pct_15m",
        ),
        notes=(
            "Pipeline Owned. Abs: oi_change_{5m,15m}, oi_velocity_1m (Difference). "
            "Pct×100: oi_change_1m + oi_change_pct_* (Return scale=100)."
        ),
    ),
    "volume": _passed_family(
        family_id="volume",
        base_feature="volume",
        transform="difference/return",
        features=(
            "volume_change_5m",
            "volume_change_15m",
            "volume_change_15s",
            "volume_change_30s",
            "volume_change_1m",
        ),
        notes=(
            "Pipeline Owned. Abs: volume_change_{5m,15m}. "
            "Pct×100: volume_change_{15s,30s,1m}. *_change_5s retired. "
            "opt_volume_flow_* → difference_clip family; "
            "ltp_x_volume_change_pct_* → ltp_x_volume_change_pct family."
        ),
    ),
    "pcr": _passed_family(
        family_id="pcr",
        base_feature="atm_pcr",
        transform="difference",
        features=("chain_pcr_change_5m", "atm_pcr_change_5m"),
        horizons=(("5m", 300.0),),
        notes="Pipeline Owned abs PCR differences @ 5m.",
        generator="difference",
    ),
    "atm_straddle": _passed_family(
        family_id="atm_straddle",
        base_feature="atm_straddle",
        transform="difference/return",
        features=(
            "atm_straddle_change_1m",
            "atm_straddle_change_5m",
            "atm_straddle_change_pct_1m",
            "atm_straddle_change_pct_5m",
        ),
        notes=(
            "Pipeline Owned lookback change/pct. "
            "zscore / accel / slope / pct_change_from_open → Phase-2 families."
        ),
    ),
    "iv": _passed_family(
        family_id="iv",
        base_feature="current_iv",
        transform="difference/return",
        features=(
            "iv_change_1m",
            "iv_change_5m",
            "iv_change_15m",
            "iv_pct_change_1m",
        ),
        notes=(
            "Pipeline Owned. Abs: iv_change_*. Pct×100: iv_pct_change_1m. "
            "iv_zscore_* → rolling_zscore family."
        ),
    ),
    "derived": _passed_family(
        family_id="derived",
        base_feature="atm_straddle",
        transform="derived",
        features=(
            "atm_straddle_slope_5m",
            "atm_straddle_slope_15m",
            "atm_straddle_change_accel",
        ),
        notes=(
            "Pipeline Owned weighted-lag algebra via Derived transform "
            "(slope_5m/15m, change_accel)."
        ),
        generator="derived",
    ),
    "difference_clip": _passed_family(
        family_id="difference_clip",
        base_feature="volume",
        transform="difference_clip",
        features=(
            "opt_volume_flow_15s",
            "opt_volume_flow_30s",
            "opt_volume_flow_1m",
        ),
        horizons=(("15s", 15.0), ("30s", 30.0), ("1m", 60.0)),
        notes=(
            "Pipeline Owned clipped diffs (clip_min=0 → *_flow_*). "
            "opt_volume_flow_5s retired (5∤3)."
        ),
        generator="difference_clip",
    ),
    "anchor_return": _passed_family(
        family_id="anchor_return",
        base_feature="atm_straddle",
        transform="anchor_return",
        features=("atm_straddle_pct_change_from_open",),
        notes=(
            "Pipeline Owned session-open anchor return via Anchor Return "
            "(scale=100, partition_by trading_day[+token])."
        ),
        generator="return",
    ),
    "rolling_zscore": _passed_family(
        family_id="rolling_zscore",
        base_feature="current_iv",
        transform="rolling_statistics",
        features=(
            "iv_zscore_1m",
            "iv_zscore_5m",
            "iv_zscore_15m",
            "iv_zscore_30m",
            "atm_straddle_zscore_30m",
        ),
        horizons=(
            ("1m", 60.0),
            ("5m", 300.0),
            ("15m", 900.0),
            ("30m", 1800.0),
        ),
        notes=(
            "Pipeline Owned rolling z-score via Rolling Statistics (stat=zscore). "
            "atm_straddle_zscore_change_5m → atm_straddle_zscore_change family."
        ),
        generator="rolling_zscore",
    ),
    "ltp_return": _passed_family(
        family_id="ltp_return",
        base_feature="ltp",
        transform="return",
        features=("ltp_return_15s", "ltp_return_30s", "ltp_return_1m"),
        horizons=(("15s", 15.0), ("30s", 30.0), ("1m", 60.0)),
        notes="Pipeline Owned Return scale=100 (Master pct). *_return_5s retired.",
        generator="return",
    ),
    "spot_return": _passed_family(
        family_id="spot_return",
        base_feature="spot",
        transform="return",
        features=("spot_change_15s", "spot_change_30s", "spot_change_1m"),
        horizons=(("15s", 15.0), ("30s", 30.0), ("1m", 60.0)),
        notes=(
            "Pipeline Owned Return scale=100. Name is change but semantics are pct×100. "
            "*_change_5s retired."
        ),
        generator="return",
    ),
    "atm_straddle_zscore_change": _passed_family(
        family_id="atm_straddle_zscore_change",
        base_feature="atm_straddle_zscore_30m",
        transform="difference",
        features=("atm_straddle_zscore_change_5m",),
        horizons=(("5m", 300.0),),
        notes=(
            "Pipeline Owned Difference of Pipeline Owned base atm_straddle_zscore_30m @ 300s. "
            "Config: difference features=[atm_straddle_zscore_30m] "
            "horizons=[{seconds:300, column:atm_straddle_zscore_change_5m}]."
        ),
        generator="difference",
    ),
    "spot_rolling_ohlc_5m": _passed_family(
        family_id="spot_rolling_ohlc_5m",
        base_feature="spot",
        transform="rolling_ohlc",
        features=(
            "spot_dist_high_5m_pct",
            "spot_dist_low_5m_pct",
            "spot_range_pos_5m",
        ),
        horizons=(("5m", 300.0),),
        notes=(
            "Pipeline Owned rolling_ohlc on spot @ 300s. "
            "outputs=[dist_high_pct, dist_low_pct, range_pos], range_eps=1e-9, "
            "column map → Master names. Tick-OHLC 10s candle family retired (not remapped)."
        ),
        generator="rolling_ohlc",
    ),
    "ltp_x_volume_change_pct": _passed_family(
        family_id="ltp_x_volume_change_pct",
        base_feature="volume",
        transform="return/interaction",
        features=(
            "ltp_x_volume_change_pct_30s",
            "ltp_x_volume_change_pct_1m",
            "ltp_x_volume_change_pct_3m",
            "ltp_x_volume_change_pct_5m",
            "ltp_x_volume_change_pct_15m",
        ),
        horizons=(
            ("30s", 30.0),
            ("1m", 60.0),
            ("3m", 180.0),
            ("5m", 300.0),
            ("15m", 900.0),
        ),
        notes=(
            "Pipeline Owned multi-step: Return(volume, scale=1, denom_eps=1e-9) then "
            "Interaction mul ltp × volume_return_* with Master column overrides. "
            "Master formula: ltp * (vol-vol_lag)/(vol_lag+ε), NOT ×100. "
            "*_10s retired (10∤3)."
        ),
        generator="interaction",
    ),
    # Former Registry interaction products — never re-admit (Interaction pipeline only).
    "registry_interaction_products": _passed_family(
        family_id="registry_interaction_products",
        base_feature="",
        transform="interaction",
        features=(
            "delta_x_spot",
            "gamma_x_spot",
            "weighted_iv_zscore_x_weighted_spot_ema_to_ltp_ratio",
            "weighted_spot_ema_to_ltp_ratio_x_delta",
            "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_1m",
            "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_5m",
            "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m",
            "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_1m_x_delta",
            "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_5m_x_delta",
            "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m_x_delta",
            "ltp_ema9_to_spot_ratio_x_iv_ema9",
            "ltp_ema20_to_spot_ratio_x_iv_ema20",
            "ltp_ema50_to_spot_ratio_x_iv_ema50",
            "ltp_ema100_to_spot_ratio_x_iv_ema100",
            "ltp_ema200_to_spot_ratio_x_iv_ema200",
            "ltp_ema300_to_spot_ratio_x_iv_ema300",
            "spot_to_ltp_ratio_x_iv_ema9",
            "spot_to_ltp_ratio_x_iv_ema20",
            "spot_to_ltp_ratio_x_iv_ema50",
            "spot_to_ltp_ratio_x_iv_ema100",
            "spot_to_ltp_ratio_x_iv_ema200",
            "spot_to_ltp_ratio_x_iv_ema300",
            "spot_to_ltp_ratio_x_iv_ema9_x_moneyness",
            "spot_to_ltp_ratio_x_iv_ema20_x_moneyness",
            "spot_to_ltp_ratio_x_iv_ema50_x_moneyness",
            "spot_to_ltp_ratio_x_iv_ema100_x_moneyness",
            "spot_to_ltp_ratio_x_iv_ema200_x_moneyness",
            "spot_to_ltp_ratio_x_iv_ema300_x_moneyness",
            "spot_ema9_to_ltp_ratio_x_moneyness",
            "spot_ema20_to_ltp_ratio_x_moneyness",
            "spot_ema50_to_ltp_ratio_x_moneyness",
            "spot_ema100_to_ltp_ratio_x_moneyness",
            "spot_ema200_to_ltp_ratio_x_moneyness",
            "spot_ema300_to_ltp_ratio_x_moneyness",
            "ltp_ema9_to_spot_ratio_x_moneyness",
            "ltp_ema20_to_spot_ratio_x_moneyness",
            "ltp_ema50_to_spot_ratio_x_moneyness",
            "ltp_ema100_to_spot_ratio_x_moneyness",
            "ltp_ema200_to_spot_ratio_x_moneyness",
            "ltp_ema300_to_spot_ratio_x_moneyness",
            "weighted_spot_ema_to_ltp_ratio_x_moneyness",
            "weighted_spot_ema_to_ltp_ratio_x_moneyness_x_delta",
        ),
        notes=(
            "Removed from Feature Registry / Master. Rebuild via InteractionTransformation "
            "pairs only. Never re-admit (admission rejects *_x_* / Interaction ops)."
        ),
        generator="interaction",
    ),
    # Architectural review: generic registry-math composites (not name-based).
    "greek_ltp_spot_registry_math": _passed_family(
        family_id="greek_ltp_spot_registry_math",
        base_feature="ltp_to_spot_ratio",
        transform="interaction",
        features=(
            "delta_ltp_to_spot_ratio",
            "gamma_ltp_to_spot_ratio",
            "moneyness_delta_ltp_to_spot_ratio",
        ),
        notes=(
            "Moved: no controller/state machine; pure row-wise products of registry columns "
            "(delta|gamma|moneyness×abs_delta) × ltp / spot ≡ × ltp_to_spot_ratio. "
            "Recreate via InteractionTransformation multiply. Not canonical market state."
        ),
        generator="interaction",
    ),
    "current_to_atm6_flow_composite": _passed_family(
        family_id="current_to_atm6_flow_composite",
        base_feature="",
        transform="interaction",
        features=("current_to_atm6_flow_delta_ltp_to_spot_ratio",),
        notes=(
            "Moved: no dedicated controller; experiment OTM-flow heuristic scaled by "
            "|delta|×ltp/spot. Not every-Master market state. Outer scaling is Interaction; "
            "7-strike 1m vol/OI flow core needs Dataset Builder chain-flow step "
            "(current_to_atm6_flow.py) before Interaction, or a future ChainController "
            "Computed Base for flow strength alone."
        ),
        generator="interaction",
    ),
    # Both operands already in Registry — pure A÷B (or add then ÷) normalization.
    "registry_normalization_ratios": _passed_family(
        family_id="registry_normalization_ratios",
        base_feature="",
        transform="interaction",
        features=(
            "ltp_to_dgt_reiv_ratio",
            "ltp_to_bs_reiv_ratio",
            "dgt_reiv_to_ltp_ratio",
            "bs_reiv_to_ltp_ratio",
            "dgt_to_spot_ratio",
            "bs_to_spot_ratio",
            "spot_rv_ratio",
            "opt_rv_ratio",
            "ce_atm6_ltp_to_spot_ratio",
            "pe_atm6_ltp_to_spot_ratio",
            "ce_pe_atm6_ltp_ratio",
            "atm6_total_to_ltp_ratio",
            "atm6_total_to_spot_ratio",
        ),
        notes=(
            "Migrated: both numerator and denominator already Registry columns. "
            "Recreate via Interaction divide (ATM6 totals: add CE+PE sums then divide). "
            "See docs/master-dataset/MIGRATION_REGISTRY_NORMALIZATION_RATIOS.md. "
            "Foundational moneyness / ltp_to_spot_ratio / strike_to_spot_ratio kept as Base."
        ),
        generator="interaction",
    ),
    # Wave 2: EMA/std packaging — controllers now emit levels; ratios via Interaction.
    "wave2_controller_ema_packaging": _passed_family(
        family_id="wave2_controller_ema_packaging",
        base_feature="",
        transform="interaction",
        features=(
            "ltp_ema9_to_ltp_ratio",
            "ltp_ema20_to_ltp_ratio",
            "ltp_ema50_to_ltp_ratio",
            "ltp_ema100_to_ltp_ratio",
            "ltp_ema200_to_ltp_ratio",
            "ltp_ema300_to_ltp_ratio",
            "ltp_ema9_to_spot_ratio",
            "ltp_ema20_to_spot_ratio",
            "ltp_ema50_to_spot_ratio",
            "ltp_ema100_to_spot_ratio",
            "ltp_ema200_to_spot_ratio",
            "ltp_ema300_to_spot_ratio",
            "ltp_std20_to_ltp_ratio",
            "ltp_std20_to_spot_ratio",
            "iv_ema9_to_ltp_ratio",
            "iv_ema20_to_ltp_ratio",
            "iv_ema50_to_ltp_ratio",
            "iv_ema100_to_ltp_ratio",
            "iv_ema200_to_ltp_ratio",
            "iv_ema300_to_ltp_ratio",
            "iv_ema9_to_spot_ratio",
            "iv_ema20_to_spot_ratio",
            "iv_ema50_to_spot_ratio",
            "iv_ema100_to_spot_ratio",
            "iv_ema200_to_spot_ratio",
            "iv_ema300_to_spot_ratio",
            "spot_ema9_to_ltp_ratio",
            "spot_ema20_to_ltp_ratio",
            "spot_ema50_to_ltp_ratio",
            "spot_ema100_to_ltp_ratio",
            "spot_ema200_to_ltp_ratio",
            "spot_ema300_to_ltp_ratio",
            "spot_high_ema20_to_ltp_ratio",
            "spot_high_ema50_to_ltp_ratio",
            "spot_high_ema100_to_ltp_ratio",
            "spot_high_ema200_to_ltp_ratio",
            "spot_high_ema300_to_ltp_ratio",
            "spot_low_ema20_to_ltp_ratio",
            "spot_low_ema50_to_ltp_ratio",
            "spot_low_ema100_to_ltp_ratio",
            "spot_low_ema200_to_ltp_ratio",
            "spot_low_ema300_to_ltp_ratio",
        ),
        notes=(
            "Wave 2: controllers emit canonical levels (ltp_emaN, iv_emaN, spot_emaN, "
            "spot_high/low_emaN, ltp_std20). Recreate packaging via Interaction divide "
            "(numerator=level, denominator=ltp|spot). See "
            "docs/master-dataset/MIGRATION_WAVE2_CONTROLLER_LEVELS.md."
        ),
        generator="interaction",
    ),
    # Wave 3: weighted blend levels promoted; packaging → Interaction.
    "wave3_weighted_blend_packaging": _passed_family(
        family_id="wave3_weighted_blend_packaging",
        base_feature="",
        transform="interaction",
        features=(
            "weighted_ltp_ema_to_ltp_ratio",
            "weighted_spot_ema_to_ltp_ratio",
            "weighted_spot_high_ema_to_ltp_ratio",
            "weighted_spot_low_ema_to_ltp_ratio",
            "weighted_spot_high_ema_to_weighted_spot_low_ema",
            "weighted_spot_ema_to_weighted_spot_low_ema",
            "weighted_spot_ema_to_weighted_spot_high_ema",
        ),
        notes=(
            "Wave 3: promote weighted_*_ema levels (incl. weighted_spot_close_ema for "
            "HL-close blend used by historical cross-ratios). Recreate: "
            "weighted_ltp_ema/ltp, weighted_spot_ema/ltp, weighted_spot_high_ema/ltp, "
            "weighted_spot_low_ema/ltp, weighted_spot_high_ema/weighted_spot_low_ema, "
            "weighted_spot_close_ema/weighted_spot_low_ema, "
            "weighted_spot_close_ema/weighted_spot_high_ema. "
            "See docs/master-dataset/MIGRATION_WAVE3_WEIGHTED_BLENDS.md."
        ),
        generator="interaction",
    ),
    # Wave 4: sharp momentum score/count levels; packaging → Interaction.
    "wave4_sharp_momentum_packaging": _passed_family(
        family_id="wave4_sharp_momentum_packaging",
        base_feature="",
        transform="interaction",
        features=(
            "spot_up_score_1m_to_ltp_ratio",
            "spot_up_score_3m_to_ltp_ratio",
            "spot_up_score_5m_to_ltp_ratio",
            "spot_up_score_10m_to_ltp_ratio",
            "spot_down_score_1m_to_ltp_ratio",
            "spot_down_score_3m_to_ltp_ratio",
            "spot_down_score_5m_to_ltp_ratio",
            "spot_down_score_10m_to_ltp_ratio",
            "ltp_to_1m_spot_up_sample_count_ratio",
            "ltp_to_3m_spot_up_sample_count_ratio",
            "ltp_to_5m_spot_up_sample_count_ratio",
            "ltp_to_10m_spot_up_sample_count_ratio",
            "ltp_to_1m_spot_down_sample_count_ratio",
            "ltp_to_3m_spot_down_sample_count_ratio",
            "ltp_to_5m_spot_down_sample_count_ratio",
            "ltp_to_10m_spot_down_sample_count_ratio",
        ),
        notes=(
            "Wave 4: SpotMomentumSnapshot emits score/count levels. "
            "Recreate score packaging: spot_{up|down}_score_{h} / ltp. "
            "Recreate count packaging: ltp / (spot_{up|down}_sample_count_{h} + 1e-6). "
            "See docs/master-dataset/MIGRATION_WAVE4_SHARP_MOMENTUM.md."
        ),
        generator="interaction",
    ),
    # Wave 5: channel width levels; packaging → Interaction.
    "wave5_channel_width_packaging": _passed_family(
        family_id="wave5_channel_width_packaging",
        base_feature="",
        transform="interaction",
        features=(
            "ltp_to_spot_ema20_channel_width_ratio",
            "ltp_to_spot_ema50_channel_width_ratio",
            "ltp_to_spot_ema100_channel_width_ratio",
            "ltp_to_spot_ema200_channel_width_ratio",
            "ltp_to_spot_ema300_channel_width_ratio",
        ),
        notes=(
            "Wave 5: spot.hl emits spot_ema{N}_channel_width = high−low. "
            "Recreate packaging: ltp / spot_ema{N}_channel_width "
            "(Master used abs(width)+1e-6; Interaction divide is soft-parity). "
            "See docs/master-dataset/MIGRATION_WAVE5_CHANNEL_WIDTH.md."
        ),
        generator="interaction",
    ),
    # Wave 6: registry % / normalized-imbalance packaging → Interaction.
    "wave6_registry_pct_packaging": _passed_family(
        family_id="wave6_registry_pct_packaging",
        base_feature="",
        transform="interaction",
        features=(
            "spot_vs_ema20_pct",
            "ema_spread_pct",
            "ema_spread_vs_spot_pct",
            "ce_pe_atm6_ltp_diff_pct",
        ),
        notes=(
            "Wave 6: reconstructible from canonical levels already in Registry. "
            "spot/EMA % via subtract→divide scale=100; "
            "ce_pe_atm6_ltp_diff_pct via subtract→add→divide (soft-parity eps). "
            "Keep spot, spot_ema9, spot_ema20, ce/pe_atm6_ltp_sum as Base/Computed Base. "
            "See docs/master-dataset/MIGRATION_WAVE6_REGISTRY_PCT_PACKAGING.md."
        ),
        generator="interaction",
    ),
    # Wave A: spread-normalized LTP step — packaging only (Difference ÷ spread).
    "wave_a_spread_normalized_ltp_step": _passed_family(
        family_id="wave_a_spread_normalized_ltp_step",
        base_feature="ltp",
        transform="difference/interaction",
        features=("ltp_step_div_bid_ask_spread",),
        notes=(
            "Wave A: Δltp / bid_ask_spread. Build via Difference on ltp then "
            "Interaction divide by bid_ask_spread (eps). Not a Registry feature; "
            "keep mid_price / microprice / OBI / depths as Computed Base levels."
        ),
        generator="interaction",
    ),
    # Option VWAP distances — packaging only (ltp vs option_vwap).
    "option_vwap_distance": _passed_family(
        family_id="option_vwap_distance",
        base_feature="option_vwap",
        transform="interaction",
        features=(
            "ltp_minus_option_vwap",
            "ltp_minus_option_vwap_div_option_vwap",
        ),
        notes=(
            "Distance points = Interaction subtract(ltp, option_vwap). "
            "Distance %% = that result ÷ option_vwap (second Interaction). "
            "Keep option_vwap as Registry Base (exchange ATP); do not admit "
            "distance columns as canonical Registry features."
        ),
        generator="interaction",
    ),
    # Futures timeline relationships — packaging only (spot / futures_ltp / futures_vwap).
    "futures_timeline_packaging": _passed_family(
        family_id="futures_timeline_packaging",
        base_feature="futures_ltp",
        transform="interaction",
        features=(
            "futures_ltp_minus_futures_vwap",
            "futures_ltp_minus_futures_vwap_div_futures_vwap",
            "spot_minus_futures_ltp",
            "spot_minus_futures_vwap",
            "spot_div_futures_ltp",
            "futures_ltp_div_spot",
        ),
        notes=(
            "Phase 1 Futures Timeline: keep futures_ltp / futures_vwap as Registry "
            "Base. All basis / premium-to-VWAP / ratio signals are Interaction only."
        ),
        generator="interaction",
    ),
}

# Per-feature generator overrides when a family mixes difference + return.
_FEATURE_GENERATOR_OVERRIDES: dict[str, str] = {
    # oi
    "oi_change_5m": "difference",
    "oi_change_15m": "difference",
    "oi_velocity_1m": "difference",
    "oi_change_1m": "return",
    "oi_change_pct_1m": "return",
    "oi_change_pct_5m": "return",
    "oi_change_pct_15m": "return",
    # volume
    "volume_change_5m": "difference",
    "volume_change_15m": "difference",
    "volume_change_15s": "return",
    "volume_change_30s": "return",
    "volume_change_1m": "return",
    # atm_straddle
    "atm_straddle_change_1m": "difference",
    "atm_straddle_change_5m": "difference",
    "atm_straddle_change_pct_1m": "return",
    "atm_straddle_change_pct_5m": "return",
    # iv
    "iv_change_1m": "difference",
    "iv_change_5m": "difference",
    "iv_change_15m": "difference",
    "iv_pct_change_1m": "return",
    # Phase-2 rolling zscore (mixed bases)
    "iv_zscore_1m": "rolling_zscore",
    "iv_zscore_5m": "rolling_zscore",
    "iv_zscore_15m": "rolling_zscore",
    "iv_zscore_30m": "rolling_zscore",
    "atm_straddle_zscore_30m": "rolling_zscore",
    # Phase-2 difference_clip
    "opt_volume_flow_15s": "difference_clip",
    "opt_volume_flow_30s": "difference_clip",
    "opt_volume_flow_1m": "difference_clip",
    # Phase-2 derived / anchor
    "atm_straddle_slope_5m": "derived",
    "atm_straddle_slope_15m": "derived",
    "atm_straddle_change_accel": "derived",
    "atm_straddle_pct_change_from_open": "return",
    # Phase-3 chained / OHLC / interaction
    "atm_straddle_zscore_change_5m": "difference",
    "spot_dist_high_5m_pct": "rolling_ohlc",
    "spot_dist_low_5m_pct": "rolling_ohlc",
    "spot_range_pos_5m": "rolling_ohlc",
    "ltp_x_volume_change_pct_30s": "interaction",
    "ltp_x_volume_change_pct_1m": "interaction",
    "ltp_x_volume_change_pct_3m": "interaction",
    "ltp_x_volume_change_pct_5m": "interaction",
    "ltp_x_volume_change_pct_15m": "interaction",
    # greeks (base differs per column)
    "delta_change_5m": "difference",
    "gamma_change_5m": "difference",
    "theta_change_5m": "difference",
    # pcr (base differs)
    "chain_pcr_change_5m": "difference",
    "atm_pcr_change_5m": "difference",
    # spot_return Master names
    "spot_change_15s": "return",
    "spot_change_30s": "return",
    "spot_change_1m": "return",
}


def _populate_pipeline_owned() -> tuple[set[str], dict[str, str]]:
    owned: set[str] = set()
    generators: dict[str, str] = {}
    for fam in _MIGRATION_FAMILIES.values():
        if not fam.pipeline_owned:
            continue
        owned.update(fam.features)
        for name in fam.features:
            generators[name] = _FEATURE_GENERATOR_OVERRIDES.get(name, fam.generator_id)
    return owned, generators


PIPELINE_OWNED_FEATURES: set[str]
PIPELINE_OWNED_GENERATORS: dict[str, str]
PIPELINE_OWNED_FEATURES, PIPELINE_OWNED_GENERATORS = _populate_pipeline_owned()


def get_migration_family(family_id: str) -> MigrationFamily:
    fam = _MIGRATION_FAMILIES.get(str(family_id or "").strip())
    if fam is None:
        raise KeyError(f"Unknown migration family: {family_id!r}")
    return fam


def list_migration_families() -> list[MigrationFamily]:
    return list(_MIGRATION_FAMILIES.values())


def migration_status_table() -> list[dict[str, Any]]:
    """Rows for the migration dashboard."""
    rows: list[dict[str, Any]] = []
    for fam in list_migration_families():
        rows.append({
            "feature_family": fam.family_id,
            "transform": fam.transform,
            "parity": fam.parity,
            "removed_from_master": fam.removed_from_master,
            "removed_from_registry": fam.removed_from_registry,
            "pipeline_owned": fam.pipeline_owned,
            "notes": fam.notes,
        })
    return rows


def mark_family_parity(
    family_id: str,
    *,
    status: ParityStatus,
    detail: dict[str, Any] | None = None,
    notes: str | None = None,
) -> MigrationFamily:
    fam = get_migration_family(family_id)
    fam.parity = status
    if detail is not None:
        fam.parity_detail = dict(detail)
    if notes is not None:
        fam.notes = notes
    if status == "passed":
        fam.pipeline_owned = True
        PIPELINE_OWNED_FEATURES.update(fam.features)
        for name in fam.features:
            PIPELINE_OWNED_GENERATORS.setdefault(
                name,
                _FEATURE_GENERATOR_OVERRIDES.get(name, fam.generator_id),
            )
    return fam


def mark_family_removed(
    family_id: str,
    *,
    from_master: bool = False,
    from_registry: bool = False,
) -> MigrationFamily:
    fam = get_migration_family(family_id)
    if from_master:
        fam.removed_from_master = True
    if from_registry:
        fam.removed_from_registry = True
    if fam.removed_from_master or fam.removed_from_registry:
        fam.pipeline_owned = True
        PIPELINE_OWNED_FEATURES.update(fam.features)
        for name in fam.features:
            PIPELINE_OWNED_GENERATORS.setdefault(
                name,
                _FEATURE_GENERATOR_OVERRIDES.get(name, fam.generator_id),
            )
    return fam


def is_pipeline_owned(feature: str) -> bool:
    return str(feature or "").strip() in PIPELINE_OWNED_FEATURES


def is_retired(feature: str) -> bool:
    return str(feature or "").strip() in RETIRED_FEATURES


def horizons_compatible_with_interval(
    horizons: tuple[tuple[str, float], ...] | list[tuple[str, float]],
    sample_interval_sec: float,
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Split horizons into (compatible, incompatible) for row-shift transforms.

    Calendar-time lookbacks are not supported; incompatible horizons must be
    retired or remapped to exact multiples of ``sample_interval_sec``.
    """
    try:
        interval = float(sample_interval_sec)
    except (TypeError, ValueError):
        interval = 0.0
    ok: list[tuple[str, float]] = []
    bad: list[tuple[str, float]] = []
    for suffix, sec in horizons:
        if interval <= 0:
            bad.append((suffix, float(sec)))
            continue
        rows = float(sec) / interval
        rows_i = int(round(rows))
        if abs(rows - rows_i) > 1e-9 or rows_i < 1:
            bad.append((suffix, float(sec)))
        else:
            ok.append((suffix, float(sec)))
    return ok, bad


__all__ = [
    "ParityStatus",
    "MigrationFamily",
    "LTP_TO_SPOT_RATIO_LAG_HORIZONS",
    "LTP_TO_SPOT_RATIO_CHANGE_HORIZONS",
    "DGT_REIV_PRED_LAG_HORIZONS",
    "DGT_REIV_PRED_CHANGE_HORIZONS",
    "DGT_PREDICTION_ERROR_LAG_HORIZONS",
    "RETIRED_FEATURES",
    "PIPELINE_OWNED_FEATURES",
    "PIPELINE_OWNED_GENERATORS",
    "get_migration_family",
    "list_migration_families",
    "migration_status_table",
    "mark_family_parity",
    "mark_family_removed",
    "is_pipeline_owned",
    "is_retired",
    "horizons_compatible_with_interval",
]
