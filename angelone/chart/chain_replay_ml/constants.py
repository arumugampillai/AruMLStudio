"""Phase 1 ML feature export constants (aligned with chain_replay_player.html)."""

from __future__ import annotations

RISK_FREE_RATE = 0.07
SECONDS_IN_YEAR = 365 * 24 * 3600
EPS_T = 1e-8

# Hybrid re-anchor defaults (predictor panel)
DEFAULT_IV_THRESHOLD_PCT = 2.0
DEFAULT_SPOT_THRESHOLD_PCT = 0.3
DEFAULT_MAX_ROLL_AGE_MIN = 15

WARMUP_MINUTES = 15
LABEL_FORWARD_MIN = 5
LOOKBACK_MINUTES = (1, 5, 15)

STRIKE_STEP_BY_INDEX = {
    "NIFTY": 50,
    "SENSEX": 100,
}

MIN_LTP_RUPEES = 1.0
DEFAULT_DELTA_PROFILE_TARGET = 0.15

PHASE1_COLUMNS = [
    # meta (training splits)
    "date",
    "underlying",
    "expiry",
    "token",
    "symbol",
    "warmup_row",
    # time
    "time",
    "minute_of_day",
    "minutes_to_close",
    "days_to_expiry",
    "is_expiry_day",
    # market
    "spot",
    "ltp",
    "current_iv",
    "roll_iv",
    "roll_age_min",
    "rows_since_roll",
    "roll_reason",
    "roll_reason_code",
    "roll_count",
    # structure
    "strike",
    "distance_from_atm_points",
    "distance_from_atm_pct",
    "strike_distance_from_atm",
    "is_call",
    "moneyness",
    # greeks (rolled)
    "delta",
    "gamma",
    "theta",
    "vega",
    # momentum
    "spot_change_1m",
    "spot_change_5m",
    "spot_change_15m",
    "iv_change_1m",
    "iv_change_5m",
    "iv_change_15m",
    "ltp_change_1m",
    "ltp_change_5m",
    "ltp_change_15m",
    "volume_change_1m",
    "volume_change_5m",
    "volume_change_15m",
    "oi_change_1m",
    "oi_change_5m",
    "oi_change_15m",
    "oi_change_pct_1m",
    "oi_change_pct_5m",
    "oi_change_pct_15m",
    # re-anchor state
    "iv_drift_from_roll",
    "spot_drift_from_roll",
    "is_reanchor_row",
    "bid_ask_spread",
    # chain-level straddle, PCR, and OI wall features
    "atm_straddle",
    "atm_straddle_change_1m",
    "atm_straddle_change_5m",
    "atm_straddle_change_15m",
    "atm_straddle_change_pct_1m",
    "atm_straddle_change_pct_5m",
    "atm_straddle_change_pct_15m",
    "atm_straddle_zscore_30m",
    "atm_straddle_zscore_change_5m",
    "atm_straddle_change_accel",
    "atm_straddle_slope_5m",
    "atm_straddle_slope_15m",
    "atm_straddle_pct_change_from_open",
    "distance_to_max_call_oi_strikes",
    "distance_to_max_put_oi_strikes",
    "max_call_oi_pct",
    "max_put_oi_pct",
    "chain_pcr",
    "atm_pcr",
    "oi_wall_bias",
    "distance_to_call_build_wall",
    "distance_to_put_build_wall",
    "chain_pcr_change_5m",
    "atm_pcr_change_5m",
    "pinning_pressure",
    # baselines at t
    "bs_reiv_pred",
    "dgt_reiv_pred",
    # label
    "actual_ltp_t",
    "actual_ltp_t_plus_5m",
    "residual_5m",
    "residual_pct_5m",
    "mfe_pct_10m",
    "mae_pct_10m",
    "future_high_first_10m",
    "future_low_first_10m",
    "time_to_high_sec_10m",
    "time_to_low_sec_10m",
    "entry_quality_score",
    "scalp_expectancy_score",
    "scalp_score",
    "scalper_score",
    "hit_10pct_before_5pct_down_60s",
    "hit_15pct_before_7pct_down_60s",
    "hit_20pct_before_10pct_down_60s",
    "hit_10pct_before_5pct_down_120s",
    "hit_15pct_before_7pct_down_120s",
    "hit_20pct_before_10pct_down_120s",
    "hit_10pct_before_5pct_down_300s",
    "hit_15pct_before_7pct_down_300s",
    "hit_20pct_before_10pct_down_300s",
    "hit_7pct_before_3pct_down_60s",
    "hit_5pct_before_2pct_down_30s",
]

LABEL_COLUMNS = [
    "actual_ltp_t",
    "actual_ltp_t_plus_5m",
    "residual_5m",
    "residual_pct_5m",
    "mfe_pct_10m",
    "mae_pct_10m",
    "future_high_first_10m",
    "future_low_first_10m",
    "time_to_high_sec_10m",
    "time_to_low_sec_10m",
    "entry_quality_score",
    "scalp_expectancy_score",
    "scalp_score",
    "scalper_score",
    "hit_10pct_before_5pct_down_60s",
    "hit_15pct_before_7pct_down_60s",
    "hit_20pct_before_10pct_down_60s",
    "hit_10pct_before_5pct_down_120s",
    "hit_15pct_before_7pct_down_120s",
    "hit_20pct_before_10pct_down_120s",
    "hit_10pct_before_5pct_down_300s",
    "hit_15pct_before_7pct_down_300s",
    "hit_20pct_before_10pct_down_300s",
    "hit_7pct_before_3pct_down_60s",
    "hit_5pct_before_2pct_down_30s",
]

META_COLUMNS = [
    "date",
    "underlying",
    "expiry",
    "token",
    "symbol",
    "time",
]

# Numeric inputs for XGBoost (excludes meta, labels, warmup flag, string roll_reason).
FEATURE_COLUMNS = [
    c
    for c in PHASE1_COLUMNS
    if c not in LABEL_COLUMNS
    and c not in META_COLUMNS
    and c not in ("warmup_row", "roll_reason")
]

# Side-specific models: same inputs minus is_call (side is implicit).
FEATURE_COLUMNS_SIDE = [c for c in FEATURE_COLUMNS if c != "is_call"]

DEFAULT_TARGET = "hit_7pct_before_3pct_down_60s"
SUPPORTED_TARGETS = (
    "hit_5pct_before_2pct_down_30s",
    "hit_7pct_before_3pct_down_60s",
    "hit_10pct_before_5pct_down_120s",
    "scalp_expectancy_score",
)
OPTION_SIDES = ("CE", "PE")

