"""
Live feature rows matching backtest ``extract_timeline_features`` / ``FEATURE_COLUMNS``.

Builds one strike's feature dict at a 10s grid timestamp from replay ``TickTimeline``
instances (typically adapted from ``TickRingStore`` via ``tick_timeline``).

No NeoApp wiring — call from ML engine / debug tools only.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from api.tick_ring import TickRingStore
from research.atm_band_ml.tick_timeline import ring_to_tick_timeline

_CHART_DIR = Path(__file__).resolve().parents[2] / "apps"

# Match chain_replay_ml.recompute_2_1_ratio data-quality gates.
FEATURE_BAR_SEC = 5
FEATURE_LOOKBACKS_SEC = (0, 5, 15, 30, 60)
DEFAULT_STRIKE_STEP = 50


def _ensure_chart_on_path() -> None:
    chart_dir = str(_CHART_DIR)
    if chart_dir not in sys.path:
        sys.path.insert(0, chart_dir)


def replay_feature_columns() -> list[str]:
    _ensure_chart_on_path()
    from chain_replay_ml.train_atm_model import FEATURE_COLUMNS

    return list(FEATURE_COLUMNS)


def _import_extract_timeline_features():
    _ensure_chart_on_path()
    from chain_replay_ml.features_atm_band import (
        extract_timeline_features,
        find_atm_strike,
    )

    return extract_timeline_features, find_atm_strike


@dataclass(frozen=True, slots=True)
class FeatureBuildResult:
    features: dict[str, Any]
    window_complete: bool
    model_complete: bool
    missing_model_columns: tuple[str, ...] = ()
    reason: str = ""

    @property
    def complete(self) -> bool:
        return self.window_complete and self.model_complete


def index_ltp_rupees_at(index_timeline, target_ts: float) -> float | None:
    if index_timeline is None:
        return None
    paise = index_timeline.ltp_paise_at(float(target_ts))
    if paise is None or paise <= 0:
        return None
    return paise / 100.0


def _bucket_has_tick(index_timeline, bucket: float) -> bool:
    """True when the index timeline has a valid tick in ``[bucket, bucket+5s)``."""
    if index_timeline is None or not index_timeline.timestamps:
        return False
    bar = float(FEATURE_BAR_SEC)
    b = float(bucket)
    for i, ts in enumerate(index_timeline.timestamps):
        if ts < b or ts >= b + bar:
            continue
        if index_timeline.ltps_paise[i] > 0:
            return True
    return False


def missing_5s_buckets(
    open_ts: float,
    close_ts: float,
    index_timeline,
    *,
    as_of_ts: float | None = None,
) -> set[float]:
    """5s buckets in [open, close] with no valid index tick (replay DQ helper).

    When ``as_of_ts`` is set (live), only buckets up to that time are considered so
    future session buckets are not marked missing.
    """
    grid_start = math.floor(float(open_ts) / FEATURE_BAR_SEC) * FEATURE_BAR_SEC
    grid_end = math.floor(float(close_ts) / FEATURE_BAR_SEC) * FEATURE_BAR_SEC
    if as_of_ts is not None:
        grid_end = min(
            grid_end,
            math.floor(float(as_of_ts) / FEATURE_BAR_SEC) * FEATURE_BAR_SEC,
        )
    if grid_end < grid_start:
        return set()
    buckets_with_tick: set[float] = set()
    if index_timeline is not None and index_timeline.timestamps:
        for i, ts in enumerate(index_timeline.timestamps):
            if ts < grid_start or ts >= grid_end + FEATURE_BAR_SEC:
                continue
            if index_timeline.ltps_paise[i] <= 0:
                continue
            bucket = math.floor(ts / FEATURE_BAR_SEC) * FEATURE_BAR_SEC
            if grid_start <= bucket <= grid_end:
                buckets_with_tick.add(bucket)
    missing: set[float] = set()
    t = grid_start
    while t <= grid_end:
        if t not in buckets_with_tick:
            missing.add(t)
        t += FEATURE_BAR_SEC
    return missing


def is_feature_window_complete(
    ts: float,
    index_timeline,
    *,
    missing_buckets: set[float] | None = None,
    open_ts: float | None = None,
    close_ts: float | None = None,
    as_of_ts: float | None = None,
    live: bool = False,
) -> bool:
    """
    Index short-horizon lookback gate (same rule as backtest ``_is_feature_window_complete``).

    When ``live`` is True, only require point-in-time LTP at each lookback (ring forward-fill).
    Replay/backtest keeps the dense 5s bucket DQ gate via ``missing_buckets``.
    """
    if index_timeline is None or not index_timeline.timestamps:
        return False
    if not live:
        if missing_buckets is None:
            if open_ts is None or close_ts is None:
                open_ts = float(index_timeline.timestamps[0])
                close_ts = float(index_timeline.timestamps[-1])
            missing_buckets = missing_5s_buckets(
                open_ts,
                close_ts,
                index_timeline,
                as_of_ts=as_of_ts if as_of_ts is not None else ts,
            )
    target = float(ts)
    for lb in FEATURE_LOOKBACKS_SEC:
        if not live:
            bucket = math.floor((target - lb) / FEATURE_BAR_SEC) * FEATURE_BAR_SEC
            if bucket in missing_buckets:
                return False
        if index_ltp_rupees_at(index_timeline, target - lb) is None:
            return False
    return True


def precompute_ema_context(index_timeline, open_ts: float, close_ts: float) -> dict[str, Any]:
    """1-minute EMA9/EMA20 + crossover history (matches export / recompute pipeline)."""
    minutes = np.arange(float(open_ts), float(close_ts) + 1.0, 60.0)
    prices_1m: list[float] = []
    last_p = index_timeline.ltps_paise[0] / 100.0 if index_timeline.ltps_paise else 0.0
    for m in minutes:
        p = index_timeline.ltp_rupees_at(float(m))
        if p is not None:
            last_p = p
        prices_1m.append(last_p)
    prices_1m_arr = np.array(prices_1m, dtype=float)
    alpha_9 = 2.0 / (9 + 1)
    alpha_20 = 2.0 / (20 + 1)
    ema9 = np.zeros_like(prices_1m_arr)
    ema20 = np.zeros_like(prices_1m_arr)
    if len(prices_1m_arr) > 0:
        ema9[0] = prices_1m_arr[0]
        ema20[0] = prices_1m_arr[0]
        for idx_m in range(1, len(prices_1m_arr)):
            ema9[idx_m] = prices_1m_arr[idx_m] * alpha_9 + ema9[idx_m - 1] * (1.0 - alpha_9)
            ema20[idx_m] = prices_1m_arr[idx_m] * alpha_20 + ema20[idx_m - 1] * (1.0 - alpha_20)
    crossovers: list[dict[str, Any]] = []
    if len(prices_1m_arr) > 0:
        last_state = ema9[0] > ema20[0]
        crossovers.append(
            {
                "ts": float(minutes[0]),
                "price": float(prices_1m_arr[0]),
                "dir": 1 if last_state else -1,
            }
        )
        for idx_m in range(1, len(prices_1m_arr)):
            curr_state = ema9[idx_m] > ema20[idx_m]
            if curr_state != last_state:
                crossovers.append(
                    {
                        "ts": float(minutes[idx_m]),
                        "price": float(prices_1m_arr[idx_m]),
                        "dir": 1 if curr_state else -1,
                    }
                )
                last_state = curr_state
    return {"minutes": minutes, "ema9": ema9, "ema20": ema20, "crossovers": crossovers}


def ema_inputs_at_ts(
    ts: float,
    spot: float,
    ema_ctx: Mapping[str, Any],
    open_ts: float,
) -> dict[str, float | None]:
    minutes = ema_ctx["minutes"]
    ema9 = ema_ctx["ema9"]
    ema20 = ema_ctx["ema20"]
    crossovers = ema_ctx["crossovers"]
    idx_ts = int((float(ts) - float(open_ts)) / 60.0)
    idx_ts = max(0, min(idx_ts, len(minutes) - 1))
    ema9_now = float(ema9[idx_ts]) if len(ema9) > 0 else None
    ema20_now = float(ema20[idx_ts]) if len(ema20) > 0 else None
    idx_1m_ago = max(0, idx_ts - 1)
    ema9_1m_ago = float(ema9[idx_1m_ago]) if len(ema9) > 0 else None
    ema9_gt_ema20 = 0.0
    ema_spread_vs_spot_pct = 0.0
    time_since_cross_min = 60.0
    price_dist_from_cross = 0.0
    if ema9_now is not None and ema20_now is not None and spot > 0:
        ema9_gt_ema20 = 1.0 if ema9_now > ema20_now else 0.0
        ema_spread_vs_spot_pct = float(100.0 * (ema9_now - ema20_now) / spot)
        latest_cross = crossovers[0]
        for cross in crossovers:
            if float(cross["ts"]) <= float(ts):
                latest_cross = cross
            else:
                break
        time_since_cross_min = float(max(0.0, (float(ts) - float(latest_cross["ts"])) / 60.0))
        price_dist_from_cross = float(
            100.0 * (spot - float(latest_cross["price"])) / float(latest_cross["price"])
        )
    return {
        "ema9_now": ema9_now,
        "ema20_now": ema20_now,
        "ema9_1m_ago": ema9_1m_ago,
        "ema9_gt_ema20": ema9_gt_ema20,
        "ema_spread_vs_spot_pct": ema_spread_vs_spot_pct,
        "time_since_cross_min": time_since_cross_min,
        "price_dist_from_cross": price_dist_from_cross,
    }


def missing_model_columns(
    features: Mapping[str, Any],
    *,
    columns: Sequence[str] | None = None,
) -> list[str]:
    cols = list(columns or replay_feature_columns())
    missing: list[str] = []
    for col in cols:
        if col not in features:
            missing.append(col)
            continue
        val = features[col]
        if val is None:
            missing.append(col)
            continue
        if isinstance(val, float) and math.isnan(val):
            missing.append(col)
    return missing


def model_feature_vector(
    features: Mapping[str, Any],
    *,
    columns: Sequence[str] | None = None,
) -> list[float] | None:
    """Ordered model inputs; None when any column is missing."""
    cols = list(columns or replay_feature_columns())
    missing = missing_model_columns(features, columns=cols)
    if missing:
        return None
    out: list[float] = []
    for col in cols:
        out.append(float(features[col]))
    return out


def build_strike_features(
    *,
    ts: float,
    index_timeline,
    option_timeline,
    option_type: str,
    strike_rupees: float,
    atm_strike_price: int | None = None,
    expiry_ts: float,
    open_ts: float,
    close_ts: float,
    ema_ctx: Mapping[str, Any] | None = None,
    missing_buckets: set[float] | None = None,
    strike_step: int = DEFAULT_STRIKE_STEP,
    live: bool = False,
    allow_partial_window: bool = False,
) -> FeatureBuildResult:
    """
    Build full feature dict for one option strike at ``ts``.

    Returns empty features when index spot is unavailable at ``ts``.
    With ``allow_partial_window``, still extracts features for probe/debug UI.
    """
    extract_timeline_features, find_atm_strike = _import_extract_timeline_features()
    target_ts = float(ts)
    window_ok = is_feature_window_complete(
        target_ts,
        index_timeline,
        missing_buckets=missing_buckets,
        open_ts=open_ts,
        close_ts=close_ts,
        as_of_ts=target_ts,
        live=live,
    )
    if not window_ok and not allow_partial_window:
        return FeatureBuildResult(
            features={},
            window_complete=False,
            model_complete=False,
            reason="index_feature_window_incomplete",
        )

    spot = index_ltp_rupees_at(index_timeline, target_ts)
    if spot is None or spot <= 0:
        return FeatureBuildResult(
            features={},
            window_complete=window_ok,
            model_complete=False,
            reason="no_index_spot_at_ts",
        )

    if ema_ctx is None:
        ema_ctx = precompute_ema_context(index_timeline, open_ts, close_ts)
    ema_inputs = ema_inputs_at_ts(target_ts, spot, ema_ctx, open_ts)

    atm = int(atm_strike_price) if atm_strike_price is not None else find_atm_strike(spot, strike_step)
    features = extract_timeline_features(
        ts=target_ts,
        index_timeline=index_timeline,
        option_timeline=option_timeline,
        option_type=str(option_type).upper(),
        strike_rupees=float(strike_rupees),
        atm_strike_price=atm,
        expiry_ts=float(expiry_ts),
        open_ts=float(open_ts),
        close_ts=float(close_ts),
        **ema_inputs,
    )
    if not features:
        return FeatureBuildResult(
            features={},
            window_complete=window_ok,
            model_complete=False,
            reason="extract_timeline_features_empty",
        )

    model_missing = missing_model_columns(features)
    return FeatureBuildResult(
        features=features,
        window_complete=window_ok,
        model_complete=len(model_missing) == 0,
        missing_model_columns=tuple(model_missing),
        reason="" if not model_missing else "incomplete_model_columns",
    )


def build_strike_features_from_rings(
    store: TickRingStore,
    *,
    index_key: str,
    option_token: str,
    ts: float,
    option_type: str,
    strike_rupees: float,
    expiry_ts: float,
    open_ts: float,
    close_ts: float,
    atm_strike_price: int | None = None,
    ema_ctx: Mapping[str, Any] | None = None,
    strike_step: int = DEFAULT_STRIKE_STEP,
) -> FeatureBuildResult:
    """Convenience: ring store → timelines → ``build_strike_features``."""
    index_tl = ring_to_tick_timeline(store, index_key)
    opt_tl = ring_to_tick_timeline(store, option_token)
    missing_buckets = missing_5s_buckets(open_ts, close_ts, index_tl)
    if ema_ctx is None and index_tl.timestamps:
        ema_ctx = precompute_ema_context(index_tl, open_ts, close_ts)
    return build_strike_features(
        ts=ts,
        index_timeline=index_tl,
        option_timeline=opt_tl,
        option_type=option_type,
        strike_rupees=strike_rupees,
        atm_strike_price=atm_strike_price,
        expiry_ts=expiry_ts,
        open_ts=open_ts,
        close_ts=close_ts,
        ema_ctx=ema_ctx,
        missing_buckets=missing_buckets,
        strike_step=strike_step,
    )


def feature_completeness_ratio(
    features: Mapping[str, Any],
    *,
    columns: Sequence[str] | None = None,
) -> float:
    """Fraction of model columns present (0..1) for UI / DQ display."""
    cols = list(columns or replay_feature_columns())
    if not cols:
        return 0.0
    present = len(cols) - len(missing_model_columns(features, columns=cols))
    return round(present / len(cols), 4)
