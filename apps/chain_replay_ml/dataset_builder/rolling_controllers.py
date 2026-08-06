"""Gap-reset rolling controllers for token-scoped statistics (LTP EMA, STD20)."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Mapping, Sequence

import numpy as np

# Default feature grid step (seconds) for IV calendar-window warmup counts.
IV_GRID_STEP_SEC = 3.0

# Features whose readiness is enforced at emit time by controllers — skip NULL pass.
CONTROLLER_OWNED_READINESS_FEATURES: frozenset[str] = frozenset({
    # Wave 2: canonical LTP / IV / spot / HL levels
    "ltp_ema9",
    "ltp_ema20",
    "ltp_ema50",
    "ltp_ema100",
    "ltp_ema200",
    "ltp_ema300",
    "ltp_std20",
    "iv_ema9",
    "iv_ema20",
    "iv_ema50",
    "iv_ema100",
    "iv_ema200",
    "iv_ema300",
    "spot_ema9",
    "spot_ema20",
    "spot_ema50",
    "spot_ema100",
    "spot_ema200",
    "spot_ema300",
    "spot_high_ema20",
    "spot_high_ema50",
    "spot_high_ema100",
    "spot_high_ema200",
    "spot_high_ema300",
    "spot_low_ema20",
    "spot_low_ema50",
    "spot_low_ema100",
    "spot_low_ema200",
    "spot_low_ema300",
    # Wave 5: channel width levels (ltp÷width → Interaction)
    "spot_ema20_channel_width",
    "spot_ema50_channel_width",
    "spot_ema100_channel_width",
    "spot_ema200_channel_width",
    "spot_ema300_channel_width",
    # Wave 3: canonical weighted blend levels
    "weighted_ltp_ema",
    "weighted_spot_ema",
    "weighted_spot_high_ema",
    "weighted_spot_low_ema",
    "weighted_spot_close_ema",
    "opt_rv_5m",
    "opt_rv_10m",
    # opt_rv_ratio → Interaction
    # Wave 6: spot_vs_ema20_pct / ema_spread_* → Interaction
    "ema9_slope",
    "ema9_gt_ema20",
    "time_since_cross_min",
    "cross_age_decay",
    "price_dist_from_cross_pct",
    "spot_rv_5m",
    "spot_rv_10m",
    # spot_rv_ratio → Interaction
    "iv_zscore_1m",
    "iv_zscore_5m",
    "iv_zscore_15m",
    "iv_zscore_30m",
    "iv_rank_session",
    "iv_change_1m",
    "iv_change_5m",
    "iv_change_15m",
    "iv_pct_change_1m",
    "roll_iv",
    "roll_age_min",
    "rows_since_roll",
    "bs_reiv_pred",
    "dgt_reiv_pred",
    "iv_drift_from_roll",
    # DGT lag/change/error lookbacks → Pipeline Owned or retired
    "dgt_prediction_error",
    # dgt_reiv_to_ltp_ratio / dgt_to_spot_ratio → Interaction
})


@dataclass(frozen=True)
class ControllerSample:
    """Stable per-tick input bundle — controllers read only the scalars they need."""

    values: Mapping[str, float]
    ts: float | None = None

    @classmethod
    def ltp(cls, price: float, ts: float | None = None) -> ControllerSample:
        return cls(values={"ltp": float(price)}, ts=ts)

    @classmethod
    def spot(cls, price: float, ts: float | None = None) -> ControllerSample:
        return cls(values={"spot": float(price)}, ts=ts)

    @classmethod
    def iv(cls, value: float, ts: float | None = None) -> ControllerSample:
        return cls(values={"iv": float(value)}, ts=ts)

    def get(self, key: str, default: float | None = None) -> float | None:
        if key in self.values:
            return float(self.values[key])
        if default is not None:
            return float(default)
        if len(self.values) == 1:
            return float(next(iter(self.values.values())))
        return None


def assert_monotonic_controller_ts(
    previous_ts: float | None,
    new_ts: float | None,
) -> None:
    """Debug-only: every update must have new_timestamp >= previous_timestamp.

    Failure almost always indicates a replay/test harness ordering bug, not bad
    controller math. Stripped when python -O.
    """
    if not __debug__:
        return
    if previous_ts is None or new_ts is None:
        return
    prev_f = float(previous_ts)
    new_f = float(new_ts)
    assert new_f >= prev_f, (
        f"controller update timestamp went backward: {new_f} < {prev_f} "
        "(check replay row order / gap timestamp simulation)"
    )


def assert_rolling_controller_reset_complete(controller: RollingController) -> None:
    """Debug-only: reset() must clear all cached rolling state. Stripped with python -O."""
    if not __debug__:
        return
    assert controller.samples == 0, "samples must be 0 after reset()"
    assert not controller.ready(), "ready() must be False after reset()"
    assert controller.value() is None, "value() must be None after reset()"
    assert controller.last_update_ts is None, "last_update_ts must be None after reset()"


class RollingController:
    """Base lifecycle for one rolling statistic on a scalar input stream."""

    warmup_period: int = 0
    sample_fields: tuple[str, ...] = ("value",)

    def __init__(self) -> None:
        self._samples: int = 0
        self._last_reset_ts: float | None = None
        self._last_update_ts: float | None = None

    def reset(self, ts: float | None = None) -> None:
        self._samples = 0
        self._last_reset_ts = float(ts) if ts is not None else None
        self._last_update_ts = None
        self._clear_value()
        assert_rolling_controller_reset_complete(self)
        self._assert_invariants()

    def reset_feature_label(self) -> str:
        return f"EMA{self.warmup_period}"

    def update(self, sample: ControllerSample | float, ts: float | None = None) -> None:
        """Accept ``ControllerSample`` (preferred) or legacy ``(value, ts)`` scalar."""
        if isinstance(sample, ControllerSample):
            self._update_sample(sample)
            return
        self._update_sample(ControllerSample(values={"value": float(sample)}, ts=ts))

    def _update_sample(self, sample: ControllerSample) -> None:
        raise NotImplementedError

    def ready(self) -> bool:
        return self._samples >= int(self.warmup_period)

    @property
    def samples(self) -> int:
        return self._samples

    @property
    def last_reset_ts(self) -> float | None:
        return self._last_reset_ts

    @property
    def last_update_ts(self) -> float | None:
        return self._last_update_ts

    def value(self) -> float | None:
        raise NotImplementedError

    def _clear_value(self) -> None:
        pass

    def _record_update(self, ts: float | None) -> None:
        if ts is not None:
            assert_monotonic_controller_ts(self._last_update_ts, ts)
        self._samples += 1
        if ts is not None:
            self._last_update_ts = float(ts)
        self._assert_invariants()

    def _assert_invariants(self) -> None:
        """Debug-only runtime checks (stripped when python -O)."""
        if not __debug__:
            return
        assert self._samples >= 0
        assert self.ready() == (self._samples >= int(self.warmup_period))
        if self._last_reset_ts is not None and self._last_update_ts is not None:
            assert self._last_update_ts >= self._last_reset_ts


class EmaController(RollingController):
    """EMA with α = 2/(period+1); first sample after reset seeds the EMA."""

    sample_fields = ("ltp",)

    def __init__(self, period: int) -> None:
        super().__init__()
        self.warmup_period = int(period)
        self._alpha = 2.0 / (float(self.warmup_period) + 1.0)
        self._ema: float | None = None

    def _update_sample(self, sample: ControllerSample) -> None:
        raw = sample.get("ltp")
        if raw is None:
            raw = sample.get("value")
        if raw is None:
            return
        price = float(raw)
        if self._samples == 0:
            self._ema = price
        else:
            prev = self._ema if self._ema is not None else price
            self._ema = price * self._alpha + prev * (1.0 - self._alpha)
        self._record_update(sample.ts)

    def value(self) -> float | None:
        if not self.ready() or self._ema is None:
            return None
        return float(self._ema)

    def _clear_value(self) -> None:
        self._ema = None


def _population_std_buffer(buffer: Deque[float]) -> float:
    """Thin dispatch to performance kernel (Numba when enabled); same as np.std ddof=0."""
    from chain_replay_ml.performance.runtime import population_std

    return float(population_std(buffer))


class StdController(RollingController):
    """Rolling population std (ddof=0) over the last N LTP samples."""

    sample_fields = ("ltp",)

    def __init__(self, period: int = 20) -> None:
        super().__init__()
        self.warmup_period = int(period)
        self._buffer: Deque[float] = deque(maxlen=self.warmup_period)
        self._cached_std: float | None = None

    def reset_feature_label(self) -> str:
        return "STD20"

    def _update_sample(self, sample: ControllerSample) -> None:
        raw = sample.get("ltp")
        if raw is None:
            raw = sample.get("value")
        if raw is None:
            return
        self._buffer.append(float(raw))
        self._record_update(sample.ts)
        if len(self._buffer) >= self.warmup_period:
            self._cached_std = _population_std_buffer(self._buffer)
        else:
            self._cached_std = None

    def value(self) -> float | None:
        if not self.ready() or len(self._buffer) < self.warmup_period:
            return None
        if self._cached_std is None:
            self._cached_std = _population_std_buffer(self._buffer)
        return float(self._cached_std)

    def _clear_value(self) -> None:
        self._buffer.clear()
        self._cached_std = None


class RvController(RollingController):
    """Rolling population std (ddof=0) of percentage returns over the last N samples."""

    sample_fields = ("ltp", "spot")

    def __init__(self, period: int) -> None:
        super().__init__()
        self.warmup_period = int(period)
        self._buffer: Deque[float] = deque(maxlen=self.warmup_period)
        self._last_price: float | None = None
        self._cached_std: float | None = None

    def reset_feature_label(self) -> str:
        return "RV5M" if self.warmup_period <= 30 else "RV10M"

    def _update_sample(self, sample: ControllerSample) -> None:
        raw = sample.get("ltp")
        if raw is None:
            raw = sample.get("spot")
        if raw is None:
            raw = sample.get("value")
        if raw is None or float(raw) <= 0:
            return
        price = float(raw)
        if self._last_price is None:
            self._last_price = price
            return
        ret = (price - self._last_price) / self._last_price * 100.0
        self._last_price = price
        self._buffer.append(ret)
        self._record_update(sample.ts)
        if len(self._buffer) >= self.warmup_period:
            self._cached_std = _population_std_buffer(self._buffer)
        else:
            self._cached_std = None

    def value(self) -> float | None:
        if not self.ready() or len(self._buffer) < self.warmup_period:
            return None
        if self._cached_std is None:
            self._cached_std = _population_std_buffer(self._buffer)
        return float(self._cached_std)

    def _clear_value(self) -> None:
        self._buffer.clear()
        self._last_price = None
        self._cached_std = None


class IvZscoreWindowController(RollingController):
    """Calendar-window IV z-score: prior entries in [ts-window, ts) vs current IV."""

    sample_fields = ("iv",)

    def __init__(self, window_sec: float, warmup_period: int) -> None:
        super().__init__()
        self.window_sec = float(window_sec)
        self.warmup_period = int(warmup_period)
        self._history: Deque[tuple[float, float]] = deque()
        self._zscore: float | None = None

    def reset_feature_label(self) -> str:
        labels = {60.0: "IVZ1M", 300.0: "IVZ5M", 900.0: "IVZ15M", 1800.0: "IVZ30M"}
        return labels.get(self.window_sec, "IVZ")

    def _update_sample(self, sample: ControllerSample) -> None:
        raw = sample.get("iv")
        if raw is None:
            raw = sample.get("value")
        if raw is None or sample.ts is None:
            return
        iv = float(raw)
        ts = float(sample.ts)
        if self._samples + 1 >= self.warmup_period:
            cutoff = ts - self.window_sec
            priors = [v for t, v in self._history if t >= cutoff and t < ts]
            if priors:
                from chain_replay_ml.performance.runtime import iv_zscore

                self._zscore = float(iv_zscore(priors, iv, eps=1e-8))
            else:
                self._zscore = 0.0
        else:
            self._zscore = None
        self._history.append((ts, iv))
        self._prune(ts)
        self._record_update(sample.ts)

    def _prune(self, current_ts: float) -> None:
        cutoff = current_ts - self.window_sec
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def value(self) -> float | None:
        if not self.ready() or self._zscore is None:
            return None
        return float(self._zscore)

    def _clear_value(self) -> None:
        self._history.clear()
        self._zscore = None


# Calendar lag windows for token.iv_history (seconds, warmup samples @ IV_GRID_STEP_SEC).
IV_HISTORY_LAG_SPECS: tuple[tuple[float, int], ...] = (
    (60.0, int(60.0 / IV_GRID_STEP_SEC)),
    (300.0, int(300.0 / IV_GRID_STEP_SEC)),
    (900.0, int(900.0 / IV_GRID_STEP_SEC)),
)


def _iv_pct_change(cur_pct: float | None, past_pct: float | None) -> float | None:
    if cur_pct is None or past_pct is None:
        return None
    if past_pct <= 0:
        return 0.0 if cur_pct == 0 else None
    return float((cur_pct - past_pct) / past_pct * 100.0)


class IvHistoryController:
    """Single deque of (ts, iv) — calendar lag lookups for iv_change_* features."""

    sample_fields = ("iv",)

    def __init__(self) -> None:
        self._history: Deque[tuple[float, float]] = deque()
        self._samples: int = 0
        self._last_reset_ts: float | None = None
        self._last_update_ts: float | None = None
        self._current_iv: float | None = None
        self._current_iv_pct: float | None = None

    def reset(self, ts: float | None = None) -> None:
        self._history.clear()
        self._samples = 0
        self._last_reset_ts = float(ts) if ts is not None else None
        self._last_update_ts = None
        self._current_iv = None
        self._current_iv_pct = None
        assert_iv_history_reset_complete(self)

    @property
    def samples(self) -> int:
        return self._samples

    @property
    def last_update_ts(self) -> float | None:
        return self._last_update_ts

    def reset_feature_label(self) -> str:
        return "IVHIST"

    def update(self, iv: float, *, ts: float) -> None:
        iv_f = float(iv)
        ts_f = float(ts)
        assert_monotonic_controller_ts(self._last_update_ts, ts_f)
        self._current_iv = iv_f
        self._current_iv_pct = iv_f * 100.0
        self._history.append((ts_f, iv_f))
        self._prune(ts_f)
        self._samples += 1
        self._last_update_ts = ts_f
        self._assert_invariants()

    def _assert_invariants(self) -> None:
        if not __debug__:
            return
        if self._last_reset_ts is not None and self._last_update_ts is not None:
            assert self._last_update_ts >= self._last_reset_ts

    def _prune(self, current_ts: float) -> None:
        max_lag = max(lag for lag, _ in IV_HISTORY_LAG_SPECS)
        cutoff = current_ts - max_lag - IV_GRID_STEP_SEC
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

    def _lookup_iv_at_or_before(self, target_ts: float) -> float | None:
        for t, v in reversed(self._history):
            if t <= target_ts + 1e-6:
                return float(v)
        return None

    def lag_iv(self, lag_sec: float) -> float | None:
        """IV (decimal) at or before current_ts − lag_sec from post-reset history only."""
        if self._last_update_ts is None:
            return None
        return self._lookup_iv_at_or_before(float(self._last_update_ts) - float(lag_sec))

    def warmup_for_lag(self, lag_sec: float) -> int:
        for lag, warmup in IV_HISTORY_LAG_SPECS:
            if float(lag) == float(lag_sec):
                return int(warmup)
        return int(float(lag_sec) / IV_GRID_STEP_SEC)

    def ready_for_lag(self, lag_sec: float) -> bool:
        warmup = self.warmup_for_lag(lag_sec)
        if self._samples < warmup:
            return False
        return self.lag_iv(lag_sec) is not None

    def iv_change_pct_points(self, lag_sec: float) -> float | None:
        """current_iv% − lag_iv% — NULL until calendar lag is covered post-reset."""
        if not self.ready_for_lag(lag_sec):
            return None
        lag = self.lag_iv(lag_sec)
        if lag is None or self._current_iv_pct is None:
            return None
        return float(self._current_iv_pct - lag * 100.0)

    def iv_pct_change_for_lag(self, lag_sec: float) -> float | None:
        if not self.ready_for_lag(lag_sec):
            return None
        lag = self.lag_iv(lag_sec)
        if lag is None or self._current_iv_pct is None:
            return None
        return _iv_pct_change(self._current_iv_pct, lag * 100.0)

    @property
    def sample_count(self) -> int | None:
        if not __debug__:
            return None
        return len(self._history)

    @property
    def oldest_timestamp(self) -> float | None:
        if not __debug__:
            return None
        return float(self._history[0][0]) if self._history else None

    @property
    def newest_timestamp(self) -> float | None:
        if not __debug__:
            return None
        return float(self._history[-1][0]) if self._history else None


def assert_iv_history_reset_complete(controller: IvHistoryController) -> None:
    """Debug-only: IvHistoryController reset() must clear deque and cached IV."""
    if not __debug__:
        return
    assert controller.samples == 0
    assert controller.last_update_ts is None
    assert controller.iv_change_pct_points(60.0) is None
    if controller.sample_count is not None:
        assert controller.sample_count == 0


class RollController:
    """Roll anchor state machine for token.roll features (gap-reset, immediate warmup)."""

    sample_fields = ("iv", "spot", "ltp")
    warmup_period = 0

    def __init__(self) -> None:
        from chain_replay_ml.reanchor import RollState

        self._roll = RollState()
        self._session_initialized = False
        self._rows_since_roll = -1
        self._last_reset_ts: float | None = None
        self._last_update_ts: float | None = None

    def reset(self, ts: float | None = None) -> None:
        from chain_replay_ml.reanchor import RollState

        self._roll = RollState()
        self._session_initialized = False
        self._rows_since_roll = -1
        self._last_reset_ts = float(ts) if ts is not None else None
        self._last_update_ts = None
        assert_roll_controller_reset_complete(self)

    def reset_feature_label(self) -> str:
        return "ROLL"

    def ready(self) -> bool:
        return self._session_initialized

    @property
    def samples(self) -> int:
        return 1 if self._session_initialized else 0

    @property
    def last_reset_ts(self) -> float | None:
        return self._last_reset_ts

    @property
    def last_update_ts(self) -> float | None:
        return self._last_update_ts

    @property
    def roll_iv(self) -> float | None:
        return self._roll.roll_iv

    @property
    def roll_anchor_ts(self) -> float:
        return self._roll.roll_anchor_ts

    @property
    def roll_spot(self) -> float | None:
        return self._roll.roll_spot

    @property
    def roll_ltp(self) -> float | None:
        return self._roll.roll_ltp

    @property
    def roll_greeks(self) -> dict[str, float]:
        return self._roll.roll_greeks

    @property
    def rows_since_roll(self) -> int:
        return self._rows_since_roll

    def roll_age_min(self, ts: float) -> float | None:
        if not self._session_initialized:
            return None
        return (float(ts) - self._roll.roll_anchor_ts) / 60.0

    def value(self) -> float | None:
        if not self.ready() or self._roll.roll_iv is None:
            return None
        return float(self._roll.roll_iv)

    def update(
        self,
        *,
        actual_iv: float | None,
        spot: float | None,
        ltp: float | None,
        ts: float,
        option_type: str,
        strike_rupees: float,
        expiry_ts: float,
        thresholds: Any | None = None,
    ) -> None:
        from chain_replay_ml import bs
        from chain_replay_ml.constants import RISK_FREE_RATE
        from chain_replay_ml.reanchor import ReanchorThresholds, apply_roll, evaluate_triggers

        if actual_iv is None or spot is None or ltp is None:
            return
        t_exp = bs.time_to_expiry_years(expiry_ts, float(ts))
        if t_exp <= 0:
            return

        ts_f = float(ts)
        assert_monotonic_controller_ts(self._last_update_ts, ts_f)
        thr = thresholds or ReanchorThresholds()

        if not self._session_initialized:
            self._roll.roll_iv = float(actual_iv)
            self._roll.roll_anchor_ts = ts_f
            self._roll.roll_spot = float(spot)
            self._roll.roll_ltp = float(ltp)
            self._roll.roll_greeks = bs.greeks(
                option_type, float(spot), strike_rupees, RISK_FREE_RATE, t_exp, float(actual_iv),
            )
            self._session_initialized = True
            self._rows_since_roll = 0
        elif ts_f > self._roll.roll_anchor_ts + 0.001:
            should_roll, _ = evaluate_triggers(
                actual_iv=float(actual_iv),
                actual_spot=float(spot),
                roll=self._roll,
                row_ts=ts_f,
                thresholds=thr,
            )
            if should_roll:
                apply_roll(
                    self._roll,
                    actual_iv=float(actual_iv),
                    actual_spot=float(spot),
                    actual_ltp=float(ltp),
                    row_ts=ts_f,
                    option_type=option_type,
                    strike_rupees=strike_rupees,
                    expiry_ts=expiry_ts,
                )
                self._rows_since_roll = 0
            else:
                self._rows_since_roll += 1

        self._last_update_ts = ts_f


def assert_roll_controller_reset_complete(controller: RollController) -> None:
    """Debug-only: RollController reset() must clear roll anchor state."""
    if not __debug__:
        return
    assert not controller.ready()
    assert controller.value() is None
    assert controller.rows_since_roll == -1
    assert controller.last_update_ts is None
    assert controller.roll_iv is None


def update_token_roll_controller(
    controllers: TokenControllers,
    *,
    actual_iv: float | None,
    spot: float | None,
    ltp: float | None,
    ts: float,
    option_type: str,
    strike_rupees: float,
    expiry_ts: float,
    thresholds: Any | None = None,
) -> None:
    """Ingest one row into token.roll controller."""
    from .build_profiler import get_profiler, profile_call

    if get_profiler() is None:
        controllers.roll.update(
            actual_iv=actual_iv,
            spot=spot,
            ltp=ltp,
            ts=ts,
            option_type=option_type,
            strike_rupees=strike_rupees,
            expiry_ts=expiry_ts,
            thresholds=thresholds,
        )
        return
    profile_call(
        "controller.roll",
        lambda: controllers.roll.update(
            actual_iv=actual_iv,
            spot=spot,
            ltp=ltp,
            ts=ts,
            option_type=option_type,
            strike_rupees=strike_rupees,
            expiry_ts=expiry_ts,
            thresholds=thresholds,
        ),
        rows=1,
    )


def update_token_dgt_controller(
    controllers: TokenControllers,
    *,
    ts: float,
    ltp: float | None,
    dgt_reiv: float | None,
) -> None:
    """Append one row to token.dgt snapshot history (after emit)."""
    from .build_profiler import get_profiler, profile_call

    if get_profiler() is None:
        controllers.dgt.record_sample(ts=ts, ltp=ltp, dgt_reiv=dgt_reiv)
        return
    profile_call(
        "controller.dgt",
        lambda: controllers.dgt.record_sample(ts=ts, ltp=ltp, dgt_reiv=dgt_reiv),
        rows=1,
    )


def emit_roll_features(
    controller: RollController,
    *,
    actual_iv: float | None,
    spot: float | None,
    ltp: float | None,
    t_exp: float,
    option_type: str,
    strike_rupees: float,
    ts: float,
) -> dict[str, float | None]:
    """Emit token.roll owned features — NULL until session initialized."""
    from chain_replay_ml import bs
    from chain_replay_ml.constants import RISK_FREE_RATE
    from chain_replay_ml.reanchor import iv_drift_from_roll_pct

    null_row: dict[str, float | None] = {
        "roll_iv": None,
        "roll_age_min": None,
        "rows_since_roll": None,
        "bs_reiv_pred": None,
        "dgt_reiv_pred": None,
        "iv_drift_from_roll": None,
    }
    if not controller.ready():
        return null_row

    roll_iv = controller.roll_iv
    roll_age_min = controller.roll_age_min(ts)
    roll_fwd_min = max(0.0, roll_age_min or 0.0)

    bs_reiv = None
    if roll_iv and roll_iv > 0 and spot is not None and t_exp > 0:
        bs_reiv = max(
            0.0,
            bs.bs_price(option_type, float(spot), strike_rupees, RISK_FREE_RATE, t_exp, roll_iv),
        )

    dgt_reiv = None
    if controller.roll_greeks and controller.roll_ltp and controller.roll_spot is not None and spot is not None:
        dgt_reiv = bs.greek_predicted_ltp(
            controller.roll_ltp,
            controller.roll_greeks,
            float(spot) - controller.roll_spot,
            roll_fwd_min,
            0.0,
        )

    roll_iv_pct = roll_iv * 100.0 if roll_iv else None
    rows_since = (
        float(controller.rows_since_roll) if controller.rows_since_roll >= 0 else None
    )

    return {
        "roll_iv": roll_iv_pct,
        "roll_age_min": roll_age_min,
        "rows_since_roll": rows_since,
        "bs_reiv_pred": bs_reiv,
        "dgt_reiv_pred": dgt_reiv,
        "iv_drift_from_roll": iv_drift_from_roll_pct(actual_iv, roll_iv),
    }


# Lag/change lookbacks are Pipeline Owned (or retired for 10s). Controller keeps
# only current-state DGT features; horizons retained empty for API compatibility.
DGT_LAG_HORIZONS: tuple[tuple[str, float], ...] = ()
_DGT_ERROR_LAG_SUFFIXES = frozenset()
_DGT_SNAPSHOT_MAX = 120

DGT_OWNED_FEATURES: frozenset[str] = frozenset({
    "dgt_prediction_error",
    # dgt_reiv_to_ltp_ratio / dgt_to_spot_ratio → Interaction (operands already Registry)
})


class DgtController:
    """Per-token DGT snapshot history — calendar lag lookups for dgt_* features."""

    sample_fields = ("ltp",)

    def __init__(self) -> None:
        self._snapshots: Deque[tuple[float, float | None, float | None]] = deque(
            maxlen=_DGT_SNAPSHOT_MAX,
        )
        self._last_reset_ts: float | None = None
        self._last_update_ts: float | None = None

    def reset(self, ts: float | None = None) -> None:
        self._snapshots.clear()
        self._last_reset_ts = float(ts) if ts is not None else None
        self._last_update_ts = None

    def reset_feature_label(self) -> str:
        return "DGT"

    def ready(self) -> bool:
        return bool(self._snapshots)

    @property
    def samples(self) -> int:
        return len(self._snapshots)

    @property
    def last_update_ts(self) -> float | None:
        return self._last_update_ts

    def value(self) -> float | None:
        return None

    def _snapshot_at_or_before(
        self,
        target_ts: float,
    ) -> tuple[float, float | None, float | None] | None:
        best = None
        for snap in self._snapshots:
            if snap[0] <= target_ts + 0.001:
                best = snap
            else:
                break
        return best

    def record_sample(
        self,
        *,
        ts: float,
        ltp: float | None,
        dgt_reiv: float | None,
    ) -> None:
        if ltp is None and dgt_reiv is None:
            return
        ts_f = float(ts)
        assert_monotonic_controller_ts(self._last_update_ts, ts_f)
        self._snapshots.append((ts_f, ltp, dgt_reiv))
        self._last_update_ts = ts_f

    def emit_features(
        self,
        *,
        ts: float,
        ltp: float | None,
        dgt_reiv: float | None,
        spot: float | None,
    ) -> dict[str, float | None]:
        """Emit current-state token.dgt features (lookbacks are Pipeline Owned)."""
        out: dict[str, float | None] = {name: None for name in DGT_OWNED_FEATURES}

        if ltp is not None and dgt_reiv is not None:
            out["dgt_prediction_error"] = float(ltp - dgt_reiv)

        return out


def emit_dgt_features(
    controller: DgtController,
    *,
    ts: float,
    ltp: float | None,
    dgt_reiv: float | None,
    spot: float | None,
) -> dict[str, float | None]:
    return controller.emit_features(ts=ts, ltp=ltp, dgt_reiv=dgt_reiv, spot=spot)


class IvSessionRankController(RollingController):
    """Session IV rank: (IV - session_min) / (session_max - session_min) * 100."""

    sample_fields = ("iv",)
    warmup_period = 1

    def __init__(self) -> None:
        super().__init__()
        self._min_iv: float | None = None
        self._max_iv: float | None = None
        self._rank: float | None = None

    def reset_feature_label(self) -> str:
        return "IVRANK"

    def _update_sample(self, sample: ControllerSample) -> None:
        raw = sample.get("iv")
        if raw is None:
            raw = sample.get("value")
        if raw is None:
            return
        iv = float(raw)
        if self._min_iv is None:
            self._min_iv = iv
            self._max_iv = iv
            self._rank = 50.0
        else:
            self._min_iv = min(self._min_iv, iv)
            self._max_iv = max(self._max_iv, iv)
            span = self._max_iv - self._min_iv
            self._rank = (iv - self._min_iv) / span * 100.0 if span > 1e-12 else 50.0
        self._record_update(sample.ts)

    def value(self) -> float | None:
        if not self.ready() or self._rank is None:
            return None
        return float(self._rank)

    def _clear_value(self) -> None:
        self._min_iv = None
        self._max_iv = None
        self._rank = None


def emit_controller_value(controller: RollingController) -> float | None:
    """Emit controller.value() only when ready and a numeric value exists."""
    if not controller.ready():
        return None
    val = controller.value()
    if val is None:
        return None
    return float(val)


def emit_controller_ratio(controller: RollingController, denominator: Any) -> float | None:
    """Emit controller.value() / denominator when ready; otherwise None."""
    val = emit_controller_value(controller)
    if val is None:
        return None
    try:
        denom = float(denominator)
    except (TypeError, ValueError):
        return None
    if denom <= 0:
        return None
    return float(val) / denom


def emit_when_ready(
    controllers: Sequence[RollingController],
    fn: Callable[[], float | None],
) -> float | None:
    """Emit fn() only when every controller is ready and has a value."""
    if not controllers or not all(
        c.ready() and c.value() is not None for c in controllers
    ):
        return None
    return fn()


def emit_controller_derived_quotient(
    numerator: RollingController,
    denominator: RollingController,
    *,
    eps: float = 1e-9,
) -> float | None:
    """ControllerDerived emission: NULL unless both sources are ready with values."""
    num_val = emit_controller_value(numerator)
    den_val = emit_controller_value(denominator)
    if num_val is None or den_val is None:
        return None
    return float(num_val / (den_val + eps))


def guard_controller_derived_rv_features(out: Mapping[str, Any]) -> None:
    """NULL *_rv_ratio when either RV source column is NULL (warm-up coherence)."""
    if not isinstance(out, dict):
        return
    for prefix in ("opt", "spot"):
        rv5 = out.get(f"{prefix}_rv_5m")
        rv10 = out.get(f"{prefix}_rv_10m")
        ratio_key = f"{prefix}_rv_ratio"
        if ratio_key not in out:
            continue
        if out[ratio_key] is None:
            continue
        if rv5 is None or rv10 is None:
            out[ratio_key] = None


@dataclass
class TokenControllers:
    ema9: EmaController = field(default_factory=lambda: EmaController(9))
    ema20: EmaController = field(default_factory=lambda: EmaController(20))
    ema50: EmaController = field(default_factory=lambda: EmaController(50))
    ema100: EmaController = field(default_factory=lambda: EmaController(100))
    ema200: EmaController = field(default_factory=lambda: EmaController(200))
    ema300: EmaController = field(default_factory=lambda: EmaController(300))
    std20: StdController = field(default_factory=StdController)
    rv5m: RvController = field(default_factory=lambda: RvController(30))
    rv10m: RvController = field(default_factory=lambda: RvController(60))
    iv_ema9: EmaController = field(default_factory=lambda: EmaController(9))
    iv_ema20: EmaController = field(default_factory=lambda: EmaController(20))
    iv_ema50: EmaController = field(default_factory=lambda: EmaController(50))
    iv_ema100: EmaController = field(default_factory=lambda: EmaController(100))
    iv_ema200: EmaController = field(default_factory=lambda: EmaController(200))
    iv_ema300: EmaController = field(default_factory=lambda: EmaController(300))
    iv_zscore_1m: IvZscoreWindowController = field(
        default_factory=lambda: IvZscoreWindowController(60.0, int(60.0 / IV_GRID_STEP_SEC)),
    )
    iv_zscore_5m: IvZscoreWindowController = field(
        default_factory=lambda: IvZscoreWindowController(300.0, int(300.0 / IV_GRID_STEP_SEC)),
    )
    iv_zscore_15m: IvZscoreWindowController = field(
        default_factory=lambda: IvZscoreWindowController(900.0, int(900.0 / IV_GRID_STEP_SEC)),
    )
    iv_zscore_30m: IvZscoreWindowController = field(
        default_factory=lambda: IvZscoreWindowController(1800.0, int(1800.0 / IV_GRID_STEP_SEC)),
    )
    iv_session_rank: IvSessionRankController = field(default_factory=IvSessionRankController)
    iv_history: IvHistoryController = field(default_factory=IvHistoryController)
    roll: RollController = field(default_factory=RollController)
    dgt: DgtController = field(default_factory=DgtController)

    def __post_init__(self) -> None:
        # Architecture: ensure Controller Registry is loaded (no emission change).
        from .controller_registry import ensure_architecture_registry

        ensure_architecture_registry()

    def reset_all(
        self,
        ts: float | None = None,
        *,
        token: str | None = None,
        previous_ts: float | None = None,
        gap_limit: float | None = None,
        reason: str = "row_gap",
    ) -> None:
        from .gap_policy_instrumentation import log_controller_reset

        gap = None
        if previous_ts is not None and ts is not None:
            gap = float(ts) - float(previous_ts)
        for ctrl in (
            self.ema9, self.ema20, self.ema50, self.ema100, self.ema200, self.ema300,
            self.std20, self.rv5m, self.rv10m,
            self.iv_ema9, self.iv_ema20, self.iv_ema50, self.iv_ema100, self.iv_ema200, self.iv_ema300,
            self.iv_zscore_1m, self.iv_zscore_5m, self.iv_zscore_15m, self.iv_zscore_30m,
            self.iv_session_rank,
        ):
            log_controller_reset(
                token=token,
                feature=ctrl.reset_feature_label(),
                previous_ts=previous_ts,
                current_ts=ts,
                gap=gap,
                gap_limit=gap_limit,
                reason=reason,
            )
            ctrl.reset(ts)
        log_controller_reset(
            token=token,
            feature=self.iv_history.reset_feature_label(),
            previous_ts=previous_ts,
            current_ts=ts,
            gap=gap,
            gap_limit=gap_limit,
            reason=reason,
        )
        self.iv_history.reset(ts)
        log_controller_reset(
            token=token,
            feature=self.roll.reset_feature_label(),
            previous_ts=previous_ts,
            current_ts=ts,
            gap=gap,
            gap_limit=gap_limit,
            reason=reason,
        )
        self.roll.reset(ts)
        log_controller_reset(
            token=token,
            feature=self.dgt.reset_feature_label(),
            previous_ts=previous_ts,
            current_ts=ts,
            gap=gap,
            gap_limit=gap_limit,
            reason=reason,
        )
        self.dgt.reset(ts)

    def ema_controller_periods(self) -> dict[str, int]:
        return {
            "token.ltp.ema9": self.ema9.warmup_period,
            "token.ltp.ema20": self.ema20.warmup_period,
            "token.ltp.ema50": self.ema50.warmup_period,
            "token.ltp.ema100": self.ema100.warmup_period,
            "token.ltp.ema200": self.ema200.warmup_period,
            "token.ltp.ema300": self.ema300.warmup_period,
            "token.iv.ema9": self.iv_ema9.warmup_period,
            "token.iv.ema20": self.iv_ema20.warmup_period,
            "token.iv.ema50": self.iv_ema50.warmup_period,
            "token.iv.ema100": self.iv_ema100.warmup_period,
            "token.iv.ema200": self.iv_ema200.warmup_period,
            "token.iv.ema300": self.iv_ema300.warmup_period,
        }


def update_token_ltp_controllers(
    controllers: TokenControllers,
    ltp: float | None,
    *,
    ts: float | None = None,
) -> None:
    """Ingest one LTP sample into all token LTP rolling controllers."""
    if ltp is None:
        return
    sample = ControllerSample.ltp(float(ltp), ts)
    from .build_profiler import get_profiler, profile_call

    if get_profiler() is None:
        controllers.ema9.update(sample)
        controllers.ema20.update(sample)
        controllers.ema50.update(sample)
        controllers.ema100.update(sample)
        controllers.ema200.update(sample)
        controllers.ema300.update(sample)
        controllers.std20.update(sample)
        return
    profile_call("controller.token.ema9", lambda: controllers.ema9.update(sample), rows=1)
    profile_call("controller.token.ema20", lambda: controllers.ema20.update(sample), rows=1)
    profile_call("controller.token.ema50", lambda: controllers.ema50.update(sample), rows=1)
    profile_call("controller.token.ema100", lambda: controllers.ema100.update(sample), rows=1)
    profile_call("controller.token.ema200", lambda: controllers.ema200.update(sample), rows=1)
    profile_call("controller.token.ema300", lambda: controllers.ema300.update(sample), rows=1)
    profile_call("controller.token.std20", lambda: controllers.std20.update(sample), rows=1)


def update_token_rv_controllers(
    controllers: TokenControllers,
    ltp: float | None,
    *,
    ts: float | None = None,
) -> None:
    """Ingest one LTP sample into token RV rolling controllers."""
    if ltp is None:
        return
    sample = ControllerSample.ltp(float(ltp), ts)
    from .build_profiler import get_profiler, profile_call

    if get_profiler() is None:
        controllers.rv5m.update(sample)
        controllers.rv10m.update(sample)
        return
    profile_call("controller.token.rv5m", lambda: controllers.rv5m.update(sample), rows=1)
    profile_call("controller.token.rv10m", lambda: controllers.rv10m.update(sample), rows=1)


def update_token_iv_controllers(
    controllers: TokenControllers,
    iv: float | None,
    *,
    ts: float | None = None,
) -> None:
    """Ingest one IV sample into token IV rolling controllers."""
    if iv is None or ts is None:
        return
    sample = ControllerSample.iv(float(iv), ts)
    from .build_profiler import get_profiler, profile_call

    if get_profiler() is None:
        controllers.iv_ema9.update(sample)
        controllers.iv_ema20.update(sample)
        controllers.iv_ema50.update(sample)
        controllers.iv_ema100.update(sample)
        controllers.iv_ema200.update(sample)
        controllers.iv_ema300.update(sample)
        controllers.iv_zscore_1m.update(sample)
        controllers.iv_zscore_5m.update(sample)
        controllers.iv_zscore_15m.update(sample)
        controllers.iv_zscore_30m.update(sample)
        controllers.iv_session_rank.update(sample)
        controllers.iv_history.update(float(iv), ts=float(ts))
        return
    profile_call("controller.token.iv_ema9", lambda: controllers.iv_ema9.update(sample), rows=1)
    profile_call("controller.token.iv_ema20", lambda: controllers.iv_ema20.update(sample), rows=1)
    profile_call("controller.token.iv_ema50", lambda: controllers.iv_ema50.update(sample), rows=1)
    profile_call("controller.token.iv_ema100", lambda: controllers.iv_ema100.update(sample), rows=1)
    profile_call("controller.token.iv_ema200", lambda: controllers.iv_ema200.update(sample), rows=1)
    profile_call("controller.token.iv_ema300", lambda: controllers.iv_ema300.update(sample), rows=1)
    profile_call("controller.token.iv_zscore_1m", lambda: controllers.iv_zscore_1m.update(sample), rows=1)
    profile_call("controller.token.iv_zscore_5m", lambda: controllers.iv_zscore_5m.update(sample), rows=1)
    profile_call("controller.token.iv_zscore_15m", lambda: controllers.iv_zscore_15m.update(sample), rows=1)
    profile_call("controller.token.iv_zscore_30m", lambda: controllers.iv_zscore_30m.update(sample), rows=1)
    profile_call("controller.token.iv_session_rank", lambda: controllers.iv_session_rank.update(sample), rows=1)
    profile_call(
        "controller.iv_history",
        lambda: controllers.iv_history.update(float(iv), ts=float(ts)),
        rows=1,
    )


def emit_iv_history_features(controller: IvHistoryController) -> dict[str, float | None]:
    """Emit iv_change_* from token.iv_history controller.

    Ownership: Pipeline Owned in the Feature Registry; Master still emits until
    the IV history emitter is fully retired in a follow-up pass.
    """
    return {
        "iv_change_1m": controller.iv_change_pct_points(60.0),
        "iv_change_5m": controller.iv_change_pct_points(300.0),
        "iv_change_15m": controller.iv_change_pct_points(900.0),
        "iv_pct_change_1m": controller.iv_pct_change_for_lag(60.0),
    }


def opt_rv_ratio(controllers: TokenControllers) -> float | None:
    """opt_rv_5m / opt_rv_10m — ControllerDerived; NULL until both sources ready."""
    return emit_controller_derived_quotient(controllers.rv5m, controllers.rv10m)


@dataclass
class SpotControllers:
    """Session-wide spot RV controllers — not reset on per-token gaps."""

    ema9: EmaController = field(default_factory=lambda: EmaController(9))
    ema20: EmaController = field(default_factory=lambda: EmaController(20))
    ema50: EmaController = field(default_factory=lambda: EmaController(50))
    ema100: EmaController = field(default_factory=lambda: EmaController(100))
    ema200: EmaController = field(default_factory=lambda: EmaController(200))
    ema300: EmaController = field(default_factory=lambda: EmaController(300))
    rv5m: RvController = field(default_factory=lambda: RvController(30))
    rv10m: RvController = field(default_factory=lambda: RvController(60))
    momentum: Any = field(default=None, repr=False)
    hl: Any = field(default=None, repr=False)
    _last_stream_ts: float | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        from .controller_registry import ensure_architecture_registry

        ensure_architecture_registry()
        if self.momentum is None:
            from .spot_momentum_registry import SpotMomentumController

            self.momentum = SpotMomentumController()
        if self.hl is None:
            from .spot_hl_controllers import SpotHlControllers

            self.hl = SpotHlControllers()

    def update(
        self,
        spot: float | None,
        ts: float | None = None,
        *,
        grid_step_sec: float | None = None,
        index_tl: Any | None = None,
        grid_origin_ts: float | None = None,
    ) -> None:
        """Ingest one spot sample — at most once per timestamp (market-wide stream)."""
        from time import perf_counter

        from .spot_controllers_profiler import (
            record_duplicate_timestamp_return,
            record_full_update,
            record_invalid_spot_return,
            spot_update_profiler_active,
            time_section,
        )

        profile_paths = spot_update_profiler_active()
        t_entry = time_section() if profile_paths else None

        if spot is None or float(spot) <= 0:
            if profile_paths and t_entry is not None:
                record_invalid_spot_return(perf_counter() - t_entry)
            return
        if ts is not None and self._last_stream_ts is not None:
            if float(ts) == float(self._last_stream_ts):
                if profile_paths and t_entry is not None:
                    record_duplicate_timestamp_return(perf_counter() - t_entry)
                return

        t_full = time_section() if profile_paths else None
        ema_sec = rv_sec = 0.0
        momentum_sec: float | None = None
        spot_hl_sec: float | None = None

        if grid_step_sec is not None:
            self.momentum.grid_step_sec = float(grid_step_sec)
        sample = ControllerSample.spot(float(spot), ts)

        if profile_paths:
            t_ema = time_section()
            self.ema9.update(sample)
            self.ema20.update(sample)
            self.ema50.update(sample)
            self.ema100.update(sample)
            self.ema200.update(sample)
            self.ema300.update(sample)
            ema_sec = perf_counter() - t_ema
            t_rv = time_section()
            self.rv5m.update(sample)
            self.rv10m.update(sample)
            rv_sec = perf_counter() - t_rv
        else:
            self.ema9.update(sample)
            self.ema20.update(sample)
            self.ema50.update(sample)
            self.ema100.update(sample)
            self.ema200.update(sample)
            self.ema300.update(sample)
            self.rv5m.update(sample)
            self.rv10m.update(sample)

        if ts is not None:
            if profile_paths:
                t_mom = time_section()
                self.momentum.update(
                    spot=float(spot),
                    ts=float(ts),
                    ema9=self.ema9.value(),
                    ema20=self.ema20.value(),
                )
                momentum_sec = perf_counter() - t_mom
            else:
                self.momentum.update(
                    spot=float(spot),
                    ts=float(ts),
                    ema9=self.ema9.value(),
                    ema20=self.ema20.value(),
                )
            if (
                index_tl is not None
                and grid_step_sec is not None
                and grid_origin_ts is not None
            ):
                if profile_paths:
                    t_hl = time_section()
                    self.hl.update_bar(
                        index_tl=index_tl,
                        ts=float(ts),
                        close=float(spot),
                        grid_step_sec=float(grid_step_sec),
                        grid_origin_ts=float(grid_origin_ts),
                    )
                    spot_hl_sec = perf_counter() - t_hl
                else:
                    self.hl.update_bar(
                        index_tl=index_tl,
                        ts=float(ts),
                        close=float(spot),
                        grid_step_sec=float(grid_step_sec),
                        grid_origin_ts=float(grid_origin_ts),
                    )
            self._last_stream_ts = float(ts)

        if profile_paths and t_full is not None:
            record_full_update(
                total_sec=perf_counter() - t_full,
                ema_sec=ema_sec,
                rv_sec=rv_sec,
                momentum_sec=momentum_sec,
                spot_hl_sec=spot_hl_sec,
            )

    def reset_all(self, ts: float | None = None) -> None:
        self.ema9.reset(ts)
        self.ema20.reset(ts)
        self.ema50.reset(ts)
        self.ema100.reset(ts)
        self.ema200.reset(ts)
        self.ema300.reset(ts)
        self.rv5m.reset(ts)
        self.rv10m.reset(ts)
        self.momentum.reset(ts)
        self.hl.reset(ts)
        self._last_stream_ts = None

    def spot_rv_ratio(self) -> float | None:
        """spot_rv_5m / spot_rv_10m — ControllerDerived; NULL until both sources ready."""
        return emit_controller_derived_quotient(self.rv5m, self.rv10m)


def build_spot_rv_cache(
    index_tl: Any,
    timestamps: Sequence[float],
    *,
    grid_step_sec: float = IV_GRID_STEP_SEC,
    grid_origin_ts: float | None = None,
) -> dict[float, dict[str, float | None]]:
    """Precompute spot RV + momentum features for sorted session timestamps (parallel build path)."""
    from .spot_momentum_registry import emit_spot_momentum_registry_features

    spot_ctrl = SpotControllers()
    cache: dict[float, dict[str, float | None]] = {}
    if grid_origin_ts is not None:
        origin: float | None = float(grid_origin_ts)
    else:
        from .tick_coverage import spot_tick_bounds

        bounds = spot_tick_bounds(index_tl) if index_tl is not None else None
        origin = float(bounds[0]) if bounds else None
    for ts in sorted({float(t) for t in timestamps}):
        spot = index_tl.ltp_rupees_at(ts) if index_tl is not None else None
        spot_ctrl.update(
            spot,
            ts=ts,
            grid_step_sec=grid_step_sec,
            index_tl=index_tl,
            grid_origin_ts=origin,
        )
        entry = {
            "spot_ema9": emit_controller_value(spot_ctrl.ema9),
            "spot_ema20": emit_controller_value(spot_ctrl.ema20),
            "spot_ema50": emit_controller_value(spot_ctrl.ema50),
            "spot_ema100": emit_controller_value(spot_ctrl.ema100),
            "spot_ema200": emit_controller_value(spot_ctrl.ema200),
            "spot_ema300": emit_controller_value(spot_ctrl.ema300),
            "spot_rv_5m": emit_controller_value(spot_ctrl.rv5m),
            "spot_rv_10m": emit_controller_value(spot_ctrl.rv10m),
            "spot_rv_ratio": spot_ctrl.spot_rv_ratio(),
        }
        entry.update(
            emit_spot_momentum_registry_features(
                spot_ctrl,
                spot=spot,
                ts=ts,
            )
        )
        # HL / channel / weighted HL — required for token-parallel Stage 6.
        # Controllers are updated above; without emitting into the cache the
        # parallel path forces these features to NULL (enrich gets controllers=None).
        from .spot_hl_registry import (
            emit_spot_hl_composite_registry_features,
            emit_spot_hl_ratio_registry_features,
        )

        entry.update(
            emit_spot_hl_ratio_registry_features(spot_ctrl, ltp=None, active_features=None)
        )
        entry.update(
            emit_spot_hl_composite_registry_features(
                spot_ctrl, ltp=None, active_features=None
            )
        )
        guard_controller_derived_rv_features(entry)
        cache[ts] = entry
    return cache


def weighted_spot_ema_level_from_values(
    ema9: float | None,
    ema20: float | None,
    ema50: float | None,
    ema200: float | None,
) -> float | None:
    """Normalized weighted spot EMA level — weights 4/3/2/1 on EMA9/20/50/200."""
    if ema9 is None or ema20 is None or ema50 is None or ema200 is None:
        return None
    return (float(ema9) * 4.0 + float(ema20) * 3.0 + float(ema50) * 2.0 + float(ema200)) / 10.0


def weighted_spot_ema_level(controllers: SpotControllers) -> float | None:
    """Composite weighted spot EMA level — bottleneck warmup = 200."""
    deps = (controllers.ema9, controllers.ema20, controllers.ema50, controllers.ema200)

    def _blend() -> float | None:
        return weighted_spot_ema_level_from_values(
            controllers.ema9.value(),
            controllers.ema20.value(),
            controllers.ema50.value(),
            controllers.ema200.value(),
        )

    return emit_when_ready(deps, _blend)


def weighted_spot_ema_ratio_from_values(
    ema9: float | None,
    ema20: float | None,
    ema50: float | None,
    ema200: float | None,
    ltp: float,
) -> float | None:
    """Weighted spot EMA blend / ltp — weights 4/3/2/1 on EMA9/20/50/200."""
    if ltp <= 0:
        return None
    level = weighted_spot_ema_level_from_values(ema9, ema20, ema50, ema200)
    if level is None:
        return None
    return float(level) / float(ltp)


def weighted_spot_ema_ratio(
    controllers: SpotControllers,
    ltp: float,
) -> float | None:
    """Composite weighted spot EMA / ltp — bottleneck warmup = 200."""
    if ltp <= 0:
        return None
    level = weighted_spot_ema_level(controllers)
    if level is None:
        return None
    return float(level) / float(ltp)


def resolve_weighted_spot_ema_to_ltp_ratio(
    raw: Mapping[str, Any],
    *,
    ltp: float,
    spot_controllers: SpotControllers | None = None,
    spot_rv_cache: Mapping[float, Mapping[str, float | None]] | None = None,
    ts: float | None = None,
) -> float | None:
    """Resolve weighted spot EMA ratio from row, live controllers, or parallel cache."""
    existing = raw.get("weighted_spot_ema_to_ltp_ratio")
    if existing is not None:
        try:
            return float(existing)
        except (TypeError, ValueError):
            return None
    # Wave 3: prefer canonical level / ltp
    level = raw.get("weighted_spot_ema")
    if level is not None and ltp > 0:
        try:
            return float(level) / float(ltp)
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    if ltp <= 0:
        return None
    if spot_controllers is not None:
        return weighted_spot_ema_ratio(spot_controllers, ltp)
    if spot_rv_cache is not None and ts is not None:
        cached = spot_rv_cache.get(float(ts), {})
        return weighted_spot_ema_ratio_from_values(
            cached.get("spot_ema9"),
            cached.get("spot_ema20"),
            cached.get("spot_ema50"),
            cached.get("spot_ema200"),
            ltp,
        )
    return None


def weighted_ltp_ema_level(controllers: TokenControllers) -> float | None:
    """Composite weighted LTP EMA level — bottleneck warmup = 200."""
    deps = (controllers.ema9, controllers.ema20, controllers.ema50, controllers.ema200)

    def _blend() -> float | None:
        e9 = controllers.ema9.value()
        e20 = controllers.ema20.value()
        e50 = controllers.ema50.value()
        e200 = controllers.ema200.value()
        if e9 is None or e20 is None or e50 is None or e200 is None:
            return None
        return float(e9 * 4.0 + e20 * 3.0 + e50 * 2.0 + e200) / 10.0

    return emit_when_ready(deps, _blend)


def weighted_ltp_ema_ratio(
    controllers: TokenControllers,
    ltp: float,
) -> float | None:
    """Composite weighted LTP EMA / ltp — bottleneck warmup = 200."""
    if ltp <= 0:
        return None
    level = weighted_ltp_ema_level(controllers)
    if level is None:
        return None
    return float(level) / float(ltp)


# Bootstrap Controller Registry on import (idempotent; no behavioural change).
from .controller_registry import ensure_architecture_registry as _ensure_controller_registry

_ensure_controller_registry()
