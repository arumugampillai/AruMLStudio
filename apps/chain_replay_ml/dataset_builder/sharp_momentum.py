"""Sharp Momentum — spot impulse scores/counts with time-normalized decay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from chain_replay_ml.ticks import TickTimeline

from .day_context import DayContext
from .extended_features import OptionFeatureState
from .feature_grid_policy import resolve_feature_grid_step_sec
from .rolling_controllers import (
    SpotControllers,
    weighted_ltp_ema_level,
    weighted_spot_ema_level,
    weighted_spot_ema_level_from_values,
)

REF_STEP_SEC = 3.0
# Kept for docs / packaging Interaction (ltp / (count + COUNT_EPS)).
COUNT_EPS = 1e-6

SHARP_MOMENTUM_HORIZONS: tuple[str, ...] = ("1m", "3m", "5m", "10m")

DECAY_AT_3S: dict[str, float] = {
    "1m": 0.900,
    "3m": 0.962,
    "5m": 0.977,
    "10m": 0.989,
}

# Wave 3: canonical weighted levels (÷ltp → Interaction)
WEIGHTED_EMA_FEATURES: tuple[str, ...] = (
    "weighted_spot_ema",
    "weighted_ltp_ema",
)

# Wave 4: canonical sharp momentum levels (packaging → Interaction)
SPOT_SCORE_FEATURES: tuple[str, ...] = tuple(
    f"spot_{side}_score_{h}"
    for h in SHARP_MOMENTUM_HORIZONS
    for side in ("up", "down")
)

SPOT_COUNT_FEATURES: tuple[str, ...] = tuple(
    f"spot_{side}_sample_count_{h}"
    for h in SHARP_MOMENTUM_HORIZONS
    for side in ("up", "down")
)

SHARP_MOMENTUM_FEATURES: frozenset[str] = frozenset(
    (*WEIGHTED_EMA_FEATURES, *SPOT_SCORE_FEATURES, *SPOT_COUNT_FEATURES)
)


def decay_factor(decay_3s: float, dt_sec: float) -> float:
    if dt_sec <= 0:
        return 1.0
    return float(decay_3s) ** (float(dt_sec) / REF_STEP_SEC)


def _zero_horizon_state() -> dict[str, float]:
    return {h: 0.0 for h in SHARP_MOMENTUM_HORIZONS}


@dataclass
class SpotMomentumSnapshot:
    up_score: dict[str, float] = field(default_factory=_zero_horizon_state)
    down_score: dict[str, float] = field(default_factory=_zero_horizon_state)
    up_count: dict[str, float] = field(default_factory=_zero_horizon_state)
    down_count: dict[str, float] = field(default_factory=_zero_horizon_state)

    def copy(self) -> SpotMomentumSnapshot:
        return SpotMomentumSnapshot(
            up_score=dict(self.up_score),
            down_score=dict(self.down_score),
            up_count=dict(self.up_count),
            down_count=dict(self.down_count),
        )


def _apply_decay(snapshot: SpotMomentumSnapshot, dt_sec: float) -> None:
    for h in SHARP_MOMENTUM_HORIZONS:
        d = decay_factor(DECAY_AT_3S[h], dt_sec)
        snapshot.up_score[h] *= d
        snapshot.down_score[h] *= d
        snapshot.up_count[h] *= d
        snapshot.down_count[h] *= d


def _apply_spot_change(snapshot: SpotMomentumSnapshot, spot_change: float) -> None:
    if spot_change > 0:
        for h in SHARP_MOMENTUM_HORIZONS:
            snapshot.up_score[h] += spot_change
            snapshot.up_count[h] += 1.0
    elif spot_change < 0:
        mag = abs(spot_change)
        for h in SHARP_MOMENTUM_HORIZONS:
            snapshot.down_score[h] += mag
            snapshot.down_count[h] += 1.0


def active_sharp_momentum_features(active: Iterable[str] | None) -> frozenset[str]:
    if active is None:
        return SHARP_MOMENTUM_FEATURES
    wanted = {str(f) for f in active if f in SHARP_MOMENTUM_FEATURES}
    return frozenset(wanted)


def needs_sharp_momentum(active: Iterable[str] | None) -> bool:
    return bool(active_sharp_momentum_features(active))


def features_from_snapshot(
    snapshot: SpotMomentumSnapshot,
    ltp: float | None = None,
    *,
    wanted: frozenset[str] | None = None,
) -> dict[str, float]:
    """Emit canonical score/count levels (Wave 4). ``ltp`` unused (kept for call-site compat)."""
    del ltp
    want = wanted or SHARP_MOMENTUM_FEATURES
    out: dict[str, float] = {}
    for h in SHARP_MOMENTUM_HORIZONS:
        up_s = f"spot_up_score_{h}"
        dn_s = f"spot_down_score_{h}"
        up_c = f"spot_up_sample_count_{h}"
        dn_c = f"spot_down_sample_count_{h}"
        if up_s in want:
            out[up_s] = float(snapshot.up_score[h])
        if dn_s in want:
            out[dn_s] = float(snapshot.down_score[h])
        if up_c in want:
            out[up_c] = float(snapshot.up_count[h])
        if dn_c in want:
            out[dn_c] = float(snapshot.down_count[h])
    return out


def _reset_spot_momentum_cache(ctx: DayContext, step_sec: int) -> None:
    ctx.spot_momentum_by_ts = {}
    ctx.spot_momentum_run_state = SpotMomentumSnapshot()
    ctx.spot_momentum_prev_ts = None
    ctx.spot_momentum_prev_spot = None
    ctx.spot_momentum_ready_through = None
    ctx.spot_momentum_step_sec = int(step_sec)
    ctx._spot_momentum_grid_ts = None


def _grid_timestamps_through(
    ctx: DayContext,
    *,
    through_ts: float,
    step_sec: int,
    max_horizon_sec: int,
) -> list[float]:
    from .tick_coverage import list_clipped_grid_timestamps

    grid = list_clipped_grid_timestamps(ctx, step_sec=step_sec, max_horizon_sec=max_horizon_sec)
    return [t for t in grid if t <= float(through_ts) + 0.001]


def ensure_spot_momentum_cache(
    ctx: DayContext,
    *,
    through_ts: float,
    step_sec: int | None = None,
    max_horizon_sec: int = 0,
) -> None:
    """Advance shared spot momentum state through ``through_ts`` on the feature grid."""
    step = resolve_feature_grid_step_sec(ctx=ctx, fallback=step_sec)
    if int(getattr(ctx, "spot_momentum_step_sec", 0) or 0) != step:
        _reset_spot_momentum_cache(ctx, step)

    ready = getattr(ctx, "spot_momentum_ready_through", None)
    if ready is not None and float(through_ts) <= float(ready) + 0.001:
        return

    grid_ts = _grid_timestamps_through(
        ctx, through_ts=float(through_ts), step_sec=step, max_horizon_sec=max_horizon_sec,
    )
    if not grid_ts:
        return

    state: SpotMomentumSnapshot = getattr(ctx, "spot_momentum_run_state", None) or SpotMomentumSnapshot()
    prev_ts: float | None = getattr(ctx, "spot_momentum_prev_ts", None)
    prev_spot: float | None = getattr(ctx, "spot_momentum_prev_spot", None)
    cache: dict[float, SpotMomentumSnapshot] = getattr(ctx, "spot_momentum_by_ts", None) or {}

    for ts in grid_ts:
        if ts in cache:
            snap = cache[ts]
            state = snap.copy()
            prev_ts = ts
            spot_at = ctx.index_tl.ltp_rupees_at(ts)
            if spot_at is not None:
                prev_spot = float(spot_at)
            continue

        spot = ctx.index_tl.ltp_rupees_at(ts)
        if prev_ts is None:
            cache[ts] = SpotMomentumSnapshot()
            if spot is not None and float(spot) > 0:
                prev_ts = ts
                prev_spot = float(spot)
            continue

        dt = float(ts) - float(prev_ts)
        if dt > 0:
            _apply_decay(state, dt)
            if spot is not None and prev_spot is not None:
                _apply_spot_change(state, float(spot) - float(prev_spot))

        cache[ts] = state.copy()
        prev_ts = ts
        if spot is not None and float(spot) > 0:
            prev_spot = float(spot)

    ctx.spot_momentum_by_ts = cache
    ctx._spot_momentum_grid_ts = sorted(cache.keys())
    ctx.spot_momentum_run_state = state
    ctx.spot_momentum_prev_ts = prev_ts
    ctx.spot_momentum_prev_spot = prev_spot
    ctx.spot_momentum_ready_through = float(through_ts)


def spot_momentum_snapshot_at(ctx: DayContext, ts: float) -> SpotMomentumSnapshot:
    cache: dict[float, SpotMomentumSnapshot] = getattr(ctx, "spot_momentum_by_ts", None) or {}
    if not cache:
        return SpotMomentumSnapshot()
    key = float(ts)
    if key in cache:
        return cache[key]
    # Nearest prior grid point (rows should be on-grid; tolerate tiny float drift).
    grid = getattr(ctx, "_spot_momentum_grid_ts", None)
    if grid is None:
        grid = sorted(cache.keys())
        ctx._spot_momentum_grid_ts = grid
    import bisect

    idx = bisect.bisect_right(grid, key + 0.001) - 1
    if idx < 0:
        return SpotMomentumSnapshot()
    return cache[grid[idx]]


def enrich_sharp_momentum_features(
    raw: dict[str, Any],
    *,
    ts: float,
    ctx: DayContext,
    opt_state: OptionFeatureState | None,
    option_timeline: TickTimeline | None,
    open_ts: float | None,
    close_ts: float | None,
    active_features: frozenset[str] | None = None,
    feature_grid_step_sec: float | None = None,
    gap_max_sec: float | None = None,
    spot_controllers: SpotControllers | None = None,
    spot_rv_cache: dict[float, dict[str, float | None]] | None = None,
) -> dict[str, Any]:
    wanted = active_sharp_momentum_features(active_features)
    if not wanted:
        return raw

    out = dict(raw)

    if "weighted_spot_ema" in wanted:
        level: float | None = None
        if spot_controllers is not None:
            level = weighted_spot_ema_level(spot_controllers)
        elif spot_rv_cache is not None:
            cached = spot_rv_cache.get(float(ts), {})
            level = weighted_spot_ema_level_from_values(
                cached.get("spot_ema9"),
                cached.get("spot_ema20"),
                cached.get("spot_ema50"),
                cached.get("spot_ema200"),
            )
        out["weighted_spot_ema"] = level

    if "weighted_ltp_ema" in wanted:
        if opt_state is not None:
            out["weighted_ltp_ema"] = weighted_ltp_ema_level(opt_state.controllers)
        else:
            out["weighted_ltp_ema"] = None

    level_wanted = wanted & frozenset((*SPOT_SCORE_FEATURES, *SPOT_COUNT_FEATURES))
    if level_wanted:
        snap = spot_momentum_snapshot_at(ctx, ts)
        out.update(features_from_snapshot(snap, wanted=level_wanted))

    return out
