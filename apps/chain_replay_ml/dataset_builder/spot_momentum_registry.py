"""Spot momentum registry features — EMA crossover state from SpotControllers.

Wave 6: percentage packaging (spot_vs_ema20_pct, ema_spread_pct,
ema_spread_vs_spot_pct) is Interaction / Pipeline Owned — not emitted here.
Canonical inputs remain: spot_ema9, spot_ema20 (from spot.ema*), plus crossover
event state below.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from .rolling_controllers import SpotControllers

SPOT_MOMENTUM_REGISTRY_FEATURES: frozenset[str] = frozenset({
    "ema9_slope",
    "ema9_gt_ema20",
    "time_since_cross_min",
    "cross_age_decay",
    "price_dist_from_cross_pct",
})

_DEFAULT_TIME_SINCE_CROSS_MIN = 60.0


def needs_spot_momentum_registry(active: Iterable[str] | None) -> bool:
    if active is None:
        return True
    return any(str(f) in SPOT_MOMENTUM_REGISTRY_FEATURES for f in active)


def active_spot_momentum_registry_features(active: Iterable[str] | None) -> frozenset[str]:
    if active is None:
        return SPOT_MOMENTUM_REGISTRY_FEATURES
    return frozenset(str(f) for f in active if str(f) in SPOT_MOMENTUM_REGISTRY_FEATURES)


@dataclass
class SpotMomentumController:
    """Crossover + EMA9 lag state for spot.momentum registry features."""

    grid_step_sec: float = 3.0
    _history: list[tuple[float, float, float | None, float | None]] = field(default_factory=list)
    _latest_cross_ts: float | None = None
    _latest_cross_price: float | None = None
    _prev_ema9_above: bool | None = None

    @property
    def bars_per_min(self) -> int:
        return max(1, int(round(60.0 / max(float(self.grid_step_sec), 1.0))))

    def reset(self, ts: float | None = None) -> None:
        self._history.clear()
        self._latest_cross_ts = None
        self._latest_cross_price = None
        self._prev_ema9_above = None

    def update(
        self,
        *,
        spot: float,
        ts: float,
        ema9: float | None,
        ema20: float | None,
    ) -> None:
        spot_f = float(spot)
        ts_f = float(ts)
        if self._latest_cross_ts is None:
            self._latest_cross_ts = ts_f
            self._latest_cross_price = spot_f
        self._history.append((ts_f, spot_f, ema9, ema20))
        if ema9 is not None and ema20 is not None:
            above = float(ema9) > float(ema20)
            if self._prev_ema9_above is not None and above != self._prev_ema9_above:
                self._latest_cross_ts = ts_f
                self._latest_cross_price = spot_f
            self._prev_ema9_above = above

    def _ema9_1m_ago(self) -> float | None:
        if not self._history:
            return None
        lag_idx = max(0, len(self._history) - 1 - self.bars_per_min)
        return self._history[lag_idx][2]

    def emit(self, *, spot: float | None, ts: float | None) -> dict[str, float | None]:
        spot_f = float(spot) if spot is not None and float(spot) > 0 else None
        ema9 = self._history[-1][2] if self._history else None
        ema20 = self._history[-1][3] if self._history else None

        out: dict[str, float | None] = {
            "ema9_slope": None,
            "ema9_gt_ema20": 0.0,
            "time_since_cross_min": _DEFAULT_TIME_SINCE_CROSS_MIN,
            "cross_age_decay": float(math.exp(-_DEFAULT_TIME_SINCE_CROSS_MIN / 30.0)),
            "price_dist_from_cross_pct": 0.0,
        }

        if ema9 is None or ema20 is None:
            return out

        ema9_f = float(ema9)
        ema20_f = float(ema20)
        out["ema9_gt_ema20"] = 1.0 if ema9_f > ema20_f else 0.0

        ema9_1m = self._ema9_1m_ago()
        if ema9_1m is not None and float(ema9_1m) > 0:
            out["ema9_slope"] = float(100.0 * (ema9_f - float(ema9_1m)) / float(ema9_1m))

        if self._latest_cross_ts is not None and ts is not None:
            mins = float(max(0.0, (float(ts) - float(self._latest_cross_ts)) / 60.0))
            out["time_since_cross_min"] = mins
            out["cross_age_decay"] = float(math.exp(-mins / 30.0))
            if self._latest_cross_price and float(self._latest_cross_price) > 0 and spot_f is not None:
                cross_price = float(self._latest_cross_price)
                out["price_dist_from_cross_pct"] = float(
                    100.0 * (spot_f - cross_price) / cross_price,
                )

        return out


def emit_spot_momentum_registry_features(
    spot_controllers: SpotControllers,
    *,
    spot: float | None,
    ts: float | None,
    active_features: frozenset[str] | None = None,
) -> dict[str, float | None]:
    wanted = active_spot_momentum_registry_features(active_features)
    if not wanted:
        return {}
    emitted = spot_controllers.momentum.emit(spot=spot, ts=ts)
    return {name: emitted.get(name) for name in wanted}


def enrich_spot_momentum_registry_features(
    raw: dict[str, Any],
    *,
    ts: float,
    spot_controllers: SpotControllers | None = None,
    spot_rv_cache: Mapping[float, Mapping[str, float | None]] | None = None,
    active_features: frozenset[str] | None = None,
) -> dict[str, Any]:
    wanted = active_spot_momentum_registry_features(active_features)
    if not wanted:
        return raw

    out = dict(raw)
    spot = out.get("spot")
    if spot_controllers is not None:
        emitted = emit_spot_momentum_registry_features(
            spot_controllers,
            spot=spot,
            ts=ts,
            active_features=wanted,
        )
    elif spot_rv_cache is not None:
        cached = dict(spot_rv_cache.get(float(ts), {}))
        emitted = {name: cached.get(name) for name in wanted}
    else:
        emitted = {}

    for name, val in emitted.items():
        out[name] = val
    return out
