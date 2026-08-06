"""Spot HL registry features — canonical HL EMA + weighted blend + channel width.

Wave 2: spot_high_ema* / spot_low_ema* Computed Base levels.
Wave 3: weighted_spot_high/low/close_ema Computed Base; packaging → Interaction.
Wave 5: spot_ema{N}_channel_width Computed Base; ltp÷width → Interaction.
"""

from __future__ import annotations

from typing import Any, Iterable

from .rolling_controllers import (
    SpotControllers,
    emit_controller_value,
    emit_when_ready,
)
from .spot_hl_controllers import SPOT_HL_CLOSE_PERIODS, SPOT_HL_PERIODS

CHANNEL_EPS = 1e-6
WEIGHTED_HL_PERIODS: tuple[int, ...] = SPOT_HL_CLOSE_PERIODS

WEIGHTED_SPOT_HIGH_EMA = "weighted_spot_high_ema"
WEIGHTED_SPOT_LOW_EMA = "weighted_spot_low_ema"
WEIGHTED_SPOT_CLOSE_EMA = "weighted_spot_close_ema"

# Wave 2: canonical HL levels (name kept for call-site compatibility).
SPOT_HL_RATIO_REGISTRY_FEATURES: frozenset[str] = frozenset(
    [f"spot_high_ema{p}" for p in SPOT_HL_PERIODS]
    + [f"spot_low_ema{p}" for p in SPOT_HL_PERIODS]
)

# Wave 5: canonical channel width levels (high − low); ltp÷width → Interaction.
SPOT_HL_CHANNEL_WIDTH_REGISTRY_FEATURES: frozenset[str] = frozenset(
    f"spot_ema{p}_channel_width" for p in SPOT_HL_PERIODS
)

# Wave 3: weighted blend levels (HL close ≠ spot.ema weighted_spot_ema).
SPOT_HL_WEIGHTED_COMPOSITE_REGISTRY_FEATURES: frozenset[str] = frozenset(
    {
        WEIGHTED_SPOT_HIGH_EMA,
        WEIGHTED_SPOT_LOW_EMA,
        WEIGHTED_SPOT_CLOSE_EMA,
    }
)

SPOT_HL_COMPOSITE_REGISTRY_FEATURES: frozenset[str] = (
    SPOT_HL_CHANNEL_WIDTH_REGISTRY_FEATURES | SPOT_HL_WEIGHTED_COMPOSITE_REGISTRY_FEATURES
)

SPOT_HL_CONTROLLER_REGISTRY_FEATURES: frozenset[str] = (
    SPOT_HL_RATIO_REGISTRY_FEATURES | SPOT_HL_COMPOSITE_REGISTRY_FEATURES
)


def needs_spot_hl_ratio_registry(active: Iterable[str] | None) -> bool:
    if active is None:
        return True
    return any(str(f) in SPOT_HL_RATIO_REGISTRY_FEATURES for f in active)


def needs_spot_hl_composite_registry(active: Iterable[str] | None) -> bool:
    if active is None:
        return True
    return any(str(f) in SPOT_HL_COMPOSITE_REGISTRY_FEATURES for f in active)


def active_spot_hl_ratio_registry_features(active: Iterable[str] | None) -> frozenset[str]:
    if active is None:
        return SPOT_HL_RATIO_REGISTRY_FEATURES
    return frozenset(str(f) for f in active if str(f) in SPOT_HL_RATIO_REGISTRY_FEATURES)


def active_spot_hl_composite_registry_features(active: Iterable[str] | None) -> frozenset[str]:
    if active is None:
        return SPOT_HL_COMPOSITE_REGISTRY_FEATURES
    return frozenset(str(f) for f in active if str(f) in SPOT_HL_COMPOSITE_REGISTRY_FEATURES)


def _weighted_side_blend_sum(side) -> float | None:
    """Raw weighted sum (×4/3/2/1) — divide by 10 for normalized level."""
    deps = [side.controller(period) for period in WEIGHTED_HL_PERIODS]

    def _blend() -> float | None:
        v20, v50, v200, v300 = (ctrl.value() for ctrl in deps)
        return float(v20) * 4.0 + float(v50) * 3.0 + float(v200) * 2.0 + float(v300)

    return emit_when_ready(deps, _blend)


def _weighted_side_level(side) -> float | None:
    blend = _weighted_side_blend_sum(side)
    if blend is None:
        return None
    return float(blend) / 10.0


def _channel_width_from_controllers(hl, period: int) -> float | None:
    """Signed high−low EMA channel width. NullUntilReady until both sides warm."""
    high_val = emit_controller_value(hl.high.controller(period))
    low_val = emit_controller_value(hl.low.controller(period))
    if high_val is None or low_val is None:
        return None
    return float(high_val) - float(low_val)


def emit_spot_hl_ratio_registry_features(
    spot_controllers: SpotControllers,
    *,
    ltp: float | None = None,
    active_features: frozenset[str] | None = None,
) -> dict[str, float | None]:
    """Emit canonical spot high/low EMA levels (Wave 2). ``ltp`` unused."""
    del ltp  # levels do not require LTP
    wanted = active_spot_hl_ratio_registry_features(active_features)
    if not wanted:
        return {}

    hl = spot_controllers.hl
    out: dict[str, float | None] = {}
    for period in SPOT_HL_PERIODS:
        feat_h = f"spot_high_ema{period}"
        if feat_h in wanted:
            out[feat_h] = emit_controller_value(hl.high.controller(period))
        feat_l = f"spot_low_ema{period}"
        if feat_l in wanted:
            out[feat_l] = emit_controller_value(hl.low.controller(period))
    return out


def emit_spot_hl_composite_registry_features(
    spot_controllers: SpotControllers,
    *,
    ltp: float | None = None,
    active_features: frozenset[str] | None = None,
) -> dict[str, float | None]:
    """Emit channel width + weighted HL levels. ``ltp`` unused (Wave 5)."""
    del ltp
    wanted = active_spot_hl_composite_registry_features(active_features)
    if not wanted:
        return {}

    hl = spot_controllers.hl
    out: dict[str, float | None] = {}

    # Wave 5: canonical channel width (no LTP).
    channel_wanted = wanted & SPOT_HL_CHANNEL_WIDTH_REGISTRY_FEATURES
    if channel_wanted:
        for period in SPOT_HL_PERIODS:
            feat = f"spot_ema{period}_channel_width"
            if feat in channel_wanted:
                out[feat] = _channel_width_from_controllers(hl, period)

    # Wave 3: canonical weighted HL levels (no LTP required).
    if WEIGHTED_SPOT_HIGH_EMA in wanted:
        out[WEIGHTED_SPOT_HIGH_EMA] = _weighted_side_level(hl.high)
    if WEIGHTED_SPOT_LOW_EMA in wanted:
        out[WEIGHTED_SPOT_LOW_EMA] = _weighted_side_level(hl.low)
    if WEIGHTED_SPOT_CLOSE_EMA in wanted:
        out[WEIGHTED_SPOT_CLOSE_EMA] = _weighted_side_level(hl.close)

    return out


def enrich_spot_hl_ratio_registry_features(
    raw: dict[str, Any],
    *,
    spot_controllers: SpotControllers | None = None,
    spot_rv_cache: dict[float, dict[str, float | None]] | None = None,
    ts: float | None = None,
    active_features: frozenset[str] | None = None,
) -> dict[str, Any]:
    wanted = active_spot_hl_ratio_registry_features(active_features)
    if not wanted:
        return raw

    out = dict(raw)
    if spot_controllers is not None:
        emitted = emit_spot_hl_ratio_registry_features(
            spot_controllers,
            ltp=out.get("ltp"),
            active_features=wanted,
        )
    elif spot_rv_cache is not None and ts is not None:
        cached = spot_rv_cache.get(float(ts), {})
        emitted = {name: cached.get(name) for name in wanted}
    else:
        emitted = {name: None for name in wanted}

    for name, val in emitted.items():
        out[name] = val
    return out


def enrich_spot_hl_composite_registry_features(
    raw: dict[str, Any],
    *,
    spot_controllers: SpotControllers | None = None,
    spot_rv_cache: dict[float, dict[str, float | None]] | None = None,
    ts: float | None = None,
    active_features: frozenset[str] | None = None,
) -> dict[str, Any]:
    wanted = active_spot_hl_composite_registry_features(active_features)
    if not wanted:
        return raw

    out = dict(raw)
    if spot_controllers is not None:
        emitted = emit_spot_hl_composite_registry_features(
            spot_controllers,
            ltp=out.get("ltp"),
            active_features=wanted,
        )
    elif spot_rv_cache is not None and ts is not None:
        cached = spot_rv_cache.get(float(ts), {})
        emitted = {name: cached.get(name) for name in wanted}
    else:
        emitted = {name: None for name in wanted}

    for name, val in emitted.items():
        out[name] = val
    return out
