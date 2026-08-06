"""Advanced cross features — weighted spot EMA ratio × moneyness / delta."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .iv_zscore_features import _mul
from .rolling_controllers import SpotControllers, resolve_weighted_spot_ema_to_ltp_ratio

ADVANCED_COMPOSITE_FEATURES: tuple[str, ...] = (
    "weighted_spot_ema_to_ltp_ratio_x_moneyness",
    "weighted_spot_ema_to_ltp_ratio_x_moneyness_x_delta",
)


def needs_advanced_composites(active: Iterable[str] | None) -> bool:
    if active is None:
        return False
    return any(str(f) in ADVANCED_COMPOSITE_FEATURES for f in active)


def enrich_advanced_composite_features(
    raw: dict[str, Any],
    *,
    active_features: frozenset[str] | None = None,
    spot_controllers: SpotControllers | None = None,
    spot_rv_cache: Mapping[float, Mapping[str, float | None]] | None = None,
    ts: float | None = None,
) -> dict[str, Any]:
    wanted = {str(f) for f in (active_features or ()) if str(f) in ADVANCED_COMPOSITE_FEATURES}
    if not wanted:
        return raw

    out = dict(raw)
    ltp = out.get("ltp")
    ltp_f = float(ltp) if ltp is not None and float(ltp) > 0 else 0.0
    w_ratio = resolve_weighted_spot_ema_to_ltp_ratio(
        out,
        ltp=ltp_f,
        spot_controllers=spot_controllers,
        spot_rv_cache=spot_rv_cache,
        ts=ts,
    )

    moneyness = out.get("moneyness")
    try:
        moneyness_f: float | None = float(moneyness) if moneyness is not None else None
    except (TypeError, ValueError):
        moneyness_f = None

    delta = out.get("delta")
    try:
        delta_f: float | None = float(delta) if delta is not None else None
    except (TypeError, ValueError):
        delta_f = None

    if "weighted_spot_ema_to_ltp_ratio_x_moneyness" in wanted:
        out["weighted_spot_ema_to_ltp_ratio_x_moneyness"] = _mul(w_ratio, moneyness_f)
    if "weighted_spot_ema_to_ltp_ratio_x_moneyness_x_delta" in wanted:
        out["weighted_spot_ema_to_ltp_ratio_x_moneyness_x_delta"] = _mul(
            w_ratio, moneyness_f, delta_f,
        )

    return out
