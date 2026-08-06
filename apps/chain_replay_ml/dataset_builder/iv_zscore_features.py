"""IV z-score composites — weighted IV z-scores crossed with spot EMA / delta."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .rolling_controllers import (
    SpotControllers,
    resolve_weighted_spot_ema_to_ltp_ratio,
)

IV_ZSCORE_BASE_FEATURES: tuple[str, ...] = (
    "iv_zscore_1m",
    "iv_zscore_5m",
    "iv_zscore_15m",
)

IV_ZSCORE_COMPOSITE_FEATURES: tuple[str, ...] = (
    "weighted_iv_zscore_x_weighted_spot_ema_to_ltp_ratio",
    "weighted_spot_ema_to_ltp_ratio_x_delta",
    "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_1m",
    "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_5m",
    "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m",
    "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_1m_x_delta",
    "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_5m_x_delta",
    "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m_x_delta",
)

IV_ZSCORE_FEATURES: frozenset[str] = frozenset(
    (*IV_ZSCORE_BASE_FEATURES, *IV_ZSCORE_COMPOSITE_FEATURES)
)


def active_iv_zscore_features(active: Iterable[str] | None) -> frozenset[str]:
    if active is None:
        return IV_ZSCORE_FEATURES
    return frozenset(str(f) for f in active if str(f) in IV_ZSCORE_FEATURES)


def needs_iv_zscore_composites(active: Iterable[str] | None) -> bool:
    if active is None:
        return True
    return any(str(f) in IV_ZSCORE_COMPOSITE_FEATURES for f in active)


def _mul(*vals: float | None) -> float | None:
    out: float | None = 1.0
    for val in vals:
        if val is None:
            return None
        out = float(out) * float(val)  # type: ignore[operator]
    return out


def weighted_iv_zscore(
    z1: float | None,
    z5: float | None,
    z15: float | None,
) -> float | None:
    if z1 is None or z5 is None or z15 is None:
        return None
    return (float(z1) * 3.0 + float(z5) * 2.0 + float(z15)) / 6.0


def enrich_iv_zscore_features(
    raw: dict[str, Any],
    *,
    active_features: frozenset[str] | None = None,
    spot_controllers: SpotControllers | None = None,
    spot_rv_cache: Mapping[float, Mapping[str, float | None]] | None = None,
    ts: float | None = None,
) -> dict[str, Any]:
    wanted = active_iv_zscore_features(active_features) & frozenset(IV_ZSCORE_COMPOSITE_FEATURES)
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
    delta = out.get("delta")
    try:
        delta_f: float | None = float(delta) if delta is not None else None
    except (TypeError, ValueError):
        delta_f = None

    z1 = out.get("iv_zscore_1m")
    z5 = out.get("iv_zscore_5m")
    z15 = out.get("iv_zscore_15m")
    try:
        z1_f = float(z1) if z1 is not None else None
        z5_f = float(z5) if z5 is not None else None
        z15_f = float(z15) if z15 is not None else None
    except (TypeError, ValueError):
        z1_f = z5_f = z15_f = None

    w_iv = weighted_iv_zscore(z1_f, z5_f, z15_f)

    if "weighted_iv_zscore_x_weighted_spot_ema_to_ltp_ratio" in wanted:
        out["weighted_iv_zscore_x_weighted_spot_ema_to_ltp_ratio"] = _mul(w_iv, w_ratio)
    if "weighted_spot_ema_to_ltp_ratio_x_delta" in wanted:
        out["weighted_spot_ema_to_ltp_ratio_x_delta"] = _mul(w_ratio, delta_f)
    if "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_1m" in wanted:
        out["weighted_spot_ema_to_ltp_ratio_x_iv_zscore_1m"] = _mul(w_ratio, z1_f)
    if "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_5m" in wanted:
        out["weighted_spot_ema_to_ltp_ratio_x_iv_zscore_5m"] = _mul(w_ratio, z5_f)
    if "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m" in wanted:
        out["weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m"] = _mul(w_ratio, z15_f)
    if "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_1m_x_delta" in wanted:
        out["weighted_spot_ema_to_ltp_ratio_x_iv_zscore_1m_x_delta"] = _mul(w_ratio, z1_f, delta_f)
    if "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_5m_x_delta" in wanted:
        out["weighted_spot_ema_to_ltp_ratio_x_iv_zscore_5m_x_delta"] = _mul(w_ratio, z5_f, delta_f)
    if "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m_x_delta" in wanted:
        out["weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m_x_delta"] = _mul(w_ratio, z15_f, delta_f)

    return out
