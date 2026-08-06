"""IV–RV spread levels — current_iv minus spot realized vol (Wave B).

Canonical Computed Base levels. Lag / difference / rolling / EMA → Pipeline.
Units: both sides expressed in percent scale (IV decimal ×100 when needed;
spot_rv is already rolling std of per-sample % returns).
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

IV_RV_SPREAD_FEATURES: tuple[str, ...] = (
    "iv_rv_spread_5m",
    "iv_rv_spread_10m",
)
IV_RV_SPREAD_FEATURE_SET: frozenset[str] = frozenset(IV_RV_SPREAD_FEATURES)

_RV_BY_SPREAD: dict[str, str] = {
    "iv_rv_spread_5m": "spot_rv_5m",
    "iv_rv_spread_10m": "spot_rv_10m",
}


def active_iv_rv_spread_features(active: Iterable[str] | None) -> frozenset[str]:
    if active is None:
        return IV_RV_SPREAD_FEATURE_SET
    return frozenset(str(f) for f in active if str(f) in IV_RV_SPREAD_FEATURE_SET)


def needs_iv_rv_spread(active: Iterable[str] | None) -> bool:
    if active is None:
        return True
    return any(str(f) in IV_RV_SPREAD_FEATURE_SET for f in active)


def iv_as_percent(iv: float) -> float:
    """Normalize IV to percent. Values > 3 treated as already percent (iv_vs_atm convention)."""
    v = float(iv)
    return v if v > 3.0 else v * 100.0


def compute_iv_rv_spread(iv: float | None, spot_rv: float | None) -> float | None:
    if iv is None or spot_rv is None:
        return None
    try:
        return float(iv_as_percent(float(iv)) - float(spot_rv))
    except (TypeError, ValueError):
        return None


def _resolve_spot_rv(
    *,
    name: str,
    raw: Mapping[str, Any],
    ts: float | None,
    spot_controllers: Any | None,
    spot_rv_cache: Mapping[float, Mapping[str, float | None]] | None,
) -> float | None:
    existing = raw.get(name)
    if existing is not None:
        try:
            return float(existing)
        except (TypeError, ValueError):
            pass
    if spot_rv_cache is not None and ts is not None:
        cached = spot_rv_cache.get(float(ts), {})
        val = cached.get(name)
        if val is not None:
            return float(val)
    if spot_controllers is not None:
        from .rolling_controllers import emit_controller_value

        ctrl = spot_controllers.rv5m if name == "spot_rv_5m" else spot_controllers.rv10m
        return emit_controller_value(ctrl)
    return None


def enrich_iv_rv_spread_features(
    raw: dict[str, Any],
    *,
    ts: float | None = None,
    spot_controllers: Any | None = None,
    spot_rv_cache: Mapping[float, Mapping[str, float | None]] | None = None,
    active_features: frozenset[str] | None = None,
) -> dict[str, Any]:
    wanted = active_iv_rv_spread_features(active_features)
    if not wanted:
        return raw
    out = dict(raw)
    iv = out.get("iv")
    if iv is None:
        iv = out.get("current_iv")
    for spread_name in IV_RV_SPREAD_FEATURES:
        if spread_name not in wanted:
            continue
        rv_name = _RV_BY_SPREAD[spread_name]
        rv = _resolve_spot_rv(
            name=rv_name,
            raw=out,
            ts=ts,
            spot_controllers=spot_controllers,
            spot_rv_cache=spot_rv_cache,
        )
        out[spread_name] = compute_iv_rv_spread(
            float(iv) if iv is not None else None,
            rv,
        )
    return out
