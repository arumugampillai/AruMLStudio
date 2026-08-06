"""IV EMA features — canonical IV EMA levels for Computed Base (Wave 2).

Packaged normalizations and crosses are Interaction / Pipeline Owned only:
  iv_emaN_to_ltp_ratio, iv_emaN_to_spot_ratio,
  ltp_emaN_to_spot_ratio_x_iv_emaN, spot_to_ltp_ratio_x_iv_emaN(_x_moneyness)

Reuses dedicated IV EMA controllers (updated once per IV sample in
update_token_iv_controllers).
"""

from __future__ import annotations

from typing import Any, Iterable

IV_EMA_PERIODS: tuple[int, ...] = (9, 20, 50, 100, 200, 300)

IV_EMA_LEVEL_FEATURES: tuple[str, ...] = tuple(f"iv_ema{p}" for p in IV_EMA_PERIODS)
IV_EMA_LEVEL_FEATURE_SET: frozenset[str] = frozenset(IV_EMA_LEVEL_FEATURES)

# Back-compat aliases used by enrichment / profiler call sites.
IV_EMA_RATIO_FEATURES: tuple[str, ...] = IV_EMA_LEVEL_FEATURES
IV_EMA_RATIO_FEATURE_SET: frozenset[str] = IV_EMA_LEVEL_FEATURE_SET


def active_iv_ema_ratio_features(active: Iterable[str] | None) -> frozenset[str]:
    if active is None:
        return IV_EMA_LEVEL_FEATURE_SET
    return frozenset(str(f) for f in active if str(f) in IV_EMA_LEVEL_FEATURE_SET)


def needs_iv_ema_ratio_features(active: Iterable[str] | None) -> bool:
    if active is None:
        return True
    return any(str(f) in IV_EMA_LEVEL_FEATURE_SET for f in active)


def _iv_ema_controller(controllers: Any, period: int) -> Any:
    return getattr(controllers, f"iv_ema{period}")


def enrich_iv_ema_ratio_features(
    raw: dict[str, Any],
    *,
    opt_state: Any | None = None,
    active_features: frozenset[str] | None = None,
) -> dict[str, Any]:
    wanted = active_iv_ema_ratio_features(active_features)
    if not wanted or opt_state is None:
        return raw

    from .rolling_controllers import emit_controller_value

    ctrl = getattr(opt_state, "controllers", None)
    if ctrl is None:
        return raw

    out = dict(raw)
    for period in IV_EMA_PERIODS:
        feat = f"iv_ema{period}"
        if feat not in wanted:
            continue
        out[feat] = emit_controller_value(_iv_ema_controller(ctrl, period))
    return out
