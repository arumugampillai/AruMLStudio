"""Spot/LTP EMA × moneyness crosses — Pipeline Owned only (Wave 2).

Registry no longer admits:
  spot_ema300_to_ltp_ratio (use spot_ema300 level + Interaction ÷ltp)
  spot_emaN_to_ltp_ratio_x_moneyness
  ltp_emaN_to_spot_ratio_x_moneyness

Canonical levels are emitted by extended_features / controllers.
This enricher is a no-op for Master Registry builds.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .rolling_controllers import SpotControllers

SPOT_EMA_MONEYNESS_PERIODS: tuple[int, ...] = (9, 20, 50, 100, 200, 300)
LTP_EMA_MONEYNESS_PERIODS: tuple[int, ...] = (9, 20, 50, 100, 200, 300)

# Empty: packaged crosses moved to Interaction / Pipeline Owned.
SPOT_RATIO_MONEYNESS_FEATURES: tuple[str, ...] = ()
SPOT_RATIO_MONEYNESS_FEATURE_SET: frozenset[str] = frozenset()


def needs_spot_ratio_moneyness(active: Iterable[str] | None) -> bool:
    del active
    return False


def active_spot_ratio_moneyness(active: Iterable[str] | None) -> frozenset[str]:
    del active
    return frozenset()


def enrich_spot_ratio_moneyness_features(
    raw: dict[str, Any],
    *,
    opt_state: Any | None = None,
    spot_controllers: SpotControllers | None = None,
    spot_rv_cache: Mapping[float, Mapping[str, float | None]] | None = None,
    ts: float | None = None,
    active_features: frozenset[str] | None = None,
) -> dict[str, Any]:
    del opt_state, spot_controllers, spot_rv_cache, ts, active_features
    return raw
