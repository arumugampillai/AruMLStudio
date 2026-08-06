"""Feature-history readiness — Layer 2 decides warmup, not replay clock."""

from __future__ import annotations

import math
from typing import Any, Mapping

import pandas as pd

from .market_state import LiveMarketState

# Canonical probes for longest lookbacks (EMA100, rolling std, 5m chain, PCR).
READINESS_PROBE_FEATURES: tuple[str, ...] = (
    "spot_ema100_to_ltp_ratio",
    "ltp_std20_to_ltp_ratio",
    "chain_pcr_change_5m",
    "chain_pcr",
)

# Minimum probes that must be present + valid to call warmup complete.
_MIN_READY_PROBES = 2


def _is_valid_value(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return False
    try:
        if pd.isna(val):
            return False
    except (TypeError, ValueError):
        pass
    return True


def assess_feature_readiness(features: Mapping[str, Any] | None) -> dict[str, object]:
    """Return warmup status from built feature values (replay-safe at any scrub position)."""
    if not features:
        return {
            "warmup_complete": False,
            "warmup_confidence": "low",
            "warmup_label": "Warmup: LOW confidence",
            "warmup_ready_probes": [],
            "warmup_missing_probes": list(READINESS_PROBE_FEATURES),
            "warmup_ready_ratio": 0.0,
            "warmup_reason": "no_features",
        }

    present = [name for name in READINESS_PROBE_FEATURES if name in features]
    ready = [name for name in present if _is_valid_value(features.get(name))]
    missing = [name for name in present if name not in ready]

    if not present:
        complete = len(features) > 0
        return {
            "warmup_complete": complete,
            "warmup_confidence": "complete" if complete else "low",
            "warmup_label": "Warmup Complete" if complete else "Warmup: LOW confidence",
            "warmup_ready_probes": [],
            "warmup_missing_probes": [],
            "warmup_ready_ratio": 1.0 if complete else 0.0,
            "warmup_reason": "no_probe_features_in_snapshot",
        }

    ratio = len(ready) / len(present)
    complete = len(ready) >= _MIN_READY_PROBES and len(missing) == 0
    return {
        "warmup_complete": complete,
        "warmup_confidence": "complete" if complete else "low",
        "warmup_label": "Warmup Complete" if complete else "Warmup: LOW confidence",
        "warmup_ready_probes": ready,
        "warmup_missing_probes": missing,
        "warmup_ready_ratio": round(ratio, 3),
        "warmup_reason": "feature_probes" if complete else "history_maturing",
    }


def history_span_sec(state: LiveMarketState, grid_ts: float) -> float:
    """Tick history available before grid_ts (for pre-build UI hints only)."""
    tl = state.index_timeline
    if not tl.timestamps:
        return 0.0
    first_tick = float(tl.timestamps[0])
    anchor = max(state.session_open_ts(), first_tick)
    return max(0.0, float(grid_ts) - anchor)


def warmup_info(
    state: LiveMarketState,
    grid_ts: float,
    *,
    features: Mapping[str, Any] | None = None,
) -> dict[str, object]:
    """Warmup from feature readiness; falls back to history span before features exist."""
    if features:
        info = dict(assess_feature_readiness(features))
        info["history_span_minutes"] = round(history_span_sec(state, grid_ts) / 60.0, 1)
        return info

    span_sec = history_span_sec(state, grid_ts)
    span_min = round(span_sec / 60.0, 1)
    return {
        "warmup_complete": False,
        "warmup_confidence": "low",
        "warmup_label": "Warmup: LOW confidence",
        "warmup_ready_probes": [],
        "warmup_missing_probes": list(READINESS_PROBE_FEATURES),
        "warmup_ready_ratio": 0.0,
        "warmup_reason": "features_pending",
        "history_span_minutes": span_min,
    }


def is_warmup_complete(
    features: Mapping[str, Any] | None,
    *,
    state: LiveMarketState | None = None,
    grid_ts: float | None = None,
) -> bool:
    if features is not None:
        return bool(assess_feature_readiness(features)["warmup_complete"])
    if state is not None and grid_ts is not None:
        return bool(warmup_info(state, grid_ts, features=None)["warmup_complete"])
    return False
