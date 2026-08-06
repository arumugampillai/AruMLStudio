"""Export ML-ready ORMP features from the live profile (no internal dump).

Engine stores durations in seconds; exported time features are in **minutes**.
"""

from __future__ import annotations

from typing import Any

from .market_context import MARKET_CONTEXT_COLUMNS
from .profile_engine import OrmpProfile

ORMP_PROFILE_COLUMNS: tuple[str, ...] = (
    "ormp_current_band",
    "ormp_current_band_time",
    "ormp_current_band_visits",
    "ormp_current_visit_duration",
    "ormp_last_visit_duration",
    "ormp_visit_count",
    "ormp_average_visit_duration",
    "ormp_highest_band",
    "ormp_lowest_band",
    "ormp_unique_band_count",
    "ormp_total_band_transitions",
    "ormp_upward_transitions",
    "ormp_downward_transitions",
    "ormp_time_above_open",
    "ormp_time_below_open",
    "ormp_return_to_open_count",
    "ormp_current_band_avg_stay",
    "ormp_time_above_ratio",
    "ormp_time_below_ratio",
    "ormp_distance_from_open",
    "ormp_distance_from_bottom",
    "ormp_distance_from_top",
    "ormp_position_in_range",
    "ormp_range_width",
    "ormp_range_expansion",
    "ormp_last_transition_distance",
    "ormp_last_transition_direction",
    "ormp_last_transition_duration",
    "ormp_crossed_bands_last_move",
)

# Profile features + Market Context (multi-TF EMA / spot ratios).
FEATURE_COLUMNS: tuple[str, ...] = ORMP_PROFILE_COLUMNS + MARKET_CONTEXT_COLUMNS

IDENTITY_COLUMNS: tuple[str, ...] = (
    "trading_day",
    "timestamp",
    "spot_open",
)

# Evaluation / labeling prices (always 1m Close — independent of ormp_price_source)
FORWARD_LTP_HORIZONS_MIN: tuple[int, ...] = (5, 10, 15)
PRICE_COLUMNS: tuple[str, ...] = (
    "spot_ltp",
    *tuple(f"future_ltp_{h}m" for h in FORWARD_LTP_HORIZONS_MIN),
)


def _sec_to_min(sec: float | None) -> float | None:
    if sec is None:
        return None
    return float(sec) / 60.0


def _derived_range_features(
    current_band: int,
    highest_band: int,
    lowest_band: int,
    *,
    previous_range_width: float = 0.0,
) -> dict[str, float]:
    """Pure derived features from band extrema (no engine state)."""
    distance_from_open = float(current_band)
    distance_from_bottom = float(current_band - lowest_band)
    distance_from_top = float(highest_band - current_band)
    span = highest_band - lowest_band
    if span == 0:
        position_in_range = 0.5
    else:
        position_in_range = float(current_band - lowest_band) / float(span)
    return {
        "ormp_distance_from_open": distance_from_open,
        "ormp_distance_from_bottom": distance_from_bottom,
        "ormp_distance_from_top": distance_from_top,
        "ormp_position_in_range": position_in_range,
        "ormp_range_width": float(span),
        "ormp_range_expansion": float(span) - float(previous_range_width),
    }


def export_features(
    profile: OrmpProfile,
    *,
    previous_range_width: float = 0.0,
) -> dict[str, Any]:
    """Snapshot features after a candle update. Time features exported in minutes."""
    idx = profile.current_band
    if idx is None:
        raise RuntimeError("profile has no current band")
    if profile.highest_band is None or profile.lowest_band is None:
        raise RuntimeError("profile missing highest/lowest band")
    band = profile.bands[idx]
    elapsed_sec = float(profile.candles_processed * profile.candle_interval_sec)
    avg_sec = band.average_visit_duration
    highest = int(profile.highest_band)
    lowest = int(profile.lowest_band)

    out: dict[str, Any] = {
        "ormp_current_band": int(idx),
        "ormp_current_band_time": _sec_to_min(band.total_time),
        "ormp_current_band_visits": int(band.visit_count),
        "ormp_current_visit_duration": _sec_to_min(band.current_visit_duration),
        "ormp_last_visit_duration": _sec_to_min(band.last_visit_duration),
        "ormp_visit_count": int(band.visit_count),
        "ormp_average_visit_duration": _sec_to_min(avg_sec),
        "ormp_highest_band": highest,
        "ormp_lowest_band": lowest,
        "ormp_unique_band_count": int(profile.unique_band_count()),
        "ormp_total_band_transitions": int(profile.total_band_transitions),
        "ormp_upward_transitions": int(profile.upward_transitions),
        "ormp_downward_transitions": int(profile.downward_transitions),
        "ormp_time_above_open": _sec_to_min(profile.time_above_open),
        "ormp_time_below_open": _sec_to_min(profile.time_below_open),
        "ormp_return_to_open_count": int(profile.return_to_open_count),
        "ormp_current_band_avg_stay": _sec_to_min(avg_sec),
        "ormp_time_above_ratio": (
            min(1.0, max(0.0, profile.time_above_open / elapsed_sec))
            if elapsed_sec > 0
            else None
        ),
        "ormp_time_below_ratio": (
            min(1.0, max(0.0, profile.time_below_open / elapsed_sec))
            if elapsed_sec > 0
            else None
        ),
    }
    out.update(
        _derived_range_features(
            int(idx),
            highest,
            lowest,
            previous_range_width=previous_range_width,
        )
    )
    dist = float(profile.last_transition_distance())
    out["ormp_last_transition_distance"] = dist
    out["ormp_last_transition_direction"] = float(profile.last_transition_direction())
    out["ormp_last_transition_duration"] = _sec_to_min(profile.last_move_duration_sec)
    out["ormp_crossed_bands_last_move"] = dist  # same as abs(curr − prev)
    return out
