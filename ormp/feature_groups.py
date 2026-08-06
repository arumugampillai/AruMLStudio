"""ORMP feature groups for Dataset Builder / Model Builder metadata."""

from __future__ import annotations

from typing import Any

# Group id → ordered feature names (must cover FEATURE_COLUMNS exactly once).
FEATURE_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "ormp_band_state",
        "Band State",
        (
            "ormp_current_band",
            "ormp_current_band_time",
            "ormp_current_band_visits",
            "ormp_current_visit_duration",
            "ormp_last_visit_duration",
            "ormp_current_band_avg_stay",
        ),
    ),
    (
        "ormp_movement",
        "Movement",
        (
            "ormp_distance_from_open",
            "ormp_last_transition_distance",
            "ormp_last_transition_direction",
            "ormp_last_transition_duration",
            "ormp_crossed_bands_last_move",
        ),
    ),
    (
        "ormp_range",
        "Range",
        (
            "ormp_highest_band",
            "ormp_lowest_band",
            "ormp_unique_band_count",
            "ormp_distance_from_bottom",
            "ormp_distance_from_top",
            "ormp_position_in_range",
            "ormp_range_width",
            "ormp_range_expansion",
        ),
    ),
    (
        "ormp_transitions",
        "Transitions",
        (
            "ormp_total_band_transitions",
            "ormp_upward_transitions",
            "ormp_downward_transitions",
            "ormp_return_to_open_count",
        ),
    ),
    (
        "ormp_time_distribution",
        "Time Distribution",
        (
            "ormp_time_above_open",
            "ormp_time_below_open",
            "ormp_time_above_ratio",
            "ormp_time_below_ratio",
        ),
    ),
    (
        "ormp_visit_statistics",
        "Visit Statistics",
        (
            "ormp_visit_count",
            "ormp_average_visit_duration",
        ),
    ),
    (
        "ormp_market_context",
        "Market Context",
        (
            "1m_ema9_to_spot_ratio",
            "1m_ema20_to_spot_ratio",
            "1m_ema50_to_spot_ratio",
            "1m_ema100_to_spot_ratio",
            "1m_ema200_to_spot_ratio",
            "5m_ema9_to_spot_ratio",
            "5m_ema20_to_spot_ratio",
            "5m_ema50_to_spot_ratio",
            "5m_ema100_to_spot_ratio",
            "5m_ema200_to_spot_ratio",
            "15m_ema9_to_spot_ratio",
            "15m_ema20_to_spot_ratio",
            "15m_ema50_to_spot_ratio",
            "15m_ema100_to_spot_ratio",
            "15m_ema200_to_spot_ratio",
        ),
    ),
)


def feature_group_catalog() -> list[dict[str, Any]]:
    return [
        {"id": gid, "label": label, "features": list(feats)}
        for gid, label, feats in FEATURE_GROUPS
    ]


def all_grouped_features() -> list[str]:
    out: list[str] = []
    for _gid, _label, feats in FEATURE_GROUPS:
        out.extend(feats)
    return out


def features_for_groups(group_ids: set[str] | list[str]) -> list[str]:
    wanted = {str(g) for g in group_ids}
    out: list[str] = []
    for gid, _label, feats in FEATURE_GROUPS:
        if gid in wanted:
            out.extend(feats)
    return out
