"""Build summary preview and metadata — sampling, strikes, gap, targets, warm-up."""

from __future__ import annotations

from typing import Any

from .expected_spec import (
    format_sampling_interval_label,
    format_strike_selection_label,
    strike_selection_metadata,
)
from .feature_plugins import horizon_label
from .gap_policy import gap_max_sec_from_policy, gap_summary_label, normalize_gap_policy


def build_summary_metadata(
    *,
    feature_names: list[str],
    sampling_interval_sec: float,
    strike_selection: dict[str, Any] | None = None,
    gap_policy: dict[str, Any] | None = None,
    prediction_targets: dict[str, Any] | None = None,
    feature_count: int | None = None,
    target_count: int | None = None,
) -> dict[str, Any]:
    """Canonical build-summary block persisted in master / dataset metadata."""
    names = list(dict.fromkeys(feature_names))
    horizons_sec = [int(h) for h in (prediction_targets or {}).get("horizonsSec") or []]
    gap_doc = normalize_gap_policy(gap_policy)
    strike_doc = strike_selection_metadata(strike_selection or {})
    interval = float(sampling_interval_sec)
    return {
        "sampling_interval_sec": interval,
        "sampling_label": format_sampling_interval_label(interval) or f"{interval:g}s",
        "strike_selection": strike_doc,
        "gap_policy": gap_doc,
        "prediction_targets": {
            "type": str((prediction_targets or {}).get("targetType") or "future_ltp"),
            "horizons_sec": horizons_sec,
            "labels": [horizon_label(h) for h in horizons_sec],
        },
        "feature_count": int(feature_count if feature_count is not None else len(names)),
        "target_count": int(target_count if target_count is not None else len(horizons_sec)),
        "feature_names": names,
    }


def build_summary_labels(
    *,
    sampling_interval_sec: float,
    strike_selection: dict[str, Any] | None = None,
    gap_policy: dict[str, Any] | None = None,
    prediction_targets: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Human-readable labels for build summary UI."""
    horizons_sec = [int(h) for h in (prediction_targets or {}).get("horizonsSec") or []]
    interval = float(sampling_interval_sec)
    return {
        "sampling": format_sampling_interval_label(interval) or f"{interval:g}s",
        "strike": format_strike_selection_label(strike_selection) or "—",
        "gap": gap_summary_label(gap_policy),
        "targets": ", ".join(horizon_label(h) for h in horizons_sec) or "—",
    }


def build_summary_preview(
    feature_names: list[str],
    *,
    sampling_interval_sec: float = 10.0,
    strike_selection: dict[str, Any] | None = None,
    gap_policy: dict[str, Any] | None = None,
    prediction_targets: dict[str, Any] | None = None,
    gap_max_sec: float | None = None,
    estimated_rows: int | None = None,
    estimated_sessions: int | None = None,
) -> dict[str, Any]:
    """Pre-build summary: configuration, features, warm-up budget, checks."""
    from chain_replay_ml.feature_policy import build_validation_preview

    resolved_gap = (
        float(gap_max_sec)
        if gap_max_sec is not None
        else gap_max_sec_from_policy(gap_policy)
    )
    preview = build_validation_preview(
        feature_names,
        sampling_interval_sec=float(sampling_interval_sec),
        gap_max_sec=resolved_gap,
        estimated_rows=estimated_rows,
        estimated_sessions=estimated_sessions,
    )
    labels = build_summary_labels(
        sampling_interval_sec=sampling_interval_sec,
        strike_selection=strike_selection,
        gap_policy=gap_policy,
        prediction_targets=prediction_targets,
    )
    meta = build_summary_metadata(
        feature_names=feature_names,
        sampling_interval_sec=sampling_interval_sec,
        strike_selection=strike_selection,
        gap_policy=gap_policy,
        prediction_targets=prediction_targets,
    )
    return {
        **preview,
        "build_config": {
            **meta,
            "strike_label": labels["strike"],
            "gap_label": labels["gap"],
            "target_labels": meta["prediction_targets"]["labels"],
            "target_labels_text": labels["targets"],
        },
        "build_summary_metadata": meta,
    }
