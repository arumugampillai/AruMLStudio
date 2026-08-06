"""Configurable thresholds for Recommendation Engine rules (Phase 5.3)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

# Defaults — override via run_compute(thresholds=...) or merge_thresholds().
DEFAULT_THRESHOLDS: dict[str, Any] = {
    # Drift / risk
    "high_drift": 0.35,
    "high_risk_score": 50.0,
    "high_risk_label": "high",
    # Importance (rank_gain: 1 = most important)
    "high_importance_rank_max": 20,  # ranks 1..N are "high importance"
    "low_importance_rank_min": 50,  # ranks >= N are "low importance" (soft if fewer feats)
    "low_importance_gain_max": 0.0,  # gain at/below → low when rank missing
    # Null drift (percentage points WF→holdout)
    "high_null_drift_pp": 5.0,
    # KS / Wasserstein / mean drift %
    "large_ks": 0.40,
    "small_ks": 0.15,
    "small_mean_drift_pct": 5.0,  # |drift_pct|
    "high_wasserstein_normalized": 1.0,
    # Priority mapping from evidence strength (internal 0–1 unit; UI shows 0–100)
    "priority_high_confidence_min": 0.75,
    "priority_medium_confidence_min": 0.50,
    # Experiment Planner v2 — family-split research experiments
    "max_experiments": 10,  # cap ranked experiments (~5–10)
    "top_contributors": 5,  # top features by risk in evidence
}

PLANNER_VERSION = "5.3.4"
SCHEMA_VERSION = 5

# Canonical category labels (UI + artifacts).
CATEGORIES = (
    "Feature Removal",
    "Feature Review",
    "Feature Addition",
    "Retraining",
    "Data Collection",
    "Threshold Review",
    "Model Refresh",
)

# Model-level vs feature-family experiment scopes (presentation only).
MODEL_EXPERIMENT_CATEGORIES = frozenset(
    {
        "Retraining",
        "Model Refresh",
        "Threshold Review",
        "Feature Addition",
    }
)

# Advisory research status (Phase 5.4 bridge); never auto-advanced by compute.
# Superseded is derived on recompute when an experiment_id leaves planner.json.
EXPERIMENT_STATUSES = (
    "Not Started",
    "In Progress",
    "Completed",
    "Rejected",
    "Superseded",
)
DEFAULT_EXPERIMENT_STATUS = "Not Started"

# Manual UI actions that may set status (not free-form dropdown).
STATUS_ACTIONS = (
    "mark_in_progress",
    "mark_complete",
    "reject",
    "add_notes",
)

# Estimated effort labels.
EFFORT_LEVELS = ("Easy", "Medium", "High")

# Feature-count bands for effort heuristics.
EFFORT_EASY_MAX_FEATURES = 5
EFFORT_HIGH_MIN_FEATURES = 25

# User-managed state sidecar (survives planner recompute).
EXPERIMENT_STATE_SCHEMA_VERSION = 1


def merge_thresholds(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    out = deepcopy(DEFAULT_THRESHOLDS)
    if overrides:
        for key, val in overrides.items():
            if key in out and val is not None:
                out[key] = val
    return out
