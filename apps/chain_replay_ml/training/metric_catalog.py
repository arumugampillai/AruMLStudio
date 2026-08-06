"""Metric taxonomy: Model Quality (endpoint) vs Trading Outcome (path).

UI labels must not collide across categories. Canonical implementations live in:

- Model Quality → ``training.evaluator``
  (``directional_accuracy_pct``, ``endpoint_hit_rate_pct``, ``evaluate_regression``)
- Trading Outcome → ``prediction_meta.outcomes``
  (``first_target_reached_ts``, path MFE/MAE / dd_before_target)
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Model Quality — predicted vs actual *endpoint*
# ---------------------------------------------------------------------------
MODEL_QUALITY: dict[str, dict[str, Any]] = {
    "mae": {"ui_label": "MAE", "unit": "₹"},
    "rmse": {"ui_label": "RMSE", "unit": "₹"},
    "mape": {"ui_label": "MAPE", "unit": "%"},
    "r2": {"ui_label": "R²", "unit": None},
    "premium_mae_pct": {"ui_label": "Premium MAE %", "unit": "%"},
    "premium_rmse_pct": {"ui_label": "Premium RMSE %", "unit": "%"},
    "directional_accuracy_pct": {"ui_label": "Direction Accuracy", "unit": "%"},
    "endpoint_hit_pct": {
        "ui_label": "Endpoint Hit %",
        "unit": "%",
        "formula": "|pred−actual|/|actual| ≤ 5%",
        "legacy_keys": ("hit_rate_pct", "target_hit_pct"),
    },
    "composite_score": {"ui_label": "Composite Score", "unit": None},
    "prediction_bias": {"ui_label": "Prediction Bias", "unit": "₹"},
    "medae": {"ui_label": "Median Abs Error", "unit": "₹"},
    "p95_error": {"ui_label": "P95 Abs Error", "unit": "₹"},
}

# ---------------------------------------------------------------------------
# Trading Outcome — prediction vs future *price path*
# ---------------------------------------------------------------------------
TRADING_OUTCOME: dict[str, dict[str, Any]] = {
    "path_touch_rate": {
        "ui_label": "Path Touch Rate",
        "unit": "fraction",
        "storage_keys": ("target_reached", "target_hit_rate", "hit_rate"),
        "formula": "LTP touched predicted premium during horizon",
    },
    "time_to_target": {"ui_label": "Time to Target", "unit": "s"},
    "dd_before_target": {"ui_label": "DD Before Target", "unit": "₹"},
    "maximum_profit": {"ui_label": "Max Profit (path)", "unit": "₹"},
    "maximum_drawdown": {"ui_label": "Max Drawdown (path)", "unit": "₹"},
}

# UI display strings — single source for rename-safe labels
UI_ENDPOINT_HIT = MODEL_QUALITY["endpoint_hit_pct"]["ui_label"]  # "Endpoint Hit %"
UI_DIRECTION = MODEL_QUALITY["directional_accuracy_pct"]["ui_label"]  # "Direction Accuracy"
UI_PATH_TOUCH = TRADING_OUTCOME["path_touch_rate"]["ui_label"]  # "Path Touch Rate"
UI_PATH_TOUCH_CLASSIFIER = "Path Touch"  # confidence classifier / binary label


def assert_no_shared_ui_labels() -> None:
    """Raise if a display string is claimed by both categories."""
    mq = {str(v["ui_label"]) for v in MODEL_QUALITY.values()}
    to = {str(v["ui_label"]) for v in TRADING_OUTCOME.values()}
    to.add(UI_PATH_TOUCH_CLASSIFIER)
    overlap = mq & to
    if overlap:
        raise RuntimeError(f"Shared UI labels across metric categories: {sorted(overlap)}")


assert_no_shared_ui_labels()
