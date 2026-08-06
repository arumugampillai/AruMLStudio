"""Preset research campaigns — hypothesis + success/failure criteria (Phase F1)."""

from __future__ import annotations

from typing import Any

from .research_objective import (
    default_failure_criteria,
    default_stopping_policy,
    default_success_criteria,
)

CAMPAIGN_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "stop_optimization",
        "name": "Stop Optimization",
        "research_question": "What is the optimal stop loss percentage?",
        "hypothesis": "A tighter stop loss (around 7%) increases profit factor without hurting win rate.",
        "description": "Test stop-loss levels one change at a time (e.g. 5%, 7%, 10%).",
        "success_criteria": {"pf_delta_min": 0.15, "win_rate_drop_max": 2.0, "trade_count_min": 100},
        "failure_criteria": {"pf_delta_max": -0.02, "trade_count_min": 50},
        "stopping": {"min_jobs": 5, "max_jobs": 30, "auto_stop": True, "plateau_jobs": 2},
    },
    {
        "id": "premium_filter",
        "name": "Premium Optimization",
        "research_question": "What premium band maximizes profit factor?",
        "hypothesis": "Entries with premium ₹20–30 outperform low-premium entries.",
        "description": "Sweep minimum premium thresholds and bands.",
        "success_criteria": {"pf_delta_min": 0.1, "trade_count_min": 80},
        "failure_criteria": {"pf_delta_max": -0.02, "trade_count_min": 40},
        "stopping": default_stopping_policy(),
    },
    {
        "id": "confidence_filter",
        "name": "Confidence Threshold",
        "research_question": "What minimum prediction confidence maximizes profit factor?",
        "hypothesis": "Raising confidence to ~72% improves PF with acceptable trade reduction.",
        "description": "Test confidence cutoffs across regimes.",
        "success_criteria": {"pf_delta_min": 0.08, "trade_count_min": 60},
        "failure_criteria": {"pf_delta_max": -0.02, "trade_count_min": 30},
        "stopping": {"min_jobs": 10, "max_jobs": 50, "auto_stop": True},
    },
    {
        "id": "holding_time",
        "name": "Holding Time",
        "research_question": "What holding duration maximizes profit factor?",
        "hypothesis": "Shorter holds (~42s) reduce theta decay exposure.",
        "description": "Time-based exit optimization.",
        "success_criteria": {"pf_delta_min": 0.05, "trade_count_min": 80},
        "failure_criteria": {"pf_delta_max": -0.02},
        "stopping": default_stopping_policy(),
    },
    {
        "id": "theta_filter",
        "name": "Theta Filter",
        "research_question": "Do high-theta trades underperform?",
        "hypothesis": "Filtering high-theta entries improves profit factor.",
        "description": "Filter or avoid high theta decay entries.",
        "success_criteria": {"pf_delta_min": 0.05, "trade_count_min": 100},
        "failure_criteria": {"pf_delta_max": -0.02, "trade_count_min": 50},
        "stopping": default_stopping_policy(),
    },
    {
        "id": "skip_range",
        "name": "Market Regime",
        "research_question": "Do range-regime entries underperform?",
        "hypothesis": "Skipping range-bound regimes improves outcomes.",
        "description": "Regime filter for range vs trending markets.",
        "success_criteria": {"pf_delta_min": 0.05, "trade_count_min": 100},
        "failure_criteria": {"pf_delta_max": -0.02},
        "stopping": default_stopping_policy(),
    },
    {
        "id": "iv_expansion",
        "name": "IV Filter",
        "research_question": "Do IV expansion features improve model accuracy?",
        "hypothesis": "IV expansion features improve walk-forward performance.",
        "description": "Feature / model changes around implied volatility.",
        "success_criteria": {"pf_delta_min": 0.03},
        "failure_criteria": {"pf_delta_max": -0.01},
        "stopping": {"min_jobs": 8, "max_jobs": 40, "auto_stop": True},
    },
    {
        "id": "custom",
        "name": "Custom (type your own)",
        "research_question": "",
        "hypothesis": "",
        "description": "Enter a single focused research question manually.",
        "success_criteria": default_success_criteria(),
        "failure_criteria": default_failure_criteria(),
        "stopping": default_stopping_policy(),
    },
]

PROGRAM_TEMPLATES: list[dict[str, Any]] = [
    {
        "id": "otm_buyer_strategy",
        "name": "Current Expiry OTM Buyer Research v1",
        "program_type": "strategy",
        "description": "Strategy filters and exits for OTM buyer on current expiry.",
        "campaign_template_ids": [
            "stop_optimization",
            "premium_filter",
            "confidence_filter",
            "holding_time",
            "theta_filter",
            "skip_range",
        ],
    },
    {
        "id": "feature_engineering",
        "name": "Feature Engineering Program",
        "program_type": "feature",
        "description": "Feature ablation and validation campaigns.",
        "campaign_template_ids": ["iv_expansion"],
    },
]


def list_campaign_templates() -> list[dict[str, Any]]:
    return [dict(t) for t in CAMPAIGN_TEMPLATES]


def get_campaign_template(template_id: str) -> dict[str, Any] | None:
    for row in CAMPAIGN_TEMPLATES:
        if row.get("id") == template_id:
            return dict(row)
    return None


def list_program_templates() -> list[dict[str, Any]]:
    return [dict(t) for t in PROGRAM_TEMPLATES]
