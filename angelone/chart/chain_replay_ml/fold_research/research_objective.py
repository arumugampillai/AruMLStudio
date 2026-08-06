"""Research Objective — schema, defaults, inheritance (Phase D1 + F1)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

IMPORTANCE_LEVELS = ("critical", "high", "medium", "low")
PROGRAM_TYPES = ("strategy", "feature", "risk", "general")
PROGRAM_STATUSES = ("draft", "active", "paused", "archived", "retired")
CAMPAIGN_STATUSES = (
    "created",
    "waiting",
    "running",
    "paused",
    "completed",
    "failed",
    "cancelled",
    "validated",
    "retired",
)
PROGRAM_RUN_STATUSES = ("waiting", "running", "paused", "completed", "failed", "cancelled")

DEFAULT_STOPPING_POLICY: dict[str, Any] = {
    "min_jobs": 10,
    "max_jobs": 50,
    "auto_stop": True,
    "min_confidence_pct": 85,
    "min_generalization": 70,
    "plateau_jobs": 3,
}

DEFAULT_SUCCESS_CRITERIA: dict[str, Any] = {
    "pf_delta_min": 0.05,
    "win_rate_drop_max": 2.0,
    "trade_count_min": 100,
}

DEFAULT_FAILURE_CRITERIA: dict[str, Any] = {
    "pf_delta_max": -0.02,
    "trade_count_min": 50,
}

DEFAULT_OBJECTIVE: dict[str, Any] = {
    "primary_goal": {
        "metric": "profit_factor",
        "direction": "maximize",
    },
    "constraints": [
        {"metric": "win_rate_pct", "op": ">=", "value": 80.0},
        {"metric": "trade_count", "op": ">=", "value": 100},
    ],
    "secondary_goals": [
        {"metric": "mae", "direction": "minimize", "weight": 0.15},
        {"metric": "information_gain", "direction": "maximize", "weight": 0.10},
    ],
    "importance": "medium",
}

DEFAULT_BUDGET: dict[str, Any] = {
    "max_experiments": 50,
    "min_experiments": 10,
    "max_gpu_hours": 15.0,
    "max_cpu_hours": 40.0,
    "max_wall_clock_hours": 72.0,
    "max_storage_gb": 50.0,
    "max_cost": None,
}


def default_stopping_policy(**overrides: Any) -> dict[str, Any]:
    policy = deepcopy(DEFAULT_STOPPING_POLICY)
    policy.update({k: v for k, v in overrides.items() if v is not None})
    return policy


def default_success_criteria(**overrides: Any) -> dict[str, Any]:
    doc = deepcopy(DEFAULT_SUCCESS_CRITERIA)
    doc.update({k: v for k, v in overrides.items() if v is not None})
    return doc


def default_failure_criteria(**overrides: Any) -> dict[str, Any]:
    doc = deepcopy(DEFAULT_FAILURE_CRITERIA)
    doc.update({k: v for k, v in overrides.items() if v is not None})
    return doc


def default_objective(**overrides: Any) -> dict[str, Any]:
    obj = deepcopy(DEFAULT_OBJECTIVE)
    for key, val in overrides.items():
        if key == "constraints" and isinstance(val, list):
            obj["constraints"] = val
        elif key == "secondary_goals" and isinstance(val, list):
            obj["secondary_goals"] = val
        elif val is not None:
            obj[key] = val
    return obj


def default_budget(**overrides: Any) -> dict[str, Any]:
    budget = deepcopy(DEFAULT_BUDGET)
    budget.update({k: v for k, v in overrides.items() if v is not None})
    return policy_budget_from_stopping(budget)


def policy_budget_from_stopping(budget: dict[str, Any]) -> dict[str, Any]:
    """Align max_experiments with stopping max_jobs when only stopping policy set."""
    out = dict(budget)
    if out.get("max_experiments") is None and out.get("max_jobs") is not None:
        out["max_experiments"] = out["max_jobs"]
    return out


def merge_stopping(
    program_stopping: dict[str, Any] | None,
    campaign_stopping: dict[str, Any] | None,
) -> dict[str, Any]:
    base = deepcopy(program_stopping or default_stopping_policy())
    base.update({k: v for k, v in (campaign_stopping or {}).items() if v is not None})
    return base


def merge_objective(
    program_objective: dict[str, Any] | None,
    campaign_objective: dict[str, Any] | None,
) -> dict[str, Any]:
    """Campaign inherits program objective; campaign keys override."""
    base = deepcopy(program_objective or default_objective())
    patch = campaign_objective or {}
    if patch.get("primary_goal"):
        base["primary_goal"] = {**(base.get("primary_goal") or {}), **patch["primary_goal"]}
    if patch.get("constraints"):
        base["constraints"] = list(patch["constraints"])
    if patch.get("secondary_goals"):
        base["secondary_goals"] = list(patch["secondary_goals"])
    if patch.get("importance"):
        base["importance"] = patch["importance"]
    return base


def merge_budget(
    program_budget: dict[str, Any] | None,
    campaign_budget: dict[str, Any] | None,
) -> dict[str, Any]:
    base = deepcopy(program_budget or default_budget())
    base.update({k: v for k, v in (campaign_budget or {}).items() if v is not None})
    return base


def validate_research_question(name: str, question: str) -> str | None:
    """Return error message if campaign scope is too broad."""
    q = str(question or name or "").strip().lower()
    if not q:
        return "research question is required"
    broad_phrases = (
        "improve entire strategy",
        "improve whole strategy",
        "fix everything",
        "all filters",
        "complete optimization",
    )
    for phrase in broad_phrases:
        if phrase in q:
            return "campaign must answer one research question — split into smaller campaigns"
    return None
