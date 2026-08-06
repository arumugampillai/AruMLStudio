"""Evidence-based campaign stopping (Phase F2)."""

from __future__ import annotations

from typing import Any

from .research_objective import default_stopping_policy, merge_stopping


def resolve_stopping_policy(
    program_stopping: dict[str, Any] | None,
    campaign_stopping: dict[str, Any] | None,
    budget: dict[str, Any] | None = None,
) -> dict[str, Any]:
    policy = merge_stopping(program_stopping, campaign_stopping)
    if budget:
        if budget.get("max_experiments") is not None:
            policy["max_jobs"] = int(budget["max_experiments"])
        if budget.get("min_experiments") is not None:
            policy["min_jobs"] = int(budget["min_experiments"])
    return policy


def evaluate_campaign_stop(
    *,
    stopping: dict[str, Any],
    completed_jobs: int,
    comparison: dict[str, Any] | None = None,
    success_criteria: dict[str, Any] | None = None,
    failure_criteria: dict[str, Any] | None = None,
    generalization: dict[str, Any] | None = None,
    confidence_pct: float | None = None,
    plateau_count: int = 0,
) -> dict[str, Any]:
    """Return whether campaign should stop and why."""
    min_jobs = int(stopping.get("min_jobs") or 10)
    max_jobs = int(stopping.get("max_jobs") or 50)
    auto_stop = bool(stopping.get("auto_stop", True))

    comparison = comparison or {}
    success = success_criteria or {}
    failure = failure_criteria or {}

    pf_delta = comparison.get("pf_delta")
    try:
        pf_delta_f = float(pf_delta) if pf_delta is not None else None
    except (TypeError, ValueError):
        pf_delta_f = None

    trade_count = comparison.get("after_trade_count") or comparison.get("trade_count")
    try:
        trade_count_i = int(trade_count) if trade_count is not None else None
    except (TypeError, ValueError):
        trade_count_i = None

    if completed_jobs >= max_jobs:
        return {
            "should_stop": True,
            "reason": "max_jobs_reached",
            "label": f"Maximum jobs ({max_jobs}) reached",
        }

    if completed_jobs < min_jobs:
        return {
            "should_stop": False,
            "reason": "below_min_jobs",
            "label": f"Need at least {min_jobs} jobs ({completed_jobs} done)",
        }

    fail_trade_min = failure.get("trade_count_min")
    if fail_trade_min is not None and trade_count_i is not None and trade_count_i < int(fail_trade_min):
        return {
            "should_stop": True,
            "reason": "failure_trade_count",
            "label": f"Trade count {trade_count_i} below failure threshold",
        }

    fail_pf = failure.get("pf_delta_max")
    if fail_pf is not None and pf_delta_f is not None and pf_delta_f <= float(fail_pf):
        return {
            "should_stop": True,
            "reason": "failure_pf",
            "label": f"PF delta {pf_delta_f:+.4f} below failure threshold",
        }

    if not auto_stop:
        return {"should_stop": False, "reason": "auto_stop_disabled", "label": "Auto-stop disabled"}

    succ_pf = success.get("pf_delta_min")
    gen_min = int(stopping.get("min_generalization") or 70)
    conf_min = float(stopping.get("min_confidence_pct") or 85)
    gen_score = int((generalization or {}).get("overall") or 0)
    conf = float(confidence_pct or gen_score or 0)

    success_met = (
        succ_pf is not None
        and pf_delta_f is not None
        and pf_delta_f >= float(succ_pf)
        and gen_score >= gen_min
        and conf >= conf_min
    )
    plateau_jobs = int(stopping.get("plateau_jobs") or 3)
    if success_met:
        return {
            "should_stop": True,
            "reason": "success_criteria_met",
            "label": f"Winner found — PF delta {pf_delta_f:+.4f}, Gen {gen_score}, confidence {conf:.0f}%",
        }

    if plateau_count >= plateau_jobs and completed_jobs >= min_jobs:
        return {
            "should_stop": True,
            "reason": "plateau",
            "label": f"No improvement over last {plateau_jobs} jobs",
        }

    return {
        "should_stop": False,
        "reason": "continue",
        "label": f"Collecting evidence ({completed_jobs}/{max_jobs})",
    }
