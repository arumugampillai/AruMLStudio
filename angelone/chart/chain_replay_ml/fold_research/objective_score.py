"""Objective Score — rank proposals against Research Objective (Phase D2)."""

from __future__ import annotations

from typing import Any

from .experiment_score import compute_experiment_score


def _importance_boost(importance: str) -> float:
    return {
        "critical": 8.0,
        "high": 5.0,
        "medium": 0.0,
        "low": -3.0,
    }.get(str(importance or "medium").lower(), 0.0)


def _constraint_penalty(objective: dict[str, Any], baseline: dict[str, Any]) -> tuple[bool, str | None]:
    """Hard reject when baseline already violates a constraint (no room to improve)."""
    constraints = objective.get("constraints") or []
    for c in constraints:
        metric = str(c.get("metric") or "")
        op = str(c.get("op") or "")
        try:
            threshold = float(c.get("value"))
            actual = baseline.get(metric)
            if actual is None:
                continue
            val = float(actual)
        except (TypeError, ValueError):
            continue
        if op == ">=" and val < threshold:
            return True, f"baseline {metric} {val} below constraint {threshold}"
        if op == "<=" and val > threshold:
            return True, f"baseline {metric} {val} above constraint {threshold}"
    return False, None


def compute_objective_score(
    data_dir: str,
    *,
    proposal: dict[str, Any],
    objective: dict[str, Any],
    importance: str = "medium",
    campaign_memory: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Score a proposal against the campaign Research Objective.

    Weights (domain model v1.1):
      expected primary-goal delta 35%, research value 25%, knowledge gap 15%,
      execution cost 10%, confidence 10%, duplicate penalty -5%.
    """
    selected = proposal.get("selected_recommendations") or []
    goal = str(proposal.get("goal") or "")
    baseline = proposal.get("baseline") or {}
    report_stub = {
        "ok": True,
        "research_report_id": proposal.get("research_report_id"),
        "prediction_run_id": proposal.get("prediction_run_id"),
        "strategy_run_id": proposal.get("strategy_run_id"),
        "executive_summary": {
            "model_id": proposal.get("model_id"),
            "strategy": proposal.get("strategy_label"),
        },
        "baseline_metrics": {
            "profit_factor": baseline.get("profit_factor"),
            "win_rate_pct": baseline.get("win_rate_pct"),
        },
    }
    exp_score = compute_experiment_score(
        data_dir,
        report_stub,
        accepted_items=selected,
        goal=goal,
    ) if selected else {"overall": 0, "expected_gain": 0, "novelty": 50, "evidence_strength": 40}

    rejected, reject_reason = _constraint_penalty(objective, baseline)
    if rejected:
        return {
            "overall": 0,
            "rejected": True,
            "reject_reason": reject_reason,
            "components": {},
            "experiment_score": exp_score,
        }

    expected_delta = float(exp_score.get("expected_gain") or exp_score.get("overall") or 0)
    research_value = float((proposal.get("score") or {}).get("follow_up", {}).get("expected_information_gain_score")
                          or exp_score.get("novelty") or 50)
    if isinstance((proposal.get("score") or {}).get("follow_up"), dict):
        gain_label = str((proposal.get("score") or {}).get("follow_up", {}).get("expected_information_gain") or "")
        gain_map = {"very high": 90, "high": 75, "medium": 55, "low": 35}
        research_value = gain_map.get(gain_label.lower(), research_value)

    knowledge_gap = float(exp_score.get("novelty") or 50)
    est_minutes = float(exp_score.get("estimated_minutes") or 10)
    execution_cost = max(0.0, min(100.0, 100.0 - est_minutes * 0.8))
    confidence = float(exp_score.get("evidence_strength") or 45)
    duplicate_penalty = 5.0 if exp_score.get("recommendation") == "Likely Duplicate" else 0.0

    raw = (
        expected_delta * 0.35
        + research_value * 0.25
        + knowledge_gap * 0.15
        + execution_cost * 0.10
        + confidence * 0.10
        - duplicate_penalty
        + _importance_boost(importance)
    )
    overall = int(max(0, min(100, round(raw))))

    memory = campaign_memory or {}
    if memory.get("experiments_run", 0) == 0 and len(selected) == 1:
        overall = min(100, overall + 3)

    return {
        "overall": overall,
        "rejected": False,
        "components": {
            "expected_primary_delta": round(expected_delta, 1),
            "research_value": round(research_value, 1),
            "knowledge_gap": round(knowledge_gap, 1),
            "execution_cost": round(execution_cost, 1),
            "confidence": round(confidence, 1),
            "duplicate_penalty": duplicate_penalty,
            "importance_boost": _importance_boost(importance),
        },
        "experiment_score": exp_score,
        "estimated_gpu_minutes": exp_score.get("gpu_minutes"),
        "estimated_cpu_minutes": exp_score.get("cpu_minutes"),
    }
