"""Research Cycle — Hypothesis → Experiment → Evidence → Decision (Phase D3)."""

from __future__ import annotations

from typing import Any

CYCLE_STEPS = ("hypothesis", "experiment", "evidence", "decision")
EXPLORATION_STAGES = ("explore", "exploit", "validate")

STAGE_LABELS = {
    "explore": "Explore — single-variable hypotheses only",
    "exploit": "Exploit — refine winning hypothesis family",
    "validate": "Validate — generalization check before promotion",
}


def _num(v: Any) -> float | None:
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def infer_exploration_stage(memory: dict[str, Any]) -> str:
    """Internal exploration policy — explore until a winner, then exploit/validate."""
    if memory.get("validation_ready") or memory.get("best_generalization", {}).get("overall", 0) >= 70:
        return "validate"
    best_verdict = memory.get("best_verdict")
    if best_verdict in ("Strong Improvement", "Improvement") or memory.get("best_profit_factor"):
        if int(memory.get("experiments_run") or 0) >= 2:
            return "exploit"
    return "explore"


def allows_multi_change(memory: dict[str, Any]) -> bool:
    """Stage 1 = single-variable templates only."""
    return infer_exploration_stage(memory) != "explore"


def get_cycle_view(memory: dict[str, Any]) -> dict[str, Any]:
    step = str(memory.get("last_cycle_step") or "hypothesis")
    stage = infer_exploration_stage(memory)
    log = list(memory.get("hypothesis_log") or [])
    return {
        "current_step": step,
        "exploration_stage": stage,
        "stage_label": STAGE_LABELS.get(stage, stage),
        "experiments_run": int(memory.get("experiments_run") or 0),
        "hypothesis_log": log[-10:],
        "validation_ready": bool(memory.get("validation_ready")),
        "best_job_id": memory.get("best_job_id"),
        "best_generalization": memory.get("best_generalization"),
    }


def _objective_met(comparison: dict[str, Any], objective: dict[str, Any] | None) -> bool:
    if not objective:
        return False
    constraints = objective.get("constraints") or []
    after_pf = _num(comparison.get("after_pf") or comparison.get("after_profit_factor"))
    after_wr = _num(comparison.get("after_win_rate_pct"))
    after_tc = _num(comparison.get("after_trade_count"))
    metrics = {
        "profit_factor": after_pf,
        "win_rate_pct": after_wr,
        "trade_count": after_tc,
    }
    for c in constraints:
        metric = str(c.get("metric") or "")
        op = str(c.get("op") or "")
        try:
            threshold = float(c.get("value"))
            actual = metrics.get(metric)
            if actual is None:
                continue
            val = float(actual)
        except (TypeError, ValueError):
            continue
        if op == ">=" and val < threshold:
            return False
        if op == "<=" and val > threshold:
            return False
    primary = objective.get("primary_goal") or {}
    if primary.get("metric") == "profit_factor" and primary.get("direction") == "maximize":
        base_pf = _num(comparison.get("baseline_pf"))
        if after_pf is not None and base_pf is not None and after_pf <= base_pf:
            return False
    return True


def record_hypothesis_trial(
    memory: dict[str, Any],
    *,
    template: dict[str, Any],
    job: dict[str, Any],
    comparison: dict[str, Any],
    verdict: dict[str, Any],
) -> dict[str, Any]:
    """Append evidence to campaign memory hypothesis log."""
    memory = dict(memory)
    log = list(memory.get("hypothesis_log") or [])
    changes = template.get("accepted_changes") or []
    entry = {
        "job_id": job.get("job_id"),
        "job_number": job.get("job_number"),
        "template_id": template.get("template_id"),
        "goal": template.get("goal"),
        "change_count": len(changes),
        "change_text": (changes[0] or {}).get("text") if len(changes) == 1 else f"{len(changes)} changes",
        "verdict": verdict.get("verdict"),
        "pf_delta": comparison.get("pf_delta"),
        "after_pf": comparison.get("after_pf"),
        "objective_score": (template.get("objective_score") or {}).get("overall"),
    }
    log.append(entry)
    memory["hypothesis_log"] = log[-50:]

    vlabel = str(verdict.get("verdict") or "")
    if vlabel in ("Strong Improvement", "Improvement"):
        memory["best_verdict"] = vlabel
    memory["last_cycle_step"] = "evidence"
    memory["exploration_stage"] = infer_exploration_stage(memory)
    return memory


def apply_cycle_decision(
    memory: dict[str, Any],
    *,
    comparison: dict[str, Any],
    verdict: dict[str, Any],
    objective: dict[str, Any] | None,
    generalization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Decision step — update memory with accept/reject/continue signals."""
    memory = dict(memory)
    memory["last_cycle_step"] = "decision"
    memory["last_decision"] = {
        "verdict": verdict.get("verdict"),
        "recommendation": verdict.get("recommendation"),
        "objective_met": _objective_met(comparison, objective),
        "continue_research": verdict.get("verdict") not in ("Strong Improvement",),
    }

    gen_overall = int((generalization or {}).get("overall") or 0)
    obj_met = bool(memory["last_decision"].get("objective_met"))
    if gen_overall >= 70 and obj_met:
        memory["validation_ready"] = True
        memory["exploration_stage"] = "validate"
    else:
        memory["validation_ready"] = bool(memory.get("validation_ready")) and gen_overall >= 50

    memory["exploration_stage"] = infer_exploration_stage(memory)
    return memory


def process_job_cycle_event(
    data_dir: str,
    campaign_id: str,
    *,
    job: dict[str, Any],
    template: dict[str, Any],
    objective: dict[str, Any] | None = None,
    generalization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .research_program import get_research_campaign, update_research_campaign

    campaign = get_research_campaign(data_dir, campaign_id)
    if not campaign:
        return {"ok": False, "error": "campaign not found"}

    memory = dict(campaign.get("memory") or {})
    comparison = job.get("comparison") or {}
    verdict = (job.get("results") or {}).get("verdict") or {}
    if not objective:
        objective = campaign.get("resolved_objective")

    memory = record_hypothesis_trial(
        memory,
        template=template,
        job=job,
        comparison=comparison,
        verdict=verdict,
    )
    memory = apply_cycle_decision(
        memory,
        comparison=comparison,
        verdict=verdict,
        objective=objective,
        generalization=generalization,
    )
    if generalization:
        memory["best_generalization"] = {
            "overall": generalization.get("overall"),
            "label": generalization.get("label"),
            "promote_recommended": generalization.get("promote_recommended"),
            "job_id": job.get("job_id"),
            "dimensions": generalization.get("dimensions"),
        }

    update_research_campaign(data_dir, campaign_id, memory=memory)
    return {
        "ok": True,
        "memory": memory,
        "cycle": get_cycle_view(memory),
        "objective_met": memory.get("last_decision", {}).get("objective_met"),
    }
