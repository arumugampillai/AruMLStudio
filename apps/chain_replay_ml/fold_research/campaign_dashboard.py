"""Campaign Dashboard — single-campaign research cockpit (Phase D4)."""

from __future__ import annotations

from typing import Any

from .campaign_scheduler import check_campaign_budget, get_campaign_scheduler_view
from .experiment_pipeline_store import ExperimentPipelineStore
from .research_program_store import ResearchProgramStore


def _num(v: Any) -> float | None:
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def _job_timeline_entry(job: dict[str, Any], template: dict[str, Any] | None) -> dict[str, Any]:
    comparison = job.get("comparison") or {}
    results = job.get("results") or {}
    verdict = results.get("verdict") or {}
    changes = (template or {}).get("accepted_changes") or []
    return {
        "job_id": job.get("job_id"),
        "job_number": job.get("job_number"),
        "status": job.get("status"),
        "template_id": job.get("template_id"),
        "template_number": (template or {}).get("template_number"),
        "goal": (template or {}).get("goal"),
        "change_text": (changes[0] or {}).get("text") if len(changes) == 1 else f"{len(changes)} changes",
        "verdict": verdict.get("verdict"),
        "recommendation": verdict.get("recommendation"),
        "pf_delta": comparison.get("pf_delta"),
        "after_pf": comparison.get("after_pf"),
        "after_win_rate_pct": comparison.get("after_win_rate_pct"),
        "started_at": job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "objective_score": ((template or {}).get("objective_score") or {}).get("overall"),
    }


def _dependency_rows(data_dir: str, campaign: dict[str, Any]) -> list[dict[str, Any]]:
    deps = campaign.get("dependencies") or []
    rows: list[dict[str, Any]] = []
    with ResearchProgramStore(data_dir) as store:
        for dep_id in deps:
            dep = store._load_campaign(str(dep_id))
            if not dep:
                rows.append({"campaign_id": dep_id, "name": "?", "status": "missing", "satisfied": False})
                continue
            satisfied = dep.get("status") in ("validated", "retired")
            rows.append({
                "campaign_id": dep.get("campaign_id"),
                "campaign_number": dep.get("campaign_number"),
                "name": dep.get("name"),
                "status": dep.get("status"),
                "satisfied": satisfied,
            })
    return rows


def get_campaign_dashboard(data_dir: str, campaign_id: str) -> dict[str, Any]:
    scheduler = get_campaign_scheduler_view(data_dir, campaign_id)
    if not scheduler.get("ok"):
        return scheduler

    with ResearchProgramStore(data_dir) as store:
        config = store.resolve_campaign_config(campaign_id)
    if not config:
        return {"ok": False, "error": "campaign not found"}

    campaign = config["campaign"]
    program = config["program"]
    memory = campaign.get("memory") or {}

    with ExperimentPipelineStore(data_dir) as store:
        jobs = store.list_jobs(campaign_id=campaign_id, limit=50)
        templates = {str(t.get("template_id") or ""): t for t in store.list_templates(campaign_id=campaign_id, limit=50)}

    timeline = []
    metrics_trend = []
    for job in sorted(jobs, key=lambda j: int(j.get("job_number") or 0)):
        if job.get("status") not in ("complete", "completed", "failed"):
            continue
        tmpl = templates.get(str(job.get("template_id") or ""))
        entry = _job_timeline_entry(job, tmpl)
        timeline.append(entry)
        if entry.get("after_pf") is not None:
            metrics_trend.append({
                "job_number": entry.get("job_number"),
                "after_pf": entry.get("after_pf"),
                "pf_delta": entry.get("pf_delta"),
                "verdict": entry.get("verdict"),
            })

    best_job_id = memory.get("best_job_id")
    best_template_id = memory.get("best_template_id")
    best_job = next((j for j in jobs if str(j.get("job_id")) == str(best_job_id)), None)
    best_template = templates.get(str(best_template_id or ""))

    budget = check_campaign_budget(data_dir, campaign_id)
    used = budget.get("used") or {}
    resolved_budget = budget.get("budget") or config.get("resolved_budget") or {}
    max_exp = resolved_budget.get("max_experiments")
    budget_burn = {
        "experiments_used": int(used.get("experiments") or 0),
        "experiments_limit": max_exp,
        "gpu_hours_used": used.get("max_gpu_hours"),
        "gpu_hours_limit": resolved_budget.get("max_gpu_hours"),
        "exhausted": bool(budget.get("exhausted")),
    }
    if max_exp:
        try:
            budget_burn["experiments_pct"] = round(int(used.get("experiments") or 0) / int(max_exp) * 100.0, 1)
        except (TypeError, ValueError, ZeroDivisionError):
            budget_burn["experiments_pct"] = None

    hypothesis_log = (scheduler.get("cycle") or {}).get("hypothesis_log") or memory.get("hypothesis_log") or []
    funnel = {
        "proposals_queued": len(scheduler.get("proposal_queue") or []),
        "experiments_completed": len(timeline),
        "hypotheses_tested": len(hypothesis_log),
        "validation_ready": bool((scheduler.get("cycle") or {}).get("validation_ready")),
    }

    pf_values = [_num(m.get("after_pf")) for m in metrics_trend]
    pf_values = [v for v in pf_values if v is not None]
    trend_summary = None
    if pf_values:
        trend_summary = {
            "first_pf": pf_values[0],
            "latest_pf": pf_values[-1],
            "best_pf": max(pf_values),
            "delta": round(pf_values[-1] - pf_values[0], 4) if len(pf_values) > 1 else 0,
        }

    return {
        "ok": True,
        "campaign_id": campaign_id,
        "campaign": campaign,
        "program": {"program_id": program.get("program_id"), "name": program.get("name")},
        "scheduler": scheduler,
        "timeline": timeline,
        "metrics_trend": metrics_trend,
        "trend_summary": trend_summary,
        "best_experiment": {
            "job": _job_timeline_entry(best_job, best_template) if best_job else None,
            "template_id": best_template_id,
            "template_number": (best_template or {}).get("template_number"),
            "generalization": memory.get("best_generalization"),
        },
        "budget_burn": budget_burn,
        "dependencies": _dependency_rows(data_dir, campaign),
        "funnel": funnel,
        "resolved_objective": config.get("resolved_objective"),
        "resolved_budget": config.get("resolved_budget"),
    }
