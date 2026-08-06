"""Campaign Scheduler — Phase D2 autonomous research loop (+ F2/F3 lifecycle)."""

from __future__ import annotations

from typing import Any

from .campaign_proposal_generator import (
    rank_campaign_proposals,
    seed_proposals_from_job,
    seed_proposals_from_report,
)
from .campaign_stopping import evaluate_campaign_stop, resolve_stopping_policy
from .experiment_pipeline_store import ExperimentPipelineStore
from .research_program_store import ResearchProgramStore


def _utc_now() -> str:
    from .research_program_store import _utc_now as _now

    return _now()


def evaluate_campaign_should_stop(data_dir: str, campaign_id: str) -> dict[str, Any]:
    """Evidence-based stop decision (Phase F2)."""
    with ResearchProgramStore(data_dir) as store:
        config = store.resolve_campaign_config(campaign_id)
    if not config:
        return {"ok": False, "error": "campaign not found", "should_stop": False}

    campaign = config["campaign"]
    memory = campaign.get("memory") or {}
    stopping = resolve_stopping_policy(
        config["program"].get("stopping"),
        campaign.get("stopping"),
        config.get("resolved_budget"),
    )

    with ExperimentPipelineStore(data_dir) as store:
        jobs = store.list_jobs(campaign_id=campaign_id, limit=500)
    completed = [j for j in jobs if j.get("status") in ("complete", "completed")]
    budget_used = int((campaign.get("budget_used") or {}).get("experiments") or 0)
    completed_jobs = max(len(completed), budget_used, int(memory.get("experiments_run") or 0))

    comparison: dict[str, Any] = {}
    best_job_id = memory.get("best_job_id")
    if best_job_id:
        best_job = next((j for j in completed if str(j.get("job_id")) == str(best_job_id)), None)
        if best_job:
            comparison = best_job.get("comparison") or {}

    gen = memory.get("best_generalization") or {}
    conf = gen.get("overall")
    plateau = int(memory.get("no_improvement_streak") or 0)

    decision = evaluate_campaign_stop(
        stopping=stopping,
        completed_jobs=completed_jobs,
        comparison=comparison,
        success_criteria=campaign.get("success_criteria"),
        failure_criteria=campaign.get("failure_criteria"),
        generalization=gen,
        confidence_pct=float(conf) if conf is not None else None,
        plateau_count=plateau,
    )
    return {"ok": True, "decision": decision, "stopping": stopping}


def check_campaign_budget(data_dir: str, campaign_id: str) -> dict[str, Any]:
    with ResearchProgramStore(data_dir) as store:
        config = store.resolve_campaign_config(campaign_id)
    if not config:
        return {"ok": False, "error": "campaign not found", "exhausted": True}

    budget = config.get("resolved_budget") or {}
    used = config["campaign"].get("budget_used") or {}
    checks: list[dict[str, Any]] = []

    max_exp = budget.get("max_experiments")
    exp_used = int(used.get("experiments") or 0)
    if max_exp is not None and exp_used >= int(max_exp):
        checks.append({"field": "max_experiments", "used": exp_used, "limit": max_exp})

    for field in ("max_gpu_hours", "max_cpu_hours", "max_wall_clock_hours"):
        limit = budget.get(field)
        val_used = float(used.get(field) or 0)
        if limit is not None and val_used >= float(limit):
            checks.append({"field": field, "used": val_used, "limit": limit})

    exhausted = bool(checks)
    stop_out = evaluate_campaign_should_stop(data_dir, campaign_id)
    stop_decision = (stop_out.get("decision") or {}) if stop_out.get("ok") else {}
    if stop_decision.get("should_stop") and int(used.get("experiments") or 0) >= int(
        (stop_out.get("stopping") or {}).get("min_jobs") or 0
    ):
        exhausted = True
        checks.append({
            "field": "evidence_stop",
            "reason": stop_decision.get("reason"),
            "label": stop_decision.get("label"),
        })

    return {
        "ok": True,
        "exhausted": exhausted,
        "violations": checks,
        "budget": budget,
        "used": used,
        "stop_decision": stop_decision,
        "remaining": {
            "max_experiments": (None if max_exp is None else max(0, int(max_exp) - exp_used)),
        },
    }


def get_campaign_scheduler_view(data_dir: str, campaign_id: str) -> dict[str, Any]:
    with ResearchProgramStore(data_dir) as store:
        config = store.resolve_campaign_config(campaign_id)
    if not config:
        return {"ok": False, "error": "campaign not found"}

    campaign = config["campaign"]
    memory = campaign.get("memory") or {}
    budget_check = check_campaign_budget(data_dir, campaign_id)

    with ExperimentPipelineStore(data_dir) as store:
        proposals = store.list_proposals(campaign_id=campaign_id, status="draft", limit=20)
        running_job = store.get_running_job_for_campaign(campaign_id)
        templates = store.list_templates(campaign_id=campaign_id, limit=10)
        jobs = store.list_jobs(campaign_id=campaign_id, limit=10)

    proposals.sort(
        key=lambda p: int((p.get("objective_score") or {}).get("overall") or 0),
        reverse=True,
    )

    from .research_cycle import get_cycle_view

    cycle = get_cycle_view(memory)

    return {
        "ok": True,
        "campaign_id": campaign_id,
        "status": campaign.get("status"),
        "research_question": campaign.get("research_question"),
        "hypothesis": campaign.get("hypothesis"),
        "success_criteria": campaign.get("success_criteria"),
        "failure_criteria": campaign.get("failure_criteria"),
        "stopping": config.get("resolved_stopping"),
        "manifest": campaign.get("manifest"),
        "memory": memory,
        "budget": budget_check,
        "running_job": running_job,
        "proposal_queue": proposals,
        "recent_templates": templates[:5],
        "recent_jobs": jobs[:5],
        "auto_run": bool(memory.get("auto_run")),
        "cycle": cycle,
    }


def update_campaign_budget_used(
    data_dir: str,
    campaign_id: str,
    *,
    job: dict[str, Any],
    template: dict[str, Any],
) -> None:
    from .research_program import get_research_campaign, update_research_campaign

    campaign = get_research_campaign(data_dir, campaign_id)
    if not campaign:
        return

    used = dict(campaign.get("budget_used") or {})
    used["experiments"] = int(used.get("experiments") or 0) + 1

    obj_score = template.get("objective_score") or {}
    exp_score = (template.get("score") or {}).get("experiment_score") or template.get("score") or {}
    gpu_min = float(obj_score.get("estimated_gpu_minutes") or exp_score.get("gpu_minutes") or 0) / 60.0
    cpu_min = float(obj_score.get("estimated_cpu_minutes") or exp_score.get("cpu_minutes") or 0) / 60.0
    used["max_gpu_hours"] = round(float(used.get("max_gpu_hours") or 0) + gpu_min, 3)
    used["max_cpu_hours"] = round(float(used.get("max_cpu_hours") or 0) + cpu_min, 3)

    if job.get("started_at") and job.get("completed_at"):
        try:
            from datetime import datetime

            start = datetime.fromisoformat(str(job["started_at"]))
            end = datetime.fromisoformat(str(job["completed_at"]))
            hours = max(0.0, (end - start).total_seconds() / 3600.0)
            used["max_wall_clock_hours"] = round(float(used.get("max_wall_clock_hours") or 0) + hours, 3)
        except (TypeError, ValueError):
            pass

    update_research_campaign(data_dir, campaign_id, budget_used=used)


def update_campaign_memory_after_job(
    data_dir: str,
    campaign_id: str,
    *,
    job: dict[str, Any],
    template: dict[str, Any],
    post: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .research_program import get_research_campaign, update_research_campaign

    campaign = get_research_campaign(data_dir, campaign_id)
    if not campaign:
        return {"ok": False, "error": "campaign not found"}

    memory = dict(campaign.get("memory") or {})
    memory["last_job_id"] = job.get("job_id")
    memory["last_template_id"] = template.get("template_id")
    memory["experiments_run"] = int(memory.get("experiments_run") or 0) + 1

    comparison = job.get("comparison") or {}
    after_pf = comparison.get("after_profit_factor") or comparison.get("profit_factor_after")
    best_pf = memory.get("best_profit_factor")
    improved = False
    try:
        if after_pf is not None and (best_pf is None or float(after_pf) > float(best_pf)):
            memory["best_profit_factor"] = float(after_pf)
            memory["best_job_id"] = job.get("job_id")
            memory["best_template_id"] = template.get("template_id")
            improved = True
    except (TypeError, ValueError):
        pass

    if improved:
        memory["no_improvement_streak"] = 0
    else:
        memory["no_improvement_streak"] = int(memory.get("no_improvement_streak") or 0) + 1

    verdict_text = ((job.get("results") or {}).get("verdict") or {}).get("verdict")
    if verdict_text in ("Reject", "Worse", "Failed"):
        memory["rejected_hypotheses"] = int(memory.get("rejected_hypotheses") or 0) + 1

    outputs = job.get("outputs") or {}
    if outputs.get("research_report_id"):
        memory["last_research_report_id"] = outputs.get("research_report_id")

    verdict = (job.get("results") or {}).get("verdict") or {}
    memory["last_verdict"] = verdict.get("verdict")

    update_research_campaign(data_dir, campaign_id, memory=memory)
    update_campaign_budget_used(data_dir, campaign_id, job=job, template=template)

    if post and post.get("next_experiments"):
        seed_proposals_from_job(data_dir, campaign_id, str(job.get("job_id") or ""))

    rank_campaign_proposals(data_dir, campaign_id)

    gen = None
    if str(memory.get("best_job_id") or "") == str(job.get("job_id") or ""):
        from .generalization_score import compute_job_generalization

        gen_out = compute_job_generalization(data_dir, str(job.get("job_id") or ""))
        if gen_out.get("ok"):
            gen = gen_out

    from .research_cycle import process_job_cycle_event

    process_job_cycle_event(
        data_dir,
        campaign_id,
        job=job,
        template=template,
        generalization=gen,
    )

    from .knowledge_pipeline import finalize_campaign_job_knowledge
    from .kb_proposal_generator import seed_kb_driven_proposals

    kb_out = finalize_campaign_job_knowledge(
        data_dir,
        job=job,
        campaign_id=campaign_id,
        program_id=str(campaign.get("program_id") or ""),
    )
    stage = str((memory.get("exploration_stage") or "explore")).lower()
    kb_seed = None
    if stage == "explore":
        kb_seed = seed_kb_driven_proposals(data_dir, campaign_id, limit=2)

    from .campaign_manifest import checkpoint_campaign_after_job

    knowledge_delta = {
        "created": int((kb_out or {}).get("created") or 0),
        "updated": int((kb_out or {}).get("updated") or 0),
    }
    checkpoint_campaign_after_job(
        data_dir,
        campaign_id,
        job=job,
        knowledge_delta=knowledge_delta,
    )

    return {"ok": True, "memory": memory, "knowledge": kb_out, "kb_proposals": kb_seed}


def pick_next_proposal(data_dir: str, campaign_id: str) -> dict[str, Any] | None:
    ranked = rank_campaign_proposals(data_dir, campaign_id)
    if not ranked.get("ok"):
        return None
    for proposal in ranked.get("proposals") or []:
        obj = proposal.get("objective_score") or {}
        if obj.get("rejected") or int(obj.get("overall") or 0) <= 0:
            continue
        return proposal
    return None


def run_next_campaign_experiment(data_dir: str, campaign_id: str) -> dict[str, Any]:
    from .experiment_pipeline import create_template_from_proposal, create_template_job

    with ResearchProgramStore(data_dir) as store:
        campaign = store._load_campaign(campaign_id)
    if not campaign:
        return {"ok": False, "error": "campaign not found"}
    if campaign.get("status") != "running":
        return {"ok": False, "error": f"campaign is not running (status={campaign.get('status')})"}

    budget = check_campaign_budget(data_dir, campaign_id)
    if budget.get("exhausted"):
        return {"ok": False, "error": "campaign budget exhausted", "budget": budget}

    with ExperimentPipelineStore(data_dir) as store:
        if store.get_running_job_for_campaign(campaign_id):
            return {"ok": False, "error": "a job is already running for this campaign"}

    proposal = pick_next_proposal(data_dir, campaign_id)
    if not proposal:
        memory = campaign.get("memory") or {}
        if memory.get("baseline_research_report_id"):
            seed = seed_proposals_from_report(data_dir, campaign_id)
            if seed.get("ok") and seed.get("count"):
                proposal = pick_next_proposal(data_dir, campaign_id)
        if not proposal:
            return {"ok": False, "error": "no ranked proposals — attach baseline and seed proposals"}

    pid = str(proposal.get("proposal_id") or "")
    tmpl_out = create_template_from_proposal(data_dir, pid)
    if not tmpl_out.get("ok"):
        return tmpl_out

    template = tmpl_out.get("template") or {}
    tid = str(template.get("template_id") or "")

    with ExperimentPipelineStore(data_dir) as store:
        template = store._load_template(tid) or template
        template["campaign_id"] = campaign_id
        template["program_id"] = proposal.get("program_id")
        template["objective_score"] = proposal.get("objective_score")
        store.save_template(template)

    job_out = create_template_job(data_dir, tid)
    if not job_out.get("ok"):
        return job_out

    job = job_out.get("job") or {}
    with ExperimentPipelineStore(data_dir) as store:
        job = store._load_job(str(job.get("job_id") or "")) or job
        job["campaign_id"] = campaign_id
        job["program_id"] = proposal.get("program_id")
        store.save_job(job)

    from .research_program import update_research_campaign

    memory = dict(campaign.get("memory") or {})
    memory["last_cycle_step"] = "experiment"
    memory["active_job_id"] = job.get("job_id")
    memory["active_template_id"] = tid
    update_research_campaign(data_dir, campaign_id, memory=memory)

    return {
        "ok": True,
        "proposal": proposal,
        "template": template,
        "job": job,
        "objective_score": proposal.get("objective_score"),
    }


def on_campaign_job_complete(data_dir: str, campaign_id: str, job_result: dict[str, Any]) -> dict[str, Any]:
    from .research_program import update_research_campaign

    job = job_result.get("job") or {}
    with ExperimentPipelineStore(data_dir) as store:
        template = store._load_template(str(job.get("template_id") or ""))

    post = (job.get("results") or {})
    update_campaign_memory_after_job(
        data_dir,
        campaign_id,
        job=job,
        template=template or {},
        post={"next_experiments": post.get("next_experiments")},
    )

    memory_update: dict[str, Any] = {"last_cycle_step": "decision"}
    with ResearchProgramStore(data_dir) as store:
        campaign = store._load_campaign(campaign_id)
    if campaign:
        memory = dict(campaign.get("memory") or {})
        memory.pop("active_job_id", None)
        memory_update["memory"] = memory

    budget = check_campaign_budget(data_dir, campaign_id)
    if budget.get("exhausted"):
        stop_reason = ((budget.get("stop_decision") or {}).get("reason") or "budget_exhausted")
        mem = memory_update.get("memory")
        update_research_campaign(
            data_dir,
            campaign_id,
            status="completed",
            memory=mem,
        )
        prog_out = None
        if mem and mem.get("program_run_id"):
            from .program_runner import on_program_campaign_complete

            prog_out = on_program_campaign_complete(data_dir, campaign_id)
        return {
            "ok": True,
            "action": stop_reason,
            "budget": budget,
            "program_run": prog_out,
        }

    update_research_campaign(data_dir, campaign_id, memory=memory_update.get("memory"))

    with ResearchProgramStore(data_dir) as store:
        campaign = store._load_campaign(campaign_id)
    if campaign and (campaign.get("memory") or {}).get("auto_run"):
        nxt = run_next_campaign_experiment(data_dir, campaign_id)
        if nxt.get("ok"):
            mem = dict((campaign.get("memory") or {}))
            mem["pending_job_id"] = (nxt.get("job") or {}).get("job_id")
            update_research_campaign(data_dir, campaign_id, memory=mem)
        return {"ok": True, "action": "auto_queued", "next": nxt}

    return {"ok": True, "action": "awaiting_next"}


def set_campaign_auto_run(data_dir: str, campaign_id: str, *, enabled: bool) -> dict[str, Any]:
    from .research_program import get_research_campaign, update_research_campaign

    campaign = get_research_campaign(data_dir, campaign_id)
    if not campaign:
        return {"ok": False, "error": "campaign not found"}
    memory = dict(campaign.get("memory") or {})
    memory["auto_run"] = bool(enabled)
    return update_research_campaign(data_dir, campaign_id, memory=memory)


def mark_campaign_validated(data_dir: str, campaign_id: str) -> dict[str, Any]:
    """Human gate — mark campaign validated when generalization passes."""
    from .campaign_report import save_campaign_report
    from .program_champion import refresh_program_champion_candidate
    from .research_program import get_research_campaign, update_research_campaign

    campaign = get_research_campaign(data_dir, campaign_id)
    if not campaign:
        return {"ok": False, "error": "campaign not found"}
    memory = campaign.get("memory") or {}
    gen = memory.get("best_generalization") or {}
    if int(gen.get("overall") or 0) < 70:
        return {
            "ok": False,
            "error": f"generalization score {gen.get('overall') or 0} below 70 — not ready to validate",
        }
    out = update_research_campaign(data_dir, campaign_id, status="validated")
    if not out.get("ok"):
        return out
    report_out = save_campaign_report(data_dir, campaign_id)
    pid = str(campaign.get("program_id") or "")
    champ_out = refresh_program_champion_candidate(data_dir, pid) if pid else None
    return {
        "ok": True,
        "campaign": out.get("campaign"),
        "report": report_out.get("report"),
        "champion_candidate": (champ_out or {}).get("champion", {}).get("candidate") if champ_out else None,
    }


def bootstrap_campaign_scheduler(data_dir: str, campaign_id: str) -> dict[str, Any]:
    """Called when campaign starts — seed proposals if baseline exists."""
    with ResearchProgramStore(data_dir) as store:
        config = store.resolve_campaign_config(campaign_id)
    if not config:
        return {"ok": False, "error": "campaign not found"}

    memory = config["campaign"].get("memory") or {}
    seeded = None
    if memory.get("baseline_research_report_id"):
        seeded = seed_proposals_from_report(data_dir, campaign_id)
    ranked = rank_campaign_proposals(data_dir, campaign_id)
    return {
        "ok": True,
        "seeded": seeded,
        "ranked_count": len(ranked.get("proposals") or []),
    }
