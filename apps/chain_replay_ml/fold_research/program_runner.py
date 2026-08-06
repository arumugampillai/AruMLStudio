"""Program Runner — execute reusable research programs on a model (Phase F3)."""

from __future__ import annotations

from typing import Any

from .campaign_proposal_generator import attach_campaign_baseline
from .program_execution_store import ProgramExecutionStore, create_program_run
from .research_program_store import ResearchProgramStore


def _utc_now() -> str:
    from .research_program_store import _utc_now as _now

    return _now()


def _append_run_event(data_dir: str, run_id: str, action: str, **payload: Any) -> None:
    with ProgramExecutionStore(data_dir) as store:
        store.append_event(run_id, {"action": action, **payload})


def _clone_campaign_for_run(
    data_dir: str,
    *,
    source: dict[str, Any],
    program_run_id: str,
    model_id: str,
) -> dict[str, Any]:
    from .research_program import create_research_campaign

    out = create_research_campaign(
        data_dir,
        str(source.get("program_id") or ""),
        name=str(source.get("name") or "Campaign"),
        research_question=str(source.get("research_question") or ""),
        description=source.get("description"),
        importance=source.get("importance"),
        hypothesis=source.get("hypothesis"),
        objective=source.get("objective"),
        budget=source.get("budget"),
        stopping=source.get("stopping"),
        success_criteria=source.get("success_criteria"),
        failure_criteria=source.get("failure_criteria"),
        dependencies=[],
    )
    if not out.get("ok"):
        return out
    campaign = out.get("campaign") or {}
    cid = str(campaign.get("campaign_id") or "")
    from .research_program import update_research_campaign

    memory = dict(campaign.get("memory") or {})
    memory.update({
        "program_run_id": program_run_id,
        "model_id": model_id,
        "source_campaign_id": source.get("campaign_id"),
        "auto_run": True,
    })
    update_research_campaign(data_dir, cid, status="waiting", memory=memory)
    campaign["campaign_id"] = cid
    campaign["memory"] = memory
    campaign["status"] = "waiting"
    return {"ok": True, "campaign": campaign}


def start_program_on_model(
    data_dir: str,
    *,
    model_id: str,
    program_id: str,
    research_report_id: str | None = None,
    prediction_run_id: str | None = None,
    strategy_run_id: str | None = None,
    campaign_ids: list[str] | None = None,
) -> dict[str, Any]:
    """One-click: create program run, clone campaigns, attach baseline, start first campaign."""
    run_out = create_program_run(
        data_dir,
        model_id=model_id,
        program_id=program_id,
        prediction_run_id=prediction_run_id,
        strategy_run_id=strategy_run_id,
        research_report_id=research_report_id,
    )
    if not run_out.get("ok"):
        return run_out

    run = run_out.get("run") or {}
    run_id = str(run.get("run_id") or "")

    with ResearchProgramStore(data_dir) as store:
        sources = store.list_campaigns(program_id=program_id, include_retired=False, limit=100)

    if campaign_ids:
        allowed = {str(x) for x in campaign_ids}
        sources = [c for c in sources if str(c.get("campaign_id")) in allowed]

    if not sources:
        return {"ok": False, "error": "program has no campaigns to run"}

    run_campaigns: list[dict[str, Any]] = []
    for src in sources:
        cloned = _clone_campaign_for_run(
            data_dir,
            source=src,
            program_run_id=run_id,
            model_id=model_id,
        )
        if not cloned.get("ok"):
            return cloned
        camp = cloned.get("campaign") or {}
        if research_report_id:
            attach_campaign_baseline(data_dir, str(camp.get("campaign_id")), research_report_id=research_report_id)
        run_campaigns.append({
            "campaign_id": camp.get("campaign_id"),
            "name": camp.get("name"),
            "status": "waiting",
            "source_campaign_id": src.get("campaign_id"),
        })

    manifest = dict(run.get("manifest") or {})
    manifest["campaigns"] = run_campaigns
    manifest["total_campaigns"] = len(run_campaigns)
    manifest["completed_campaigns"] = 0
    checkpoint = {"current_campaign_index": 0, "current_campaign_id": run_campaigns[0]["campaign_id"]}

    with ProgramExecutionStore(data_dir) as store:
        run = store.save_run({
            **run,
            "status": "running",
            "started_at": _utc_now(),
            "manifest": manifest,
            "checkpoint": checkpoint,
        })

    _append_run_event(
        data_dir,
        run_id,
        "program_run_started",
        model_id=model_id,
        program_id=program_id,
        campaign_count=len(run_campaigns),
    )

    first = _start_campaign_at_index(data_dir, run_id, index=0)
    return {"ok": True, "run": run, "first_campaign": first}


def _start_campaign_at_index(data_dir: str, run_id: str, *, index: int) -> dict[str, Any]:
    from .research_program import start_research_campaign, update_research_campaign

    with ProgramExecutionStore(data_dir) as store:
        run = store.get_run(run_id)
    if not run:
        return {"ok": False, "error": "program run not found"}

    campaigns = (run.get("manifest") or {}).get("campaigns") or []
    if index >= len(campaigns):
        return _complete_program_run(data_dir, run_id)

    entry = campaigns[index]
    cid = str(entry.get("campaign_id") or "")
    start_out = start_research_campaign(data_dir, cid)
    if not start_out.get("ok"):
        return start_out

    entry["status"] = "running"
    manifest = dict(run.get("manifest") or {})
    manifest["campaigns"] = campaigns
    checkpoint = dict(run.get("checkpoint") or {})
    checkpoint.update({
        "current_campaign_index": index,
        "current_campaign_id": cid,
    })

    with ProgramExecutionStore(data_dir) as store:
        store.save_run({
            **run,
            "manifest": manifest,
            "checkpoint": checkpoint,
            "status": "running",
        })

    _append_run_event(data_dir, run_id, "campaign_started", campaign_id=cid, index=index)
    return {"ok": True, "campaign_id": cid, "index": index, "scheduler": start_out.get("scheduler")}


def _complete_program_run(data_dir: str, run_id: str) -> dict[str, Any]:
    from .research_portfolio_report import build_model_research_portfolio_report

    with ProgramExecutionStore(data_dir) as store:
        run = store.get_run(run_id)
    if not run:
        return {"ok": False, "error": "program run not found"}

    model_id = str(run.get("model_id") or "")
    portfolio = build_model_research_portfolio_report(data_dir, model_id)

    with ProgramExecutionStore(data_dir) as store:
        run = store.save_run({
            **run,
            "status": "completed",
            "completed_at": _utc_now(),
            "summary": portfolio.get("report"),
        })

    _append_run_event(data_dir, run_id, "program_run_completed", portfolio=portfolio.get("report"))
    return {"ok": True, "action": "program_completed", "run": run, "portfolio": portfolio}


def on_program_campaign_complete(data_dir: str, campaign_id: str) -> dict[str, Any]:
    """Advance program run when a run-scoped campaign finishes."""
    from .research_program import get_research_campaign, update_research_campaign

    campaign = get_research_campaign(data_dir, campaign_id)
    if not campaign:
        return {"ok": False, "error": "campaign not found"}
    memory = campaign.get("memory") or {}
    run_id = str(memory.get("program_run_id") or "")
    if not run_id:
        return {"ok": False, "error": "campaign not part of a program run"}

    update_research_campaign(data_dir, campaign_id, status="completed")

    with ProgramExecutionStore(data_dir) as store:
        run = store.get_run(run_id)
    if not run:
        return {"ok": False, "error": "program run not found"}

    manifest = dict(run.get("manifest") or {})
    campaigns = list(manifest.get("campaigns") or [])
    idx = None
    for i, c in enumerate(campaigns):
        if str(c.get("campaign_id")) == campaign_id:
            c["status"] = "completed"
            idx = i
            break

    manifest["completed_campaigns"] = int(manifest.get("completed_campaigns") or 0) + 1
    manifest["campaigns"] = campaigns

    with ProgramExecutionStore(data_dir) as store:
        store.save_run({**run, "manifest": manifest})

    _append_run_event(data_dir, run_id, "campaign_completed", campaign_id=campaign_id)

    next_idx = (idx + 1) if idx is not None else int((run.get("checkpoint") or {}).get("current_campaign_index") or 0) + 1
    if next_idx >= len(campaigns):
        return _complete_program_run(data_dir, run_id)
    return _start_campaign_at_index(data_dir, run_id, index=next_idx)


def resume_program_run(data_dir: str, run_id: str) -> dict[str, Any]:
    """Resume after power failure — campaign manifest → job."""
    from .campaign_manifest import build_campaign_manifest
    from .research_program import start_research_campaign, update_research_campaign

    with ProgramExecutionStore(data_dir) as store:
        run = store.get_run(run_id)
    if not run:
        return {"ok": False, "error": "program run not found"}

    checkpoint = dict(run.get("checkpoint") or {})
    idx = int(checkpoint.get("current_campaign_index") or 0)
    campaigns = (run.get("manifest") or {}).get("campaigns") or []
    if idx >= len(campaigns):
        return _complete_program_run(data_dir, run_id)

    cid = str(campaigns[idx].get("campaign_id") or checkpoint.get("current_campaign_id") or "")
    if not cid:
        return {"ok": False, "error": "no campaign to resume"}

    manifest_out = build_campaign_manifest(data_dir, cid)
    manifest = manifest_out.get("manifest") or {}

    _append_run_event(data_dir, run_id, "program_run_resumed", campaign_id=cid, manifest=manifest)

    if manifest.get("resume_required") and manifest.get("resume_from_job_id"):
        from .experiment_pipeline import execute_job_pipeline

        job_id = str(manifest.get("resume_from_job_id"))
        job_out = execute_job_pipeline(data_dir, job_id)
        return {"ok": True, "action": "resume_job", "job_id": job_id, "job": job_out}

    campaign = campaigns[idx]
    status = str(campaign.get("status") or "waiting")
    if status in ("waiting", "created", "paused"):
        start_out = start_research_campaign(data_dir, cid)
        if not start_out.get("ok"):
            return start_out
        campaign["status"] = "running"
    elif status == "running":
        from .campaign_scheduler import run_next_campaign_experiment

        nxt = run_next_campaign_experiment(data_dir, cid)
        if nxt.get("ok"):
            mem_update = {}
            from .research_program import get_research_campaign as _get

            c = _get(data_dir, cid)
            if c:
                mem = dict(c.get("memory") or {})
                mem["pending_job_id"] = (nxt.get("job") or {}).get("job_id")
                mem_update["memory"] = mem
            update_research_campaign(data_dir, cid, **mem_update)
        return {"ok": True, "action": "resume_campaign", "next": nxt}

    manifest_run = dict(run.get("manifest") or {})
    manifest_run["campaigns"] = campaigns
    with ProgramExecutionStore(data_dir) as store:
        store.save_run({**run, "status": "running", "manifest": manifest_run})

    return {"ok": True, "action": "campaign_resumed", "campaign_id": cid, "index": idx}


def get_program_run_view(data_dir: str, run_id: str) -> dict[str, Any]:
    with ProgramExecutionStore(data_dir) as store:
        run = store.get_run(run_id)
    if not run:
        return {"ok": False, "error": "program run not found"}
    return {"ok": True, "run": run}
