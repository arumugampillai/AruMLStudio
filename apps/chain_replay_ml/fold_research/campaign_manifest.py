"""Campaign Manifest — primary checkpoint object for campaign state (Phase F1)."""

from __future__ import annotations

from typing import Any

from .research_program_store import ResearchProgramStore, _utc_now


def empty_manifest(campaign: dict[str, Any]) -> dict[str, Any]:
    return {
        "campaign_id": campaign.get("campaign_id"),
        "campaign_number": campaign.get("campaign_number"),
        "name": campaign.get("name"),
        "research_question": campaign.get("research_question"),
        "hypothesis": campaign.get("hypothesis"),
        "status": campaign.get("status") or "waiting",
        "current_job_id": None,
        "current_job_number": None,
        "completed_jobs": 0,
        "failed_jobs": 0,
        "evidence_count": 0,
        "knowledge_created": 0,
        "knowledge_updated": 0,
        "reports_count": 0,
        "confidence_pct": None,
        "winner": None,
        "resume_required": False,
        "resume_from_job_id": None,
        "last_checkpoint_at": None,
        "program_run_id": None,
        "model_id": None,
    }


def build_campaign_manifest(data_dir: str, campaign_id: str) -> dict[str, Any]:
    from .experiment_pipeline_store import ExperimentPipelineStore

    with ResearchProgramStore(data_dir) as store:
        campaign = store._load_campaign(campaign_id)
    if not campaign:
        return {"ok": False, "error": "campaign not found"}

    manifest = dict(campaign.get("manifest") or empty_manifest(campaign))
    manifest["campaign_id"] = campaign_id
    manifest["name"] = campaign.get("name")
    manifest["research_question"] = campaign.get("research_question")
    manifest["hypothesis"] = campaign.get("hypothesis")
    manifest["status"] = campaign.get("status")
    manifest["success_criteria"] = campaign.get("success_criteria")
    manifest["failure_criteria"] = campaign.get("failure_criteria")
    manifest["stopping"] = campaign.get("stopping")

    memory = campaign.get("memory") or {}
    manifest["program_run_id"] = memory.get("program_run_id") or manifest.get("program_run_id")
    manifest["model_id"] = memory.get("model_id") or manifest.get("model_id")

    with ExperimentPipelineStore(data_dir) as pipe:
        jobs = pipe.list_jobs(campaign_id=campaign_id, limit=500)

    completed = [j for j in jobs if j.get("status") in ("complete", "completed")]
    failed = [j for j in jobs if j.get("status") == "failed"]
    running = [j for j in jobs if j.get("status") == "running"]

    manifest["completed_jobs"] = len(completed)
    manifest["failed_jobs"] = len(failed)
    manifest["reports_count"] = len(completed)

    if running:
        rj = running[0]
        manifest["current_job_id"] = rj.get("job_id")
        manifest["current_job_number"] = rj.get("job_number")
        manifest["resume_required"] = True
        manifest["resume_from_job_id"] = rj.get("job_id")
    elif memory.get("pending_job_id"):
        manifest["resume_required"] = True
        manifest["resume_from_job_id"] = memory.get("pending_job_id")
        manifest["current_job_id"] = memory.get("pending_job_id")
    else:
        manifest["resume_required"] = False
        manifest["resume_from_job_id"] = None

    best_pf = memory.get("best_profit_factor")
    best_job_id = memory.get("best_job_id")
    if best_job_id:
        best_job = next((j for j in completed if str(j.get("job_id")) == str(best_job_id)), None)
        if best_job:
            comp = best_job.get("comparison") or {}
            tmpl_id = best_job.get("template_id")
            with ExperimentPipelineStore(data_dir) as pipe:
                tmpl = pipe._load_template(str(tmpl_id or "")) if tmpl_id else None
            changes = (tmpl or {}).get("accepted_changes") or []
            manifest["winner"] = {
                "job_id": best_job_id,
                "job_number": best_job.get("job_number"),
                "change_text": (changes[0] or {}).get("text") if changes else None,
                "pf": comp.get("after_pf") or best_pf,
                "pf_delta": comp.get("pf_delta"),
            }

    gen = memory.get("best_generalization") or {}
    if gen.get("overall") is not None:
        manifest["confidence_pct"] = int(gen.get("overall"))

    manifest["evidence_count"] = int(memory.get("experiments_run") or len(completed))
    manifest["knowledge_created"] = int(memory.get("knowledge_created") or 0)
    manifest["knowledge_updated"] = int(memory.get("knowledge_updated") or 0)
    manifest["last_checkpoint_at"] = memory.get("last_checkpoint_at") or campaign.get("updated_at")

    return {"ok": True, "manifest": manifest}


def save_campaign_manifest(data_dir: str, campaign_id: str, manifest: dict[str, Any]) -> dict[str, Any]:
    from .research_program import update_research_campaign

    manifest = dict(manifest)
    manifest["last_checkpoint_at"] = _utc_now()
    return update_research_campaign(data_dir, campaign_id, manifest=manifest)


def checkpoint_campaign_after_job(
    data_dir: str,
    campaign_id: str,
    *,
    job: dict[str, Any],
    knowledge_delta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist manifest + memory checkpoint immediately after each job."""
    from .research_program import get_research_campaign, update_research_campaign

    built = build_campaign_manifest(data_dir, campaign_id)
    if not built.get("ok"):
        return built
    manifest = built.get("manifest") or {}

    campaign = get_research_campaign(data_dir, campaign_id)
    memory = dict((campaign or {}).get("memory") or {})
    memory["last_checkpoint_at"] = _utc_now()
    memory["last_job_id"] = job.get("job_id")
    memory.pop("pending_job_id", None)
    memory.pop("active_job_id", None)

    if knowledge_delta:
        memory["knowledge_created"] = int(memory.get("knowledge_created") or 0) + int(
            knowledge_delta.get("created") or 0
        )
        memory["knowledge_updated"] = int(memory.get("knowledge_updated") or 0) + int(
            knowledge_delta.get("updated") or 0
        )

    update_research_campaign(data_dir, campaign_id, memory=memory, manifest=manifest)
    return {"ok": True, "manifest": manifest}
