"""Phase D1 — Research Program and Campaign service layer."""

from __future__ import annotations

from typing import Any

from .research_objective import validate_research_question
from .research_program_store import ResearchProgramStore, _utc_now


def create_research_program(
    data_dir: str,
    *,
    name: str,
    description: str | None = None,
    importance: str = "medium",
    program_type: str = "strategy",
    objective: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
    stopping: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not str(name or "").strip():
        return {"ok": False, "error": "program name is required"}
    program = {
        "name": str(name).strip(),
        "description": description,
        "importance": importance,
        "program_type": program_type,
        "status": "active",
        "objective": objective,
        "budget": budget,
        "stopping": stopping,
    }
    with ResearchProgramStore(data_dir) as store:
        saved = store.save_program(program)
    return {"ok": True, "program": saved}


def get_research_program(data_dir: str, program_id: str) -> dict[str, Any] | None:
    with ResearchProgramStore(data_dir) as store:
        return store._load_program(program_id)


def list_research_programs(data_dir: str, *, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    with ResearchProgramStore(data_dir) as store:
        programs = store.list_programs(status=status, limit=limit)
        for prog in programs:
            prog["campaign_stats"] = store.count_campaigns_for_program(str(prog.get("program_id") or ""))
        return programs


def update_research_program(
    data_dir: str,
    program_id: str,
    *,
    name: str | None = None,
    description: str | None = None,
    importance: str | None = None,
    objective: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
    champion: dict[str, Any] | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    with ResearchProgramStore(data_dir) as store:
        program = store._load_program(program_id)
        if not program:
            return {"ok": False, "error": "program not found"}
        if name is not None:
            program["name"] = str(name).strip()
        if description is not None:
            program["description"] = description
        if importance is not None:
            program["importance"] = importance
        if objective is not None:
            program["objective"] = objective
        if budget is not None:
            program["budget"] = budget
        if champion is not None:
            program["champion"] = champion
        if status is not None:
            program["status"] = status
        saved = store.save_program(program)
    return {"ok": True, "program": saved}


def retire_research_program(
    data_dir: str,
    program_id: str,
    *,
    reason: str,
) -> dict[str, Any]:
    with ResearchProgramStore(data_dir) as store:
        program = store._load_program(program_id)
        if not program:
            return {"ok": False, "error": "program not found"}
        program["status"] = "retired"
        program["retired_reason"] = str(reason or "").strip()
        saved = store.save_program(program)
    return {"ok": True, "program": saved}


def create_research_campaign(
    data_dir: str,
    program_id: str,
    *,
    name: str,
    research_question: str,
    description: str | None = None,
    importance: str | None = None,
    hypothesis: str | None = None,
    objective: dict[str, Any] | None = None,
    budget: dict[str, Any] | None = None,
    stopping: dict[str, Any] | None = None,
    success_criteria: dict[str, Any] | None = None,
    failure_criteria: dict[str, Any] | None = None,
    dependencies: list[str] | None = None,
) -> dict[str, Any]:
    err = validate_research_question(name, research_question)
    if err:
        return {"ok": False, "error": err}
    with ResearchProgramStore(data_dir) as store:
        program = store._load_program(program_id)
        if not program:
            return {"ok": False, "error": "program not found"}
        if program.get("status") == "retired":
            return {"ok": False, "error": "cannot add campaigns to a retired program"}
        campaign = {
            "program_id": program_id,
            "name": str(name).strip(),
            "research_question": str(research_question).strip(),
            "hypothesis": hypothesis,
            "description": description,
            "importance": importance,
            "objective": objective,
            "budget": budget,
            "stopping": stopping,
            "success_criteria": success_criteria,
            "failure_criteria": failure_criteria,
            "dependencies": dependencies or [],
            "status": "created",
        }
        saved = store.save_campaign(campaign)
    return {"ok": True, "campaign": saved}


def get_research_campaign(data_dir: str, campaign_id: str) -> dict[str, Any] | None:
    with ResearchProgramStore(data_dir) as store:
        campaign = store._load_campaign(campaign_id)
        if not campaign:
            return None
        config = store.resolve_campaign_config(campaign_id)
        if config:
            campaign["resolved_objective"] = config.get("resolved_objective")
            campaign["resolved_budget"] = config.get("resolved_budget")
            campaign["resolved_stopping"] = config.get("resolved_stopping")
            campaign["resolved_importance"] = config.get("importance")
            campaign["program_name"] = (config.get("program") or {}).get("name")
        return campaign


def list_research_campaigns(
    data_dir: str,
    *,
    program_id: str | None = None,
    status: str | None = None,
    include_retired: bool = False,
    limit: int = 50,
) -> list[dict[str, Any]]:
    with ResearchProgramStore(data_dir) as store:
        return store.list_campaigns(
            program_id=program_id,
            status=status,
            include_retired=include_retired,
            limit=limit,
        )


def update_research_campaign(
    data_dir: str,
    campaign_id: str,
    **fields: Any,
) -> dict[str, Any]:
    with ResearchProgramStore(data_dir) as store:
        campaign = store._load_campaign(campaign_id)
        if not campaign:
            return {"ok": False, "error": "campaign not found"}
        for key in (
            "name", "research_question", "hypothesis", "description", "importance",
            "objective", "budget", "stopping", "success_criteria", "failure_criteria",
            "memory", "manifest", "dependencies", "status",
            "best_template_id", "retired_reason", "budget_used",
        ):
            if key in fields and fields[key] is not None:
                campaign[key] = fields[key]
        if "research_question" in fields or "name" in fields:
            err = validate_research_question(
                str(campaign.get("name") or ""),
                str(campaign.get("research_question") or ""),
            )
            if err:
                return {"ok": False, "error": err}
        saved = store.save_campaign(campaign)
    return {"ok": True, "campaign": saved}


def start_research_campaign(data_dir: str, campaign_id: str) -> dict[str, Any]:
    with ResearchProgramStore(data_dir) as store:
        campaign = store._load_campaign(campaign_id)
        if not campaign:
            return {"ok": False, "error": "campaign not found"}
        if campaign.get("status") not in ("created", "waiting"):
            return {"ok": False, "error": f"campaign cannot start from status {campaign.get('status')}"}
        deps = campaign.get("dependencies") or []
        for dep_id in deps:
            dep = store._load_campaign(str(dep_id))
            if not dep or dep.get("status") not in ("validated", "retired"):
                return {
                    "ok": False,
                    "error": f"dependency campaign {dep_id} not validated",
                }
        campaign["status"] = "running"
        campaign["started_at"] = campaign.get("started_at") or _utc_now()
        saved = store.save_campaign(campaign)
    from .campaign_scheduler import bootstrap_campaign_scheduler

    boot = bootstrap_campaign_scheduler(data_dir, campaign_id)
    return {"ok": True, "campaign": saved, "scheduler": boot}


def retire_research_campaign(
    data_dir: str,
    campaign_id: str,
    *,
    reason: str,
) -> dict[str, Any]:
    with ResearchProgramStore(data_dir) as store:
        campaign = store._load_campaign(campaign_id)
        if not campaign:
            return {"ok": False, "error": "campaign not found"}
        campaign["status"] = "retired"
        campaign["retired_reason"] = str(reason or "").strip()
        campaign["completed_at"] = _utc_now()
        saved = store.save_campaign(campaign)
    return {"ok": True, "campaign": saved}


def get_campaign_config(data_dir: str, campaign_id: str) -> dict[str, Any]:
    with ResearchProgramStore(data_dir) as store:
        config = store.resolve_campaign_config(campaign_id)
    if not config:
        return {"ok": False, "error": "campaign not found"}
    return {"ok": True, **config}
