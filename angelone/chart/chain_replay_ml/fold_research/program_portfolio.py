"""Program Portfolio — cross-program research overview (Phase D4)."""

from __future__ import annotations

from typing import Any

from .campaign_scheduler import check_campaign_budget
from .experiment_pipeline_store import ExperimentPipelineStore
from .research_cycle import get_cycle_view, infer_exploration_stage
from .research_program_store import ResearchProgramStore

IMPORTANCE_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _budget_pct(used: dict[str, Any], budget: dict[str, Any]) -> float | None:
    max_exp = budget.get("max_experiments")
    if max_exp is None:
        return None
    try:
        return round(int(used.get("experiments") or 0) / max(int(max_exp), 1) * 100.0, 1)
    except (TypeError, ValueError):
        return None


def _campaign_card(
    data_dir: str,
    campaign: dict[str, Any],
    *,
    program_name: str = "",
    program_importance: str = "medium",
) -> dict[str, Any]:
    cid = str(campaign.get("campaign_id") or "")
    memory = campaign.get("memory") or {}
    cycle = get_cycle_view(memory)
    budget_check = check_campaign_budget(data_dir, cid)

    with ExperimentPipelineStore(data_dir) as store:
        running = store.get_running_job_for_campaign(cid)
        top_proposals = store.list_proposals(campaign_id=cid, status="draft", limit=1)

    top_score = None
    if top_proposals:
        top_score = int((top_proposals[0].get("objective_score") or {}).get("overall") or 0)

    gen = memory.get("best_generalization") or {}
    resolved_budget = budget_check.get("budget") or campaign.get("budget") or {}
    used = budget_check.get("used") or campaign.get("budget_used") or {}

    return {
        "campaign_id": cid,
        "campaign_number": campaign.get("campaign_number"),
        "program_id": campaign.get("program_id"),
        "program_name": program_name,
        "name": campaign.get("name"),
        "research_question": campaign.get("research_question"),
        "status": campaign.get("status"),
        "importance": campaign.get("importance") or program_importance,
        "exploration_stage": cycle.get("exploration_stage") or infer_exploration_stage(memory),
        "cycle_step": cycle.get("current_step"),
        "experiments_run": int(memory.get("experiments_run") or used.get("experiments") or 0),
        "budget_pct": _budget_pct(used, resolved_budget),
        "budget_exhausted": bool(budget_check.get("exhausted")),
        "best_profit_factor": memory.get("best_profit_factor"),
        "best_job_id": memory.get("best_job_id"),
        "generalization": gen,
        "validation_ready": bool(cycle.get("validation_ready")),
        "auto_run": bool(memory.get("auto_run")),
        "job_running": bool(running),
        "top_objective_score": top_score,
        "updated_at": campaign.get("updated_at"),
    }


def _aggregate_campaign_stats(cards: list[dict[str, Any]]) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    total_experiments = 0
    running = 0
    validated = 0
    validation_ready = 0
    for card in cards:
        status = str(card.get("status") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        total_experiments += int(card.get("experiments_run") or 0)
        if card.get("job_running"):
            running += 1
        if status == "validated":
            validated += 1
        if card.get("validation_ready"):
            validation_ready += 1
    return {
        "total_campaigns": len(cards),
        "by_status": by_status,
        "total_experiments": total_experiments,
        "running_jobs": running,
        "validated": validated,
        "validation_ready": validation_ready,
    }


def get_program_portfolio(data_dir: str, program_id: str) -> dict[str, Any]:
    with ResearchProgramStore(data_dir) as store:
        program = store._load_program(program_id)
        if not program:
            return {"ok": False, "error": "program not found"}
        campaigns = store.list_campaigns(program_id=program_id, include_retired=True, limit=100)

    cards = [
        _campaign_card(
            data_dir,
            c,
            program_name=str(program.get("name") or ""),
            program_importance=str(program.get("importance") or "medium"),
        )
        for c in campaigns
    ]
    cards.sort(key=lambda c: (IMPORTANCE_RANK.get(str(c.get("importance") or "medium"), 9), c.get("campaign_number") or 0))

    active = [c for c in cards if c.get("status") not in ("retired",)]
    stats = _aggregate_campaign_stats(active)

    best_gen = None
    for card in cards:
        gen = card.get("generalization") or {}
        if gen.get("overall") is None:
            continue
        if best_gen is None or int(gen.get("overall") or 0) > int(best_gen.get("overall") or 0):
            best_gen = {**gen, "campaign_name": card.get("name"), "campaign_id": card.get("campaign_id")}

    return {
        "ok": True,
        "program": program,
        "campaigns": cards,
        "active_campaigns": active,
        "stats": stats,
        "best_generalization": best_gen,
    }


def get_research_portfolio(data_dir: str, *, limit: int = 50) -> dict[str, Any]:
    """Desk-wide portfolio — all programs ranked by importance."""
    with ResearchProgramStore(data_dir) as store:
        programs = store.list_programs(status=None, limit=limit)

    program_rows: list[dict[str, Any]] = []
    all_cards: list[dict[str, Any]] = []

    for program in programs:
        pid = str(program.get("program_id") or "")
        port = get_program_portfolio(data_dir, pid)
        if not port.get("ok"):
            continue
        cards = port.get("campaigns") or []
        all_cards.extend(cards)
        program_rows.append({
            "program_id": pid,
            "program_number": program.get("program_number"),
            "name": program.get("name"),
            "status": program.get("status"),
            "importance": program.get("importance"),
            "stats": port.get("stats"),
            "campaign_count": len(cards),
            "active_campaign_count": len(port.get("active_campaigns") or []),
            "best_generalization": port.get("best_generalization"),
        })

    program_rows.sort(key=lambda p: IMPORTANCE_RANK.get(str(p.get("importance") or "medium"), 9))

    with ExperimentPipelineStore(data_dir) as store:
        global_running = len(store.list_running_jobs())

    active_cards = [c for c in all_cards if c.get("status") not in ("retired",)]
    global_stats = _aggregate_campaign_stats(active_cards)
    global_stats["global_running_jobs"] = global_running

    priority_queue = sorted(
        [c for c in active_cards if c.get("status") == "running"],
        key=lambda c: (
            IMPORTANCE_RANK.get(str(c.get("importance") or "medium"), 9),
            -(int(c.get("top_objective_score") or 0)),
        ),
    )

    return {
        "ok": True,
        "programs": program_rows,
        "global_stats": global_stats,
        "priority_queue": priority_queue[:10],
        "campaigns": active_cards,
    }
