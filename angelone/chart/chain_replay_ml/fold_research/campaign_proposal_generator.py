"""Campaign Proposal Generator — Phase D2."""

from __future__ import annotations

import re
from typing import Any

from .experiment_pipeline_store import ExperimentPipelineStore
from .experiment_score import compute_experiment_score
from .objective_score import compute_objective_score
from .research_program_store import ResearchProgramStore


def _question_tokens(question: str) -> set[str]:
    words = re.findall(r"[a-z]{3,}", str(question or "").lower())
    stop = {"what", "the", "does", "how", "is", "are", "for", "and", "with", "from", "this", "that"}
    return {w for w in words if w not in stop}


def _item_matches_question(item: dict[str, Any], tokens: set[str]) -> bool:
    if not tokens:
        return True
    text = str(item.get("text") or "").lower()
    return any(tok in text for tok in tokens)


def _single_change_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [dict(item, accepted_default=True) for item in items]


def attach_campaign_baseline(
    data_dir: str,
    campaign_id: str,
    *,
    research_report_id: str,
) -> dict[str, Any]:
    from .research_program import update_research_campaign

    from .research_report_store import get_research_report, resolve_research_report_id

    resolved_id = resolve_research_report_id(data_dir, research_report_id)
    if not resolved_id:
        return {"ok": False, "error": "research report not found — use full ID or unique prefix"}
    report = get_research_report(data_dir, resolved_id)
    if not report or not report.get("ok"):
        return {"ok": False, "error": "research report not found"}
    exec_sum = report.get("executive_summary") or {}
    memory = {
        "baseline_research_report_id": resolved_id,
        "baseline_prediction_run_id": report.get("prediction_run_id"),
        "baseline_strategy_run_id": report.get("strategy_run_id"),
        "baseline_model_id": exec_sum.get("model_id"),
        "baseline_strategy_label": exec_sum.get("strategy"),
    }
    return update_research_campaign(data_dir, campaign_id, memory=memory)


def _save_scored_proposal(
    data_dir: str,
    *,
    campaign_id: str,
    program_id: str,
    proposal: dict[str, Any],
    objective: dict[str, Any],
    importance: str,
    memory: dict[str, Any],
) -> dict[str, Any]:
    obj_score = compute_objective_score(
        data_dir,
        proposal=proposal,
        objective=objective,
        importance=importance,
        campaign_memory=memory,
    )
    proposal["campaign_id"] = campaign_id
    proposal["program_id"] = program_id
    proposal["objective_score"] = obj_score
    with ExperimentPipelineStore(data_dir) as store:
        saved = store.save_proposal(proposal)
    return saved


def seed_proposals_from_report(
    data_dir: str,
    campaign_id: str,
    *,
    research_report_id: str | None = None,
    single_change_only: bool = True,
    limit: int = 8,
) -> dict[str, Any]:
    from .experiment_planner import build_experiment_planner_view
    from .experiment_pipeline import _baseline_from_report
    from .research_report_store import get_research_report

    with ResearchProgramStore(data_dir) as store:
        config = store.resolve_campaign_config(campaign_id)
    if not config:
        return {"ok": False, "error": "campaign not found"}

    campaign = config["campaign"]
    program = config["program"]
    memory = dict(campaign.get("memory") or {})
    from .research_cycle import get_cycle_view

    if single_change_only:
        single_change_only = get_cycle_view(memory).get("exploration_stage") == "explore"
    report_id = research_report_id or memory.get("baseline_research_report_id")
    if not report_id:
        return {"ok": False, "error": "attach a baseline research report first"}

    report = get_research_report(data_dir, str(report_id))
    if not report or not report.get("ok"):
        return {"ok": False, "error": "research report not found"}

    view = build_experiment_planner_view(report)
    available = view.get("items") or []
    question = str(campaign.get("research_question") or "")
    tokens = _question_tokens(question)
    matched = [i for i in available if _item_matches_question(i, tokens)]
    if not matched:
        matched = available

    created: list[dict[str, Any]] = []
    objective = config.get("resolved_objective") or {}
    importance = str(config.get("importance") or "medium")
    baseline = _baseline_from_report(report)
    goal_prefix = question or view.get("suggested_goal") or ""

    if single_change_only:
        for item in matched[:limit]:
            selected = _single_change_items([item])
            goal = f"{goal_prefix} — {item.get('text', '')[:48]}"
            score = compute_experiment_score(data_dir, report, accepted_items=selected, goal=goal)
            proposal = {
                "status": "draft",
                "research_report_id": report.get("report_id"),
                "prediction_run_id": report.get("prediction_run_id"),
                "strategy_run_id": report.get("strategy_run_id"),
                "model_id": (report.get("executive_summary") or {}).get("model_id"),
                "strategy_label": (report.get("executive_summary") or {}).get("strategy"),
                "goal": goal,
                "tags": list(dict.fromkeys([*(score.get("tags") or []), "campaign", "seed"])),
                "available_recommendations": available,
                "selected_recommendations": selected,
                "baseline": baseline,
                "score": score,
            }
            saved = _save_scored_proposal(
                data_dir,
                campaign_id=campaign_id,
                program_id=str(program.get("program_id") or ""),
                proposal=proposal,
                objective=objective,
                importance=importance,
                memory=memory,
            )
            created.append(saved)
    else:
        selected = [i for i in matched if i.get("accepted_default")] or matched[:3]
        goal = goal_prefix or view.get("suggested_goal") or ""
        score = compute_experiment_score(data_dir, report, accepted_items=selected, goal=goal)
        proposal = {
            "status": "draft",
            "research_report_id": report.get("report_id"),
            "prediction_run_id": report.get("prediction_run_id"),
            "strategy_run_id": report.get("strategy_run_id"),
            "model_id": (report.get("executive_summary") or {}).get("model_id"),
            "strategy_label": (report.get("executive_summary") or {}).get("strategy"),
            "goal": goal,
            "tags": score.get("tags") or [],
            "available_recommendations": available,
            "selected_recommendations": selected,
            "baseline": baseline,
            "score": score,
        }
        saved = _save_scored_proposal(
            data_dir,
            campaign_id=campaign_id,
            program_id=str(program.get("program_id") or ""),
            proposal=proposal,
            objective=objective,
            importance=importance,
            memory=memory,
        )
        created.append(saved)

    return {"ok": True, "proposals": created, "count": len(created)}


def seed_proposals_from_job(
    data_dir: str,
    campaign_id: str,
    job_id: str,
    *,
    limit: int = 6,
) -> dict[str, Any]:
    from .experiment_pipeline import create_proposal_from_suggestion

    with ExperimentPipelineStore(data_dir) as store:
        job = store._load_job(job_id)
        if not job:
            return {"ok": False, "error": "job not found"}
        template = store._load_template(str(job.get("template_id") or ""))
        if not template:
            return {"ok": False, "error": "template not found"}

    results = job.get("results") or {}
    suggestions = results.get("next_experiments") or []
    if not suggestions:
        return {"ok": False, "error": "job has no follow-up suggestions"}

    with ResearchProgramStore(data_dir) as store:
        config = store.resolve_campaign_config(campaign_id)
    if not config:
        return {"ok": False, "error": "campaign not found"}

    objective = config.get("resolved_objective") or {}
    importance = str(config.get("importance") or "medium")
    memory = dict(config["campaign"].get("memory") or {})
    program_id = str(config["program"].get("program_id") or "")
    template_id = str(template.get("template_id") or "")

    created: list[dict[str, Any]] = []
    for suggestion in suggestions[:limit]:
        out = create_proposal_from_suggestion(
            data_dir,
            template_id,
            suggestion,
            source_job_id=job_id,
        )
        if not out.get("ok"):
            continue
        proposal = out.get("proposal") or {}
        saved = _save_scored_proposal(
            data_dir,
            campaign_id=campaign_id,
            program_id=program_id,
            proposal=proposal,
            objective=objective,
            importance=importance,
            memory=memory,
        )
        created.append(saved)

    return {"ok": True, "proposals": created, "count": len(created)}


def rank_campaign_proposals(data_dir: str, campaign_id: str) -> dict[str, Any]:
    with ResearchProgramStore(data_dir) as store:
        config = store.resolve_campaign_config(campaign_id)
    if not config:
        return {"ok": False, "error": "campaign not found"}

    objective = config.get("resolved_objective") or {}
    importance = str(config.get("importance") or "medium")
    memory = dict(config["campaign"].get("memory") or {})

    with ExperimentPipelineStore(data_dir) as store:
        proposals = store.list_proposals(campaign_id=campaign_id, status="draft", limit=100)

    ranked: list[dict[str, Any]] = []
    for proposal in proposals:
        obj_score = compute_objective_score(
            data_dir,
            proposal=proposal,
            objective=objective,
            importance=importance,
            campaign_memory=memory,
        )
        proposal["objective_score"] = obj_score
        with ExperimentPipelineStore(data_dir) as store:
            store.save_proposal(proposal)
        ranked.append({**proposal, "objective_score": obj_score})

    ranked.sort(key=lambda p: int((p.get("objective_score") or {}).get("overall") or 0), reverse=True)
    return {"ok": True, "proposals": ranked, "top": ranked[0] if ranked else None}
