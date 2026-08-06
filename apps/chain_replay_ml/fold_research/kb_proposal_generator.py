"""KB-driven proposal generator — Phase E."""

from __future__ import annotations

import re
from typing import Any

from .campaign_proposal_generator import _item_matches_question, _question_tokens, _save_scored_proposal
from .experiment_pipeline import _baseline_from_report
from .experiment_score import compute_experiment_score
from .experiment_planner import build_experiment_planner_view
from .knowledge_retrieval import _findings_by_keys, _enrich_finding
from .knowledge_store import KnowledgeStore, list_knowledge_findings
from .research_program_store import ResearchProgramStore


def _planner_item_from_finding(finding: dict[str, Any]) -> dict[str, Any]:
    text = str(finding.get("finding") or "")
    category = str(finding.get("category") or "strategy")
    target = "strategy_registry"
    if category == "feature":
        target = "feature_registry"
    elif category == "model":
        target = "model_builder"
    return {
        "text": text,
        "target": target,
        "target_label": target,
        "accepted_default": True,
        "stars": 4 if finding.get("status") == "knowledge" else 3,
        "source": "knowledge_base",
        "finding_key": finding.get("finding_key"),
        "finding_status": finding.get("status"),
    }


def get_knowledge_gaps_for_campaign(data_dir: str, campaign_id: str) -> dict[str, Any]:
    with ResearchProgramStore(data_dir) as store:
        config = store.resolve_campaign_config(campaign_id)
    if not config:
        return {"ok": False, "error": "campaign not found"}

    question = str(config["campaign"].get("research_question") or "")
    tokens = _question_tokens(question)
    all_findings = list_knowledge_findings(data_dir, limit=100)
    enriched = [_enrich_finding(f) for f in all_findings]

    relevant = []
    for f in enriched:
        blob = f"{f.get('finding_key', '')} {f.get('finding', '')}".lower()
        if tokens and not any(tok in blob for tok in tokens):
            continue
        relevant.append(f)

    known_keys = {str(f.get("finding_key") or "") for f in relevant if f.get("status") in ("knowledge", "confirmed", "contradicted")}
    gaps: list[dict[str, Any]] = []

    for f in relevant:
        if f.get("status") == "knowledge":
            continue
        if f.get("status") == "contradicted":
            gaps.append({
                "type": "contradicted",
                "finding_key": f.get("finding_key"),
                "finding": f.get("finding"),
                "note": "Prior evidence contradicts — avoid or re-test",
            })
            continue
        if f.get("status") in ("candidate", "supported") and int(f.get("evidence_count") or 0) < 3:
            gaps.append({
                "type": "needs_evidence",
                "finding_key": f.get("finding_key"),
                "finding": f.get("finding"),
                "evidence_count": f.get("evidence_count"),
                "note": "Needs more experiments to confirm",
            })

    # Topic keywords from question not covered by knowledge
    covered_tokens: set[str] = set()
    for f in relevant:
        covered_tokens.update(re.findall(r"[a-z]{3,}", str(f.get("finding") or "").lower()))
    for tok in tokens:
        if tok not in covered_tokens and tok not in known_keys:
            gaps.append({
                "type": "unexplored",
                "topic": tok,
                "note": f"No knowledge yet for «{tok}» in this campaign scope",
            })

    return {
        "ok": True,
        "campaign_id": campaign_id,
        "research_question": question,
        "relevant_findings": relevant,
        "knowledge_gaps": gaps[:12],
        "known_count": len([f for f in relevant if f.get("status") == "knowledge"]),
    }


def seed_kb_driven_proposals(
    data_dir: str,
    campaign_id: str,
    *,
    limit: int = 4,
) -> dict[str, Any]:
    """Create proposals from KB gaps + report items not yet known."""
    from .research_report_store import get_research_report

    gaps_out = get_knowledge_gaps_for_campaign(data_dir, campaign_id)
    if not gaps_out.get("ok"):
        return gaps_out

    with ResearchProgramStore(data_dir) as store:
        config = store.resolve_campaign_config(campaign_id)
    if not config:
        return {"ok": False, "error": "campaign not found"}

    campaign = config["campaign"]
    program = config["program"]
    memory = dict(campaign.get("memory") or {})
    objective = config.get("resolved_objective") or {}
    importance = str(config.get("importance") or "medium")
    question = str(campaign.get("research_question") or "")
    tokens = _question_tokens(question)

    report_id = memory.get("baseline_research_report_id") or memory.get("last_research_report_id")
    if not report_id:
        return {"ok": False, "error": "attach a baseline research report first"}

    report = get_research_report(data_dir, str(report_id))
    if not report or not report.get("ok"):
        return {"ok": False, "error": "research report not found"}

    view = build_experiment_planner_view(report)
    available = view.get("items") or []
    baseline = _baseline_from_report(report)

    known_keys: set[str] = set()
    for f in gaps_out.get("relevant_findings") or []:
        if f.get("status") in ("knowledge", "confirmed", "contradicted"):
            known_keys.add(str(f.get("finding_key") or ""))

    created: list[dict[str, Any]] = []

    # Proposals from planner items that address knowledge gaps
    for item in available:
        if len(created) >= limit:
            break
        if not _item_matches_question(item, tokens):
            continue
        text = str(item.get("text") or "").lower()
        skip = False
        for f in gaps_out.get("relevant_findings") or []:
            if f.get("status") == "contradicted" and any(w in text for w in str(f.get("finding") or "").lower().split()[:3]):
                skip = True
                break
        if skip:
            continue

        selected = [dict(item, accepted_default=True)]
        goal = f"{question} — KB gap: {item.get('text', '')[:48]}"
        score = compute_experiment_score(data_dir, report, accepted_items=selected, goal=goal)
        score["kb_driven"] = {"source": "planner_gap", "knowledge_gaps": len(gaps_out.get("knowledge_gaps") or [])}
        proposal = {
            "status": "draft",
            "research_report_id": report.get("report_id"),
            "prediction_run_id": report.get("prediction_run_id"),
            "strategy_run_id": report.get("strategy_run_id"),
            "model_id": (report.get("executive_summary") or {}).get("model_id"),
            "strategy_label": (report.get("executive_summary") or {}).get("strategy"),
            "goal": goal,
            "tags": list(dict.fromkeys([*(score.get("tags") or []), "kb_driven", "campaign"])),
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

    # Proposals from findings that need more evidence
    for gap in gaps_out.get("knowledge_gaps") or []:
        if len(created) >= limit:
            break
        if gap.get("type") != "needs_evidence":
            continue
        key = str(gap.get("finding_key") or "")
        findings = _findings_by_keys(data_dir, {key})
        if not findings:
            continue
        item = _planner_item_from_finding(findings[0])
        selected = [item]
        goal = f"{question} — confirm: {item.get('text', '')[:48]}"
        score = compute_experiment_score(data_dir, report, accepted_items=selected, goal=goal)
        score["kb_driven"] = {"source": "needs_evidence", "finding_key": key}
        proposal = {
            "status": "draft",
            "research_report_id": report.get("report_id"),
            "prediction_run_id": report.get("prediction_run_id"),
            "strategy_run_id": report.get("strategy_run_id"),
            "model_id": (report.get("executive_summary") or {}).get("model_id"),
            "strategy_label": (report.get("executive_summary") or {}).get("strategy"),
            "goal": goal,
            "tags": ["kb_driven", "confirm_finding", "campaign"],
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

    return {
        "ok": True,
        "proposals": created,
        "count": len(created),
        "gaps": gaps_out.get("knowledge_gaps") or [],
    }
