"""Campaign outcome summary — what the campaign produced."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .campaign_dashboard import get_campaign_dashboard
from .campaign_report import _conclusion
from .knowledge_pipeline import get_knowledge_pipeline_view


def _changes_tested(jobs: list[dict[str, Any]], templates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for job in jobs:
        if job.get("status") not in ("complete", "completed"):
            continue
        tmpl = templates.get(str(job.get("template_id") or "")) or {}
        changes = tmpl.get("accepted_changes") or []
        if len(changes) == 1:
            label = str((changes[0] or {}).get("text") or "1 change")
        elif changes:
            label = f"{len(changes)} bundled changes"
        else:
            label = "unknown change"
        counts[label] += 1
    return [{"change": k, "count": v} for k, v in counts.most_common()]


def _pf_from_jobs(jobs: list[dict[str, Any]]) -> tuple[float | None, float | None, int | None]:
    baseline_pf: float | None = None
    best_pf: float | None = None
    best_job_number: int | None = None
    for job in jobs:
        if job.get("status") not in ("complete", "completed"):
            continue
        comp = job.get("comparison") or {}
        bp = comp.get("baseline_pf") or comp.get("baseline_profit_factor")
        ap = comp.get("after_pf") or comp.get("after_profit_factor")
        if bp is not None and baseline_pf is None:
            baseline_pf = float(bp)
        if ap is not None and (best_pf is None or float(ap) > best_pf):
            best_pf = float(ap)
            best_job_number = int(job.get("job_number") or 0) or None
    return baseline_pf, best_pf, best_job_number


def _assess_question(
    *,
    research_question: str,
    changes: list[dict[str, Any]],
    baseline_pf: float | None,
    best_pf: float | None,
) -> str:
    q = str(research_question or "").lower()
    unique_changes = len(changes)
    dominant = changes[0]["count"] if changes else 0
    total = sum(c["count"] for c in changes) or 0

    if total == 0:
        return "No experiments completed yet — outcome pending."

    if unique_changes == 1 and total > 1 and dominant == total:
        if "optimal" in q or "percentage" in q:
            return (
                f"Partially answered: only one change was tested repeatedly "
                f"({changes[0]['change'][:48]}). Run distinct hypotheses to compare options."
            )
        return f"Single hypothesis tested {total} time(s) — result is directional, not a full sweep."

    if baseline_pf is not None and best_pf is not None:
        delta = best_pf - baseline_pf
        if delta > 0.02:
            return f"Positive outcome: PF improved {baseline_pf:.4f} → {best_pf:.4f} (+{delta:.4f})."
        if abs(delta) <= 0.02:
            return f"Neutral outcome: PF unchanged at ~{best_pf:.4f} — baseline may already include the change."
        return f"Negative outcome: PF declined {baseline_pf:.4f} → {best_pf:.4f} ({delta:+.4f})."

    return "Experiments completed — review verdicts and knowledge below."


def get_campaign_outcome(data_dir: str, campaign_id: str) -> dict[str, Any]:
    dash = get_campaign_dashboard(data_dir, campaign_id)
    if not dash.get("ok"):
        return dash

    campaign = dash.get("campaign") or {}
    memory = campaign.get("memory") or {}
    timeline = dash.get("timeline") or []
    budget = dash.get("budget_burn") or {}
    best_exp = dash.get("best_experiment") or {}
    best_job = best_exp.get("job") or {}
    gen = best_exp.get("generalization") or memory.get("best_generalization") or {}

    from .experiment_pipeline_store import ExperimentPipelineStore

    with ExperimentPipelineStore(data_dir) as store:
        jobs = store.list_jobs(campaign_id=campaign_id, limit=200)
        templates = {
            str(t.get("template_id") or ""): t
            for t in store.list_templates(campaign_id=campaign_id, limit=200)
        }

    changes = _changes_tested(jobs, templates)
    baseline_pf, best_pf, best_job_num = _pf_from_jobs(jobs)
    pf_delta = (best_pf - baseline_pf) if baseline_pf is not None and best_pf is not None else None

    verdicts = Counter(str(t.get("verdict") or "Unknown") for t in timeline)
    completed = sum(1 for j in jobs if j.get("status") in ("complete", "completed"))

    kb = get_knowledge_pipeline_view(data_dir, campaign_id=campaign_id, limit=30)
    from .kb_proposal_generator import get_knowledge_gaps_for_campaign

    gaps_out = get_knowledge_gaps_for_campaign(data_dir, campaign_id)
    knowledge = kb.get("knowledge") or []
    findings = kb.get("active_findings") or []
    gaps = (gaps_out.get("knowledge_gaps") or []) if gaps_out.get("ok") else []

    question = str(campaign.get("research_question") or "")
    assessment = _assess_question(
        research_question=question,
        changes=changes,
        baseline_pf=baseline_pf,
        best_pf=best_pf,
    )

    conclusion = _conclusion(
        research_question=question,
        best=best_job if best_job else (timeline[-1] if timeline else None),
        gen=gen,
        timeline=timeline,
    )

    recommendation = "Continue research"
    if campaign.get("status") == "validated":
        recommendation = "Validated — review program champion"
    elif budget.get("exhausted"):
        recommendation = "Budget exhausted — evaluate generalization, validate, or retire"
    elif int(gen.get("overall") or 0) >= 70:
        recommendation = "Ready for validation"
    elif pf_delta and pf_delta > 0.05:
        recommendation = "Promising PF gain — evaluate generalization"

    return {
        "ok": True,
        "campaign_id": campaign_id,
        "campaign_number": campaign.get("campaign_number"),
        "campaign_name": campaign.get("name"),
        "research_question": question,
        "status": campaign.get("status"),
        "executive_summary": {
            "conclusion": conclusion,
            "assessment": assessment,
            "recommendation": recommendation,
            "baseline_pf": baseline_pf,
            "best_pf": best_pf,
            "pf_delta": round(pf_delta, 4) if pf_delta is not None else None,
            "best_job_number": best_job_num or best_job.get("job_number"),
            "best_change": best_job.get("change_text") or (changes[0]["change"] if changes else None),
            "best_verdict": best_job.get("verdict"),
            "generalization": gen,
            "experiments_completed": completed,
            "experiments_budget": budget.get("experiments_limit"),
            "budget_exhausted": bool(budget.get("exhausted")),
        },
        "verdict_distribution": dict(verdicts),
        "changes_tested": changes,
        "knowledge_gained": [
            {"finding": k.get("finding"), "status": k.get("status"), "confidence": k.get("confidence")}
            for k in knowledge
        ],
        "findings": [
            {"finding": f.get("finding"), "status": f.get("status"), "evidence_count": f.get("evidence_count")}
            for f in findings[:8]
        ],
        "knowledge_gaps": gaps[:8],
        "timeline": timeline[-10:],
    }
