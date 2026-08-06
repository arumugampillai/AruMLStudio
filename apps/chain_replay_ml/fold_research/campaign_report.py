"""Campaign Report — synthesized outcome document (Phase D5)."""

from __future__ import annotations

from collections import Counter
from typing import Any

from .campaign_dashboard import get_campaign_dashboard
from .research_program_store import ResearchProgramStore


def _utc_now() -> str:
    from .research_program_store import _utc_now as _now

    return _now()


def _conclusion(
    *,
    research_question: str,
    best: dict[str, Any] | None,
    gen: dict[str, Any] | None,
    timeline: list[dict[str, Any]],
) -> str:
    if not best:
        return f"Campaign explored «{research_question}» but no decisive winner was recorded."
    change = best.get("change_text") or "the tested hypothesis"
    pf = best.get("after_pf")
    verdict = best.get("verdict") or "—"
    gen_score = (gen or {}).get("overall")
    parts = [f"Best result: {change}"]
    if pf is not None:
        parts.append(f"PF {pf}")
    parts.append(f"verdict {verdict}")
    if gen_score is not None:
        parts.append(f"generalization {gen_score} {(gen or {}).get('label') or ''}".strip())
    if len(timeline) > 1:
        parts.append(f"across {len(timeline)} experiments")
    return " — ".join(parts) + "."


def build_campaign_report(data_dir: str, campaign_id: str) -> dict[str, Any]:
    dash = get_campaign_dashboard(data_dir, campaign_id)
    if not dash.get("ok"):
        return dash

    campaign = dash.get("campaign") or {}
    memory = campaign.get("memory") or {}
    timeline = dash.get("timeline") or []
    hypothesis_log = (dash.get("scheduler") or {}).get("cycle", {}).get("hypothesis_log") or memory.get("hypothesis_log") or []
    best_exp = dash.get("best_experiment") or {}
    best_job = best_exp.get("job") or {}
    gen = best_exp.get("generalization") or memory.get("best_generalization") or {}
    trend = dash.get("trend_summary") or {}
    budget = dash.get("budget_burn") or {}

    verdicts = Counter(str(t.get("verdict") or "Unknown") for t in timeline)

    recommendation = "Continue research"
    if campaign.get("status") == "validated":
        recommendation = "Campaign validated — promote best experiment to program champion review"
    elif int(gen.get("overall") or 0) >= 70:
        recommendation = "Ready for validation — generalization threshold met"
    elif best_job.get("verdict") in ("Strong Improvement", "Improvement"):
        recommendation = "Promising result — run generalization evaluation before validation"

    report = {
        "ok": True,
        "report_id": f"campaign_report_{campaign_id}",
        "campaign_id": campaign_id,
        "campaign_number": campaign.get("campaign_number"),
        "program_id": campaign.get("program_id"),
        "generated_at": _utc_now(),
        "executive_summary": {
            "campaign_name": campaign.get("name"),
            "research_question": campaign.get("research_question"),
            "status": campaign.get("status"),
            "conclusion": _conclusion(
                research_question=str(campaign.get("research_question") or ""),
                best=best_job,
                gen=gen,
                timeline=timeline,
            ),
            "recommendation": recommendation,
            "best_change": best_job.get("change_text"),
            "best_pf": best_job.get("after_pf") or memory.get("best_profit_factor"),
            "best_pf_delta": best_job.get("pf_delta"),
            "best_verdict": best_job.get("verdict"),
            "generalization": gen,
            "experiments_run": int(memory.get("experiments_run") or budget.get("experiments_used") or 0),
            "hypotheses_tested": len(hypothesis_log),
        },
        "timeline": timeline,
        "hypothesis_log": hypothesis_log,
        "metrics_trend": dash.get("metrics_trend") or [],
        "trend_summary": trend,
        "budget": budget,
        "best_experiment": best_exp,
        "verdict_distribution": dict(verdicts),
        "resolved_objective": dash.get("resolved_objective"),
        "dependencies": dash.get("dependencies") or [],
    }
    return report


def save_campaign_report(data_dir: str, campaign_id: str, report: dict[str, Any] | None = None) -> dict[str, Any]:
    from .research_program import update_research_campaign

    doc = report or build_campaign_report(data_dir, campaign_id)
    if not doc.get("ok"):
        return doc

    from .research_program import get_research_campaign

    campaign = get_research_campaign(data_dir, campaign_id)
    if not campaign:
        return {"ok": False, "error": "campaign not found"}
    memory = dict(campaign.get("memory") or {})
    memory["campaign_report"] = doc
    memory["campaign_report_generated_at"] = doc.get("generated_at")
    out = update_research_campaign(data_dir, campaign_id, memory=memory)
    if out.get("ok"):
        out["report"] = doc
    return out


def get_campaign_report(data_dir: str, campaign_id: str) -> dict[str, Any]:
    from .research_program import get_research_campaign

    campaign = get_research_campaign(data_dir, campaign_id)
    if not campaign:
        return {"ok": False, "error": "campaign not found"}
    memory = campaign.get("memory") or {}
    report = memory.get("campaign_report")
    if report:
        return {"ok": True, "report": report, "source": "memory"}
    if campaign.get("status") in ("validated", "retired"):
        built = build_campaign_report(data_dir, campaign_id)
        if built.get("ok"):
            return {"ok": True, "report": built, "source": "generated"}
    return {"ok": False, "error": "no campaign report available"}
