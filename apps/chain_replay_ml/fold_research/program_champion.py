"""Program Champion — candidate selection and human approval (Phase D5)."""

from __future__ import annotations

from typing import Any

from .campaign_report import build_campaign_report
from .experiment_pipeline_store import ExperimentPipelineStore
from .program_portfolio import get_program_portfolio
from .research_program_store import ResearchProgramStore


def _utc_now() -> str:
    from .research_program_store import _utc_now as _now

    return _now()


def _num(v: Any) -> float | None:
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def _candidate_score(*, gen_overall: int, best_pf: float | None, experiments: int) -> int:
    pf_part = min(100.0, (_num(best_pf) or 0) * 30.0) if best_pf else 0
    exp_part = min(25.0, experiments * 3.0)
    return int(round(gen_overall * 0.55 + pf_part * 0.30 + exp_part * 0.15))


def _experiment_refs(data_dir: str, job_id: str | None, template_id: str | None) -> dict[str, Any]:
    refs: dict[str, Any] = {
        "job_id": job_id,
        "template_id": template_id,
    }
    if not job_id:
        return refs
    with ExperimentPipelineStore(data_dir) as store:
        job = store._load_job(str(job_id))
        if job:
            outputs = job.get("outputs") or {}
            refs.update({
                "job_number": job.get("job_number"),
                "strategy_version_id": outputs.get("strategy_version_id"),
                "strategy_run_id": outputs.get("strategy_run_id"),
                "prediction_run_id": outputs.get("prediction_run_id"),
                "research_report_id": outputs.get("research_report_id"),
                "model_name": outputs.get("model_name"),
            })
        if template_id:
            tmpl = store._load_template(str(template_id))
            if tmpl:
                refs["template_number"] = tmpl.get("template_number")
                refs["goal"] = tmpl.get("goal")
                refs["model_id"] = tmpl.get("model_id")
                refs["strategy_label"] = tmpl.get("strategy_label")
                changes = tmpl.get("accepted_changes") or []
                if changes:
                    refs["change_text"] = changes[0].get("text")
    return refs


def _build_candidate_from_campaign(
    data_dir: str,
    campaign: dict[str, Any],
    *,
    program_id: str,
    program_name: str,
) -> dict[str, Any] | None:
    memory = campaign.get("memory") or {}
    gen = memory.get("best_generalization") or {}
    gen_overall = int(gen.get("overall") or 0)
    if campaign.get("status") not in ("validated",) and gen_overall < 70:
        return None

    best_pf = _num(memory.get("best_profit_factor"))
    experiments = int(memory.get("experiments_run") or 0)
    job_id = str(memory.get("best_job_id") or "")
    template_id = str(memory.get("best_template_id") or "")
    refs = _experiment_refs(data_dir, job_id or None, template_id or None)

    objective_score = None
    with ExperimentPipelineStore(data_dir) as store:
        if template_id:
            tmpl = store._load_template(template_id)
            if tmpl:
                objective_score = (tmpl.get("objective_score") or {}).get("overall")

    return {
        "campaign_id": campaign.get("campaign_id"),
        "campaign_number": campaign.get("campaign_number"),
        "campaign_name": campaign.get("name"),
        "research_question": campaign.get("research_question"),
        "program_id": program_id,
        "program_name": program_name,
        "status": campaign.get("status"),
        "generalization": gen,
        "best_profit_factor": best_pf,
        "experiments_run": experiments,
        "objective_score": objective_score,
        "composite_score": _candidate_score(gen_overall=gen_overall, best_pf=best_pf, experiments=experiments),
        "evidence_label": (
            "Very High" if experiments >= 10 else
            "High" if experiments >= 5 else
            "Medium" if experiments >= 2 else
            "Low"
        ),
        "recommendation": "Promote Strategy" if gen_overall >= 70 else "Review",
        "refs": refs,
        "recommended_at": _utc_now(),
    }


def build_candidate_program_champion(data_dir: str, program_id: str) -> dict[str, Any]:
    """Pick best validated campaign experiment as program champion candidate."""
    with ResearchProgramStore(data_dir) as store:
        program = store._load_program(program_id)
        if not program:
            return {"ok": False, "error": "program not found"}
        campaigns = store.list_campaigns(program_id=program_id, include_retired=True, limit=100)

    candidates: list[dict[str, Any]] = []
    for camp in campaigns:
        if camp.get("status") not in ("validated", "running", "retired"):
            continue
        cand = _build_candidate_from_campaign(
            data_dir,
            camp,
            program_id=program_id,
            program_name=str(program.get("name") or ""),
        )
        if cand:
            candidates.append(cand)

    if not candidates:
        return {"ok": True, "program_id": program_id, "candidate": None, "note": "no eligible campaigns"}

    candidates.sort(
        key=lambda c: (
            -int(c.get("composite_score") or 0),
            -int((c.get("generalization") or {}).get("overall") or 0),
        ),
    )
    best = candidates[0]
    return {"ok": True, "program_id": program_id, "candidate": best, "alternates": candidates[1:3]}


def refresh_program_champion_candidate(data_dir: str, program_id: str) -> dict[str, Any]:
    from .research_program import get_research_program, update_research_program

    program = get_research_program(data_dir, program_id)
    if not program:
        return {"ok": False, "error": "program not found"}

    champion = dict(program.get("champion") or {})
    if champion.get("approved"):
        return {"ok": True, "action": "already_approved", "champion": champion}

    built = build_candidate_program_champion(data_dir, program_id)
    if not built.get("ok"):
        return built

    candidate = built.get("candidate")
    if not candidate:
        champion.pop("candidate", None)
    else:
        existing = champion.get("candidate") or {}
        if int(candidate.get("composite_score") or 0) >= int(existing.get("composite_score") or 0):
            champion["candidate"] = candidate
    champion["updated_at"] = _utc_now()
    return update_research_program(data_dir, program_id, champion=champion)


def get_program_champion_view(data_dir: str, program_id: str) -> dict[str, Any]:
    from .research_program import get_research_program

    program = get_research_program(data_dir, program_id)
    if not program:
        return {"ok": False, "error": "program not found"}

    champion = program.get("champion") or {}
    candidate = champion.get("candidate")
    approved = champion.get("approved")

    portfolio = get_program_portfolio(data_dir, program_id)
    validated = [c for c in (portfolio.get("campaigns") or []) if c.get("status") == "validated"]

    return {
        "ok": True,
        "program_id": program_id,
        "program_name": program.get("name"),
        "champion": champion,
        "candidate": candidate,
        "approved": approved,
        "has_approved_champion": bool(approved),
        "validated_campaigns": len(validated),
        "can_approve": bool(candidate) and not approved,
    }


def approve_program_champion(
    data_dir: str,
    program_id: str,
    *,
    note: str | None = None,
) -> dict[str, Any]:
    """Human gate — never called automatically."""
    from .research_program import get_research_program, update_research_program

    program = get_research_program(data_dir, program_id)
    if not program:
        return {"ok": False, "error": "program not found"}

    champion = dict(program.get("champion") or {})
    candidate = champion.get("candidate")
    if not candidate:
        built = build_candidate_program_champion(data_dir, program_id)
        candidate = (built.get("candidate") if built.get("ok") else None)
    if not candidate:
        return {"ok": False, "error": "no champion candidate available"}

    gen_overall = int((candidate.get("generalization") or {}).get("overall") or 0)
    if gen_overall < 70:
        return {
            "ok": False,
            "error": f"generalization {gen_overall} below 70 — cannot approve",
        }

    approved = dict(candidate)
    approved["status"] = "approved"
    approved["approved_at"] = _utc_now()
    if note:
        approved["approval_note"] = str(note).strip()

    champion["approved"] = approved
    champion["candidate"] = None
    champion["approved_at"] = approved["approved_at"]
    return update_research_program(data_dir, program_id, champion=champion)


def dismiss_program_champion_candidate(data_dir: str, program_id: str) -> dict[str, Any]:
    from .research_program import get_research_program, update_research_program

    program = get_research_program(data_dir, program_id)
    if not program:
        return {"ok": False, "error": "program not found"}
    champion = dict(program.get("champion") or {})
    champion.pop("candidate", None)
    champion["candidate_dismissed_at"] = _utc_now()
    return update_research_program(data_dir, program_id, champion=champion)
