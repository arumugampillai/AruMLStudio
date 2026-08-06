"""Model Research view — programs, knowledge, certification for one model (Phase F5)."""

from __future__ import annotations

from typing import Any

from .campaign_manifest import build_campaign_manifest
from .model_certification import build_model_certification
from .program_execution_store import ProgramExecutionStore
from .research_portfolio_report import build_model_research_portfolio_report
from .research_program_store import ResearchProgramStore


def get_model_research_view(data_dir: str, model_id: str) -> dict[str, Any]:
    """Aggregate view for Model → Research tab."""
    if not str(model_id or "").strip():
        return {"ok": False, "error": "model_id is required"}

    with ProgramExecutionStore(data_dir) as store:
        runs = store.list_runs(model_id=model_id, limit=30)

    programs: list[dict[str, Any]] = []
    with ResearchProgramStore(data_dir) as store:
        for run in runs:
            pid = str(run.get("program_id") or "")
            prog = store._load_program(pid) if pid else None
            manifest = run.get("manifest") or {}
            campaigns = []
            for c in manifest.get("campaigns") or []:
                cid = str(c.get("campaign_id") or "")
                man = build_campaign_manifest(data_dir, cid) if cid else {}
                campaigns.append({
                    "campaign_id": cid,
                    "name": c.get("name"),
                    "status": c.get("status") or (man.get("manifest") or {}).get("status"),
                    "manifest": man.get("manifest"),
                })
            programs.append({
                "run_id": run.get("run_id"),
                "run_number": run.get("run_number"),
                "program_id": pid,
                "program_name": (prog or {}).get("name") or manifest.get("program_name"),
                "program_type": (prog or {}).get("program_type") or manifest.get("program_type"),
                "status": run.get("status"),
                "campaigns": campaigns,
                "event_log": (run.get("event_log") or [])[-20:],
                "started_at": run.get("started_at"),
                "completed_at": run.get("completed_at"),
            })

    cert = build_model_certification(data_dir, model_id)
    portfolio = build_model_research_portfolio_report(data_dir, model_id)

    knowledge_summary = (portfolio.get("report") or {}).get("recommended_settings") or {}
    certification = cert.get("certification") or {}

    return {
        "ok": True,
        "model_id": model_id,
        "programs": programs,
        "knowledge": {
            "count": certification.get("knowledge_count"),
            "promoted": certification.get("promoted_knowledge"),
            "best_stop": knowledge_summary.get("stop_pct"),
            "premium": knowledge_summary.get("premium"),
            "hold": knowledge_summary.get("hold"),
            "confidence_pct": knowledge_summary.get("confidence_pct"),
        },
        "certification": {
            "research_grade": certification.get("research_grade"),
            "generalization_pct": certification.get("generalization_pct"),
            "production_ready": certification.get("production_ready"),
            "champion_candidate": certification.get("champion_candidate"),
            "recommended_settings": certification.get("recommended_settings"),
        },
        "portfolio_report": portfolio.get("report"),
    }
