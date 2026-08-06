"""Research Portfolio Report — permanent research record per model (Phase F4)."""

from __future__ import annotations

from typing import Any

from .experiment_pipeline_store import ExperimentPipelineStore
from .knowledge_store import KnowledgeStore
from .model_certification import build_model_certification
from .program_execution_store import ProgramExecutionStore
from .research_program_store import ResearchProgramStore


def build_model_research_portfolio_report(data_dir: str, model_id: str) -> dict[str, Any]:
    """Aggregate report when all (or available) programs finish on a model."""
    if not str(model_id or "").strip():
        return {"ok": False, "error": "model_id is required"}

    with ProgramExecutionStore(data_dir) as store:
        runs = store.list_runs(model_id=model_id, limit=100)

    program_ids: set[str] = set()
    campaign_ids: list[str] = []
    experiments = 0
    gpu_hours = 0.0
    rejected = 0

    with ResearchProgramStore(data_dir) as store:
        for run in runs:
            program_ids.add(str(run.get("program_id") or ""))
            for c in (run.get("manifest") or {}).get("campaigns") or []:
                cid = str(c.get("campaign_id") or "")
                if cid:
                    campaign_ids.append(cid)
                camp = store._load_campaign(cid) if cid else None
                if camp:
                    used = camp.get("budget_used") or {}
                    experiments += int(used.get("experiments") or 0)
                    gpu_hours += float(used.get("max_gpu_hours") or 0)
                    mem = camp.get("memory") or {}
                    rejected += int(mem.get("rejected_hypotheses") or 0)

    with ExperimentPipelineStore(data_dir) as pipe:
        for cid in campaign_ids:
            jobs = pipe.list_jobs(campaign_id=cid, limit=500)
            for j in jobs:
                verdict = ((j.get("results") or {}).get("verdict") or {}).get("verdict")
                if verdict in ("Reject", "Worse", "Failed"):
                    rejected += 1

    knowledge_count = 0
    with KnowledgeStore(data_dir) as kstore:
        row = kstore.conn.execute(
            """
            SELECT COUNT(DISTINCT f.finding_id) AS n
            FROM knowledge_findings f
            INNER JOIN finding_links l ON l.finding_id = f.finding_id
            WHERE l.link_type = 'model' AND l.link_ref = ?
            """,
            (model_id,),
        ).fetchone()
        knowledge_count = int(row["n"] if row else 0)

    cert_out = build_model_certification(data_dir, model_id)
    cert = cert_out.get("certification") or {}

    report = {
        "model_id": model_id,
        "programs": len(program_ids),
        "program_runs": len(runs),
        "campaigns": len(campaign_ids),
        "experiments": experiments,
        "knowledge": knowledge_count,
        "rejected_hypotheses": rejected,
        "gpu_hours": round(gpu_hours, 1),
        "best_pf": cert.get("best_profit_factor"),
        "generalization_pct": cert.get("generalization_pct"),
        "certification": cert.get("certification_grade"),
        "certification_score": cert.get("certification_score"),
        "production_ready": cert.get("production_ready"),
        "recommended_settings": cert.get("recommended_settings"),
    }

    return {"ok": True, "report": report, "certification": cert}
