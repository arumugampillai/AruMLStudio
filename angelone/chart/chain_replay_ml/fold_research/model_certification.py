"""Model Certification — Phase 3 output from accumulated research knowledge (Phase F4)."""

from __future__ import annotations

from typing import Any

from .experiment_pipeline_store import ExperimentPipelineStore
from .knowledge_store import KnowledgeStore
from .program_execution_store import ProgramExecutionStore
from .research_program_store import ResearchProgramStore


def _grade_from_score(score: int) -> str:
    if score >= 93:
        return "A+"
    if score >= 88:
        return "A"
    if score >= 80:
        return "B+"
    if score >= 72:
        return "B"
    if score >= 60:
        return "C"
    return "D"


def _collect_model_knowledge(data_dir: str, model_id: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    with KnowledgeStore(data_dir) as store:
        rows = store.conn.execute(
            """
            SELECT f.* FROM knowledge_findings f
            INNER JOIN finding_links l ON l.finding_id = f.finding_id
            WHERE l.link_type = 'model' AND l.link_ref = ?
            ORDER BY f.updated_at DESC
            LIMIT 200
            """,
            (model_id,),
        ).fetchall()
        for row in rows:
            doc = dict(row)
            meta_raw = doc.pop("metadata_json", None)
            import json

            doc["metadata"] = json.loads(meta_raw) if meta_raw else {}
            findings.append(doc)
    return findings


def _recommended_settings(findings: list[dict[str, Any]]) -> dict[str, Any]:
    rec: dict[str, Any] = {}
    for f in findings:
        if str(f.get("status")) not in ("confirmed", "promoted", "active"):
            continue
        text = str(f.get("finding") or "").lower()
        cat = str(f.get("category") or "")
        if "stop" in text or cat == "stop_loss":
            if "%" in text:
                import re

                m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
                if m:
                    rec["stop_pct"] = float(m.group(1))
        if "premium" in text or cat == "premium":
            rec.setdefault("premium", text)
        if "confidence" in text or cat == "confidence":
            import re

            m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
            if m:
                rec["confidence_pct"] = float(m.group(1))
        if "hold" in text or "second" in text or cat == "holding_time":
            rec.setdefault("hold", text)
    return rec


def build_model_certification(data_dir: str, model_id: str) -> dict[str, Any]:
    """Certification report for a single model from research runs + knowledge."""
    if not str(model_id or "").strip():
        return {"ok": False, "error": "model_id is required"}

    runs = []
    with ProgramExecutionStore(data_dir) as store:
        runs = store.list_runs(model_id=model_id, limit=50)

    findings = _collect_model_knowledge(data_dir, model_id)
    promoted = [f for f in findings if str(f.get("status")) in ("confirmed", "promoted", "active")]

    campaigns_completed = 0
    campaigns_total = 0
    best_pf = None
    gen_scores: list[int] = []
    gpu_hours = 0.0

    with ResearchProgramStore(data_dir) as store:
        for run in runs:
            for c in (run.get("manifest") or {}).get("campaigns") or []:
                campaigns_total += 1
                if c.get("status") == "completed":
                    campaigns_completed += 1
                cid = str(c.get("campaign_id") or "")
                if not cid:
                    continue
                camp = store._load_campaign(cid)
                if not camp:
                    continue
                mem = camp.get("memory") or {}
                try:
                    pf = float(mem.get("best_profit_factor")) if mem.get("best_profit_factor") is not None else None
                    if pf is not None and (best_pf is None or pf > best_pf):
                        best_pf = pf
                except (TypeError, ValueError):
                    pass
                gen = mem.get("best_generalization") or {}
                if gen.get("overall") is not None:
                    gen_scores.append(int(gen["overall"]))
                used = camp.get("budget_used") or {}
                gpu_hours += float(used.get("max_gpu_hours") or 0)

    gen_overall = int(round(sum(gen_scores) / len(gen_scores))) if gen_scores else 0
    knowledge_conf = "High" if len(promoted) >= 5 else ("Medium" if len(promoted) >= 2 else "Low")
    completion_pct = int(round(campaigns_completed / max(campaigns_total, 1) * 100))

    cert_score = int(
        min(100, gen_overall * 0.45 + completion_pct * 0.25 + min(len(promoted) * 8, 30))
    )
    grade = _grade_from_score(cert_score)
    production_ready = (
        gen_overall >= 70
        and len(promoted) >= 2
        and campaigns_completed >= max(1, campaigns_total // 2)
        and cert_score >= 72
    )

    recommended = _recommended_settings(promoted)

    report = {
        "model_id": model_id,
        "programs_run": len(runs),
        "campaigns_completed": campaigns_completed,
        "campaigns_total": campaigns_total,
        "knowledge_count": len(findings),
        "promoted_knowledge": len(promoted),
        "knowledge_confidence": knowledge_conf,
        "generalization_pct": gen_overall,
        "best_profit_factor": best_pf,
        "recommended_settings": recommended,
        "expected_pf": best_pf,
        "certification_grade": grade,
        "certification_score": cert_score,
        "champion_candidate": production_ready,
        "production_ready": production_ready,
        "research_grade": grade,
    }

    return {"ok": True, "certification": report}
