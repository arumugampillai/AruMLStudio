"""Evidence → Finding → Knowledge pipeline (Phase E)."""

from __future__ import annotations

from typing import Any

from .finding_extraction import extract_findings_from_job
from .knowledge_store import KnowledgeStore, lifecycle_stage_for_status, list_knowledge_findings


def _auto_promote_eligible_findings(data_dir: str, finding_ids: list[str]) -> list[dict[str, Any]]:
    promoted: list[dict[str, Any]] = []
    with KnowledgeStore(data_dir) as store:
        for fid in finding_ids:
            finding = store.get_finding(fid)
            if not finding:
                continue
            if finding.get("status") != "confirmed":
                continue
            if int(finding.get("experiment_count") or 0) < 3:
                continue
            support = int(finding.get("supporting_count") or 0)
            total = int(finding.get("evidence_count") or 0) or 1
            if support / total < 0.75:
                continue
            out = store.promote_to_knowledge(fid)
            if out.get("ok"):
                promoted.append(out.get("finding") or {})
    return promoted


def process_job_knowledge_pipeline(
    data_dir: str,
    *,
    template: dict[str, Any],
    job: dict[str, Any],
    comparison: dict[str, Any] | None = None,
    campaign_id: str | None = None,
    program_id: str | None = None,
) -> dict[str, Any]:
    """Run after job completion — evidence extraction, linking, auto-promotion."""
    comparison = comparison or job.get("comparison") or {}
    trade_count = (comparison.get("after_trade_count") or (comparison.get("collected") or {}).get("strategy", {}).get("trade_count"))
    if trade_count is None:
        trade_count = (job.get("results") or {}).get("collected", {}).get("strategy", {}).get("trade_count")

    extraction = extract_findings_from_job(
        data_dir,
        template=template,
        job=job,
        comparison=comparison,
        trade_count=trade_count,
        campaign_id=campaign_id,
        program_id=program_id,
    )

    finding_ids = [str(f.get("finding_id") or "") for f in (extraction.get("findings") or []) if f.get("finding_id")]
    promoted = _auto_promote_eligible_findings(data_dir, finding_ids)

    stages: dict[str, int] = {}
    with KnowledgeStore(data_dir) as store:
        for fid in finding_ids:
            finding = store.get_finding(fid)
            if not finding:
                continue
            stage = lifecycle_stage_for_status(str(finding.get("status") or ""))
            stages[stage] = stages.get(stage, 0) + 1

    return {
        "ok": True,
        "extraction": extraction,
        "findings_updated": extraction.get("findings_updated", 0),
        "promoted_to_knowledge": len(promoted),
        "knowledge_entries": promoted,
        "lifecycle_counts": stages,
    }


def get_knowledge_pipeline_view(
    data_dir: str,
    *,
    program_id: str | None = None,
    campaign_id: str | None = None,
    limit: int = 30,
) -> dict[str, Any]:
    findings = list_knowledge_findings(data_dir, limit=limit)
    if campaign_id or program_id:
        filtered: list[dict[str, Any]] = []
        with KnowledgeStore(data_dir) as store:
            for f in findings:
                links = store.list_links_for_finding(str(f.get("finding_id") or ""))
                link_refs = {(lnk.get("link_type"), lnk.get("link_ref")) for lnk in links}
                if campaign_id and ("research_campaign", campaign_id) in link_refs:
                    filtered.append(f)
                elif program_id and ("research_program", program_id) in link_refs:
                    filtered.append(f)
        findings = filtered or findings

    enriched: list[dict[str, Any]] = []
    lifecycle_counts: dict[str, int] = {}
    for f in findings:
        stage = lifecycle_stage_for_status(str(f.get("status") or ""))
        lifecycle_counts[stage] = lifecycle_counts.get(stage, 0) + 1
        enriched.append({**f, "lifecycle_stage": stage})

    knowledge = [f for f in enriched if f.get("status") == "knowledge"]
    active_findings = [f for f in enriched if f.get("lifecycle_stage") == "finding"]
    evidence = [f for f in enriched if f.get("lifecycle_stage") == "evidence_linked"]

    return {
        "ok": True,
        "findings": enriched,
        "knowledge": knowledge,
        "active_findings": active_findings,
        "evidence_linked": evidence,
        "lifecycle_counts": lifecycle_counts,
        "totals": {
            "all": len(enriched),
            "knowledge": len(knowledge),
            "finding": len(active_findings),
            "evidence_linked": len(evidence),
        },
    }


def promote_finding_to_knowledge(data_dir: str, finding_id: str) -> dict[str, Any]:
    with KnowledgeStore(data_dir) as store:
        return store.promote_to_knowledge(finding_id)


def finalize_campaign_job_knowledge(
    data_dir: str,
    *,
    job: dict[str, Any],
    campaign_id: str,
    program_id: str | None = None,
) -> dict[str, Any]:
    """Post-extraction step — auto-promote findings from a completed campaign job."""
    knowledge = (job.get("results") or {}).get("knowledge") or {}
    finding_ids = [
        str(f.get("finding_id") or "")
        for f in (knowledge.get("findings") or [])
        if f.get("finding_id")
    ]
    promoted = _auto_promote_eligible_findings(data_dir, finding_ids)

    stages: dict[str, int] = {}
    with KnowledgeStore(data_dir) as store:
        for fid in finding_ids:
            finding = store.get_finding(fid)
            if not finding:
                continue
            stage = lifecycle_stage_for_status(str(finding.get("status") or ""))
            stages[stage] = stages.get(stage, 0) + 1

    return {
        "ok": True,
        "campaign_id": campaign_id,
        "program_id": program_id,
        "findings_seen": len(finding_ids),
        "promoted_to_knowledge": len(promoted),
        "knowledge_entries": promoted,
        "lifecycle_counts": stages,
    }
