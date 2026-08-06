"""Phase 3 — proactive Knowledge Base retrieval and experiment scoring."""

from __future__ import annotations

import re
from typing import Any

from .experiment_similarity import _change_signature, check_experiment_before_create
from .finding_extraction import _finding_from_change, _goal_finding
from .knowledge_store import KnowledgeStore, list_knowledge_findings


def _confidence_stars(confidence: str) -> int:
    return {
        "very_high": 5,
        "high": 4,
        "medium": 3,
        "low": 2,
    }.get(str(confidence or ""), 1)


def _confidence_pct(finding: dict[str, Any]) -> int:
    base = {
        "very_high": 97,
        "high": 85,
        "medium": 70,
        "low": 52,
    }.get(str(finding.get("confidence") or ""), 35)
    exp = int(finding.get("experiment_count") or 0)
    trades = int(finding.get("trade_count") or 0)
    if exp >= 15:
        base = min(99, base + 4)
    elif exp >= 8:
        base = min(97, base + 2)
    if trades >= 10000:
        base = min(99, base + 2)
    support = int(finding.get("supporting_count") or 0)
    total = int(finding.get("evidence_count") or 0) or 1
    ratio = support / total
    if ratio < 0.5 and finding.get("status") != "contradicted":
        base = max(30, base - 15)
    return base


def _enrich_finding(finding: dict[str, Any]) -> dict[str, Any]:
    out = dict(finding)
    out["stars"] = _confidence_stars(str(finding.get("confidence") or ""))
    out["confidence_pct"] = _confidence_pct(finding)
    support = int(finding.get("supporting_count") or 0)
    total = int(finding.get("evidence_count") or 0) or 1
    out["support_ratio_pct"] = round(support / total * 100, 1)
    return out


def _keys_from_report(report: dict[str, Any]) -> set[str]:
    keys: set[str] = set()
    premium_threshold = float(report.get("premium_threshold") or 25.0)

    for item in (report.get("root_cause_analysis") or {}).get("items") or []:
        label = str(item.get("label") or "")
        low = label.lower()
        if "theta" in low:
            keys.update({"theta_decay_is_major_failure_mode", "high_theta_trades_underperform"})
        if "premium" in low:
            m = re.search(r"₹?\s*(\d+)", label)
            th = int(m.group(1)) if m else int(premium_threshold)
            keys.add(f"premium_below_{th}_poor")
            keys.add("low_premium_is_major_failure_mode")
        if "direction" in low or "prediction" in low:
            keys.add("wrong_direction_is_major_failure_mode")
        if "confidence" in low:
            keys.add("low_confidence_below_70_poor")
        if "range" in low:
            keys.add("range_regime_underperform")

    for rec in report.get("recommendations") or []:
        spec = _finding_from_change({
            "text": rec.get("text"),
            "target": "strategy_registry",
            "filters": {"min_premium": premium_threshold},
        })
        if spec:
            keys.add(str(spec["finding_key"]))

    return keys


def _findings_by_keys(data_dir: str, keys: set[str]) -> list[dict[str, Any]]:
    if not keys:
        return []
    with KnowledgeStore(data_dir) as store:
        placeholders = ",".join("?" for _ in keys)
        rows = store.conn.execute(
            f"""
            SELECT finding_id, finding_key, finding, category, status, confidence,
                   evidence_count, supporting_count, contradicting_count,
                   trade_count, experiment_count, time_span_days, last_confirmed_at
            FROM knowledge_findings
            WHERE finding_key IN ({placeholders})
            ORDER BY evidence_count DESC
            """,
            list(keys),
        ).fetchall()
    return [_enrich_finding(dict(r)) for r in rows]


def get_known_findings_for_report(data_dir: str, report: dict[str, Any]) -> dict[str, Any]:
    if not report.get("ok"):
        return report
    keys = _keys_from_report(report)
    matched = _findings_by_keys(data_dir, keys)
    if not matched:
        all_findings = list_knowledge_findings(data_dir, limit=20)
        matched = [_enrich_finding(f) for f in all_findings[:5]]
        source = "global" if matched else "none"
    else:
        source = "report_context"
    matched.sort(key=lambda f: (f.get("stars", 0), f.get("confidence_pct", 0)), reverse=True)
    return {"ok": True, "source": source, "finding_keys": sorted(keys), "findings": matched}


def _aggregate_evidence_quality(findings: list[dict[str, Any]]) -> str:
    if not findings:
        return "weak"
    exp = max(int(f.get("experiment_count") or 0) for f in findings)
    trades = max(int(f.get("trade_count") or 0) for f in findings)
    if exp >= 10 and trades >= 5000:
        return "strong"
    if exp >= 3 or trades >= 500:
        return "moderate"
    return "weak"


def _improvement_probability(
    *,
    similarity_check: dict[str, Any],
    findings: list[dict[str, Any]],
) -> str:
    if similarity_check.get("should_warn"):
        return "Low"
    if not findings:
        novelty = 100 - float(similarity_check.get("top_similarity_pct") or 0)
        return "Medium" if novelty >= 70 else "Medium-Low"

    contradicted = [f for f in findings if f.get("status") == "contradicted"]
    if contradicted and len(contradicted) >= len(findings) / 2:
        return "Low"

    confirmed = [f for f in findings if f.get("status") in ("confirmed", "supported")]
    if confirmed:
        avg_support = sum(f.get("support_ratio_pct", 50) for f in confirmed) / len(confirmed)
        if avg_support >= 75 and any(f.get("status") == "confirmed" for f in confirmed):
            return "High"
        if avg_support >= 60:
            return "Medium-High"

    top_match = (similarity_check.get("matches") or [None])[0]
    if top_match and top_match.get("outcome") == "improved":
        return "Medium-High"

    return "Medium"


def score_experiment_proposal(
    data_dir: str,
    report: dict[str, Any],
    *,
    accepted_items: list[dict[str, Any]],
    goal: str | None = None,
) -> dict[str, Any]:
    similarity = check_experiment_before_create(
        data_dir,
        report,
        accepted_items=accepted_items,
        goal=goal,
    )
    keys: set[str] = set(_change_signature(item) for item in accepted_items)
    goal_spec = _goal_finding(str(goal or ""))
    if goal_spec:
        keys.add(str(goal_spec["finding_key"]))
    findings = _findings_by_keys(data_dir, keys)

    novelty_score = round(100.0 - float(similarity.get("top_similarity_pct") or 0), 1)
    evidence_quality = _aggregate_evidence_quality(findings)
    improvement_probability = _improvement_probability(
        similarity_check=similarity,
        findings=findings,
    )

    return {
        **similarity,
        "novelty_score": novelty_score,
        "evidence_quality": evidence_quality,
        "improvement_probability": improvement_probability,
        "matched_findings": findings,
    }


def get_feature_knowledge(data_dir: str, feature_names: list[str]) -> list[dict[str, Any]]:
    names = {str(n).lower() for n in feature_names if n}
    if not names:
        return []
    out: list[dict[str, Any]] = []
    with KnowledgeStore(data_dir) as store:
        rows = store.conn.execute(
            """
            SELECT DISTINCT f.finding_id, f.finding_key, f.finding, f.category, f.status,
                   f.confidence, f.evidence_count, f.experiment_count, f.trade_count,
                   f.supporting_count, l.link_ref
            FROM knowledge_findings f
            JOIN finding_links l ON l.finding_id = f.finding_id
            WHERE l.link_type = 'feature'
            """,
        ).fetchall()
    for row in rows:
        ref = str(row["link_ref"] or "").lower()
        if ref not in names and not any(ref in n or n in ref for n in names):
            continue
        finding = _enrich_finding(dict(row))
        support = int(finding.get("supporting_count") or 0)
        total = int(finding.get("evidence_count") or 0) or 1
        avg_gain = None
        with KnowledgeStore(data_dir) as store:
            ev_rows = store.conn.execute(
                """
                SELECT AVG(pf_change) AS avg_pf
                FROM finding_evidence
                WHERE finding_id = ? AND supports_finding = 1 AND pf_change IS NOT NULL
                """,
                (finding["finding_id"],),
            ).fetchone()
            if ev_rows and ev_rows["avg_pf"] is not None:
                avg_gain = round(float(ev_rows["avg_pf"]), 2)
        out.append({
            **finding,
            "feature": ref,
            "average_pf_gain": avg_gain,
            "evaluated_experiments": finding.get("experiment_count"),
        })
    out.sort(key=lambda x: x.get("experiment_count") or 0, reverse=True)
    return out


def get_strategy_filter_knowledge(data_dir: str, config: dict[str, Any]) -> list[dict[str, Any]]:
    """Hints for strategy config knobs present in champion config."""
    entry = config.get("entry") or {}
    conf = config.get("confidence") or {}
    hints: list[dict[str, Any]] = []

    checks: list[tuple[str, dict[str, Any]]] = []
    if entry.get("premium_min") is not None:
        checks.append((f"Premium floor ₹{entry['premium_min']}", {
            "text": f"Avoid premium below ₹{entry['premium_min']}",
            "filters": {"min_premium": entry["premium_min"]},
        }))
    min_conf = entry.get("min_confidence") or conf.get("min_signal_strength")
    if min_conf is not None:
        checks.append((f"Confidence >{min_conf}%", {
            "text": f"Increase confidence threshold to {min_conf}%",
            "filters": {"min_confidence": min_conf},
        }))

    for label, change in checks:
        spec = _finding_from_change(change)
        if not spec:
            continue
        matched = _findings_by_keys(data_dir, {spec["finding_key"]})
        if not matched:
            continue
        f = matched[0]
        hints.append({
            "filter_label": label,
            "finding": f.get("finding"),
            "status": f.get("status"),
            "experiments": f.get("experiment_count"),
            "confidence_pct": f.get("confidence_pct"),
            "stars": f.get("stars"),
        })
    return hints
