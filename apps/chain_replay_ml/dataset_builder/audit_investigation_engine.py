"""Automatic audit investigations, timelines, conclusions, and training readiness."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from .audit_progress import system_metrics
from .audit_rca import investigate_audit_failure
from .investigation_history import append_investigation
from .pipeline_identity import BUILDER_VERSION, VALIDATOR_VERSION
from .writer import _safe_filename

ROOT_CAUSE_CATEGORIES = (
    "Builder Bug",
    "Validator Bug",
    "Configuration Mismatch",
    "Dataset Issue",
    "Expected Statistical Behaviour",
    "Audit Methodology",
    "Threshold Too Strict",
    "Numerical Precision",
    "Replay Data Issue",
    "Missing Source Data",
    "Performance Issue",
    "Unable to Classify",
    "Unknown",
)

SEVERITY_LEVELS = ("critical", "warning", "information")

TRAINING_RECOMMENDATION_READY = "READY"
TRAINING_RECOMMENDATION_WARNINGS = "READY_WITH_WARNINGS"
TRAINING_RECOMMENDATION_NOT_READY = "NOT_READY"

TRAINING_RECOMMENDATION_LABELS: dict[str, str] = {
    TRAINING_RECOMMENDATION_READY: "🟢 READY",
    TRAINING_RECOMMENDATION_WARNINGS: "🟡 READY WITH WARNINGS",
    TRAINING_RECOMMENDATION_NOT_READY: "🔴 NOT READY",
}


def training_recommendation_display(recommendation: str) -> str:
    """Human-readable label for a canonical training_recommendation value."""
    return TRAINING_RECOMMENDATION_LABELS.get(recommendation, recommendation)


def is_training_allowed(recommendation: str) -> bool:
    return recommendation in (TRAINING_RECOMMENDATION_READY, TRAINING_RECOMMENDATION_WARNINGS)


def normalize_training_recommendation(source: dict[str, Any] | None) -> str:
    """Resolve canonical training_recommendation from audit cache or report."""
    if not source:
        return TRAINING_RECOMMENDATION_NOT_READY

    rec = source.get("training_recommendation")
    if rec in TRAINING_RECOMMENDATION_LABELS:
        return str(rec)

    text = str(rec or source.get("recommendation") or "")
    upper = text.upper()
    if "NOT READY" in upper:
        return TRAINING_RECOMMENDATION_NOT_READY
    if "WARNING" in upper:
        return TRAINING_RECOMMENDATION_WARNINGS
    if upper.endswith("READY") or "🟢" in text:
        return TRAINING_RECOMMENDATION_READY

    tr = source.get("training_readiness") or {}
    if isinstance(tr, dict):
        nested = tr.get("training_recommendation")
        if nested in TRAINING_RECOMMENDATION_LABELS:
            return str(nested)
        if int(tr.get("critical_count") or 0) > 0:
            return TRAINING_RECOMMENDATION_NOT_READY
        if tr.get("ready"):
            warnings_n = int(tr.get("warnings_count") or 0)
            return (
                TRAINING_RECOMMENDATION_READY
                if warnings_n == 0
                else TRAINING_RECOMMENDATION_WARNINGS
            )

    decision = source.get("audit_decision") or {}
    if isinstance(decision, dict):
        drec = decision.get("training_recommendation")
        if drec in TRAINING_RECOMMENDATION_LABELS:
            return str(drec)
        ddisplay = str(decision.get("training_recommendation_display") or drec or "")
        dup = ddisplay.upper()
        if "NOT READY" in dup:
            return TRAINING_RECOMMENDATION_NOT_READY
        if "WARNING" in dup:
            return TRAINING_RECOMMENDATION_WARNINGS
        if "READY" in dup:
            return TRAINING_RECOMMENDATION_READY

    blocking = source.get("blocking_issues")
    if blocking is not None:
        if len(blocking) > 0:
            return TRAINING_RECOMMENDATION_NOT_READY
        if source.get("ready_for_training") is True or source.get("status") in ("pass", "warn", "fail"):
            warnings_n = int(source.get("warnings") or 0)
            return (
                TRAINING_RECOMMENDATION_READY
                if warnings_n == 0
                else TRAINING_RECOMMENDATION_WARNINGS
            )

    if source.get("ready_for_training") is True:
        return TRAINING_RECOMMENDATION_WARNINGS

    return TRAINING_RECOMMENDATION_NOT_READY


class InvestigationTimeline:
    """Records real investigation steps with timing; streams via WebSocket."""

    def __init__(
        self,
        on_progress: Callable[[dict[str, Any]], None] | None,
        *,
        feature: str = "",
        category: str = "",
        check_label: str = "",
    ) -> None:
        self._on_progress = on_progress
        self.feature = feature
        self.category = category
        self.check_label = check_label or feature
        self.steps: list[dict[str, Any]] = []
        self._open: dict[str, float] = {}

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def _emit(self, *, status: str = "running", message: str | None = None) -> None:
        if not self._on_progress:
            return
        running = next((s for s in reversed(self.steps) if s.get("status") == "running"), None)
        self._on_progress({
            "phase": "investigation",
            "feature": self.feature,
            "category": self.category,
            "check_label": self.check_label,
            "status": status,
            "message": message or (running or {}).get("label"),
            "current_step": (running or {}).get("label"),
            "timeline": [dict(s) for s in self.steps],
            "timeline_step_count": len(self.steps),
        })

    def begin(self, step_id: str, label: str, *, message: str | None = None) -> None:
        self._open[step_id] = time.monotonic()
        self.steps.append({
            "id": step_id,
            "label": label,
            "status": "running",
            "start": self._now_iso(),
            "end": None,
            "elapsed_sec": None,
            "message": message or label,
        })
        self._emit(status="running", message=label)

    def complete(
        self,
        step_id: str,
        status: str = "pass",
        *,
        message: str | None = None,
    ) -> None:
        t0 = self._open.pop(step_id, time.monotonic())
        elapsed = round(time.monotonic() - t0, 4)
        for row in reversed(self.steps):
            if row["id"] == step_id and row["status"] == "running":
                row["status"] = status
                row["end"] = self._now_iso()
                row["elapsed_sec"] = elapsed
                if message:
                    row["message"] = message
                break
        self._emit(status=status, message=message)

    def run(self, step_id: str, label: str, func: Callable[[], Any]) -> Any:
        self.begin(step_id, label)
        try:
            out = func()
            self.complete(step_id, "pass")
            return out
        except Exception as exc:
            self.complete(step_id, "fail", message=str(exc))
            raise


def _qualitative_confidence_level(
    pct: int | None,
    rc: str,
    *,
    matched: int = 0,
    total: int = 0,
) -> str:
    """Map investigation evidence to High / Medium / Low."""
    if rc == "Unable to Classify":
        return "Low"
    if pct is not None:
        if pct >= 85:
            return "High"
        if pct >= 65:
            return "Medium"
        return "Low"
    if total:
        ratio = matched / total
        if ratio >= 0.85:
            return "High"
        if ratio >= 0.65:
            return "Medium"
    return "Low"


def _confidence_evidence(result: dict[str, Any]) -> dict[str, Any]:
    """Build confidence evidence checklist from investigation result."""
    diagnosis = result.get("diagnosis") or {}
    rc_block = result.get("root_cause") or {}
    unable = (
        rc_block.get("title") == "Unable to Classify"
        or rc_block.get("category") == "Unable to Classify"
        or diagnosis.get("problem") == "Unable to Classify"
    )
    checks = list(result.get("confidence_checks") or [])
    if not checks:
        for c in result.get("causes") or []:
            checks.append({
                "check": c.get("label") or c.get("id"),
                "status": c.get("status"),
                "confidence_pct": c.get("confidence_pct"),
            })
    matched = sum(1 for c in checks if c.get("status") in ("pass", "derived"))
    total = len(checks)
    raw_pct = diagnosis.get("confidence_pct")
    if unable:
        pct = None
    elif raw_pct is not None:
        pct = int(raw_pct)
    elif total:
        pct = int(round(100 * matched / total))
    else:
        pct = None
    evidence_items = []
    for c in checks:
        if c.get("status") in ("pass", "derived"):
            label = c.get("check") or "—"
            if c.get("status") == "derived" and c.get("confidence_label"):
                label = f"{label} ({c['confidence_label']})"
            evidence_items.append(label)
    default_reason = None
    if unable:
        default_reason = "Evidence is insufficient to determine a single dominant cause."
    elif pct is not None and pct < 70:
        default_reason = "No single cause exceeded the confidence threshold."
    return {
        "confidence": pct,
        "evidence": {
            "matched": matched,
            "total": total,
            "summary": f"{matched} of {total} checks matched" if total else "No checks recorded",
            "items": evidence_items,
        },
        "confidence_reason": diagnosis.get("confidence_reason") or default_reason,
    }


def classify_root_cause(result: dict[str, Any]) -> dict[str, Any]:
    """Map investigation result to exactly one root-cause category."""
    status = str(result.get("status") or "")
    diagnosis = result.get("diagnosis") or {}
    policy = result.get("policy_alignment") or {}
    category = str(result.get("category") or "")
    rc_block = result.get("root_cause") or {}
    evidence = _confidence_evidence(result)
    possible_causes = list(result.get("possible_causes") or [])
    recommended_fix = result.get("recommended_fix") or {}

    if status == "configuration_mismatch" or policy.get("configuration_mismatch"):
        rc = "Configuration Mismatch"
        severity = "warning"
        confidence = evidence["confidence"]
        action = recommended_fix.get("action") or (
            policy.get("suggested_fix") or "Align validator lookback policy with dataset specification."
        )
        reason = recommended_fix.get("reason")
    elif category == "correlation":
        rc = rc_block.get("category") or "Audit Methodology"
        if rc in ("Dataset Issue", "Unknown"):
            rc = "Audit Methodology"
        severity = "information" if rc in ("Expected Statistical Behaviour", "Audit Methodology") else "warning"
        confidence = evidence["confidence"]
        action = recommended_fix.get("action") or diagnosis.get("summary") or (
            "Replace correlation with monotonicity validation where appropriate."
        )
        reason = recommended_fix.get("reason") or diagnosis.get("summary")
    elif category == "distribution":
        rc = rc_block.get("category") or "Threshold Too Strict"
        severity = "warning" if status == "diagnosed" else "information"
        confidence = evidence["confidence"]
        action = recommended_fix.get("action") or diagnosis.get("summary") or "Review distribution bounds for this feature."
        reason = recommended_fix.get("reason") or (
            "Distribution bounds may be too strict for this feature's natural range."
        )
    elif status == "not_found":
        rc = "Missing Source Data"
        severity = "warning"
        confidence = 70
        action = "Could not locate a failing sample row for investigation."
        reason = diagnosis.get("summary")
    elif rc_block.get("title") == "Unable to Classify" or rc_block.get("category") == "Unable to Classify":
        rc = "Unable to Classify"
        severity = "information"
        confidence = None
        action = recommended_fix.get("action") or "Review individual cause checks and failing row samples."
        reason = evidence["confidence_reason"] or diagnosis.get("summary")
        if not possible_causes:
            possible_causes = [
                "Snapshot selection mismatch",
                "IV interpolation difference",
                "BS implementation difference",
            ]
    elif rc_block.get("category"):
        rc = str(rc_block["category"])
        severity = "warning"
        confidence = evidence["confidence"]
        action = recommended_fix.get("action") or diagnosis.get("summary") or "Review failing rows."
        reason = recommended_fix.get("reason") or diagnosis.get("summary")
    elif rc_block.get("title"):
        title = str(rc_block["title"])
        if "timestamp" in title.lower() or "lookback" in title.lower():
            rc = "Validator Bug"
        elif "BS" in title or "builder" in title.lower():
            rc = "Builder Bug"
        else:
            rc = "Unable to Classify"
        severity = "warning"
        confidence = evidence["confidence"]
        action = recommended_fix.get("action") or diagnosis.get("summary") or "Review failing rows."
        reason = recommended_fix.get("reason") or diagnosis.get("summary")
    elif category == "independent":
        failed = [c for c in (result.get("causes") or []) if c.get("status") == "fail"]
        if failed:
            top = failed[0]
            fid = str(top.get("id") or "")
            if "lookback" in fid or "timestamp" in fid:
                rc = "Validator Bug"
            elif "bs" in fid.lower():
                rc = "Builder Bug"
            else:
                rc = "Threshold Too Strict"
            action = top.get("detail") or diagnosis.get("summary")
        else:
            rc = "Unable to Classify"
            action = recommended_fix.get("action") or diagnosis.get("summary") or (
                "Sample validates on replay — review audit tolerance or other failing rows."
            )
            possible_causes = possible_causes or [
                "Snapshot selection mismatch",
                "IV interpolation difference",
                "BS implementation difference",
            ]
        severity = "warning" if failed else "information"
        confidence = evidence["confidence"]
        reason = evidence["confidence_reason"] or diagnosis.get("summary")
    else:
        rc = "Unable to Classify"
        severity = "information"
        confidence = evidence["confidence"]
        action = recommended_fix.get("action") or diagnosis.get("summary") or (
            "Investigation incomplete — review audit checks manually."
        )
        reason = evidence["confidence_reason"] or "No single cause exceeded the confidence threshold."
        possible_causes = possible_causes or [
            "Snapshot selection mismatch",
            "IV interpolation difference",
            "BS implementation difference",
        ]

    affected_rows = int(
        (result.get("comparison") or {}).get("rows")
        or (result.get("fix_impact") or {}).get("independent_validation", {}).get("current_failures")
        or result.get("audit_reported_failures")
        or (result.get("failure_summary") or {}).get("total_failures")
        or 0
    )
    if rc == "Unable to Classify":
        confidence = None
    recommendation_reason = recommended_fix.get("reason") or diagnosis.get("summary")
    confidence_reason = evidence["confidence_reason"] or diagnosis.get("confidence_reason")
    if rc == "Unable to Classify" and not confidence_reason:
        confidence_reason = "Evidence is insufficient to determine a single dominant cause."
    ev = evidence["evidence"]
    classification_confidence = _qualitative_confidence_level(
        confidence, rc, matched=ev["matched"], total=ev["total"],
    )
    return {
        "root_cause": rc,
        "confidence": confidence,
        "classification_confidence": classification_confidence,
        "severity": severity,
        "recommended_action": action,
        "recommendation_reason": recommendation_reason,
        "recommendation_items": list(recommended_fix.get("items") or result.get("recommended_validations") or []),
        "possible_causes": possible_causes,
        "confidence_evidence": ev,
        "confidence_reason": confidence_reason,
        "affected_rows": affected_rows,
        "affected_features": [result.get("feature") or result.get("check") or result.get("check_id") or "—"],
    }


def collect_investigation_targets(report: dict[str, Any]) -> list[dict[str, Any]]:
    """WARN and FAIL checks that trigger automatic investigation."""
    targets: list[dict[str, Any]] = []
    ea = report.get("extended_audit") or {}

    ind = ea.get("independent_formulas") or {}
    failures_by_feature: dict[str, dict[str, Any]] = {}
    for fail_row in ind.get("failures") or []:
        feat = fail_row.get("feature")
        if feat and feat not in failures_by_feature:
            failures_by_feature[str(feat)] = fail_row
    if ind.get("status") == "configuration_mismatch":
        targets.append({
            "category": "independent",
            "feature": (ind.get("checks") or [{}])[0].get("feature") or "lookback_policy",
            "check": "Lookback Policy",
            "status": "configuration_mismatch",
            "audit_failure_count": sum(int(c.get("failed") or 0) for c in ind.get("checks") or []),
        })
    for chk in ind.get("checks") or []:
        st = chk.get("status")
        if st in ("fail", "warn"):
            feat = chk.get("feature")
            targets.append({
                "category": "independent",
                "feature": feat,
                "check": chk.get("label") or feat,
                "status": st,
                "audit_failure_count": int(chk.get("failed") or 0),
                "failure_sample": failures_by_feature.get(str(feat or "")),
            })

    dist = ea.get("feature_distributions") or {}
    for feat in dist.get("features") or []:
        if feat.get("status") in ("fail", "warn"):
            targets.append({
                "category": "distribution",
                "feature": feat.get("feature"),
                "check": feat.get("label") or feat.get("feature"),
                "status": feat.get("status"),
            })

    corr = ea.get("correlation_checks") or {}
    for chk in corr.get("checks") or []:
        if chk.get("status") in ("fail", "warn"):
            targets.append({
                "category": "correlation",
                "feature": chk.get("id") or chk.get("check"),
                "check": chk.get("pair_label") or chk.get("check"),
                "status": chk.get("status"),
                "audit_row": chk,
            })

    return targets


def _investigation_key(target: dict[str, Any]) -> str:
    return f"{target.get('category')}::{target.get('feature')}"


def merge_root_causes(investigations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Merge investigations with the same root cause."""
    groups: dict[str, dict[str, Any]] = {}
    for inv in investigations:
        rc = inv.get("root_cause_category") or inv.get("classification", {}).get("root_cause") or "Unable to Classify"
        if rc == "Unknown":
            rc = "Unable to Classify"
        action = inv.get("classification", {}).get("recommended_action") or ""
        merge_key = f"{rc}::{action[:80]}"
        feat = inv.get("feature") or inv.get("check")
        if merge_key not in groups:
            clf = inv.get("classification") or {}
            groups[merge_key] = {
                "root_cause": rc,
                "affected_checks": [],
                "affected_samples": 0,
                "confidence": clf.get("confidence"),
                "classification_confidence": clf.get("classification_confidence"),
                "severity": clf.get("severity") or "warning",
                "recommendation": action,
                "recommendation_items": list(clf.get("recommendation_items") or []),
            }
        g = groups[merge_key]
        g["affected_checks"].append(feat)
        g["affected_samples"] += int(inv.get("classification", {}).get("affected_rows") or 0)
        clf = inv.get("classification") or {}
        if clf.get("classification_confidence"):
            g["classification_confidence"] = clf["classification_confidence"]
        if clf.get("confidence") is not None:
            g["confidence"] = max(g.get("confidence") or 0, int(clf["confidence"]))
        if inv.get("classification", {}).get("recommendation_reason"):
            g["recommendation_reason"] = inv["classification"]["recommendation_reason"]
        if inv.get("classification", {}).get("confidence_evidence"):
            g["confidence_evidence"] = inv["classification"]["confidence_evidence"]

    return [
        {
            **g,
            "affected_checks_count": len(g["affected_checks"]),
            "recommendation": g.get("recommendation") or g.get("recommendation_reason"),
        }
        for g in groups.values()
    ]


def _append_critical(
    critical: list[dict[str, Any]],
    check: str,
    source: str,
    *,
    detail: str | None = None,
) -> None:
    entry: dict[str, Any] = {"check": check, "severity": "critical", "source": source}
    if detail:
        entry["detail"] = detail
    critical.append(entry)


def _critical_failures(report: dict[str, Any]) -> list[dict[str, Any]]:
    """Failures that block training — validator methodology issues are excluded."""
    critical: list[dict[str, Any]] = []
    files = report.get("files") or {}
    dataset_exists = bool(
        (files.get("dataset") or {}).get("exists")
        or (files.get("parquet") or {}).get("exists")
    )
    if not dataset_exists:
        _append_critical(critical, "Corrupt dataset: missing parquet file", "corrupt_dataset")

    for msg in list(report.get("errors") or []):
        low = str(msg).lower()
        if "missing parquet" in low:
            _append_critical(critical, str(msg)[:120], "corrupt_dataset")
        elif "pyarrow is required" in low:
            _append_critical(critical, str(msg)[:120], "corrupt_dataset")
        elif "100% null" in low:
            _append_critical(critical, str(msg)[:120], "corrupt_dataset")
        elif "missing target" in low:
            _append_critical(critical, str(msg)[:120], "missing_targets")
        elif "duplicate row" in low:
            _append_critical(critical, str(msg)[:120], "duplicate_keys")
        elif "invalid timestamp" in low:
            _append_critical(critical, str(msg)[:120], "invalid_timestamps")
        elif "target quality check failed" in low:
            _append_critical(critical, str(msg)[:120], "missing_targets")
        elif "missing expected" in low:
            _append_critical(critical, str(msg)[:120], "missing_required_columns")

    if not (files.get("metadata") or {}).get("exists"):
        _append_critical(critical, "Missing required file: metadata JSON", "missing_required_columns")
    spec_exists = (files.get("expected") or {}).get("exists") or (files.get("specification") or {}).get("exists")
    if not spec_exists:
        _append_critical(critical, "Missing required file: expected specification", "missing_required_columns")

    targets = report.get("targets_audit") or {}
    for col in targets.get("columns") or []:
        if not col.get("present"):
            name = col.get("column") or "?"
            _append_critical(critical, f"Missing target column: {name}", "missing_target_columns")

    fr = (report.get("feature_audit") or {}).get("formula_recalc") or {}
    if fr.get("status") == "fail":
        failed = int(fr.get("comparisons_failed") or fr.get("failures") or 0)
        _append_critical(
            critical,
            f"Formula recalculation failure ({failed:,} comparisons failed)",
            "formula_recalc",
        )

    ea = report.get("extended_audit") or {}
    replay = ea.get("replay_verification") or {}
    if replay.get("status") == "fail":
        _append_critical(critical, "Replay verification failure", "replay_verification")

    ia = report.get("integrity_audit") or {}
    dup_rows = int(ia.get("duplicate_rows") or 0)
    if dup_rows > 0:
        _append_critical(critical, f"Duplicate primary keys ({dup_rows:,} rows)", "duplicate_keys")
    invalid_ts = int(ia.get("invalid_timestamps") or 0)
    if invalid_ts > 0:
        _append_critical(critical, f"Invalid timestamps ({invalid_ts:,} rows)", "invalid_timestamps")
    all_null = int(ia.get("all_null_count") or 0)
    if all_null > 0:
        _append_critical(
            critical,
            f"Corrupt dataset: {all_null} feature column(s) are 100% NULL",
            "corrupt_dataset",
        )
    missing_targets = int(ia.get("missing_target_values") or 0)
    if missing_targets > 0:
        _append_critical(
            critical,
            f"Missing target values ({missing_targets:,} rows)",
            "missing_targets",
        )
    elif int(targets.get("missing_values") or 0) > 0 and targets.get("status") == "fail":
        _append_critical(
            critical,
            f"Missing target values ({int(targets['missing_values']):,} rows)",
            "missing_targets",
        )

    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for item in critical:
        key = item["check"]
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


_DATASET_HEALTH_WEIGHTS = {
    "configuration": 12,
    "sampling": 12,
    "strike_selection": 8,
    "prediction_targets": 12,
    "feature_generation": 8,
    "integrity": 10,
    "performance": 5,
}

_RC_FIX_MINUTES: dict[str, int] = {
    "Unable to Classify": 25,
    "Audit Methodology": 10,
    "Expected Statistical Behaviour": 10,
    "Threshold Too Strict": 10,
    "Builder Bug": 45,
    "Validator Bug": 30,
    "Configuration Mismatch": 15,
    "Missing Source Data": 20,
    "Dataset Issue": 20,
    "Numerical Precision": 15,
    "Replay Data Issue": 30,
    "Performance Issue": 20,
}


def _section_pass_rate(status: str) -> float:
    if status == "pass":
        return 1.0
    if status == "warn":
        return 0.85
    return 0.0


def _compute_dataset_health(report: dict[str, Any]) -> float:
    """Data-centric quality score (excludes validator methodology sections)."""
    qs = report.get("quality_score") or {}
    sections = {s["id"]: s.get("status", "pass") for s in (qs.get("sections") or [])}
    total_w = sum(_DATASET_HEALTH_WEIGHTS.values())
    if not total_w:
        return float(qs.get("confidence_pct") or 0.0)
    score = sum(
        _DATASET_HEALTH_WEIGHTS[key] * _section_pass_rate(sections.get(key, "pass"))
        for key in _DATASET_HEALTH_WEIGHTS
    )
    return round(min(100.0, score / total_w * 100.0), 1)


def _compute_builder_confidence(report: dict[str, Any]) -> float:
    fa = report.get("feature_audit") or {}
    fr = fa.get("formula_recalc") or {}
    comparisons = int(fr.get("comparisons") or 0)
    failed = int(fr.get("comparisons_failed") or fr.get("failures") or 0)
    if comparisons > 0:
        return round(100.0 * (comparisons - failed) / comparisons, 1)
    passed = int(fa.get("formula_groups_passed") or 0)
    total = int(fa.get("formula_groups_total") or 0)
    if total > 0:
        return round(100.0 * passed / total, 1)
    return 100.0 if fr.get("status") == "pass" else 0.0


def _compute_validator_confidence(report: dict[str, Any]) -> float:
    ea = report.get("extended_audit") or {}
    rates: list[float] = []

    ind = ea.get("independent_formulas") or {}
    ind_comp = int(ind.get("comparisons") or 0)
    ind_fail = int(ind.get("comparisons_failed") or 0)
    if ind_comp > 0:
        rates.append((ind_comp - ind_fail) / ind_comp)

    dist = ea.get("feature_distributions") or {}
    dist_feats = [f for f in (dist.get("features") or []) if f.get("status") != "info"]
    if dist_feats:
        rates.append(sum(1 for f in dist_feats if f.get("status") == "pass") / len(dist_feats))

    corr = ea.get("correlation_checks") or {}
    corr_checks = [c for c in (corr.get("checks") or []) if c.get("status") != "info"]
    if corr_checks:
        rates.append(sum(1 for c in corr_checks if c.get("status") == "pass") / len(corr_checks))

    replay = ea.get("replay_verification") or {}
    if replay.get("status"):
        rates.append(1.0 if replay.get("status") == "pass" else 0.0)

    if not rates:
        return 100.0
    return round(100.0 * sum(rates) / len(rates), 1)


def _training_recommendation_label(readiness: dict[str, Any]) -> str:
    key = readiness.get("training_recommendation")
    if key in TRAINING_RECOMMENDATION_LABELS:
        return training_recommendation_display(str(key))
    if readiness.get("recommendation"):
        return str(readiness["recommendation"])
    if readiness.get("critical_count", 0) > 0:
        return training_recommendation_display(TRAINING_RECOMMENDATION_NOT_READY)
    return training_recommendation_display(TRAINING_RECOMMENDATION_WARNINGS)


def _priority_label(group: dict[str, Any]) -> str:
    rc = str(group.get("root_cause") or "")
    checks = [str(c) for c in (group.get("affected_checks") or [])]
    items = list(group.get("recommendation_items") or [])
    rec = str(group.get("recommendation") or group.get("recommendation_reason") or "")

    if rc in ("Audit Methodology", "Expected Statistical Behaviour"):
        for chk in checks:
            low = chk.lower()
            if "spot" in low and "delta" in low:
                return "Replace Spot–Delta correlation validation."
        if items:
            return str(items[0]) + "."
    if rc == "Threshold Too Strict" and checks:
        feat = checks[0].replace("_", " ")
        return f"Relax {feat.title()} threshold."
    if items:
        return str(items[0]).rstrip(".") + "."
    if rec:
        line = rec.split("\n")[0].strip()
        return line if line.endswith(".") else line + "."
    if checks:
        return f"Review {checks[0].replace('_', ' ')}."
    return f"Review {rc}."


_RC_PRIORITY_RANK: dict[str, int] = {
    "Audit Methodology": 0,
    "Expected Statistical Behaviour": 0,
    "Configuration Mismatch": 1,
    "Threshold Too Strict": 2,
    "Unable to Classify": 3,
    "Numerical Precision": 4,
    "Builder Bug": 5,
    "Validator Bug": 5,
    "Dataset Issue": 6,
    "Replay Data Issue": 6,
    "Missing Source Data": 7,
    "Performance Issue": 8,
}


def _merged_sort_key(group: dict[str, Any]) -> tuple[int, int, int]:
    severity_rank = {"critical": 0, "warning": 1, "information": 2}
    rc = str(group.get("root_cause") or "")
    return (
        _RC_PRIORITY_RANK.get(rc, 9),
        severity_rank.get(str(group.get("severity") or "warning"), 9),
        -int(group.get("affected_samples") or 0),
    )


def _estimate_fix_minutes(merged: list[dict[str, Any]]) -> int:
    if not merged:
        return 0
    ordered = sorted(merged, key=_merged_sort_key)
    return sum(_RC_FIX_MINUTES.get(str(g.get("root_cause") or ""), 15) for g in ordered[:2])


def _format_fix_time(minutes: int) -> str:
    if minutes <= 0:
        return "None required"
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours, rem = divmod(minutes, 60)
    if rem:
        return f"{hours} hour{'s' if hours != 1 else ''} {rem} minutes"
    return f"{hours} hour{'s' if hours != 1 else ''}"


def build_audit_decision(
    report: dict[str, Any],
    investigations: list[dict[str, Any]],
    merged: list[dict[str, Any]],
) -> dict[str, Any]:
    """Single executive summary for training go/no-go."""
    readiness = build_training_readiness(report)
    summary = report.get("summary") or {}
    ordered_merged = sorted(merged, key=_merged_sort_key)
    priorities = [_priority_label(g) for g in ordered_merged]
    fix_minutes = _estimate_fix_minutes(merged)
    root_causes_found = sum(
        1 for inv in investigations if inv.get("status") not in ("error",)
    )
    info_n = sum(
        1 for inv in investigations
        if inv.get("classification", {}).get("severity") == "information"
    )

    return {
        "title": "AUDIT DECISION",
        "dataset_health_pct": _compute_dataset_health(report),
        "builder_confidence_pct": _compute_builder_confidence(report),
        "validator_confidence_pct": _compute_validator_confidence(report),
        "root_causes_found": root_causes_found,
        "merged_root_causes": len(merged),
        "critical_issues": readiness["critical_count"],
        "warnings": int(summary.get("warnings") or readiness["warnings_count"] or 0),
        "information": info_n,
        "training_recommendation": readiness["training_recommendation"],
        "training_recommendation_display": training_recommendation_display(readiness["training_recommendation"]),
        "training_ready": readiness["ready"],
        "blocking_issues": readiness["blocking_issues"],
        "estimated_fix_time": _format_fix_time(fix_minutes),
        "estimated_fix_minutes": fix_minutes,
        "top_priority": priorities[0] if priorities else None,
        "second_priority": priorities[1] if len(priorities) > 1 else None,
        "priorities": priorities,
    }


def compute_audit_overall_status(report: dict[str, Any]) -> dict[str, str]:
    """Audit PASS/WARN/FAIL for display — FAIL only when critical failures exist."""
    readiness = build_training_readiness(report)
    critical_n = readiness["critical_count"]
    summary = report.get("summary") or {}
    error_count = int(summary.get("errors") or 0)
    warn_count = int(summary.get("warnings") or 0)

    if critical_n > 0:
        return {"status": "fail", "label": "FAIL"}
    if error_count > 0 or warn_count > 0:
        return {"status": "warn", "label": "WARN"}
    return {"status": "pass", "label": "PASS"}


def normalize_audit_status(
    audit_cache: dict[str, Any] | None,
    training_recommendation: str,
) -> str | None:
    """Map cached audit status — non-critical FAILs display as WARN."""
    if not audit_cache:
        return None
    status = audit_cache.get("status")
    if status == "fail" and is_training_allowed(training_recommendation):
        return "warn"
    return status


def build_training_readiness(report: dict[str, Any]) -> dict[str, Any]:
    critical = _critical_failures(report)
    warnings_n = int(
        (report.get("summary") or {}).get("warnings")
        or len(report.get("warnings_list") or report.get("warnings") or [])
    )
    blocking_issues = [c["check"] for c in critical]
    if critical:
        training_recommendation = TRAINING_RECOMMENDATION_NOT_READY
        return {
            "ready": False,
            "label": "Not Ready",
            "training_recommendation": training_recommendation,
            "recommendation": training_recommendation_display(training_recommendation),
            "critical_count": len(critical),
            "critical_failures": critical,
            "blocking_issues": blocking_issues,
            "warnings_count": warnings_n,
        }
    training_recommendation = (
        TRAINING_RECOMMENDATION_READY
        if warnings_n == 0
        else TRAINING_RECOMMENDATION_WARNINGS
    )
    return {
        "ready": True,
        "label": "Ready with Warnings" if warnings_n > 0 else "Ready",
        "training_recommendation": training_recommendation,
        "recommendation": training_recommendation_display(training_recommendation),
        "critical_count": 0,
        "critical_failures": [],
        "blocking_issues": [],
        "warnings_count": warnings_n,
    }


def apply_training_readiness_to_report(report: dict[str, Any]) -> dict[str, Any]:
    """Sync training gate fields on the audit report from critical failures only."""
    tr = build_training_readiness(report)
    report["training_readiness"] = tr
    report["training_recommendation"] = tr["training_recommendation"]

    overall = report.setdefault("overall", {})
    overall["ready_for_training"] = is_training_allowed(tr["training_recommendation"])
    overall["safe_to_train"] = overall["ready_for_training"]
    overall["safe_label"] = tr["recommendation"]

    result = report.setdefault("result", {})
    result["ready_for_training"] = overall["ready_for_training"]
    result["safe_to_train"] = overall["ready_for_training"]
    result["training_readiness_label"] = tr["label"]
    result["training_recommendation"] = tr["training_recommendation"]
    result["blocking_issues"] = tr["blocking_issues"]
    result["critical_failures"] = tr["critical_failures"]

    qs = report.setdefault("quality_score", {})
    qs["ready_for_training"] = overall["ready_for_training"]
    qs["ready_label"] = tr["recommendation"]
    qs["training_recommendation"] = tr["training_recommendation"]

    decision = report.get("audit_decision")
    if isinstance(decision, dict):
        decision["training_recommendation"] = tr["training_recommendation"]
        decision["training_recommendation_display"] = tr["recommendation"]
        decision["training_ready"] = overall["ready_for_training"]
        decision["blocking_issues"] = tr["blocking_issues"]
        decision["critical_issues"] = tr["critical_count"]

    conclusion = report.get("audit_conclusion")
    if isinstance(conclusion, dict):
        conclusion["training_readiness"] = tr["label"]
        conclusion["training_ready"] = overall["ready_for_training"]
        conclusion["training_recommendation"] = tr["training_recommendation"]
        conclusion["training_recommendation_display"] = tr["recommendation"]
        conclusion["blocking_issues"] = tr["blocking_issues"]

    audit_overall = compute_audit_overall_status(report)
    overall["status"] = audit_overall["status"]
    overall["label"] = audit_overall["label"]
    report["passed"] = tr["critical_count"] == 0

    return tr


def build_audit_conclusion(
    report: dict[str, Any],
    investigations: list[dict[str, Any]],
    merged: list[dict[str, Any]],
) -> dict[str, Any]:
    overall = (report.get("overall") or {}).get("status") or "pass"
    summary = report.get("summary") or {}
    readiness = build_training_readiness(report)
    critical_n = readiness["critical_count"]
    warn_n = readiness["warnings_count"]
    info_n = sum(1 for inv in investigations if inv.get("classification", {}).get("severity") == "information")

    if critical_n > 0:
        status = "FAIL"
    elif warn_n > 0 or overall == "warn":
        status = "WARNING"
    else:
        status = "PASS"

    return {
        "title": "FINAL AUDIT CONCLUSION",
        "overall_status": status,
        "critical_errors": critical_n,
        "warnings": warn_n,
        "information": info_n,
        "training_readiness": readiness["label"],
        "training_ready": readiness["ready"],
        "training_recommendation": readiness["training_recommendation"],
        "training_recommendation_display": readiness["recommendation"],
        "blocking_issues": readiness["blocking_issues"],
        "investigations_total": len(investigations),
        "investigations_completed": sum(1 for i in investigations if i.get("status") != "error"),
        "merged_root_causes": len(merged),
    }


def build_execution_tree(
    report: dict[str, Any],
    investigations: list[dict[str, Any]],
) -> dict[str, Any]:
    """Hierarchical debugger-style audit tree for the UI."""
    inv_by_key = {
        _investigation_key({"category": i.get("category"), "feature": i.get("feature")}): i
        for i in investigations
    }
    ea = report.get("extended_audit") or {}

    def _status_icon(st: str) -> str:
        if st in ("pass", "done", "passed"):
            return "pass"
        if st in ("fail", "failed"):
            return "fail"
        if st in ("warn", "warning", "configuration_mismatch"):
            return "warn"
        if st in ("info", "information"):
            return "info"
        return "pending"

    def _node(label: str, status: str, children: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        return {"label": label, "status": status, "children": children or []}

    def _timeline_nodes(
        timeline: list[dict[str, Any]] | None,
        step_map: list[tuple[str, str]],
    ) -> list[dict[str, Any]]:
        by_id = {s.get("id"): s for s in (timeline or [])}
        out: list[dict[str, Any]] = []
        for step_id, label in step_map:
            step = by_id.get(step_id)
            if not step:
                continue
            st = _status_icon(step.get("status") or "pass")
            out.append(_node(label, st))
        return out

    independent_steps = [
        ("replay_db", "Replay Data"),
        ("lookback_policy_loaded", "Lookback Policy"),
        ("black_scholes_recalculated", "Greek Calculation"),
        ("difference_computed", "Validator Comparison"),
        ("root_cause_classified", "Root Cause"),
        ("recommendation_generated", "Recommendation"),
    ]
    correlation_steps = [
        ("subset_extracted", "Pearson"),
        ("cross_strike_confound", "Statistical Assumption"),
        ("moneyness_correlation", "Moneyness Analysis"),
        ("recommendation_generated", "Recommendation"),
    ]
    distribution_steps = [
        ("distribution_analysis", "Distribution"),
        ("root_cause_classified", "Bounds"),
        ("recommendation_generated", "Recommendation"),
    ]

    integrity_children = [
        _node("Metadata", _status_icon("pass" if (report.get("files") or {}).get("metadata", {}).get("exists") else "fail")),
        _node("Required Columns", _status_icon("fail" if report.get("errors") else "pass")),
        _node("Duplicate Keys", _status_icon(
            "fail" if int((report.get("integrity_audit") or {}).get("duplicate_rows") or 0) else "pass"
        )),
    ]

    formula_children: list[dict[str, Any]] = []
    fr = (report.get("feature_audit") or {}).get("formula_recalc") or {}
    for grp in fr.get("groups") or []:
        formula_children.append(_node(
            grp.get("label") or grp.get("id") or "—",
            _status_icon(grp.get("status") or "pass"),
        ))

    ind = ea.get("independent_formulas") or {}
    for chk in ind.get("checks") or []:
        feat = chk.get("feature")
        key = f"independent::{feat}"
        inv = inv_by_key.get(key)
        st = _status_icon(chk.get("status") or "pass")
        label = feat or "—"
        sub = _timeline_nodes(inv.get("timeline") if inv else None, independent_steps)
        formula_children.append(_node(label, st, sub if sub else None))

    dist = ea.get("feature_distributions") or {}
    dist_children: list[dict[str, Any]] = []
    for feat in dist.get("features") or []:
        col = feat.get("feature") or feat.get("label")
        key = f"distribution::{col}"
        inv = inv_by_key.get(key)
        st = _status_icon(feat.get("status") or "pass")
        sub = _timeline_nodes(inv.get("timeline") if inv else None, distribution_steps)
        if inv and inv.get("classification"):
            sub = sub or [
                _node("Distribution", "pass"),
                _node("Bounds", st),
                _node("Recommendation", "pass"),
            ]
        dist_children.append(_node(feat.get("label") or col, st, sub if sub else None))

    corr = ea.get("correlation_checks") or {}
    corr_children: list[dict[str, Any]] = []
    for chk in corr.get("checks") or []:
        if chk.get("status") == "info":
            continue
        cid = chk.get("id") or chk.get("check")
        key = f"correlation::{cid}"
        inv = inv_by_key.get(key)
        st = _status_icon(chk.get("status") or "pass")
        label = chk.get("pair_label") or chk.get("check") or cid
        sub = _timeline_nodes(inv.get("timeline") if inv else None, correlation_steps)
        corr_children.append(_node(label, st, sub if sub else None))

    replay = ea.get("replay_verification") or {}
    replay_st = _status_icon(replay.get("status") or "pass")

    conclusion = report.get("audit_conclusion") or {}
    conclusion_st = _status_icon(
        "pass" if conclusion.get("overall_status") == "PASS"
        else ("fail" if conclusion.get("overall_status") == "FAIL" else "warn")
    )

    nodes = [
        _node("Dataset Integrity", "pass", integrity_children),
        _node("Formula Validation", _status_icon(ind.get("status") or fr.get("status") or "pass"), formula_children),
        _node("Distribution Validation", _status_icon(dist.get("status") or "pass"), dist_children),
        _node("Correlation Validation", _status_icon(corr.get("status") or "pass"), corr_children),
        _node("Replay Validation", replay_st),
        _node("Final Audit Conclusion", conclusion_st),
    ]
    return {"label": "Audit", "children": nodes}


def run_auto_investigations(
    report: dict[str, Any],
    *,
    data_dir: str,
    chart_dir: str,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Phase 1–2: auto-investigate WARN/FAIL checks with live timeline."""
    import json as _json

    import pandas as pd

    from .expected_spec import expected_spec_path
    from .writer import _safe_filename, datasets_dir

    dataset_name = report.get("dataset_name") or ""
    targets = collect_investigation_targets(report)
    investigations: list[dict[str, Any]] = []
    by_key: dict[str, dict[str, Any]] = {}

    safe_name = _safe_filename(dataset_name)
    out_dir = datasets_dir(data_dir)
    parquet_path = os.path.join(out_dir, f"{safe_name}.parquet")
    meta_path = os.path.join(out_dir, f"{safe_name}.json")
    expected_path = expected_spec_path(data_dir, safe_name)
    shared_df: pd.DataFrame | None = None
    shared_meta: dict[str, Any] = {}
    shared_expected: dict[str, Any] = {}
    if os.path.isfile(parquet_path):
        try:
            shared_df = pd.read_parquet(parquet_path)
        except Exception:
            shared_df = None
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, encoding="utf-8") as fh:
                shared_meta = _json.load(fh)
        except (OSError, _json.JSONDecodeError):
            shared_meta = {}
    if os.path.isfile(expected_path):
        try:
            with open(expected_path, encoding="utf-8") as fh:
                shared_expected = _json.load(fh)
        except (OSError, _json.JSONDecodeError):
            shared_expected = {}

    def emit_dashboard(**extra: Any) -> None:
        if not on_progress:
            return
        completed = sum(1 for i in investigations if i.get("status") != "error")
        on_progress({
            "phase": "dashboard",
            "running_stage": extra.get("running_stage", "investigation"),
            "current_investigation": extra.get("current_investigation"),
            "investigations_total": len(targets),
            "investigations_completed": completed,
            "investigations_pending": len(targets) - completed,
            "warnings": sum(1 for t in targets if t.get("status") in ("warn", "configuration_mismatch")),
            "failures": sum(1 for t in targets if t.get("status") == "fail"),
            **system_metrics(),
            **extra,
        })

    emit_dashboard(running_stage="auto_investigation", current_investigation=None)

    for idx, target in enumerate(targets):
        feat = str(target.get("feature") or "")
        cat = str(target.get("category") or "")
        check_label = str(target.get("check") or feat)
        key = _investigation_key(target)

        emit_dashboard(
            running_stage="investigation",
            current_investigation=check_label,
            current_index=idx + 1,
        )

        timeline = InvestigationTimeline(
            on_progress,
            feature=feat,
            category=cat,
            check_label=check_label,
        )

        failure_sample = target.get("failure_sample") or {}
        try:
            result = investigate_audit_failure(
                data_dir=data_dir,
                chart_dir=chart_dir,
                dataset_name=dataset_name,
                category=cat,
                feature=feat,
                trading_day=failure_sample.get("trading_day"),
                timestamp=failure_sample.get("timestamp"),
                strike=failure_sample.get("strike"),
                option_type=failure_sample.get("option_type"),
                audit_failure_count=target.get("audit_failure_count"),
                timeline=timeline,
                persist_history=False,
                audit_context=target.get("audit_row"),
                fast_mode=True,
                df=shared_df,
                meta_doc=shared_meta,
                expected_doc=shared_expected,
            )
            classification = classify_root_cause(result)
            inv_doc = {
                "key": key,
                "category": cat,
                "feature": feat,
                "check": check_label,
                "status": result.get("status") or "diagnosed",
                "result": result,
                "timeline": timeline.steps,
                "classification": classification,
                "root_cause_category": classification["root_cause"],
                "confidence": classification["confidence"],
                "classification_confidence": classification["classification_confidence"],
                "severity": classification["severity"],
                "recommended_action": classification["recommended_action"],
                "affected_rows": classification["affected_rows"],
                "affected_features": classification["affected_features"],
            }
            investigations.append(inv_doc)
            by_key[key] = inv_doc
            _archive_investigation(data_dir, dataset_name, report, inv_doc)
            emit_dashboard(
                running_stage="investigation",
                current_investigation=check_label,
                current_index=idx + 1,
                investigations_completed=len(investigations),
                investigations_pending=len(targets) - len(investigations),
            )
        except Exception as exc:
            err_doc = {
                "key": key,
                "category": cat,
                "feature": feat,
                "check": check_label,
                "status": "error",
                "error": str(exc),
                "timeline": timeline.steps,
            }
            investigations.append(err_doc)
            by_key[key] = err_doc

    merged = merge_root_causes(investigations)
    conclusion = build_audit_conclusion(report, investigations, merged)
    execution_tree = build_execution_tree(report, investigations)
    readiness = build_training_readiness(report)
    audit_decision = build_audit_decision(report, investigations, merged)

    return {
        "investigations": investigations,
        "investigations_by_key": by_key,
        "merged_root_causes": merged,
        "audit_conclusion": conclusion,
        "audit_decision": audit_decision,
        "execution_tree": execution_tree,
        "training_readiness": readiness,
    }


def _archive_investigation(
    data_dir: str,
    dataset_name: str,
    report: dict[str, Any],
    inv_doc: dict[str, Any],
) -> None:
    """Phase 8 — persist to investigation-history.json."""
    meta_fp = (report.get("fingerprint") or {}).get("pipeline_fingerprint") or {}
    append_investigation(
        data_dir,
        dataset_name,
        {
            "failed_check": inv_doc.get("check"),
            "feature": inv_doc.get("feature"),
            "category": inv_doc.get("category"),
            "root_cause": inv_doc.get("root_cause_category"),
            "confidence": inv_doc.get("confidence"),
            "severity": inv_doc.get("severity"),
            "recommendation": inv_doc.get("recommended_action"),
            "affected_rows": inv_doc.get("affected_rows"),
            "affected_features": inv_doc.get("affected_features"),
            "timeline": inv_doc.get("timeline"),
            "validator_version": VALIDATOR_VERSION,
            "builder_version": BUILDER_VERSION,
            "dataset_version": (report.get("fingerprint") or {}).get("dataset_version"),
            "spec_hash": (report.get("fingerprint") or {}).get("dataset_spec_hash"),
            "pipeline_fingerprint": meta_fp,
            "full_result": {
                "status": inv_doc.get("status"),
                "classification": inv_doc.get("classification"),
            },
        },
    )


def enrich_report_with_investigations(
    report: dict[str, Any],
    *,
    data_dir: str,
    chart_dir: str,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run auto-investigations and attach to audit report."""
    if on_progress:
        on_progress({"phase": "audit", "step": "auto_investigation", "status": "running"})
    bundle = run_auto_investigations(
        report, data_dir=data_dir, chart_dir=chart_dir, on_progress=on_progress,
    )
    report.update(bundle)
    apply_training_readiness_to_report(report)
    if on_progress:
        on_progress({"phase": "audit", "step": "auto_investigation", "status": "done"})
        on_progress({
            "phase": "conclusion",
            "conclusion": bundle["audit_conclusion"],
            "audit_decision": bundle["audit_decision"],
            "execution_tree": bundle["execution_tree"],
            "merged_root_causes": bundle["merged_root_causes"],
        })
    return report
