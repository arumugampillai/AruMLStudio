"""Plain-text formatters for dataset registry views (Tk)."""

from __future__ import annotations

import json
from typing import Any


def fmt_num(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "—"


def fmt_lifecycle(status: dict[str, Any] | None) -> str:
    lc = (status or {}).get("lifecycle") or {}
    order = ("build", "audit", "validation", "spec", "training")
    parts: list[str] = []
    for key in order:
        step = lc.get(key)
        if not step:
            continue
        icon = step.get("icon") or "○"
        label = step.get("label") or key
        parts.append(f"{icon} {label}")
    return " | ".join(parts) if parts else "—"


def fmt_readiness(row: dict[str, Any]) -> str:
    readiness = row.get("readiness") or (row.get("status") or {}).get("readiness") or {}
    return str(readiness.get("display") or readiness.get("title") or row.get("training_recommendation") or "—")


def fmt_audit_cell(row: dict[str, Any]) -> str:
    display = row.get("audit_display")
    label = str(row.get("audit_label") or "").upper()
    if not display or display == "—":
        return "—"
    if label == "PASS":
        return f"✅ {display}"
    if label == "WARN":
        return f"⚠ {display}"
    if label == "FAIL":
        return f"❌ {display}"
    if label == "PENDING":
        return f"⏳ {display}"
    return str(display)


def fmt_selection(row: dict[str, Any]) -> str:
    sm = row.get("selection_method")
    if isinstance(sm, dict):
        return str(sm.get("summary") or sm.get("label") or sm.get("method") or sm.get("source") or "—")
    if isinstance(sm, str) and sm.strip():
        return sm.strip()
    src = row.get("selection_source")
    if src:
        return str(src)
    tdf = row.get("trading_day_filter")
    if isinstance(tdf, dict) and tdf.get("mode") and str(tdf.get("mode")) != "all":
        from chain_replay_ml.dataset_builder.trading_day_filter import trading_day_filter_label

        return trading_day_filter_label(str(tdf.get("mode")))
    return "—"


def fmt_files(row: dict[str, Any]) -> str:
    files = row.get("files") or []
    if not files:
        return "—"
    count = sum(1 for f in files if f.get("exists"))
    return f"{count} file{'s' if count != 1 else ''}"


def fmt_created(row: dict[str, Any]) -> str:
    ts = row.get("created_at")
    if not ts:
        return "—"
    try:
        import datetime as dt

        if isinstance(ts, (int, float)):
            return dt.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
        return str(ts)[:19]
    except (TypeError, ValueError, OSError):
        return str(ts)


def format_summary(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    name = summary.get("dataset_name") or "—"
    lines.append(f"Dataset Summary — {name}")
    lines.append("=" * 60)

    ds = summary.get("dataset") or summary.get("overview") or {}
    sampling = summary.get("sampling") or {}
    for key, label, block in (
        ("market", "Market", ds),
        ("trading_days", "Trading days", ds),
        ("rows", "Rows", ds),
        ("row_count", "Rows", ds),
        ("total_columns", "Columns", ds),
        ("column_count", "Columns", ds),
        ("features", "Features", ds),
        ("feature_count", "Features", ds),
        ("targets", "Targets", ds),
        ("target_count", "Targets", ds),
    ):
        val = block.get(key)
        if val is None:
            continue
        if key in ("row_count", "trading_days", "column_count", "feature_count", "target_count", "rows", "features", "targets", "total_columns"):
            val = fmt_num(val)
        if any(f"{label}:" in ln for ln in lines):
            continue
        lines.append(f"  {label}: {val}")

    day_labels = ds.get("trading_day_labels")
    if day_labels and day_labels != "—":
        lines.append(f"  Trading dates: {day_labels}")

    lineage = summary.get("dataset_lineage") or {}
    if not day_labels and lineage.get("trading_day_labels") and lineage.get("trading_day_labels") != "—":
        lines.append(f"  Trading dates: {lineage['trading_day_labels']}")

    filter_rows = summary.get("filter_summary") or []
    if filter_rows:
        lines.append("")
        lines.append("Filters")
        lines.append("-" * 40)
        for row in filter_rows:
            if isinstance(row, dict):
                label = str(row.get("label") or "").strip()
                value = str(row.get("value") or "—").strip()
                if label:
                    lines.append(f"  {label}: {value}")

    if sampling:
        lines.append(f"  Sampling: {sampling.get('interval_label') or sampling.get('interval_sec', '—')}")
        if sampling.get("strikes_label"):
            lines.append(f"  Strikes: {sampling['strikes_label']}")

    status = summary.get("status") or {}
    if status.get("display"):
        lines.append(f"  Training readiness: {status['display']}")

    audit = summary.get("audit") or {}
    if audit:
        lines.append("")
        lines.append("Audit")
        lines.append("-" * 40)
        for key in ("status", "label", "training_recommendation", "warnings", "errors"):
            if key in audit:
                lines.append(f"  {key}: {audit[key]}")

    validation = summary.get("validation") or {}
    if validation:
        lines.append("")
        lines.append("Validation")
        lines.append("-" * 40)
        for key in ("status", "label", "features_checked", "mismatches", "comparisons"):
            if key in validation:
                lines.append(f"  {key}: {validation[key]}")

    targets = summary.get("prediction_targets") or summary.get("targets") or []
    target_details = summary.get("target_details") or []
    if targets or target_details:
        lines.append("")
        lines.append("Targets")
        lines.append("-" * 40)
        items = target_details if target_details else targets
        for t in items[:20]:
            if isinstance(t, dict):
                lines.append(
                    f"  {t.get('target') or t.get('name', '—')}: "
                    f"nulls={t.get('null_count', t.get('nulls', '—'))}"
                )
            else:
                lines.append(f"  {t}")
        if len(items) > 20:
            lines.append(f"  … +{len(items) - 20} more")

    classifier_labels = summary.get("classifier_labels") or {}
    if isinstance(classifier_labels, dict) and classifier_labels:
        lines.append("")
        lines.append("Classifier Labels")
        lines.append("-" * 40)
        for key in (
            "rr_1_1_hit",
            "rr_2_3_hit",
            "rr_1_2_hit",
            "rr_1_3_hit",
            "rr_1_4_hit",
        ):
            if classifier_labels.get(key):
                lines.append(f"  ✓ {key}")
        for key, val in classifier_labels.items():
            if key in (
                "rr_1_1_hit",
                "rr_2_3_hit",
                "rr_1_2_hit",
                "rr_1_3_hit",
                "rr_1_4_hit",
            ):
                continue
            if val:
                lines.append(f"  ✓ {key}")

    rr_enrichment = summary.get("rr_enrichment") or {}
    if isinstance(rr_enrichment, dict) and rr_enrichment:
        lines.append("")
        lines.append("RR Enrichment")
        lines.append("-" * 40)
        for key, label in (
            ("source_dataset", "Source dataset"),
            ("prediction_lab", "Prediction Lab"),
            ("prediction_dataset_version", "Prediction Dataset Version"),
            ("prediction_build_timestamp", "Prediction Build Timestamp"),
            ("join_keys", "Join keys"),
            ("matched", "Matched"),
        ):
            val = rr_enrichment.get(key)
            if val is None or val == "":
                continue
            if key == "join_keys" and isinstance(val, (list, tuple)):
                val = ", ".join(str(x) for x in val)
            elif key == "matched":
                val = fmt_num(val)
            lines.append(f"  {label}: {val}")

    certification = summary.get("dataset_certification") or {}
    if certification:
        lines.append("")
        lines.append("Certification")
        lines.append("-" * 40)
        for k, v in certification.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                lines.append(f"  {k}: {v}")

    confidence = summary.get("dataset_confidence") or {}
    if confidence:
        lines.append("")
        lines.append("Confidence")
        lines.append("-" * 40)
        for k in ("score", "label", "display"):
            if k in confidence:
                lines.append(f"  {k}: {confidence[k]}")

    integrity = summary.get("integrity") or (audit.get("integrity") if isinstance(audit, dict) else {}) or {}
    if integrity:
        lines.append("")
        lines.append("Integrity")
        lines.append("-" * 40)
        for k, v in integrity.items():
            lines.append(f"  {k}: {v}")

    causes = summary.get("root_causes") or summary.get("merged_root_causes") or []
    if causes:
        lines.append("")
        lines.append("Root causes")
        lines.append("-" * 40)
        for g in causes[:15]:
            if isinstance(g, dict):
                lines.append(f"  [{g.get('severity', '?')}] {g.get('title') or g.get('message') or g}")
            else:
                lines.append(f"  {g}")

    return "\n".join(lines)


def format_audit_report(report: dict[str, Any]) -> str:
    lines: list[str] = []
    name = report.get("dataset_name") or "—"
    lines.append(f"Audit Report — {name}")
    lines.append("=" * 60)
    lines.append(f"Status: {report.get('overall_label') or report.get('status') or '—'}")
    lines.append(f"Rows: {fmt_num(report.get('dataset_rows'))}  Columns: {fmt_num(report.get('dataset_columns'))}")
    if report.get("audit_duration_sec") is not None:
        lines.append(f"Duration: {report.get('audit_duration_sec')}s")

    summary = report.get("summary") or {}
    if summary:
        lines.append(f"Warnings: {summary.get('warnings', 0)}  Errors: {summary.get('errors', 0)}")

    readiness = report.get("training_readiness") or report.get("readiness") or {}
    if readiness:
        lines.append(f"Training readiness: {readiness.get('recommendation') or readiness.get('ready')}")

    for section_key, title in (
        ("feature_audit", "Feature audit"),
        ("sampling_audit", "Sampling"),
        ("strike_audit", "Strike selection"),
        ("targets_audit", "Targets"),
        ("integrity_audit", "Integrity"),
        ("performance_audit", "Performance"),
        ("coverage_audit", "Coverage"),
        ("quality_score", "Quality score"),
    ):
        block = report.get(section_key)
        if not block:
            continue
        lines.append("")
        lines.append(title)
        lines.append("-" * 40)
        if isinstance(block, dict):
            for k, v in block.items():
                if isinstance(v, (list, dict)) and len(str(v)) > 200:
                    lines.append(f"  {k}: … ({type(v).__name__})")
                else:
                    lines.append(f"  {k}: {v}")
        else:
            lines.append(f"  {block}")

    errors = report.get("errors") or []
    warnings = report.get("warnings_list") or report.get("warnings") or []
    if errors:
        lines.append("")
        lines.append("Errors")
        for e in errors[:30]:
            lines.append(f"  • {e}")
    if warnings and isinstance(warnings, list):
        lines.append("")
        lines.append("Warnings")
        for w in warnings[:30]:
            lines.append(f"  • {w}")

    investigations = report.get("investigations") or report.get("merged_root_causes") or []
    if investigations:
        lines.append("")
        lines.append("Investigations")
        for inv in investigations[:20]:
            if isinstance(inv, dict):
                lines.append(f"  [{inv.get('severity', '?')}] {inv.get('title') or inv.get('message')}")

    return "\n".join(lines)


def format_compare(result: dict[str, Any]) -> str:
    lines = [
        f"Compare: {result.get('dataset_a')} vs {result.get('dataset_b')}",
        "=" * 60,
    ]
    for row in result.get("rows") or []:
        metric = row.get("metric", "—")
        a = row.get("a", "—")
        b = row.get("b", "—")
        delta = row.get("delta")
        d = f"  (Δ {delta})" if delta else ""
        lines.append(f"  {metric}: {a}  →  {b}{d}")
    return "\n".join(lines)


def format_schema_view(data: dict[str, Any]) -> str:
    ov = data.get("overview") or {}
    lines = [
        "Schema Registry",
        "=" * 60,
        f"Version: {ov.get('schema_version')}",
        f"Columns: {ov.get('column_count')}  Features: {ov.get('feature_count')}  Targets: {ov.get('target_count')}",
        f"Schema hash: {ov.get('schema_registry_hash')}",
        f"Implementation hash: {ov.get('implementation_hash')}",
        f"Validation rules: {ov.get('validation_rules_version')} ({ov.get('validation_rules_hash')})",
        f"Lookback: {ov.get('lookback_policy')} ({ov.get('lookback_policy_hash')})",
        "",
        "Groups",
        "-" * 40,
    ]
    schema = data.get("schema") or {}
    groups = schema.get("groups") or {}
    for gid, gmeta in sorted(groups.items()):
        label = (gmeta or {}).get("label") or gid
        lines.append(f"  {gid}: {label}")
    return "\n".join(lines)


def format_validation_report(report: dict[str, Any]) -> str:
    lines = [
        f"Validation — {report.get('dataset_name', '—')}",
        "=" * 60,
        f"Status: {report.get('status') or report.get('label') or '—'}",
        f"Sample size: {report.get('n_sample_requested') or report.get('n_sample')}",
        f"Tolerance: {report.get('tolerance')}",
    ]
    summary = report.get("summary") or {}
    if summary:
        for k, v in summary.items():
            lines.append(f"  {k}: {v}")
    mismatches = report.get("mismatches") or report.get("feature_mismatches") or []
    if mismatches:
        lines.append("")
        lines.append("Mismatches")
        for m in mismatches[:40]:
            if isinstance(m, dict):
                lines.append(f"  {m.get('feature') or m.get('column')}: {m.get('message') or m}")
            else:
                lines.append(f"  {m}")
    return "\n".join(lines)


def format_golden_status(status: dict[str, Any]) -> str:
    lines = [
        "Golden Dataset Regression",
        "=" * 60,
        f"Dataset: {status.get('dataset_name', 'golden')}{'' if status.get('dataset_exists') else ' (missing)'}",
        f"Status: {status.get('label', 'PENDING')}",
        f"Last checked: {status.get('checked_label', '—')}",
        f"Last mode: {status.get('mode') or '—'}",
        f"Manifest configured: {status.get('manifest_configured')}",
        f"Manifest update allowed: {status.get('manifest_update_allowed')}",
    ]
    if status.get("manifest_update_block_reason"):
        lines.append(f"Manifest gate: {status['manifest_update_block_reason']}")
    manifest = status.get("manifest") or {}
    if manifest:
        lines.append(f"Spec hash (ref): {manifest.get('expected_spec_hash', '—')}")
        lines.append(f"Expected audit: {manifest.get('expected_audit', '—')}")
    lines.append(f"Audit (gate): {status.get('manifest_audit_label', '—')}")
    lines.append(f"Validation (gate): {status.get('manifest_validation_label', '—')}")

    stages = status.get("stage_results") or (status.get("last_run") or {}).get("stage_results") or []
    if stages:
        lines.append("")
        lines.append("Pipeline stages")
        for s in stages:
            icon = "✓" if s.get("status") == "pass" else ("✗" if s.get("status") == "fail" else "○")
            lines.append(f"  {icon} {s.get('label') or s.get('stage')}")

    checks = (status.get("last_run") or {}).get("checks") or []
    visible = [c for c in checks if not str(c.get("check") or "").startswith("stage_")]
    if visible:
        lines.append("")
        lines.append("Last run checks")
        for c in visible:
            icon = "✓" if c.get("status") == "pass" else ("✗" if c.get("status") == "fail" else "○")
            lines.append(f"  {icon} {c.get('check')}: {c.get('actual', '—')}")
    return "\n".join(lines)


def format_merge_plan(plan: dict[str, Any]) -> str:
    lines = [
        f"Feature merge plan — {plan.get('dataset_name', '—')}",
        "=" * 60,
        f"Parquet columns: {plan.get('parquet_column_count', '—')}",
        f"Stored features: {len(plan.get('stored_features') or [])}",
        f"Present in parquet: {len(plan.get('present') or [])}",
        f"Missing from build: {len(plan.get('missing_from_build') or [])}",
        f"Merge candidates: {len(plan.get('merge_candidates') or [])}",
        f"New since build: {len(plan.get('new_since_build') or [])}",
    ]
    missing = plan.get("missing_from_build") or []
    if missing:
        lines.append("")
        lines.append("Missing from parquet (expected at build)")
        for f in missing[:30]:
            lines.append(f"  • {f}")
    return "\n".join(lines)


def format_json(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)
