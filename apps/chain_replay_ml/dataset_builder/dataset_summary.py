"""Compact dataset summary for registry Summary tab."""

from __future__ import annotations

import json
import os
from typing import Any

from .audit_investigation_engine import (
    is_training_allowed,
    normalize_audit_status,
    normalize_training_recommendation,
    training_recommendation_display,
)
from .auditor import _dataset_artifact_files
from .dataset_validator import audit_cache_path, build_registry_status, load_audit_cache, load_validation_cache
from .expected_spec import DATASET_METADATA_COLUMNS, expected_spec_path
from .golden_regression import golden_regression_status
from .pipeline_identity import BUILDER_VERSION, VALIDATOR_VERSION
from .schema_registry import (
    column_display_name,
    column_interpretation,
    target_predicts_label,
)
from .validation_rules import confidence_weights
from .writer import _safe_filename, datasets_dir

from path_config import CHART_DATA_ROOT as _CHART_DIR
_CONFIDENCE_WEIGHTS: dict[str, tuple[str, int]] = confidence_weights()


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_feature_registry() -> dict[str, Any]:
    from .schema_registry import load_feature_registry

    return load_feature_registry()


def _target_predicts_label(column: str) -> str:
    return target_predicts_label(column)


def _fmt_expiry_label(expiry: str | None) -> str:
    if not expiry:
        return "—"
    raw = str(expiry)[:10]
    try:
        from datetime import datetime

        return datetime.strptime(raw, "%Y-%m-%d").strftime("%d%b%y").upper()
    except ValueError:
        return raw


def _fmt_build_duration(meta: dict[str, Any]) -> str:
    perf = meta.get("build_performance") or {}
    label = perf.get("total_elapsed_label")
    if label:
        return str(label).replace("m ", " min ").replace("s", " sec")
    sec = perf.get("total_elapsed_sec")
    if sec is None:
        return "—"
    sec = float(sec)
    if sec >= 60:
        mins = int(sec // 60)
        rem = int(round(sec - mins * 60))
        return f"{mins} min {rem} sec"
    return f"{int(round(sec))} sec"


def _feature_group_coverage(meta: dict[str, Any], expected_doc: dict[str, Any]) -> list[dict[str, Any]]:
    registry = _load_feature_registry()
    enabled = list(meta.get("feature_groups") or expected_doc.get("feature_groups") or [])
    if not enabled:
        return []
    implemented_cols = set(meta.get("feature_columns") or [])
    order = list(registry.get("groupOrder") or enabled)
    groups_meta = registry.get("groups") or {}
    rows: list[dict[str, Any]] = []
    for gid in order:
        if gid not in enabled:
            continue
        block = groups_meta.get(gid) or {}
        feats = list(block.get("features") or [])
        expected_n = len(feats)
        actual_n = sum(1 for feat in feats if feat in implemented_cols)
        complete = expected_n > 0 and actual_n >= expected_n
        label = str(block.get("label") or gid)
        rows.append({
            "id": gid,
            "label": label,
            "expected": expected_n,
            "implemented": actual_n,
            "complete": complete,
            "display": f"{label} ({actual_n}/{expected_n})",
        })
    return rows


def _trading_day_labels(meta: dict[str, Any]) -> str:
    """Human-readable trading-day list for registry summary views.

    Prefers concrete resolved dates (``trading_day_filter`` dates, then the
    actually-exported ``days``/``sources`` list) over a vague scope label
    like "All days" — every current builder (Master Dataset panel and
    Feature Transformation Auto) records those explicit dates. Older
    datasets that only stored the ``all_days``/``selected_days`` flag still
    fall back to the label so nothing breaks on legacy metadata.
    """
    tdf = meta.get("trading_day_filter") if isinstance(meta.get("trading_day_filter"), dict) else {}
    tdf_dates = sorted({
        str(d).strip()
        for d in (tdf.get("exported_dates") or tdf.get("selected_dates") or [])
        if str(d).strip()
    })
    if tdf_dates:
        return ", ".join(tdf_dates)

    blocks = meta.get("days") or meta.get("sources") or []
    labels = sorted({
        str(d.get("trading_day") or "").strip()
        for d in blocks
        if isinstance(d, dict) and str(d.get("trading_day") or "").strip()
    })
    if labels:
        return ", ".join(labels)

    from .dataset_selection_engine import format_day_scope_label

    mf = meta.get("master_filter") if isinstance(meta.get("master_filter"), dict) else {}
    if mf:
        label = format_day_scope_label(
            all_days=bool(mf.get("all_days")),
            selected_days=mf.get("selected_days"),
            trading_day=mf.get("trading_day"),
        )
        if label != "—":
            return label

    sm = meta.get("selection_method")
    if isinstance(sm, dict):
        summary = str(sm.get("summary") or "").strip()
        if summary and summary != "—":
            return summary.split(" · ")[0]
    return "—"


def _master_day_rows_for_filter_backfill(meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Read day-tag metadata from the source master DB when filter dates were not stored."""
    rel = str(meta.get("master_db_path") or "").strip()
    if not rel:
        return []
    # build_dataset_summary works from data_dir; master path in meta is relative to data_dir.
    # Callers pass absolute via temporary injection; try common resolves.
    candidates: list[str] = []
    data_dir = str(meta.get("_data_dir") or "").strip()
    if data_dir and rel:
        candidates.append(os.path.normpath(os.path.join(data_dir, rel)))
    if os.path.isabs(rel):
        candidates.append(rel)
    for path in candidates:
        if not path or not os.path.isfile(path):
            continue
        try:
            from .master_store import MasterStore

            store = MasterStore(path)
            store.open()
            try:
                return store.read_master_days()
            finally:
                store.close()
        except Exception:
            continue
    return []


def build_filter_summary_rows(meta: dict[str, Any]) -> list[dict[str, str]]:
    """Structured master-filter rows for registry Summary tab."""
    from .expected_spec import format_strike_selection_label, strike_selection_display_label

    rows: list[dict[str, str]] = []
    mf = meta.get("master_filter") if isinstance(meta.get("master_filter"), dict) else {}
    sm = meta.get("selection_method") if isinstance(meta.get("selection_method"), dict) else {}
    crit: dict[str, Any] = dict(mf)
    if not crit and isinstance(sm.get("criteria"), dict):
        crit = dict(sm["criteria"])
    if crit.get("no_null_data") is None and meta.get("no_null_dropped_columns"):
        crit["no_null_data"] = True

    day_labels = _trading_day_labels(meta)
    if day_labels and day_labels != "—":
        rows.append({"label": "Trading dates", "value": day_labels})

    from .trading_day_filter import trading_day_filter_summary_rows

    exported_from_meta = [
        str(d.get("trading_day") or "").strip()
        for d in (meta.get("days") or meta.get("sources") or [])
        if isinstance(d, dict) and str(d.get("trading_day") or "").strip()
    ]
    master_day_rows: list[dict[str, Any]] = []
    tdf = meta.get("trading_day_filter") if isinstance(meta.get("trading_day_filter"), dict) else None
    needs_backfill = bool(
        tdf
        and str(tdf.get("mode") or "") in ("exclude_expiry", "expiry_only")
        and not (tdf.get("excluded_dates") or tdf.get("expiry_dates") or tdf.get("selected_dates"))
    )
    if needs_backfill:
        master_day_rows = _master_day_rows_for_filter_backfill(meta)

    rows.extend(
        trading_day_filter_summary_rows(
            tdf,
            exported_dates=exported_from_meta,
            master_day_rows=master_day_rows or None,
        )
    )

    token = crit.get("token")
    if token:
        rows.append({"label": "Token", "value": str(token)})

    atm = crit.get("atm_band_filter")
    if atm is not None and str(atm).lower() != "all":
        try:
            rows.append({"label": "ATM band", "value": f"±{int(atm)}"})
        except (TypeError, ValueError):
            rows.append({"label": "ATM band", "value": str(atm)})
    elif atm is not None and str(atm).lower() == "all":
        rows.append({"label": "ATM band", "value": "All"})
    else:
        strike_sel = meta.get("strike_selection") if isinstance(meta.get("strike_selection"), dict) else {}
        strike_label = strike_selection_display_label(meta) or format_strike_selection_label(strike_sel)
        if strike_label:
            rows.append({"label": "Strike selection", "value": strike_label})
        elif crit or mf or sm:
            rows.append({"label": "ATM band", "value": "All"})

    prem_on = bool(
        crit.get("premium_enabled")
        or (crit.get("premium_min") is not None and crit.get("premium_max") is not None)
    )
    if prem_on and crit.get("premium_min") is not None and crit.get("premium_max") is not None:
        rows.append({
            "label": "LTP / Premium",
            "value": f"{float(crit['premium_min']):g}–{float(crit['premium_max']):g}",
        })
    else:
        rows.append({"label": "LTP / Premium", "value": "Off"})

    delta_on = bool(
        crit.get("delta_enabled")
        or (crit.get("delta_min") is not None and crit.get("delta_max") is not None)
    )
    if delta_on and crit.get("delta_min") is not None and crit.get("delta_max") is not None:
        rows.append({
            "label": "Delta",
            "value": f"|δ| {float(crit['delta_min']):g}–{float(crit['delta_max']):g}",
        })
    else:
        rows.append({"label": "Delta", "value": "Off"})

    if crit.get("no_null_data"):
        rows.append({"label": "No null data", "value": "On"})
    else:
        rows.append({"label": "No null data", "value": "Off"})

    dropped = meta.get("no_null_dropped_columns")
    if isinstance(dropped, list) and dropped:
        rows.append({
            "label": "Dropped all-null columns",
            "value": str(len(dropped)),
        })

    if not rows and isinstance(sm, dict):
        summary = str(sm.get("summary") or "").strip()
        if summary and summary != "—":
            rows.append({"label": "Selection", "value": summary})

    return rows


def build_dataset_build_snapshot(
    meta: dict[str, Any],
    *,
    dataset_name: str | None = None,
    snapshotted_at: str | None = None,
) -> dict[str, Any]:
    """Freeze dataset registry build metadata for model packages."""
    from datetime import datetime, timezone

    from .expected_spec import sampling_interval_display_label, strike_selection_display_label

    if not isinstance(meta, dict) or not meta:
        return {}

    sampling_label = sampling_interval_display_label(meta) or "—"
    strikes_label = strike_selection_display_label(meta) or "—"
    lineage = _build_dataset_lineage(
        meta,
        sampling_label=sampling_label,
        strikes_label=strikes_label,
    )
    filter_summary = build_filter_summary_rows(meta)
    selection_method = meta.get("selection_method")
    master_filter = meta.get("master_filter")
    trading_day_filter = meta.get("trading_day_filter")
    dropped = meta.get("no_null_dropped_columns")
    pipeline_fp = meta.get("pipeline_fingerprint")

    return {
        "dataset_name": str(dataset_name or meta.get("dataset_name") or "").strip() or None,
        "snapshotted_at": snapshotted_at or datetime.now(timezone.utc).isoformat(),
        "export_source": meta.get("export_source") or meta.get("source"),
        "created_at": meta.get("created_at"),
        "market": meta.get("market"),
        "row_count": meta.get("row_count"),
        "feature_count": meta.get("feature_count"),
        "target_count": meta.get("target_count"),
        "column_count": meta.get("column_count"),
        "trading_days": meta.get("trading_days") or lineage.get("trading_days"),
        "trading_day_labels": lineage.get("trading_day_labels"),
        "selection_method": dict(selection_method) if isinstance(selection_method, dict) else selection_method,
        "master_filter": dict(master_filter) if isinstance(master_filter, dict) else None,
        "trading_day_filter": dict(trading_day_filter) if isinstance(trading_day_filter, dict) else None,
        "filter_summary": filter_summary,
        "dataset_lineage": lineage,
        "sampling": dict(meta["sampling"]) if isinstance(meta.get("sampling"), dict) else None,
        "sampling_label": sampling_label,
        "strike_selection": dict(meta["strike_selection"]) if isinstance(meta.get("strike_selection"), dict) else None,
        "strike_selection_label": strikes_label,
        "pipeline_fingerprint": dict(pipeline_fp) if isinstance(pipeline_fp, dict) else None,
        "no_null_dropped_columns": list(dropped) if isinstance(dropped, list) and dropped else None,
        "builder_version": meta.get("builder_version") or meta.get("dataset_version"),
        "dataset_version": meta.get("dataset_version") or meta.get("builder_version"),
        "audit_validation_required": meta.get("audit_validation_required"),
    }


def _build_dataset_lineage(
    meta: dict[str, Any],
    *,
    sampling_label: str,
    strikes_label: str,
) -> dict[str, Any]:
    days = meta.get("days") or []
    expiries: list[str] = []
    for day in days:
        if isinstance(day, dict):
            exp = str(day.get("expiry") or "")[:10]
        else:
            continue
        if exp and exp not in expiries:
            expiries.append(exp)
    builder_version = str(meta.get("builder_version") or meta.get("dataset_version") or BUILDER_VERSION)
    return {
        "trading_days": int(meta.get("trading_days") or len(days) or 0),
        "trading_day_labels": _trading_day_labels(meta),
        "expiry": _fmt_expiry_label(expiries[0] if expiries else None),
        "expiries": [_fmt_expiry_label(e) for e in expiries],
        "sampling": sampling_label,
        "atm_band": strikes_label,
        "source_db": f"chain replay v{builder_version.lstrip('v')}",
        "build_time": _fmt_build_duration(meta),
        "created_at": meta.get("created_at"),
    }


def _audit_finding_links(merged_causes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    warnings: list[dict[str, Any]] = []
    information: list[dict[str, Any]] = []
    for idx, group in enumerate(merged_causes):
        severity = str(group.get("severity") or "").lower()
        checks = [str(c) for c in (group.get("affected_checks") or [])]
        anchor = f"audit-finding-{idx}"
        if checks and checks[0].replace("_", "-").isascii():
            anchor = f"audit-finding-{checks[0].replace('_', '-')}"
        entry = {
            "anchor": anchor,
            "index": idx,
            "severity": severity or "information",
            "root_cause": group.get("root_cause"),
            "affected_checks": checks,
            "title": checks[0] if checks else str(group.get("root_cause") or "Finding"),
            "recommendation": group.get("recommendation"),
        }
        if severity == "warning":
            warnings.append(entry)
        elif severity == "information":
            information.append(entry)
        else:
            information.append(entry)
    return {"warnings": warnings, "information": information}


def _validation_label(status: str | None, label: str | None = None) -> str:
    if label:
        return str(label).upper()
    st = str(status or "").lower()
    if st == "pass":
        return "PASS"
    if st == "warn":
        return "WARN"
    if st == "fail":
        return "FAIL"
    return "—"


def _comparison_pass_rate(comparisons: int, failed: int) -> float | None:
    if comparisons <= 0:
        return None
    return round(100.0 * max(0, comparisons - failed) / comparisons, 1)


def _validation_block(
    *,
    features_checked: int = 0,
    comparisons: int = 0,
    comparisons_failed: int = 0,
    rows_sampled: int = 0,
    status: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    lbl = _validation_label(status, label)
    comp = int(comparisons or 0)
    failed = int(comparisons_failed or 0)
    passed = max(0, comp - failed)
    return {
        "features_checked": int(features_checked or 0),
        "comparisons": comp,
        "comparisons_failed": failed,
        "comparisons_passed": passed,
        "rows_sampled": int(rows_sampled or 0),
        "status": status,
        "label": lbl,
        "pass_rate_pct": _comparison_pass_rate(comp, failed),
        "summary": f"{comp:,} comparisons · {lbl}" if comp else "—",
    }


def _target_stats_from_parquet(
    parquet_path: str,
    target_columns: list[str],
    *,
    total_rows: int,
) -> list[dict[str, Any]]:
    if not target_columns or not os.path.isfile(parquet_path):
        return []
    try:
        import pandas as pd
    except ImportError:
        return []

    cols = [c for c in target_columns if c]
    if not cols:
        return []
    try:
        df = pd.read_parquet(parquet_path, columns=cols)
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for col in cols:
        if col not in df.columns:
            rows.append({
                "target": col,
                "display_name": column_display_name(col),
                "predicts": _target_predicts_label(col),
                "target_type": "Regression",
                "rows": total_rows,
                "missing": total_rows,
                "mean": None,
                "std_dev": None,
                "min": None,
                "max": None,
            })
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        valid = series.dropna()
        rows.append({
            "target": col,
            "display_name": column_display_name(col),
            "predicts": _target_predicts_label(col),
            "target_type": "Regression",
            "rows": total_rows,
            "missing": int(series.isna().sum()),
            "mean": round(float(valid.mean()), 4) if len(valid) else None,
            "std_dev": round(float(valid.std()), 4) if len(valid) > 1 else (0.0 if len(valid) == 1 else None),
            "min": round(float(valid.min()), 4) if len(valid) else None,
            "max": round(float(valid.max()), 4) if len(valid) else None,
        })
    return rows


def _compute_dataset_confidence(
    *,
    quality: dict[str, Any],
    formula_validation: dict[str, Any],
    replay_validation: dict[str, Any],
    audit: dict[str, Any],
    spec_hash_match: bool | None,
    coverage_pct: float,
    has_audit_cache: bool,
) -> dict[str, Any]:
    integ_issues = (
        int(quality.get("duplicate_rows") or 0)
        + int(quality.get("invalid_timestamps") or 0)
        + int(quality.get("invalid_strike_rows") or 0)
        + int(quality.get("unexpected_nulls") or 0)
        + int(quality.get("missing_targets") or 0)
    )
    integrity_pct = 100.0 if integ_issues == 0 else max(0.0, 100.0 - min(40.0, integ_issues))

    formula_pct = formula_validation.get("pass_rate_pct")
    if formula_pct is None and formula_validation.get("comparisons"):
        formula_pct = _comparison_pass_rate(
            int(formula_validation["comparisons"]),
            int(formula_validation.get("comparisons_failed") or 0),
        )
    if formula_pct is None:
        formula_pct = 100.0 if str(formula_validation.get("status") or "").lower() == "pass" else None

    replay_pct = replay_validation.get("pass_rate_pct")
    if replay_pct is None and replay_validation.get("comparisons"):
        replay_pct = _comparison_pass_rate(
            int(replay_validation["comparisons"]),
            int(replay_validation.get("comparisons_failed") or 0),
        )
    if replay_pct is None:
        replay_pct = 100.0 if str(replay_validation.get("status") or "").lower() == "pass" else None

    audit_pct = audit.get("dataset_health_pct")
    if audit_pct is None:
        st = str(audit.get("status_raw") or audit.get("status") or "").lower()
        if st == "pass":
            audit_pct = 100.0
        elif st == "warn":
            audit_pct = 92.0
        elif st == "fail":
            audit_pct = 40.0
        elif has_audit_cache:
            audit_pct = 85.0
        else:
            audit_pct = None

    if spec_hash_match is True:
        spec_pct = 100.0
    elif spec_hash_match is False:
        spec_pct = 0.0
    else:
        spec_pct = None

    coverage = float(coverage_pct or 0)

    scores: dict[str, float | None] = {
        "integrity": integrity_pct,
        "formula_validation": float(formula_pct) if formula_pct is not None else None,
        "replay_validation": float(replay_pct) if replay_pct is not None else None,
        "feature_coverage": coverage,
        "audit": float(audit_pct) if audit_pct is not None else None,
        "spec_match": spec_pct,
    }

    components: list[dict[str, Any]] = []
    weighted = 0.0
    total_w = 0.0
    for key, (label, weight) in _CONFIDENCE_WEIGHTS.items():
        score = scores.get(key)
        contribution = None
        if score is not None:
            contribution = round(float(score) * weight / 100.0, 2)
            weighted += contribution
            total_w += float(weight)
        components.append({
            "id": key,
            "label": label,
            "weight_pct": weight,
            "score_pct": score,
            "contribution_pct": contribution,
        })

    overall = round(weighted, 1) if total_w else None
    if overall is None and audit.get("dataset_health_pct") is not None:
        overall = float(audit["dataset_health_pct"])

    filled = int(round((overall or 0) / 5)) if overall is not None else 0
    filled = max(0, min(20, filled))
    bar = ("█" * filled) + ("░" * (20 - filled))

    formula_lines = [
        f"{comp['label']:<22} {comp['weight_pct']:>3}%"
        for comp in components
    ]

    return {
        "pct": overall,
        "bar": bar,
        "components": components,
        "formula": {
            "title": "Confidence Formula",
            "lines": formula_lines,
            "final_score_label": f"Final Score = {overall}%" if overall is not None else "Final Score = —",
        },
    }


def _fmt_certification_date(iso: str | None) -> str | None:
    if not iso:
        return None
    try:
        from datetime import datetime

        raw = str(iso).replace("Z", "+00:00")
        dt = datetime.fromisoformat(raw)
        return dt.strftime("%d-%b-%Y")
    except (ValueError, TypeError):
        return None


def _certification_item(label: str, certified: bool | None) -> dict[str, Any]:
    if certified is True:
        display = "✓ Certified"
        status = "certified"
    elif certified is False:
        display = "✗ Not Certified"
        status = "failed"
    else:
        display = "○ Pending"
        status = "pending"
    return {"label": label, "certified": certified, "display": display, "status": status}


def _validation_certified(cache: dict[str, Any] | None) -> bool | None:
    if not cache:
        return None
    st = str(cache.get("status") or "").lower()
    if st in ("pass", "warn"):
        return True
    if st == "fail":
        return False
    label = str(cache.get("label") or "").upper()
    if "PASS" in label or "WARN" in label:
        return True
    if "FAIL" in label:
        return False
    return None


def _replay_certified(replay_validation: dict[str, Any]) -> bool | None:
    if not replay_validation:
        return None
    st = str(replay_validation.get("status") or "").lower()
    label = str(replay_validation.get("label") or "").upper()
    if st == "pass" or label == "PASS":
        return True
    if st == "fail" or label == "FAIL":
        return False
    comparisons = int(replay_validation.get("comparisons") or 0)
    failed = int(replay_validation.get("comparisons_failed") or 0)
    if comparisons > 0 and failed == 0:
        return True
    if comparisons > 0 and failed > 0:
        return False
    return None


def _build_dataset_certification(
    *,
    has_parquet: bool,
    row_count: int,
    audit_cache: dict[str, Any] | None,
    audit_status: str | None,
    critical_issues: int,
    validation_cache: dict[str, Any] | None,
    spec_hash_match: bool | None,
    replay_validation: dict[str, Any],
    training_recommendation: str,
    meta: dict[str, Any],
) -> dict[str, Any]:
    builder_ok: bool | None
    if has_parquet and row_count > 0:
        builder_ok = True
    elif has_parquet:
        builder_ok = False
    else:
        builder_ok = None

    audit_ok: bool | None
    if not audit_cache:
        audit_ok = None
    elif critical_issues > 0 and not is_training_allowed(training_recommendation):
        audit_ok = False
    elif str(audit_status or "").lower() == "fail" and not is_training_allowed(training_recommendation):
        audit_ok = False
    elif is_training_allowed(training_recommendation):
        audit_ok = True
    elif str(audit_status or "").lower() in ("pass", "warn"):
        audit_ok = True
    else:
        audit_ok = False

    spec_ok: bool | None
    if spec_hash_match is True:
        spec_ok = True
    elif spec_hash_match is False:
        spec_ok = False
    else:
        spec_ok = None

    checks = [
        _certification_item("Dataset Builder", builder_ok),
        _certification_item("Audit", audit_ok),
        _certification_item("Validation", _validation_certified(validation_cache)),
        _certification_item("Specification", spec_ok),
        _certification_item("Replay Verification", _replay_certified(replay_validation)),
    ]

    approved = is_training_allowed(training_recommendation)
    overall_display = (
        "Dataset approved for ML Training"
        if approved
        else "Dataset not approved for ML Training"
    )

    cert_date = (
        _fmt_certification_date((audit_cache or {}).get("audited_at"))
        or _fmt_certification_date((validation_cache or {}).get("validated_at"))
        or _fmt_certification_date(meta.get("created_at"))
    )

    validator_v = str(VALIDATOR_VERSION).lstrip("v")
    return {
        "checks": checks,
        "overall": {
            "approved": approved,
            "display": overall_display,
            "status": "certified" if approved else "failed",
        },
        "certification_date": cert_date,
        "certified_by": f"ML Validation Engine v{validator_v}",
    }


def _build_model_training_readiness(
    *,
    data_dir: str,
    safe_name: str,
    meta: dict[str, Any],
    expected_doc: dict[str, Any],
    has_parquet: bool,
    has_expected: bool,
    audit_cache: dict[str, Any] | None,
    validation_cache: dict[str, Any] | None,
    training_recommendation: str,
    audit_label: str,
    audit_status_raw: str | None,
    feature_expected: int,
    feature_implemented: int,
    target_cols: list[str],
) -> dict[str, Any]:
    registry = build_registry_status(
        has_parquet=has_parquet,
        has_expected=has_expected,
        meta=meta,
        audit_cache=audit_cache,
        validation_cache=validation_cache,
    )
    row_count = int(meta.get("row_count") or 0)
    target_expected = int(
        meta.get("target_count")
        or len(expected_doc.get("prediction_target_columns") or [])
        or len(target_cols)
    )

    val_status = (validation_cache or {}).get("status")
    val_label = (validation_cache or {}).get("label") or (
        "PASS" if val_status == "pass" else ("FAIL" if val_status == "fail" else ("WARN" if val_status == "warn" else "PENDING"))
    )

    audit_display = audit_label
    if audit_status_raw == "warn" and audit_label == "WARN":
        audit_display = "PASS (Warnings)"
    elif audit_status_raw == "pass":
        audit_display = "PASS"

    golden = golden_regression_status(data_dir)
    if not golden.get("manifest_configured"):
        golden_display = "Not Applicable"
        golden_status = "na"
    elif str(golden.get("dataset_name") or "") == safe_name:
        golden_display = str(golden.get("label") or golden.get("status") or "PENDING").upper()
        golden_status = str(golden.get("status") or "pending").lower()
    else:
        golden_display = "Not Applicable"
        golden_status = "na"

    train_allowed = is_training_allowed(training_recommendation)
    overall_display = training_recommendation_display(training_recommendation)
    if train_allowed and training_recommendation == "READY":
        overall_short = "READY TO TRAIN"
    elif train_allowed:
        overall_short = "READY TO TRAIN (Warnings)"
    else:
        overall_short = "NOT READY TO TRAIN"

    def _row_status(kind: str, value: str) -> str:
        v = str(value or "").upper()
        if kind == "na" or "NOT APPLICABLE" in v:
            return "na"
        if "FAIL" in v or v == "NOT READY":
            return "fail"
        if "PENDING" in v or v == "—":
            return "pending"
        if "WARN" in v:
            return "warn"
        return "pass"

    checks = [
        {
            "label": "Dataset",
            "value": "PASS" if has_parquet and row_count > 0 else "FAIL",
            "status": _row_status("dataset", "PASS" if has_parquet and row_count > 0 else "FAIL"),
        },
        {
            "label": "Features",
            "value": f"{feature_implemented}/{feature_expected}",
            "status": _row_status(
                "features",
                "PASS" if feature_expected and feature_implemented >= feature_expected else "WARN",
            ),
        },
        {
            "label": "Targets",
            "value": f"{len(target_cols)}/{target_expected}",
            "status": _row_status(
                "targets",
                "PASS" if target_expected and len(target_cols) >= target_expected else "WARN",
            ),
        },
        {
            "label": "Audit",
            "value": audit_display,
            "status": _row_status("audit", audit_display),
        },
        {
            "label": "Validation",
            "value": val_label,
            "status": _row_status("validation", val_label),
        },
        {
            "label": "Golden Regression",
            "value": golden_display,
            "status": golden_status if golden_status != "na" else "na",
        },
    ]

    return {
        "checks": checks,
        "overall": {
            "display": overall_display,
            "short_label": overall_short,
            "training_recommendation": training_recommendation,
            "allowed": train_allowed,
        },
        "lifecycle": registry.get("lifecycle") or {},
    }


def extract_summary_snapshot(report: dict[str, Any]) -> dict[str, Any]:
    """Persist compact audit fields for the Summary tab (written to audit cache)."""
    ia = report.get("integrity_audit") or {}
    sa = report.get("sampling_audit") or {}
    fa = report.get("feature_audit") or {}
    ea = report.get("extended_audit") or {}
    fr = fa.get("formula_recalc") or {}
    ind = ea.get("independent_formulas") or {}
    replay = ea.get("replay_verification") or {}
    skip = sa.get("skip_explanation") or {}
    breakdown = {b.get("id"): int(b.get("count") or 0) for b in (sa.get("missing_breakdown") or [])}
    intentional = int(skip.get("intentional_skip_total") or 0)
    if not intentional:
        intentional = sum(
            breakdown.get(k, 0)
            for k in ("session_start_trimming", "lookback_trimming", "future_target_trimming", "missing_tick_timestamps")
        )
    verified_features = int(fr.get("features_checked") or 0)
    fr_comp = int(fr.get("comparisons") or 0)
    fr_fail = int(fr.get("comparisons_failed") or fr.get("failures") or 0)
    ind_comp = int(ind.get("comparisons") or 0)
    ind_fail = int(ind.get("comparisons_failed") or 0)
    replay_samples = replay.get("samples") or []

    formula_validation = _validation_block(
        features_checked=verified_features,
        comparisons=fr_comp,
        comparisons_failed=fr_fail,
        rows_sampled=int(fr.get("rows_checked") or 0),
        status=fr.get("status"),
        label=fr.get("label"),
    )
    replay_validation = _validation_block(
        comparisons=ind_comp,
        comparisons_failed=ind_fail,
        rows_sampled=int(ind.get("rows_checked") or 0),
        status=ind.get("status"),
        label=ind.get("label"),
    )

    return {
        "integrity": {
            "duplicate_rows": int(ia.get("duplicate_rows") or 0),
            "missing_target_values": int(ia.get("missing_target_values") or 0),
            "invalid_strike_rows": int(ia.get("invalid_strike_rows") or 0),
            "invalid_timestamps": int(ia.get("invalid_timestamps") or 0),
            "missing_feature_values": int(ia.get("missing_feature_values") or 0),
            "expected_nulls": intentional,
            "unexpected_nulls": int(sa.get("unexpected_missing") or skip.get("unexpected_skipped") or 0),
        },
        "features": {
            "expected": int(fa.get("formula_groups_total") or fa.get("features_checked") or 0),
            "implemented": int(fa.get("features_checked") or 0),
            "verified_features": verified_features,
            "coverage_pct": float(fa.get("coverage_pct") or 0),
            "formula_validation": formula_validation,
            "replay_validation": replay_validation,
        },
        "audit_decision": report.get("audit_decision") or {},
        "audit_conclusion": report.get("audit_conclusion") or {},
        "spec_hash_match": (report.get("specification_summary") or {}).get("spec_hash_match"),
        "policies_match": ((report.get("specification_summary") or {}).get("validator") or {}).get("policies_match"),
        "replay_samples": {
            "samples_checked": len(replay_samples),
            "status": replay.get("status"),
            "label": _validation_label(replay.get("status")),
        },
    }


def _fmt_interval(sec: int | float | None) -> str:
    if sec is None:
        return "—"
    s = int(sec)
    if s >= 60 and s % 60 == 0:
        m = s // 60
        return f"{m} Minute{'s' if m != 1 else ''}"
    return f"{s} Second{'s' if s != 1 else ''}"


def _fmt_sampling_method(method: str | None) -> str:
    if not method:
        return "Fixed Interval"
    m = str(method).lower().replace("_", " ")
    return m.title()


def build_dataset_summary(data_dir: str, dataset_name: str) -> dict[str, Any]:
    """Build registry Summary tab payload from metadata + audit/validation caches."""
    safe_name = _safe_filename(dataset_name)
    out_dir = datasets_dir(data_dir)
    meta_path = os.path.join(out_dir, f"{safe_name}.json")
    expected_path = expected_spec_path(data_dir, safe_name)

    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"Metadata not found for {safe_name}")

    meta = _load_json(meta_path)
    meta["_data_dir"] = data_dir
    expected_doc: dict[str, Any] = {}
    if os.path.isfile(expected_path):
        try:
            expected_doc = _load_json(expected_path)
        except (OSError, json.JSONDecodeError):
            expected_doc = {}

    audit_cache = load_audit_cache(data_dir, safe_name)
    validation_cache = load_validation_cache(data_dir, safe_name)
    snapshot = (audit_cache or {}).get("summary_snapshot") or {}
    has_summary_snapshot = bool(snapshot)
    training_recommendation = normalize_training_recommendation(audit_cache or {})
    audit_status_raw = (audit_cache or {}).get("status")
    audit_status = normalize_audit_status(audit_cache, training_recommendation) or audit_status_raw

    sampling_meta = meta.get("sampling") or {}
    strike_meta = meta.get("strike_selection") or {}
    from .expected_spec import resolve_atm_band

    resolved_band = resolve_atm_band(meta)
    band = int(resolved_band) if resolved_band not in (None, "all") else 0
    strikes_per_side = 2 * band + 1 if band else 0

    target_cols = list(
        meta.get("prediction_target_columns")
        or expected_doc.get("prediction_target_columns")
        or []
    )
    if not target_cols and meta.get("prediction_targets"):
        target_cols = [f"future_ltp_{h}" if not str(h).startswith("future") else str(h) for h in meta["prediction_targets"]]

    integrity = snapshot.get("integrity") or {}
    features_snap = snapshot.get("features") or {}
    decision = snapshot.get("audit_decision") or (audit_cache or {}).get("audit_decision") or {}
    conclusion = snapshot.get("audit_conclusion") or (audit_cache or {}).get("audit_conclusion") or {}
    merged_causes = (audit_cache or {}).get("merged_root_causes") or []
    summary_warnings = int(decision.get("warnings") or (audit_cache or {}).get("warnings") or 0)
    summary_information = int(decision.get("information") or conclusion.get("information") or 0)
    if merged_causes:
        merged_warn = sum(1 for g in merged_causes if str(g.get("severity") or "").lower() == "warning")
        merged_info = sum(1 for g in merged_causes if str(g.get("severity") or "").lower() == "information")
        if merged_warn:
            summary_warnings = merged_warn
        if merged_info:
            summary_information = merged_info

    feature_expected = int(
        expected_doc.get("expected_feature_count")
        or meta.get("feature_count")
        or features_snap.get("expected")
        or 0
    )
    feature_implemented = int(meta.get("feature_count") or features_snap.get("implemented") or 0)
    formula_validation = dict(features_snap.get("formula_validation") or {})
    replay_validation = dict(features_snap.get("replay_validation") or {})
    if not formula_validation and features_snap.get("verified") is not None:
        legacy_verified = int(features_snap.get("verified") or 0)
        if legacy_verified <= max(feature_implemented, feature_expected):
            formula_validation = _validation_block(
                features_checked=legacy_verified,
                status="pass" if legacy_verified >= feature_implemented else None,
            )
        else:
            formula_validation = _validation_block(
                comparisons=legacy_verified,
                status="pass",
            )
    if not replay_validation and features_snap.get("replay_verified") is not None:
        legacy_replay = int(features_snap.get("replay_verified") or 0)
        replay_validation = _validation_block(
            comparisons=legacy_replay,
            comparisons_failed=0,
            status="pass",
            label="PASS",
        )
    coverage_pct = features_snap.get("coverage_pct")
    if coverage_pct is None:
        coverage_pct = 100.0 if feature_implemented and feature_implemented >= feature_expected else 0.0

    parquet_path = os.path.join(out_dir, f"{safe_name}.parquet")
    has_parquet = os.path.isfile(parquet_path)
    row_count = int(meta.get("row_count") or 0)
    target_details = _target_stats_from_parquet(
        parquet_path,
        target_cols,
        total_rows=row_count,
    )

    audit_label = "—"
    if audit_status == "pass":
        audit_label = "PASS"
    elif audit_status == "warn":
        audit_label = "WARN"
    elif audit_status == "fail":
        audit_label = "FAIL"
    elif audit_cache:
        audit_label = str(audit_cache.get("label") or "—").upper()

    spec_hash_match = snapshot.get("spec_hash_match")
    if spec_hash_match is None and audit_cache:
        spec_hash_match = audit_cache.get("spec_hash_match")

    files = _dataset_artifact_files(data_dir, safe_name)
    file_rows = [
        {
            "key": f["key"],
            "label": f["label"],
            "exists": bool(f.get("exists")),
        }
        for f in files
    ]

    fp = meta.get("pipeline_fingerprint") or expected_doc.get("pipeline_fingerprint") or {}

    quality_block = {
        "duplicate_rows": int(integrity.get("duplicate_rows") or 0),
        "missing_targets": int(integrity.get("missing_target_values") or 0),
        "invalid_strike_rows": int(integrity.get("invalid_strike_rows") or 0),
        "invalid_timestamps": int(integrity.get("invalid_timestamps") or 0),
        "expected_nulls": int(integrity.get("expected_nulls") or 0),
        "unexpected_nulls": int(integrity.get("unexpected_nulls") or 0),
    }
    audit_block = {
        "status": audit_label,
        "status_raw": audit_status,
        "audited_at": (audit_cache or {}).get("audited_at"),
        "critical_issues": int(decision.get("critical_issues") or conclusion.get("critical_errors") or 0),
        "warnings": summary_warnings,
        "information": summary_information,
        "builder_confidence_pct": decision.get("builder_confidence_pct"),
        "dataset_health_pct": decision.get("dataset_health_pct"),
        "training_recommendation": training_recommendation,
        "training_recommendation_display": training_recommendation_display(training_recommendation),
        "blocking_issues": list(decision.get("blocking_issues") or (audit_cache or {}).get("blocking_issues") or []),
        "findings": _audit_finding_links(merged_causes),
    }
    feature_summary_block = {
        "expected": feature_expected,
        "implemented": feature_implemented,
        "coverage_pct": round(float(coverage_pct), 1),
        "formula_validation": formula_validation,
        "replay_validation": replay_validation,
        "formula_display": (
            (
                f"{formula_validation.get('features_checked', 0):,} features · "
                f"{formula_validation.get('comparisons', 0):,} comparisons · "
                f"{formula_validation.get('label', '—')}"
            )
            if formula_validation.get("comparisons")
            else (
                f"{formula_validation.get('features_checked', 0):,} features verified · "
                f"{formula_validation.get('label', '—')}"
                if formula_validation.get("features_checked")
                else "—"
            )
        ),
        "replay_display": replay_validation.get("summary") or "—",
    }
    dataset_confidence = _compute_dataset_confidence(
        quality=quality_block,
        formula_validation=formula_validation,
        replay_validation=replay_validation,
        audit=audit_block,
        spec_hash_match=spec_hash_match,
        coverage_pct=float(coverage_pct),
        has_audit_cache=audit_cache is not None,
    )
    model_training_readiness = _build_model_training_readiness(
        data_dir=data_dir,
        safe_name=safe_name,
        meta=meta,
        expected_doc=expected_doc,
        has_parquet=has_parquet,
        has_expected=os.path.isfile(expected_path),
        audit_cache=audit_cache,
        validation_cache=validation_cache,
        training_recommendation=training_recommendation,
        audit_label=audit_label,
        audit_status_raw=audit_status,
        feature_expected=feature_expected,
        feature_implemented=feature_implemented,
        target_cols=target_cols,
    )

    interval_label = _fmt_interval(sampling_meta.get("interval_sec") or expected_doc.get("sampling_interval_sec") or 10)
    strikes_label = f"±{band} ({strikes_per_side} Strikes)" if band else "—"
    prediction_target_rows = [
        {
            "target": col,
            "display_name": column_display_name(col),
            "predicts": _target_predicts_label(col),
            "interpretation": column_interpretation(col),
        }
        for col in target_cols
    ]
    feature_group_coverage = _feature_group_coverage(meta, expected_doc)
    dataset_lineage = _build_dataset_lineage(
        meta,
        sampling_label=interval_label,
        strikes_label=f"±{band}" if band else "—",
    )
    dataset_certification = _build_dataset_certification(
        has_parquet=has_parquet,
        row_count=row_count,
        audit_cache=audit_cache,
        audit_status=audit_status,
        critical_issues=int(decision.get("critical_issues") or conclusion.get("critical_errors") or 0),
        validation_cache=validation_cache,
        spec_hash_match=spec_hash_match,
        replay_validation=replay_validation,
        training_recommendation=training_recommendation,
        meta=meta,
    )

    return {
        "dataset_name": safe_name,
        "status": {
            "training_recommendation": training_recommendation,
            "display": training_recommendation_display(training_recommendation),
        },
        "dataset": {
            "market": meta.get("market") or "—",
            "trading_days": int(meta.get("trading_days") or len(meta.get("days") or []) or 0),
            "trading_day_labels": _trading_day_labels(meta),
            "rows": int(meta.get("row_count") or 0),
            "features": feature_implemented,
            "targets": int(meta.get("target_count") or len(target_cols)),
            "metadata_columns": int(expected_doc.get("expected_metadata_columns") or len(DATASET_METADATA_COLUMNS)),
            "total_columns": int(meta.get("column_count") or 0),
        },
        "sampling": {
            "interval_sec": int(sampling_meta.get("interval_sec") or expected_doc.get("sampling_interval_sec") or 10),
            "interval_label": interval_label,
            "method": _fmt_sampling_method(sampling_meta.get("method")),
            "atm_band": band,
            "strikes_label": strikes_label,
        },
        "prediction_targets": target_cols,
        "prediction_target_rows": prediction_target_rows,
        "target_details": target_details,
        "classifier_labels": meta.get("classifier_labels") or {},
        "rr_enrichment": meta.get("rr_enrichment") or {},
        "feature_group_coverage": feature_group_coverage,
        "dataset_lineage": dataset_lineage,
        "filter_summary": build_filter_summary_rows(meta),
        "master_filter": meta.get("master_filter"),
        "selection_method": meta.get("selection_method"),
        "dataset_certification": dataset_certification,
        "dataset_confidence": dataset_confidence,
        "model_training_readiness": model_training_readiness,
        "feature_summary": feature_summary_block,
        "quality": quality_block,
        "audit": audit_block,
        "validation": {
            "status": (validation_cache or {}).get("status"),
            "label": (validation_cache or {}).get("label"),
            "validated_at": (validation_cache or {}).get("validated_at"),
        },
        "files": file_rows,
        "pipeline": {
            "dataset_version": meta.get("dataset_version") or meta.get("builder_version"),
            "builder_version": meta.get("builder_version"),
            "validator_version": VALIDATOR_VERSION,
            "spec_hash": meta.get("dataset_spec_hash") or expected_doc.get("dataset_spec_hash") or fp.get("spec_hash"),
            "fingerprint_match": spec_hash_match,
            "policies_match": snapshot.get("policies_match") if snapshot.get("policies_match") is not None else (audit_cache or {}).get("policies_match"),
            "git_commit": meta.get("git_commit") or expected_doc.get("git_commit"),
        },
        "has_audit_cache": audit_cache is not None,
        "has_summary_snapshot": has_summary_snapshot,
    }
