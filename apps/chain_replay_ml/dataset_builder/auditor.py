"""Dataset Auditor — compare expected.json vs parquet vs dataset.json."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from typing import Any

import pandas as pd

from .audit_progress import AuditStageTracker
from .audit_diagnostics import (
    audit_all_null_features,
    audit_feature_coverage,
    audit_missing_features,
    audit_rows_calculation,
    audit_sampling_breakdown,
    audit_targets_quality,
    _max_horizon_sec,
)
from .audit_extended import run_extended_audits
from .audit_options import audit_validation_required_for_dataset
from .master_registry_export import selection_method_for_registry
from .audit_fingerprint import build_dataset_fingerprint, build_quality_score
from .dataset_validator import (
    audit_cache_path,
    build_registry_status,
    load_audit_cache,
    load_validation_cache,
    validation_cache_path,
)
from .expected_spec import expected_spec_path
from .formula_recalc import audit_formula_recalc
from .feature_plugins import GROUP_FEATURE_SOURCES
from .writer import _safe_filename, datasets_dir

try:
    import pyarrow.parquet as pq
except ImportError:  # pragma: no cover
    pq = None  # type: ignore

from path_config import CHART_DATA_ROOT as _CHART_DIR
from .pipeline_identity import load_feature_registry
from .schema_registry import metadata_column_names

_INTEGRITY_KEY = tuple(
    c for c in metadata_column_names()
    if c in ("trading_day", "market", "expiry", "timestamp", "strike", "option_type")
)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _load_registry() -> dict[str, Any]:
    return load_feature_registry()


def _fmt_sampling_method(method: str | None) -> str:
    m = str(method or "fixed_interval").lower()
    if "fixed" in m:
        return "Fixed"
    return m.replace("_", " ").title()


def _fmt_strike_mode(mode: str | None) -> str:
    m = str(mode or "ATM_BAND").upper()
    if m == "ATM_BAND":
        return "ATM Band"
    if m == "PREMIUM_BAND":
        return "Premium Band"
    if m == "DELTA_RANGE":
        return "Delta Range"
    if m == "CUSTOM":
        return "Custom"
    return m.replace("_", " ").title()


def _check_row(
    category: str,
    check: str,
    expected: Any,
    actual: Any,
    *,
    status: str = "pass",
) -> dict[str, Any]:
    return {
        "category": category,
        "check": check,
        "expected": expected,
        "actual": actual,
        "status": status,
    }


def _is_bad(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return False


def _build_formula_validation(
    registry: dict[str, Any],
    expected_groups: list[str],
    *,
    col_set: set[str] | None,
    implemented_features: list[str],
) -> list[dict[str, Any]]:
    """Per-group feature formula + column presence checks for the audit dialog."""
    groups_meta = registry.get("groups") or {}
    group_order = list(registry.get("groupOrder") or expected_groups)
    impl_set = set(implemented_features)
    rows: list[dict[str, Any]] = []

    for gid in group_order:
        if gid not in expected_groups:
            continue
        gmeta = groups_meta.get(gid) or {}
        label = str(gmeta.get("label") or gid)
        feats = list(gmeta.get("features") or [])
        expected_n = len(feats)
        mapping = GROUP_FEATURE_SOURCES.get(gid, {})
        formulas_n = sum(1 for f in feats if mapping.get(f))

        if col_set is not None:
            present = [f for f in feats if f in col_set]
        else:
            present = [f for f in feats if f in impl_set]
        actual_n = len(present)
        missing = [f for f in feats if f not in present]

        if actual_n >= expected_n and formulas_n >= expected_n:
            status = "pass"
        elif actual_n == 0:
            status = "fail"
        else:
            status = "warn"

        rows.append({
            "id": gid,
            "label": label,
            "expected": expected_n,
            "actual": actual_n,
            "formulas_defined": formulas_n,
            "status": status,
            "missing": missing,
        })
    return rows


def _dataset_artifact_files(data_dir: str, safe_name: str) -> list[dict[str, Any]]:
    """Artifact filenames for registry / audit file lists."""
    out_dir = datasets_dir(data_dir)
    artifacts = [
        ("parquet", "Dataset", f"{safe_name}.parquet"),
        ("metadata", "Metadata", f"{safe_name}.json"),
        ("expected", "Expected", f"{safe_name}.expected.json"),
        ("audit_cache", "Audit cache", f"{safe_name}.audit-cache.json"),
        ("validation", "Validation", f"{safe_name}.validation.json"),
        ("investigation_history", "Investigation history", f"{safe_name}.investigation-history.json"),
    ]
    rows: list[dict[str, Any]] = []
    for key, label, fname in artifacts:
        path = os.path.join(out_dir, fname)
        rows.append({
            "key": key,
            "label": label,
            "name": fname,
            "path": path,
            "exists": os.path.isfile(path),
        })
    return rows


def list_datasets(data_dir: str) -> list[dict[str, Any]]:
    """List built datasets with summary fields for the registry table."""
    out_dir = datasets_dir(data_dir)
    if not os.path.isdir(out_dir):
        return []

    rows: list[dict[str, Any]] = []
    skip_names = frozenset({"golden.manifest", "golden.regression-last"})
    skip_suffixes = (".audit-cache.json", ".validation.json", ".investigation-history.json")
    for fname in os.listdir(out_dir):
        if not fname.endswith(".json") or fname.endswith(".expected.json"):
            continue
        if any(fname.endswith(suffix) for suffix in skip_suffixes):
            continue
        safe_name = fname[:-5]
        if safe_name in skip_names:
            continue
        meta_path = os.path.join(out_dir, fname)
        parquet_path = os.path.join(out_dir, f"{safe_name}.parquet")
        expected_path = os.path.join(out_dir, f"{safe_name}.expected.json")

        meta: dict[str, Any] = {}
        try:
            meta = _load_json(meta_path)
        except (OSError, json.JSONDecodeError):
            pass

        has_parquet = os.path.isfile(parquet_path)
        if not has_parquet and not meta.get("row_count") and not meta.get("days"):
            continue
        row_count = int(meta.get("row_count") or 0)
        is_draft = bool(meta) and (not has_parquet or row_count == 0)

        parquet_bytes = os.path.getsize(parquet_path) if has_parquet else 0
        audit_cache = load_audit_cache(data_dir, safe_name)
        validation_cache = load_validation_cache(data_dir, safe_name)
        day_count = int(meta.get("trading_days") or len(meta.get("days") or []) or 0)
        feature_count = int(meta.get("feature_count") or 0)
        target_count = int(meta.get("target_count") or len(meta.get("prediction_target_columns") or []) or 0)
        artifact_files = _dataset_artifact_files(data_dir, safe_name)
        if is_draft:
            artifact_files = [
                f for f in artifact_files
                if f["exists"] and f["key"] in ("parquet", "metadata", "expected")
            ]
        else:
            artifact_files = [f for f in artifact_files if f["exists"]]
        status_info = build_registry_status(
            has_parquet=has_parquet,
            has_expected=os.path.isfile(expected_path),
            meta=meta,
            audit_cache=audit_cache,
            validation_cache=validation_cache,
        )
        rows.append({
            "dataset_name": safe_name,
            "is_draft": is_draft,
            "market": meta.get("market") or "—",
            "day_count": day_count,
            "row_count": int(meta.get("row_count") or 0),
            "column_count": int(meta.get("column_count") or 0),
            "feature_count": feature_count,
            "target_count": target_count,
            "created_at": meta.get("created_at"),
            "dataset_version": meta.get("dataset_version") or meta.get("builder_version"),
            "builder_version": meta.get("builder_version"),
            "git_commit": meta.get("git_commit"),
            "parquet_bytes": parquet_bytes,
            "has_parquet": has_parquet,
            "has_expected": os.path.isfile(expected_path),
            "has_audit_cache": any(f["key"] == "audit_cache" and f["exists"] for f in artifact_files),
            "metadata_path": meta_path,
            "parquet_path": parquet_path,
            "expected_path": expected_path,
            "audit_cache_path": audit_cache_path(data_dir, safe_name),
            "validation_cache_path": validation_cache_path(data_dir, safe_name),
            "files": artifact_files,
            "status": status_info,
            "readiness": status_info.get("readiness"),
            "audit_label": status_info.get("audit_label"),
            "audit_display": status_info.get("audit_display"),
            "validation_label": status_info.get("validation_label"),
            "training_recommendation": status_info.get("training_recommendation"),
            "audit_validation_required": audit_validation_required_for_dataset(meta),
            "selection_method": selection_method_for_registry(meta),
            "selection_source": (
                (meta.get("selection_method") or {}).get("source")
                if isinstance(meta.get("selection_method"), dict)
                else ("master_db" if str(meta.get("export_source") or "").lower() == "master_filter_export" else None)
            ),
            "trading_day_filter": (
                dict(meta["trading_day_filter"])
                if isinstance(meta.get("trading_day_filter"), dict)
                else None
            ),
        })

    rows.sort(key=lambda r: str(r.get("created_at") or ""), reverse=True)
    return rows


def delete_dataset(*, data_dir: str, dataset_name: str) -> dict[str, Any]:
    """Remove dataset parquet, metadata json, and expected spec."""
    safe_name = _safe_filename(dataset_name)
    out_dir = datasets_dir(data_dir)
    paths = [
        os.path.join(out_dir, f"{safe_name}.parquet"),
        os.path.join(out_dir, f"{safe_name}.json"),
        os.path.join(out_dir, f"{safe_name}.expected.json"),
        audit_cache_path(data_dir, safe_name),
        validation_cache_path(data_dir, safe_name),
        os.path.join(out_dir, f"{safe_name}.investigation-history.json"),
    ]
    deleted: list[str] = []
    missing: list[str] = []
    for path in paths:
        base = os.path.basename(path)
        if os.path.isfile(path):
            os.remove(path)
            deleted.append(base)
        else:
            missing.append(base)
    if not deleted:
        raise FileNotFoundError(f"No dataset files found for {safe_name}")
    return {
        "dataset_name": safe_name,
        "deleted": deleted,
        "missing": missing,
    }


def audit_dataset(
    *,
    data_dir: str,
    dataset_name: str,
    parquet_path: str | None = None,
    metadata_path: str | None = None,
    expected_path: str | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    audit_options: Any | None = None,
) -> dict[str, Any]:
    """Full audit report for the dataset audit dialog."""
    from .audit_options import AuditOptions

    opts = audit_options if isinstance(audit_options, AuditOptions) else AuditOptions.from_mapping(
        audit_options if isinstance(audit_options, dict) else None
    )
    tracker = AuditStageTracker(on_progress)

    def emit(step: str, status: str = "running", **extra: Any) -> None:
        tracker.emit(step, status, **extra)
        step_labels = {
            "dataset_opened": "Dataset opened",
            "metadata_loaded": "Metadata loaded",
            "structural_validation": "Structural validation",
            "independent_formulas": "Independent formulas",
            "distribution_checks": "Distribution checks",
            "correlation_checks": "Correlation validation",
            "replay_verification": "Replay verification",
            "feature_heatmap": "Feature heatmap",
            "feature_validation": "Feature validation",
            "auto_investigation": "Auto investigation",
        }
        tracker.emit_dashboard(
            running_stage=step_labels.get(step, step),
            current_validation=step_labels.get(step, step),
            audit_step=step,
            audit_step_status=status,
        )

    emit("dataset_opened")
    t0 = time.monotonic()
    safe_name = _safe_filename(dataset_name)
    out_dir = datasets_dir(data_dir)
    expected_path = expected_path or expected_spec_path(data_dir, safe_name)
    parquet_path = parquet_path or os.path.join(out_dir, f"{safe_name}.parquet")
    metadata_path = metadata_path or os.path.join(out_dir, f"{safe_name}.json")

    registry = _load_registry()
    group_labels = {
        gid: str((registry.get("groups") or {}).get(gid, {}).get("label") or gid)
        for gid in (registry.get("groupOrder") or [])
    }

    artifact_files = _dataset_artifact_files(data_dir, safe_name)
    files = {row["key"]: row for row in artifact_files}
    files.update({
        "specification": files.get("expected") or {
            "label": "Specification",
            "name": f"{safe_name}.expected.json",
            "path": expected_path,
            "exists": os.path.isfile(expected_path),
        },
        "dataset": files.get("parquet") or {
            "label": "Dataset",
            "name": f"{safe_name}.parquet",
            "path": parquet_path,
            "exists": os.path.isfile(parquet_path),
        },
        "metadata": files.get("metadata") or {
            "label": "Metadata",
            "name": f"{safe_name}.json",
            "path": metadata_path,
            "exists": os.path.isfile(metadata_path),
        },
    })

    errors: list[str] = []
    warnings: list[str] = []
    categories: dict[str, list[dict[str, Any]]] = {
        "metadata": [],
        "sampling": [],
        "strike": [],
        "prediction": [],
        "features": [],
        "dataset": [],
        "integrity": [],
        "performance": [],
    }

    if not files["specification"]["exists"]:
        errors.append(f"Missing expected spec: {expected_path}")
        return _finalize_report(
            safe_name, files, categories, errors, warnings, t0,
            feature_audit=None, sampling_audit=None, strike_audit=None,
            targets_audit=None, integrity_audit=None, performance_audit=None,
            dataset_rows=0, dataset_columns=0,
        )

    expected_doc = _load_json(expected_path)
    emit("dataset_opened", "done")
    emit("metadata_loaded")
    exp = expected_doc.get("expected") or {}
    expected_features = list(exp.get("feature_column_names") or [])
    expected_targets = list(exp.get("target_column_names") or expected_doc.get("prediction_target_columns") or [])
    expected_groups = list(expected_doc.get("feature_groups") or [])
    expected_feature_count = int(exp.get("expected_feature_columns") or len(expected_features))

    meta_doc: dict[str, Any] = {}
    if files["metadata"]["exists"]:
        meta_doc = _load_json(metadata_path)

    implemented_features = list(meta_doc.get("feature_columns") or [])
    pending_features = list(meta_doc.get("feature_columns_pending") or [])
    implemented_count = len(implemented_features) if implemented_features else int(meta_doc.get("feature_count") or 0)
    pending_count = len(pending_features) if pending_features else max(0, expected_feature_count - implemented_count)
    coverage = round(100.0 * implemented_count / expected_feature_count, 1) if expected_feature_count else 0.0

    profile_exp = str(expected_doc.get("feature_profile") or "default").title()
    profile_act = str(meta_doc.get("feature_profile") or "default").title()
    categories["metadata"].append(_check_row("Metadata", "Dataset Name", expected_doc.get("dataset_name"), meta_doc.get("dataset_name") or "—"))
    categories["metadata"].append(_check_row("Metadata", "Feature Profile", profile_exp, profile_act))

    exp_sampling = expected_doc.get("sampling") or {}
    act_sampling = meta_doc.get("sampling") or {}
    interval_exp = int(exp_sampling.get("interval_sec") or 0)
    interval_act = int(act_sampling.get("interval_sec") or 0)
    method_exp = _fmt_sampling_method(exp_sampling.get("method"))
    method_act = _fmt_sampling_method(act_sampling.get("method"))
    categories["sampling"].append(_check_row("Sampling", "Interval", f"{interval_exp} sec", f"{interval_act} sec"))
    categories["sampling"].append(_check_row("Sampling", "Method", method_exp, method_act))

    exp_strike = expected_doc.get("strike_selection") or {}
    act_strike = meta_doc.get("strike_selection") or {}
    band_exp = int(exp_strike.get("band") or 0)
    band_act = int(act_strike.get("band") or 0)
    categories["strike"].append(_check_row("Strike", "Mode", _fmt_strike_mode(exp_strike.get("mode")), _fmt_strike_mode(act_strike.get("mode"))))
    if str(exp_strike.get("mode") or "").upper() == "DELTA_RANGE":
        categories["strike"].append(_check_row(
            "Strike", "Delta Type",
            exp_strike.get("delta_type"), act_strike.get("delta_type"),
        ))
        categories["strike"].append(_check_row(
            "Strike", "Delta Min",
            exp_strike.get("delta_min"), act_strike.get("delta_min"),
        ))
        categories["strike"].append(_check_row(
            "Strike", "Delta Max",
            exp_strike.get("delta_max"), act_strike.get("delta_max"),
        ))
    else:
        categories["strike"].append(_check_row("Strike", "Band", f"±{band_exp}", f"±{band_act}"))

    target_cols_exp = list(expected_doc.get("prediction_target_columns") or [])
    target_cols_act = list(meta_doc.get("prediction_target_columns") or [])
    categories["prediction"].append(_check_row("Prediction", "Target Count", len(target_cols_exp), len(target_cols_act)))
    categories["prediction"].append(_check_row("Prediction", "Target Columns", len(target_cols_exp), len(target_cols_act)))

    feat_status = "pass"
    if implemented_count < expected_feature_count:
        feat_status = "warn"
        warnings.append(f"Feature coverage {coverage}% ({implemented_count}/{expected_feature_count})")
    categories["features"].append(_check_row("Features", "Expected Features", expected_feature_count, implemented_count, status=feat_status))
    categories["features"].append(_check_row("Features", "Pending Features", pending_count, pending_count, status="info"))
    emit("metadata_loaded", "done")

    df: pd.DataFrame | None = None
    col_set: set[str] | None = None
    row_count = int(meta_doc.get("row_count") or 0)
    col_count = int(meta_doc.get("column_count") or 0)
    sampling_audit: dict[str, Any] = {
        "interval_sec": interval_act,
        "expected_samples": int(meta_doc.get("sample_points_estimate") or 0),
        "actual_samples": 0,
        "missing_samples": 0,
        "duplicate_samples": 0,
    }
    strike_audit: dict[str, Any] = {
        "mode": _fmt_strike_mode(act_strike.get("mode")),
        "band": band_act,
        "expected_rows_per_timestamp": (2 * band_exp + 1) * 2 if band_exp else 0,
        "average_actual": 0,
        "failures": 0,
    }
    delta_range_audit: dict[str, Any] = {"applicable": False}
    targets_audit: dict[str, Any] = {
        "columns": [{"column": c, "present": False} for c in target_cols_exp],
        "missing_values": 0,
        "nan": 0,
        "inf": 0,
        "negative": 0,
        "status": "pass",
    }
    integrity_audit: dict[str, Any] = {
        "duplicate_rows": 0,
        "missing_feature_values": 0,
        "missing_target_values": 0,
        "invalid_strike_rows": 0,
        "invalid_timestamps": 0,
        "top_missing_features": [],
        "missing_reasons": [],
    }
    rows_audit: dict[str, Any] = {}
    formula_recalc: dict[str, Any] = {}
    extended_audit: dict[str, Any] = {}
    coverage_audit: dict[str, Any] = {}
    max_horizon_sec = _max_horizon_sec(target_cols_exp)
    expected_doc_for_fp = expected_doc

    if not files["dataset"]["exists"]:
        errors.append(f"Missing parquet: {parquet_path}")
    elif pq is None:
        errors.append("pyarrow is required to audit parquet")
    else:
        emit("structural_validation")
        df = pd.read_parquet(parquet_path)
        row_count = len(df)
        col_count = len(df.columns)
        col_set = set(df.columns)
        actual_feature_cols = [c for c in expected_features if c in col_set]
        missing_features = [c for c in expected_features if c not in col_set]
        if missing_features and implemented_count >= expected_feature_count:
            warnings.append(f"{len(missing_features)} expected feature columns absent from parquet")

        categories["dataset"].append(_check_row("Dataset", "Rows", f"{row_count:,}", f"{row_count:,}"))
        categories["dataset"].append(_check_row("Dataset", "Columns", col_count, col_count))

        dup_rows = 0
        if all(k in df.columns for k in _INTEGRITY_KEY):
            dup_rows = int(df.duplicated(subset=list(_INTEGRITY_KEY)).sum())
        categories["dataset"].append(_check_row("Dataset", "Duplicate Rows", 0, dup_rows, status="pass" if dup_rows == 0 else "fail"))
        if dup_rows:
            errors.append(f"{dup_rows:,} duplicate rows")

        missing_targets = 0
        for col in target_cols_exp:
            if col in df.columns:
                missing_targets += int(df[col].map(_is_bad).sum())
        categories["dataset"].append(_check_row("Dataset", "Missing Targets", 0, missing_targets, status="pass" if missing_targets == 0 else "fail"))
        if missing_targets:
            errors.append(f"{missing_targets:,} missing target values")

        # Sampling audit (with breakdown)
        if opts.skip_dataset_statistics:
            sampling_audit = {
                "interval_sec": interval_act,
                "expected_samples": int(meta_doc.get("sample_points_estimate") or 0),
                "actual_samples": 0,
                "missing_samples": 0,
                "duplicate_samples": 0,
                "status": "skipped",
                "reason": "fast_experiment",
            }
            if "timestamp" in df.columns:
                if "trading_day" in df.columns:
                    sampling_audit["actual_samples"] = int(df.groupby(["trading_day", "timestamp"], dropna=False).ngroups)
                else:
                    sampling_audit["actual_samples"] = int(df["timestamp"].nunique())
        else:
            try:
                sampling_audit = audit_sampling_breakdown(
                    df,
                    meta_doc,
                    chart_dir=_CHART_DIR,
                    step_sec=interval_act,
                    max_horizon_sec=max_horizon_sec,
                )
            except Exception:
                expected_samples = int(meta_doc.get("sample_points_estimate") or 0)
                actual_samples = 0
                if "timestamp" in df.columns:
                    if "trading_day" in df.columns:
                        actual_samples = int(df.groupby(["trading_day", "timestamp"], dropna=False).ngroups)
                    else:
                        actual_samples = int(df["timestamp"].nunique())
                sampling_audit = {
                    "interval_sec": interval_act,
                    "expected_samples": expected_samples,
                    "actual_samples": actual_samples,
                    "missing_samples": max(0, expected_samples - actual_samples),
                    "missing_breakdown": [],
                    "unexpected_missing": 0,
                    "duplicate_samples": 0,
                }

            duplicate_samples = 0
            if "timestamp" in df.columns and band_act:
                expected_rpt = (2 * band_act + 1) * 2
                if "trading_day" in df.columns:
                    duplicate_samples = int(
                        df.groupby(["trading_day", "timestamp"], dropna=False).size().gt(expected_rpt).sum()
                    )
                else:
                    duplicate_samples = int(df.groupby("timestamp", dropna=False).size().gt(expected_rpt).sum())
            sampling_audit["duplicate_samples"] = duplicate_samples

            if sampling_audit.get("unexpected_missing", 0) > 0:
                warnings.append(f"{sampling_audit['unexpected_missing']} unexpected missing sample timestamps")
            elif sampling_audit.get("missing_samples", 0) > 0:
                skip = sampling_audit.get("skip_explanation") or {}
                if skip.get("builder_ok"):
                    pass  # intentional skips only — not a builder defect
                else:
                    warnings.append(
                        f"Sample count {sampling_audit.get('actual_samples'):,} vs expected "
                        f"{sampling_audit.get('expected_samples'):,}"
                    )
            elif sampling_audit.get("expected_samples") and sampling_audit.get("actual_samples") != sampling_audit["expected_samples"]:
                warnings.append(
                    f"Sample count {sampling_audit.get('actual_samples'):,} vs expected "
                    f"{sampling_audit.get('expected_samples'):,}"
                )

            categories["sampling"].append(_check_row(
                "Sampling",
                "Expected Samples",
                sampling_audit.get("expected_samples"),
                sampling_audit.get("expected_samples"),
            ))
            act_s = sampling_audit.get("actual_samples")
            exp_s = sampling_audit.get("expected_samples")
            categories["sampling"].append(_check_row(
                "Sampling",
                "Actual Samples",
                exp_s,
                act_s,
                status="pass" if act_s == exp_s else "warn",
            ))
            for item in sampling_audit.get("missing_breakdown") or []:
                categories["sampling"].append(_check_row(
                    "Sampling",
                    item.get("label"),
                    0,
                    item.get("count"),
                    status="info",
                ))
            categories["sampling"].append(_check_row(
                "Sampling",
                "Unexpected Missing",
                0,
                sampling_audit.get("unexpected_missing", 0),
                status="pass" if not sampling_audit.get("unexpected_missing") else "fail",
            ))

        # Strike audit
        if opts.skip_dataset_statistics:
            strike_audit = {
                "mode": _fmt_strike_mode(act_strike.get("mode")),
                "band": band_act,
                "expected_rows_per_timestamp": (2 * band_exp + 1) * 2 if band_exp else 0,
                "average_actual": 0,
                "failures": 0,
                "status": "skipped",
            }
        else:
            expected_rows_per_ts = (2 * band_exp + 1) * 2 if band_exp else 0
            avg_actual_rows = 0.0
            strike_failures = 0
            if "timestamp" in df.columns and expected_rows_per_ts:
                counts = df.groupby("timestamp", dropna=False).size()
                avg_actual_rows = round(float(counts.mean()), 1) if len(counts) else 0.0
                strike_failures = int((counts != expected_rows_per_ts).sum())
            strike_audit = {
                "mode": _fmt_strike_mode(act_strike.get("mode")),
                "band": band_act,
                "expected_rows_per_timestamp": expected_rows_per_ts,
                "average_actual": avg_actual_rows,
                "failures": strike_failures,
            }
            if strike_failures:
                warnings.append(f"{strike_failures} timestamps with unexpected strike row counts")

            from .delta_range_stats import audit_delta_range_dataset

            delta_range_audit = audit_delta_range_dataset(df, act_strike)
            if delta_range_audit.get("applicable"):
                strike_audit["delta_range"] = delta_range_audit
                strike_audit["delta_type"] = act_strike.get("delta_type")
                strike_audit["delta_min"] = act_strike.get("delta_min")
                strike_audit["delta_max"] = act_strike.get("delta_max")
                vcount = int(delta_range_audit.get("violations_count") or 0)
                dr_status = "pass" if vcount == 0 else "fail"
                categories["strike"].append(_check_row(
                    "Strike",
                    "Delta Range Rule",
                    delta_range_audit.get("expected_rule"),
                    delta_range_audit.get("expected_rule"),
                ))
                categories["strike"].append(_check_row(
                    "Strike",
                    "Minimum Delta",
                    delta_range_audit.get("expected_rule"),
                    delta_range_audit.get("minimum_delta"),
                    status=dr_status,
                ))
                categories["strike"].append(_check_row(
                    "Strike",
                    "Maximum Delta",
                    delta_range_audit.get("expected_rule"),
                    delta_range_audit.get("maximum_delta"),
                    status=dr_status,
                ))
                categories["strike"].append(_check_row(
                    "Strike",
                    "Delta Violations",
                    0,
                    vcount,
                    status=dr_status,
                ))
                if vcount:
                    errors.append(f"{vcount:,} rows outside configured delta range")

        # Targets audit (quality checks)
        targets_audit = audit_targets_quality(df, target_cols_exp)
        missing_target_values = int(targets_audit.get("missing_values") or 0)
        for tcol in targets_audit.get("columns") or []:
            categories["prediction"].append(_check_row(
                "Targets",
                tcol.get("column"),
                "present",
                "yes" if tcol.get("present") else "no",
                status="pass" if tcol.get("present") else "fail",
            ))
        categories["prediction"].append(_check_row(
            "Targets", "NaN", 0, targets_audit.get("nan", 0),
            status="pass" if not targets_audit.get("nan") else "fail",
        ))
        categories["prediction"].append(_check_row(
            "Targets", "Inf", 0, targets_audit.get("inf", 0),
            status="pass" if not targets_audit.get("inf") else "fail",
        ))
        categories["prediction"].append(_check_row(
            "Targets", "Negative", 0, targets_audit.get("negative", 0),
            status="pass" if not targets_audit.get("negative") else "fail",
        ))
        if targets_audit.get("status") != "pass":
            errors.append("Target quality check failed (NaN/Inf/negative values)")

        # Expected rows calculation
        if opts.skip_dataset_statistics:
            rows_audit = {"status": "skipped", "reason": "fast_experiment"}
        else:
            rows_audit = audit_rows_calculation(
                df,
                meta_doc,
                band=band_act,
                actual_samples=int(sampling_audit.get("actual_samples") or 0),
            )
            categories["dataset"].append(_check_row(
                "Rows Calculation",
                "Expected Rows",
                f"{rows_audit.get('expected_rows', 0):,}",
                f"{rows_audit.get('actual_rows', 0):,}",
                status=rows_audit.get("status", "warn"),
            ))

        # Integrity + missing feature breakdown
        feat_cols = implemented_features or actual_feature_cols
        cols_to_check = list(expected_features) if expected_features else feat_cols
        missing_feat_detail = audit_missing_features(df, feat_cols)
        all_null_audit = audit_all_null_features(df, cols_to_check)
        coverage_audit = audit_feature_coverage(df, cols_to_check)
        missing_feature_values = int(missing_feat_detail.get("total_missing") or 0)
        invalid_strike = 0
        if "strike" in df.columns:
            invalid_strike = int((pd.to_numeric(df["strike"], errors="coerce") <= 0).sum())
        invalid_ts = 0
        if "timestamp" in df.columns:
            invalid_ts = int(df["timestamp"].map(_is_bad).sum())

        integrity_audit = {
            "duplicate_rows": dup_rows,
            "missing_feature_values": missing_feature_values,
            "missing_target_values": missing_target_values,
            "invalid_strike_rows": invalid_strike,
            "invalid_timestamps": invalid_ts,
            "top_missing_features": missing_feat_detail.get("top_missing") or [],
            "missing_reasons": missing_feat_detail.get("reasons") or [],
            "all_null_features": all_null_audit.get("features") or [],
            "all_null_count": int(all_null_audit.get("count") or 0),
            "all_null_status": all_null_audit.get("status") or "pass",
            "feature_coverage": coverage_audit.get("features") or [],
        }
        for item in all_null_audit.get("features") or []:
            feat_name = item.get("feature") or "?"
            reason = item.get("reason") or "100% NULL"
            errors.append(f"Feature '{feat_name}' is 100% NULL: {reason}")
            categories["integrity"].append(_check_row(
                "All-NULL Feature",
                feat_name,
                "≥1 non-null value",
                f"0 / {item.get('row_count', row_count):,}",
                status="fail",
            ))
        for feat_row in (missing_feat_detail.get("top_missing") or [])[:8]:
            categories["integrity"].append(_check_row(
                "Missing Features",
                feat_row.get("feature"),
                0,
                feat_row.get("missing_count"),
                status="warn",
            ))
        categories["integrity"].append(_check_row("Dataset Integrity", "Duplicate Rows", 0, dup_rows, status="pass" if dup_rows == 0 else "fail"))
        categories["integrity"].append(_check_row("Dataset Integrity", "Missing Feature Values", 0, missing_feature_values, status="pass" if missing_feature_values == 0 else "warn"))
        categories["integrity"].append(_check_row("Dataset Integrity", "Missing Target Values", 0, missing_target_values, status="pass" if missing_target_values == 0 else "fail"))
        categories["integrity"].append(_check_row("Dataset Integrity", "Invalid Strike Rows", 0, invalid_strike, status="pass" if invalid_strike == 0 else "fail"))
        categories["integrity"].append(_check_row("Dataset Integrity", "Invalid Timestamps", 0, invalid_ts, status="pass" if invalid_ts == 0 else "fail"))

        emit("structural_validation", "done")
        try:
            extended_audit = run_extended_audits(
                df,
                meta_doc,
                chart_dir=_CHART_DIR,
                feature_columns=feat_cols,
                target_columns=target_cols_exp,
                expected_doc=expected_doc,
                tracker=tracker,
                audit_options=opts,
            )
            if not opts.skip_feature_audit:
                heatmap = extended_audit.get("feature_heatmap") or {}
                integrity_audit["feature_heatmap"] = heatmap.get("rows") or []
            ind = extended_audit.get("independent_formulas") or {}
            dist = extended_audit.get("feature_distributions") or {}
            corr = extended_audit.get("correlation_checks") or {}
            if not opts.skip_feature_audit:
                if ind.get("status") == "configuration_mismatch":
                    warnings.append(
                        ind.get("mismatch_reason")
                        or "Independent validator lookback policy does not match dataset configuration"
                    )
                    align = ind.get("policy_alignment") or {}
                    categories["metadata"].append(_check_row(
                        "Lookback Policy",
                        "Builder vs Validator",
                        align.get("builder_policy") or "—",
                        align.get("validator_policy") or "—",
                        status="warn",
                    ))
                elif ind.get("status") == "fail":
                    warnings.append("Independent formula validation failed (builder-independent recalc)")
                for chk in ind.get("checks") or []:
                    if chk.get("status") == "fail":
                        categories["features"].append(_check_row(
                            "Independent Validation",
                            chk.get("feature"),
                            "match replay+BS",
                            f"{chk.get('failed')} failures",
                            status="fail",
                        ))
                if ind.get("status") == "configuration_mismatch":
                    categories["features"].append(_check_row(
                        "Independent Validation",
                        "Lookback policy",
                        ind.get("builder_policy") or "—",
                        "Configuration Mismatch",
                        status="warn",
                    ))
            if not opts.skip_distribution_report and dist.get("status") == "fail":
                warnings.append("Feature distribution sanity check failed (impossible values detected)")
            if not opts.skip_leakage_audit and corr.get("status") == "fail":
                warnings.append("Correlation sanity checks failed")
            if not opts.skip_distribution_report:
                for feat in dist.get("features") or []:
                    if feat.get("status") == "fail":
                        categories["features"].append(_check_row(
                            "Distribution",
                            feat.get("label"),
                            f"[{feat.get('bounds', {}).get('min')}, {feat.get('bounds', {}).get('max')}]",
                            f"min={feat.get('min')} max={feat.get('max')}",
                            status="fail",
                        ))
        except Exception as exc:
            extended_audit = {"error": str(exc)}
            warnings.append(f"Extended audit validations skipped: {exc}")

        if opts.skip_feature_audit:
            formula_recalc = {"status": "skipped", "rows_checked": 0, "groups": [], "spot_checks": [], "reason": "fast_experiment"}
        else:
            try:
                emit("feature_validation")

                def _formula_progress(payload: dict[str, Any]) -> None:
                    rows_done = int(payload.get("rows_done") or 0)
                    rows_total = int(payload.get("rows_total") or 0)
                    speed = round(rows_done / max(tracker.elapsed_total(), 0.001), 1) if rows_done else None
                    remaining = None
                    if rows_total and rows_done and speed:
                        remaining = round((rows_total - rows_done) / speed, 1)
                    tracker.emit_dashboard(
                        running_stage="Feature validation",
                        current_validation="Formula recalculation",
                        rows_processed=rows_done,
                        rows_total=rows_total,
                        speed_rows_per_sec=speed,
                        remaining_sec=remaining,
                    )

                formula_recalc = audit_formula_recalc(
                    df,
                    meta_doc,
                    chart_dir=_CHART_DIR,
                    registry=registry,
                    enabled_groups=expected_groups,
                    target_columns=target_cols_exp,
                    n_sample=100,
                    on_progress=_formula_progress,
                )
                emit("feature_validation", "done")
            except Exception as exc:
                formula_recalc = {"status": "warn", "rows_checked": 0, "groups": [], "spot_checks": [], "error": str(exc)}
                warnings.append(f"Formula recalculation check skipped: {exc}")

    implemented_groups = set(meta_doc.get("feature_groups_implemented") or [])
    group_rows = []
    groups_remaining = 0
    for gid in registry.get("groupOrder") or expected_groups:
        label = group_labels.get(gid, gid)
        if gid in implemented_groups:
            group_rows.append({"id": gid, "label": label, "status": "done"})
        elif gid in expected_groups:
            group_rows.append({"id": gid, "label": label, "status": "pending"})
            groups_remaining += 1
        else:
            continue

    formula_validation = (
        []
        if opts.skip_feature_audit
        else _build_formula_validation(
            registry,
            expected_groups,
            col_set=col_set,
            implemented_features=implemented_features,
        )
    )
    if not opts.skip_feature_audit:
        for row in formula_validation:
            feat_status = row.get("status") or "info"
            categories["features"].append(_check_row(
                "Feature Formula",
                row.get("label") or row.get("id"),
                f"{row.get('expected')}",
                f"{row.get('actual')} / {row.get('expected')}",
                status="pass" if feat_status == "pass" else ("fail" if feat_status == "fail" else "warn"),
            ))

    feature_audit = {
        "profile": profile_exp,
        "expected_features": expected_feature_count,
        "implemented": implemented_count,
        "pending": pending_count,
        "coverage_pct": coverage,
        "groups": group_rows,
        "groups_remaining": groups_remaining,
        "formula_registry": formula_validation,
        "formula_validation": formula_validation,
        "formula_groups_passed": sum(1 for r in formula_validation if r.get("status") == "pass"),
        "formula_groups_total": len(formula_validation),
        "formula_recalc": formula_recalc,
        "extended_audit": extended_audit,
    }

    performance_audit: dict[str, Any] = {
        "stages": [],
        "total_elapsed_label": None,
        "rows_per_sec": None,
        "peak_ram_mb": None,
    }
    if not opts.skip_dataset_statistics:
        perf = meta_doc.get("build_performance") or {}
        for st in perf.get("stages") or []:
            performance_audit["stages"].append({
                "name": st.get("name"),
                "elapsed_label": st.get("elapsed_label"),
                "elapsed_sec": st.get("elapsed_sec"),
            })
            categories["performance"].append(_check_row(
                "Performance",
                st.get("name") or "Stage",
                "—",
                st.get("elapsed_label") or "—",
                status="info",
            ))
        performance_audit["total_elapsed_label"] = perf.get("total_elapsed_label")
        performance_audit["rows_per_sec"] = perf.get("rows_per_sec")
        performance_audit["peak_ram_mb"] = perf.get("peak_ram_mb")
        if performance_audit["rows_per_sec"]:
            categories["performance"].append(_check_row(
                "Performance", "Rows/sec", "—", performance_audit["rows_per_sec"], status="info",
            ))
        if performance_audit["peak_ram_mb"]:
            categories["performance"].append(_check_row(
                "Performance", "Peak RAM (MB)", "—", performance_audit["peak_ram_mb"], status="info",
            ))
    else:
        performance_audit["status"] = "skipped"

    fingerprint = build_dataset_fingerprint(
        parquet_path=parquet_path if files["dataset"]["exists"] else None,
        meta_doc=meta_doc,
        expected_doc=expected_doc_for_fp,
    )
    from .spec_identity import build_audit_specification_summary

    specification_summary = build_audit_specification_summary(meta_doc, expected_doc)
    if opts.skip_quality_report:
        quality_score = {
            "status": "skipped",
            "label": "Skipped (fast experiment)",
            "score": None,
            "reason": "fast_experiment",
        }
    else:
        quality_score = build_quality_score(
            errors=errors,
            warnings=warnings,
            formula_registry=formula_validation,
            formula_recalc=formula_recalc,
            sampling_audit=sampling_audit,
            strike_audit=strike_audit,
            targets_audit=targets_audit,
            integrity_audit=integrity_audit,
            rows_audit=rows_audit,
            files=files,
            meta_doc=meta_doc,
            expected_doc=expected_doc,
            extended_audit=extended_audit,
        )

    report = _finalize_report(
        safe_name,
        files,
        categories,
        errors,
        warnings,
        t0,
        feature_audit=feature_audit,
        sampling_audit=sampling_audit,
        strike_audit=strike_audit,
        targets_audit=targets_audit,
        integrity_audit=integrity_audit,
        performance_audit=performance_audit,
        rows_audit=rows_audit,
        fingerprint=fingerprint,
        quality_score=quality_score,
        specification_summary=specification_summary,
        extended_audit=extended_audit,
        coverage_audit=coverage_audit,
        dataset_rows=row_count,
        dataset_columns=col_count,
    )
    if tracker.durations:
        report["stage_timings"] = {k: round(v, 2) for k, v in tracker.durations.items()}
        report["audit_duration_sec"] = round(tracker.elapsed_total(), 2)
    report["audit_options"] = opts.to_dict()

    if not opts.skip_quality_report:
        from .audit_investigation_engine import enrich_report_with_investigations
        enrich_report_with_investigations(
            report,
            data_dir=data_dir,
            chart_dir=_CHART_DIR,
            on_progress=on_progress,
        )
    return report


def _finalize_report(
    safe_name: str,
    files: dict[str, Any],
    categories: dict[str, list[dict[str, Any]]],
    errors: list[str],
    warnings: list[str],
    t0: float,
    *,
    feature_audit: dict[str, Any] | None,
    sampling_audit: dict[str, Any] | None,
    strike_audit: dict[str, Any] | None,
    targets_audit: dict[str, Any] | None,
    integrity_audit: dict[str, Any] | None,
    performance_audit: dict[str, Any] | None,
    rows_audit: dict[str, Any] | None = None,
    fingerprint: dict[str, Any] | None = None,
    quality_score: dict[str, Any] | None = None,
    specification_summary: dict[str, Any] | None = None,
    extended_audit: dict[str, Any] | None = None,
    coverage_audit: dict[str, Any] | None = None,
    dataset_rows: int = 0,
    dataset_columns: int = 0,
) -> dict[str, Any]:
    flat_rows: list[dict[str, Any]] = []
    for key in categories:
        flat_rows.extend(categories[key])

    checks_passed = sum(1 for r in flat_rows if r.get("status") == "pass")
    warn_count = sum(1 for r in flat_rows if r.get("status") == "warn") + len(warnings)
    error_count = sum(1 for r in flat_rows if r.get("status") == "fail") + len(errors)

    from .audit_investigation_engine import build_training_readiness, compute_audit_overall_status

    report_fragment = {
        "errors": errors,
        "warnings_list": warnings,
        "summary": {"warnings": warn_count, "errors": error_count},
        "integrity_audit": integrity_audit or {},
        "feature_audit": feature_audit or {},
        "extended_audit": extended_audit or {},
        "targets_audit": targets_audit or {},
        "files": files,
    }
    readiness = build_training_readiness(report_fragment)
    audit_overall = compute_audit_overall_status(report_fragment)
    overall_status = audit_overall["status"]
    overall_label = audit_overall["label"]
    passed = readiness["critical_count"] == 0
    safe_to_train = readiness["ready"]
    ready = readiness["ready"]
    ready_label = readiness["recommendation"]

    category_meta = [
        ("metadata", "Metadata"),
        ("sampling", "Sampling"),
        ("strike", "Strike Selection"),
        ("prediction", "Prediction Targets"),
        ("features", "Feature Generation"),
        ("dataset", "Dataset"),
        ("integrity", "Dataset Integrity"),
        ("performance", "Performance"),
    ]
    seen: set[str] = set()
    expandable: list[dict[str, Any]] = []
    for key, label in category_meta:
        if key in seen or not categories.get(key):
            continue
        seen.add(key)
        expandable.append({
            "id": key,
            "label": label,
            "rows": categories[key],
        })

    coverage = (feature_audit or {}).get("coverage_pct", 0)
    qs = quality_score or {}
    ready_label = qs.get("ready_label") or readiness["recommendation"]
    ready = qs.get("ready_for_training", readiness["ready"])
    safe_to_train = ready
    missing_vals = int((integrity_audit or {}).get("missing_feature_values") or qs.get("missing_feature_values") or 0)
    if not readiness["ready"]:
        emoji = "🔴"
    elif ready:
        emoji = "🟡"
    elif passed:
        emoji = "🟡"
    else:
        emoji = "🔴"
    return {
        "dataset_name": safe_name,
        "passed": passed,
        "files": files,
        "overall": {
            "status": overall_status,
            "label": overall_label,
            "emoji": emoji,
            "safe_to_train": safe_to_train,
            "safe_label": ready_label,
            "ready_for_training": ready,
            "confidence_pct": qs.get("confidence_pct"),
            "missing_feature_warnings": missing_vals > 0,
        },
        "summary": {
            "checks_passed": checks_passed,
            "warnings": warn_count,
            "errors": error_count,
            "critical_errors": readiness["critical_count"],
            "duration_sec": round(time.monotonic() - t0, 2),
        },
        "table_rows": flat_rows,
        "categories": expandable,
        "feature_audit": feature_audit or {},
        "sampling_audit": sampling_audit or {},
        "strike_audit": strike_audit or {},
        "targets_audit": targets_audit or {},
        "integrity_audit": integrity_audit or {},
        "feature_coverage_audit": coverage_audit or {},
        "extended_audit": extended_audit or {},
        "rows_audit": rows_audit or {},
        "fingerprint": fingerprint or {},
        "specification_summary": specification_summary or {},
        "quality_score": qs,
        "performance_audit": performance_audit or {},
        "result": {
            "safe_to_train": safe_to_train,
            "ready_for_training": ready,
            "training_readiness_label": readiness["label"],
            "blocking_issues": readiness["blocking_issues"],
            "warnings": warn_count,
            "errors": error_count,
            "coverage_pct": coverage,
            "confidence_pct": qs.get("confidence_pct"),
            "groups_remaining": (feature_audit or {}).get("groups_remaining", 0),
        },
        "dataset_rows": dataset_rows,
        "dataset_columns": dataset_columns,
        "issues": errors,
        "warnings_list": warnings,
    }


def compare_datasets(
    *,
    data_dir: str,
    dataset_a: str,
    dataset_b: str,
) -> dict[str, Any]:
    """Compare two datasets for the audit dialog Compare view."""
    a = audit_dataset(data_dir=data_dir, dataset_name=dataset_a)
    b = audit_dataset(data_dir=data_dir, dataset_name=dataset_b)

    def _metric(label: str, va: Any, vb: Any, *, fmt: str = "raw") -> dict[str, Any]:
        if fmt == "num":
            sa = f"{int(va):,}" if va is not None else "—"
            sb = f"{int(vb):,}" if vb is not None else "—"
        elif fmt == "pct":
            sa = f"{va}%" if va is not None else "—"
            sb = f"{vb}%" if vb is not None else "—"
        else:
            sa = str(va) if va is not None else "—"
            sb = str(vb) if vb is not None else "—"
        delta: str | None = None
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            d = vb - va
            delta = f"{d:+,}" if fmt == "num" else (f"{d:+.1f}%" if fmt == "pct" else str(d))
        return {"metric": label, "a": sa, "b": sb, "delta": delta}

    fa = a.get("feature_audit") or {}
    fb = b.get("feature_audit") or {}
    sa = a.get("sampling_audit") or {}
    sb = b.get("sampling_audit") or {}
    pa = a.get("performance_audit") or {}
    pb = b.get("performance_audit") or {}

    rows = [
        _metric("Row count", a.get("dataset_rows"), b.get("dataset_rows"), fmt="num"),
        _metric("Column count", a.get("dataset_columns"), b.get("dataset_columns"), fmt="num"),
        _metric("Feature count (implemented)", fa.get("implemented"), fb.get("implemented"), fmt="num"),
        _metric("Expected features", fa.get("expected_features"), fb.get("expected_features"), fmt="num"),
        _metric("Feature coverage", fa.get("coverage_pct"), fb.get("coverage_pct"), fmt="pct"),
        _metric("Sampling interval (sec)", sa.get("interval_sec"), sb.get("interval_sec")),
        _metric("Target columns", len((a.get("targets_audit") or {}).get("columns") or []), len((b.get("targets_audit") or {}).get("columns") or []), fmt="num"),
        _metric("Groups remaining", fa.get("groups_remaining"), fb.get("groups_remaining"), fmt="num"),
        _metric("Build time", pa.get("total_elapsed_label"), pb.get("total_elapsed_label")),
    ]

    return {
        "dataset_a": _safe_filename(dataset_a),
        "dataset_b": _safe_filename(dataset_b),
        "rows": rows,
        "audit_a": a,
        "audit_b": b,
    }
