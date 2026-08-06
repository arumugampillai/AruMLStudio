"""Golden dataset regression — manifest-based reference checks."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Callable

import pandas as pd

from .audit_fingerprint import _file_sha256
from .auditor import audit_dataset
from .dataset_validator import (
    load_audit_cache,
    load_validation_cache,
    run_dataset_validation,
    save_audit_cache,
    save_validation_cache,
)
from .expected_spec import expected_spec_path
from .pipeline_identity import pipeline_stage_hashes_from_fingerprint
from .spec_identity import dataset_spec_hash
from .writer import _safe_filename, datasets_dir

_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REGRESSION_DIR = os.path.join(_PACKAGE_DIR, "regression")
_CHART_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAST_VALIDATION_SAMPLE = 50
FULL_VALIDATION_SAMPLE = 200
SAMPLE_ROW_COUNT = 100
SAMPLE_RANDOM_STATE = 42

_STAGE_LABELS: dict[str, str] = {
    "sampling": "Sampling",
    "feature": "Features",
    "target": "Targets",
    "validation": "Validation",
}


def _current_pipeline_stage_hashes(
    meta_doc: dict[str, Any],
    expected_doc: dict[str, Any],
) -> dict[str, str]:
    fp = meta_doc.get("pipeline_fingerprint") or expected_doc.get("pipeline_fingerprint") or {}
    return pipeline_stage_hashes_from_fingerprint(fp, meta_doc=meta_doc, expected_doc=expected_doc)


def _stage_regression_checks(
    checks: list[dict[str, Any]],
    *,
    manifest: dict[str, Any],
    current: dict[str, str],
    manifest_configured: bool,
) -> list[dict[str, Any]]:
    """Append per-stage hash checks and return UI-friendly stage rows."""
    expected = manifest.get("expected_stage_hashes") or {}
    rows: list[dict[str, Any]] = []
    for key, label in _STAGE_LABELS.items():
        actual = current.get(key)
        exp = expected.get(key)
        if manifest_configured and exp:
            ok = _check(checks, name=f"stage_{key}", expected=exp, actual=actual)
            rows.append({
                "stage": key,
                "label": label,
                "status": "pass" if ok else "fail",
                "expected": exp,
                "actual": actual,
            })
        else:
            rows.append({
                "stage": key,
                "label": label,
                "status": "info",
                "expected": exp or "—",
                "actual": actual,
            })
    return rows


def _manifest_update_gate(
    data_dir: str,
    dataset_name: str,
    *,
    audit_report: dict[str, Any] | None = None,
    validation_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Whether golden manifest may be updated (audit PASS + validation PASS)."""
    audit_label = None
    validation_label = None
    if audit_report is not None:
        audit_label = _normalize_audit_label(audit_report)
    else:
        cache = load_audit_cache(data_dir, dataset_name)
        if cache:
            st = str(cache.get("status") or "").lower()
            audit_label = "PASS" if st == "pass" else ("WARN" if st == "warn" else "FAIL" if st == "fail" else None)

    if validation_report is not None:
        validation_label = _normalize_validation_label(validation_report)
    else:
        cache = load_validation_cache(data_dir, dataset_name)
        if cache:
            st = str(cache.get("status") or "").lower()
            validation_label = "PASS" if st == "pass" else ("WARN" if st == "warn" else "FAIL" if st == "fail" else None)

    reasons: list[str] = []
    if audit_label != "PASS":
        reasons.append(f"Audit must be PASS (current: {audit_label or 'not run'})")
    if validation_label != "PASS":
        reasons.append(f"Validation must be PASS (current: {validation_label or 'not run'})")

    return {
        "allowed": not reasons,
        "audit_label": audit_label,
        "validation_label": validation_label,
        "reasons": reasons,
        "block_reason": "; ".join(reasons) if reasons else None,
    }


def _load_json(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _save_json(path: str, doc: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def _short_hash(value: str | None, *, length: int = 8) -> str | None:
    if not value:
        return None
    return str(value).upper()[:length]


def _file_short_hash(path: str | None) -> str | None:
    full = _file_sha256(path or "")
    return _short_hash(full)


def manifest_path(data_dir: str) -> str:
    """User manifest in data dir overrides packaged template."""
    custom = os.path.join(datasets_dir(data_dir), "golden.manifest.json")
    if os.path.isfile(custom):
        return custom
    return os.path.join(_REGRESSION_DIR, "golden_manifest.json")


def golden_config_path() -> str:
    return os.path.join(_REGRESSION_DIR, "golden_config.json")


def last_run_path(data_dir: str) -> str:
    return os.path.join(datasets_dir(data_dir), "golden.regression-last.json")


def load_manifest(data_dir: str) -> dict[str, Any]:
    path = manifest_path(data_dir)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Golden manifest not found: {path}")
    return _load_json(path)


def load_last_run(data_dir: str) -> dict[str, Any] | None:
    path = last_run_path(data_dir)
    if not os.path.isfile(path):
        return None
    try:
        return _load_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def _round_val(v: Any) -> Any:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int, bool)):
        return v
    if isinstance(v, float):
        return round(v, 6)
    return str(v)


def compute_feature_hash(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    n: int = SAMPLE_ROW_COUNT,
    random_state: int = SAMPLE_RANDOM_STATE,
) -> str | None:
    cols = [c for c in feature_columns if c in df.columns]
    if not cols or df.empty:
        return None
    sample = df[cols].sample(n=min(n, len(df)), random_state=random_state).sort_index()
    payload = {
        "columns": cols,
        "rows": [
            {c: _round_val(sample.at[idx, c]) for c in cols}
            for idx in sample.index
        ],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8].upper()


def compute_sample_hash(
    df: pd.DataFrame,
    *,
    n: int = SAMPLE_ROW_COUNT,
    random_state: int = SAMPLE_RANDOM_STATE,
) -> str | None:
    if df.empty:
        return None
    sample = df.sample(n=min(n, len(df)), random_state=random_state).sort_index()
    records = [
        {str(k): _round_val(row[k]) for k in sorted(sample.columns)}
        for _, row in sample.iterrows()
    ]
    blob = json.dumps(records, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8].upper()


def compute_dataset_hashes(
    parquet_path: str,
    meta_doc: dict[str, Any],
    *,
    read_parquet: bool = True,
) -> dict[str, Any]:
    meta_path = parquet_path.replace(".parquet", ".json")
    out: dict[str, Any] = {
        "dataset_hash": _file_short_hash(parquet_path),
        "metadata_hash": _file_short_hash(meta_path),
        "feature_hash": None,
        "sample_hash": None,
    }
    if not read_parquet or not os.path.isfile(parquet_path):
        return out
    df = pd.read_parquet(parquet_path)
    feature_cols = list(meta_doc.get("feature_columns") or [])
    out["feature_hash"] = compute_feature_hash(df, feature_cols)
    out["sample_hash"] = compute_sample_hash(df)
    return out


def _check(
    checks: list[dict[str, Any]],
    *,
    name: str,
    expected: Any,
    actual: Any,
    ok: bool | None = None,
) -> bool:
    if ok is None:
        ok = expected == actual
    checks.append({
        "check": name,
        "expected": expected,
        "actual": actual,
        "status": "pass" if ok else "fail",
    })
    return bool(ok)


def _normalize_audit_label(report: dict[str, Any]) -> str:
    overall = report.get("overall") or {}
    st = str(overall.get("status") or "").lower()
    if st == "pass" and not overall.get("missing_feature_warnings"):
        return "PASS"
    if st == "pass":
        return "PASS"
    if st == "warn":
        return "WARN"
    return "FAIL"


def _normalize_validation_label(report: dict[str, Any]) -> str:
    st = str(report.get("status") or "").lower()
    if st == "pass":
        return "PASS"
    if st == "warn":
        return "WARN"
    return "FAIL"


def build_manifest_from_dataset(data_dir: str, dataset_name: str | None = None) -> dict[str, Any]:
    """Capture current dataset state as the golden manifest reference."""
    manifest = load_manifest(data_dir)
    name = _safe_filename(dataset_name or manifest.get("dataset_name") or "golden")
    out_dir = datasets_dir(data_dir)
    parquet_path = os.path.join(out_dir, f"{name}.parquet")
    meta_path = os.path.join(out_dir, f"{name}.json")
    expected_path = expected_spec_path(data_dir, name)
    if not os.path.isfile(parquet_path):
        raise FileNotFoundError(f"Golden dataset parquet not found: {parquet_path}")

    meta_doc = _load_json(meta_path) if os.path.isfile(meta_path) else {}
    expected_doc = _load_json(expected_path) if os.path.isfile(expected_path) else {}

    audit_report = audit_dataset(data_dir=data_dir, dataset_name=name)
    validation_report = run_dataset_validation(
        data_dir=data_dir,
        dataset_name=name,
        n_sample=FULL_VALIDATION_SAMPLE,
        save_cache=False,
    )
    gate = _manifest_update_gate(
        data_dir, name,
        audit_report=audit_report,
        validation_report=validation_report,
    )
    if not gate["allowed"]:
        raise ValueError(
            "Cannot update golden manifest: " + (gate["block_reason"] or "audit and validation must pass")
        )

    audit_label = _normalize_audit_label(audit_report)
    validation_label = _normalize_validation_label(validation_report)
    hashes = compute_dataset_hashes(parquet_path, meta_doc, read_parquet=True)
    fp = meta_doc.get("pipeline_fingerprint") or expected_doc.get("pipeline_fingerprint") or {}
    stage_hashes = _current_pipeline_stage_hashes(meta_doc, expected_doc)

    updated = {
        **manifest,
        "version": int(manifest.get("version") or 1),
        "dataset_name": name,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "expected_spec_hash": dataset_spec_hash(meta_doc, expected_doc),
        "expected_pipeline_fingerprint": fp,
        "expected_stage_hashes": stage_hashes,
        "expected_rows": int(meta_doc.get("row_count") or len(pd.read_parquet(parquet_path))),
        "expected_columns": int(meta_doc.get("column_count") or 0),
        "expected_audit": audit_label,
        "expected_validation": validation_label,
        "expected_dataset_hash": hashes.get("dataset_hash"),
        "expected_metadata_hash": hashes.get("metadata_hash"),
        "expected_feature_hash": hashes.get("feature_hash"),
        "expected_sample_hash": hashes.get("sample_hash"),
    }
    custom_path = os.path.join(datasets_dir(data_dir), "golden.manifest.json")
    _save_json(custom_path, updated)
    return updated


def _maybe_rebuild_golden(data_dir: str, on_progress: Callable[[str], None] | None = None) -> bool:
    cfg_path = golden_config_path()
    if not os.path.isfile(cfg_path):
        return False
    cfg = _load_json(cfg_path)
    if not cfg.get("rebuild_on_full"):
        return False
    build = cfg.get("build") or {}
    if not build.get("sources"):
        if on_progress:
            on_progress("Rebuild skipped — no sources in golden_config.json")
        return False

    from .orchestrator import DatasetBuildConfig, DatasetBuildOrchestrator

    if on_progress:
        on_progress("Rebuilding golden dataset…")
    config = DatasetBuildConfig(
        dataset_name=str(build.get("dataset_name") or "golden"),
        sources=list(build.get("sources") or []),
        sampling=dict(build.get("sampling") or {}),
        strike_selection=dict(build.get("strike_selection") or {}),
        prediction_targets=dict(build.get("prediction_targets") or {}),
        feature_selection=dict(build.get("feature_selection") or {}),
        data_dir=data_dir,
    )
    orchestrator = DatasetBuildOrchestrator(config=config)
    orchestrator.run()
    return True


def run_golden_regression(
    data_dir: str,
    *,
    mode: str = "fast",
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run golden regression. mode: ``fast`` (spec + audit + validation) or ``full`` (+ hashes, optional rebuild)."""
    mode = str(mode or "fast").strip().lower()
    if mode not in ("fast", "full"):
        mode = "fast"

    manifest = load_manifest(data_dir)
    dataset_name = _safe_filename(manifest.get("dataset_name") or "golden")
    out_dir = datasets_dir(data_dir)
    parquet_path = os.path.join(out_dir, f"{dataset_name}.parquet")
    meta_path = os.path.join(out_dir, f"{dataset_name}.json")
    expected_path = expected_spec_path(data_dir, dataset_name)

    checks: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    errors: list[str] = []

    def step(label: str, detail: str = "") -> None:
        steps.append({"label": label, "detail": detail, "at": datetime.now(timezone.utc).isoformat()})
        if on_progress:
            on_progress(label if not detail else f"{label} — {detail}")

    if mode == "full":
        step("Full regression", "optional rebuild")
        try:
            _maybe_rebuild_golden(data_dir, on_progress=on_progress)
        except Exception as exc:
            errors.append(f"Rebuild failed: {exc}")
            step("Rebuild", f"failed: {exc}")

    if not os.path.isfile(parquet_path):
        msg = f"Golden dataset not found: {dataset_name}.parquet"
        result = {
            "status": "fail",
            "label": "FAIL",
            "mode": mode,
            "dataset_name": dataset_name,
            "passed": False,
            "checks": [{"check": "dataset_exists", "status": "fail", "expected": True, "actual": False}],
            "steps": steps,
            "errors": [msg],
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_json(last_run_path(data_dir), result)
        return result

    meta_doc = _load_json(meta_path) if os.path.isfile(meta_path) else {}
    expected_doc = _load_json(expected_path) if os.path.isfile(expected_path) else {}
    current_stages = _current_pipeline_stage_hashes(meta_doc, expected_doc)

    # --- Spec ---
    step("Spec", "compare manifest identity")
    spec_hash = dataset_spec_hash(meta_doc, expected_doc)
    fp = meta_doc.get("pipeline_fingerprint") or expected_doc.get("pipeline_fingerprint") or {}
    exp_fp = manifest.get("expected_pipeline_fingerprint") or {}
    manifest_configured = bool(manifest.get("expected_spec_hash"))

    if manifest_configured:
        _check(checks, name="spec_hash", expected=manifest.get("expected_spec_hash"), actual=spec_hash)
        fp_ok = exp_fp == fp if exp_fp else True
        _check(checks, name="pipeline_fingerprint", expected="match", actual="match" if fp_ok else "drift", ok=fp_ok)
        if int(manifest.get("expected_rows") or 0) > 0:
            _check(
                checks,
                name="row_count",
                expected=manifest.get("expected_rows"),
                actual=int(meta_doc.get("row_count") or 0),
            )
        if int(manifest.get("expected_columns") or 0) > 0:
            _check(
                checks,
                name="column_count",
                expected=manifest.get("expected_columns"),
                actual=int(meta_doc.get("column_count") or 0),
            )
    else:
        checks.append({
            "check": "spec_hash",
            "status": "warn",
            "expected": "manifest not configured",
            "actual": spec_hash,
            "note": "Run update-manifest after creating the golden dataset.",
        })

    # --- Pipeline stage hashes ---
    step("Pipeline stages", "sampling · features · targets · validation")
    stage_results = _stage_regression_checks(
        checks,
        manifest=manifest,
        current=current_stages,
        manifest_configured=manifest_configured,
    )

    # --- Audit ---
    step("Audit")
    audit_report = audit_dataset(data_dir=data_dir, dataset_name=dataset_name)
    try:
        save_audit_cache(data_dir, dataset_name, audit_report)
    except OSError:
        pass
    audit_label = _normalize_audit_label(audit_report)
    if manifest_configured:
        _check(checks, name="audit", expected=manifest.get("expected_audit"), actual=audit_label)
    else:
        checks.append({"check": "audit", "status": "info", "expected": "—", "actual": audit_label})

    # --- Validation ---
    step("Validation")
    n_sample = FULL_VALIDATION_SAMPLE if mode == "full" else FAST_VALIDATION_SAMPLE
    validation_report = run_dataset_validation(
        data_dir=data_dir,
        dataset_name=dataset_name,
        n_sample=n_sample,
        save_cache=True,
    )
    validation_label = _normalize_validation_label(validation_report)
    if manifest_configured:
        _check(checks, name="validation", expected=manifest.get("expected_validation"), actual=validation_label)
    else:
        checks.append({"check": "validation", "status": "info", "expected": "—", "actual": validation_label})

    # --- Hashes (full mode only) ---
    if mode == "full" and manifest_configured:
        step("Hashes", "dataset · metadata · features · sample")
        hashes = compute_dataset_hashes(parquet_path, meta_doc, read_parquet=True)
        for key, manifest_key in (
            ("dataset_hash", "expected_dataset_hash"),
            ("metadata_hash", "expected_metadata_hash"),
            ("feature_hash", "expected_feature_hash"),
            ("sample_hash", "expected_sample_hash"),
        ):
            exp = manifest.get(manifest_key)
            if exp:
                _check(checks, name=key, expected=exp, actual=hashes.get(key))

    failed = [c for c in checks if c.get("status") == "fail"]
    warned = [c for c in checks if c.get("status") == "warn"]
    passed = not failed

    result = {
        "status": "pass" if passed and not failed else "fail",
        "label": "PASS" if passed and not failed else "FAIL",
        "mode": mode,
        "dataset_name": dataset_name,
        "passed": passed and not failed,
        "checks": checks,
        "stage_results": stage_results,
        "current_stage_hashes": current_stages,
        "steps": steps,
        "errors": errors,
        "audit_label": audit_label,
        "validation_label": validation_label,
        "spec_hash": spec_hash,
        "manifest_path": manifest_path(data_dir),
        "manifest_configured": manifest_configured,
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "duration_note": "~30 sec (fast)" if mode == "fast" else "~5 min (full)",
    }
    _save_json(last_run_path(data_dir), result)
    return result


def golden_regression_status(data_dir: str) -> dict[str, Any]:
    """Status for Dataset Registry golden panel."""
    try:
        manifest = load_manifest(data_dir)
    except FileNotFoundError:
        manifest = {}
    last = load_last_run(data_dir) or {}
    name = _safe_filename(manifest.get("dataset_name") or "golden")
    parquet_path = os.path.join(datasets_dir(data_dir), f"{name}.parquet")
    manifest_configured = bool(manifest.get("expected_spec_hash"))

    checked_at = last.get("checked_at")
    checked_label = None
    if checked_at:
        try:
            dt = datetime.fromisoformat(str(checked_at).replace("Z", "+00:00"))
            checked_label = dt.astimezone().strftime("%Y-%m-%d %H:%M")
        except (TypeError, ValueError):
            checked_label = str(checked_at)

    status = last.get("status") or "pending"
    label = last.get("label") or ("PENDING" if not manifest_configured else "—")
    manifest_gate = _manifest_update_gate(data_dir, name)

    return {
        "dataset_name": name,
        "dataset_exists": os.path.isfile(parquet_path),
        "manifest_configured": manifest_configured,
        "manifest_update_allowed": manifest_gate["allowed"],
        "manifest_update_block_reason": manifest_gate["block_reason"],
        "manifest_audit_label": manifest_gate["audit_label"],
        "manifest_validation_label": manifest_gate["validation_label"],
        "manifest_path": manifest_path(data_dir) if manifest else None,
        "status": status,
        "label": label,
        "passed": last.get("passed"),
        "mode": last.get("mode"),
        "checked_at": checked_at,
        "checked_label": checked_label,
        "last_run": last,
        "stage_results": last.get("stage_results") or [],
        "manifest": {
            "expected_spec_hash": manifest.get("expected_spec_hash"),
            "expected_stage_hashes": manifest.get("expected_stage_hashes") or {},
            "expected_rows": manifest.get("expected_rows"),
            "expected_columns": manifest.get("expected_columns"),
            "expected_audit": manifest.get("expected_audit"),
            "expected_validation": manifest.get("expected_validation"),
            "description": manifest.get("description"),
        },
    }
