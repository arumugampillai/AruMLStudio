"""Write / load Production Validation Phase B artifacts."""

from __future__ import annotations

import json
import os
from typing import Any


ARTIFACT_DIRNAME = "production_validation"

ARTIFACT_FILES = (
    "comparison.json",
    "summary.json",
    "run_meta.json",
)


def artifacts_dir(package_dir: str) -> str:
    path = os.path.join(package_dir, ARTIFACT_DIRNAME)
    os.makedirs(path, exist_ok=True)
    return path


def write_validation_artifacts(
    package_dir: str,
    *,
    comparison: list[dict[str, Any]],
    summary: dict[str, Any],
    run_meta: dict[str, Any],
) -> str:
    """Write JSON under ``package_dir/production_validation/``."""
    out = artifacts_dir(package_dir)
    payloads = {
        "comparison.json": {"rows": comparison},
        "summary.json": summary,
        "run_meta.json": run_meta,
    }
    for name, doc in payloads.items():
        path = os.path.join(out, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, default=str)
    return out


def load_validation_artifacts(package_dir: str) -> dict[str, Any] | None:
    """Load Phase B artifacts if ``comparison.json`` exists.

    Pre-v1.1 rows (importances + ``collapse_pct``, no ranks) are enriched in
    memory from stored importances — no permutation recompute required.
    """
    out = os.path.join(package_dir, ARTIFACT_DIRNAME)
    comparison_path = os.path.join(out, "comparison.json")
    if not os.path.isfile(comparison_path):
        return None
    with open(comparison_path, encoding="utf-8") as fh:
        comparison_doc = json.load(fh)
    rows = comparison_doc.get("rows") if isinstance(comparison_doc, dict) else None
    if not isinstance(rows, list):
        return None

    summary: dict[str, Any] = {}
    summary_path = os.path.join(out, "summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as fh:
            loaded = json.load(fh)
            if isinstance(loaded, dict):
                summary = loaded

    meta: dict[str, Any] = {}
    meta_path = os.path.join(out, "run_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            loaded = json.load(fh)
            if isinstance(loaded, dict):
                meta = loaded

    unseen_status: dict[str, Any] = {}
    status_path = os.path.join(out, "unseen_dataset.json")
    if os.path.isfile(status_path):
        with open(status_path, encoding="utf-8") as fh:
            loaded = json.load(fh)
            if isinstance(loaded, dict):
                unseen_status = loaded

    from .rules import (
        build_dual_confidence,
        build_feature_validation_summary,
        enrich_comparison_rows_from_importances,
    )

    rows = [r for r in rows if isinstance(r, dict)]
    rows, enriched = enrich_comparison_rows_from_importances(rows)
    if enriched:
        feature_summary = build_feature_validation_summary(rows)
        summary = dict(summary)
        summary["feature_validation"] = feature_summary
        day_count = int(
            summary.get("unseen_day_count")
            or meta.get("unseen_day_count")
            or (unseen_status.get("unseen_day_count") if unseen_status else 0)
            or 0
        )
        dual = build_dual_confidence(
            rows,
            unseen_day_count=day_count,
            feature_summary=feature_summary,
        )
        summary["diagnosis"] = dual.get("diagnosis")
        summary["production_confirmation"] = dual.get("production_confirmation")
        summary["thresholds"] = dual.get("thresholds")
        meta = dict(meta)
        meta["rank_fields_enriched_from_importances"] = True
        meta["studio_version"] = meta.get("studio_version") or "1.1.0-rank"

    return {
        "rows": rows,
        "summary": summary,
        "meta": meta,
        "unseen_status": unseen_status,
        "artifacts_dir": out,
        "rank_fields_enriched": enriched,
    }


def patch_unseen_status_compute_note(
    package_dir: str,
    *,
    compute_note: str,
    extra: dict[str, Any] | None = None,
) -> None:
    """Update Phase A status file after Phase B compute."""
    path = os.path.join(package_dir, ARTIFACT_DIRNAME, "unseen_dataset.json")
    doc: dict[str, Any] = {}
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                doc = loaded
        except (OSError, json.JSONDecodeError):
            doc = {}
    doc["compute_note"] = compute_note
    if extra:
        doc.update(extra)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, default=str)
