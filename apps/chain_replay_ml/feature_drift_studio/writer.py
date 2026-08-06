"""Write Feature Drift Studio JSON artifacts (compute↔UI contract)."""

from __future__ import annotations

import json
import os
from typing import Any


ARTIFACT_DIRNAME = "feature_drift_studio"

ARTIFACT_FILES = (
    "drift_rows.json",
    "comparison.json",
    "run_meta.json",
)


def studio_artifacts_dir(package_dir: str) -> str:
    path = os.path.join(package_dir, ARTIFACT_DIRNAME)
    os.makedirs(path, exist_ok=True)
    return path


# Artifact JSON contract version (Phase 5.2+). Older files may omit this field.
SCHEMA_VERSION = 2


def write_studio_artifacts(
    package_dir: str,
    *,
    drift_rows: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    run_meta: dict[str, Any],
) -> str:
    out = studio_artifacts_dir(package_dir)
    meta = dict(run_meta or {})
    meta.setdefault("schema_version", SCHEMA_VERSION)
    payloads = {
        "drift_rows.json": {"schema_version": SCHEMA_VERSION, "rows": drift_rows},
        "comparison.json": {"schema_version": SCHEMA_VERSION, "rows": comparison},
        "run_meta.json": meta,
    }
    for name, doc in payloads.items():
        path = os.path.join(out, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, default=str)
    return out


def load_studio_artifacts(package_dir: str) -> dict[str, Any] | None:
    """Load comparison + meta. Backward-compatible with pre-schema_version artifacts."""
    out = os.path.join(package_dir, ARTIFACT_DIRNAME)
    comparison_path = os.path.join(out, "comparison.json")
    meta_path = os.path.join(out, "run_meta.json")
    if not os.path.isfile(comparison_path):
        return None
    with open(comparison_path, encoding="utf-8") as fh:
        comparison_doc = json.load(fh)
    meta: dict[str, Any] = {}
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            loaded = json.load(fh)
            if isinstance(loaded, dict):
                meta = loaded
    rows = comparison_doc.get("rows") if isinstance(comparison_doc, dict) else None
    if not isinstance(rows, list):
        return None
    # Prefer explicit schema_version; fall back to legacy drift_schema / omit.
    if "schema_version" not in meta and isinstance(comparison_doc, dict):
        if comparison_doc.get("schema_version") is not None:
            meta = {**meta, "schema_version": comparison_doc.get("schema_version")}
    return {"artifacts_dir": out, "comparison": rows, "meta": meta}
