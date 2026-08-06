"""Write Feature Importance Studio JSON artifacts (compute↔UI contract)."""

from __future__ import annotations

import json
import os
from typing import Any


ARTIFACT_DIRNAME = "feature_importance_studio"

ARTIFACT_FILES = (
    "native_xgb.json",
    "permutation.json",
    "shap.json",
    "comparison.json",
    "run_meta.json",
)


def studio_artifacts_dir(package_dir: str) -> str:
    path = os.path.join(package_dir, ARTIFACT_DIRNAME)
    os.makedirs(path, exist_ok=True)
    return path


def write_studio_artifacts(
    package_dir: str,
    *,
    native: list[dict[str, Any]],
    permutation: list[dict[str, Any]],
    shap: list[dict[str, Any]],
    comparison: list[dict[str, Any]],
    run_meta: dict[str, Any],
) -> str:
    """Write JSON artifacts under ``package_dir/feature_importance_studio/``."""
    out = studio_artifacts_dir(package_dir)
    payloads = {
        "native_xgb.json": {"rows": native},
        "permutation.json": {"rows": permutation},
        "shap.json": {"rows": shap},
        "comparison.json": {"rows": comparison},
        "run_meta.json": run_meta,
    }
    for name, doc in payloads.items():
        path = os.path.join(out, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, default=str)
    return out


def load_studio_artifacts(package_dir: str) -> dict[str, Any] | None:
    """Load existing Studio artifacts if present."""
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
    # Backfill rank Δ for older artifacts.
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("rank_delta_gain_shap") is None:
            rg, rs = row.get("rank_gain"), row.get("rank_shap")
            if rg is not None and rs is not None:
                try:
                    row["rank_delta_gain_shap"] = abs(int(rg) - int(rs))
                except (TypeError, ValueError):
                    pass
    return {"artifacts_dir": out, "comparison": rows, "meta": meta}
