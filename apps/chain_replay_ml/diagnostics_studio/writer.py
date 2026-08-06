"""Write Diagnostics Studio JSON artifacts."""

from __future__ import annotations

import json
import os
from typing import Any


ARTIFACT_DIRNAME = "diagnostics_studio"


def studio_artifacts_dir(package_dir: str) -> str:
    path = os.path.join(package_dir, ARTIFACT_DIRNAME)
    os.makedirs(path, exist_ok=True)
    return path


def write_studio_artifacts(
    package_dir: str,
    *,
    summary: dict[str, Any],
    narrative: list[str],
    comparison: list[dict[str, Any]],
    run_meta: dict[str, Any],
) -> str:
    out = studio_artifacts_dir(package_dir)
    payloads = {
        "summary.json": summary,
        "narrative.json": {"bullets": narrative},
        "comparison.json": {"rows": comparison},
        "run_meta.json": run_meta,
    }
    for name, doc in payloads.items():
        with open(os.path.join(out, name), "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, default=str)
    return out


def load_studio_artifacts(package_dir: str) -> dict[str, Any] | None:
    out = os.path.join(package_dir, ARTIFACT_DIRNAME)
    summary_path = os.path.join(out, "summary.json")
    if not os.path.isfile(summary_path):
        return None
    with open(summary_path, encoding="utf-8") as fh:
        summary = json.load(fh)
    if not isinstance(summary, dict):
        return None
    narrative: list[str] = []
    narr_path = os.path.join(out, "narrative.json")
    if os.path.isfile(narr_path):
        with open(narr_path, encoding="utf-8") as fh:
            narr = json.load(fh)
        if isinstance(narr, dict) and isinstance(narr.get("bullets"), list):
            narrative = [str(b) for b in narr["bullets"]]
    comparison: list[dict[str, Any]] = []
    comp_path = os.path.join(out, "comparison.json")
    if os.path.isfile(comp_path):
        with open(comp_path, encoding="utf-8") as fh:
            comp = json.load(fh)
        rows = comp.get("rows") if isinstance(comp, dict) else None
        if isinstance(rows, list):
            comparison = [r for r in rows if isinstance(r, dict)]
    meta: dict[str, Any] = {}
    meta_path = os.path.join(out, "run_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            loaded = json.load(fh)
            if isinstance(loaded, dict):
                meta = loaded
    return {
        "artifacts_dir": out,
        "summary": summary,
        "narrative": narrative,
        "comparison": comparison,
        "meta": meta,
    }
