"""Diagnostics Studio compute — join/summarize only (no holdout recompute)."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

from chain_replay_ml.diagnostics_studio.comparison import (
    build_comparison_rows,
    index_by_feature,
)
from chain_replay_ml.diagnostics_studio.summarize import build_summary_and_narrative
from chain_replay_ml.diagnostics_studio.types import DiagnosticsStudioResult
from chain_replay_ml.diagnostics_studio.writer import write_studio_artifacts
from chain_replay_ml.feature_distribution_studio.writer import (
    load_studio_artifacts as load_distribution,
)
from chain_replay_ml.feature_drift_studio.writer import load_studio_artifacts as load_drift
from chain_replay_ml.feature_importance_studio.writer import (
    load_studio_artifacts as load_importance,
)
from chain_replay_ml.training.paths import model_package_dir, safe_model_name
from chain_replay_ml.training.registry import load_model_detail

ProgressCb = Callable[[dict[str, Any]], None]


def _read_json(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc if isinstance(doc, dict) else {}


def run_compute(
    *,
    data_dir: str,
    model_name: str,
    package_dir: str | None = None,
    require: tuple[str, ...] = (),
    progress: ProgressCb | None = None,
) -> DiagnosticsStudioResult:
    def _tick(stage: str, **extra: Any) -> None:
        if progress:
            progress({"stage": stage, **extra})

    safe = safe_model_name(model_name)
    pkg = package_dir or model_package_dir(data_dir, safe)
    empty = DiagnosticsStudioResult(
        ok=False, model_name=safe, package_dir=pkg, artifacts_dir=""
    )
    if not os.path.isdir(pkg):
        empty.error = f"Model package not found: {pkg}"
        return empty

    t0 = time.perf_counter()
    try:
        doc = load_model_detail(data_dir, safe)
        if not isinstance(doc, dict):
            doc = {"model_name": safe}
    except Exception:
        doc = {"model_name": safe}

    # Soft-merge package metrics files when detail omits them.
    if not isinstance(doc.get("metrics"), dict):
        metrics = _read_json(os.path.join(pkg, "metrics.json"))
        if metrics:
            doc["metrics"] = metrics
    if not isinstance(doc.get("production_metrics"), dict):
        prod = _read_json(os.path.join(pkg, "production_metrics.json"))
        if prod:
            doc["production_metrics"] = prod

    _tick("load_studios")
    imp_art = load_importance(pkg)
    dist_art = load_distribution(pkg)
    drift_art = load_drift(pkg)

    joins = {
        "importance": bool(imp_art and imp_art.get("comparison")),
        "distribution": bool(dist_art and dist_art.get("comparison")),
        "drift": bool(drift_art and (drift_art.get("comparison") or drift_art.get("meta"))),
    }

    for key in require:
        k = str(key).strip().lower()
        if k not in joins:
            empty.error = f"Unknown require studio: {key}"
            return empty
        if not joins[k]:
            empty.error = (
                f"Missing {k} studio artifacts on {safe}. Run that studio first."
            )
            return empty

    if not any(joins.values()):
        empty.error = (
            "No Feature Importance / Distribution / Drift artifacts found. "
            "Run Phases 4.1–4.3 first."
        )
        return empty

    imp_idx = index_by_feature((imp_art or {}).get("comparison"))
    dist_idx = index_by_feature((dist_art or {}).get("comparison"))
    drift_rows = list((drift_art or {}).get("comparison") or [])
    drift_idx = index_by_feature(drift_rows)
    drift_meta = dict((drift_art or {}).get("meta") or {})

    _tick("comparison")
    comparison = build_comparison_rows(
        importance=imp_idx,
        drift=drift_idx,
        distribution=dist_idx,
    )

    _tick("summarize")
    summary, narrative = build_summary_and_narrative(
        doc=doc if isinstance(doc, dict) else {},
        drift_rows=drift_rows,
        drift_meta=drift_meta,
        comparison=comparison,
        joins=joins,
    )

    run_meta = {
        "studio_version": "4.5.0",
        "model_name": safe,
        "package_dir": pkg,
        "joins": summary.get("joins"),
        "require": list(require),
        "row_count": len(comparison),
        "primary_cause": summary.get("primary_cause"),
        "wall_time_sec": round(time.perf_counter() - t0, 3),
    }

    _tick("write_artifacts")
    artifacts_dir = write_studio_artifacts(
        pkg,
        summary=summary,
        narrative=narrative,
        comparison=comparison,
        run_meta=run_meta,
    )

    return DiagnosticsStudioResult(
        ok=True,
        model_name=safe,
        package_dir=pkg,
        artifacts_dir=artifacts_dir,
        summary=summary,
        narrative=narrative,
        comparison=comparison,
        meta=run_meta,
    )
