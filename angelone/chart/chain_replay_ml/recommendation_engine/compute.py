"""Recommendation Engine compute — read studio artifacts, emit planner suggestions.

No Dataset Engine, no retrain, no Importance/Distribution/Drift recompute.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Callable

from chain_replay_ml.recommendation_engine.config import (
    PLANNER_VERSION,
    SCHEMA_VERSION,
    merge_thresholds,
)
from chain_replay_ml.recommendation_engine.experiments import stamp_generation_meta
from chain_replay_ml.recommendation_engine.join import build_feature_rows, index_by_feature
from chain_replay_ml.recommendation_engine.rules import apply_rules, build_summary
from chain_replay_ml.recommendation_engine.types import RecommendationEngineResult
from chain_replay_ml.recommendation_engine.writer import write_studio_artifacts
from chain_replay_ml.training.paths import model_package_dir, safe_model_name

ProgressCb = Callable[[dict[str, Any]], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_optional_diagnostics(package_dir: str) -> dict[str, Any] | None:
    try:
        from chain_replay_ml.diagnostics_studio.writer import (
            load_studio_artifacts as load_diagnostics,
        )
    except ImportError:
        return None
    art = load_diagnostics(package_dir)
    if not art:
        return None
    summary = art.get("summary")
    return summary if isinstance(summary, dict) else None


def run_compute(
    *,
    data_dir: str,
    model_name: str,
    package_dir: str | None = None,
    thresholds: dict[str, Any] | None = None,
    family_by_name: dict[str, str] | None = None,
    require: tuple[str, ...] = (),
    progress: ProgressCb | None = None,
) -> RecommendationEngineResult:
    def _tick(stage: str, **extra: Any) -> None:
        if progress:
            progress({"stage": stage, **extra})

    safe = safe_model_name(model_name)
    pkg = package_dir or model_package_dir(data_dir, safe)
    empty = RecommendationEngineResult(
        ok=False, model_name=safe, package_dir=pkg, artifacts_dir=""
    )
    if not os.path.isdir(pkg):
        empty.error = f"Model package not found: {pkg}"
        return empty

    started_at = _utc_now()
    t0 = time.perf_counter()
    th = merge_thresholds(thresholds)

    _tick("load_studios")
    from chain_replay_ml.feature_distribution_studio.writer import (
        load_studio_artifacts as load_distribution,
    )
    from chain_replay_ml.feature_drift_studio.writer import (
        load_studio_artifacts as load_drift,
    )
    from chain_replay_ml.feature_importance_studio.writer import (
        load_studio_artifacts as load_importance,
    )

    imp_art = load_importance(pkg)
    dist_art = load_distribution(pkg)
    drift_art = load_drift(pkg)
    diag_summary = _load_optional_diagnostics(pkg)

    input_artifacts = {
        "importance": bool(imp_art and imp_art.get("comparison")),
        "distribution": bool(dist_art and dist_art.get("comparison")),
        "drift": bool(drift_art and (drift_art.get("comparison") or drift_art.get("meta"))),
        "diagnostics": bool(diag_summary),
    }

    for key in require:
        k = str(key).strip().lower()
        if k not in input_artifacts:
            empty.error = f"Unknown require studio: {key}"
            return empty
        if not input_artifacts[k]:
            empty.error = (
                f"Missing {k} studio artifacts on {safe}. Run that studio first."
            )
            return empty

    if not any(
        input_artifacts[k] for k in ("importance", "distribution", "drift")
    ):
        empty.error = (
            "No Feature Importance / Distribution / Drift artifacts found. "
            "Run Phases 4.1–4.3 first."
        )
        return empty

    _tick("join")
    imp_idx = index_by_feature((imp_art or {}).get("comparison"))
    dist_idx = index_by_feature((dist_art or {}).get("comparison"))
    drift_idx = index_by_feature((drift_art or {}).get("comparison"))
    rows = build_feature_rows(
        importance=imp_idx,
        drift=drift_idx,
        distribution=dist_idx,
    )

    _tick("rules")
    suggestions = apply_rules(
        rows,
        thresholds=th,
        diagnostics_summary=diag_summary,
        family_by_name=family_by_name,
    )
    suggestions = stamp_generation_meta(
        suggestions,
        model_name=safe,
        generated_at=started_at,
        planner_version=PLANNER_VERSION,
    )
    summary = build_summary(suggestions, rows)

    finished_at = _utc_now()
    wall = round(time.perf_counter() - t0, 3)
    run_meta = {
        "started_at": started_at,
        "finished_at": finished_at,
        "compute_time": wall,
        "wall_time_sec": wall,
        "planner_version": PLANNER_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact_version": SCHEMA_VERSION,
        "model_name": safe,
        "package_dir": pkg,
        "input_artifacts": input_artifacts,
        "require": list(require),
        "thresholds": th,
        "suggestion_count": len(suggestions),
        "feature_row_count": len(rows),
    }

    _tick("write_artifacts")
    artifacts_dir = write_studio_artifacts(
        pkg,
        suggestions=suggestions,
        summary=summary,
        run_meta=run_meta,
    )

    return RecommendationEngineResult(
        ok=True,
        model_name=safe,
        package_dir=pkg,
        artifacts_dir=artifacts_dir,
        suggestions=suggestions,
        summary=summary,
        meta=run_meta,
    )
