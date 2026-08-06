"""Multi-model Feature Studio compute — join-only (no studio recompute)."""

from __future__ import annotations

import os
import time
from typing import Any, Callable

from chain_replay_ml.feature_distribution_studio.writer import (
    load_studio_artifacts as load_distribution,
)
from chain_replay_ml.feature_drift_studio.writer import load_studio_artifacts as load_drift
from chain_replay_ml.feature_importance_studio.writer import (
    load_studio_artifacts as load_importance,
)
from chain_replay_ml.multi_model_studio.comparison import (
    build_comparison_rows,
    index_by_feature,
)
from chain_replay_ml.multi_model_studio.types import MultiModelStudioResult
from chain_replay_ml.multi_model_studio.writer import (
    pair_dirname,
    write_studio_artifacts,
)
from chain_replay_ml.training.paths import model_package_dir, safe_model_name

ProgressCb = Callable[[dict[str, Any]], None]

_STUDIO_LOADERS = {
    "importance": load_importance,
    "distribution": load_distribution,
    "drift": load_drift,
}


def _load_side(
    package_dir: str,
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, bool]]:
    indexed: dict[str, dict[str, dict[str, Any]]] = {
        "importance": {},
        "distribution": {},
        "drift": {},
    }
    flags = {k: False for k in _STUDIO_LOADERS}
    for key, loader in _STUDIO_LOADERS.items():
        art = loader(package_dir)
        if not art:
            continue
        rows = art.get("comparison") or []
        if not isinstance(rows, list):
            continue
        indexed[key] = index_by_feature(rows)
        flags[key] = bool(indexed[key])
    return indexed, flags


def run_compute(
    *,
    data_dir: str,
    model_a: str,
    model_b: str,
    package_dir_a: str | None = None,
    package_dir_b: str | None = None,
    require: tuple[str, ...] = (),
    progress: ProgressCb | None = None,
) -> MultiModelStudioResult:
    def _tick(stage: str, **extra: Any) -> None:
        if progress:
            progress({"stage": stage, **extra})

    safe_a = safe_model_name(model_a)
    safe_b = safe_model_name(model_b)
    empty = MultiModelStudioResult(
        ok=False,
        model_a=safe_a,
        model_b=safe_b,
        pair_dir="",
        artifacts_dir="",
    )
    if safe_a == safe_b:
        empty.error = "Model A and Model B must be different packages"
        return empty

    pkg_a = package_dir_a or model_package_dir(data_dir, safe_a)
    pkg_b = package_dir_b or model_package_dir(data_dir, safe_b)
    if not os.path.isdir(pkg_a):
        empty.error = f"Model package not found: {pkg_a}"
        return empty
    if not os.path.isdir(pkg_b):
        empty.error = f"Model package not found: {pkg_b}"
        return empty

    t0 = time.perf_counter()
    _tick("load_a")
    side_a, flags_a = _load_side(pkg_a)
    _tick("load_b")
    side_b, flags_b = _load_side(pkg_b)

    for key in require:
        k = str(key).strip().lower()
        if k not in flags_a:
            empty.error = f"Unknown require studio: {key}"
            return empty
        if not flags_a[k]:
            empty.error = (
                f"Missing {k} studio artifacts on Model A ({safe_a}). "
                f"Run that studio first."
            )
            return empty
        if not flags_b[k]:
            empty.error = (
                f"Missing {k} studio artifacts on Model B ({safe_b}). "
                f"Run that studio first."
            )
            return empty

    if not any(flags_a.values()) and not any(flags_b.values()):
        empty.error = (
            "No Feature Importance / Distribution / Drift artifacts found "
            "on either model. Run Phases 4.1–4.3 first."
        )
        return empty
    if not any(flags_a.values()):
        empty.error = f"No studio artifacts on Model A ({safe_a})"
        return empty
    if not any(flags_b.values()):
        empty.error = f"No studio artifacts on Model B ({safe_b})"
        return empty

    _tick("join")
    comparison = build_comparison_rows(
        imp_a=side_a["importance"],
        imp_b=side_b["importance"],
        drift_a=side_a["drift"],
        drift_b=side_b["drift"],
        dist_a=side_a["distribution"],
        dist_b=side_b["distribution"],
    )

    feats_a = (
        set(side_a["importance"])
        | set(side_a["drift"])
        | set(side_a["distribution"])
    )
    feats_b = (
        set(side_b["importance"])
        | set(side_b["drift"])
        | set(side_b["distribution"])
    )
    common = feats_a & feats_b
    only_a = feats_a - feats_b
    only_b = feats_b - feats_a

    pair = pair_dirname(safe_a, safe_b)
    run_meta = {
        "studio_version": "4.4.0",
        "model_a": safe_a,
        "model_b": safe_b,
        "package_dir_a": pkg_a,
        "package_dir_b": pkg_b,
        "pair_dirname": pair,
        "artifacts_loaded": {"a": flags_a, "b": flags_b},
        "feature_count_a": len(feats_a),
        "feature_count_b": len(feats_b),
        "common_count": len(common),
        "only_a_count": len(only_a),
        "only_b_count": len(only_b),
        "row_count": len(comparison),
        "require": list(require),
        "wall_time_sec": round(time.perf_counter() - t0, 3),
    }

    _tick("write_artifacts")
    artifacts_dir = write_studio_artifacts(
        data_dir,
        safe_a,
        safe_b,
        comparison=comparison,
        run_meta=run_meta,
    )
    run_meta["pair_dir"] = artifacts_dir

    return MultiModelStudioResult(
        ok=True,
        model_a=safe_a,
        model_b=safe_b,
        pair_dir=artifacts_dir,
        artifacts_dir=artifacts_dir,
        comparison=comparison,
        meta=run_meta,
    )
