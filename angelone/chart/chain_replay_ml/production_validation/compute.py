"""Production Validation Phase B — Holdout vs Unseen importance compute."""

from __future__ import annotations

import json
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from chain_replay_ml.feature_drift_studio.writer import load_studio_artifacts as load_drift_artifacts
from chain_replay_ml.feature_importance_studio.compute import _load_holdout_xy
from chain_replay_ml.feature_importance_studio.permutation import compute_permutation_importance
from chain_replay_ml.production_validation.load_unseen import load_unseen_xy
from chain_replay_ml.production_validation.rules import (
    build_dual_confidence,
    build_feature_rows,
    build_feature_validation_summary,
)
from chain_replay_ml.production_validation.types import ProductionValidationResult
from chain_replay_ml.production_validation.unseen_dataset import (
    load_unseen_dataset_status,
    resolve_unseen_dataset,
)
from chain_replay_ml.production_validation.writer import (
    patch_unseen_status_compute_note,
    write_validation_artifacts,
)
from chain_replay_ml.training.inference_runtime import (
    configure_prediction_model_for_inference,
)
from chain_replay_ml.training.model_runtime import (
    load_prediction_model,
    resolve_production_model_path,
)
from chain_replay_ml.training.paths import model_package_dir, safe_model_name
from chain_replay_ml.training.registry import _selected_feature_names, load_model_detail

ProgressCb = Callable[[dict[str, Any]], None]


def _read_json(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc if isinstance(doc, dict) else {}


def _prediction_kind(config: dict[str, Any]) -> str:
    raw = str(
        config.get("prediction_type") or config.get("predictionType") or ""
    ).strip().lower()
    if raw in ("binary", "classification", "classifier"):
        return "binary"
    return "regression"


def _perm_map(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in rows:
        feat = str(row.get("feature") or "").strip()
        if not feat:
            continue
        try:
            out[feat] = float(row.get("permutation_mean"))
        except (TypeError, ValueError):
            continue
    return out


def run_production_validation_compute(
    *,
    data_dir: str,
    model_name: str,
    package_dir: str | None = None,
    holdout_max_rows: int | None = 50_000,
    unseen_max_rows: int | None = None,
    permutation_n_repeats: int = 5,
    resolve_unseen_if_needed: bool = True,
    progress: ProgressCb | None = None,
) -> ProductionValidationResult:
    """Compare Holdout vs Unseen permutation importance for model selected features.

    Unseen coverage defaults to **all rows** for the resolved unseen trading
    day(s). Pass ``unseen_max_rows`` only with a clear UI/meta cap label.
    """

    def _tick(stage: str, **extra: Any) -> None:
        if progress:
            progress({"stage": stage, **extra})

    def _perm_progress(phase: str) -> ProgressCb | None:
        """Rewrite permutation feature ticks with holdout/unseen stage labels."""
        if not progress:
            return None

        def _cb(info: dict[str, Any]) -> None:
            payload = dict(info) if isinstance(info, dict) else {}
            payload["stage"] = phase
            progress(payload)

        return _cb

    safe = safe_model_name(model_name)
    pkg = package_dir or model_package_dir(data_dir, safe)
    if not os.path.isdir(pkg):
        return ProductionValidationResult(
            ok=False,
            model_name=safe,
            package_dir=pkg,
            artifacts_dir="",
            error=f"Model package not found: {pkg}",
        )

    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    _tick("resolve_unseen")
    t_u = time.perf_counter()
    status = load_unseen_dataset_status(data_dir, safe)
    if (not status or str(status.get("status") or "") != "ready") and resolve_unseen_if_needed:
        resolved = resolve_unseen_dataset(
            data_dir=data_dir,
            model_name=safe,
            create_if_missing=True,
        )
        status = resolved.as_dict()
    timings["resolve_unseen_sec"] = round(time.perf_counter() - t_u, 3)

    if not status or not status.get("ok"):
        return ProductionValidationResult(
            ok=False,
            model_name=safe,
            package_dir=pkg,
            artifacts_dir="",
            error=str(
                (status or {}).get("error")
                or (status or {}).get("message")
                or "Unseen dataset not ready — Resolve Unseen Dataset first."
            ),
            unseen_status=status or {},
        )
    if str(status.get("status") or "") == "empty":
        return ProductionValidationResult(
            ok=False,
            model_name=safe,
            package_dir=pkg,
            artifacts_dir="",
            error="No unseen days for this model (Master − Seen is empty).",
            unseen_status=status,
        )
    if str(status.get("status") or "") != "ready":
        return ProductionValidationResult(
            ok=False,
            model_name=safe,
            package_dir=pkg,
            artifacts_dir="",
            error=str(status.get("message") or "Unseen dataset status is not ready."),
            unseen_status=status,
        )

    dataset_name = str(status.get("dataset_name") or "").strip()
    if not dataset_name:
        return ProductionValidationResult(
            ok=False,
            model_name=safe,
            package_dir=pkg,
            artifacts_dir="",
            error="Unseen dataset_name missing from resolve status.",
            unseen_status=status,
        )

    try:
        doc = load_model_detail(data_dir, safe)
    except Exception:
        doc = {"model_name": safe, "config": _read_json(os.path.join(pkg, "config.json"))}

    config_raw = _read_json(os.path.join(pkg, "config.json"))
    if not config_raw:
        config_raw = _read_json(os.path.join(pkg, "training_config.json"))
    algorithm = str(config_raw.get("algorithm") or "xgboost")
    kind = _prediction_kind(config_raw)

    paths = {
        "config_json": os.path.join(pkg, "config.json"),
        "package_dir": pkg,
    }
    selected = _selected_feature_names(data_dir, safe, paths)
    if not selected:
        return ProductionValidationResult(
            ok=False,
            model_name=safe,
            package_dir=pkg,
            artifacts_dir="",
            error=(
                "Model has no selected feature list "
                "(config selected_features / features / walk_forward)."
            ),
            unseen_status=status,
        )

    _tick("load_model")
    t_model = time.perf_counter()
    try:
        model_path = resolve_production_model_path(pkg, algorithm=algorithm)
        model = load_prediction_model(model_path, algorithm)
    except Exception as exc:
        return ProductionValidationResult(
            ok=False,
            model_name=safe,
            package_dir=pkg,
            artifacts_dir="",
            error=f"Failed to load model: {exc}",
            unseen_status=status,
        )
    timings["load_model_sec"] = round(time.perf_counter() - t_model, 3)

    # Prefer XGB GPU predict (device=cuda, tree_method=hist) — same path as
    # prediction builds. Smoke-tests once; falls back to CPU with a clear reason.
    _tick("configure_inference")
    t_inf = time.perf_counter()
    infer_info = configure_prediction_model_for_inference(
        model, algorithm, prefer_gpu=True
    )
    timings["configure_inference_sec"] = round(time.perf_counter() - t_inf, 3)
    device_note = (
        f"{infer_info.device_label}"
        + (
            f" · {infer_info.fallback_reason}"
            if infer_info.fallback_reason and not infer_info.gpu_active
            else ""
        )
    )
    _tick(
        "configure_inference",
        device=infer_info.device_label,
        gpu_active=infer_info.gpu_active,
        fallback_reason=infer_info.fallback_reason,
        message=f"Inference device: {device_note}",
    )

    _tick("load_holdout")
    t_ho = time.perf_counter()
    try:
        X_ho, y_ho, holdout_features, holdout_meta = _load_holdout_xy(
            data_dir=data_dir,
            package_dir=pkg,
            model_name=safe,
            doc=doc,
            holdout_max_rows=holdout_max_rows,
        )
    except Exception as exc:
        return ProductionValidationResult(
            ok=False,
            model_name=safe,
            package_dir=pkg,
            artifacts_dir="",
            error=f"Failed to load holdout: {exc}",
            unseen_status=status,
        )
    timings["load_holdout_sec"] = round(time.perf_counter() - t_ho, 3)

    missing_ho = [f for f in selected if f not in holdout_features]
    if missing_ho:
        preview = ", ".join(missing_ho[:12])
        more = f" (+{len(missing_ho) - 12} more)" if len(missing_ho) > 12 else ""
        return ProductionValidationResult(
            ok=False,
            model_name=safe,
            package_dir=pkg,
            artifacts_dir="",
            error=(
                f"{len(missing_ho)} selected feature(s) missing from holdout matrix: "
                f"{preview}{more}"
            ),
            unseen_status=status,
        )

    target = str(holdout_meta.get("target") or config_raw.get("target") or "").strip()
    unseen_days = list(status.get("unseen_days") or [])

    _tick("load_unseen")
    t_un = time.perf_counter()
    try:
        X_un, y_un, unseen_meta = load_unseen_xy(
            data_dir=data_dir,
            dataset_name=dataset_name,
            features=selected,
            target=target,
            unseen_days=unseen_days,
            prediction_type=kind,
            parquet_path=status.get("parquet_path"),
            json_path=status.get("json_path"),
            max_rows=unseen_max_rows,
        )
    except Exception as exc:
        return ProductionValidationResult(
            ok=False,
            model_name=safe,
            package_dir=pkg,
            artifacts_dir="",
            error=f"Failed to load unseen: {exc}",
            unseen_status=status,
        )
    timings["load_unseen_sec"] = round(time.perf_counter() - t_un, 3)

    # Score intersection present on both matrices (selected already validated).
    features = [f for f in selected if f in X_ho.columns and f in X_un.columns]
    if not features:
        return ProductionValidationResult(
            ok=False,
            model_name=safe,
            package_dir=pkg,
            artifacts_dir="",
            error="No overlapping model features between holdout and unseen matrices.",
            unseen_status=status,
        )

    _tick("permutation_holdout", done=0, total=len(features))
    t_ph = time.perf_counter()
    holdout_perm = compute_permutation_importance(
        model,
        X_ho,
        y_ho,
        features,
        n_repeats=permutation_n_repeats,
        kind=kind,  # type: ignore[arg-type]
        progress=_perm_progress("permutation_holdout"),
    )
    timings["permutation_holdout_sec"] = round(time.perf_counter() - t_ph, 3)
    _tick("permutation_holdout", done=len(features), total=len(features))

    _tick("permutation_unseen", done=0, total=len(features))
    t_pu = time.perf_counter()
    unseen_perm = compute_permutation_importance(
        model,
        X_un,
        y_un,
        features,
        n_repeats=permutation_n_repeats,
        kind=kind,  # type: ignore[arg-type]
        progress=_perm_progress("permutation_unseen"),
    )
    timings["permutation_unseen_sec"] = round(time.perf_counter() - t_pu, 3)
    _tick("permutation_unseen", done=len(features), total=len(features))

    ho_map = _perm_map(holdout_perm)
    un_map = _perm_map(unseen_perm)
    scored = [f for f in features if f in ho_map and f in un_map]
    if not scored:
        return ProductionValidationResult(
            ok=False,
            model_name=safe,
            package_dir=pkg,
            artifacts_dir="",
            error=(
                "Permutation importance produced no scores "
                "(need ≥10 finite rows on holdout and unseen)."
            ),
            unseen_status=status,
        )

    # Soft-join Feature Drift Studio artifacts when present (KS / W / drift).
    drift_by_feature: dict[str, dict[str, Any]] = {}
    drift_meta: dict[str, Any] = {}
    try:
        drift_loaded = load_drift_artifacts(pkg)
    except Exception:
        drift_loaded = None
    if drift_loaded and isinstance(drift_loaded.get("comparison"), list):
        for drow in drift_loaded["comparison"]:
            if not isinstance(drow, dict):
                continue
            feat = str(drow.get("feature") or "").strip()
            if feat:
                drift_by_feature[feat] = drow
        if isinstance(drift_loaded.get("meta"), dict):
            drift_meta = drift_loaded["meta"]

    comparison, join_meta = build_feature_rows(
        holdout_importance={f: ho_map[f] for f in scored},
        unseen_importance={f: un_map[f] for f in scored},
        drift_by_feature=drift_by_feature or None,
    )

    day_count = int(
        unseen_meta.get("unseen_day_count")
        or status.get("unseen_day_count")
        or len(unseen_days)
        or 0
    )
    feature_summary = build_feature_validation_summary(comparison)
    dual = build_dual_confidence(
        comparison,
        unseen_day_count=day_count,
        feature_summary=feature_summary,
    )
    summary = {
        "model_name": safe,
        "dataset_name": dataset_name,
        "target": target,
        "feature_count_selected": len(selected),
        "feature_count_scored": len(comparison),
        "holdout_rows": holdout_meta.get("holdout_rows"),
        "unseen_rows": unseen_meta.get("unseen_rows"),
        "unseen_rows_full": unseen_meta.get("unseen_rows_full"),
        "unseen_rows_capped": unseen_meta.get("unseen_rows_capped"),
        "unseen_day_count": day_count,
        "unseen_days": unseen_meta.get("unseen_days_present") or unseen_days,
        "coverage": unseen_meta.get("coverage"),
        "feature_validation": feature_summary,
        "diagnosis": dual.get("diagnosis"),
        "production_confirmation": dual.get("production_confirmation"),
        "thresholds": dual.get("thresholds"),
        "drift_join": join_meta,
        "inference_device": infer_info.device_label,
        "gpu_active": infer_info.gpu_active,
        "inference_fallback_reason": infer_info.fallback_reason,
        "studio_version": "1.1.0-rank",
    }

    run_meta = {
        "run_id": str(uuid.uuid4()),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_name": safe,
        "package_dir": pkg,
        "dataset": holdout_meta.get("dataset"),
        "unseen_dataset": dataset_name,
        "target": target,
        "prediction_type": kind,
        "algorithm": algorithm,
        "selected_feature_count": len(selected),
        "scored_feature_count": len(comparison),
        "holdout_row_count": holdout_meta.get("holdout_rows"),
        "holdout_start": holdout_meta.get("holdout_start"),
        "holdout_stop": holdout_meta.get("holdout_stop"),
        "holdout_max_rows": holdout_max_rows,
        "unseen_row_count": unseen_meta.get("unseen_rows"),
        "unseen_rows_full": unseen_meta.get("unseen_rows_full"),
        "unseen_max_rows": unseen_max_rows,
        "unseen_rows_capped": unseen_meta.get("unseen_rows_capped"),
        "unseen_coverage": unseen_meta.get("coverage"),
        "unseen_days": unseen_meta.get("unseen_days_present") or unseen_days,
        "permutation_n_repeats": permutation_n_repeats,
        "importance_metric": "permutation_mean",
        "rank_formula": "rank by abs(permutation_mean); 1 = most important",
        "rank_change_formula": "holdout_rank - unseen_rank (negative = less important on unseen)",
        "importance_difference_formula": "unseen - holdout",
        "drift_join": join_meta,
        "drift_studio_meta": {
            "schema_version": drift_meta.get("schema_version"),
            "artifacts_present": bool(drift_by_feature),
        },
        "inference_device": infer_info.device_label,
        "inference_device_param": infer_info.device_param,
        "gpu_requested": infer_info.gpu_requested,
        "gpu_active": infer_info.gpu_active,
        "inference_fallback_reason": infer_info.fallback_reason,
        "predict_api": infer_info.predict_api,
        "timings_sec": timings,
        "wall_time_sec": round(time.perf_counter() - t0, 3),
        "studio_version": "1.1.0-rank",
        "holdout_load": holdout_meta,
        "unseen_load": unseen_meta,
    }

    _tick("write_artifacts")
    artifacts = write_validation_artifacts(
        pkg,
        comparison=comparison,
        summary=summary,
        run_meta=run_meta,
    )

    capped = bool(unseen_meta.get("unseen_rows_capped"))
    note = (
        f"computed · {len(comparison)} features · "
        f"{day_count} unseen day(s) · "
        + (
            f"unseen rows capped {unseen_meta.get('unseen_rows')}/{unseen_meta.get('unseen_rows_full')}"
            if capped
            else f"whole unseen day(s) · {unseen_meta.get('unseen_rows')} rows"
        )
    )
    patch_unseen_status_compute_note(
        pkg,
        compute_note=note,
        extra={"last_compute_ok": True, "last_compute_wall_sec": run_meta["wall_time_sec"]},
    )
    status = dict(status)
    status["compute_note"] = note

    _tick("done", done=1, total=1)
    return ProductionValidationResult(
        ok=True,
        model_name=safe,
        package_dir=pkg,
        artifacts_dir=artifacts,
        rows=comparison,
        summary=summary,
        meta=run_meta,
        unseen_status=status,
    )
