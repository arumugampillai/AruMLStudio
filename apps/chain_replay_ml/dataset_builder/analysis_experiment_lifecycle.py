"""Experiment lifecycle — Train → Holdout → WF → SHAP → Validation → result artifact.

Consumes only ``experiment_hypothesis`` (+ verifies bound discovery_bundle
fingerprint). Does not read live Discovery working tables.
"""
from __future__ import annotations

import os
from typing import Any, Callable

import numpy as np
import pandas as pd

from .analysis_artifacts import (
    KIND_EXPERIMENT_HYPOTHESIS,
    KIND_EXPERIMENT_RESULT,
    format_artifact_card,
    publish_artifact,
    require_artifact,
    verify_artifact_fingerprint,
)
from .analysis_experiments import (
    STATUS_MODEL_PRODUCED,
    STATUS_TRAINING,
    STATUS_VALIDATED,
    VALIDATION_GOOD,
    VALIDATION_PENDING,
    VALIDATION_UNSTABLE,
    ensure_experiments_schema,
    load_experiment,
    update_experiment_metrics,
)
from .analysis_lab_store import _AnalysisDb, _now_iso, resolve_parquet_path

ProgressCb = Callable[[float, str], None]


def _run_dataset(data_dir: str, run_id: str) -> dict[str, Any]:
    with _AnalysisDb(data_dir) as conn:
        run = conn.execute(
            "SELECT * FROM analysis_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if not run:
            raise ValueError(f"Unknown analysis run {run_id!r}")
        ds = conn.execute(
            "SELECT * FROM datasets WHERE dataset_id = ?",
            (str(run["dataset_id"]),),
        ).fetchone()
        if not ds:
            raise ValueError(f"Dataset missing for run {run_id!r}")
        return dict(ds)


def _pick_target(columns: list[str], preferred: str = "") -> str:
    if preferred and preferred in columns:
        return preferred
    for c in columns:
        if str(c).startswith("future_"):
            return str(c)
    raise ValueError("No prediction target column found (expected future_*)")


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - float(np.mean(yt))) ** 2))
    if ss_tot <= 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def _features_fingerprint(features: list[str]) -> str:
    import hashlib

    blob = "\n".join(sorted(str(f) for f in features)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def extract_model_feature_names(model: Any) -> list[str]:
    """Feature names the fitted XGBoost model was trained on (train order)."""
    names: list[str] = []
    # sklearn / XGBRegressor
    fn = getattr(model, "feature_names_in_", None)
    if fn is not None:
        try:
            names = [str(x) for x in list(fn)]
        except Exception:
            names = []
    if names:
        return names
    # Booster (native)
    booster = None
    getter = getattr(model, "get_booster", None)
    if callable(getter):
        try:
            booster = getter()
        except Exception:
            booster = None
    if booster is None and hasattr(model, "feature_names"):
        booster = model
    if booster is not None:
        try:
            raw = getattr(booster, "feature_names", None)
            if raw:
                names = [str(x) for x in list(raw)]
        except Exception:
            names = []
    return names


def assert_selected_features_match_model(
    selected_features: list[str],
    model: Any,
    *,
    context: str = "experiment train",
) -> list[str]:
    """Sanity-check: persisted feature list == columns passed into XGBoost.

    Compares length and exact order. Returns the model feature-name list.
    Raises ``ValueError`` on mismatch so a Champion Bundle cannot silently
    record a different input set than the trained model.
    """
    selected = [str(f) for f in selected_features]
    model_names = extract_model_feature_names(model)
    if not model_names:
        # Older / numpy-only fits may omit names — still require count via n_features_in_
        n_in = getattr(model, "n_features_in_", None)
        if n_in is not None and int(n_in) != len(selected):
            raise ValueError(
                f"{context}: selected_features has {len(selected)} names but "
                f"model.n_features_in_={n_in}"
            )
        return selected
    if len(selected) != len(model_names):
        raise ValueError(
            f"{context}: selected_features len={len(selected)} != "
            f"model feature_names len={len(model_names)}"
        )
    if selected != model_names:
        # Find first mismatch for a clear error
        for i, (a, b) in enumerate(zip(selected, model_names)):
            if a != b:
                raise ValueError(
                    f"{context}: feature order mismatch at index {i}: "
                    f"selected={a!r} model={b!r}"
                )
        raise ValueError(
            f"{context}: selected_features != model.feature_names "
            f"(len={len(selected)})"
        )
    return model_names


def _train_eval_hypothesis(
    parquet_path: str,
    features: list[str],
    target: str,
    *,
    holdout_pct: float = 0.15,
    n_wf_folds: int = 3,
) -> dict[str, Any]:
    """Lightweight XGB train for Experiment Manager (analysis parquet only).

    GPU-first via ``analysis_train_device``; falls back to CPU automatically.
    """
    from .analysis_train_device import (
        DEVICE_CPU,
        compute_shap_mean_abs_gpu_first,
        fit_xgb_regressor_gpu_first,
        format_device_label,
    )

    feats = list(dict.fromkeys(str(f) for f in features if f))
    if not feats:
        raise ValueError("No features in experiment hypothesis")
    df = pd.read_parquet(parquet_path, columns=list(dict.fromkeys(feats + [target])))
    df = df.dropna(subset=feats + [target]).reset_index(drop=True)
    if len(df) < 40:
        raise ValueError(f"Too few complete rows for experiment train ({len(df)})")

    X = df[feats].astype(float)
    y = df[target].astype(float).to_numpy()
    n = len(df)
    hold_n = max(int(n * holdout_pct), 10)
    train_end = n - hold_n
    if train_end < 20:
        raise ValueError("Not enough rows after holdout split")

    model, device_info = fit_xgb_regressor_gpu_first(
        X.iloc[:train_end],
        y[:train_end],
        base_params={
            "n_estimators": 40,
            "max_depth": 3,
            "learning_rate": 0.1,
            "objective": "reg:squarederror",
            "verbosity": 0,
        },
    )
    # Hard guarantee: Champion/result feature list == XGBoost train columns
    model_feature_names = assert_selected_features_match_model(
        feats, model, context="experiment holdout train"
    )
    train_device = str(device_info.get("train_device") or DEVICE_CPU)
    hold_pred = model.predict(X.iloc[train_end:])
    holdout_r2 = _r2(y[train_end:], hold_pred)
    holdout_rmse = _rmse(y[train_end:], hold_pred)

    # Simple expanding walk-forward on the train region (same device policy)
    fold_scores: list[float] = []
    fold_size = max(train_end // (n_wf_folds + 1), 15)
    for i in range(n_wf_folds):
        te_end = min(fold_size * (i + 2), train_end)
        tr_end = max(te_end - fold_size, 15)
        if tr_end < 15 or te_end - tr_end < 5:
            continue
        m, _ = fit_xgb_regressor_gpu_first(
            X.iloc[:tr_end],
            y[:tr_end],
            base_params={
                "n_estimators": 30,
                "max_depth": 3,
                "learning_rate": 0.1,
                "objective": "reg:squarederror",
                "verbosity": 0,
            },
        )
        pred = m.predict(X.iloc[tr_end:te_end])
        fold_scores.append(_r2(y[tr_end:te_end], pred))
    wf_r2 = float(np.mean(fold_scores)) if fold_scores else holdout_r2
    wf_std = float(np.std(fold_scores)) if len(fold_scores) > 1 else 0.0

    # SHAP on holdout sample — GPU path when the trained model is CUDA
    sample = X.iloc[train_end : train_end + min(200, hold_n)]
    shap_rows, shap_device, shap_error = compute_shap_mean_abs_gpu_first(
        model,
        sample,
        feats,
        prefer_gpu=(train_device == "GPU"),
    )

    # Stricter labels so saturated R²≈0.99 does not all look identical/"Good"
    if holdout_r2 >= 0.85 and wf_r2 >= 0.80 and wf_std <= 0.05:
        validation_label = "Excellent"
    elif holdout_r2 >= 0.05 and wf_r2 >= 0.0 and wf_std <= 0.15:
        validation_label = VALIDATION_GOOD
    elif holdout_r2 < 0.0 or wf_r2 < -0.05:
        validation_label = VALIDATION_UNSTABLE
    else:
        validation_label = VALIDATION_PENDING

    device_label = format_device_label(
        train_device,
        shap_device=shap_device,
        fallback_reason=device_info.get("fallback_reason"),
    )
    return {
        "holdout_r2": float(holdout_r2),
        "holdout_rmse": float(holdout_rmse),
        "walk_forward_r2": float(wf_r2),
        "wf_fold_scores": fold_scores,
        "wf_std": wf_std,
        "n_rows": n,
        "n_train": train_end,
        "n_holdout": hold_n,
        "features": feats,
        "features_fingerprint": _features_fingerprint(feats),
        "n_features": len(feats),
        "model_feature_names": model_feature_names,
        "feature_names_match": model_feature_names == list(feats),
        "target": target,
        "shap": shap_rows,
        "shap_error": shap_error,
        "validation_label": validation_label,
        "model": model,
        "train_device": train_device,
        "shap_device": shap_device,
        "device_label": device_label,
        "executed_device": device_info.get("executed_device"),
        "gpu_name": device_info.get("gpu_name"),
        "device_fallback_reason": device_info.get("fallback_reason"),
    }


def run_experiment_lifecycle(
    data_dir: str,
    experiment_id: str,
    *,
    target: str = "",
    on_progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Full lifecycle owned by Experiment Manager.

    Steps: verify discovery_bundle fingerprint → Train → Holdout →
    Walk-forward → SHAP → Validation → publish experiment_result.
    """
    eid = str(experiment_id or "").strip()
    exp = load_experiment(data_dir, eid, verify_bundle=True)
    if not exp:
        raise ValueError(f"Unknown experiment {eid!r}")

    rid = str(exp["run_id"])
    hyp_id = str(exp.get("hypothesis_artifact_id") or "").strip()
    if not hyp_id:
        raise ValueError(f"{eid} has no hypothesis_artifact_id")

    if on_progress:
        on_progress(0.05, "Loading experiment_hypothesis artifact…")
    hyp = require_artifact(
        data_dir, hyp_id, expected_kind=KIND_EXPERIMENT_HYPOTHESIS
    )
    # Re-verify discovery bundle fingerprint stored on the experiment
    bundle_id = str(exp.get("discovery_bundle_id") or "").strip()
    bundle_fp = str(exp.get("discovery_bundle_fingerprint") or "").strip()
    if bundle_id and bundle_fp:
        verify_artifact_fingerprint(data_dir, bundle_id, bundle_fp)

    payload = dict(hyp.get("payload") or {})
    # Prefer authoritative selected_features (Top-N / flat strategies)
    features = [
        str(f)
        for f in (
            payload.get("selected_features")
            or (payload.get("feature_set") or {}).get("features")
            or (payload.get("final_feature_dataset") or {}).get("features")
            or []
        )
        if str(f).strip()
    ]
    if not features:
        features = [
            str(r.get("representative"))
            for r in (payload.get("families") or [])
            if r.get("representative")
        ]
        # Expand Top-N representatives when present
        extra: list[str] = []
        for r in payload.get("families") or []:
            for name in r.get("representatives") or []:
                s = str(name).strip()
                if s and s not in features and s not in extra:
                    extra.append(s)
        features = list(dict.fromkeys(features + extra))
    if not features:
        features = [
            str(v) for v in dict(payload.get("family_reps") or {}).values() if v
        ]
    features = list(dict.fromkeys(features))
    if not features:
        raise ValueError(f"{eid} hypothesis has no feature representatives")

    feature_selection = dict(payload.get("feature_selection") or {})

    from .analysis_experiments import (
        build_feature_set,
        build_parent_diff,
        _load_parent_family_reps,
    )

    feature_set = dict(payload.get("feature_set") or {})
    if not feature_set.get("features"):
        feature_set = build_feature_set(
            families=list(payload.get("families") or []),
            features=features,
            discovery_bundle_id=bundle_id or None,
            parent_experiment_id=str(exp.get("parent_experiment_id") or "")
            or None,
            variant_changes=list(payload.get("variant_changes") or []),
        )
    else:
        # Ensure trained feature list is authoritative
        feature_set = dict(feature_set)
        feature_set["features"] = features
        feature_set["count"] = len(features)
        from .analysis_experiments import features_fingerprint

        feature_set["hash"] = features_fingerprint(features)

    parent_reps = _load_parent_family_reps(
        data_dir, exp.get("parent_experiment_id")
    )
    parent_diff = build_parent_diff(
        parent_experiment_id=str(exp.get("parent_experiment_id") or "") or None,
        parent_family_reps=parent_reps,
        current_family_reps=exp.get("family_reps"),
        variant_changes=list(
            payload.get("variant_changes")
            or exp.get("variant_changes_list")
            or []
        ),
    )

    if on_progress:
        on_progress(0.1, f"Status → {STATUS_TRAINING}")
    with _AnalysisDb(data_dir) as conn:
        ensure_experiments_schema(conn)
        conn.execute(
            "UPDATE experiments SET status = ?, updated_at = ? WHERE experiment_id = ?",
            (STATUS_TRAINING, _now_iso(), eid),
        )

    ds = _run_dataset(data_dir, rid)
    path = resolve_parquet_path(data_dir, ds)
    if on_progress:
        on_progress(0.2, f"Loading parquet · {os.path.basename(path)}")

    # Discover columns / target
    import pyarrow.parquet as pq

    columns = [str(n) for n in pq.read_schema(path).names]
    missing = [f for f in features if f not in columns]
    if missing:
        raise ValueError(
            f"Hypothesis features missing from dataset: {missing[:8]}"
        )
    tgt = _pick_target(columns, preferred=str(target or "").strip())

    if on_progress:
        device_hint = ""
        try:
            from .analysis_train_device import resolve_experiment_xgb_plan

            plan = resolve_experiment_xgb_plan()
            device_hint = (
                f" · {plan.device_label}"
                + (f" ({plan.gpu_name})" if plan.gpu_name else "")
            )
        except Exception:
            device_hint = ""
        on_progress(0.35, f"Train → Holdout → Walk-forward{device_hint}…")
    eval_out = _train_eval_hypothesis(path, features, tgt)
    # Authoritative train column list (already asserted == model feature names)
    trained_features = [
        str(f)
        for f in (
            eval_out.get("model_feature_names")
            or eval_out.get("features")
            or features
        )
    ]
    if trained_features != list(features):
        raise ValueError(
            f"{eid}: train column list drifted from hypothesis features "
            f"({len(features)} vs {len(trained_features)})"
        )
    features = trained_features
    feature_set = dict(feature_set)
    feature_set["features"] = features
    feature_set["count"] = len(features)
    from .analysis_experiments import features_fingerprint

    feature_set["hash"] = features_fingerprint(features)
    feature_set["model_feature_names"] = list(
        eval_out.get("model_feature_names") or features
    )
    feature_set["feature_names_match"] = bool(
        eval_out.get("feature_names_match", True)
    )

    model_name = f"Exp_{eid.replace('-', '_')}"
    train_device = str(eval_out.get("train_device") or "CPU")
    shap_device = str(eval_out.get("shap_device") or "CPU")
    device_label = str(eval_out.get("device_label") or train_device)
    # Persist a lightweight model package for SHAP reuse / inspection
    model_path = ""
    try:
        from chain_replay_ml.training.paths import model_package_dir
        import json
        import tempfile

        def _atomic_json(path: str, payload: dict) -> None:
            directory = os.path.dirname(path) or "."
            fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=directory)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp, path)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise

        pkg = model_package_dir(data_dir, model_name)
        os.makedirs(pkg, exist_ok=True)
        model_ubj = os.path.join(pkg, "model.ubj")
        eval_out["model"].save_model(model_ubj)
        model_path = model_ubj
        _atomic_json(
            os.path.join(pkg, "config.json"),
            {
                "selected_features": features,
                "features": features,
                "model_feature_names": list(
                    eval_out.get("model_feature_names") or features
                ),
                "feature_names_match": bool(
                    eval_out.get("feature_names_match", True)
                ),
                "feature_set": feature_set,
                "algorithm": "xgboost",
                "target": tgt,
                "experiment_id": eid,
                "registry_scope": "experiment",
                "origin": "research_experiment",
                "train_device": train_device,
                "shap_device": shap_device,
                "device_label": device_label,
                "executed_device": eval_out.get("executed_device"),
                "gpu_name": eval_out.get("gpu_name"),
            },
        )
        _atomic_json(
            os.path.join(pkg, "registry.json"),
            {
                "model_name": model_name,
                "algorithm": "xgboost",
                "target": tgt,
                "status": "trained",
                "experiment_id": eid,
                "registry_scope": "experiment",
                "origin": "research_experiment",
                "train_device": train_device,
                "device_label": device_label,
            },
        )
    except Exception:
        model_name = model_name  # keep name even if package write fails

    if on_progress:
        on_progress(
            0.65,
            f"Status → {STATUS_MODEL_PRODUCED} · device {device_label}",
        )
    with _AnalysisDb(data_dir) as conn:
        ensure_experiments_schema(conn)
        conn.execute(
            "UPDATE experiments SET status = ?, model_name = ?, updated_at = ? "
            "WHERE experiment_id = ?",
            (STATUS_MODEL_PRODUCED, model_name, _now_iso(), eid),
        )

    if on_progress:
        on_progress(0.75, f"Holdout → Walk-forward → SHAP ({shap_device}) → Validation…")

    result_payload = {
        "experiment_id": eid,
        "hypothesis_artifact_id": hyp_id,
        "discovery_bundle_id": bundle_id,
        "discovery_bundle_fingerprint": bundle_fp,
        "model_name": model_name,
        "model_path": model_path or None,
        "target": tgt,
        "selected_features": features,
        "features": features,
        "model_feature_names": list(
            eval_out.get("model_feature_names") or features
        ),
        "feature_names_match": bool(eval_out.get("feature_names_match", True)),
        "feature_set": feature_set,
        "feature_selection": feature_selection or None,
        "final_feature_dataset": {
            "count": len(features),
            "features": features,
            "hash": feature_set.get("hash"),
            "feature_selection": feature_selection or None,
        },
        "parent_diff": parent_diff,
        "features_fingerprint": feature_set.get("hash")
        or eval_out.get("features_fingerprint"),
        "n_features": feature_set.get("count")
        or eval_out.get("n_features")
        or len(features),
        "holdout_r2": eval_out["holdout_r2"],
        "holdout_rmse": eval_out.get("holdout_rmse"),
        "walk_forward_r2": eval_out["walk_forward_r2"],
        "wf_fold_scores": eval_out["wf_fold_scores"],
        "wf_std": eval_out.get("wf_std"),
        "shap": eval_out["shap"],
        "shap_error": eval_out.get("shap_error") or "",
        "validation_label": eval_out["validation_label"],
        "n_rows": eval_out["n_rows"],
        "train_device": train_device,
        "shap_device": shap_device,
        "device_label": device_label,
        "executed_device": eval_out.get("executed_device"),
        "gpu_name": eval_out.get("gpu_name"),
        "device_fallback_reason": eval_out.get("device_fallback_reason"),
        "parent_experiment_id": exp.get("parent_experiment_id"),
        "variant_changes": list(
            payload.get("variant_changes")
            or exp.get("variant_changes_list")
            or []
        ),
        "card": {
            "dataset": ds.get("name") or ds.get("dataset_id"),
            "n_features": len(features),
            "n_families": len(feature_set.get("families") or features),
            "holdout": round(float(eval_out["holdout_r2"]), 6),
            "walk_forward": round(float(eval_out["walk_forward_r2"]), 6),
            "holdout_rmse": round(float(eval_out.get("holdout_rmse") or 0), 6),
            "device": device_label,
            "train_device": train_device,
            "shap_device": shap_device,
            "features_fingerprint": str(
                feature_set.get("hash")
                or eval_out.get("features_fingerprint")
                or ""
            )[:16],
            "parent_experiment_id": exp.get("parent_experiment_id"),
            "n_changed_families": parent_diff.get("n_changed_families"),
        },
        "completed_at": _now_iso(),
    }
    # Drop non-serializable model before publish
    result_art = publish_artifact(
        data_dir,
        rid,
        KIND_EXPERIMENT_RESULT,
        result_payload,
        parent_ids=[hyp_id],
        label=f"Result {eid}",
        reuse_identical=False,
    )

    summary = (
        f"Holdout R²={eval_out['holdout_r2']:.6f} · "
        f"WF R²={eval_out['walk_forward_r2']:.6f} · "
        f"RMSE={float(eval_out.get('holdout_rmse') or 0):.6f} · "
        f"device={device_label} · "
        f"feats={eval_out.get('n_features')} · "
        f"fp={str(eval_out.get('features_fingerprint') or '')[:10]} · "
        f"{eval_out['validation_label']}"
    )
    update_experiment_metrics(
        data_dir,
        eid,
        model_name=model_name,
        holdout_score=float(eval_out["holdout_r2"]),
        walk_forward_score=float(eval_out["walk_forward_r2"]),
        validation_label=str(eval_out["validation_label"]),
        validation_summary=summary,
        status=STATUS_VALIDATED,
        train_device=train_device,
        shap_device=shap_device,
        device_label=device_label,
    )
    with _AnalysisDb(data_dir) as conn:
        ensure_experiments_schema(conn)
        conn.execute(
            """
            UPDATE experiments
            SET result_artifact_id = ?, updated_at = ?
            WHERE experiment_id = ?
            """,
            (result_art["artifact_id"], _now_iso(), eid),
        )

    if on_progress:
        on_progress(1.0, f"Validated · {summary}")

    out_exp = load_experiment(data_dir, eid, verify_bundle=True) or {}
    return {
        "experiment_id": eid,
        "status": STATUS_VALIDATED,
        "started": True,
        "message": (
            f"{eid} Validated · model scores: {summary}\n"
            "(Holdout / Walk-forward / Validation belong to the model, "
            "not the feature-set snapshot.)"
        ),
        "result_artifact_id": result_art["artifact_id"],
        "result_fingerprint": result_art.get("fingerprint"),
        "result_card": format_artifact_card(result_art),
        "holdout_r2": eval_out["holdout_r2"],
        "walk_forward_r2": eval_out["walk_forward_r2"],
        "holdout_rmse": eval_out.get("holdout_rmse"),
        "features_fingerprint": eval_out.get("features_fingerprint"),
        "n_features": eval_out.get("n_features"),
        "validation_label": eval_out["validation_label"],
        "model_name": model_name,
        "train_device": train_device,
        "shap_device": shap_device,
        "device_label": device_label,
        "experiment": out_exp,
    }


__all__ = [
    "assert_selected_features_match_model",
    "extract_model_feature_names",
    "run_experiment_lifecycle",
]
