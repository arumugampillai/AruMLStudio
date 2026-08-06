"""Confidence Model inference over Prediction Dataset (batch, persist 0/1 only).

Operating Threshold is read from each Confidence Model's persisted metadata —
never hard-coded. Confidence Dataset is not used at inference time.

Inference is per model key (Target Hit, RR 1:1, …); each writes its own
``confidence_*_pred`` columns on the Prediction Dataset.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from .confidence_dataset import resolve_regression_selected_features
from .confidence_manifest import (
    TARGET_BY_KEY,
    mark_inference_out_of_date,
    model_package_dir_for,
    read_manifest,
    write_manifest,
)
from .prediction_feature_store import PredictionFeatureStore
from .store import ModelLabStore

# model_key → prediction_dataset column names (Market + Replay-Based)
from .target_spec import build_inference_columns_map

INFERENCE_COLUMNS: dict[str, dict[str, str]] = build_inference_columns_map()

# Backward-compatible aliases (Target Hit / Path Touch)
INFERENCE_MODEL_KEY = "target_hit"
PRED_COL = INFERENCE_COLUMNS["target_hit"]["pred"]
MODEL_ID_COL = INFERENCE_COLUMNS["target_hit"]["model_id"]
THRESHOLD_COL = INFERENCE_COLUMNS["target_hit"]["threshold"]
CREATED_COL = INFERENCE_COLUMNS["target_hit"]["created"]

DEFAULT_BATCH_SIZE = 50_000

ProgressCb = Callable[[dict[str, Any]], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_inference_block() -> dict[str, Any]:
    return {
        "status": "not_run",
        "rows": 0,
        "positive": 0,
        "negative": 0,
        "nulls": 0,
        "model_key": None,
        "model_id": None,
        "threshold": None,
        "completed_at": None,
        "validation": None,
        "error": None,
        "stale_reason": None,
    }


def inference_columns_for(model_key: str) -> dict[str, str]:
    cols = INFERENCE_COLUMNS.get(model_key)
    if not cols:
        raise ValueError(f"No inference columns for model key: {model_key}")
    return cols


def _load_classifier(model_path: str, features: list[str]) -> Any:
    import xgboost as xgb

    from chain_replay_ml.training.xgb_trainer import booster_wrapper

    bst = xgb.Booster()
    bst.load_model(model_path)
    return booster_wrapper(bst, features)


def resolve_operating_threshold(lab_db_path: str, model_key: str = INFERENCE_MODEL_KEY) -> dict[str, Any]:
    """Load Operating Threshold from Confidence Model metadata."""
    if model_key not in TARGET_BY_KEY:
        return {"ok": False, "error": f"Unknown confidence model: {model_key}"}
    doc = read_manifest(lab_db_path)
    entry = (doc.get("models") or {}).get(model_key) or {}
    if str(entry.get("status") or "") not in ("ready", "stale"):
        return {
            "ok": False,
            "error": f"{TARGET_BY_KEY[model_key]['label']} is not trained.",
        }
    thr = entry.get("operating_threshold")
    if thr is None:
        pkg = entry.get("package_dir") or model_package_dir_for(lab_db_path, model_key)
        metrics_path = os.path.join(pkg, "metrics.json") if pkg else None
        if metrics_path and os.path.isfile(metrics_path):
            try:
                with open(metrics_path, encoding="utf-8") as fh:
                    meta = json.load(fh)
                if isinstance(meta, dict) and meta.get("operating_threshold") is not None:
                    thr = meta.get("operating_threshold")
            except (OSError, json.JSONDecodeError):
                pass
    if thr is None:
        return {
            "ok": False,
            "error": (
                f"No Operating Threshold set for {TARGET_BY_KEY[model_key]['label']}. "
                "Open Evaluate → Threshold Analysis, choose a threshold "
                "(e.g. 0.70), and Save Operating Threshold before inference."
            ),
        }
    try:
        thr_f = float(thr)
    except (TypeError, ValueError):
        return {"ok": False, "error": f"Invalid operating threshold: {thr}"}
    if thr_f < 0.0 or thr_f > 1.0 or thr_f != thr_f:
        return {"ok": False, "error": f"Operating threshold out of range: {thr}"}
    return {
        "ok": True,
        "threshold": thr_f,
        "model_key": model_key,
        "entry": entry,
        "manifest": doc,
    }


def inference_status(lab_db_path: str, model_key: str = INFERENCE_MODEL_KEY) -> dict[str, Any]:
    """Confidence page Inference panel payload for one model."""
    if model_key not in TARGET_BY_KEY:
        return {
            "status": "not_run",
            "status_display": "Not Run",
            "can_run": False,
            "error": f"Unknown model: {model_key}",
            "model_key": model_key,
            "action_label": "Run Inference",
        }

    cols_map = inference_columns_for(model_key)
    pred_col = cols_map["pred"]
    doc = read_manifest(lab_db_path)
    inf = ((doc.get("inference") or {}).get(model_key) or {}).copy()
    thr_res = resolve_operating_threshold(lab_db_path, model_key)
    entry = (doc.get("models") or {}).get(model_key) or {}
    label = TARGET_BY_KEY[model_key]["label"]
    model_status = str(entry.get("status") or "not_created")

    with ModelLabStore(lab_db_path) as store:
        store.ensure_prediction_schema()
        cols = set(store._prediction_table_columns())
        pred_n = int(store.prediction_row_count() or 0)
        positive = 0
        negative = 0
        nulls = pred_n
        if pred_col in cols and pred_n > 0:
            row = store.conn.execute(
                f"""
                SELECT
                    SUM(CASE WHEN "{pred_col}" = 1 THEN 1 ELSE 0 END) AS pos,
                    SUM(CASE WHEN "{pred_col}" = 0 THEN 1 ELSE 0 END) AS neg,
                    SUM(CASE WHEN "{pred_col}" IS NULL THEN 1 ELSE 0 END) AS nulls
                FROM prediction_dataset
                """
            ).fetchone()
            if row:
                positive = int(row[0] or 0)
                negative = int(row[1] or 0)
                nulls = int(row[2] or 0)

    status = str(inf.get("status") or "not_run")
    scored_n = positive + negative
    if status == "completed" and (nulls > 0 or (pred_n > 0 and scored_n != pred_n)):
        status = "out_of_date"
        inf["stale_reason"] = "Prediction Dataset row count / nulls mismatch"

    can_run = bool(
        thr_res.get("ok")
        and model_status in ("ready", "stale")
        and pred_n > 0
    )

    if status == "out_of_date":
        action_label = f"Update Inference ({label})"
    elif status == "completed":
        action_label = f"Re-run Inference ({label})"
    elif status == "failed":
        action_label = f"Retry Inference ({label})"
    else:
        action_label = f"Run Inference ({label})"

    completed_disp = None
    if inf.get("completed_at"):
        completed_disp = str(inf["completed_at"])[:19].replace("T", " ")

    return {
        "status": status,
        "status_display": {
            "not_run": "Not Run",
            "running": "Running",
            "completed": "Completed",
            "out_of_date": "Out of Date",
            "failed": "Failed",
        }.get(status, status.replace("_", " ").title()),
        "rows": int(
            inf.get("rows")
            or (scored_n if status in ("completed", "out_of_date") else 0)
        ),
        "positive": int(
            inf.get("positive")
            or (positive if status in ("completed", "out_of_date") else 0)
        ),
        "negative": int(
            inf.get("negative")
            or (negative if status in ("completed", "out_of_date") else 0)
        ),
        "nulls": nulls,
        "prediction_rows": pred_n,
        "model_key": model_key,
        "model_label": label,
        "model_status": model_status,
        "model_id": inf.get("model_id") or entry.get("created_at"),
        "threshold": thr_res.get("threshold") if thr_res.get("ok") else inf.get("threshold"),
        "operating_threshold_set": bool(thr_res.get("ok")),
        "operating_threshold_error": thr_res.get("error") if not thr_res.get("ok") else None,
        "active_model_key": doc.get("active_model_key"),
        "completed_at": inf.get("completed_at"),
        "completed_at_display": completed_disp,
        "validation": inf.get("validation"),
        "stale_reason": inf.get("stale_reason"),
        "error": inf.get("error"),
        "can_run": can_run,
        "action_label": action_label,
        "has_predictions": scored_n > 0,
        "pred_column": pred_col,
    }


def clear_confidence_inference(
    lab_db_path: str,
    model_key: str = INFERENCE_MODEL_KEY,
) -> dict[str, Any]:
    """Clear confidence columns for one model and reset its inference status."""
    if model_key not in INFERENCE_COLUMNS:
        return {"ok": False, "error": f"Unknown model key: {model_key}"}
    cols_map = inference_columns_for(model_key)
    with ModelLabStore(lab_db_path) as store:
        store.ensure_prediction_schema()
        cols = set(store._prediction_table_columns())
        if cols_map["pred"] in cols:
            store.conn.execute(
                f"""
                UPDATE prediction_dataset SET
                    "{cols_map['pred']}" = NULL,
                    "{cols_map['model_id']}" = NULL,
                    "{cols_map['threshold']}" = NULL,
                    "{cols_map['created']}" = NULL
                """
            )
            store.conn.commit()

    doc = read_manifest(lab_db_path)
    doc.setdefault("inference", {})[model_key] = _empty_inference_block()
    write_manifest(lab_db_path, doc)
    return {"ok": True, "inference": inference_status(lab_db_path, model_key)}


def run_confidence_inference(
    lab_db_path: str,
    *,
    model_key: str = INFERENCE_MODEL_KEY,
    batch_size: int = DEFAULT_BATCH_SIZE,
    data_dir: str | None = None,
    on_progress: ProgressCb | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """
    Score every Prediction Dataset row with the selected Confidence Model.

    Uses regression selected features + that model's persisted Operating Threshold.
    Writes only final 0/1 decisions into that model's confidence_* columns.
    """
    _ = data_dir
    if model_key not in TARGET_BY_KEY or model_key not in INFERENCE_COLUMNS:
        return {"ok": False, "error": f"Unknown confidence model: {model_key}"}

    cols_map = inference_columns_for(model_key)
    pred_col = cols_map["pred"]
    model_id_col = cols_map["model_id"]
    threshold_col = cols_map["threshold"]
    created_col = cols_map["created"]
    label = TARGET_BY_KEY[model_key]["label"]

    thr_res = resolve_operating_threshold(lab_db_path, model_key)
    if not thr_res.get("ok"):
        return {"ok": False, "error": thr_res.get("error")}
    threshold = float(thr_res["threshold"])
    entry = thr_res["entry"]

    pkg = entry.get("package_dir") or model_package_dir_for(lab_db_path, model_key)
    model_path = entry.get("model_path") or os.path.join(pkg, "model.json")
    if not os.path.isfile(model_path):
        return {"ok": False, "error": f"Model package missing: {model_path}"}

    feat_res = resolve_regression_selected_features(lab_db_path)
    if not feat_res.get("ok"):
        return {"ok": False, "error": feat_res.get("error")}
    features = list(feat_res["features"])
    if not features:
        return {"ok": False, "error": "No regression selected features for inference."}

    metrics_path = os.path.join(pkg, "metrics.json")
    if os.path.isfile(metrics_path):
        try:
            with open(metrics_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            stored_feats = list(meta.get("features") or [])
            if stored_feats:
                features = stored_feats
        except (OSError, json.JSONDecodeError):
            pass

    try:
        model = _load_classifier(model_path, features)
    except Exception as exc:
        return {"ok": False, "error": f"Failed to load classifier: {exc}"}

    model_id = f"{model_key}@{entry.get('created_at') or 'unknown'}"
    created_at = _utc_now()
    batch_size = max(1_000, int(batch_size))

    def _prog(payload: dict[str, Any]) -> None:
        if on_progress:
            try:
                on_progress(payload)
            except Exception:
                pass

    doc = read_manifest(lab_db_path)
    doc.setdefault("inference", {}).setdefault(model_key, _empty_inference_block())
    doc["inference"][model_key].update(
        {
            "status": "running",
            "error": None,
            "stale_reason": None,
            "model_key": model_key,
            "model_id": model_id,
            "threshold": threshold,
        }
    )
    write_manifest(lab_db_path, doc)

    t0 = time.perf_counter()
    processed = 0
    positive = 0
    negative = 0
    last_id = 0
    batch_i = 0

    try:
        with ModelLabStore(lab_db_path) as store:
            store.ensure_prediction_schema()
            total = int(store.prediction_row_count() or 0)
            if total <= 0:
                raise RuntimeError("Prediction Dataset is empty.")

            store.conn.execute(
                f"""
                UPDATE prediction_dataset SET
                    "{pred_col}" = NULL,
                    "{model_id_col}" = NULL,
                    "{threshold_col}" = NULL,
                    "{created_col}" = NULL
                """
            )
            store.conn.commit()

            fs = PredictionFeatureStore.from_store(store)
            available = {n for n, _ in fs.feature_map()}
            missing_feats = [f for f in features if f not in available]
            if missing_feats and len(missing_feats) == len(features):
                raise RuntimeError(
                    "None of the Confidence Model features are available on the "
                    "Prediction Dataset (check embedded/referenced feature storage)."
                )
            use_features = [f for f in features if f in available]
            if not use_features:
                raise RuntimeError("No usable features for Confidence inference.")

            _prog(
                {
                    "phase": "start",
                    "message": (
                        f"{label} inference · {total:,} rows · thr={threshold:.2f}"
                    ),
                    "total": total,
                    "processed": 0,
                    "batch": 0,
                    "threshold": threshold,
                    "model_key": model_key,
                }
            )

            while True:
                if cancel_check and cancel_check():
                    raise RuntimeError("Inference cancelled.")

                batch_i += 1
                rows = fs.fetch_rows(
                    outcome_cols=["id"],
                    feature_names=use_features,
                    where_sql="p.id > ?",
                    where_args=[last_id],
                    limit=batch_size,
                    order_by_sql="p.id ASC",
                )
                if not rows:
                    break

                ids = [int(r["id"]) for r in rows if r.get("id") is not None]
                if not ids:
                    break
                last_id = int(ids[-1])

                X = pd.DataFrame([{f: r.get(f) for f in use_features} for r in rows])
                for f in features:
                    if f not in X.columns:
                        X[f] = np.nan
                X = X[features].apply(pd.to_numeric, errors="coerce")

                y_prob = np.asarray(model.predict(X), dtype=float)
                y_hat = (y_prob >= threshold).astype(int)

                updates = list(
                    zip(
                        y_hat.tolist(),
                        [model_id] * len(ids),
                        [threshold] * len(ids),
                        [created_at] * len(ids),
                        ids,
                    )
                )
                store.conn.executemany(
                    f"""
                    UPDATE prediction_dataset SET
                        "{pred_col}" = ?,
                        "{model_id_col}" = ?,
                        "{threshold_col}" = ?,
                        "{created_col}" = ?
                    WHERE id = ?
                    """,
                    updates,
                )
                store.conn.commit()

                processed += len(ids)
                positive += int(y_hat.sum())
                negative += int(len(y_hat) - int(y_hat.sum()))
                elapsed = max(time.perf_counter() - t0, 1e-6)
                rate = processed / elapsed
                remaining = max(total - processed, 0)
                eta = remaining / rate if rate > 0 else None
                _prog(
                    {
                        "phase": "batch",
                        "message": (
                            f"{label} · batch {batch_i} · {processed:,}/{total:,} "
                            f"({rate:,.0f} rows/s)"
                        ),
                        "total": total,
                        "processed": processed,
                        "positive": positive,
                        "negative": negative,
                        "batch": batch_i,
                        "rows_per_sec": rate,
                        "eta_sec": eta,
                        "threshold": threshold,
                        "model_key": model_key,
                    }
                )

            row = store.conn.execute(
                f"""
                SELECT
                    COUNT(*) AS n,
                    SUM(CASE WHEN "{pred_col}" IS NOT NULL THEN 1 ELSE 0 END) AS updated,
                    SUM(CASE WHEN "{pred_col}" IS NULL THEN 1 ELSE 0 END) AS nulls,
                    SUM(CASE WHEN "{pred_col}" = 1 THEN 1 ELSE 0 END) AS pos,
                    SUM(CASE WHEN "{pred_col}" = 0 THEN 1 ELSE 0 END) AS neg
                FROM prediction_dataset
                """
            ).fetchone()
            pred_rows = int(row[0] or 0)
            updated = int(row[1] or 0)
            nulls = int(row[2] or 0)
            pos = int(row[3] or 0)
            neg = int(row[4] or 0)
            passed = pred_rows == updated and nulls == 0 and pred_rows > 0
            validation = {
                "prediction_rows": pred_rows,
                "rows_updated": updated,
                "null_predictions": nulls,
                "positive_predictions": pos,
                "negative_predictions": neg,
                "passed": passed,
                "threshold": threshold,
                "model_id": model_id,
                "model_key": model_key,
            }

        completed_at = _utc_now()
        doc = read_manifest(lab_db_path)
        doc.setdefault("inference", {})[model_key] = {
            "status": "completed" if passed else "failed",
            "rows": updated,
            "positive": pos,
            "negative": neg,
            "nulls": nulls,
            "model_key": model_key,
            "model_id": model_id,
            "threshold": threshold,
            "completed_at": completed_at,
            "validation": validation,
            "error": None if passed else "Validation failed — null or incomplete predictions.",
            "stale_reason": None,
        }
        write_manifest(lab_db_path, doc)

        _prog(
            {
                "phase": "done",
                "message": (
                    f"{label} complete · {updated:,} rows · "
                    f"+{pos:,} / −{neg:,} · thr={threshold:.2f}"
                ),
                "total": pred_rows,
                "processed": updated,
                "positive": pos,
                "negative": neg,
                "validation": validation,
                "model_key": model_key,
            }
        )
        return {
            "ok": passed,
            "validation": validation,
            "threshold": threshold,
            "model_id": model_id,
            "model_key": model_key,
            "inference": inference_status(lab_db_path, model_key),
            "error": None if passed else validation,
        }
    except Exception as exc:
        doc = read_manifest(lab_db_path)
        doc.setdefault("inference", {}).setdefault(model_key, _empty_inference_block())
        doc["inference"][model_key].update(
            {
                "status": "failed",
                "error": str(exc),
            }
        )
        write_manifest(lab_db_path, doc)
        _prog({"phase": "error", "message": str(exc), "model_key": model_key})
        return {
            "ok": False,
            "error": str(exc),
            "inference": inference_status(lab_db_path, model_key),
        }


def mark_stale_on_prediction_rebuild(lab_db_path: str) -> None:
    """Call when Prediction Dataset is rebuilt — all model inferences Out of Date."""
    mark_inference_out_of_date(
        lab_db_path, reason="Prediction Dataset rebuilt", model_key="*"
    )


def target_hit_filter_available(lab_db_path: str) -> dict[str, Any]:
    """Whether Research Dashboard may enable Target Hit confidence filter."""
    return confidence_filter_available(lab_db_path, "target_hit")


def confidence_filter_available(
    lab_db_path: str, model_key: str = "target_hit"
) -> dict[str, Any]:
    """Whether Research Dashboard may filter on a Confidence Classifier's predictions."""
    from .confidence_manifest import TARGET_BY_KEY

    key = str(model_key or "target_hit").strip().lower()
    label = (TARGET_BY_KEY.get(key) or {}).get("label") or key
    if key not in INFERENCE_COLUMNS:
        return {
            "available": False,
            "model_key": key,
            "label": label,
            "reason": f"Unknown confidence classifier: {key}",
        }
    st = inference_status(lab_db_path, key)
    # Unlock when Prediction Dataset has 0/1 scores for this classifier
    # (Operating Threshold alone is not enough).
    available = bool(st.get("has_predictions"))
    if available:
        reason = None
    elif not st.get("operating_threshold_set"):
        reason = (
            f"{label}: save an Operating Threshold, then Run Inference "
            f"on the Confidence Model tab (relaunch not required)."
        )
    else:
        reason = (
            f"{label}: Operating Threshold is saved, but Inference has not been run. "
            f"Open Confidence Model → select {label} → "
            f"{st.get('action_label') or 'Run Inference'}. "
            f"Reloading the app is not required."
        )
    return {
        "available": available,
        "model_key": key,
        "label": label,
        "status": st.get("status"),
        "status_display": st.get("status_display"),
        "operating_threshold_set": bool(st.get("operating_threshold_set")),
        "reason": reason,
        "inference": st,
    }


def dashboard_confidence_filter_options(lab_db_path: str) -> dict[str, Any]:
    """Availability map for every Confidence Classifier on the Research Dashboard."""
    from .confidence_manifest import CONFIDENCE_TARGETS

    classifiers = []
    for spec in CONFIDENCE_TARGETS:
        gate = confidence_filter_available(lab_db_path, spec["key"])
        classifiers.append(
            {
                "key": spec["key"],
                "label": spec["label"],
                "available": bool(gate.get("available")),
                "reason": gate.get("reason"),
                "status": gate.get("status"),
            }
        )
    return {"classifiers": classifiers}


def _percentile(sorted_vals: np.ndarray, q: float) -> float | None:
    if sorted_vals.size == 0:
        return None
    return float(np.percentile(sorted_vals, q))


def score_probability_distribution(
    lab_db_path: str,
    model_key: str,
    *,
    batch_size: int = 50_000,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """
    Rescore Prediction Dataset with the saved classifier (pre-threshold probs).

    Returns probability stats + positives at the saved Operating Threshold, and
    compares Evaluate/manifest threshold vs threshold written during inference.
    """
    if model_key not in TARGET_BY_KEY or model_key not in INFERENCE_COLUMNS:
        return {"ok": False, "error": f"Unknown confidence model: {model_key}"}

    label = TARGET_BY_KEY[model_key]["label"]
    cols_map = inference_columns_for(model_key)
    pred_col = cols_map["pred"]
    threshold_col = cols_map["threshold"]

    thr_res = resolve_operating_threshold(lab_db_path, model_key)
    evaluate_thr = float(thr_res["threshold"]) if thr_res.get("ok") else None
    entry = (thr_res.get("entry") if thr_res.get("ok") else None) or {}
    if not entry:
        doc = read_manifest(lab_db_path)
        entry = (doc.get("models") or {}).get(model_key) or {}

    status = str(entry.get("status") or "not_created")
    if status not in ("ready", "stale"):
        return {
            "ok": False,
            "model_key": model_key,
            "label": label,
            "error": f"{label} is not trained.",
            "model_status": status,
        }

    pkg = entry.get("package_dir") or model_package_dir_for(lab_db_path, model_key)
    model_path = entry.get("model_path") or os.path.join(pkg, "model.json")
    if not os.path.isfile(model_path):
        return {
            "ok": False,
            "model_key": model_key,
            "label": label,
            "error": f"Model package missing: {model_path}",
        }

    metrics_thr = None
    metrics_path = os.path.join(pkg, "metrics.json")
    if os.path.isfile(metrics_path):
        try:
            with open(metrics_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            if isinstance(meta, dict) and meta.get("operating_threshold") is not None:
                metrics_thr = float(meta["operating_threshold"])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            metrics_thr = None

    doc = read_manifest(lab_db_path)
    inference_thr = ((doc.get("inference") or {}).get(model_key) or {}).get("threshold")
    try:
        inference_thr_f = float(inference_thr) if inference_thr is not None else None
    except (TypeError, ValueError):
        inference_thr_f = None

    feat_res = resolve_regression_selected_features(lab_db_path)
    if not feat_res.get("ok"):
        return {"ok": False, "model_key": model_key, "label": label, "error": feat_res.get("error")}
    features = list(feat_res["features"] or [])
    if os.path.isfile(metrics_path):
        try:
            with open(metrics_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            stored = list((meta or {}).get("features") or [])
            if stored:
                features = stored
        except (OSError, json.JSONDecodeError):
            pass
    if not features:
        return {"ok": False, "model_key": model_key, "label": label, "error": "No features."}

    try:
        model = _load_classifier(model_path, features)
    except Exception as exc:
        return {"ok": False, "model_key": model_key, "label": label, "error": str(exc)}

    stored_thr_values: list[float] = []
    pred_pos = pred_neg = pred_null = 0
    probs: list[float] = []
    last_id = 0
    scored = 0

    with ModelLabStore(lab_db_path) as store:
        store.ensure_prediction_schema()
        table_cols = set(store._prediction_table_columns())
        total = int(store.prediction_row_count() or 0)
        if pred_col in table_cols and total > 0:
            row = store.conn.execute(
                f"""
                SELECT
                    SUM(CASE WHEN "{pred_col}" = 1 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN "{pred_col}" = 0 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN "{pred_col}" IS NULL THEN 1 ELSE 0 END)
                FROM prediction_dataset
                """
            ).fetchone()
            pred_pos = int(row[0] or 0)
            pred_neg = int(row[1] or 0)
            pred_null = int(row[2] or 0)
        if threshold_col in table_cols and total > 0:
            for (v,) in store.conn.execute(
                f"""
                SELECT DISTINCT "{threshold_col}"
                FROM prediction_dataset
                WHERE "{threshold_col}" IS NOT NULL
                LIMIT 20
                """
            ):
                try:
                    stored_thr_values.append(float(v))
                except (TypeError, ValueError):
                    pass

        fs = PredictionFeatureStore.from_store(store)
        available = {n for n, _ in fs.feature_map()}
        use_features = [f for f in features if f in available]
        if not use_features:
            return {
                "ok": False,
                "model_key": model_key,
                "label": label,
                "error": "No usable features on Prediction Dataset.",
            }

        limit_total = int(max_rows) if max_rows else total
        while scored < limit_total:
            take = min(batch_size, limit_total - scored)
            rows = fs.fetch_rows(
                outcome_cols=["id"],
                feature_names=use_features,
                where_sql="p.id > ?",
                where_args=[last_id],
                limit=take,
                order_by_sql="p.id ASC",
            )
            if not rows:
                break
            ids = [int(r["id"]) for r in rows if r.get("id") is not None]
            if not ids:
                break
            last_id = int(ids[-1])
            X = pd.DataFrame([{f: r.get(f) for f in use_features} for r in rows])
            for f in features:
                if f not in X.columns:
                    X[f] = np.nan
            X = X[features].apply(pd.to_numeric, errors="coerce")
            y_prob = np.asarray(model.predict(X), dtype=float)
            probs.extend(float(p) for p in y_prob.tolist())
            scored += len(ids)

    arr = np.asarray(probs, dtype=float)
    arr = arr[np.isfinite(arr)]
    thr_for_pos = evaluate_thr if evaluate_thr is not None else inference_thr_f
    pos_at_thr = int((arr >= thr_for_pos).sum()) if thr_for_pos is not None and arr.size else None
    sorted_arr = np.sort(arr) if arr.size else arr

    stored_thr_unique = sorted(set(round(v, 6) for v in stored_thr_values))
    inference_written_thr = stored_thr_unique[0] if len(stored_thr_unique) == 1 else None
    thr_match = None
    if evaluate_thr is not None and inference_written_thr is not None:
        thr_match = abs(evaluate_thr - inference_written_thr) < 1e-6
    elif evaluate_thr is not None and inference_thr_f is not None:
        thr_match = abs(evaluate_thr - inference_thr_f) < 1e-6

    return {
        "ok": True,
        "model_key": model_key,
        "label": label,
        "model_status": status,
        "rows_scored": int(arr.size),
        "probability": {
            "min": float(arr.min()) if arr.size else None,
            "max": float(arr.max()) if arr.size else None,
            "mean": float(arr.mean()) if arr.size else None,
            "median": float(np.median(arr)) if arr.size else None,
            "p10": _percentile(sorted_arr, 10),
            "p25": _percentile(sorted_arr, 25),
            "p50": _percentile(sorted_arr, 50),
            "p75": _percentile(sorted_arr, 75),
            "p90": _percentile(sorted_arr, 90),
            "p95": _percentile(sorted_arr, 95),
            "p99": _percentile(sorted_arr, 99),
        },
        "operating_threshold_evaluate": evaluate_thr,
        "operating_threshold_metrics_json": metrics_thr,
        "operating_threshold_inference_meta": inference_thr_f,
        "operating_threshold_written_on_rows": (
            inference_written_thr
            if inference_written_thr is not None
            else (stored_thr_unique or None)
        ),
        "threshold_match": thr_match,
        "positives_at_operating_threshold": pos_at_thr,
        "positive_rate_at_threshold": (
            round(100.0 * pos_at_thr / arr.size, 4) if pos_at_thr is not None and arr.size else None
        ),
        "stored_pred_counts": {
            "pred_1": pred_pos,
            "pred_0": pred_neg,
            "null": pred_null,
        },
    }


def audit_probability_distributions(
    lab_db_path: str,
    *,
    max_rows: int | None = None,
) -> dict[str, Any]:
    """Score every trained Confidence Model; return + printable summary."""
    from .confidence_manifest import CONFIDENCE_TARGETS

    results: list[dict[str, Any]] = []
    for spec in CONFIDENCE_TARGETS:
        results.append(
            score_probability_distribution(
                lab_db_path, spec["key"], max_rows=max_rows
            )
        )
    return {"ok": True, "lab_db_path": lab_db_path, "models": results}


def format_probability_distribution_report(payload: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("Confidence probability distributions (pre-threshold)")
    lines.append("=" * 72)
    for m in payload.get("models") or []:
        label = m.get("label") or m.get("model_key")
        lines.append("")
        lines.append(f"## {label} ({m.get('model_key')})")
        if not m.get("ok"):
            lines.append(f"  SKIP: {m.get('error')}")
            continue
        p = m.get("probability") or {}
        lines.append(f"  Rows scored:     {int(m.get('rows_scored') or 0):,}")
        lines.append(
            f"  Prob min/max:    {p.get('min'):.4f} / {p.get('max'):.4f}"
            if p.get("min") is not None
            else "  Prob min/max:    —"
        )
        lines.append(
            f"  Prob mean/med:   {p.get('mean'):.4f} / {p.get('median'):.4f}"
            if p.get("mean") is not None
            else "  Prob mean/med:   —"
        )
        lines.append(
            "  Percentiles:     "
            f"p10={p.get('p10'):.4f}  p25={p.get('p25'):.4f}  "
            f"p50={p.get('p50'):.4f}  p75={p.get('p75'):.4f}  "
            f"p90={p.get('p90'):.4f}  p95={p.get('p95'):.4f}  "
            f"p99={p.get('p99'):.4f}"
            if p.get("p10") is not None
            else "  Percentiles:     —"
        )
        thr = m.get("operating_threshold_evaluate")
        pos = m.get("positives_at_operating_threshold")
        rate = m.get("positive_rate_at_threshold")
        lines.append(
            f"  Operating thr:   {thr:.4f}" if thr is not None else "  Operating thr:   (not set)"
        )
        if pos is not None and rate is not None:
            lines.append(f"  Positives@thr:   {pos:,}  ({rate:.2f}%)")
        stored = m.get("stored_pred_counts") or {}
        lines.append(
            f"  Stored pred 1/0/null: "
            f"{int(stored.get('pred_1') or 0):,} / "
            f"{int(stored.get('pred_0') or 0):,} / "
            f"{int(stored.get('null') or 0):,}"
        )
        written = m.get("operating_threshold_written_on_rows")
        match = m.get("threshold_match")
        lines.append(f"  Evaluate thr:    {thr}")
        lines.append(f"  metrics.json:    {m.get('operating_threshold_metrics_json')}")
        lines.append(f"  inference meta:  {m.get('operating_threshold_inference_meta')}")
        lines.append(f"  written on rows: {written}")
        if match is True:
            lines.append("  Threshold match: YES (Evaluate == inference)")
        elif match is False:
            lines.append("  Threshold match: NO — Evaluate vs written/inference differ")
        else:
            lines.append("  Threshold match: n/a (missing Evaluate thr or no inference yet)")
    return "\n".join(lines)
