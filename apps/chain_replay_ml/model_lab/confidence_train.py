"""Train / evaluate lab-local Confidence Models (binary XGB)."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd

from chain_replay_ml.training.evaluator import (
    DEFAULT_THRESHOLD_SWEEP,
    evaluate_classification,
    threshold_analysis,
)
from chain_replay_ml.training.xgb_trainer import train_xgb_binary_classifier

from .confidence_dataset import confidence_dataset_paths, resolve_regression_selected_features
from .confidence_manifest import (
    COLUMN_BY_KEY,
    TARGET_BY_KEY,
    model_package_dir_for,
    read_manifest,
    write_manifest,
)
from .service import load_lab


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)!r} is not JSON serializable")


def model_has_threshold_analysis(
    lab_db_path: str,
    model_key: str,
    *,
    entry: dict[str, Any] | None = None,
) -> bool:
    """
    True when the model package can populate Threshold Analysis
    (stored sweep and/or eval_predictions.npz for recompute).
    """
    if model_key not in TARGET_BY_KEY:
        return False
    if entry is None:
        entry = (read_manifest(lab_db_path).get("models") or {}).get(model_key) or {}
    ta = entry.get("threshold_analysis")
    if isinstance(ta, list) and len(ta) >= len(DEFAULT_THRESHOLD_SWEEP):
        return True
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    nested = metrics.get("threshold_analysis") if isinstance(metrics, dict) else None
    if isinstance(nested, list) and len(nested) >= len(DEFAULT_THRESHOLD_SWEEP):
        return True

    pkg = entry.get("package_dir") or model_package_dir_for(lab_db_path, model_key)
    if not pkg:
        return False
    pred_path = os.path.join(pkg, "eval_predictions.npz")
    if os.path.isfile(pred_path):
        return True
    metrics_path = os.path.join(pkg, "metrics.json")
    if os.path.isfile(metrics_path):
        try:
            with open(metrics_path, encoding="utf-8") as fh:
                stored = json.load(fh)
            if isinstance(stored, dict):
                top = stored.get("threshold_analysis")
                if isinstance(top, list) and len(top) >= len(DEFAULT_THRESHOLD_SWEEP):
                    return True
                inner = (stored.get("metrics") or {}).get("threshold_analysis")
                if isinstance(inner, list) and len(inner) >= len(DEFAULT_THRESHOLD_SWEEP):
                    return True
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    return False


def _time_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> dict[str, Any]:
    n = len(X)
    if n < 30:
        raise ValueError(f"Need at least 30 rows to train (found {n}).")
    i_train = max(1, int(n * train_frac))
    i_val = max(i_train + 1, int(n * (train_frac + val_frac)))
    if i_val >= n:
        i_val = n - 1
    if i_train >= i_val:
        i_train = max(1, i_val - 1)
    return {
        "train_X": X.iloc[:i_train],
        "train_y": y.iloc[:i_train],
        "val_X": X.iloc[i_train:i_val],
        "val_y": y.iloc[i_train:i_val],
        "test_X": X.iloc[i_val:],
        "test_y": y.iloc[i_val:],
    }


def train_confidence_model(
    lab_db_path: str,
    model_key: str,
    *,
    parameters: dict[str, Any] | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Train one binary confidence classifier; store under the Research Lab package."""
    if model_key not in TARGET_BY_KEY:
        return {"ok": False, "error": f"Unknown target: {model_key}"}

    paths = confidence_dataset_paths(lab_db_path)
    parquet = paths["parquet"]
    meta_path = paths["json"]
    if not os.path.isfile(parquet) or not os.path.isfile(meta_path):
        return {
            "ok": False,
            "error": "Confidence Dataset not created. Run Build Confidence Dataset first.",
        }

    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    target_col = COLUMN_BY_KEY[model_key]

    # Always inherit regression selected features (never full export matrix)
    lab = load_lab(lab_db_path)
    feature_resolve = resolve_regression_selected_features(lab_db_path, lab=lab)
    if feature_resolve.get("ok"):
        features = list(feature_resolve["features"])
        feature_source = "regression_model"
    else:
        # Legacy confidence datasets: accept meta only if tagged as regression_model
        if str(meta.get("feature_source") or "") == "regression_model":
            features = list(meta.get("feature_columns") or meta.get("selected_features") or [])
            feature_source = "confidence_dataset_meta"
        else:
            return {
                "ok": False,
                "error": (
                    str(feature_resolve.get("error")
                        or "Cannot resolve regression selected features.")
                    + " Rebuild Confidence Dataset after the lab has a selected feature snapshot."
                ),
            }
    if not features:
        return {"ok": False, "error": "No regression selected features available for training."}

    if on_progress:
        on_progress(
            {
                "phase": "loading",
                "message": (
                    f"Loading Confidence Dataset · {len(features)} regression features…"
                ),
            }
        )

    df = pd.read_parquet(parquet)
    features = [f for f in features if f in df.columns]
    if not features:
        return {
            "ok": False,
            "error": "Regression selected features are not present in the Confidence Dataset parquet.",
        }
    if target_col not in df.columns:
        return {"ok": False, "error": f"Target column missing: {target_col}"}

    # Stable row order for time split
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    elif "trading_day" in df.columns:
        df = df.sort_values("trading_day", kind="mergesort").reset_index(drop=True)

    y = pd.to_numeric(df[target_col], errors="coerce")
    mask = y.notna()
    df = df.loc[mask].reset_index(drop=True)
    y = y.loc[mask].astype(int).reset_index(drop=True)
    X = df[features].apply(pd.to_numeric, errors="coerce")

    # Drop rows with all-NaN features
    row_ok = X.notna().any(axis=1)
    X = X.loc[row_ok].reset_index(drop=True)
    y = y.loc[row_ok].reset_index(drop=True)
    if len(X) < 30:
        return {"ok": False, "error": f"Need at least 30 labeled rows (found {len(X)})."}
    if y.nunique() < 2:
        return {"ok": False, "error": f"Target {target_col} has only one class."}

    parts = _time_split(X, y)
    params = {
        "n_estimators": 400,
        "early_stopping_rounds": 40,
        "learning_rate": 0.05,
        "max_depth": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "prediction_type": "binary",
        **(parameters or {}),
    }
    params["prediction_type"] = "binary"
    # Prefer GPU via shared factory (xgb_device defaults to cuda unless overridden).
    params.setdefault(
        "xgb_device",
        str(os.environ.get("XGB_TRAIN_DEVICE") or os.environ.get("ML_TRAIN_DEVICE") or "cuda")
        .strip()
        .lower(),
    )

    if on_progress:
        on_progress({"phase": "training", "message": f"Training {TARGET_BY_KEY[model_key]['label']}…"})

    try:
        result = train_xgb_binary_classifier(
            train_X=parts["train_X"],
            train_y=parts["train_y"],
            val_X=parts["val_X"],
            val_y=parts["val_y"],
            features=features,
            parameters=params,
            cancel_check=cancel_check,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    model = result.get("model") or result.get("wrapper")
    if model is None:
        # train_xgb returns booster wrapper under various keys
        model = result.get("booster_wrapper") or result.get("regressor")
    if model is None:
        for v in result.values():
            if hasattr(v, "predict") and hasattr(v, "save_model"):
                model = v
                break
    if model is None:
        return {"ok": False, "error": "Trainer did not return a model object.", "raw_keys": list(result.keys())}

    test_X = parts["test_X"]
    test_y = parts["test_y"]
    if len(test_X) == 0:
        test_X, test_y = parts["val_X"], parts["val_y"]

    y_prob = np.asarray(model.predict(test_X), dtype=float)
    metrics = evaluate_classification(test_y, y_prob)
    from .confidence import compute_calibration_bins

    calibration = compute_calibration_bins(test_y.tolist(), y_prob.tolist())

    # Always compute + persist Threshold Analysis for every Confidence Model
    # (Target Hit, RR 1:1, RR 2:3, …) — never optional.
    thr_analysis = threshold_analysis(test_y, y_prob)
    if not thr_analysis or len(thr_analysis) < len(DEFAULT_THRESHOLD_SWEEP):
        return {
            "ok": False,
            "error": (
                f"Threshold Analysis failed for {TARGET_BY_KEY[model_key]['label']} "
                f"(got {len(thr_analysis) if thr_analysis else 0} rows, "
                f"expected {len(DEFAULT_THRESHOLD_SWEEP)}). "
                "Cannot save model without Threshold Analysis."
            ),
        }
    metrics["threshold_analysis"] = thr_analysis

    pkg = model_package_dir_for(lab_db_path, model_key)
    os.makedirs(pkg, exist_ok=True)
    model_path = os.path.join(pkg, "model.json")
    model.save_model(model_path)

    # Persist test labels + probabilities for Threshold Analysis on Evaluate
    pred_path = os.path.join(pkg, "eval_predictions.npz")
    try:
        yt_arr = np.asarray(test_y, dtype=np.int8)
        yp_arr = np.asarray(y_prob, dtype=np.float32)
        if len(yt_arr) == 0 or len(yp_arr) == 0:
            raise RuntimeError("Empty eval predictions — cannot build Threshold Analysis.")
        np.savez_compressed(pred_path, y_true=yt_arr, y_prob=yp_arr)
    except Exception as exc:
        return {
            "ok": False,
            "error": (
                f"Failed to save eval_predictions.npz for Threshold Analysis: {exc}"
            ),
        }

    created_at = _utc_now()
    meta_out = {
        "model_key": model_key,
        "label": TARGET_BY_KEY[model_key]["label"],
        "column": target_col,
        "features": features,
        "feature_count": len(features),
        "feature_source": feature_source,
        "feature_source_label": "Regression Model",
        "metrics": metrics,
        "calibration": calibration,
        "threshold_analysis": thr_analysis,
        "threshold_sweep": list(DEFAULT_THRESHOLD_SWEEP),
        "created_at": created_at,
        "train_rows": int(len(parts["train_X"])),
        "val_rows": int(len(parts["val_X"])),
        "test_rows": int(len(test_X)),
        "parameters": {k: v for k, v in params.items() if k != "prediction_type"},
        "lab_db_path": lab_db_path,
        "eval_predictions_path": pred_path,
        "has_threshold_analysis": True,
    }
    with open(os.path.join(pkg, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(meta_out, fh, indent=2, default=_json_default)
        fh.write("\n")

    # Feature importance
    try:
        from chain_replay_ml.training.xgb_trainer import feature_importance_df

        fi = feature_importance_df(model, features)
        fi.to_csv(os.path.join(pkg, "feature_importance.csv"), index=False)
        meta_out["feature_importance"] = fi.head(40).to_dict(orient="records")
    except Exception:
        meta_out["feature_importance"] = []

    doc = read_manifest(lab_db_path)
    prev_entry = (doc.get("models") or {}).get(model_key) or {}
    # Clear previous active flags if linking this as only ready? keep previous active unless none
    entry = {
        "key": model_key,
        "label": TARGET_BY_KEY[model_key]["label"],
        "column": target_col,
        "status": "ready",
        "metrics": metrics,
        "calibration": calibration,
        "created_at": created_at,
        "package_dir": pkg,
        "model_path": model_path,
        "active": doc.get("active_model_key") == model_key
        or (doc.get("active_model_key") is None),
        "feature_importance": meta_out.get("feature_importance") or [],
        "feature_count": len(features),
        "feature_source": feature_source,
        "threshold_analysis": thr_analysis,
        "has_threshold_analysis": True,
        "train_rows": meta_out["train_rows"],
        "test_rows": meta_out["test_rows"],
        # Preserve Operating Threshold across retrain (still marks inference Out of Date)
        "operating_threshold": prev_entry.get("operating_threshold"),
    }
    if entry["active"]:
        for k, m in (doc.get("models") or {}).items():
            m["active"] = k == model_key
        doc["active_model_key"] = model_key
    doc.setdefault("models", {})[model_key] = entry
    write_manifest(lab_db_path, doc)

    if on_progress:
        on_progress({"phase": "done", "message": "Training complete"})

    # Retrain invalidates persisted Prediction Dataset scores for this model
    from .confidence_manifest import mark_inference_out_of_date

    mark_inference_out_of_date(
        lab_db_path, reason="Confidence Model retrained", model_key=model_key
    )

    return {
        "ok": True,
        "model_key": model_key,
        "label": TARGET_BY_KEY[model_key]["label"],
        "metrics": metrics,
        "calibration": calibration,
        "threshold_analysis": thr_analysis,
        "has_threshold_analysis": True,
        "features": features,
        "feature_count": len(features),
        "feature_source": feature_source,
        "package_dir": pkg,
        "manifest": read_manifest(lab_db_path),
        "operating_threshold": entry.get("operating_threshold"),
    }


def evaluate_confidence_model(lab_db_path: str, model_key: str) -> dict[str, Any]:
    """Return stored evaluation payload for a trained confidence model."""
    if model_key not in TARGET_BY_KEY:
        return {"ok": False, "error": f"Unknown target: {model_key}"}
    doc = read_manifest(lab_db_path)
    entry = (doc.get("models") or {}).get(model_key) or {}
    if str(entry.get("status") or "") not in ("ready", "stale"):
        return {"ok": False, "error": f"{TARGET_BY_KEY[model_key]['label']} is not trained."}

    metrics = entry.get("metrics") or {}
    pkg = entry.get("package_dir") or model_package_dir_for(lab_db_path, model_key)
    metrics_path = os.path.join(pkg, "metrics.json") if pkg else None
    detail = dict(entry)
    if metrics_path and os.path.isfile(metrics_path):
        try:
            with open(metrics_path, encoding="utf-8") as fh:
                stored = json.load(fh)
            if isinstance(stored, dict):
                detail.update(stored)
                metrics = stored.get("metrics") or metrics
        except (OSError, json.JSONDecodeError):
            pass

    confusion = (metrics or {}).get("confusion") or {}
    tp = int(confusion.get("tp") or 0)
    fp = int(confusion.get("fp") or 0)
    tn = int(confusion.get("tn") or 0)
    fn = int(confusion.get("fn") or 0)
    total = tp + fp + tn + fn
    actual_pos = metrics.get("actual_positives")
    actual_neg = metrics.get("actual_negatives")
    pred_pos = metrics.get("predicted_positives")
    pred_neg = metrics.get("predicted_negatives")
    if actual_pos is None:
        actual_pos = tp + fn
    if actual_neg is None:
        actual_neg = tn + fp
    if pred_pos is None:
        pred_pos = tp + fp
    if pred_neg is None:
        pred_neg = tn + fn
    pos_rate = metrics.get("positive_rate_pct")
    if pos_rate is None and total > 0:
        pos_rate = round(100.0 * float(actual_pos) / float(total), 2)

    from chain_replay_ml.training.evaluator import (
        threshold_analysis,
        trading_filter_summary_from_confusion,
    )

    thr_analysis = (
        detail.get("threshold_analysis")
        or metrics.get("threshold_analysis")
        or entry.get("threshold_analysis")
        or []
    )
    pred_file = None
    if pkg:
        cand = os.path.join(pkg, "eval_predictions.npz")
        if os.path.isfile(cand):
            pred_file = cand
        elif detail.get("eval_predictions_path") and os.path.isfile(
            str(detail.get("eval_predictions_path"))
        ):
            pred_file = str(detail.get("eval_predictions_path"))
    if pred_file:
        try:
            loaded = np.load(pred_file)
            thr_analysis = threshold_analysis(loaded["y_true"], loaded["y_prob"])
        except Exception:
            pass

    # Enrich legacy threshold rows missing trading-filter fields
    enriched_ta: list[dict[str, Any]] = []
    for row in thr_analysis or []:
        r = dict(row)
        if r.get("good_trades_kept_pct") is None and r.get("recall_pct") is not None:
            r["good_trades_kept_pct"] = r.get("recall_pct")
        if r.get("bad_trades_filtered_pct") is None:
            tp_r = int(r.get("tp") or 0)
            fp_r = int(r.get("fp") or 0)
            tn_r = int(r.get("tn") or 0)
            fn_r = int(r.get("fn") or 0)
            filt = trading_filter_summary_from_confusion(
                tp=tp_r, fp=fp_r, tn=tn_r, fn=fn_r, threshold=float(r.get("threshold") or 0.5)
            )
            for k, v in filt.items():
                if k != "threshold" and r.get(k) is None:
                    r[k] = v
        if r.get("good_trades_filtered_pct") is None and r.get("good_trades_kept_pct") is not None:
            try:
                r["good_trades_filtered_pct"] = round(
                    100.0 - float(r["good_trades_kept_pct"]), 2
                )
            except (TypeError, ValueError):
                pass
        if r.get("bad_trades_passed_pct") is None and r.get("bad_trades_filtered_pct") is not None:
            try:
                r["bad_trades_passed_pct"] = round(
                    100.0 - float(r["bad_trades_filtered_pct"]), 2
                )
            except (TypeError, ValueError):
                pass
        enriched_ta.append(r)
    thr_analysis = enriched_ta

    # Legacy packages (trained before Threshold Analysis) have no sweep and no npz.
    is_legacy = not bool(thr_analysis)
    has_ta = not is_legacy

    default_thr = float(metrics.get("threshold") or 0.5)
    trading_summary = trading_filter_summary_from_confusion(
        tp=tp, fp=fp, tn=tn, fn=fn, threshold=default_thr
    )
    # Prefer matching sweep row when available (same source as Threshold Analysis)
    for row in thr_analysis:
        try:
            if abs(float(row.get("threshold")) - default_thr) < 1e-9:
                trading_summary = {
                    "threshold": round(default_thr, 2),
                    "good_trades_kept_pct": row.get("good_trades_kept_pct"),
                    "good_trades_filtered_pct": row.get("good_trades_filtered_pct"),
                    "bad_trades_filtered_pct": row.get("bad_trades_filtered_pct"),
                    "bad_trades_passed_pct": row.get("bad_trades_passed_pct"),
                }
                break
        except (TypeError, ValueError):
            continue

    evaluation = {
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "core_metrics": {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision_pct": metrics.get("precision_pct"),
            "recall_pct": metrics.get("recall_pct"),
            "f1_pct": metrics.get("f1_pct"),
            "accuracy_pct": metrics.get("accuracy_pct"),
            "roc_auc": metrics.get("roc_auc"),
            "pr_auc": metrics.get("pr_auc"),
            "brier_score": metrics.get("brier_score"),
        },
        "class_distribution": {
            "total_rows": metrics.get("n_samples") or total,
            "actual_positives": int(actual_pos),
            "actual_negatives": int(actual_neg),
            "predicted_positives": int(pred_pos),
            "predicted_negatives": int(pred_neg),
            "positive_rate_pct": pos_rate,
        },
        "threshold": {
            "classification_threshold": default_thr,
            "mean_prob_actual_positive": metrics.get("mean_prob_actual_positive"),
            "mean_prob_actual_negative": metrics.get("mean_prob_actual_negative"),
            "note": (
                f"Headline metrics use threshold {default_thr:.2f} "
                f"(Predict Positive if P ≥ {default_thr:.2f}). "
                "With a high base rate (~86% hits), 0.50 often predicts almost all positives — "
                "compare Trading Filter Summary and Threshold Analysis below."
            ),
        },
        "trading_filter_summary": trading_summary,
        "threshold_analysis": thr_analysis,
    }

    return {
        "ok": True,
        "model_key": model_key,
        "label": TARGET_BY_KEY[model_key]["label"],
        "status": entry.get("status"),
        "status_display": "Legacy" if is_legacy else None,
        "is_legacy": is_legacy,
        "has_threshold_analysis": has_ta,
        "legacy_message": (
            "This Confidence Model is Legacy: Threshold Analysis was not saved "
            "when it was trained. Retrain to generate the 0.50–0.90 sweep, then "
            "select and Save Operating Threshold."
            if is_legacy
            else None
        ),
        "metrics": metrics,
        "evaluation": evaluation,
        "threshold_analysis": thr_analysis,
        "calibration": detail.get("calibration") or entry.get("calibration") or [],
        "confusion": confusion,
        "feature_importance": detail.get("feature_importance") or entry.get("feature_importance") or [],
        "created_at": entry.get("created_at"),
        "operating_threshold": entry.get("operating_threshold")
        if entry.get("operating_threshold") is not None
        else detail.get("operating_threshold"),
        "test_rows": detail.get("test_rows") or entry.get("test_rows"),
        "package_dir": pkg,
    }
