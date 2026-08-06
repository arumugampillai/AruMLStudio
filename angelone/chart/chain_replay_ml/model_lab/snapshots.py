"""Immutable Phase-1 snapshots + artifact pointers for Model Lab."""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

from chain_replay_ml.training.paths import model_artifact_paths, model_package_dir


def _meta(doc: dict[str, Any]) -> dict[str, Any]:
    return doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}


def _cfg(doc: dict[str, Any]) -> dict[str, Any]:
    return doc.get("config") if isinstance(doc.get("config"), dict) else {}


def _metrics(doc: dict[str, Any]) -> dict[str, Any]:
    return doc.get("metrics") if isinstance(doc.get("metrics"), dict) else {}


def _wf(doc: dict[str, Any]) -> dict[str, Any]:
    return doc.get("walk_forward") if isinstance(doc.get("walk_forward"), dict) else {}


def _summary(doc: dict[str, Any]) -> dict[str, Any]:
    return doc.get("training_summary") if isinstance(doc.get("training_summary"), dict) else {}


def sha256_file(path: str, *, chunk_size: int = 1024 * 1024) -> str | None:
    if not path or not os.path.isfile(path):
        return None
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(chunk_size)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def compute_model_checksum(data_dir: str, model_name: str) -> str | None:
    """Hash parent model binary (prefer model.ubj, fallback model.json)."""
    paths = model_artifact_paths(data_dir, model_name)
    for key in ("model_ubj", "tuned_model_ubj", "baseline_model_ubj", "model_json"):
        checksum = sha256_file(paths.get(key, ""))
        if checksum:
            return checksum
    return None


def _pointer(path: str, *, kind: str) -> dict[str, Any]:
    exists = os.path.isfile(path)
    return {
        "kind": kind,
        "path": os.path.abspath(path) if path else "",
        "available": exists,
        "status": "available" if exists else "unavailable",
    }


def build_artifact_pointers(data_dir: str, model_name: str) -> dict[str, Any]:
    paths = model_artifact_paths(data_dir, model_name)
    pkg = model_package_dir(data_dir, model_name)
    wf_dir = os.path.join(pkg, "walk_forward")
    pointers: dict[str, Any] = {
        "package_dir": {
            "kind": "directory",
            "path": os.path.abspath(pkg),
            "available": os.path.isdir(pkg),
            "status": "available" if os.path.isdir(pkg) else "unavailable",
        },
        "model_ubj": _pointer(paths.get("model_ubj", ""), kind="model"),
        "model_json": _pointer(paths.get("model_json", ""), kind="metadata"),
        "config_json": _pointer(paths.get("config_json", ""), kind="config"),
        "metrics_json": _pointer(paths.get("metrics_json", ""), kind="metrics"),
        "registry_json": _pointer(paths.get("registry_json", ""), kind="registry"),
        "feature_importance_csv": _pointer(paths.get("feature_importance_csv", ""), kind="importance"),
        "training_summary_json": _pointer(paths.get("training_summary_json", ""), kind="training"),
        "training_metadata_json": _pointer(paths.get("training_metadata_json", ""), kind="training"),
        "dataset_build_snapshot_json": _pointer(
            paths.get("dataset_build_snapshot_json", ""), kind="dataset",
        ),
        "walk_forward_summary_json": _pointer(
            os.path.join(wf_dir, "summary.json"), kind="walk_forward",
        ),
        "walk_forward_champion_aggregate_json": _pointer(
            os.path.join(wf_dir, "champion_aggregate.json"), kind="walk_forward",
        ),
        "selected_features_csv": _pointer(
            os.path.join(wf_dir, "selected_features.csv"), kind="features",
        ),
        "holdout_metrics": _pointer(paths.get("metrics_json", ""), kind="holdout"),
    }
    # Optional aliases matching roadmap naming
    pointers["model.pkl"] = pointers["model_ubj"]
    pointers["feature_importance.json"] = pointers["feature_importance_csv"]
    pointers["walk_forward_results.json"] = pointers["walk_forward_summary_json"]
    pointers["holdout_metrics.json"] = pointers["metrics_json"]
    pointers["selected_features.csv"] = pointers["selected_features_csv"]
    return pointers


def extract_selected_features(doc: dict[str, Any]) -> list[str]:
    feats = doc.get("selected_features")
    if isinstance(feats, list) and feats:
        return [str(f).strip() for f in feats if str(f).strip()]

    wf = _wf(doc)
    sel_art = wf.get("selected_features") or {}
    rows = sel_art.get("rows") if isinstance(sel_art.get("rows"), list) else []
    names: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        feat = str(row.get("feature") or row.get("Feature") or "").strip()
        if not feat:
            continue
        selected = str(row.get("selected") or row.get("Selected") or "").strip().lower()
        if selected in ("", "yes", "true", "1", "y"):
            names.append(feat)
    if names:
        return names

    cfg = _cfg(doc)
    for key in ("selected_features", "features"):
        vals = cfg.get(key)
        if isinstance(vals, list) and vals:
            return [str(f).strip() for f in vals if str(f).strip()]
    return []


def extract_feature_ranking(doc: dict[str, Any]) -> dict[str, Any]:
    """Prefer walk_forward/selected_features.csv rows already loaded into detail doc."""
    wf = _wf(doc)
    sel_art = wf.get("selected_features") if isinstance(wf.get("selected_features"), dict) else {}
    rows = sel_art.get("rows") if isinstance(sel_art.get("rows"), list) else []
    if rows:
        ranked: list[dict[str, Any]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            feat = str(row.get("feature") or row.get("Feature") or "").strip()
            if not feat:
                continue
            ranked.append({
                "feature": feat,
                "final_rank": row.get("final_rank") or row.get("Final Rank"),
                "selected_in_folds": row.get("selected_in_folds") or row.get("Selected in Folds"),
                "selected": row.get("selected") or row.get("Selected"),
                "gain_importance_pct": row.get("gain_importance_pct") or row.get("Gain Importance %"),
                "selection_frequency": row.get("selection_frequency") or row.get("Selection Frequency"),
            })
        if ranked:
            return {
                "source": "walk_forward/selected_features.csv",
                "available": True,
                "rows": ranked,
            }

    # Registry / detail metadata fallback (do not regenerate rankings)
    feats = extract_selected_features(doc)
    if feats:
        return {
            "source": "registry_metadata",
            "available": True,
            "rows": [
                {"feature": name, "final_rank": i, "selected": "Yes"}
                for i, name in enumerate(feats, start=1)
            ],
            "note": "Ordered list only — fold ranking unavailable.",
        }

    return {
        "source": None,
        "available": False,
        "rows": [],
        "message": "Feature ranking unavailable.",
    }


def extract_original_feature_count(doc: dict[str, Any], selected_count: int) -> int | None:
    """Original (pre-RFE) feature universe size when available."""
    candidates: list[int] = []
    wf = _wf(doc)
    display = wf.get("display") if isinstance(wf.get("display"), dict) else {}
    summary_art = wf.get("summary") if isinstance(wf.get("summary"), dict) else {}
    summary_data = summary_art.get("data") if isinstance(summary_art.get("data"), dict) else {}
    fs = summary_data.get("feature_selection") if isinstance(summary_data.get("feature_selection"), dict) else {}
    for key in ("started_features", "features_before", "initial_features", "n_features_before"):
        val = fs.get(key)
        try:
            if val is not None:
                candidates.append(int(val))
        except (TypeError, ValueError):
            pass
    elim = fs.get("elimination") if isinstance(fs.get("elimination"), dict) else {}
    for key in ("started_features", "features_before"):
        val = elim.get(key)
        try:
            if val is not None:
                candidates.append(int(val))
        except (TypeError, ValueError):
            pass

    # Model name often encodes total features: ..._239f_...
    name = str(doc.get("model_name") or "")
    m = re.search(r"_(\d+)f_", name, re.I)
    if m:
        candidates.append(int(m.group(1)))

    meta = _meta(doc)
    cfg = _cfg(doc)
    for val in (
        meta.get("original_feature_count"),
        meta.get("input_feature_count"),
        display.get("input_feature_count"),
        len(cfg.get("features") or []) if isinstance(cfg.get("features"), list) else None,
    ):
        try:
            if val is not None:
                candidates.append(int(val))
        except (TypeError, ValueError):
            pass

    candidates = [c for c in candidates if c > 0]
    if not candidates:
        return selected_count if selected_count > 0 else None
    best = max(candidates)
    if selected_count > 0:
        return max(best, selected_count)
    return best


def build_model_snapshot(doc: dict[str, Any], *, model_checksum: str | None = None) -> dict[str, Any]:
    meta = _meta(doc)
    cfg = _cfg(doc)
    row = doc.get("table_row") if isinstance(doc.get("table_row"), dict) else {}
    return {
        "model_name": doc.get("model_name") or meta.get("model_name") or row.get("model_name"),
        "model_id": doc.get("model_name") or meta.get("model_name") or row.get("model_name"),
        "model_version": meta.get("model_version") or cfg.get("model_version"),
        "model_checksum": model_checksum,
        "algorithm": meta.get("algorithm") or cfg.get("algorithm_label") or cfg.get("algorithm"),
        "target": meta.get("target") or cfg.get("target") or row.get("target"),
        "dataset": meta.get("dataset") or cfg.get("dataset") or row.get("dataset"),
        "trained_at": meta.get("trained_at") or row.get("trained_at"),
        "feature_count": meta.get("feature_count") or row.get("feature_count"),
        "row_count": meta.get("row_count") or row.get("rows"),
        "status": row.get("status") or "ready",
        "is_walk_forward": bool(doc.get("is_walk_forward")),
        "description": meta.get("model_description") or cfg.get("model_description"),
    }


def build_dataset_snapshot(doc: dict[str, Any], data_dir: str) -> dict[str, Any]:
    meta = _meta(doc)
    cfg = _cfg(doc)
    row = doc.get("table_row") if isinstance(doc.get("table_row"), dict) else {}
    name = meta.get("dataset") or cfg.get("dataset") or row.get("dataset")
    snapshot: dict[str, Any] = {
        "dataset_name": name,
        "row_count": meta.get("row_count") or row.get("rows"),
        "sampling_interval_label": doc.get("sampling_interval_label"),
        "strike_selection_label": doc.get("strike_selection_label"),
    }
    # Prefer model's own dataset_build_snapshot artifact when present
    paths = model_artifact_paths(data_dir, str(doc.get("model_name") or ""))
    snap_path = paths.get("dataset_build_snapshot_json") or ""
    if snap_path and os.path.isfile(snap_path):
        try:
            with open(snap_path, encoding="utf-8") as fh:
                loaded = json.load(fh)
            if isinstance(loaded, dict):
                snapshot["dataset_build_snapshot"] = loaded
                if loaded.get("master_db_path") and not snapshot.get("master_db_path"):
                    snapshot["master_db_path"] = loaded.get("master_db_path")
        except (OSError, json.JSONDecodeError, TypeError):
            pass
    # Registry / training dataset JSON often carries master_db_path
    if name and data_dir and not snapshot.get("master_db_path"):
        try:
            from chain_replay_ml.model_lab.prediction_io import (
                load_dataset_meta,
                resolve_dataset_parquet,
            )

            _pq, meta_path = resolve_dataset_parquet(data_dir, str(name))
            ds_meta = load_dataset_meta(meta_path)
            if ds_meta.get("master_db_path"):
                snapshot["master_db_path"] = ds_meta.get("master_db_path")
        except Exception:
            pass
    return snapshot


def build_training_config_snapshot(doc: dict[str, Any]) -> dict[str, Any]:
    cfg = _cfg(doc)
    return {
        "algorithm": cfg.get("algorithm") or cfg.get("algorithm_label"),
        "parameters": cfg.get("parameters") if isinstance(cfg.get("parameters"), dict) else {},
        "split": cfg.get("split") if isinstance(cfg.get("split"), dict) else {},
        "target": cfg.get("target"),
        "dataset": cfg.get("dataset"),
        "selected_features": cfg.get("selected_features") or cfg.get("features"),
        "model_version": cfg.get("model_version"),
    }


def build_wf_snapshot(doc: dict[str, Any]) -> dict[str, Any]:
    wf = _wf(doc)
    cfg = _cfg(doc)
    split = cfg.get("split") if isinstance(cfg.get("split"), dict) else {}
    wf_cfg = split.get("walk_forward") if isinstance(split.get("walk_forward"), dict) else {}
    display = wf.get("display") if isinstance(wf.get("display"), dict) else {}
    summary_art = wf.get("summary") if isinstance(wf.get("summary"), dict) else {}
    summary_data = summary_art.get("data") if isinstance(summary_art.get("data"), dict) else {}
    return {
        "available": bool(wf.get("available")),
        "config": wf_cfg,
        "display": display,
        "aggregated": summary_data.get("aggregated") if isinstance(summary_data.get("aggregated"), dict) else {},
        "n_folds": display.get("n_folds") or wf_cfg.get("n_folds"),
        "selected_feature_count": display.get("selected_feature_count"),
        "feature_selection_method": display.get("feature_selection_method")
        or wf_cfg.get("feature_selection_method"),
    }


def build_metrics_snapshot(doc: dict[str, Any]) -> dict[str, Any]:
    metrics = _metrics(doc)
    summary = _summary(doc)
    prod = doc.get("production_metrics") if isinstance(doc.get("production_metrics"), dict) else {}
    return {
        "training": {
            "trees_trained": summary.get("trees_trained"),
            "early_stopped": summary.get("early_stopped"),
            "training_time_sec": summary.get("training_time_sec"),
            "train_rmse": metrics.get("train_rmse") or (metrics.get("training") or {}).get("rmse")
            if isinstance(metrics.get("training"), dict)
            else metrics.get("train_rmse"),
            "best_iteration": (doc.get("training_metadata") or {}).get("best_iteration")
            if isinstance(doc.get("training_metadata"), dict)
            else None,
        },
        "validation": metrics.get("validation") if isinstance(metrics.get("validation"), dict) else {},
        "holdout": metrics.get("test") if isinstance(metrics.get("test"), dict) else metrics.get("holdout"),
        "test": metrics.get("test") if isinstance(metrics.get("test"), dict) else {},
        "walk_forward": metrics.get("walk_forward") if isinstance(metrics.get("walk_forward"), dict) else {},
        "production_walk_forward": metrics.get("production_walk_forward")
        if isinstance(metrics.get("production_walk_forward"), dict)
        else prod,
        "production_metrics": prod,
    }


def build_lab_snapshots(data_dir: str, doc: dict[str, Any]) -> dict[str, Any]:
    model_name = str(doc.get("model_name") or "").strip()
    selected = extract_selected_features(doc)
    selected_count = len(selected)
    ranking = extract_feature_ranking(doc)
    model_checksum = compute_model_checksum(data_dir, model_name)
    model_snap = build_model_snapshot(doc, model_checksum=model_checksum)
    original_count = extract_original_feature_count(doc, selected_count)
    return {
        "model_snapshot": model_snap,
        "dataset_snapshot": build_dataset_snapshot(doc, data_dir),
        "training_config_snapshot": build_training_config_snapshot(doc),
        "wf_snapshot": build_wf_snapshot(doc),
        "metrics_snapshot": build_metrics_snapshot(doc),
        "selected_features_snapshot": selected,
        "feature_ranking_snapshot": ranking,
        "artifact_pointers": build_artifact_pointers(data_dir, model_name),
        "original_feature_count": original_count,
        "selected_feature_count": selected_count or model_snap.get("feature_count"),
        "training_rows": model_snap.get("row_count"),
        "target": model_snap.get("target"),
        "algorithm": model_snap.get("algorithm"),
        "parent_model_id": str(model_snap.get("model_id") or model_name),
        "parent_model_name": model_name,
        "model_version": model_snap.get("model_version"),
        "model_checksum": model_checksum,
    }
