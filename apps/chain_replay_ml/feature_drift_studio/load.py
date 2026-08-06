"""Load WF region + holdout slices for Drift Studio (Dataset Engine path)."""

from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd

from chain_replay_ml.training.config import normalize_training_config
from chain_replay_ml.training.dataset_loader import load_training_xy
from chain_replay_ml.training.holdout_performance import resolve_holdout_slice_with_fallback
from chain_replay_ml.training.paths import safe_model_name
from chain_replay_ml.training.registry import _selected_feature_names


def _read_json(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        doc = json.load(fh)
    return doc if isinstance(doc, dict) else {}


def load_wf_holdout_xy(
    *,
    data_dir: str,
    package_dir: str,
    model_name: str,
    doc: dict[str, Any],
    holdout_max_rows: int | None = None,
    wf_max_rows: int | None = None,
    X: pd.DataFrame | None = None,
    y: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, list[str], dict[str, Any]]:
    """Load training matrix, return (X_wf, y_wf, X_ho, y_ho, features, meta).

    Optional ``X``/``y`` reuse the Create Model in-memory matrix (no disk reload).
    """
    config_path = os.path.join(package_dir, "config.json")
    if not os.path.isfile(config_path):
        config_path = os.path.join(package_dir, "training_config.json")
    config_raw = _read_json(config_path)
    if not config_raw and isinstance(doc.get("config"), dict):
        nested = doc["config"]
        config_raw = nested.get("data") if isinstance(nested.get("data"), dict) else nested

    training_cfg = normalize_training_config(config_raw)
    paths = {
        "config_json": os.path.join(package_dir, "config.json"),
        "package_dir": package_dir,
    }
    selected = _selected_feature_names(data_dir, safe_model_name(model_name), paths)
    if selected:
        training_cfg.features = list(selected)

    passthrough = X is not None and y is not None
    if passthrough:
        assert X is not None and y is not None
        features = list(training_cfg.features) or list(X.columns)
        use_features = [f for f in features if f in X.columns]
        if not use_features:
            raise ValueError("In-memory matrix has none of the selected features")
        X_mat = X[use_features]
        y_mat = y
        metadata: dict[str, Any] = {
            "dataset_load": {"backend": "in_memory_passthrough", "rows_returned": int(len(X_mat))},
        }
    else:
        X_mat, y_mat, features, metadata, _expected, _ctx = load_training_xy(
            data_dir, training_cfg
        )
        use_features = [f for f in features if f in X_mat.columns]
        X_mat = X_mat[use_features]

    n_rows = len(X_mat)
    start, stop = resolve_holdout_slice_with_fallback(doc, n_rows, training_cfg)

    X_wf = X_mat.iloc[:start].copy()
    y_wf = y_mat.iloc[:start].copy()
    X_ho = X_mat.iloc[start:stop].copy()
    y_ho = y_mat.iloc[start:stop].copy()

    if wf_max_rows is not None and len(X_wf) > int(wf_max_rows):
        # keep the tail of WF (closest to holdout) for representativeness
        X_wf = X_wf.iloc[-int(wf_max_rows) :]
        y_wf = y_wf.iloc[-int(wf_max_rows) :]
    if holdout_max_rows is not None and len(X_ho) > int(holdout_max_rows):
        X_ho = X_ho.iloc[: int(holdout_max_rows)]
        y_ho = y_ho.iloc[: int(holdout_max_rows)]

    load_meta = {
        "dataset": training_cfg.dataset,
        "target": training_cfg.target,
        "holdout_start": int(start),
        "holdout_stop": int(stop),
        "wf_rows": int(len(X_wf)),
        "holdout_rows": int(len(X_ho)),
        "feature_count": len(use_features),
        "dataset_load": dict((metadata or {}).get("dataset_load") or {}),
        "prediction_type": str(
            config_raw.get("prediction_type")
            or config_raw.get("predictionType")
            or "regression"
        ).strip().lower(),
        "frame_backend": "polars_stats",
        "matrix_source": "in_memory" if passthrough else "disk",
    }
    return X_wf, y_wf, X_ho, y_ho, use_features, load_meta
