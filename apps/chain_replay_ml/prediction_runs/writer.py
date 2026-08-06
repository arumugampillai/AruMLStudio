"""Write prediction rows during walk-forward validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .hashes import (
    dataset_fingerprint,
    feature_snapshot_hash,
    training_config_hash,
    walk_forward_config_hash,
)
from .store import PredictionRunStore
from chain_replay_ml.training.evaluator import direction_correct_flag

IDENTITY_COLUMNS = (
    "trading_day", "timestamp", "token", "strike", "option_type",
    "spot", "ltp", "symbol", "market", "expiry",
)

_BATCH = 2000


class PredictionRunWriter:
    """Accumulates fold metadata and prediction rows for one run."""

    def __init__(self, store: PredictionRunStore, run_id: str) -> None:
        self.store = store
        self.run_id = run_id
        self._fold_ids: dict[int, str] = {}
        self._row_count = 0

    @property
    def row_count(self) -> int:
        return self._row_count

    def write_fold_predictions(
        self,
        *,
        fold_number: int,
        fold_def: dict[str, Any],
        metrics: dict[str, Any],
        val_context: pd.DataFrame,
        val_pred: np.ndarray,
        val_y: pd.Series,
        baseline_ltp: pd.Series | None = None,
    ) -> str:
        run_id = self.run_id
        tr = fold_def.get("train") or {}
        va = fold_def.get("validation") or {}
        fold_id = self.store.insert_fold({
            "run_id": run_id,
            "fold_number": fold_number,
            "train_start": tr.get("start"),
            "train_end": tr.get("stop"),
            "validation_start": va.get("start"),
            "validation_end": va.get("stop"),
            "train_rows": tr.get("rows"),
            "validation_rows": va.get("rows"),
            "mae": metrics.get("mae"),
            "rmse": metrics.get("rmse"),
            "directional_accuracy_pct": metrics.get("directional_accuracy_pct"),
            "prediction_count": len(val_y),
            "meta": {"window_mode": fold_def.get("window_mode")},
        })
        self._fold_ids[fold_number] = fold_id

        n = len(val_y)
        pred = np.asarray(val_pred, dtype=float).reshape(-1)
        actual = pd.to_numeric(val_y, errors="coerce").to_numpy(dtype=float)
        ctx = val_context.reset_index(drop=True)
        base = None
        if baseline_ltp is not None:
            base = pd.to_numeric(baseline_ltp.reset_index(drop=True), errors="coerce").to_numpy(dtype=float)

        batch: list[dict[str, Any]] = []
        for i in range(n):
            row = _context_row(ctx, i)
            predicted = float(pred[i]) if i < len(pred) and np.isfinite(pred[i]) else None
            actual_v = float(actual[i]) if i < len(actual) and np.isfinite(actual[i]) else None
            err = (predicted - actual_v) if predicted is not None and actual_v is not None else None
            dir_ok = None
            ltp_val = row.get("ltp")
            if base is not None and i < len(base) and np.isfinite(base[i]):
                b = float(base[i])
                if ltp_val is None:
                    ltp_val = b
                if predicted is not None and actual_v is not None:
                    dir_ok = direction_correct_flag(predicted, actual_v, b)
            batch.append({
                "prediction_id": f"{run_id}:{fold_id}:{i}",
                "run_id": run_id,
                "fold_id": fold_id,
                "row_index": i,
                "timestamp": row.get("timestamp"),
                "trading_day": row.get("trading_day"),
                "token": row.get("token"),
                "strike": row.get("strike"),
                "option_type": row.get("option_type"),
                "spot": row.get("spot"),
                "ltp": ltp_val,
                "predicted_ltp": predicted,
                "actual_ltp": actual_v,
                "prediction_error": err,
                "direction_correct": dir_ok,
                "confidence": None,
            })
            if len(batch) >= _BATCH:
                self.store.insert_rows_batch(batch)
                self._row_count += len(batch)
                batch = []
        if batch:
            self.store.insert_rows_batch(batch)
            self._row_count += len(batch)
        return fold_id


def _context_row(ctx: pd.DataFrame, i: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if i >= len(ctx):
        return out
    for col in IDENTITY_COLUMNS:
        if col not in ctx.columns:
            continue
        val = ctx.iloc[i][col]
        if pd.isna(val):
            continue
        if col == "timestamp":
            try:
                out[col] = float(val)
            except (TypeError, ValueError):
                out[col] = None
        elif col in ("strike", "spot", "ltp"):
            try:
                out[col] = float(val)
            except (TypeError, ValueError):
                out[col] = val
        else:
            out[col] = str(val)
    return out


def create_prediction_run(
    data_dir: str,
    *,
    model_id: str,
    model_version: str | None,
    dataset_name: str,
    target: str,
    metadata: dict[str, Any] | None,
    features: list[str],
    wf_cfg: dict[str, Any],
    config_dict: dict[str, Any],
    package_dir: str | None = None,
    run_kind: str = "walk_forward_production",
) -> tuple[PredictionRunStore, PredictionRunWriter, dict[str, Any]]:
    store = PredictionRunStore(data_dir)
    store.open()
    run = store.create_run({
        "model_id": model_id,
        "model_version": model_version,
        "dataset_name": dataset_name,
        "target": target,
        "dataset_fingerprint": dataset_fingerprint(metadata, dataset_name),
        "feature_snapshot_hash": feature_snapshot_hash(features),
        "walk_forward_config_hash": walk_forward_config_hash(wf_cfg),
        "training_config_hash": training_config_hash(config_dict),
        "package_dir": package_dir,
        "run_kind": run_kind,
        "status": "running",
    })
    writer = PredictionRunWriter(store, run["run_id"])
    return store, writer, run


def load_context_columns(
    data_dir: str,
    dataset_name: str,
    features: list[str],
) -> list[str]:
    """Identity columns available in dataset parquet (excluding features already loaded)."""
    from chain_replay_ml.training.dataset_loader import load_dataset_frame

    try:
        df, _, _ = load_dataset_frame(data_dir, dataset_name, columns=list(IDENTITY_COLUMNS))
        return [c for c in IDENTITY_COLUMNS if c in df.columns]
    except Exception:
        return []


def record_champion_prediction_run(
    *,
    data_dir: str,
    model_id: str,
    model_version: str | None,
    config_dict: dict[str, Any],
    metadata: dict[str, Any] | None,
    dataset_name: str,
    target: str,
    features: list[str],
    parameters: dict[str, Any],
    wf_cfg: dict[str, Any],
    fold_defs: list[dict[str, Any]],
    X: pd.DataFrame,
    y: pd.Series,
    context_df: pd.DataFrame,
    package_dir: str | None,
    training_duration_sec: float | None,
    algorithm: str = "xgboost",
    prediction_type: str = "regression",
    score_refs: dict[str, float] | None = None,
    cancel_check=None,
) -> dict[str, Any]:
    """Re-run champion production WF and persist all validation predictions."""
    from ..training.walk_forward_runner import evaluate_hyperparameters_on_walk_forward

    store, writer, run = create_prediction_run(
        data_dir,
        model_id=model_id,
        model_version=model_version,
        dataset_name=dataset_name,
        target=target,
        metadata=metadata,
        features=features,
        wf_cfg=wf_cfg,
        config_dict=config_dict,
        package_dir=package_dir,
    )
    try:
        evaluate_hyperparameters_on_walk_forward(
            X=X,
            y=y,
            features=features,
            parameters=parameters,
            fold_defs=fold_defs,
            algorithm=algorithm,
            prediction_type=prediction_type,
            score_refs=score_refs,
            cancel_check=cancel_check,
            context_df=context_df if context_df is not None and len(context_df) == len(X) else None,
            prediction_writer=writer,
        )
        store.finalize_run(
            run["run_id"],
            status="completed",
            training_duration_sec=training_duration_sec,
            prediction_count=writer.row_count,
            fold_count=len(fold_defs),
        )
        detail = store.get_run(run["run_id"]) or run
        detail["prediction_count_stored"] = writer.row_count
        if package_dir:
            import json
            import os

            os.makedirs(package_dir, exist_ok=True)
            doc = {
                "run_id": detail.get("run_id"),
                "model_id": detail.get("model_id"),
                "model_version": detail.get("model_version"),
                "dataset_name": detail.get("dataset_name"),
                "target": detail.get("target"),
                "status": detail.get("status"),
                "run_kind": detail.get("run_kind"),
                "created_at": detail.get("created_at"),
                "finished_at": detail.get("finished_at"),
                "training_duration_sec": detail.get("training_duration_sec"),
                "prediction_count": writer.row_count,
                "fold_count": len(fold_defs),
                "dataset_fingerprint": detail.get("dataset_fingerprint"),
                "feature_snapshot_hash": detail.get("feature_snapshot_hash"),
                "walk_forward_config_hash": detail.get("walk_forward_config_hash"),
                "training_config_hash": detail.get("training_config_hash"),
            }
            with open(os.path.join(package_dir, "prediction_run.json"), "w", encoding="utf-8") as fh:
                json.dump(doc, fh, indent=2)
        return detail
    except Exception:
        store.finalize_run(run["run_id"], status="failed")
        raise
    finally:
        store.close()

