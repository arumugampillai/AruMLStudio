"""Batch regression inference — same models + feature columns as live replay."""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd

from chain_replay_ml.training.model_runtime import load_prediction_model_cached

from live_inference.predictor import _feature_value

from .model_registry import slot_columns_for_count


class BatchModelRunner:
    """Load all ready models once; vectorized predict on master feature rows."""

    def __init__(self, specs: list[dict[str, Any]]) -> None:
        self.specs = list(specs)
        self.model_names = [str(s.get("model_name") or f"model_{i}") for i, s in enumerate(self.specs, 1)]
        preds, deltas, ranks = slot_columns_for_count(len(self.specs))
        self.pred_columns = preds
        self.delta_columns = deltas
        self.rank_columns = ranks
        self._models: list[Any] = []
        self._feature_lists: list[list[str]] = []
        self.models_loaded_from_disk = 0
        for spec in self.specs:
            path = spec.get("model_path")
            algo = spec.get("algorithm")
            model, _load_ms, from_disk = load_prediction_model_cached(path, algo)
            if from_disk:
                self.models_loaded_from_disk += 1
            self._models.append(model)
            self._feature_lists.append(list(spec.get("features") or []))

    def predict_batch(
        self,
        rows_df: pd.DataFrame,
        feature_cols: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame, float, float, float]:
        """Return (predictions, status flags, batch_time_ms, prediction_timestamp, feature_valid_pct)."""
        n = len(rows_df)
        pred_out = {col: np.full(n, np.nan, dtype=float) for col in self.pred_columns}
        ok_out = {f"model_{i}_ok": np.zeros(n, dtype=int) for i in range(1, len(self.specs) + 1)}

        shared_rows = rows_df.to_dict(orient="records")
        prediction_timestamp = time.time()
        t0 = time.perf_counter()
        feature_checks = 0
        feature_ok = 0

        for spec, model, feats, col, ok_col in zip(
            self.specs, self._models, self._feature_lists, self.pred_columns, ok_out.keys(),
        ):
            if not feats:
                continue
            matrix: list[list[Any]] = []
            valid_mask = np.ones(n, dtype=bool)
            for row_i, shared in enumerate(shared_rows):
                row_vals = []
                missing = False
                for name in feats:
                    if name not in shared:
                        missing = True
                        break
                    row_vals.append(_feature_value(shared, name))
                feature_checks += 1
                if not missing:
                    feature_ok += 1
                if missing:
                    valid_mask[row_i] = False
                    matrix.append([np.nan] * len(feats))
                else:
                    matrix.append(row_vals)

            X = pd.DataFrame(matrix, columns=feats)
            try:
                preds = model.predict(X)
            except Exception:
                continue

            for row_i, pred in enumerate(preds):
                if not valid_mask[row_i]:
                    continue
                try:
                    val = float(pred)
                except (TypeError, ValueError):
                    continue
                if math.isnan(val) or math.isinf(val):
                    continue
                pred_out[col][row_i] = round(val, 2)
                ok_out[ok_col][row_i] = 1

        batch_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        feature_valid_pct = round(100.0 * feature_ok / max(feature_checks, 1), 1)
        return pd.DataFrame(pred_out), pd.DataFrame(ok_out), batch_ms, prediction_timestamp, feature_valid_pct

    def model_catalog(self) -> dict[str, Any]:
        return {
            "model_count": len(self.specs),
            "model_columns": self.pred_columns,
            "model_names": self.model_names,
            "targets": [str(s.get("target") or "") for s in self.specs],
        }
