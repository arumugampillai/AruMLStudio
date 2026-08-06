"""Unit tests for Feature Importance Studio compute (synthetic package)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd


class FeatureImportanceStudioTests(unittest.TestCase):
    def _train_tiny_package(self, root: str) -> str:
        import xgboost as xgb

        from chain_replay_ml.training.paths import model_package_dir, safe_model_name

        rng = np.random.default_rng(0)
        n = 800
        df = pd.DataFrame(
            {
                "trading_day": ["2026-07-23"] * (n // 2) + ["2026-07-24"] * (n // 2),
                "timestamp": np.arange(n),
                "token": rng.integers(1, 5, size=n),
                "ltp": rng.uniform(10, 50, size=n),
                "f_a": rng.normal(size=n),
                "f_b": rng.normal(size=n),
                "f_c": rng.normal(size=n),
            }
        )
        df["future_ltp_5m"] = (
            10 + 2.0 * df["f_a"] + 0.5 * df["f_b"] + rng.normal(scale=0.1, size=n)
        )
        data_dir = root
        ds_dir = os.path.join(data_dir, "datasets")
        os.makedirs(ds_dir, exist_ok=True)
        ds_name = "tiny_fis"
        df.to_parquet(os.path.join(ds_dir, f"{ds_name}.parquet"), index=False)
        with open(os.path.join(ds_dir, f"{ds_name}.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "dataset_name": ds_name,
                    "feature_columns": ["f_a", "f_b", "f_c", "ltp"],
                    "row_count": n,
                },
                fh,
            )

        features = ["f_a", "f_b", "f_c"]
        X = df[features].astype("float32")
        y = df["future_ltp_5m"].astype("float32")
        dtrain = xgb.DMatrix(X, label=y, feature_names=features)
        bst = xgb.train(
            {"objective": "reg:squarederror", "max_depth": 3, "eta": 0.2, "verbosity": 0},
            dtrain,
            num_boost_round=20,
        )
        model_name = safe_model_name("Tiny_FIS_Model")
        pkg = model_package_dir(data_dir, model_name)
        os.makedirs(pkg, exist_ok=True)
        ubj = os.path.join(pkg, "model.ubj")
        bst.save_model(ubj)
        config = {
            "dataset": ds_name,
            "target": "future_ltp_5m",
            "algorithm": "xgboost",
            "prediction_type": "regression",
            "features": features,
            "model_version": "1.0",
            "split": {
                "train": 70,
                "validation": 15,
                "test": 15,
                "strategy": "walk_forward",
                "walk_forward": {
                    "n_folds": 2,
                    "window_mode": "expanding",
                    "fold_placement": "distributed",
                    "train_window_size": 200,
                    "validation_window_size": 50,
                    "feature_selection_method": "none",
                },
            },
            "xgboost": {"n_estimators": 20, "max_depth": 3, "learning_rate": 0.2},
            "skip_dataset_audit": True,
            "skip_dataset_validation": True,
        }
        with open(os.path.join(pkg, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(config, fh)
        with open(os.path.join(pkg, "metadata.json"), "w", encoding="utf-8") as fh:
            json.dump({"model_name": model_name, "dataset": ds_name, "feature_count": 3}, fh)
        # walk_forward summary with explicit holdout for resolve_holdout_slice
        wf = os.path.join(pkg, "walk_forward")
        os.makedirs(wf, exist_ok=True)
        with open(os.path.join(wf, "summary.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "test_holdout": {"start": 600, "stop": 800},
                    "meta": {"n_folds": 2},
                },
                fh,
            )
        return model_name

    def test_native_gain_weight_cover(self) -> None:
        import xgboost as xgb

        from chain_replay_ml.feature_importance_studio.native import (
            compute_native_xgb_importance,
        )

        rng = np.random.default_rng(1)
        X = pd.DataFrame({"a": rng.normal(size=100), "b": rng.normal(size=100)})
        y = X["a"] * 2 + rng.normal(scale=0.05, size=100)
        bst = xgb.train(
            {"objective": "reg:squarederror", "max_depth": 2, "verbosity": 0},
            xgb.DMatrix(X, label=y, feature_names=["a", "b"]),
            num_boost_round=10,
        )

        class _M:
            def get_booster(self):
                return bst

        rows = compute_native_xgb_importance(_M(), ["a", "b"])
        self.assertEqual(len(rows), 2)
        self.assertIn("gain", rows[0])
        self.assertIn("weight", rows[0])
        self.assertIn("cover", rows[0])
        self.assertGreater(float(rows[0]["gain"]) + float(rows[1]["gain"]), 0)

    def test_end_to_end_artifacts(self) -> None:
        from chain_replay_ml.feature_importance_studio import run_feature_importance_studio

        with tempfile.TemporaryDirectory() as tmp:
            model_name = self._train_tiny_package(tmp)
            result = run_feature_importance_studio(
                data_dir=tmp,
                model_name=model_name,
                holdout_max_rows=200,
                permutation_n_repeats=2,
                shap_sample_size=80,
            )
            self.assertTrue(result.ok, result.error)
            self.assertTrue(result.comparison)
            art = result.artifacts_dir
            for name in (
                "native_xgb.json",
                "permutation.json",
                "shap.json",
                "comparison.json",
                "run_meta.json",
            ):
                self.assertTrue(os.path.isfile(os.path.join(art, name)), name)
            with open(os.path.join(art, "comparison.json"), encoding="utf-8") as fh:
                comp = json.load(fh)
            row = comp["rows"][0]
            for key in (
                "feature",
                "gain",
                "weight",
                "cover",
                "permutation_mean",
                "permutation_std",
                "shap_mean_abs",
                "rank_gain",
                "rank_permutation",
                "rank_shap",
                "rank_delta_gain_shap",
            ):
                self.assertIn(key, row)
            with open(os.path.join(art, "run_meta.json"), encoding="utf-8") as fh:
                meta = json.load(fh)
            self.assertEqual(meta.get("model_name"), model_name)
            self.assertIn("timings_sec", meta)
            self.assertIn("dataset_engine_backend", meta)


if __name__ == "__main__":
    unittest.main()
