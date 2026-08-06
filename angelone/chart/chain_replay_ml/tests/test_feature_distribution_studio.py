"""Unit tests for Feature Distribution Studio compute (synthetic package)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd


class FeatureDistributionStudioTests(unittest.TestCase):
    def _tiny_package(self, root: str) -> str:
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
                "f_b": rng.normal(loc=5, scale=2, size=n),
                "f_c": rng.exponential(scale=1.0, size=n),
            }
        )
        # inject nulls + skew signal
        df.loc[0:19, "f_a"] = np.nan
        df["future_ltp_5m"] = (
            10 + 2.0 * df["f_a"].fillna(0) + 0.5 * df["f_b"] + rng.normal(scale=0.1, size=n)
        )
        data_dir = root
        ds_dir = os.path.join(data_dir, "datasets")
        os.makedirs(ds_dir, exist_ok=True)
        ds_name = "tiny_fds"
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
        model_name = safe_model_name("Tiny_FDS_Model")
        pkg = model_package_dir(data_dir, model_name)
        os.makedirs(pkg, exist_ok=True)
        # Dist Studio does not need a booster; config + holdout meta suffice.
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
            "skip_dataset_audit": True,
            "skip_dataset_validation": True,
        }
        with open(os.path.join(pkg, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(config, fh)
        with open(os.path.join(pkg, "metadata.json"), "w", encoding="utf-8") as fh:
            json.dump({"model_name": model_name, "dataset": ds_name, "feature_count": 3}, fh)
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
        # Optional Importance join fixture
        imp_dir = os.path.join(pkg, "feature_importance_studio")
        os.makedirs(imp_dir, exist_ok=True)
        with open(os.path.join(imp_dir, "comparison.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "rows": [
                        {
                            "feature": "f_a",
                            "gain": 10.0,
                            "rank_gain": 1,
                            "shap_mean_abs": 0.5,
                            "rank_shap": 1,
                            "rank_delta_gain_shap": 0,
                            "permutation_mean": 0.1,
                        },
                        {
                            "feature": "f_b",
                            "gain": 5.0,
                            "rank_gain": 2,
                            "shap_mean_abs": 0.2,
                            "rank_shap": 2,
                            "rank_delta_gain_shap": 0,
                            "permutation_mean": 0.05,
                        },
                        {
                            "feature": "f_c",
                            "gain": 1.0,
                            "rank_gain": 3,
                            "shap_mean_abs": 0.05,
                            "rank_shap": 3,
                            "rank_delta_gain_shap": 0,
                            "permutation_mean": 0.01,
                        },
                    ]
                },
                fh,
            )
        return model_name

    def test_feature_distribution_row_nulls(self) -> None:
        from chain_replay_ml.feature_distribution_studio.stats import (
            feature_distribution_row,
        )

        s = pd.Series([1.0, 2.0, np.nan, 4.0, np.inf])
        row = feature_distribution_row("x", s)
        self.assertEqual(row["feature"], "x")
        self.assertEqual(row["count"], 5)
        self.assertEqual(row["n_finite"], 3)
        self.assertGreater(row["null_pct"], 0)
        self.assertIsNotNone(row["p50"])
        self.assertIn("skew", row)

    def test_end_to_end_artifacts(self) -> None:
        from chain_replay_ml.feature_distribution_studio import (
            run_feature_distribution_studio,
        )

        with tempfile.TemporaryDirectory() as tmp:
            model_name = self._tiny_package(tmp)
            result = run_feature_distribution_studio(
                data_dir=tmp,
                model_name=model_name,
                holdout_max_rows=200,
            )
            self.assertTrue(result.ok, result.error)
            self.assertTrue(result.comparison)
            art = result.artifacts_dir
            for name in ("holdout_stats.json", "comparison.json", "run_meta.json"):
                self.assertTrue(os.path.isfile(os.path.join(art, name)), name)
            with open(os.path.join(art, "comparison.json"), encoding="utf-8") as fh:
                comp = json.load(fh)
            row = next(r for r in comp["rows"] if r["feature"] == "f_a")
            for key in (
                "feature",
                "null_pct",
                "mean",
                "std",
                "min",
                "p1",
                "p5",
                "p25",
                "p50",
                "p75",
                "p95",
                "p99",
                "max",
                "skew",
                "rank_gain",
                "importance_joined",
            ):
                self.assertIn(key, row)
            self.assertTrue(row["importance_joined"])
            self.assertEqual(row["rank_gain"], 1)
            with open(os.path.join(art, "run_meta.json"), encoding="utf-8") as fh:
                meta = json.load(fh)
            self.assertEqual(meta.get("model_name"), model_name)
            self.assertTrue(meta.get("importance_joined"))
            self.assertIn("timings_sec", meta)


if __name__ == "__main__":
    unittest.main()
