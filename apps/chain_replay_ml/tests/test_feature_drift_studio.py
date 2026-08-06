"""Unit tests for Feature Drift Studio compute (synthetic package)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd


class FeatureDriftStudioTests(unittest.TestCase):
    def _tiny_package(self, root: str, *, with_joins: bool = True) -> str:
        from chain_replay_ml.training.paths import model_package_dir, safe_model_name

        rng = np.random.default_rng(0)
        n = 800
        # deliberate holdout shift on f_a
        f_a = rng.normal(size=n)
        f_a[600:] = f_a[600:] + 3.0
        df = pd.DataFrame(
            {
                "trading_day": ["2026-07-23"] * (n // 2) + ["2026-07-24"] * (n // 2),
                "timestamp": np.arange(n),
                "token": rng.integers(1, 5, size=n),
                "ltp": rng.uniform(10, 50, size=n),
                "f_a": f_a,
                "f_b": rng.normal(loc=5, scale=2, size=n),
                "f_c": rng.normal(size=n),
            }
        )
        df["future_ltp_5m"] = (
            10 + 2.0 * df["f_a"] + 0.5 * df["f_b"] + rng.normal(scale=0.1, size=n)
        )
        data_dir = root
        ds_dir = os.path.join(data_dir, "datasets")
        os.makedirs(ds_dir, exist_ok=True)
        ds_name = "tiny_drift"
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
        model_name = safe_model_name("Tiny_Drift_Model")
        pkg = model_package_dir(data_dir, model_name)
        os.makedirs(pkg, exist_ok=True)
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
        if with_joins:
            imp_dir = os.path.join(pkg, "feature_importance_studio")
            os.makedirs(imp_dir, exist_ok=True)
            with open(os.path.join(imp_dir, "comparison.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "rows": [
                            {"feature": "f_a", "gain": 10.0, "rank_gain": 1, "shap_mean_abs": 0.5},
                            {"feature": "f_b", "gain": 5.0, "rank_gain": 2, "shap_mean_abs": 0.2},
                            {"feature": "f_c", "gain": 1.0, "rank_gain": 3, "shap_mean_abs": 0.05},
                        ]
                    },
                    fh,
                )
            dist_dir = os.path.join(pkg, "feature_distribution_studio")
            os.makedirs(dist_dir, exist_ok=True)
            with open(os.path.join(dist_dir, "comparison.json"), "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "rows": [
                            {"feature": "f_a", "null_pct": 0.0, "skew": 0.1},
                            {"feature": "f_b", "null_pct": 1.5, "skew": 0.2},
                            {"feature": "f_c", "null_pct": 0.0, "skew": 0.0},
                        ]
                    },
                    fh,
                )
        return model_name

    def test_end_to_end_artifacts(self) -> None:
        from chain_replay_ml.feature_drift_studio import run_feature_drift_studio

        with tempfile.TemporaryDirectory() as tmp:
            model_name = self._tiny_package(tmp, with_joins=True)
            result = run_feature_drift_studio(
                data_dir=tmp,
                model_name=model_name,
                holdout_max_rows=200,
                wf_max_rows=600,
            )
            self.assertTrue(result.ok, result.error)
            self.assertTrue(result.comparison)
            art = result.artifacts_dir
            for name in ("drift_rows.json", "comparison.json", "run_meta.json"):
                self.assertTrue(os.path.isfile(os.path.join(art, name)), name)
            row = next(r for r in result.comparison if r["feature"] == "f_a")
            for key in (
                "feature",
                "wf_mean",
                "holdout_mean",
                "drift_pct",
                "drift",
                "ks_statistic",
                "ks_pvalue",
                "wasserstein_distance",
                "wasserstein_normalized",
                "null_pct_wf",
                "null_pct_ho",
                "null_drift_pp",
                "importance",
                "risk",
                "risk_score",
                "rank_gain",
                "importance_joined",
                "distribution_joined",
            ):
                self.assertIn(key, row)
            self.assertTrue(row["importance_joined"])
            self.assertTrue(row["distribution_joined"])
            self.assertEqual(row["rank_gain"], 1)
            # shifted feature should have elevated drift / KS / Wasserstein / risk
            self.assertGreater(float(row["drift"]), 0.2)
            self.assertGreater(float(row["ks_statistic"]), 0.2)
            self.assertGreater(float(row["wasserstein_distance"]), 0.5)
            self.assertGreaterEqual(float(row["risk_score"]), 0.0)
            self.assertLessEqual(float(row["risk_score"]), 100.0)
            with open(os.path.join(art, "run_meta.json"), encoding="utf-8") as fh:
                meta = json.load(fh)
            self.assertEqual(meta.get("model_name"), model_name)
            self.assertIn("feature_drift_pct", meta)
            self.assertIn("similarity_pct", meta)
            self.assertIn("average_ks", meta)
            self.assertIn("average_wasserstein", meta)
            self.assertIn("average_drift_pct", meta)
            self.assertEqual(meta.get("studio_version"), "5.2.0")
            self.assertEqual(meta.get("schema_version"), 2)
            self.assertEqual(meta.get("drift_schema"), "v2")
            self.assertIn("average_wasserstein_normalized", meta)
            self.assertTrue(meta.get("importance_joined"))
            with open(os.path.join(art, "comparison.json"), encoding="utf-8") as fh:
                comparison_doc = json.load(fh)
            self.assertEqual(comparison_doc.get("schema_version"), 2)
            with open(os.path.join(art, "drift_rows.json"), encoding="utf-8") as fh:
                drift_doc = json.load(fh)
            self.assertEqual(drift_doc.get("schema_version"), 2)

    def test_without_joins(self) -> None:
        from chain_replay_ml.feature_drift_studio import run_feature_drift_studio

        with tempfile.TemporaryDirectory() as tmp:
            model_name = self._tiny_package(tmp, with_joins=False)
            result = run_feature_drift_studio(
                data_dir=tmp,
                model_name=model_name,
                holdout_max_rows=200,
                wf_max_rows=600,
            )
            self.assertTrue(result.ok, result.error)
            self.assertFalse(result.meta.get("importance_joined"))
            self.assertFalse(result.meta.get("distribution_joined"))
            for row in result.comparison:
                self.assertFalse(row.get("importance_joined"))

    def test_format_si_and_importance_display(self) -> None:
        from master_dataset_tk.feature_drift_studio_panel import (
            format_importance_cell,
            format_si_number,
        )

        self.assertEqual(format_si_number(835_041_077.5622), "835M")
        self.assertEqual(format_si_number(444_703_586_255.097), "445B")
        self.assertEqual(format_si_number(31_700_000), "31.7M")
        self.assertEqual(format_si_number(None), "—")
        self.assertEqual(format_si_number(0), "0")
        self.assertEqual(format_si_number(1.2345), "1.2345")

        self.assertEqual(
            format_importance_cell({"importance": 0.0, "importance_joined": False}),
            "—",
        )
        self.assertEqual(
            format_importance_cell({"importance": None, "importance_joined": True}),
            "—",
        )
        self.assertEqual(
            format_importance_cell({"importance": 0.0, "importance_joined": True}),
            "0.0000",
        )
        self.assertEqual(
            format_importance_cell({"importance": 0.0}),  # old artifact, ambiguous zero
            "—",
        )
        joined = format_importance_cell(
            {"importance": 0.125, "importance_joined": True}
        )
        self.assertIn("0.125", joined)

    def test_loader_backward_compatible_without_schema_version(self) -> None:
        from chain_replay_ml.feature_drift_studio.writer import load_studio_artifacts
        from chain_replay_ml.training.paths import model_package_dir, safe_model_name

        with tempfile.TemporaryDirectory() as tmp:
            model_name = safe_model_name("Legacy_Drift")
            pkg = model_package_dir(tmp, model_name)
            art = os.path.join(pkg, "feature_drift_studio")
            os.makedirs(art, exist_ok=True)
            with open(os.path.join(art, "comparison.json"), "w", encoding="utf-8") as fh:
                json.dump({"rows": [{"feature": "f_a", "risk_score": 1.0}]}, fh)
            with open(os.path.join(art, "run_meta.json"), "w", encoding="utf-8") as fh:
                json.dump({"model_name": model_name, "studio_version": "5.2.0"}, fh)
            loaded = load_studio_artifacts(pkg)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(len(loaded["comparison"]), 1)
            self.assertNotIn("schema_version", loaded["meta"])


if __name__ == "__main__":
    unittest.main()

