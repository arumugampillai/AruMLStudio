"""Tests for SHAP Analysis (Phase 2 Research Lab)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from chain_replay_ml.dataset_builder.analysis_feature_profiles import (
    build_feature_profiles,
    load_feature_profile,
    load_feature_scorecard,
)
from chain_replay_ml.dataset_builder.analysis_lab_store import (
    ensure_analysis_run,
    module_statuses,
    register_dataset,
)
from chain_replay_ml.dataset_builder.analysis_mutual_information import analysis_timeline
from chain_replay_ml.dataset_builder.analysis_shap import (
    load_shap_results,
    rehydrate_shap_into_profiles,
    run_shap_analysis,
    shap_already_computed,
)
from chain_replay_ml.training.paths import model_package_dir, models_dir


def _train_tiny_xgb(tmp: str, features: list[str], model_name: str) -> None:
    from xgboost import XGBRegressor

    rng = np.random.default_rng(0)
    n = 120
    X = {f: rng.normal(size=n) for f in features}
    # Make first feature drive the target
    y = X[features[0]] * 2.0 + rng.normal(scale=0.1, size=n)
    df_x = pd.DataFrame(X)
    model = XGBRegressor(
        n_estimators=8,
        max_depth=2,
        learning_rate=0.3,
        objective="reg:squarederror",
        verbosity=0,
    )
    model.fit(df_x, y)

    pkg = model_package_dir(tmp, model_name)
    os.makedirs(pkg, exist_ok=True)
    model_path = os.path.join(pkg, "model.ubj")
    model.save_model(model_path)
    with open(os.path.join(pkg, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_features": features,
                "features": features,
                "algorithm": "xgboost",
                "target": "future_ltp_5m",
            },
            f,
        )
    with open(os.path.join(pkg, "registry.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": model_name,
                "algorithm": "xgboost",
                "target": "future_ltp_5m",
                "status": "trained",
                "dataset": "shap_demo",
            },
            f,
        )


class ShapAnalysisTests(unittest.TestCase):
    def _make_env(self, tmp: str) -> tuple[str, dict, str]:
        features = ["spot", "current_iv", "noise_feat"]
        n = 100
        rng = np.random.default_rng(1)
        spot = pd.Series(np.linspace(100, 140, n)) + rng.normal(0, 0.2, n)
        df = pd.DataFrame(
            {
                "trading_day": ["2026-07-24"] * n,
                "spot": spot,
                "current_iv": np.linspace(0.1, 0.2, n),
                "noise_feat": rng.normal(size=n),
                "future_ltp_5m": spot * 1.01 + rng.normal(0, 0.5, n),
            }
        )
        path = os.path.join(tmp, "shap_demo.parquet")
        df.to_parquet(path, index=False)
        register_dataset(tmp, path, name="shap_demo")
        run = ensure_analysis_run(tmp, "shap_demo")
        model_name = "Tiny_SHAP_XGB"
        _train_tiny_xgb(tmp, features, model_name)
        ds = {"path": path, "dataset_id": "shap_demo", "name": "shap_demo"}
        return run["run_id"], ds, model_name

    def test_compute_persist_reuse_and_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_id, ds, model_name = self._make_env(tmp)
            progress_log: list[str] = []

            def _progress(msg: str, elapsed: float) -> None:
                progress_log.append(f"{elapsed:.1f}:{msg}")

            out = run_shap_analysis(
                tmp,
                run_id,
                ds,
                model_name,
                sample_size=80,
                progress=_progress,
            )
            self.assertFalse(out["reused"])
            self.assertGreater(out["features"], 0)
            self.assertTrue(progress_log)
            self.assertIn("SHAP done", out["message"])
            self.assertTrue(shap_already_computed(tmp, run_id, model_name))

            rows = load_shap_results(tmp, run_id, model_name)
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["rank"], 1)
            self.assertGreaterEqual(float(rows[0]["percentile"]), 90.0)
            # Strong driver should rank at or near top
            self.assertEqual(rows[0]["feature"], "spot")

            out2 = run_shap_analysis(tmp, run_id, ds, model_name, sample_size=80)
            self.assertTrue(out2["reused"])

            build_feature_profiles(tmp, run_id, ds)
            rehydrate_shap_into_profiles(tmp, run_id, model_name=model_name)
            prof = load_feature_profile(tmp, run_id, "spot")
            self.assertIsNotNone(prof)
            assert prof is not None
            self.assertIsNotNone(prof.get("shap_importance"))
            self.assertEqual(prof.get("shap_model"), model_name)
            self.assertEqual(int(prof.get("shap_rank") or 0), 1)

            card = load_feature_scorecard(tmp, run_id)
            shap_rows = [r for r in card if r.get("shap_importance") is not None]
            self.assertGreaterEqual(len(shap_rows), 3)

            statuses = {
                m["module_id"]: m["status"] for m in module_statuses(tmp, run_id)
            }
            self.assertEqual(statuses.get("shap"), "completed")

            tl = analysis_timeline(tmp, run_id, "spot")
            by_id = {s["id"]: s["state"] for s in tl}
            self.assertEqual(by_id["shap"], "done")

    def test_missing_model_features_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_id, ds, _ = self._make_env(tmp)
            # Train a model that needs a column not in parquet
            _train_tiny_xgb(tmp, ["spot", "missing_col"], "Bad_Model")
            with self.assertRaises(KeyError):
                run_shap_analysis(tmp, run_id, ds, "Bad_Model", sample_size=50)


if __name__ == "__main__":
    unittest.main()
