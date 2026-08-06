"""Tests for Permutation Importance (Phase 2.4)."""

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
    register_dataset,
)
from chain_replay_ml.dataset_builder.analysis_permutation import (
    CancelToken,
    interpret_permutation,
    load_permutation_results,
    perm_already_complete,
    run_permutation_importance,
)
from chain_replay_ml.training.paths import model_package_dir


def _train_tiny_xgb(tmp: str, features: list[str], model_name: str) -> None:
    from xgboost import XGBRegressor

    rng = np.random.default_rng(0)
    n = 120
    X = {f: rng.normal(size=n) for f in features}
    y = X[features[0]] * 3.0 + rng.normal(scale=0.05, size=n)
    model = XGBRegressor(
        n_estimators=12,
        max_depth=2,
        learning_rate=0.3,
        objective="reg:squarederror",
        verbosity=0,
    )
    model.fit(pd.DataFrame(X), y)
    pkg = model_package_dir(tmp, model_name)
    os.makedirs(pkg, exist_ok=True)
    model.save_model(os.path.join(pkg, "model.ubj"))
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
                "dataset": "perm_demo",
            },
            f,
        )


class PermutationImportanceTests(unittest.TestCase):
    def _env(self, tmp: str) -> tuple[str, dict, str]:
        features = ["spot", "current_iv", "noise_feat"]
        n = 100
        rng = np.random.default_rng(7)
        spot = pd.Series(np.linspace(100, 140, n)) + rng.normal(0, 0.2, n)
        df = pd.DataFrame(
            {
                "trading_day": ["2026-07-24"] * n,
                "spot": spot,
                "current_iv": np.linspace(0.1, 0.2, n),
                "noise_feat": rng.normal(size=n),
                "future_ltp_5m": spot * 1.02 + rng.normal(0, 0.3, n),
                "label_up_5m": (spot.diff().fillna(0) > 0).astype(int),
            }
        )
        path = os.path.join(tmp, "perm_demo.parquet")
        df.to_parquet(path, index=False)
        with open(os.path.join(tmp, "perm_demo.json"), "w", encoding="utf-8") as f:
            json.dump(
                {"prediction_target_columns": ["future_ltp_5m", "label_up_5m"]},
                f,
            )
        register_dataset(tmp, path, name="perm_demo")
        run = ensure_analysis_run(tmp, "perm_demo")
        model_name = "Tiny_Perm_XGB"
        _train_tiny_xgb(tmp, features, model_name)
        ds = {"path": path, "dataset_id": "perm_demo", "name": "perm_demo"}
        return run["run_id"], ds, model_name

    def test_interpret_bands(self) -> None:
        self.assertEqual(interpret_permutation(99), "Critical")
        self.assertEqual(interpret_permutation(10), "Negligible")

    def test_compute_rank_reuse_and_profiles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_id, ds, model = self._env(tmp)
            progress: list[dict] = []

            out = run_permutation_importance(
                tmp,
                run_id,
                ds,
                model,
                "future_ltp_5m",
                sample_size=80,
                progress=lambda info: progress.append(dict(info)),
            )
            self.assertFalse(out["reused"])
            self.assertFalse(out["cancelled"])
            self.assertIn("Permutation done", out["message"])
            self.assertTrue(progress)
            self.assertTrue(perm_already_complete(tmp, run_id, model, "future_ltp_5m"))

            rows = load_permutation_results(tmp, run_id, model, "future_ltp_5m")
            self.assertEqual(len(rows), 3)
            self.assertEqual(rows[0]["importance_rank"], 1)
            self.assertGreaterEqual(float(rows[0]["delta_rmse"] or 0), 0.0)
            names = {r["feature_name"] for r in rows}
            self.assertIn("spot", names)
            self.assertNotIn("future_ltp_5m", names)
            self.assertNotIn("label_up_5m", names)

            out2 = run_permutation_importance(
                tmp, run_id, ds, model, "future_ltp_5m", sample_size=80
            )
            self.assertTrue(out2["reused"])

            build_feature_profiles(tmp, run_id, ds)
            top = rows[0]["feature_name"]
            prof = load_feature_profile(tmp, run_id, top)
            self.assertIsNotNone(prof)
            assert prof is not None
            self.assertIsNotNone(prof.get("permutation_importance"))
            self.assertEqual(int(prof.get("permutation_rank") or 0), 1)

            card = load_feature_scorecard(tmp, run_id)
            perm_rows = [
                r for r in card if r.get("permutation_importance") is not None
            ]
            self.assertGreaterEqual(len(perm_rows), 3)

    def test_cancel_and_resume(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_id, ds, model = self._env(tmp)
            token = CancelToken()
            # Cancel immediately after first progress tick with a feature
            seen = {"n": 0}

            def _prog(info: dict) -> None:
                if info.get("feature"):
                    seen["n"] += 1
                    if seen["n"] >= 1:
                        token.cancel()

            out = run_permutation_importance(
                tmp,
                run_id,
                ds,
                model,
                "future_ltp_5m",
                sample_size=80,
                progress=_prog,
                cancel=token,
            )
            self.assertTrue(out["cancelled"])
            self.assertGreaterEqual(int(out["features"]), 1)
            self.assertFalse(
                perm_already_complete(tmp, run_id, model, "future_ltp_5m")
            )

            # Resume completes remaining features
            out2 = run_permutation_importance(
                tmp, run_id, ds, model, "future_ltp_5m", sample_size=80
            )
            self.assertFalse(out2["cancelled"])
            self.assertTrue(perm_already_complete(tmp, run_id, model, "future_ltp_5m"))
            rows = load_permutation_results(tmp, run_id, model, "future_ltp_5m")
            self.assertEqual(len(rows), 3)


if __name__ == "__main__":
    unittest.main()
