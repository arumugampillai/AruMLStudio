"""Tests for Mutual Information (Phase 2.3 Research Lab)."""

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
from chain_replay_ml.dataset_builder.analysis_mutual_information import (
    analysis_timeline,
    discover_mi_targets,
    interpret_mi,
    load_mi_results,
    mi_already_computed,
    mi_stars,
    rehydrate_mi_into_profiles,
    run_mutual_information,
)


class MutualInformationTests(unittest.TestCase):
    def _make_dataset(self, tmp: str) -> tuple[str, dict]:
        n = 200
        rng = np.random.default_rng(42)
        spot = pd.Series(np.linspace(100, 200, n)) + rng.normal(0, 0.5, n)
        # Strong predictor of future_ltp_5m
        signal = spot * 1.01
        noise = rng.normal(0, 5, n)
        future = signal + noise
        df = pd.DataFrame(
            {
                "trading_day": ["2026-07-24"] * n,
                "spot": spot,
                "spot_ema200_to_ltp_ratio": signal / spot,
                "current_iv": np.linspace(0.1, 0.25, n),
                "weekday": [i % 5 for i in range(n)],
                "noise_feat": rng.normal(size=n),
                "future_ltp_1m": future * 0.99,
                "future_ltp_5m": future,
                "label_up_5m": (future > spot).astype(int),
            }
        )
        path = os.path.join(tmp, "mi_demo.parquet")
        df.to_parquet(path, index=False)
        with open(os.path.join(tmp, "mi_demo.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "prediction_target_columns": [
                        "future_ltp_1m",
                        "future_ltp_5m",
                        "label_up_5m",
                    ]
                },
                f,
            )
        register_dataset(tmp, path, name="mi_demo")
        run = ensure_analysis_run(tmp, "mi_demo")
        ds = {"path": path, "dataset_id": "mi_demo", "name": "mi_demo"}
        return run["run_id"], ds

    def test_interpret_and_stars(self) -> None:
        self.assertEqual(interpret_mi(99), "Excellent predictor")
        self.assertEqual(interpret_mi(2), "Very weak")
        self.assertEqual(mi_stars(99), "★★★★★")
        self.assertEqual(mi_stars(None), "Pending")

    def test_discover_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, ds = self._make_dataset(tmp)
            targets = discover_mi_targets(tmp, ds)
            self.assertIn("future_ltp_5m", targets)
            self.assertIn("label_up_5m", targets)

    def test_compute_persist_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_id, ds = self._make_dataset(tmp)
            out = run_mutual_information(tmp, run_id, ds, "future_ltp_5m")
            self.assertFalse(out["reused"])
            self.assertGreater(out["features"], 3)
            self.assertTrue(mi_already_computed(tmp, run_id, "future_ltp_5m"))

            rows = load_mi_results(tmp, run_id, "future_ltp_5m")
            self.assertEqual(len(rows), out["features"])
            # Targets / labels must not appear as features
            feats = {r["feature"] for r in rows}
            self.assertNotIn("future_ltp_5m", feats)
            self.assertNotIn("label_up_5m", feats)
            # Rank 1 should be a real feature with high percentile
            self.assertEqual(rows[0]["rank"], 1)
            self.assertGreaterEqual(float(rows[0]["percentile"]), 80.0)

            out2 = run_mutual_information(tmp, run_id, ds, "future_ltp_5m")
            self.assertTrue(out2["reused"])

            statuses = {
                m["module_id"]: m["status"] for m in module_statuses(tmp, run_id)
            }
            self.assertEqual(statuses.get("mutual_information"), "completed")

    def test_profiles_and_timeline_after_mi(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_id, ds = self._make_dataset(tmp)
            run_mutual_information(tmp, run_id, ds, "future_ltp_5m")
            build_feature_profiles(tmp, run_id, ds)
            # rebuild clears then rehydrates MI
            rehydrate_mi_into_profiles(tmp, run_id, target="future_ltp_5m")

            top = load_mi_results(tmp, run_id, "future_ltp_5m", limit=1)[0]
            prof = load_feature_profile(tmp, run_id, top["feature"])
            self.assertIsNotNone(prof)
            assert prof is not None
            self.assertIsNotNone(prof.get("mi_score"))
            self.assertEqual(prof.get("mi_target"), "future_ltp_5m")

            card = load_feature_scorecard(tmp, run_id)
            mi_rows = [r for r in card if r.get("mi_score") is not None]
            self.assertGreaterEqual(len(mi_rows), 3)

            tl = analysis_timeline(
                tmp, run_id, top["feature"], mi_target="future_ltp_5m"
            )
            by_id = {s["id"]: s["state"] for s in tl}
            self.assertEqual(by_id["mutual_information"], "done")
            self.assertEqual(by_id["shap"], "pending")


if __name__ == "__main__":
    unittest.main()
