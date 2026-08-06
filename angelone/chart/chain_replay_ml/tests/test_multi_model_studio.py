"""Unit tests for Multi-model Feature Studio (join-only fixtures)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest


class MultiModelStudioTests(unittest.TestCase):
    def _write_studio(
        self, pkg: str, dirname: str, rows: list[dict]
    ) -> None:
        path = os.path.join(pkg, dirname)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "comparison.json"), "w", encoding="utf-8") as fh:
            json.dump({"rows": rows}, fh)
        with open(os.path.join(path, "run_meta.json"), "w", encoding="utf-8") as fh:
            json.dump({"ok": True}, fh)

    def _pkg(self, data_dir: str, name: str) -> str:
        from chain_replay_ml.training.paths import model_package_dir, safe_model_name

        pkg = model_package_dir(data_dir, safe_model_name(name))
        os.makedirs(pkg, exist_ok=True)
        with open(os.path.join(pkg, "metadata.json"), "w", encoding="utf-8") as fh:
            json.dump({"model_name": safe_model_name(name)}, fh)
        return pkg

    def test_pair_dirname_order_independent(self) -> None:
        from chain_replay_ml.multi_model_studio.writer import pair_dirname

        self.assertEqual(pair_dirname("B_Model", "A_Model"), pair_dirname("A_Model", "B_Model"))

    def test_join_deltas(self) -> None:
        from chain_replay_ml.multi_model_studio import run_multi_model_studio
        from chain_replay_ml.multi_model_studio.writer import pair_dirname
        from chain_replay_ml.training.paths import safe_model_name

        with tempfile.TemporaryDirectory() as tmp:
            pkg_a = self._pkg(tmp, "Model_A")
            pkg_b = self._pkg(tmp, "Model_B")
            self._write_studio(
                pkg_a,
                "feature_importance_studio",
                [
                    {"feature": "f_a", "rank_gain": 1, "gain": 10, "shap_mean_abs": 0.5},
                    {"feature": "f_b", "rank_gain": 2, "gain": 5, "shap_mean_abs": 0.2},
                    {"feature": "f_only_a", "rank_gain": 3, "gain": 1, "shap_mean_abs": 0.01},
                ],
            )
            self._write_studio(
                pkg_b,
                "feature_importance_studio",
                [
                    {"feature": "f_a", "rank_gain": 2, "gain": 8, "shap_mean_abs": 0.4},
                    {"feature": "f_b", "rank_gain": 1, "gain": 9, "shap_mean_abs": 0.6},
                    {"feature": "f_only_b", "rank_gain": 3, "gain": 1, "shap_mean_abs": 0.01},
                ],
            )
            self._write_studio(
                pkg_a,
                "feature_drift_studio",
                [
                    {"feature": "f_a", "drift": 0.2, "risk": "medium", "risk_score": 0.05},
                    {"feature": "f_b", "drift": 0.1, "risk": "low", "risk_score": 0.01},
                ],
            )
            self._write_studio(
                pkg_b,
                "feature_drift_studio",
                [
                    {"feature": "f_a", "drift": 0.4, "risk": "high", "risk_score": 0.12},
                    {"feature": "f_b", "drift": 0.05, "risk": "low", "risk_score": 0.005},
                ],
            )
            self._write_studio(
                pkg_a,
                "feature_distribution_studio",
                [{"feature": "f_a", "null_pct": 0.0, "skew": 0.1, "mean": 1.0, "p50": 1.0}],
            )
            self._write_studio(
                pkg_b,
                "feature_distribution_studio",
                [{"feature": "f_a", "null_pct": 2.5, "skew": 0.3, "mean": 1.2, "p50": 1.1}],
            )

            result = run_multi_model_studio(
                data_dir=tmp,
                model_a="Model_A",
                model_b="Model_B",
            )
            self.assertTrue(result.ok, result.error)
            self.assertTrue(os.path.isdir(result.artifacts_dir))
            expected_pair = pair_dirname(
                safe_model_name("Model_A"), safe_model_name("Model_B")
            )
            self.assertIn(expected_pair, result.artifacts_dir)

            # reverse order → same pair dir
            result2 = run_multi_model_studio(
                data_dir=tmp,
                model_a="Model_B",
                model_b="Model_A",
            )
            self.assertTrue(result2.ok, result2.error)
            self.assertEqual(result.artifacts_dir, result2.artifacts_dir)

            row_a = next(r for r in result.comparison if r["feature"] == "f_a")
            self.assertEqual(row_a["rank_gain_a"], 1)
            self.assertEqual(row_a["rank_gain_b"], 2)
            self.assertEqual(row_a["rank_gain_delta"], 1)
            self.assertEqual(row_a["risk_b"], "high")
            self.assertAlmostEqual(float(row_a["null_pct_delta"]), 2.5)

            only_a = next(r for r in result.comparison if r["feature"] == "f_only_a")
            self.assertTrue(only_a["in_a"])
            self.assertFalse(only_a["in_b"])
            self.assertIsNone(only_a["rank_gain_b"])

            meta = result.meta
            self.assertEqual(meta["common_count"], 2)  # f_a, f_b appear on both sides somehow
            # f_a and f_b on both; f_only_a and f_only_b exclusive
            self.assertEqual(meta["only_a_count"], 1)
            self.assertEqual(meta["only_b_count"], 1)

    def test_require_importance_missing(self) -> None:
        from chain_replay_ml.multi_model_studio import run_multi_model_studio

        with tempfile.TemporaryDirectory() as tmp:
            self._pkg(tmp, "Model_A")
            pkg_b = self._pkg(tmp, "Model_B")
            self._write_studio(
                pkg_b,
                "feature_importance_studio",
                [{"feature": "f_a", "rank_gain": 1, "gain": 1}],
            )
            result = run_multi_model_studio(
                data_dir=tmp,
                model_a="Model_A",
                model_b="Model_B",
                require=("importance",),
            )
            self.assertFalse(result.ok)
            self.assertIn("importance", (result.error or "").lower())


if __name__ == "__main__":
    unittest.main()
