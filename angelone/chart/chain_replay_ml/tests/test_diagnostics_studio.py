"""Unit tests for Diagnostics Studio (join fixtures)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest


class DiagnosticsStudioTests(unittest.TestCase):
    def _write_studio(self, pkg: str, dirname: str, rows: list[dict], meta: dict | None = None) -> None:
        path = os.path.join(pkg, dirname)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "comparison.json"), "w", encoding="utf-8") as fh:
            json.dump({"rows": rows}, fh)
        with open(os.path.join(path, "run_meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta or {"ok": True}, fh)

    def _pkg(self, data_dir: str, name: str = "Diag_Model") -> tuple[str, str]:
        from chain_replay_ml.training.paths import model_package_dir, safe_model_name

        model_name = safe_model_name(name)
        pkg = model_package_dir(data_dir, model_name)
        os.makedirs(pkg, exist_ok=True)
        with open(os.path.join(pkg, "metadata.json"), "w", encoding="utf-8") as fh:
            json.dump({"model_name": model_name}, fh)
        with open(os.path.join(pkg, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "validation": {"mae": 1.0, "rmse": 1.5, "premium_mae_pct": 10.0},
                    "test": {"mae": 1.5, "rmse": 2.2, "premium_mae_pct": 18.0},
                },
                fh,
            )
        return model_name, pkg

    def test_end_to_end(self) -> None:
        from chain_replay_ml.diagnostics_studio import run_diagnostics_studio

        with tempfile.TemporaryDirectory() as tmp:
            model_name, pkg = self._pkg(tmp)
            self._write_studio(
                pkg,
                "feature_importance_studio",
                [
                    {"feature": "f_a", "rank_gain": 1, "gain": 10, "shap_mean_abs": 0.5},
                    {"feature": "f_b", "rank_gain": 2, "gain": 5, "shap_mean_abs": 0.2},
                ],
            )
            self._write_studio(
                pkg,
                "feature_distribution_studio",
                [
                    {"feature": "f_a", "null_pct": 0.0, "skew": 0.1},
                    {"feature": "f_b", "null_pct": 6.0, "skew": 2.5},
                ],
            )
            self._write_studio(
                pkg,
                "feature_drift_studio",
                [
                    {
                        "feature": "f_a",
                        "drift": 0.55,
                        "drift_pct": 40.0,
                        "risk": "high",
                        "risk_score": 62.0,
                        "wf_mean": 0.0,
                        "holdout_mean": 1.0,
                    },
                    {
                        "feature": "f_b",
                        "drift": 0.1,
                        "drift_pct": 5.0,
                        "risk": "low",
                        "risk_score": 8.0,
                        "wf_mean": 1.0,
                        "holdout_mean": 1.05,
                    },
                ],
                meta={
                    "similarity_pct": 62.0,
                    "feature_drift_pct": 38.0,
                    "target_drift_pct": 12.0,
                    "drift_scores": {"feature": 38.0, "target": 12.0, "premium": 0.0, "volatility": 0.0},
                },
            )

            result = run_diagnostics_studio(data_dir=tmp, model_name=model_name)
            self.assertTrue(result.ok, result.error)
            for name in ("summary.json", "narrative.json", "comparison.json", "run_meta.json"):
                self.assertTrue(os.path.isfile(os.path.join(result.artifacts_dir, name)), name)
            self.assertIn(result.summary.get("primary_cause"), {
                "data_drift",
                "overfitting",
                "difficult_market",
                "stable",
                "unknown",
                "insufficient_inputs",
            })
            # high feature drift → data_drift expected
            self.assertEqual(result.summary.get("primary_cause"), "data_drift")
            self.assertTrue(result.narrative)
            row_a = next(r for r in result.comparison if r["feature"] == "f_a")
            self.assertEqual(row_a["diagnostic_flag"], "high_risk")
            row_b = next(r for r in result.comparison if r["feature"] == "f_b")
            self.assertEqual(row_b["diagnostic_flag"], "high_null")
            self.assertTrue(result.summary["joins"]["importance"])
            self.assertTrue(result.summary["joins"]["metrics"])
            self.assertIsNotNone(result.summary.get("mae_pct_change"))

    def test_soft_missing_distribution(self) -> None:
        from chain_replay_ml.diagnostics_studio import run_diagnostics_studio

        with tempfile.TemporaryDirectory() as tmp:
            model_name, pkg = self._pkg(tmp)
            self._write_studio(
                pkg,
                "feature_importance_studio",
                [{"feature": "f_a", "rank_gain": 1, "gain": 1}],
            )
            self._write_studio(
                pkg,
                "feature_drift_studio",
                [{"feature": "f_a", "drift": 0.2, "risk": "low", "risk_score": 8.0}],
                meta={"similarity_pct": 80.0, "drift_scores": {"feature": 10.0}},
            )
            result = run_diagnostics_studio(data_dir=tmp, model_name=model_name)
            self.assertTrue(result.ok, result.error)
            self.assertFalse(result.summary["joins"]["distribution"])
            self.assertTrue(any("distribution" in b.lower() for b in result.narrative))


if __name__ == "__main__":
    unittest.main()
