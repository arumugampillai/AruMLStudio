"""Tests for Analysis Lab correlation engine."""

from __future__ import annotations

import os
import tempfile
import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.analysis_correlation import (
    load_clusters,
    load_correlation_summary,
    load_matrix_slice,
    load_top_pairs,
    run_correlation_analysis,
)
from chain_replay_ml.dataset_builder.analysis_lab_store import (
    ensure_analysis_run,
    register_dataset,
)


class AnalysisCorrelationTests(unittest.TestCase):
    def test_run_correlation_builds_summary_pairs_clusters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # Highly correlated price family + orthogonal noise
            n = 200
            spot = pd.Series(range(n), dtype=float)
            df = pd.DataFrame(
                {
                    "trading_day": ["2026-07-24"] * n,
                    "spot": spot,
                    "spot_ema9": spot * 1.001,
                    "spot_ema20": spot * 1.002,
                    "noise_a": pd.Series(range(n, 0, -1), dtype=float),
                    "noise_b": (spot * 0.01).radd(50.0),
                }
            )
            # Make noise_b less correlated
            import numpy as np

            rng = np.random.default_rng(0)
            df["noise_b"] = rng.normal(size=n)

            path = os.path.join(tmp, "corr_demo.parquet")
            df.to_parquet(path, index=False)
            register_dataset(tmp, path, name="corr_demo")
            run = ensure_analysis_run(tmp, "corr_demo")
            summary = run_correlation_analysis(
                tmp, run["run_id"], {"path": path, "dataset_id": "corr_demo"}
            )
            self.assertGreaterEqual(int(summary["features_analysed"]), 4)
            self.assertGreater(int(summary["pairs"]), 0)
            self.assertGreaterEqual(int(summary["pairs_ge_095"]), 1)

            loaded = load_correlation_summary(tmp, run["run_id"])
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(
                int(loaded["features_analysed"]), int(summary["features_analysed"])
            )

            pairs = load_top_pairs(tmp, run["run_id"], min_abs=0.95, limit=20)
            self.assertTrue(pairs)
            self.assertGreaterEqual(abs(float(pairs[0]["correlation"])), 0.95)

            clusters = load_clusters(tmp, run["run_id"])
            multi = [c for c in clusters if int(c["size"]) > 1]
            self.assertTrue(multi)
            # Price family should group spot / emas
            members = set()
            for c in multi:
                members |= set(c["members"])
            self.assertTrue({"spot", "spot_ema9", "spot_ema20"} <= members)

            mat = load_matrix_slice(tmp, run["run_id"], ["spot", "spot_ema9"])
            self.assertAlmostEqual(mat["spot"]["spot"], 1.0)
            self.assertGreater(abs(mat["spot"]["spot_ema9"]), 0.95)

    def test_day_constant_features_excluded_from_pairs(self) -> None:
        """Day-level constants must not appear as |r|=1.0 across days."""
        from chain_replay_ml.dataset_builder.analysis_correlation import (
            compute_correlation_frame,
            _pair_rows,
        )

        with tempfile.TemporaryDirectory() as tmp:
            n = 100
            df = pd.DataFrame(
                {
                    "trading_day": ["2026-07-23"] * n + ["2026-07-24"] * n,
                    # Constant within each day — spurious pooled r=1.0 without demean.
                    "days_to_expiry": [4.0] * n + [5.0] * n,
                    "spot_low": [23606.3] * n + [23807.2] * n,
                    # Real within-day signal
                    "noise": list(range(n)) + list(range(n)),
                    "noise2": [x * 1.001 for x in list(range(n)) + list(range(n))],
                }
            )
            path = os.path.join(tmp, "day_const.parquet")
            df.to_parquet(path, index=False)
            corr, features = compute_correlation_frame(path)
            self.assertNotIn("days_to_expiry", features)
            self.assertNotIn("spot_low", features)
            excluded = list(corr.attrs.get("excluded_constant_features") or [])
            self.assertIn("days_to_expiry", excluded)
            self.assertIn("spot_low", excluded)
            pairs = _pair_rows(corr)
            names = {(a, b) for a, b, _ in pairs} | {(b, a) for a, b, _ in pairs}
            self.assertNotIn(("days_to_expiry", "spot_low"), names)

    def test_constant_and_day_constant_helpers_match_mask(self) -> None:
        """Vectorized screens keep the same keep/exclude set as per-column masks."""
        import numpy as np

        from chain_replay_ml.dataset_builder.analysis_correlation import (
            _constant_feature_columns,
            _day_constant_feature_columns,
            _zero_variance_mask,
        )

        n = 80
        days = pd.Series(["d1"] * (n // 2) + ["d2"] * (n // 2))
        num = pd.DataFrame(
            {
                "const": [1.0] * n,
                "day_const": [10.0] * (n // 2) + [20.0] * (n // 2),
                "signal": np.arange(n, dtype=float),
                "signal2": np.arange(n, dtype=float) * 1.01,
            }
        )
        keep, excluded = _constant_feature_columns(num)
        legacy_keep = [c for c in num.columns if not _zero_variance_mask(num[c])]
        legacy_excl = [c for c in num.columns if _zero_variance_mask(num[c])]
        self.assertEqual(set(keep), set(legacy_keep))
        self.assertEqual(set(excluded), set(legacy_excl))
        self.assertIn("const", excluded)
        self.assertIn("day_const", keep)  # varies across days; not globally constant

        keep2, day_excl = _day_constant_feature_columns(num, keep, days)
        self.assertIn("day_const", day_excl)
        self.assertNotIn("day_const", keep2)
        self.assertIn("signal", keep2)

    def test_pair_rows_matches_upper_triangle(self) -> None:
        from chain_replay_ml.dataset_builder.analysis_correlation import _pair_rows

        corr = pd.DataFrame(
            [[1.0, 0.5, float("nan")], [0.5, 1.0, -0.25], [float("nan"), -0.25, 1.0]],
            index=["a", "b", "c"],
            columns=["a", "b", "c"],
        )
        pairs = {(a, b, round(r, 6)) for a, b, r in _pair_rows(corr)}
        self.assertEqual(pairs, {("a", "b", 0.5), ("b", "c", -0.25)})

    def test_correlation_samples_without_full_frame_default(self) -> None:
        """Default max_rows must cap rows so large parquets stay memory-safe."""
        from chain_replay_ml.dataset_builder.analysis_correlation import (
            DEFAULT_CORR_MAX_ROWS,
            compute_correlation_frame,
        )

        with tempfile.TemporaryDirectory() as tmp:
            n = 5_000
            df = pd.DataFrame(
                {
                    "trading_day": ["2026-07-24"] * n,
                    "spot": list(range(n)),
                    "spot_ema9": [x * 1.001 for x in range(n)],
                    "noise": list(range(n, 0, -1)),
                }
            )
            path = os.path.join(tmp, "big.parquet")
            df.to_parquet(path, index=False)
            corr, features = compute_correlation_frame(path, max_rows=800)
            self.assertGreaterEqual(len(features), 2)
            self.assertEqual(corr.shape[0], len(features))
            self.assertLessEqual(800, DEFAULT_CORR_MAX_ROWS)
            load = (getattr(corr, "attrs", None) or {}).get("dataset_load") or {}
            self.assertIn(load.get("backend"), ("dataset_engine", "pandas"))
            self.assertLessEqual(int(load.get("rows_returned") or 0), 800)

    def test_engine_pandas_correlation_matrix_parity(self) -> None:
        """Full-frame Engine vs Pandas Pearson matrices match within fp tolerance."""
        try:
            import duckdb  # noqa: F401
        except ImportError:
            self.skipTest("duckdb not installed")

        from chain_replay_ml.dataset_builder.analysis_correlation import (
            compare_correlation_matrices,
        )

        with tempfile.TemporaryDirectory() as tmp:
            n = 400
            spot = pd.Series(range(n), dtype=float)
            df = pd.DataFrame(
                {
                    "trading_day": ["2026-07-24"] * n,
                    "spot": spot,
                    "spot_ema9": spot * 1.001,
                    "spot_ema20": spot * 1.002,
                    "noise_a": pd.Series(range(n, 0, -1), dtype=float),
                }
            )
            path = os.path.join(tmp, "parity.parquet")
            df.to_parquet(path, index=False)
            # max_rows=None → both backends load identical full frames
            report = compare_correlation_matrices(path, max_rows=None)
            self.assertTrue(report["ok"], report)
            self.assertTrue(report["matrices_close"])
            self.assertEqual(report["engine_load"].get("backend"), "dataset_engine")
            self.assertEqual(report["pandas_load"].get("backend"), "pandas")


if __name__ == "__main__":
    unittest.main()
