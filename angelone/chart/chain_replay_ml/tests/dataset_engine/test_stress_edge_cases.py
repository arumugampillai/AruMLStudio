"""Unit tests for Dataset Engine stress / edge-case builder (small scale)."""

from __future__ import annotations

import unittest

from chain_replay_ml.tests.dataset_engine._fixtures import require_duckdb


class DatasetEngineStressTests(unittest.TestCase):
    def test_small_edge_case_parity(self) -> None:
        require_duckdb()
        from chain_replay_ml.dataset_engine.stress_test import run_case

        report = run_case(n_rows=5_000, n_features=50)
        self.assertTrue(report["ok"], report)

    def test_mixed_dtypes_and_missings_present(self) -> None:
        from chain_replay_ml.dataset_engine.stress_test import build_stress_frame

        df = build_stress_frame(n_rows=200, n_features=20)
        self.assertIn("const_f32", df.columns)
        self.assertIn("high_card", df.columns)
        self.assertTrue(any(c.startswith("f32_") for c in df.columns))
        self.assertTrue(df[[c for c in df.columns if c.startswith("f32_")][0]].isna().any())


if __name__ == "__main__":
    unittest.main()
