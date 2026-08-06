"""Unit tests for optional GPU/CPU CorrelationEngine (no GPU required)."""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import pandas as pd

from chain_replay_ml.analytics.correlation import (
    CorrelationEngine,
    is_gpu_available,
    resolve_backend,
)
from chain_replay_ml.analytics.correlation.cpu_engine import pearson_corr_cpu


class CorrelationEngineTests(unittest.TestCase):
    def _frame(self, n: int = 200) -> pd.DataFrame:
        rng = np.random.default_rng(0)
        base = rng.standard_normal(n)
        return pd.DataFrame(
            {
                "a": base,
                "b": base * 1.001 + rng.normal(0, 0.01, n),
                "c": rng.standard_normal(n),
                "d": np.linspace(0, 1, n),
            }
        )

    def test_selects_cpu_when_gpu_unavailable(self) -> None:
        with mock.patch(
            "chain_replay_ml.analytics.correlation.engine.is_gpu_available",
            return_value=False,
        ):
            self.assertEqual(resolve_backend("auto"), "cpu")
            self.assertEqual(resolve_backend("gpu"), "cpu")
            self.assertEqual(resolve_backend("cpu"), "cpu")
            res = CorrelationEngine(preference="auto").compute(self._frame())
            self.assertEqual(res.backend_used, "cpu")
            self.assertFalse(res.gpu_available)

    def test_selects_gpu_when_available_and_auto(self) -> None:
        with mock.patch(
            "chain_replay_ml.analytics.correlation.engine.is_gpu_available",
            return_value=True,
        ):
            self.assertEqual(resolve_backend("auto"), "gpu")
            self.assertEqual(resolve_backend("gpu"), "gpu")
            self.assertEqual(resolve_backend("cpu"), "cpu")

    def test_cpu_path_matches_pandas_corr(self) -> None:
        frame = self._frame()
        expected = frame.corr(method="pearson", min_periods=2)
        got, timing = pearson_corr_cpu(frame, min_periods=2)
        self.assertIsNotNone(timing.cpu_compute_sec)
        np.testing.assert_allclose(
            got.to_numpy(dtype=float),
            expected.to_numpy(dtype=float),
            rtol=0.0,
            atol=0.0,
            equal_nan=True,
        )
        self.assertEqual(list(got.columns), list(expected.columns))
        self.assertEqual(list(got.index), list(expected.index))

    def test_engine_cpu_matches_existing_analysis_path(self) -> None:
        frame = self._frame()
        legacy = frame.corr(method="pearson", min_periods=2)
        res = CorrelationEngine(preference="cpu").compute(frame)
        self.assertEqual(res.backend_used, "cpu")
        np.testing.assert_allclose(
            res.matrix.to_numpy(dtype=float),
            legacy.to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-15,
            equal_nan=True,
        )

    def test_fallback_on_gpu_error(self) -> None:
        frame = self._frame()
        with mock.patch(
            "chain_replay_ml.analytics.correlation.engine.is_gpu_available",
            return_value=True,
        ), mock.patch(
            "chain_replay_ml.analytics.correlation.gpu_engine.pearson_corr_gpu",
            side_effect=MemoryError("simulated OOM"),
        ):
            res = CorrelationEngine(preference="gpu").compute(frame)
        self.assertEqual(res.backend_used, "cpu")
        self.assertIsNotNone(res.timing.fallback_reason)
        self.assertIn("MemoryError", str(res.timing.fallback_reason))
        legacy = frame.corr(method="pearson", min_periods=2)
        np.testing.assert_allclose(
            res.matrix.to_numpy(dtype=float),
            legacy.to_numpy(dtype=float),
            equal_nan=True,
        )

    def test_gpu_vs_cpu_within_tolerance_or_skip(self) -> None:
        if not is_gpu_available():
            self.skipTest("cudf/cupy not available")
        frame = self._frame(5_000)
        cpu = CorrelationEngine(preference="cpu").compute(frame)
        gpu = CorrelationEngine(preference="gpu").compute(frame)
        self.assertEqual(gpu.backend_used, "gpu")
        self.assertEqual(list(gpu.matrix.columns), list(cpu.matrix.columns))
        self.assertEqual(list(gpu.matrix.index), list(cpu.matrix.index))
        np.testing.assert_allclose(
            gpu.matrix.to_numpy(dtype=float),
            cpu.matrix.to_numpy(dtype=float),
            rtol=0.0,
            atol=1e-8,
            equal_nan=True,
        )

    def test_import_safe_without_cudf(self) -> None:
        # Package import must succeed even when RAPIDS is missing.
        from chain_replay_ml.analytics import correlation as corr_pkg

        self.assertTrue(hasattr(corr_pkg, "CorrelationEngine"))
        # Real probe on this machine (Windows typically False).
        self.assertIsInstance(is_gpu_available(), bool)


if __name__ == "__main__":
    unittest.main()
