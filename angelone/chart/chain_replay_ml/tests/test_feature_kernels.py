"""Equality tests for Phase 6.0 Numba feature kernels vs reference implementations."""

from __future__ import annotations

import unittest

import numpy as np

from chain_replay_ml.dataset_builder.rolling_controllers import (
    IV_GRID_STEP_SEC,
    EmaController,
    IvZscoreWindowController,
    RvController,
    StdController,
)
from chain_replay_ml.performance import runtime
from chain_replay_ml.performance.feature_kernels import (
    ema_series_python,
    iv_zscore_python,
    population_std_kernel,
    population_std_numpy,
)


def _prices(n: int = 200, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (100.0 + np.cumsum(rng.normal(0.0, 0.2, size=n))).astype(np.float64)


class PopulationStdKernelTests(unittest.TestCase):
    def test_matches_numpy_exactish(self) -> None:
        arr = _prices(20)
        ref = population_std_numpy(arr)
        got = float(population_std_kernel(arr))
        self.assertTrue(np.isclose(got, ref, rtol=1e-12, atol=1e-15), f"{got} vs {ref}")

    def test_runtime_numba_vs_numpy(self) -> None:
        arr = _prices(20)
        runtime.set_numba_enabled(False)
        a = runtime.population_std(arr)
        runtime.set_numba_enabled(True)
        b = runtime.population_std(arr)
        runtime.set_numba_enabled(None)
        self.assertTrue(np.isclose(a, b, rtol=1e-12, atol=1e-15))


class EmaKernelTests(unittest.TestCase):
    def test_ema_series_matches_python(self) -> None:
        prices = _prices(500)
        runtime.set_numba_enabled(False)
        ref = runtime.ema_series(prices, 20)
        runtime.set_numba_enabled(True)
        got = runtime.ema_series(prices, 20)
        runtime.set_numba_enabled(None)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-15)

    def test_ema_series_matches_controller(self) -> None:
        prices = _prices(300)
        ctrl = EmaController(20)
        series = []
        for i, px in enumerate(prices):
            ctrl.update(float(px), ts=float(i))
            # Controller only emits when ready; compare underlying EMA via private state.
            series.append(ctrl._ema)
        runtime.set_numba_enabled(True)
        got = runtime.ema_series(prices, 20)
        runtime.set_numba_enabled(None)
        np.testing.assert_allclose(got, np.asarray(series, dtype=np.float64), rtol=1e-12, atol=1e-15)

    def test_python_reference_matches_numpy_path(self) -> None:
        prices = _prices(100)
        a = ema_series_python(prices, 9)
        runtime.set_numba_enabled(False)
        b = runtime.ema_series(prices, 9)
        runtime.set_numba_enabled(None)
        np.testing.assert_array_equal(a, b)


class IvZscoreKernelTests(unittest.TestCase):
    def test_matches_python_reference(self) -> None:
        priors = [0.12, 0.13, 0.11, 0.14, 0.125]
        iv = 0.15
        ref = iv_zscore_python(priors, iv)
        runtime.set_numba_enabled(True)
        got = runtime.iv_zscore(priors, iv)
        runtime.set_numba_enabled(None)
        self.assertTrue(np.isclose(got, ref, rtol=1e-12, atol=1e-15))

    def test_empty_priors_zero(self) -> None:
        self.assertEqual(runtime.iv_zscore([], 0.1), 0.0)


class StdControllerParityTests(unittest.TestCase):
    def test_numba_on_off_identical(self) -> None:
        prices = _prices(80)
        runtime.set_numba_enabled(False)
        off = StdController(20)
        off_vals = []
        for i, px in enumerate(prices):
            off.update(float(px), ts=float(i))
            off_vals.append(off.value())

        runtime.set_numba_enabled(True)
        on = StdController(20)
        on_vals = []
        for i, px in enumerate(prices):
            on.update(float(px), ts=float(i))
            on_vals.append(on.value())
        runtime.set_numba_enabled(None)

        for a, b in zip(off_vals, on_vals):
            if a is None and b is None:
                continue
            self.assertIsNotNone(a)
            self.assertIsNotNone(b)
            self.assertTrue(np.isclose(a, b, rtol=1e-12, atol=1e-15), f"{a} vs {b}")

    def test_matches_numpy_std(self) -> None:
        prices = list(_prices(20))
        ctrl = StdController(20)
        for i, px in enumerate(prices):
            ctrl.update(float(px), ts=float(i))
        expected = float(np.std(prices, ddof=0))
        self.assertTrue(np.isclose(ctrl.value(), expected, rtol=1e-12, atol=1e-15))


class RvControllerParityTests(unittest.TestCase):
    def test_numba_on_off_identical(self) -> None:
        prices = np.abs(_prices(120)) + 1.0
        runtime.set_numba_enabled(False)
        off = RvController(30)
        off_vals = []
        for i, px in enumerate(prices):
            off.update(float(px), ts=float(i))
            off_vals.append(off.value())

        runtime.set_numba_enabled(True)
        on = RvController(30)
        on_vals = []
        for i, px in enumerate(prices):
            on.update(float(px), ts=float(i))
            on_vals.append(on.value())
        runtime.set_numba_enabled(None)

        for a, b in zip(off_vals, on_vals):
            if a is None and b is None:
                continue
            self.assertTrue(np.isclose(a, b, rtol=1e-12, atol=1e-15), f"{a} vs {b}")


class IvZscoreControllerParityTests(unittest.TestCase):
    def test_numba_on_off_identical(self) -> None:
        rng = np.random.default_rng(3)
        ivs = (0.12 + rng.normal(0.0, 0.008, size=400)).astype(np.float64)
        warmup = int(300.0 / IV_GRID_STEP_SEC)

        runtime.set_numba_enabled(False)
        off = IvZscoreWindowController(300.0, warmup)
        off_vals = []
        for i, iv in enumerate(ivs):
            off.update(float(iv), ts=float(i) * IV_GRID_STEP_SEC)
            off_vals.append(off.value())

        runtime.set_numba_enabled(True)
        on = IvZscoreWindowController(300.0, warmup)
        on_vals = []
        for i, iv in enumerate(ivs):
            on.update(float(iv), ts=float(i) * IV_GRID_STEP_SEC)
            on_vals.append(on.value())
        runtime.set_numba_enabled(None)

        for a, b in zip(off_vals, on_vals):
            if a is None and b is None:
                continue
            # Two-pass mean/var in Numba vs pure Python can differ by ~1 ULP.
            self.assertTrue(np.isclose(a, b, rtol=1e-12, atol=1e-14), f"{a} vs {b}")


class RollingMeanStdTests(unittest.TestCase):
    def test_matches_numpy_windows(self) -> None:
        arr = _prices(250)
        window = 20
        runtime.set_numba_enabled(False)
        m0, s0 = runtime.rolling_mean_std(arr, window)
        runtime.set_numba_enabled(True)
        m1, s1 = runtime.rolling_mean_std(arr, window)
        runtime.set_numba_enabled(None)
        np.testing.assert_allclose(m0, m1, rtol=1e-12, atol=1e-15, equal_nan=True)
        np.testing.assert_allclose(s0, s1, rtol=1e-12, atol=1e-15, equal_nan=True)


class RuntimeFlagTests(unittest.TestCase):
    def test_numba_available(self) -> None:
        # Soft assert: package must not crash; prefer True in CI with numba installed.
        self.assertIsInstance(runtime.numba_available(), bool)

    def test_warmup_alias_and_stats(self) -> None:
        from chain_replay_ml.performance import warmup_kernels

        runtime.reset_perf_counters_for_tests()
        timings = warmup_kernels(verbose=False)
        self.assertIsInstance(timings, dict)
        stats = runtime.performance_stats()
        self.assertIn("kernel_hits", stats)
        self.assertIn("cache_hits", stats)
        self.assertTrue(stats["warmed"])

    def test_regression_helper_against_baseline(self) -> None:
        from chain_replay_ml.performance.benchmark import check_regression

        # Synthetic report at exactly the baseline rate → pass.
        report = {
            "suite": {
                "optimized": {
                    "rows_per_sec": 11301.38,
                    "features_per_sec": 2328083.36,
                }
            }
        }
        result = check_regression(report)
        self.assertTrue(result["ok"])

        # 25% drop → fail under 20% limit.
        report["suite"]["optimized"]["rows_per_sec"] = 11301.38 * 0.75
        result = check_regression(report)
        self.assertFalse(result["ok"])

    def test_dashboard_format(self) -> None:
        from chain_replay_ml.performance.dashboard import (
            build_dashboard,
            format_dashboard_text,
        )

        text = format_dashboard_text(
            build_dashboard(
                report={
                    "rows": 1000,
                    "features_assumed": 206,
                    "numba_available": True,
                    "compile_overhead_sec": {"population_std": 0.1},
                    "suite": {
                        "optimized": {
                            "label": "controller_suite_numba",
                            "rows_per_sec": 1000.0,
                            "features_per_sec": 206000.0,
                            "rows": 1000,
                            "features": 206,
                        }
                    },
                }
            )
        )
        self.assertIn("Feature Engine Performance", text)
        self.assertIn("Rows/sec", text)
        self.assertIn("Numba Status", text)

    def tearDown(self) -> None:
        runtime.set_numba_enabled(None)


if __name__ == "__main__":
    unittest.main()
