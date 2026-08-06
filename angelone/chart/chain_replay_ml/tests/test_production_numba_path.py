"""Assert Numba is dispatched on the production Create Dataset controller path."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.rolling_controllers import (
    IV_GRID_STEP_SEC,
    IvZscoreWindowController,
    RvController,
    StdController,
    update_token_ltp_controllers,
    update_token_rv_controllers,
)
from chain_replay_ml.dataset_builder.extended_features import OptionFeatureState
from chain_replay_ml.performance import runtime
from chain_replay_ml.performance.numba_utils import reset_python_fallback_for_tests


class ProductionNumbaPathTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_python_fallback_for_tests()
        runtime.set_numba_enabled(None)
        runtime.reset_perf_counters_for_tests()

    def tearDown(self) -> None:
        runtime.set_numba_enabled(None)
        reset_python_fallback_for_tests()

    @unittest.skipUnless(runtime.numba_available(), "Numba not installed")
    def test_create_dataset_controllers_hit_numba_kernels(self) -> None:
        runtime.begin_create_dataset_session(verbose=False)
        self.assertTrue(runtime.numba_enabled())

        prices = [100.0 + i * 0.1 for i in range(60)]
        std = StdController(20)
        rv = RvController(30)
        ivz = IvZscoreWindowController(60.0, int(60.0 / IV_GRID_STEP_SEC))
        for i, px in enumerate(prices):
            ts = float(i * IV_GRID_STEP_SEC)
            std.update(px, ts=ts)
            rv.update(px + 1.0, ts=ts)
            ivz.update(0.15 + i * 0.001, ts=ts)

        opt = OptionFeatureState()
        for i, px in enumerate(prices):
            ts = float(1000 + i * IV_GRID_STEP_SEC)
            update_token_ltp_controllers(opt.controllers, px, ts=ts)
            update_token_rv_controllers(opt.controllers, px + 1.0, ts=ts)

        stats = runtime.end_create_dataset_session(verbose=False)
        self.assertEqual(stats["numba_enabled_label"], "YES")
        self.assertGreater(stats["kernel_hits"], 0)
        self.assertEqual(stats["python_fallback_hits"], 0)

    @unittest.skipUnless(runtime.numba_available(), "Numba not installed")
    def test_env_off_counts_python_path_hits(self) -> None:
        runtime.set_numba_enabled(False)
        runtime.reset_perf_counters_for_tests()
        ctrl = StdController(20)
        for i in range(25):
            ctrl.update(100.0 + i, ts=float(i))
            _ = ctrl.value()
        stats = runtime.performance_stats()
        self.assertFalse(stats["numba_enabled"])
        self.assertEqual(stats["kernel_hits"], 0)
        self.assertGreater(stats["python_fallback_hits"], 0)


if __name__ == "__main__":
    unittest.main()
