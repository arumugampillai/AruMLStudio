"""Tests for shared production build_day_rows helper."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from chain_replay_ml.dataset_builder.production_day_build import (
    build_production_day_rows,
    production_performance_debug,
)
from chain_replay_ml.feature_policy.performance_debug import PerformanceDebugLevel


class ProductionDayBuildTests(unittest.TestCase):
    def test_production_perf_is_off(self) -> None:
        perf = production_performance_debug()
        self.assertEqual(perf.level, PerformanceDebugLevel.OFF)
        self.assertFalse(perf.collect_gap_profile())
        self.assertFalse(perf.collect_readiness_profile())

    def test_build_production_day_rows_passes_off_debug(self) -> None:
        with patch("chain_replay_ml.dataset_builder.stages.build_day_rows", return_value=([], {})) as mock:
            build_production_day_rows(
                object(),
                step_sec=10,
                strike_selection={},
                horizons_sec=[300],
                enabled_groups=[],
                group_labels={},
                implemented_features=[],
                per_group_features={},
            )
        _, kwargs = mock.call_args
        perf = kwargs.get("performance_debug")
        self.assertEqual(perf.level, PerformanceDebugLevel.OFF)
        self.assertFalse(kwargs.get("gap_profile"))
        self.assertFalse(kwargs.get("readiness_profile"))
        self.assertTrue(kwargs.get("skip_readiness_compliance"))


if __name__ == "__main__":
    unittest.main()
