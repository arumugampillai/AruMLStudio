"""Tests for readiness enforcement profiler."""

from __future__ import annotations

import unittest

from chain_replay_ml.feature_policy.build_readiness import (
    enforce_readiness_on_rows,
    validate_readiness_compliance,
)
from chain_replay_ml.feature_policy.readiness_profiler import (
    compare_readiness_profiles,
    start_readiness_profiler,
    stop_readiness_profiler,
)


class ReadinessProfilerTests(unittest.TestCase):
    def test_per_timestamp_cache_reduces_is_ready_calls(self) -> None:
        feat = "ltp_ema20_to_ltp_ratio"
        rows = [
            {"trading_day": "d", "timestamp": float(i * 10), feat: 1.0 + i * 0.01}
            for i in range(25)
        ]
        for _ in range(3):
            batch = [dict(r) for r in rows]
            start_readiness_profiler(gap_max_sec=30.0, feature_count=1, row_count=len(batch))
            enforce_readiness_on_rows(
                batch,
                feature_names=[feat],
                sampling_interval_sec=10.0,
                gap_max_sec=30.0,
                readiness_profile=True,
            )
            stats = stop_readiness_profiler()
            assert stats is not None
        calls = int((stats.function_calls or {}).get("is_ready", 0))
        self.assertLessEqual(calls, 30)

    def test_compare_readiness_profiles_delta(self) -> None:
        off = {
            "total_wall_sec": 0.004,
            "by_function": [{"function": "is_ready", "time_sec": 0.003, "calls": 100, "avg_time_us": 30}],
        }
        on = {
            "total_wall_sec": 7.67,
            "by_function": [{"function": "is_ready", "time_sec": 6.5, "calls": 1_000_000, "avg_time_us": 6.5}],
        }
        doc = compare_readiness_profiles(off, on)
        self.assertGreater(float(doc["delta_total_sec"]), 7.0)
        self.assertEqual(doc["by_function"][0]["function"], "is_ready")


if __name__ == "__main__":
    unittest.main()
