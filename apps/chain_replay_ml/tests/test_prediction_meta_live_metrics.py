"""Tests for live dashboard metrics."""

from __future__ import annotations

import unittest

from chain_replay_ml.prediction_meta.live_metrics import LiveMetricsTracker


class TestLiveMetrics(unittest.TestCase):
    def test_eta_and_throughput(self) -> None:
        live = LiveMetricsTracker(models_per_row=11, rows_at_start=0, rows_done=0, rows_total=1000)
        live.update_batch(
            batch_rows=100,
            feature_ms=27.0,
            prediction_ms=7100.0,
            sqlite_ms=340.0,
            feature_valid_pct=96.8,
            rows_done=100,
            batch_stats={
                "predictions_ok": 1100,
                "predictions_failed": 0,
                "skipped_rows": 0,
                "failed_model_rows": 0,
                "outcome_completed": 90,
                "outcome_pending": 10,
                "agreement_values": [0.874],
                "spread_values": [4.31],
                "direction_values": [1.0, 0.0],
            },
        )
        snap = live.snapshot()
        self.assertEqual(snap["rows_done"], 100)
        self.assertGreater(snap["rows_per_sec"], 0)
        self.assertGreater(snap["predictions_per_sec"], snap["rows_per_sec"])
        self.assertAlmostEqual(snap["quality"]["avg_agreement_pct"], 87.4, places=1)
        self.assertEqual(snap["quality"]["avg_spread"], 4.31)


if __name__ == "__main__":
    unittest.main()
