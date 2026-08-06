"""Tests for build summary preview and metadata."""

from __future__ import annotations

import unittest


class BuildSummaryTests(unittest.TestCase):
    def test_build_summary_metadata(self) -> None:
        from chain_replay_ml.dataset_builder.build_summary import (
            build_summary_metadata,
            build_summary_preview,
        )

        meta = build_summary_metadata(
            feature_names=["spot", "ltp_ema20_to_ltp_ratio"],
            sampling_interval_sec=3.0,
            strike_selection={"mode": "atm_band", "atmBand": 10},
            gap_policy={"preset": "20", "gapMaxSec": 20.0},
            prediction_targets={"horizonsSec": [3, 60, 300]},
        )
        self.assertEqual(meta["sampling_interval_sec"], 3.0)
        self.assertEqual(meta["sampling_label"], "3s")
        self.assertEqual(meta["strike_selection"]["mode"], "ATM_BAND")
        self.assertEqual(meta["gap_policy"]["gapMaxSec"], 20.0)
        self.assertEqual(meta["prediction_targets"]["labels"], ["3s", "1m", "5m"])
        self.assertEqual(meta["feature_count"], 2)
        self.assertEqual(meta["target_count"], 3)

        preview = build_summary_preview(
            ["spot", "ltp_ema20_to_ltp_ratio"],
            sampling_interval_sec=3.0,
            strike_selection={"mode": "atm_band", "atmBand": 10},
            gap_policy={"preset": "20", "gapMaxSec": 20.0},
            prediction_targets={"horizonsSec": [3, 60, 300]},
        )
        self.assertIn("build_config", preview)
        self.assertIn("build_summary_metadata", preview)
        self.assertEqual(preview["build_config"]["strike_label"], "ATM ±10")
        self.assertEqual(preview["build_config"]["gap_label"], "20s")
        self.assertIn("3s", preview["build_config"]["target_labels_text"])


if __name__ == "__main__":
    unittest.main()
