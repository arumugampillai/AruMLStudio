"""Tests for day_build_info loader."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from master_dataset_tk.day_build_info import (
    format_feature_names_text,
    load_master_build_info,
)


class DayBuildInfoTests(unittest.TestCase):
    def test_load_master_build_info_merges_meta_keys(self) -> None:
        store = MagicMock()
        store.get_meta.side_effect = lambda key: {
            "build_summary": {
                "sampling_interval_sec": 10,
                "sliding_stride_sec": 5,
                "feature_window_sec": 10,
                "sampling_label": "10s",
                "sliding_stride_label": "5s",
                "feature_names": ["feat_a", "feat_b"],
                "prediction_targets": {
                    "horizons_sec": [300],
                    "labels": ["5m"],
                },
                "strike_selection": {"mode": "atm_band", "atmBand": 2},
                "gap_policy": {"enabled": True, "gapMaxSec": 20},
            },
            "master_config": {
                "market": "NIFTY",
                "atm_band": 2,
                "strike_selection": {"mode": "atm_band", "atmBand": 2},
                "gap_policy": {"enabled": True, "gapMaxSec": 20},
                "prediction_targets": {"horizons_sec": [300], "labels": ["5m"]},
            },
            "build_schema": {
                "feature_columns": ["feat_a", "feat_b"],
                "target_columns": ["target_5m"],
            },
            "dataset_configuration": {
                "lookback_policy": {"method": "nearest_snapshot"},
                "feature_groups": ["greeks", "volatility"],
            },
            "feature_policy": {
                "registry_version": "reg-v1",
                "feature_profile": "default",
            },
        }.get(key)
        store.read_master_meta_dict.return_value = {
            "market": "NIFTY",
            "builder_version": "build-1",
            "feature_registry_version": "reg-v1",
            "schema_hash": "abc123def456",
            "created_at": "2026-08-14T10:00:00+00:00",
            "updated_at": "2026-08-14T11:00:00+00:00",
        }

        info = load_master_build_info(store)
        kv = info["kv_fields"]
        self.assertEqual(kv["market"], "NIFTY")
        self.assertEqual(kv["sampling"], "10s")
        self.assertEqual(kv["sliding_stride"], "5s")
        self.assertIn("ATM", kv["strike_selection"])
        self.assertEqual(kv["feature_count"], "2")
        self.assertEqual(kv["target_count"], "1")
        self.assertEqual(kv["feature_profile"], "default")
        self.assertIn("greeks", kv["feature_groups"])
        text = format_feature_names_text(info)
        self.assertIn("feat_a", text)
        self.assertIn("target_5m", text)


if __name__ == "__main__":
    unittest.main()
