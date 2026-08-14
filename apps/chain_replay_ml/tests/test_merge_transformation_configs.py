"""Tests for transformation config merge helper."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.pipeline_features_config import (
    build_pipeline_features_transformation_config,
)
from chain_replay_ml.dataset_builder.transformations.config import (
    merge_transformation_configs,
    prune_transformation_config_for_interval,
)


class MergeTransformationConfigsTests(unittest.TestCase):
    def test_appends_enabled_stages_after_base(self) -> None:
        base = {
            "transformations": [
                {"id": "lag", "enabled": True, "order": 10, "params": {"features": ["a"]}},
            ],
        }
        extra = {
            "transformations": [
                {"id": "return", "enabled": True, "order": 5, "params": {"features": ["b"]}},
                {"id": "math", "enabled": False, "order": 6, "params": {"features": ["c"]}},
            ],
        }
        merged = merge_transformation_configs(base, extra)
        stages = merged.get("transformations") or []
        self.assertEqual(len(stages), 2)
        self.assertEqual(stages[0]["id"], "lag")
        self.assertEqual(stages[1]["id"], "return")
        self.assertGreater(int(stages[1]["order"]), int(stages[0]["order"]))

    def test_prune_keeps_rolling_ohlc_string_outputs_at_6s(self) -> None:
        cfg = build_pipeline_features_transformation_config(sample_interval_sec=6.0)
        pruned = prune_transformation_config_for_interval(cfg, 6.0)
        ohlc = [
            st for st in pruned.get("transformations") or []
            if str((st or {}).get("id") or "") == "rolling_ohlc"
        ]
        self.assertEqual(len(ohlc), 1)
        params = ohlc[0].get("params") or {}
        self.assertIn("dist_high_pct", params.get("outputs") or [])
        self.assertIn("spot_dist_high_5m_pct", (params.get("column_map") or {}).values())

    def test_opt_volume_flow_15s_horizon_compatible_at_6s(self) -> None:
        cfg = build_pipeline_features_transformation_config(sample_interval_sec=6.0)
        pruned = prune_transformation_config_for_interval(cfg, 6.0)
        clip = [
            st for st in pruned.get("transformations") or []
            if str((st or {}).get("id") or "") == "difference_clip"
        ]
        self.assertEqual(len(clip), 1)
        horizons = (clip[0].get("params") or {}).get("horizons") or []
        flow_15 = [
            h for h in horizons
            if str((h or {}).get("column") or "") == "opt_volume_flow_15s"
        ]
        self.assertEqual(len(flow_15), 1)
        self.assertEqual(float(flow_15[0]["seconds"]), 18.0)

    def test_prune_drops_15s_horizons_at_6s_interval(self) -> None:
        cfg = build_pipeline_features_transformation_config(sample_interval_sec=6.0)
        pruned = prune_transformation_config_for_interval(cfg, 6.0)
        for stage in pruned.get("transformations") or []:
            params = (stage or {}).get("params") or {}
            for key in ("horizons", "windows", "periods"):
                for item in params.get(key) or []:
                    if isinstance(item, dict) and item.get("seconds") is not None:
                        sec = float(item["seconds"])
                        if sec > 0:
                            self.assertEqual(sec % 6.0, 0.0, msg=item)


if __name__ == "__main__":
    unittest.main()
