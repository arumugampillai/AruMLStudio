"""Tests for self-describing pipeline catalog and Interaction source universe."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.transformations.describe import (
    MASTER_STAGE_ID,
    describe_pipeline_stages,
)
from chain_replay_ml.dataset_builder.transformations.interaction_ui import (
    available_interaction_features,
    available_interaction_features_from_config,
    columns_for_interaction_source,
    interaction_source_choices,
)
from chain_replay_ml.dataset_builder.transformations.lag import LagTransformation
from chain_replay_ml.dataset_builder.transformations.rolling import RollingTransformation


class PipelineDescribeTests(unittest.TestCase):
    def test_lag_describe_lists_planned_outputs(self) -> None:
        lag = LagTransformation()
        stage = lag.describe(
            {
                "features": ["ltp"],
                "lag_seconds": [30, 60],
                "sample_interval_sec": 3,
            },
            enabled=True,
            sample_interval_sec=3,
        )
        self.assertEqual(stage.id, "lag")
        self.assertEqual(stage.order, 10)
        self.assertTrue(stage.enabled)
        self.assertEqual(
            stage.output_names,
            ["ltp_lag_30s", "ltp_lag_60s"],
        )

    def test_pipeline_stages_available_before_interaction(self) -> None:
        cfg = {
            "transformations": [
                {
                    "id": "lag",
                    "enabled": True,
                    "params": {
                        "features": ["ltp"],
                        "lag_seconds": [30],
                        "sample_interval_sec": 3,
                    },
                },
                {
                    "id": "rolling",
                    "enabled": True,
                    "params": {
                        "features": ["ltp"],
                        "windows": [20],
                        "operations": ["std"],
                    },
                },
                {
                    "id": "interaction",
                    "enabled": True,
                    "params": {
                        "pairs": [
                            {
                                "left": "ltp",
                                "right": "ltp_roll_std_20",
                                "op": "divide",
                                "output": "ltp_div_ltp_roll_std_20",
                            }
                        ]
                    },
                },
            ]
        }
        desc = describe_pipeline_stages(
            cfg,
            master_features=["ltp", "iv"],
            sample_interval_sec=3,
        )
        self.assertEqual(desc.stage(MASTER_STAGE_ID).output_names, ["ltp", "iv"])
        before = desc.available_before("interaction")
        self.assertIn("ltp", before)
        self.assertIn("ltp_lag_30s", before)
        self.assertIn("ltp_roll_std_20", before)
        self.assertNotIn("ltp_div_ltp_roll_std_20", before)

        # Derived (order 60) must not appear before Interaction.
        derived = desc.stage("derived")
        self.assertIsNotNone(derived)
        self.assertGreater(int(derived.order), 50)

        sources = interaction_source_choices(
            cfg,
            master_features=["ltp", "iv"],
            sample_interval_sec=3,
        )
        source_ids = [sid for sid, _ in sources]
        self.assertIn(MASTER_STAGE_ID, source_ids)
        self.assertIn("lag", source_ids)
        self.assertIn("rolling", source_ids)
        self.assertIn("interaction", source_ids)
        self.assertNotIn("derived", source_ids)

        roll_cols = columns_for_interaction_source(
            cfg,
            "rolling",
            master_features=["ltp", "iv"],
            sample_interval_sec=3,
        )
        self.assertEqual(roll_cols, ["ltp_roll_std_20"])

        avail = available_interaction_features_from_config(
            cfg,
            master_features=["ltp", "iv"],
            sample_interval_sec=3,
        )
        self.assertIn("ltp_roll_std_20", avail)
        self.assertIn("ltp_div_ltp_roll_std_20", avail)

    def test_rolling_describe_empty_when_incomplete(self) -> None:
        rolling = RollingTransformation()
        stage = rolling.describe({"features": ["ltp"]}, enabled=True)
        self.assertEqual(stage.output_names, [])

    def test_available_interaction_features_includes_rolling_via_kwargs(self) -> None:
        avail = available_interaction_features(
            master_features=["ltp"],
            lag_features=["ltp"],
            lag_seconds=[30],
            lag_enabled=True,
            rolling_enabled=True,
            rolling_features=["ltp"],
            rolling_windows=[20],
            rolling_operations=["mean"],
            sample_interval_sec=3,
        )
        self.assertIn("ltp_lag_30s", avail)
        self.assertIn("ltp_roll_mean_20", avail)


if __name__ == "__main__":
    unittest.main()
