"""Multi-stage same-id transform params must not clobber each other."""

from __future__ import annotations

import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.transformations.pipeline import (
    run_transformation_pipeline,
)
from chain_replay_ml.dataset_builder.transformations.time_shift import (
    resolve_transform_params,
)


class ResolveTransformParamsTests(unittest.TestCase):
    def test_instance_params_not_overwritten_by_first_config_entry(self) -> None:
        cfg = {
            "transformations": [
                {
                    "id": "difference",
                    "enabled": True,
                    "params": {
                        "features": ["ltp_to_spot_ratio"],
                        "horizons": [{"seconds": 60, "suffix": "1m"}],
                    },
                },
                {
                    "id": "difference",
                    "enabled": True,
                    "params": {
                        "features": ["ltp"],
                        "horizons": [{"seconds": 3, "column": "ltp_step"}],
                    },
                },
            ],
        }
        inst = {
            "features": ["ltp"],
            "horizons": [{"seconds": 3, "column": "ltp_step"}],
            "partition_by": ["trading_day", "token"],
            "sample_interval_sec": 3.0,
        }
        got = resolve_transform_params("difference", inst, cfg)
        self.assertEqual(got["features"], ["ltp"])
        self.assertEqual(got["horizons"][0].get("column"), "ltp_step")

    def test_pipeline_creates_ltp_step_before_interaction(self) -> None:
        rows = []
        for i in range(8):
            rows.append({
                "trading_day": "2024-01-02",
                "token": "T1",
                "ltp": 100.0 + i,
                "bid_ask_spread": 0.5,
            })
        df = pd.DataFrame(rows)
        cfg = {
            "transformation_pipeline_version": 1,
            "transformations": [
                {
                    "id": "difference",
                    "enabled": True,
                    "order": 20,
                    "params": {
                        "features": ["a_unused"],
                        "horizons": [{"seconds": 3, "suffix": "3s"}],
                        "partition_by": ["trading_day", "token"],
                        "sample_interval_sec": 3.0,
                    },
                },
                {
                    "id": "difference",
                    "enabled": True,
                    "order": 20,
                    "params": {
                        "features": ["ltp"],
                        "horizons": [{"seconds": 3, "column": "ltp_step"}],
                        "partition_by": ["trading_day", "token"],
                        "sample_interval_sec": 3.0,
                    },
                },
                {
                    "id": "interaction",
                    "enabled": True,
                    "order": 50,
                    "params": {
                        "pairs": [{
                            "left": "ltp_step",
                            "right": "bid_ask_spread",
                            "op": "divide",
                            "output": "ltp_step_div_bid_ask_spread",
                            "eps": 1e-9,
                        }],
                        "fail_on_duplicate_output": False,
                        "overwrite": True,
                    },
                },
            ],
        }
        # First difference needs a dummy column so it does not fail hard.
        df["a_unused"] = df["ltp"]
        pipe = run_transformation_pipeline(df, cfg)
        self.assertIn("ltp_step", pipe.frame.columns)
        self.assertIn("ltp_step_div_bid_ask_spread", pipe.frame.columns)


if __name__ == "__main__":
    unittest.main()
