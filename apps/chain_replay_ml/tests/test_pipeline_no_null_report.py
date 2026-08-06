"""Unit tests for Pipeline No-Null diagnostic report."""

from __future__ import annotations

import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.pipeline_no_null_report import (
    CLASS_INHERITED,
    CLASS_MATH,
    CLASS_PIPELINE,
    CLASS_WARMUP,
    build_pipeline_lineage_map,
    build_pipeline_no_null_report_text,
)


class PipelineNoNullReportTests(unittest.TestCase):
    def test_inherited_interaction_is_class1(self) -> None:
        df = pd.DataFrame(
            {
                "ltp_ema300": [1.0, None, None, 1.0, 1.0],
                "spot_ema300": [1.0, 1.0, None, 1.0, 1.0],
                "ltp_ema300_x_spot_ema300": [1.0, None, None, 1.0, None],
            }
        )
        # Last row: both parents present but product NULL → pipeline-created
        # Rows 1-2: parent NULL → inherited
        cfg = {
            "version": 1,
            "transformations": [
                {
                    "id": "interaction",
                    "enabled": True,
                    "params": {
                        "pairs": [
                            {
                                "left": "ltp_ema300",
                                "right": "spot_ema300",
                                "op": "multiply",
                                "output": "ltp_ema300_x_spot_ema300",
                            }
                        ]
                    },
                }
            ],
        }
        lineage = build_pipeline_lineage_map(cfg)
        self.assertEqual(
            lineage["ltp_ema300_x_spot_ema300"]["parents"],
            ["ltp_ema300", "spot_ema300"],
        )

        text = build_pipeline_no_null_report_text(
            df,
            pipeline_columns=["ltp_ema300_x_spot_ema300"],
            transformation_config=cfg,
            filter_applied=False,
        )
        self.assertIn("Pipeline Stage", text)
        self.assertIn("Rows physically removed    0", text)
        self.assertIn("Would remove", text)
        self.assertIn("Pipeline NULL Summary", text)
        self.assertIn("Inherited from Registry", text)
        self.assertIn("Top NEW Pipeline NULLs", text)
        self.assertIn("ltp_ema300_x_spot_ema300", text)

    def test_zscore_lineage_and_warmup_class(self) -> None:
        df = pd.DataFrame(
            {
                "current_iv": [1.0, 1.0, 1.0, 1.0, 1.0],
                "iv_zscore_30m": [None, None, None, 0.1, 0.2],
            }
        )
        cfg = {
            "version": 1,
            "transformations": [
                {
                    "id": "rolling_statistics",
                    "enabled": True,
                    "params": {
                        "features": ["current_iv"],
                        "windows": [{"seconds": 1800, "column": "iv_zscore_30m"}],
                        "stat": "zscore",
                    },
                }
            ],
        }
        lineage = build_pipeline_lineage_map(cfg)
        self.assertEqual(lineage["iv_zscore_30m"]["parents"], ["current_iv"])

        text = build_pipeline_no_null_report_text(
            df,
            pipeline_columns=["iv_zscore_30m"],
            transformation_config=cfg,
            filter_applied=False,
        )
        # current_iv is Explicit Nullable → zscore is Inherited Nullable,
        # excluded from Step-2 (Would remove 0) and not a Class-2 bug.
        self.assertIn("Pipeline (Inherited)", text)
        self.assertIn("✓ iv_zscore_30m", text)
        self.assertIn("↳ inherits from current_iv", text)
        self.assertIn("Would remove               0", text)
        self.assertIn("iv_zscore_30m", text)

    def test_divide_by_zero_is_mathematical(self) -> None:
        df = pd.DataFrame(
            {
                "opt_rv_5m": [1.0, 1.0, 1.0, 1.0],
                "opt_rv_10m": [1.0, 0.0, 0.0, 2.0],
                "opt_rv_ratio": [1.0, None, None, 0.5],
            }
        )
        cfg = {
            "version": 1,
            "transformations": [
                {
                    "id": "interaction",
                    "enabled": True,
                    "params": {
                        "pairs": [
                            {
                                "left": "opt_rv_5m",
                                "right": "opt_rv_10m",
                                "op": "divide",
                                "output": "opt_rv_ratio",
                            }
                        ]
                    },
                }
            ],
        }
        text = build_pipeline_no_null_report_text(
            df,
            pipeline_columns=["opt_rv_ratio"],
            transformation_config=cfg,
            filter_applied=False,
        )
        self.assertIn("Mathematical", text)
        self.assertIn("opt_rv_ratio", text)

    def test_warmup_lag_expected(self) -> None:
        df = pd.DataFrame(
            {
                "ltp": [1.0, 1.0, 1.0, 1.0],
                "ltp_lag_9s": [None, None, None, 1.0],
            }
        )
        text = build_pipeline_no_null_report_text(
            df,
            pipeline_columns=["ltp_lag_9s"],
            transformation_config=None,
            filter_applied=False,
        )
        self.assertIn("Would remove", text)
        self.assertIn("ltp_lag_9s", text)
        self.assertIn("Rows physically removed    0", text)


if __name__ == "__main__":
    unittest.main()
