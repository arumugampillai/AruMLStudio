"""Tests for Explicit + Inherited Nullable propagation via lineage."""

from __future__ import annotations

import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.non_null_filter import apply_non_null_filter_frame
from chain_replay_ml.dataset_builder.nullable_features import (
    expand_nullable_via_lineage,
    format_nullable_classification,
    mandatory_columns_for_step2,
)
from chain_replay_ml.dataset_builder.pipeline_no_null_report import (
    build_pipeline_lineage_map,
    build_pipeline_no_null_report_text,
    validate_pipeline_lineage_parents,
)
from chain_replay_ml.dataset_builder.transformations.pipeline import describe_pipeline


def _iv_chain_config() -> dict:
    return {
        "version": 1,
        "transformations": [
            {
                "id": "rolling_statistics",
                "enabled": True,
                "params": {
                    "features": ["current_iv"],
                    "windows": [
                        {"seconds": 60, "column": "iv_zscore_1m"},
                        {"seconds": 900, "column": "iv_zscore_15m"},
                    ],
                    "stat": "zscore",
                },
            },
            {
                "id": "interaction",
                "enabled": True,
                "params": {
                    "pairs": [
                        {
                            "left": "weighted_spot_ema_to_ltp_ratio",
                            "right": "iv_zscore_15m",
                            "op": "multiply",
                            "output": "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m",
                        },
                        {
                            "left": "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m",
                            "right": "delta",
                            "op": "multiply",
                            "output": (
                                "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m_x_delta"
                            ),
                        },
                    ]
                },
            },
        ],
    }


class TestNullableInheritance(unittest.TestCase):
    def test_recursive_chain_from_current_iv(self) -> None:
        cfg = _iv_chain_config()
        lineage = build_pipeline_lineage_map(cfg)
        res = expand_nullable_via_lineage(
            lineage,
            column_names=[
                "current_iv",
                "delta",
                "weighted_spot_ema_to_ltp_ratio",
                "iv_zscore_1m",
                "iv_zscore_15m",
                "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m",
                "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m_x_delta",
                "ltp",
            ],
        )
        self.assertIn("current_iv", res.explicit)
        self.assertEqual(res.inheritance_parents("iv_zscore_15m"), ("current_iv",))
        self.assertEqual(
            res.inheritance_parents(
                "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m"
            ),
            ("iv_zscore_15m",),
        )
        self.assertEqual(
            res.inheritance_parents(
                "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m_x_delta"
            ),
            ("weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m",),
        )
        for name in (
            "iv_zscore_1m",
            "iv_zscore_15m",
            "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m",
            "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m_x_delta",
        ):
            self.assertIn(name, res.inherited, msg=name)
        self.assertNotIn("delta", res.effective)
        self.assertNotIn("ltp", res.effective)
        self.assertNotIn("weighted_spot_ema_to_ltp_ratio", res.effective)

    def test_classification_shows_inherit_trace(self) -> None:
        cfg = _iv_chain_config()
        lineage = build_pipeline_lineage_map(cfg)
        res = expand_nullable_via_lineage(lineage, column_names=None)
        text = "\n".join(format_nullable_classification(res))
        self.assertIn("Nullable Classification", text)
        self.assertIn("Registry (Explicit)", text)
        self.assertIn("Pipeline (Inherited)", text)
        self.assertIn("✓ iv_zscore_15m", text)
        self.assertIn("↳ inherits from current_iv", text)
        self.assertIn(
            "↳ inherits from weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m",
            text,
        )

    def test_step2_excludes_inherited(self) -> None:
        cfg = _iv_chain_config()
        cols = [
            "ltp",
            "current_iv",
            "iv_zscore_15m",
            "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m",
        ]
        mandatory = mandatory_columns_for_step2(
            cols,
            transformation_config=cfg,
        )
        self.assertEqual(mandatory, ["ltp"])

    def test_filter_frame_keeps_rows_with_inherited_nulls(self) -> None:
        cfg = _iv_chain_config()
        frame = pd.DataFrame(
            {
                "trading_day": ["2026-07-24"] * 4,
                "ltp": [10.0, 11.0, 12.0, 13.0],
                "current_iv": [0.1, None, 0.2, None],
                "iv_zscore_15m": [0.0, None, None, None],
                "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m": [
                    1.0,
                    None,
                    None,
                    None,
                ],
            }
        )
        result = apply_non_null_filter_frame(
            frame,
            transformation_config=cfg,
        )
        self.assertEqual(result["report"]["rows_after"], 4)
        self.assertIn("current_iv", result["report"]["nullable_explicit"])
        self.assertIn("iv_zscore_15m", result["report"]["nullable_inherited"])
        self.assertIn(
            "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m",
            result["report"]["nullable_inherited"],
        )

    def test_report_marks_inherited_not_class2(self) -> None:
        cfg = _iv_chain_config()
        df = pd.DataFrame(
            {
                "current_iv": [0.1, None, 0.2, 0.2, 0.2],
                "iv_zscore_15m": [None, None, None, 0.1, 0.2],
                "ltp_lag_9s": [None, None, 1.0, 1.0, 1.0],
            }
        )
        text = build_pipeline_no_null_report_text(
            df,
            pipeline_columns=["iv_zscore_15m", "ltp_lag_9s"],
            transformation_config=cfg,
            filter_applied=False,
        )
        self.assertIn("Nullable Classification", text)
        self.assertIn("Pipeline (Inherited)", text)
        self.assertIn("✓ iv_zscore_15m", text)
        self.assertIn("↳ inherits from current_iv", text)
        self.assertIn("Registry (Explicit)", text)
        self.assertIn("✓ current_iv", text)
        top_section = text.split("Top NEW Pipeline NULLs")[1].split("----")[0]
        self.assertNotIn("iv_zscore_15m", top_section)

    def test_lineage_gap_warning_for_missing_upstream(self) -> None:
        # Interaction references a pipeline-looking parent that was never produced.
        cfg = {
            "version": 1,
            "transformations": [
                {
                    "id": "interaction",
                    "enabled": True,
                    "params": {
                        "pairs": [
                            {
                                "left": "ltp",
                                "right": "iv_zscore_15m",
                                "op": "multiply",
                                "output": "ltp_x_iv_zscore_15m",
                            }
                        ]
                    },
                }
            ],
        }
        lineage = build_pipeline_lineage_map(cfg)
        warnings = validate_pipeline_lineage_parents(lineage, known_columns=None)
        self.assertTrue(
            any("iv_zscore_15m" in w and "ltp_x_iv_zscore_15m" in w for w in warnings),
            msg=warnings,
        )
        # With known columns that include the missing parent, no gap.
        warnings_ok = validate_pipeline_lineage_parents(
            lineage,
            known_columns=["ltp", "iv_zscore_15m"],
        )
        self.assertEqual(warnings_ok, [])

        desc = describe_pipeline(cfg)
        self.assertTrue(any("iv_zscore_15m" in w for w in desc.lineage_warnings))


if __name__ == "__main__":
    unittest.main()
