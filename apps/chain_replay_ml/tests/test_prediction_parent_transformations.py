"""Prediction must recreate parent Feature Transformations before validation."""

from __future__ import annotations

import os
import tempfile
import unittest

import pandas as pd

from chain_replay_ml.model_lab.prediction_job_schema import (
    DAY_CP_FAILED,
    DAY_CP_PENDING,
)
from chain_replay_ml.model_lab.prediction_job_store import (
    assign_days_round_robin,
    create_job,
    update_checkpoint,
    worker_pending_days,
)
from chain_replay_ml.model_lab.prediction_transformations import (
    apply_parent_dataset_transformations,
    expand_columns_for_master_load,
    pipeline_has_enabled_transforms,
    sample_interval_sec_from_meta,
    source_columns_for_transformations,
    transformation_config_from_dataset_meta,
)


def _lag_meta() -> dict:
    return {
        "sampling": {"interval_sec": 3},
        "transformations": [
            {
                "id": "lag",
                "enabled": True,
                "params": {
                    "features": ["atm_pcr", "ltp"],
                    "lag_seconds": [30, 60],
                    "partition_by": ["trading_day", "token"],
                    "sample_interval_sec": 3,
                },
            }
        ],
    }


class PredictionTransformationAdapterTests(unittest.TestCase):
    def test_reads_parent_pipeline_and_source_columns(self) -> None:
        cfg = transformation_config_from_dataset_meta(_lag_meta())
        self.assertTrue(pipeline_has_enabled_transforms(cfg))
        sources = source_columns_for_transformations(cfg)
        self.assertIn("atm_pcr", sources)
        self.assertIn("ltp", sources)
        self.assertIn("trading_day", sources)
        self.assertIn("token", sources)
        self.assertEqual(sample_interval_sec_from_meta(_lag_meta()), 3.0)

        expanded = expand_columns_for_master_load(
            ["atm_pcr_lag_30s", "future_ltp_5m", "timestamp"],
            cfg,
        )
        self.assertIn("atm_pcr", expanded)
        self.assertIn("ltp", expanded)
        self.assertIn("atm_pcr_lag_30s", expanded)

    def test_top_level_sample_interval_is_experiment_identity(self) -> None:
        """Top-level sample_interval_sec wins so 3s vs 6s experiments stay distinct."""
        meta = {
            "sample_interval_sec": 6,
            "sampling": {"interval_sec": 3},
            "transformations": [
                {
                    "id": "rolling",
                    "enabled": True,
                    "params": {"sample_interval_sec": 3},
                }
            ],
        }
        self.assertEqual(sample_interval_sec_from_meta(meta), 6.0)

    def test_apply_shared_lag_before_feature_space_ready(self) -> None:
        """Master-shaped frame gains lag columns via shared pipeline only."""
        rows = []
        for i in range(12):
            rows.append(
                {
                    "trading_day": "2026-05-26",
                    "token": "T1",
                    "timestamp": 1_000_000.0 + i * 3.0,
                    "atm_pcr": float(i),
                    "ltp": 100.0 + i,
                }
            )
        frame = pd.DataFrame(rows)
        out = apply_parent_dataset_transformations(
            frame,
            transformation_config=transformation_config_from_dataset_meta(_lag_meta()),
            sample_interval_sec=3,
        )
        self.assertIn("atm_pcr_lag_30s", out.columns)
        self.assertIn("ltp_lag_60s", out.columns)
        # 30s / 3s = 10 rows
        self.assertTrue(pd.isna(out.loc[9, "atm_pcr_lag_30s"]))
        self.assertEqual(out.loc[10, "atm_pcr_lag_30s"], 0.0)
        self.assertEqual(out.loc[11, "atm_pcr_lag_30s"], 1.0)

    def test_no_enabled_pipeline_is_noop(self) -> None:
        frame = pd.DataFrame({"a": [1.0, 2.0]})
        out = apply_parent_dataset_transformations(
            frame,
            transformation_config={"transformations": []},
        )
        self.assertEqual(list(out.columns), ["a"])

    def test_interaction_pairs_left_right_collected(self) -> None:
        cfg = {
            "transformations": [
                {
                    "id": "interaction",
                    "enabled": True,
                    "params": {
                        "pairs": [
                            {"left": "atm_pcr", "right": "ltp", "op": "multiply"},
                            {"left": "delta", "right": "gamma", "op": "add"},
                        ],
                        "partition_by": ["trading_day"],
                    },
                }
            ],
        }
        sources = source_columns_for_transformations(cfg)
        self.assertIn("atm_pcr", sources)
        self.assertIn("ltp", sources)
        self.assertIn("delta", sources)
        self.assertIn("gamma", sources)
        self.assertIn("trading_day", sources)

    def test_derived_and_anchor_return_outputs_feature_collected(self) -> None:
        cfg = {
            "transformations": [
                {
                    "id": "derived",
                    "enabled": True,
                    "params": {
                        "outputs": [
                            {"feature": "spot", "column": "spot_derived", "terms": []},
                            {"feature": "fut_ltp", "column": "fut_derived", "terms": []},
                        ],
                    },
                },
                {
                    "id": "anchor_return",
                    "enabled": True,
                    "params": {
                        "outputs": [
                            {"feature": "nifty", "column": "nifty_ret"},
                        ],
                    },
                },
            ],
        }
        sources = source_columns_for_transformations(cfg)
        self.assertIn("spot", sources)
        self.assertIn("fut_ltp", sources)
        self.assertIn("nifty", sources)

    def test_params_feature_singular_alias_collected(self) -> None:
        cfg = {
            "transformations": [
                {
                    "id": "rolling",
                    "enabled": True,
                    "params": {"feature": "atm_iv", "windows": [10]},
                }
            ],
        }
        sources = source_columns_for_transformations(cfg)
        self.assertEqual(sources, ["atm_iv"])

    def test_ohlc_string_outputs_not_collected_as_sources(self) -> None:
        """OHLC ``outputs`` are field names (open/high/...), not source columns."""
        cfg = {
            "transformations": [
                {
                    "id": "ohlc_aggregation",
                    "enabled": True,
                    "params": {
                        "features": ["ltp"],
                        "outputs": ["open", "high", "low", "close"],
                        "partition_by": ["token"],
                    },
                }
            ],
        }
        sources = source_columns_for_transformations(cfg)
        self.assertIn("ltp", sources)
        self.assertIn("token", sources)
        for fld in ("open", "high", "low", "close"):
            self.assertNotIn(fld, sources)

    def test_flat_features_and_partition_by_still_work(self) -> None:
        cfg = transformation_config_from_dataset_meta(_lag_meta())
        sources = source_columns_for_transformations(cfg)
        self.assertIn("atm_pcr", sources)
        self.assertIn("ltp", sources)
        self.assertIn("trading_day", sources)
        self.assertIn("token", sources)


class FailedDayNotRetriedTests(unittest.TestCase):
    def test_failed_checkpoint_excluded_from_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab_db = os.path.join(tmp, "lab.db")
            days = ["2026-05-26", "2026-05-27"]
            assignments = assign_days_round_robin(days, 1)
            create_job(
                lab_db,
                job_id="jobfail01",
                lab_uuid="lab-uuid",
                data_dir=tmp,
                worker_count=1,
                day_assignments=assignments,
                config={"wanted_columns": []},
            )
            update_checkpoint(
                lab_db,
                "jobfail01",
                "2026-05-26",
                status=DAY_CP_FAILED,
                finished=True,
            )
            pending = worker_pending_days(lab_db, "jobfail01", 1)
            self.assertEqual([p["trading_day"] for p in pending], ["2026-05-27"])
            self.assertEqual(pending[0]["status"], DAY_CP_PENDING)


if __name__ == "__main__":
    unittest.main()
