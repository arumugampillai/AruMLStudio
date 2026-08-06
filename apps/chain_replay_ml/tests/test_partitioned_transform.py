"""Day-at-a-time / Fast-Safe partitioned transformation pipeline tests."""

from __future__ import annotations

import os
import tempfile
import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.transformations.pipeline import (
    run_transformation_pipeline_on_parquet,
)


class PartitionedTransformTests(unittest.TestCase):
    def test_day_partitioned_lag(self) -> None:
        rows = []
        for day, base in (("2024-01-02", 100.0), ("2024-01-03", 200.0)):
            for i in range(5):
                rows.append({
                    "trading_day": day,
                    "token": "T1",
                    "ltp": base + i,
                })
        df = pd.DataFrame(rows)
        logs: list[str] = []
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sample.parquet")
            df.to_parquet(path, index=False)
            cfg = {
                "transformation_pipeline_version": 1,
                "transformations": [{
                    "id": "lag",
                    "enabled": True,
                    "params": {
                        "features": ["ltp"],
                        "horizons": [{"seconds": 3, "suffix": "3s"}],
                        "partition_by": ["trading_day", "token"],
                        "sample_interval_sec": 3.0,
                    },
                }],
            }
            result = run_transformation_pipeline_on_parquet(
                path, cfg, log_fn=logs.append,
            )
            self.assertIn("ltp_lag_3s", result.created_columns)
            self.assertEqual(result.execution.get("strategy"), "day_at_a_time")
            self.assertEqual(result.execution.get("mode_by_day", {}).get("2024-01-02"), "fast")
            self.assertEqual(result.execution.get("mode_by_day", {}).get("2024-01-03"), "fast")
            self.assertTrue(any("day-at-a-time" in line for line in logs))
            out = pd.read_parquet(path)
            self.assertEqual(len(out), len(df))
            self.assertIn("ltp_lag_3s", out.columns)
            # First row of each day is NaN lag
            for day in ("2024-01-02", "2024-01-03"):
                day_df = out[out["trading_day"] == day].reset_index(drop=True)
                self.assertTrue(pd.isna(day_df.loc[0, "ltp_lag_3s"]))
                self.assertEqual(day_df.loc[1, "ltp_lag_3s"], day_df.loc[0, "ltp"])

    def test_processes_days_sequentially(self) -> None:
        rows = []
        for day in ("2024-01-02", "2024-01-03", "2024-01-04"):
            for tok in ("A", "B"):
                for i in range(4):
                    rows.append({
                        "trading_day": day,
                        "token": tok,
                        "ltp": float(i + 1),
                    })
        df = pd.DataFrame(rows)
        order: list[str] = []

        def _progress(msg: str, cur: int, tot: int) -> None:
            if "done · mode=" in msg and "(" in msg:
                start = msg.index("(") + 1
                end = msg.index(")", start)
                order.append(msg[start:end])

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "multi.parquet")
            df.to_parquet(path, index=False)
            cfg = {
                "transformation_pipeline_version": 1,
                "transformations": [{
                    "id": "lag",
                    "enabled": True,
                    "params": {
                        "features": ["ltp"],
                        "horizons": [{"seconds": 3, "suffix": "3s"}],
                        "partition_by": ["trading_day", "token"],
                        "sample_interval_sec": 3.0,
                    },
                }],
            }
            run_transformation_pipeline_on_parquet(
                path, cfg, on_partition_progress=_progress,
            )
            self.assertEqual(order, ["2024-01-02", "2024-01-03", "2024-01-04"])


if __name__ == "__main__":
    unittest.main()
