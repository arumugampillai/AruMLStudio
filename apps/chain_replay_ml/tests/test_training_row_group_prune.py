"""Tests for Create Model Opt-1 tail row-group pruning."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from chain_replay_ml.training.memory_utils import MAX_TRAINING_ROWS
from chain_replay_ml.training.row_group_prune import (
    plan_tail_row_groups,
    read_parquet_row_groups,
    target_rows_for_prune,
)


def _write_multi_rg(path: Path, n_rows: int = 1000, rg_size: int = 100) -> None:
    df = pd.DataFrame(
        {
            "trading_day": [f"2026-07-{(i // 100) + 1:02d}" for i in range(n_rows)],
            "timestamp": [float(i) for i in range(n_rows)],
            "token": ["A"] * n_rows,
            "feat_a": [float(i) for i in range(n_rows)],
            "future_ltp_60": [float(i + 1) for i in range(n_rows)],
        }
    )
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path, row_group_size=rg_size)


class TailRowGroupPruneTests(unittest.TestCase):
    def test_target_overread_with_premium(self) -> None:
        with patch.dict(os.environ, {"ARUNEO_TRAIN_ROW_GROUP_OVERREAD": "2"}):
            self.assertEqual(target_rows_for_prune(premium_filter=False, max_rows=100), 100)
            self.assertEqual(target_rows_for_prune(premium_filter=True, max_rows=100), 200)

    def test_plan_selects_trailing_groups(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ds.parquet"
            _write_multi_rg(path, n_rows=1000, rg_size=100)
            plan = plan_tail_row_groups(
                str(path),
                max_rows=250,
                metadata={"is_sorted": True},
                mode="on",
            )
            self.assertIsNotNone(plan)
            assert plan is not None
            self.assertTrue(plan.pruned)
            # 250 rows → last 3 groups (700,800,900) = indices 7,8,9
            self.assertEqual(list(plan.indices), [7, 8, 9])
            self.assertEqual(plan.rows_in_plan, 300)

            table = read_parquet_row_groups(str(path), plan.indices, columns=["timestamp", "feat_a"])
            self.assertEqual(table.num_rows, 300)
            ts = table.column("timestamp").to_pylist()
            self.assertEqual(ts[0], 700.0)
            self.assertEqual(ts[-1], 999.0)

    def test_below_cap_no_prune(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ds.parquet"
            _write_multi_rg(path, n_rows=200, rg_size=50)
            plan = plan_tail_row_groups(
                str(path),
                max_rows=500,
                metadata={"is_sorted": True},
                mode="on",
            )
            self.assertIsNotNone(plan)
            assert plan is not None
            self.assertFalse(plan.pruned)
            self.assertEqual(plan.skip_reason, "below_cap")

    def test_auto_skips_without_chronology(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ds.parquet"
            # Unsorted timestamps across groups — write reverse order chunks.
            chunks = []
            for g in range(5):
                start = (4 - g) * 100
                df = pd.DataFrame(
                    {
                        "timestamp": [float(start + i) for i in range(100)],
                        "feat_a": [float(i) for i in range(100)],
                    }
                )
                chunks.append(pa.Table.from_pandas(df, preserve_index=False))
            table = pa.concat_tables(chunks)
            pq.write_table(table, path, row_group_size=100)
            plan = plan_tail_row_groups(str(path), max_rows=150, metadata={}, mode="auto")
            self.assertIsNotNone(plan)
            assert plan is not None
            self.assertFalse(plan.pruned)
            self.assertEqual(plan.skip_reason, "chronology_unproven")

    def test_pandas_load_uses_prune(self) -> None:
        from chain_replay_ml.dataset_builder.writer import _safe_filename, datasets_dir
        from chain_replay_ml.training.dataset_loader import load_dataset_frame

        with tempfile.TemporaryDirectory() as tmp:
            out = datasets_dir(tmp)
            os.makedirs(out, exist_ok=True)
            name = "prune_train"
            safe = _safe_filename(name)
            path = Path(out) / f"{safe}.parquet"
            meta_path = Path(out) / f"{safe}.json"
            _write_multi_rg(path, n_rows=1000, rg_size=100)
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "dataset_name": name,
                        "row_count": 1000,
                        "is_sorted": True,
                        "row_order": ["trading_day", "timestamp", "token"],
                    },
                    fh,
                )
            with patch(
                "chain_replay_ml.training.row_group_prune.MAX_TRAINING_ROWS",
                250,
            ):
                # Also patch the imported constant used by target_rows default path
                with patch.dict(os.environ, {"ARUNEO_TRAIN_ROW_GROUP_PRUNE": "on"}):
                    df, meta, _ = load_dataset_frame(
                        tmp,
                        name,
                        columns=["timestamp", "feat_a", "trading_day", "token"],
                        max_rows_hint=250,
                    )
            self.assertEqual(len(df), 300)
            self.assertTrue((meta.get("row_group_prune") or {}).get("applied"))
            self.assertEqual(float(df["timestamp"].iloc[0]), 700.0)


if __name__ == "__main__":
    unittest.main()
