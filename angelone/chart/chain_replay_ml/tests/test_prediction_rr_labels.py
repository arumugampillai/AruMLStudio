"""Tests for Reward/Risk classifier labels on prediction_dataset rows."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.model_lab.prediction_schema import (
    RR_HIT_COLUMNS,
    compute_rr_hit_labels,
)
from chain_replay_ml.model_lab.store import ModelLabStore

_ALL_ZERO = {c: 0 for c in RR_HIT_COLUMNS}
_ALL_ONE = {c: 1 for c in RR_HIT_COLUMNS}


class ComputeRrHitLabelsTests(unittest.TestCase):
    def test_example_1_all_hits(self) -> None:
        # 12 / 3 = 4.0 → all thresholds
        labels = compute_rr_hit_labels(
            target_reached=1,
            maximum_profit=12.0,
            maximum_drawdown=3.0,
        )
        self.assertEqual(labels, _ALL_ONE)

    def test_example_2_partial_hits(self) -> None:
        # 8 / 3 ≈ 2.67 → 1:1, 2:3, 1:2 only
        labels = compute_rr_hit_labels(
            target_reached=1,
            maximum_profit=8.0,
            maximum_drawdown=3.0,
        )
        self.assertEqual(
            labels,
            {
                "rr_1_1_hit": 1,
                "rr_2_3_hit": 1,
                "rr_1_2_hit": 1,
                "rr_1_3_hit": 0,
                "rr_1_4_hit": 0,
            },
        )

    def test_rr_1_1_and_2_3_thresholds(self) -> None:
        # 1.0× → RR 1:1 only
        self.assertEqual(
            compute_rr_hit_labels(
                target_reached=1, maximum_profit=3.0, maximum_drawdown=3.0
            ),
            {
                "rr_1_1_hit": 1,
                "rr_2_3_hit": 0,
                "rr_1_2_hit": 0,
                "rr_1_3_hit": 0,
                "rr_1_4_hit": 0,
            },
        )
        # 1.5× → RR 1:1 + RR 2:3
        self.assertEqual(
            compute_rr_hit_labels(
                target_reached=1, maximum_profit=4.5, maximum_drawdown=3.0
            ),
            {
                "rr_1_1_hit": 1,
                "rr_2_3_hit": 1,
                "rr_1_2_hit": 0,
                "rr_1_3_hit": 0,
                "rr_1_4_hit": 0,
            },
        )

    def test_example_3_target_not_reached(self) -> None:
        labels = compute_rr_hit_labels(
            target_reached=0,
            maximum_profit=15.0,
            maximum_drawdown=2.0,
        )
        self.assertEqual(labels, _ALL_ZERO)

    def test_target_miss_and_null(self) -> None:
        self.assertEqual(
            compute_rr_hit_labels(
                target_reached=0, maximum_profit=10.0, maximum_drawdown=2.0
            ),
            _ALL_ZERO,
        )
        self.assertEqual(
            compute_rr_hit_labels(
                target_reached=None, maximum_profit=10.0, maximum_drawdown=2.0
            ),
            _ALL_ZERO,
        )


class RrLabelSchemaTests(unittest.TestCase):
    def test_migration_adds_columns_and_persists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lab.db")
            with ModelLabStore(path) as store:
                store.ensure_prediction_schema()
                cols = set(store._prediction_table_columns())
                for name in RR_HIT_COLUMNS:
                    self.assertIn(name, cols)
                store.write_prediction_summary(
                    lab_uuid="u1",
                    status="ready",
                    row_count=2,
                    trading_days=1,
                    target_column="future_ltp_5m",
                )
                store.insert_prediction_rows(
                    [
                        {
                            "lab_uuid": "u1",
                            "prediction_id": "a",
                            "trading_day": "2026-07-01",
                            "timestamp": 1.0,
                            "target_reached": 1,
                            "maximum_profit": 12.0,
                            "maximum_drawdown": 3.0,
                            **compute_rr_hit_labels(
                                target_reached=1,
                                maximum_profit=12.0,
                                maximum_drawdown=3.0,
                            ),
                        },
                        {
                            "lab_uuid": "u1",
                            "prediction_id": "b",
                            "trading_day": "2026-07-01",
                            "timestamp": 2.0,
                            "target_reached": 1,
                            "maximum_profit": 4.5,
                            "maximum_drawdown": 3.0,
                            **compute_rr_hit_labels(
                                target_reached=1,
                                maximum_profit=4.5,
                                maximum_drawdown=3.0,
                            ),
                        },
                    ]
                )
                sums = store.conn.execute(
                    """
                    SELECT
                        SUM(rr_1_1_hit),
                        SUM(rr_2_3_hit),
                        SUM(rr_1_2_hit),
                        SUM(rr_1_3_hit),
                        SUM(rr_1_4_hit)
                    FROM prediction_dataset
                    """
                ).fetchone()
                self.assertEqual(tuple(int(x or 0) for x in sums), (2, 2, 1, 1, 1))


if __name__ == "__main__":
    unittest.main()
