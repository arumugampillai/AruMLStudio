"""Partition pruning — planner selects only requested trading days."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_engine import SampleSpec, query_dataset
from chain_replay_ml.dataset_engine.planner import plan_query
from chain_replay_ml.tests.dataset_engine._fixtures import (
    DemoParquetCase,
    require_duckdb,
)


class TestPartitionPruningPlanner(unittest.TestCase):
    def test_prune_by_trading_days(self) -> None:
        plan = plan_query(
            "analysis_demo",
            filters={"trading_days": ["2026-07-23", "2026-07-24"]},
            available_partitions=[
                "2026-07-22",
                "2026-07-23",
                "2026-07-24",
                "2026-07-25",
            ],
            sample=SampleSpec(max_rows=1000),
        )
        self.assertEqual(plan.partitions, ("2026-07-23", "2026-07-24"))
        self.assertEqual(plan.partitions_pruned, 2)

    def test_single_trading_day(self) -> None:
        plan = plan_query(
            "x",
            filters={"trading_day": "2026-07-23"},
            available_partitions=["2026-07-22", "2026-07-23"],
        )
        self.assertEqual(plan.partitions, ("2026-07-23",))
        self.assertEqual(plan.partitions_pruned, 1)

    def test_no_day_filter_keeps_all(self) -> None:
        plan = plan_query(
            "x",
            filters={"ltp_min": 15},
            available_partitions=["2026-07-23", "2026-07-24"],
        )
        self.assertEqual(plan.partitions, ("2026-07-23", "2026-07-24"))
        self.assertEqual(plan.partitions_pruned, 0)


class TestPartitionPruningQuery(DemoParquetCase):
    def setUp(self) -> None:
        require_duckdb()

    def test_stats_reflect_planner_prune(self) -> None:
        result = query_dataset(
            self.path,
            filters={"trading_days": ["2026-07-23"]},
            available_partitions=["2026-07-22", "2026-07-23", "2026-07-24"],
        )
        self.assertEqual(result.stats.partitions_pruned, 2)
        self.assertEqual(result.stats.partitions_scanned, 1)
        days = set(result.table.to_pandas()["trading_day"].astype(str))
        self.assertEqual(days, {"2026-07-23"})


if __name__ == "__main__":
    unittest.main()
