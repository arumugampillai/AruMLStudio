"""Phase 2: Fixed Horizon migration — old Stage 5 / backfill math vs OLE must match."""

from __future__ import annotations

import unittest
from typing import Any

from chain_replay_ml.dataset_builder.feature_plugins import horizon_column_name
from chain_replay_ml.dataset_builder.target_backfill import compute_targets_for_row
from chain_replay_ml.outcome_label_engine import (
    FIXED_HORIZON_STRATEGY_ID,
    LabelSourceContext,
    LabelStrategyConfig,
    ensure_builtin_strategies,
    get_strategy,
)
from chain_replay_ml.outcome_label_engine.fixed_horizon import (
    FixedHorizonStrategy,
    compute_fixed_horizon_targets,
)
from chain_replay_ml.ticks import TickTimeline


def _legacy_stage5_targets(
    *,
    ts: float,
    opt_tl: Any,
    horizons_sec: list[int],
    max_stale_sec: float,
) -> dict[str, Any]:
    """Inline copy of pre-migration Stage 5 loop (reference oracle for identity)."""
    out: dict[str, Any] = {}
    for h in horizons_sec:
        col = horizon_column_name(h)
        future_ts = ts + float(h)
        if opt_tl.is_fresh_at(future_ts, max_stale_sec):
            out[col] = opt_tl.ltp_rupees_at(future_ts)
        else:
            out[col] = None
    return out


def _opt_tl_with_future(
    *,
    base_ts: float,
    ltp: float,
    future_ltp: float,
    horizon: float,
) -> TickTimeline:
    tl = TickTimeline()
    tl.append(base_ts, int(round(ltp * 100)))
    tl.append(base_ts + horizon, int(round(future_ltp * 100)))
    return tl


class FixedHorizonMigrationTests(unittest.TestCase):
    def setUp(self) -> None:
        ensure_builtin_strategies()

    def test_registry_defaults_to_fixed_horizon(self) -> None:
        s = get_strategy(FIXED_HORIZON_STRATEGY_ID)
        self.assertIsInstance(s, FixedHorizonStrategy)
        self.assertEqual(s.metadata.strategy_id, "fixed_horizon")
        schema = s.get_config_schema()
        self.assertIn("horizons_sec", schema)
        self.assertIn("max_stale_sec", schema)
        defs = s.get_target_definitions()
        self.assertEqual(defs.primary_target, "future_ltp_5m")

    def test_ole_matches_legacy_stage5_fresh(self) -> None:
        opt_tl = _opt_tl_with_future(
            base_ts=1000.0, ltp=100.0, future_ltp=112.5, horizon=5.0
        )
        horizons = [5, 10, 60, 300]
        legacy = _legacy_stage5_targets(
            ts=1000.0,
            opt_tl=opt_tl,
            horizons_sec=horizons,
            max_stale_sec=10.0,
        )
        ole = compute_fixed_horizon_targets(
            ts=1000.0,
            opt_tl=opt_tl,
            horizons_sec=horizons,
            max_stale_sec=10.0,
        )
        backfill = compute_targets_for_row(
            ts=1000.0,
            opt_tl=opt_tl,
            horizons_sec=horizons,
            max_stale_sec=10.0,
        )
        self.assertEqual(ole, legacy)
        self.assertEqual(backfill, legacy)
        self.assertEqual(ole["future_ltp_5s"], 112.5)
        self.assertEqual(ole["future_ltp_1m"], None)  # no tick at +60s → stale/missing
        self.assertEqual(ole["future_ltp_5m"], None)

    def test_ole_matches_legacy_stage5_stale(self) -> None:
        stale_tl = TickTimeline()
        stale_tl.append(990.0, 10000)
        horizons = [5]
        legacy = _legacy_stage5_targets(
            ts=1000.0,
            opt_tl=stale_tl,
            horizons_sec=horizons,
            max_stale_sec=10.0,
        )
        ole = compute_fixed_horizon_targets(
            ts=1000.0,
            opt_tl=stale_tl,
            horizons_sec=horizons,
            max_stale_sec=10.0,
        )
        self.assertEqual(ole, legacy)
        self.assertIsNone(ole["future_ltp_5s"])

    def test_ole_matches_legacy_with_column_subset(self) -> None:
        opt_tl = _opt_tl_with_future(
            base_ts=1000.0, ltp=100.0, future_ltp=111.0, horizon=5.0
        )
        horizons = [5, 10]
        columns = ["future_ltp_5s"]
        # Legacy backfill filtered by columns; replicate that filter on Stage 5 oracle.
        legacy_all = _legacy_stage5_targets(
            ts=1000.0,
            opt_tl=opt_tl,
            horizons_sec=horizons,
            max_stale_sec=10.0,
        )
        legacy = {c: legacy_all[c] for c in columns}
        ole = compute_fixed_horizon_targets(
            ts=1000.0,
            opt_tl=opt_tl,
            horizons_sec=horizons,
            max_stale_sec=10.0,
            columns=columns,
        )
        self.assertEqual(ole, legacy)
        self.assertEqual(ole, {"future_ltp_5s": 111.0})

    def test_build_labels_batch_matches_row_api(self) -> None:
        opt_tl = _opt_tl_with_future(
            base_ts=2000.0, ltp=50.0, future_ltp=55.0, horizon=10.0
        )
        strategy = FixedHorizonStrategy()
        samples = [
            {"timestamp": 2000.0, "_opt_tl": opt_tl, "token": "T1"},
            {"timestamp": 2000.0, "opt_tl": opt_tl, "token": "T2"},
        ]
        batch = strategy.build_labels(
            LabelSourceContext(source_kind="master", day="2024-01-02"),
            samples,
            LabelStrategyConfig(
                strategy_id="fixed_horizon",
                version="1.0",
                params={"horizons_sec": [10], "max_stale_sec": 10.0},
            ),
        )
        expected = compute_fixed_horizon_targets(
            ts=2000.0,
            opt_tl=opt_tl,
            horizons_sec=[10],
            max_stale_sec=10.0,
        )
        self.assertEqual(len(batch.rows), 2)
        for row in batch.rows:
            self.assertEqual(row["future_ltp_10s"], expected["future_ltp_10s"])
            self.assertNotIn("_opt_tl", row)
            self.assertNotIn("opt_tl", row)
            self.assertTrue(row["is_valid"])
        self.assertEqual(batch.target_columns, ["future_ltp_10s"])


if __name__ == "__main__":
    unittest.main()
