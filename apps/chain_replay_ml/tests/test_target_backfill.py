"""Tests for target horizon backfill on master SQLite datasets."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.dataset_builder.day_context import DayContext, SourceSpec
from chain_replay_ml.dataset_builder.feature_plugins import horizon_column_name
from chain_replay_ml.dataset_builder.master_store import MasterStore
from chain_replay_ml.dataset_builder.target_backfill import (
    analyze_target_backfill,
    backfill_day_targets,
    compute_targets_for_row,
    maybe_backfill_expanded_targets,
)
from chain_replay_ml.ticks import TickTimeline


def _opt_tl_with_future(*, base_ts: float, ltp: float, future_ltp: float, horizon: float) -> TickTimeline:
    tl = TickTimeline()
    tl.append(base_ts, int(round(ltp * 100)))
    tl.append(base_ts + horizon, int(round(future_ltp * 100)))
    return tl


class TestTargetBackfill(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "master_backfill_test.db")

    def _store(self) -> MasterStore:
        store = MasterStore(self.db_path)
        store.open()
        return store

    def _seed_day_with_null_targets(self, store: MasterStore) -> None:
        store.set_meta("build_schema", {
            "feature_columns": ["ltp"],
            "target_columns": ["future_ltp_10s"],
            "feature_count": 1,
            "target_count": 1,
        })
        cols = [
            "trading_day", "timestamp", "token", "market", "expiry",
            "strike", "option_type", "future_ltp_10s", "future_ltp_5s",
        ]
        store.begin_day("2026-01-02", cols)
        store.insert_rows([
            {
                "trading_day": "2026-01-02",
                "timestamp": 1000.0,
                "token": "TOK1",
                "market": "NIFTY",
                "expiry": "2026-01-08",
                "strike": 25000.0,
                "option_type": "CE",
                "future_ltp_10s": 105.0,
                "future_ltp_5s": None,
            },
            {
                "trading_day": "2026-01-02",
                "timestamp": 1010.0,
                "token": "TOK1",
                "market": "NIFTY",
                "expiry": "2026-01-08",
                "strike": 25000.0,
                "option_type": "CE",
                "future_ltp_10s": 110.0,
                "future_ltp_5s": None,
            },
        ])
        store.commit_day("2026-01-02")

    def test_analyze_detects_new_horizon_columns(self) -> None:
        store = self._store()
        try:
            self._seed_day_with_null_targets(store)
            analysis = analyze_target_backfill(
                store,
                ["future_ltp_10s", "future_ltp_5s"],
            )
            self.assertTrue(analysis["needed"])
            self.assertEqual(analysis["columns"], ["future_ltp_5s"])
            self.assertEqual(analysis["trading_days"], ["2026-01-02"])
        finally:
            store.close()

    def test_analyze_only_flags_days_missing_the_column(self) -> None:
        """Regression: adding a new day must not re-queue days that already
        have the target column populated (previously this returned *every*
        trading day in the master DB, causing a full re-scan — including a
        full tick-DB reload — of already-complete days on every build)."""
        store = self._store()
        try:
            self._seed_day_with_null_targets(store)
            # Second, newer day already has both target columns fully populated —
            # it must be left untouched even though the schema mismatch check
            # would otherwise treat "future_ltp_5s" as globally missing.
            cols = [
                "trading_day", "timestamp", "token", "market", "expiry",
                "strike", "option_type", "future_ltp_10s", "future_ltp_5s",
            ]
            store.begin_day("2026-01-05", cols)
            store.insert_rows([
                {
                    "trading_day": "2026-01-05",
                    "timestamp": 2000.0,
                    "token": "TOK1",
                    "market": "NIFTY",
                    "expiry": "2026-01-08",
                    "strike": 25000.0,
                    "option_type": "CE",
                    "future_ltp_10s": 120.0,
                    "future_ltp_5s": 118.0,
                },
            ])
            store.commit_day("2026-01-05")

            analysis = analyze_target_backfill(
                store,
                ["future_ltp_10s", "future_ltp_5s"],
            )
            self.assertTrue(analysis["needed"])
            self.assertEqual(analysis["columns"], ["future_ltp_5s"])
            # Only the day that actually lacks the column should be queued.
            self.assertEqual(analysis["trading_days"], ["2026-01-02"])
        finally:
            store.close()

    def test_analyze_no_backfill_when_all_days_populated(self) -> None:
        store = self._store()
        try:
            self._seed_day_with_null_targets(store)
            store.conn.execute('UPDATE samples SET "future_ltp_5s" = 100.0')
            store.conn.commit()
            analysis = analyze_target_backfill(
                store,
                ["future_ltp_10s", "future_ltp_5s"],
            )
            self.assertFalse(analysis["needed"])
            self.assertEqual(analysis["columns"], [])
            self.assertEqual(analysis["trading_days"], [])
        finally:
            store.close()

    def test_compute_targets_matches_freshness_policy(self) -> None:
        opt_tl = _opt_tl_with_future(base_ts=1000.0, ltp=100.0, future_ltp=112.5, horizon=5.0)
        values = compute_targets_for_row(
            ts=1000.0,
            opt_tl=opt_tl,
            horizons_sec=[5],
            max_stale_sec=10.0,
            columns=["future_ltp_5s"],
        )
        self.assertEqual(values["future_ltp_5s"], 112.5)

        stale_tl = TickTimeline()
        stale_tl.append(990.0, 10000)
        stale = compute_targets_for_row(
            ts=1000.0,
            opt_tl=stale_tl,
            horizons_sec=[5],
            max_stale_sec=10.0,
            columns=["future_ltp_5s"],
        )
        self.assertIsNone(stale["future_ltp_5s"])

    def test_backfill_populates_null_target_columns(self) -> None:
        store = self._store()
        try:
            self._seed_day_with_null_targets(store)
            opt_tl = _opt_tl_with_future(base_ts=1000.0, ltp=100.0, future_ltp=111.0, horizon=5.0)
            opt_tl.append(1010.0, 10000)
            opt_tl.append(1015.0, 11500)

            ctx = DayContext(
                source=SourceSpec(
                    source_id="2026-01-02|NIFTY|2026-01-08",
                    trading_day="2026-01-02",
                    market="NIFTY",
                    expiry="2026-01-08",
                ),
                db_path="",
                expiry_norm="2026-01-08",
                open_ts=900.0,
                close_ts=2000.0,
                expiry_ts=3000.0,
                index_tl=TickTimeline(),
                strike_mapping={
                    (25000.0, "CE"): ("TOK1", "SYM", opt_tl),
                },
            )

            result = backfill_day_targets(
                store,
                trading_day="2026-01-02",
                target_columns=["future_ltp_5s"],
                horizons_sec=[5],
                ctx=ctx,
            )
            self.assertFalse(result.get("skipped"))
            self.assertEqual(result["rows"], 2)

            rows = store.conn.execute(
                'SELECT timestamp, "future_ltp_5s" FROM samples ORDER BY timestamp'
            ).fetchall()
            self.assertEqual(rows[0][1], 111.0)
            self.assertEqual(rows[1][1], 115.0)
        finally:
            store.close()

    def test_backfill_emits_heartbeat_before_loading_tick_database(self) -> None:
        """Loading the tick DB for a day has no internal progress hooks and can
        take minutes; a heartbeat right before it must fire so the UI's last
        received-at timestamp refreshes and it doesn't look frozen."""
        store = self._store()
        try:
            self._seed_day_with_null_targets(store)
            opt_tl = _opt_tl_with_future(base_ts=1000.0, ltp=100.0, future_ltp=111.0, horizon=5.0)
            opt_tl.append(1010.0, 10000)
            opt_tl.append(1015.0, 11500)
            ctx = DayContext(
                source=SourceSpec(
                    source_id="2026-01-02|NIFTY|2026-01-08",
                    trading_day="2026-01-02",
                    market="NIFTY",
                    expiry="2026-01-08",
                ),
                db_path="",
                expiry_norm="2026-01-08",
                open_ts=900.0,
                close_ts=2000.0,
                expiry_ts=3000.0,
                index_tl=TickTimeline(),
                strike_mapping={
                    (25000.0, "CE"): ("TOK1", "SYM", opt_tl),
                },
            )

            from unittest.mock import patch

            calls: list[tuple[str, int, int, str]] = []

            def _capture(msg: str, cur: int, tot: int, unit: str = "days") -> None:
                calls.append((msg, cur, tot, unit))

            with patch(
                "chain_replay_ml.dataset_builder.target_backfill.load_day_context",
                return_value=ctx,
            ) as mock_load:
                backfill_day_targets(
                    store,
                    trading_day="2026-01-02",
                    target_columns=["future_ltp_5s"],
                    horizons_sec=[5],
                    on_progress=_capture,
                )
                mock_load.assert_called_once()

            self.assertGreaterEqual(len(calls), 1)
            first_msg, first_cur, first_tot, first_unit = calls[0]
            self.assertIn("loading tick database", first_msg)
            self.assertIn("2026-01-02", first_msg)
            self.assertEqual(first_cur, 0)
            self.assertEqual(first_tot, 2)  # two seeded rows for the day
            self.assertEqual(first_unit, "rows")
        finally:
            store.close()

    def test_maybe_backfill_updates_build_schema(self) -> None:
        store = self._store()
        try:
            self._seed_day_with_null_targets(store)
            opt_tl = _opt_tl_with_future(base_ts=1000.0, ltp=100.0, future_ltp=111.0, horizon=5.0)
            opt_tl.append(1010.0, 10000)
            opt_tl.append(1015.0, 11500)
            ctx = DayContext(
                source=SourceSpec(
                    source_id="2026-01-02|NIFTY|2026-01-08",
                    trading_day="2026-01-02",
                    market="NIFTY",
                    expiry="2026-01-08",
                ),
                db_path="",
                expiry_norm="2026-01-08",
                open_ts=900.0,
                close_ts=2000.0,
                expiry_ts=3000.0,
                index_tl=TickTimeline(),
                strike_mapping={
                    (25000.0, "CE"): ("TOK1", "SYM", opt_tl),
                },
            )

            from unittest.mock import patch

            target_cols = [horizon_column_name(10), horizon_column_name(5)]
            with patch(
                "chain_replay_ml.dataset_builder.target_backfill.load_day_context",
                return_value=ctx,
            ):
                bf = maybe_backfill_expanded_targets(
                    store,
                    target_columns=target_cols,
                    horizons_sec=[10, 5],
                )

            self.assertTrue(bf.get("backfilled"))
            schema = store.get_meta("build_schema") or {}
            self.assertEqual(schema.get("target_columns"), target_cols)
            self.assertEqual(schema.get("target_count"), 2)

            nulls = store.conn.execute(
                'SELECT COUNT(*) FROM samples WHERE "future_ltp_5s" IS NULL'
            ).fetchone()[0]
            self.assertEqual(nulls, 0)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
