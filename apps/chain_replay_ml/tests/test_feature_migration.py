"""Tests for master feature migration engine."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.dataset_builder.feature_migration_engine import (
    MIGRATION_TEMP_TABLE,
    analyze_master_feature_migration,
    commit_migration,
    make_sample_id,
    rollback_migration,
    start_feature_migration,
    validate_migration,
)
from chain_replay_ml.dataset_builder.master_store import MasterStore


class TestFeatureMigration(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "master_mig.db")

    def _seed_master(self) -> MasterStore:
        store = MasterStore(self.db_path)
        store.open()
        cols = [
            "trading_day", "timestamp", "token", "strike", "option_type",
            "expiry", "market", "ltp", "delta", "abs_delta", "feature_a",
        ]
        store.begin_day("2026-05-27", cols)
        store.insert_rows([
            {
                "trading_day": "2026-05-27",
                "timestamp": 1000.0,
                "token": "A",
                "strike": 25000.0,
                "option_type": "CE",
                "expiry": "2026-05-29",
                "market": "NIFTY",
                "ltp": 25.0,
                "delta": 0.15,
                "abs_delta": 0.15,
                "feature_a": 1.0,
            },
            {
                "trading_day": "2026-05-27",
                "timestamp": 1003.0,
                "token": "B",
                "strike": 25050.0,
                "option_type": "PE",
                "expiry": "2026-05-29",
                "market": "NIFTY",
                "ltp": 5.0,
                "delta": -0.25,
                "abs_delta": 0.25,
                "feature_a": 2.0,
            },
        ])
        store.commit_day("2026-05-27")
        store.set_meta("build_schema", {
            "feature_columns": ["feature_a"],
            "feature_count": 1,
            "target_columns": [],
        })
        return store

    def test_analyze_reports_missing(self) -> None:
        store = self._seed_master()
        try:
            out = analyze_master_feature_migration(store)
            self.assertGreaterEqual(out["registry_feature_count"], out["current_feature_count"])
            missing_names = {f["name"] for f in out["missing_features"]}
            for infra in ("ltp", "delta", "abs_delta", "strike"):
                self.assertNotIn(infra, missing_names, f"{infra} should not be a migration target")
            # Registry features not in build_schema.feature_columns must appear.
            self.assertGreater(out["missing_count"], 0)
            self.assertIn("feature_a", out.get("stored_features") or [])
            # Schema-lag (in DB / not in meta) is reported on missing rows.
            lag = [f for f in out["missing_features"] if f.get("reason") == "not_in_schema"]
            need_col = [f for f in out["missing_features"] if f.get("reason") == "missing_column"]
            self.assertTrue(lag or need_col)
        finally:
            store.close()

    def test_validate_and_commit(self) -> None:
        store = self._seed_master()
        try:
            conn = store.conn
            conn.execute(
                f"""
                CREATE TABLE "{MIGRATION_TEMP_TABLE}" (
                    sample_id TEXT PRIMARY KEY,
                    trading_day TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    token TEXT NOT NULL,
                    new_feat REAL
                )
                """
            )
            rows = conn.execute(
                "SELECT trading_day, timestamp, token FROM samples ORDER BY timestamp"
            ).fetchall()
            for r in rows:
                sid = make_sample_id(str(r[0]), float(r[1]), str(r[2]))
                conn.execute(
                    f'INSERT INTO "{MIGRATION_TEMP_TABLE}" VALUES (?, ?, ?, ?, ?)',
                    (sid, r[0], r[1], r[2], 42.0),
                )
            conn.commit()
            store.set_meta("feature_migration", {"features": ["new_feat"], "status": "computing"})

            report = validate_migration(store, ["new_feat"])
            self.assertTrue(report["passed"])
            self.assertEqual(report["master_row_count"], report["temp_row_count"])

            result = commit_migration(store, validation=report)
            self.assertIn("new_feat", result["features_merged"])
            cols = {row[1] for row in conn.execute("PRAGMA table_info(samples)").fetchall()}
            self.assertIn("new_feat", cols)
            val = conn.execute('SELECT "new_feat" FROM samples WHERE token = ?', ("A",)).fetchone()[0]
            self.assertEqual(float(val), 42.0)
        finally:
            store.close()

    def test_resume_keeps_started_at_and_commit_records_clocks(self) -> None:
        store = self._seed_master()
        try:
            started = start_feature_migration(store, ["gamma"])
            self.assertFalse(started.get("resumed"))
            started_at = started["progress"]["started_at"]
            self.assertTrue(started_at)

            # Simulate mid-job: leave pending days / computing status.
            progress = store.get_meta("feature_migration")
            progress["status"] = "computing"
            if not progress.get("pending_days"):
                progress["pending_days"] = ["2026-05-27"]
            if not progress.get("completed_days"):
                progress["completed_days"] = []
            store.set_meta("feature_migration", progress)

            resumed = start_feature_migration(store, ["gamma"], resume=True)
            self.assertTrue(resumed.get("resumed"))
            self.assertEqual(resumed["progress"]["started_at"], started_at)
            self.assertTrue(resumed["progress"].get("resumed_at"))

            # Prepare temp values so commit can finish with clocks.
            conn = store.conn
            features = ["gamma"]
            # Ensure temp has feature column + identity rows.
            cols = {r[1] for r in conn.execute(f'PRAGMA table_info("{MIGRATION_TEMP_TABLE}")').fetchall()}
            if "gamma" not in cols:
                conn.execute(f'ALTER TABLE "{MIGRATION_TEMP_TABLE}" ADD COLUMN gamma REAL')
                conn.commit()
            rows = conn.execute(
                f'SELECT sample_id FROM "{MIGRATION_TEMP_TABLE}"'
            ).fetchall()
            if not rows:
                sample_rows = conn.execute(
                    "SELECT trading_day, timestamp, token FROM samples"
                ).fetchall()
                for r in sample_rows:
                    sid = make_sample_id(str(r[0]), float(r[1]), str(r[2]))
                    conn.execute(
                        f'INSERT INTO "{MIGRATION_TEMP_TABLE}" (sample_id, trading_day, timestamp, token, gamma) '
                        f"VALUES (?, ?, ?, ?, ?)",
                        (sid, r[0], r[1], r[2], 0.1),
                    )
            else:
                conn.execute(f'UPDATE "{MIGRATION_TEMP_TABLE}" SET gamma = 0.1')
            conn.commit()

            report = validate_migration(store, features)
            self.assertTrue(report["passed"], report)
            result = commit_migration(store, validation=report)
            self.assertEqual(result["started_at"], started_at)
            self.assertTrue(result["committed_at"])
            self.assertIsNotNone(result["total_elapsed_sec"])
            self.assertEqual(result["progress"]["status"], "completed")
            self.assertEqual(result["progress"]["started_at"], started_at)
            self.assertEqual(result["progress"]["committed_at"], result["committed_at"])
        finally:
            store.close()

    def test_resume_succeeds_even_if_feature_column_already_in_samples(self) -> None:
        """Continue must not fail when samples already has the column (partial commit)."""
        store = self._seed_master()
        try:
            started = start_feature_migration(store, ["gamma"])
            self.assertTrue(started.get("ok"))
            # Simulate columns landing in samples before the job finished.
            store.conn.execute("ALTER TABLE samples ADD COLUMN gamma REAL")
            store.conn.commit()

            resumed = start_feature_migration(store, ["gamma"], resume=True)
            self.assertTrue(resumed.get("resumed"))
            self.assertEqual(resumed.get("pending_days"), ["2026-05-27"])
        finally:
            store.close()

    def test_start_blocks_fully_registered_features(self) -> None:
        """Features already in build_schema with data are not migration targets."""
        from chain_replay_ml.dataset_builder.feature_migration_engine import FeatureMigrationError

        store = self._seed_master()
        try:
            store.conn.execute("ALTER TABLE samples ADD COLUMN gamma REAL")
            store.conn.execute("UPDATE samples SET gamma = 1.0")
            store.conn.commit()
            bs = dict(store.get_meta("build_schema") or {})
            cols = list(bs.get("feature_columns") or []) + ["gamma"]
            bs["feature_columns"] = cols
            bs["feature_count"] = len(cols)
            store.set_meta("build_schema", bs)

            with self.assertRaises(FeatureMigrationError):
                start_feature_migration(store, ["gamma"], resume=False)
            # Mixed selection: skip complete features and migrate the rest.
            started = start_feature_migration(store, ["gamma", "theta"], resume=False)
            self.assertTrue(started.get("ok"))
            self.assertEqual(started.get("features"), ["theta"])
            self.assertNotIn("gamma", started.get("features") or [])
        finally:
            store.close()

    def test_rollback_drops_temp(self) -> None:
        store = self._seed_master()
        try:
            start_feature_migration(store, ["gamma"])
            rollback_migration(store)
            exists = store.conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (MIGRATION_TEMP_TABLE,),
            ).fetchone()
            self.assertIsNone(exists)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()

