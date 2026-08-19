"""Comprehensive Unit Tests for Phase 4D.1: Research Memory Schema & DB Initialization."""

import hashlib
import os
import shutil
import sqlite3
import tempfile
import unittest

from chain_replay_ml.research_memory import (
    EXPECTED_INDICES,
    EXPECTED_TABLES,
    analysis_db_path,
    connect_analysis_db,
    init_analysis_db,
    verify_analysis_db_schema,
)


class TestResearchMemorySchema(unittest.TestCase):
    """Test suite verifying analysis.db initialization, schema correctness, WAL mode, and integrity."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_analysis_db_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_fresh_analysis_db_initialization(self):
        """1. Verify fresh initialization creates analysis.db and all expected tables."""
        db_path = init_analysis_db(self.tmp_dir)
        self.assertTrue(os.path.isfile(db_path))
        self.assertEqual(db_path, analysis_db_path(self.tmp_dir))

        diag = verify_analysis_db_schema(self.tmp_dir)
        self.assertTrue(diag["exists"])
        self.assertTrue(diag["is_valid"])
        self.assertEqual(len(diag["tables_missing"]), 0)
        self.assertEqual(len(diag["indices_missing"]), 0)
        self.assertTrue(diag["foreign_keys_enabled"])
        self.assertEqual(diag["journal_mode"], "wal")

    def test_all_tables_and_columns_exist(self):
        """2. Verify exact columns exist for all 9 core research tables."""
        init_analysis_db(self.tmp_dir)
        conn = connect_analysis_db(self.tmp_dir)
        try:
            # Table 1: research_campaigns
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(research_campaigns);").fetchall()}
            self.assertIn("campaign_id", cols)
            self.assertIn("campaign_name", cols)
            self.assertIn("context_key", cols)
            self.assertIn("ranking_policy_version", cols)
            self.assertIn("status", cols)
            self.assertIn("max_experiments_limit", cols)
            self.assertIn("max_duration_seconds", cols)
            self.assertIn("memory_limit_mb", cols)
            self.assertIn("total_planned", cols)
            self.assertIn("completed_count", cols)
            self.assertIn("skipped_duplicate_count", cols)
            self.assertIn("failed_count", cols)

            # Table 2: experiment_signatures
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(experiment_signatures);").fetchall()}
            self.assertIn("signature_hash", cols)
            self.assertIn("context_key", cols)
            self.assertIn("market", cols)
            self.assertIn("sampling_interval_sec", cols)
            self.assertIn("task_type", cols)
            self.assertIn("prediction_horizon", cols)
            self.assertIn("regime_id", cols)
            self.assertIn("regime_definition_hash", cols)
            self.assertIn("dataset_snapshot_hash", cols)
            self.assertIn("feature_set_hash", cols)
            self.assertIn("algorithm", cols)
            self.assertIn("hyperparameters_hash", cols)
            self.assertIn("walk_forward_hash", cols)
            self.assertIn("canonical_payload_json", cols)

            # Table 3: campaign_experiments
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(campaign_experiments);").fetchall()}
            self.assertIn("campaign_exp_id", cols)
            self.assertIn("campaign_id", cols)
            self.assertIn("trial_index", cols)
            self.assertIn("signature_hash", cols)
            self.assertIn("execution_status", cols)
            self.assertIn("memory_peak_mb", cols)

            # Table 4: benchmark_runs
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(benchmark_runs);").fetchall()}
            self.assertIn("benchmark_run_id", cols)
            self.assertIn("context_key", cols)
            self.assertIn("benchmark_scope", cols)
            self.assertIn("ranking_policy_version", cols)

            # Table 5: model_benchmarks
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(model_benchmarks);").fetchall()}
            self.assertIn("benchmark_id", cols)
            self.assertIn("benchmark_run_id", cols)
            self.assertIn("model_name", cols)
            self.assertIn("signature_hash", cols)
            self.assertIn("primary_metric_name", cols)
            self.assertIn("primary_metric_value", cols)
            self.assertIn("temporal_stability_score", cols)
            self.assertIn("brier_score", cols)
            self.assertIn("expected_calibration_error", cols)
            self.assertIn("robustness_score", cols)
            self.assertIn("rank_in_context", cols)
            self.assertIn("recommendation_status", cols)

            # Table 6: benchmark_metrics (Normalized metric store)
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(benchmark_metrics);").fetchall()}
            self.assertIn("metric_id", cols)
            self.assertIn("benchmark_id", cols)
            self.assertIn("metric_name", cols)
            self.assertIn("metric_stage", cols)
            self.assertIn("fold_index", cols)
            self.assertIn("metric_value", cols)
            self.assertIn("metric_type", cols)

            # Table 7: regime_evaluations
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(regime_evaluations);").fetchall()}
            self.assertIn("eval_id", cols)
            self.assertIn("model_name", cols)
            self.assertIn("tested_regime_id", cols)
            self.assertIn("tested_regime_hash", cols)
            self.assertIn("is_native_regime", cols)
            self.assertIn("regime_degradation_pct", cols)

            # Table 8: feature_set_evaluations
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(feature_set_evaluations);").fetchall()}
            self.assertIn("feature_eval_id", cols)
            self.assertIn("total_features", cols)
            self.assertIn("base_pipeline_count", cols)
            self.assertIn("registry_feature_count", cols)
            self.assertIn("experimental_feature_count", cols)
            self.assertIn("deprecated_feature_count", cols)
            self.assertIn("experimental_dependency_ratio", cols)

            # Table 9: champion_history
            cols = {r["name"] for r in conn.execute("PRAGMA table_info(champion_history);").fetchall()}
            self.assertIn("transition_id", cols)
            self.assertIn("context_key", cols)
            self.assertIn("previous_champion_name", cols)
            self.assertIn("new_champion_name", cols)
            self.assertIn("previous_robustness_score", cols)
            self.assertIn("new_robustness_score", cols)
            self.assertIn("score_delta", cols)
            self.assertIn("ranking_policy_version", cols)
            self.assertIn("promoted_by", cols)
        finally:
            conn.close()

    def test_idempotent_initialization(self):
        """3. Verify repeated initialization calls are strictly idempotent."""
        path1 = init_analysis_db(self.tmp_dir)
        path2 = init_analysis_db(self.tmp_dir)
        path3 = init_analysis_db(self.tmp_dir)

        self.assertEqual(path1, path2)
        self.assertEqual(path2, path3)

        diag = verify_analysis_db_schema(self.tmp_dir)
        self.assertTrue(diag["is_valid"])
        self.assertEqual(len(diag["tables_found"]), len(EXPECTED_TABLES))

    def test_foreign_key_enforcement(self):
        """4. Verify foreign key enforcement rejects orphaned relational records."""
        init_analysis_db(self.tmp_dir)
        conn = connect_analysis_db(self.tmp_dir)
        try:
            # Inserting a benchmark_metric with non-existent benchmark_id should fail
            with self.assertRaises(sqlite3.IntegrityError):
                with conn:
                    conn.execute(
                        """
                        INSERT INTO benchmark_metrics (
                            benchmark_id, metric_name, metric_stage, metric_value, metric_type, created_at
                        ) VALUES (9999, 'test_roc_auc', 'TEST', 0.85, 'SCALAR_FLOAT', '2026-08-19T00:00:00Z');
                        """
                    )
        finally:
            conn.close()

    def test_transaction_atomicity_and_rollback(self):
        """5. Verify atomic rollback on failure preserves clean state."""
        init_analysis_db(self.tmp_dir)
        conn = connect_analysis_db(self.tmp_dir)
        try:
            # Execute a failing transaction block
            try:
                with conn:
                    conn.execute(
                        """
                        INSERT INTO research_campaigns (
                            campaign_id, campaign_name, context_key, created_at, updated_at
                        ) VALUES ('CMP_FAIL_TEST', 'Test Campaign', 'NIFTY_3s_DIR_5m_R001', '2026-08-19T00:00:00Z', '2026-08-19T00:00:00Z');
                        """
                    )
                    # Trigger a foreign key failure inside the same transaction
                    conn.execute(
                        """
                        INSERT INTO benchmark_metrics (
                            benchmark_id, metric_name, metric_stage, metric_value, metric_type, created_at
                        ) VALUES (99999, 'invalid', 'TEST', 0.0, 'FLOAT', '2026-08-19T00:00:00Z');
                        """
                    )
            except sqlite3.IntegrityError:
                pass

            # Verify that CMP_FAIL_TEST was rolled back completely
            row = conn.execute(
                "SELECT campaign_id FROM research_campaigns WHERE campaign_id = 'CMP_FAIL_TEST';"
            ).fetchone()
            self.assertIsNone(row)
        finally:
            conn.close()

    def test_concurrent_read_safety(self):
        """6. Verify multiple concurrent readers operate cleanly without lock contention in WAL mode."""
        init_analysis_db(self.tmp_dir)
        conn1 = connect_analysis_db(self.tmp_dir)
        conn2 = connect_analysis_db(self.tmp_dir)
        try:
            with conn1:
                conn1.execute(
                    """
                    INSERT INTO research_campaigns (
                        campaign_id, campaign_name, context_key, created_at, updated_at
                    ) VALUES ('CMP_001', 'Read Concurrency Test', 'NIFTY_3s_DIR_5m_R001', '2026-08-19T00:00:00Z', '2026-08-19T00:00:00Z');
                    """
                )

            # Reader on conn2 can read immediately
            row = conn2.execute("SELECT campaign_name FROM research_campaigns WHERE campaign_id = 'CMP_001';").fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["campaign_name"], "Read Concurrency Test")
        finally:
            conn1.close()
            conn2.close()

    def test_evidence_db_immutability(self):
        """7. Verify initializing analysis.db does NOT touch or mutate feature_recommendation_evidence.db."""
        ev_path = "apps/feature_recommendation_evidence.db"
        self.assertTrue(os.path.isfile(ev_path))
        with open(ev_path, "rb") as fh:
            sha_before = hashlib.sha256(fh.read()).hexdigest()

        # Initialize analysis.db in tmp_dir
        init_analysis_db(self.tmp_dir)

        with open(ev_path, "rb") as fh:
            sha_after = hashlib.sha256(fh.read()).hexdigest()

        self.assertEqual(sha_before, sha_after)
        self.assertEqual(sha_after, "6f91afca94ec87a1210d8f4bcef356b9c16a6ef5a488268f41c5b1b81431ade2")


if __name__ == "__main__":
    unittest.main()
