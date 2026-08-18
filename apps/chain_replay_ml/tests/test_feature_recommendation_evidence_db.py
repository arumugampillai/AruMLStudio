"""Comprehensive test suite for Feature Recommendation Evidence DB, Dual Projections & Pre-Training Gate."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest

from chain_replay_ml.production_validation.dataset_context import (
    LEGACY_UNKNOWN_CONTEXT_ID,
    build_context_key,
    build_dataset_context,
    generate_context_id,
    resolve_context_from_model_package,
    resolve_context_or_legacy,
)
from chain_replay_ml.production_validation.evidence_store import (
    append_validation_evidence,
    compute_feature_identity_key,
    ensure_schema,
    get_connection,
    get_experimental_lineage_summaries,
    get_feature_context_summaries,
    query_blocked_candidates,
    rebuild_all_projections,
)
from chain_replay_ml.production_validation.recommendation_migration import (
    is_migration_completed,
    migrate_legacy_recommendation_json,
)
from chain_replay_ml.production_validation.recommendation_policy import (
    RecommendationPolicy,
    ScoringPolicy,
    compute_evidence_score,
    load_recommendation_policy,
    save_recommendation_policy,
)


class TestDatasetContext(unittest.TestCase):
    def test_context_id_generation_deterministic(self) -> None:
        key1 = build_context_key(
            market="NIFTY", sampling_interval_sec=3, sliding_window="atm_15", feature_project_id="all"
        )
        key2 = build_context_key(
            market="nifty", sampling_interval_sec=3, sliding_window="ATM_15", feature_project_id="ALL"
        )
        self.assertEqual(key1, key2)
        cid1 = generate_context_id(key1)
        cid2 = generate_context_id(key2)
        self.assertEqual(cid1, cid2)
        self.assertTrue(cid1.startswith("ctx_"))

    def test_context_isolation_different_markets_intervals(self) -> None:
        ctx_nifty_3s = build_dataset_context(
            market="NIFTY", sampling_interval_sec=3, sliding_window="atm_15", feature_project_id="all"
        )
        ctx_nifty_6s = build_dataset_context(
            market="NIFTY", sampling_interval_sec=6, sliding_window="atm_15", feature_project_id="all"
        )
        ctx_sensex_1s = build_dataset_context(
            market="SENSEX", sampling_interval_sec=1, sliding_window="standard", feature_project_id="all"
        )
        self.assertNotEqual(ctx_nifty_3s.context_id, ctx_nifty_6s.context_id)
        self.assertNotEqual(ctx_nifty_3s.context_id, ctx_sensex_1s.context_id)
        self.assertNotEqual(ctx_nifty_6s.context_id, ctx_sensex_1s.context_id)

    def test_feature_identity_keys(self) -> None:
        reg_key = compute_feature_identity_key("registry", "oi_pcr")
        self.assertEqual(reg_key, "registry:oi_pcr")

        base_key = compute_feature_identity_key("base_pipeline", "nifty_spot_lag_6s")
        self.assertEqual(base_key, "base_pipeline:nifty_spot_lag_6s")

        exp_key = compute_feature_identity_key(
            "experimental", "nifty_spot_lag_6s", pipeline_id="PL_0005", pipeline_snapshot_id="snap_123"
        )
        self.assertEqual(exp_key, "exp:nifty_spot_lag_6s:PL_0005:snap_123")


class TestEvidenceStoreAndProjections(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.ctx = build_dataset_context(
            market="NIFTY", sampling_interval_sec=3, sliding_window="atm_15", feature_project_id="all"
        )
        self.conn = get_connection(self.tmp)

    def tearDown(self) -> None:
        self.conn.close()

    def test_append_evidence_and_projections(self) -> None:
        rows = [
            {
                "evidence_id": "ev_001",
                "feature_name": "feat_exp_1",
                "feature_source": "experimental",
                "pipeline_id": "PL_0002",
                "pipeline_snapshot_id": "snap_aaa",
                "recommendation": "KEEP",
                "validation_run_id": "run_1",
                "model_name": "Model_A",
                "run_timestamp": "2026-08-16T10:00:00Z",
            },
            {
                "evidence_id": "ev_002",
                "feature_name": "feat_exp_1",
                "feature_source": "experimental",
                "pipeline_id": "PL_0002",
                "pipeline_snapshot_id": "snap_aaa",
                "recommendation": "KEEP",
                "validation_run_id": "run_2",
                "model_name": "Model_B",
                "run_timestamp": "2026-08-16T11:00:00Z",
            },
            {
                "evidence_id": "ev_003",
                "feature_name": "feat_exp_1",
                "feature_source": "experimental",
                "pipeline_id": "PL_0002",
                "pipeline_snapshot_id": "snap_aaa",
                "recommendation": "KEEP",
                "validation_run_id": "run_3",
                "model_name": "Model_C",
                "run_timestamp": "2026-08-16T12:00:00Z",
            },
        ]
        res = append_validation_evidence(self.conn, context=self.ctx, evidence_rows=rows)
        self.assertEqual(res["inserted"], 3)

        # Verify context summary
        ctx_sums = get_feature_context_summaries(self.conn, self.ctx.context_id)
        self.assertEqual(len(ctx_sums), 1)
        self.assertEqual(ctx_sums[0]["feature_name"], "feat_exp_1")
        self.assertEqual(ctx_sums[0]["keep_runs"], 3)
        self.assertEqual(ctx_sums[0]["consecutive_keep_count"], 3)
        # Context summary status CANNOT be promotion_candidate (only active, held, blocked, alert)
        self.assertEqual(ctx_sums[0]["lifecycle_status"], "active")

        # Verify lineage summary -> SHOULD be promotion_candidate because 3 consecutive KEEPs on unique models
        lin_sums = get_experimental_lineage_summaries(self.conn, self.ctx.context_id)
        self.assertEqual(len(lin_sums), 1)
        self.assertEqual(lin_sums[0]["pipeline_id"], "PL_0002")
        self.assertEqual(lin_sums[0]["pipeline_snapshot_id"], "snap_aaa")
        self.assertEqual(lin_sums[0]["lifecycle_status"], "promotion_candidate")
        self.assertGreaterEqual(lin_sums[0]["lineage_evidence_score"], 75.0)

    def test_experimental_remove_blocks_in_context(self) -> None:
        rows = [
            {
                "evidence_id": "ev_r1",
                "feature_name": "bad_exp_feat",
                "feature_source": "experimental",
                "pipeline_id": "PL_0001",
                "pipeline_snapshot_id": "snap_1",
                "recommendation": "REMOVE",
                "validation_run_id": "run_r1",
                "model_name": "Model_X",
                "run_timestamp": "2026-08-16T10:00:00Z",
            },
            {
                "evidence_id": "ev_r2",
                "feature_name": "bad_exp_feat",
                "feature_source": "experimental",
                "pipeline_id": "PL_0001",
                "pipeline_snapshot_id": "snap_1",
                "recommendation": "REMOVE",
                "validation_run_id": "run_r2",
                "model_name": "Model_Y",
                "run_timestamp": "2026-08-16T11:00:00Z",
            },
        ]
        append_validation_evidence(self.conn, context=self.ctx, evidence_rows=rows)

        # Check context summary status is blocked
        ctx_sums = get_feature_context_summaries(self.conn, self.ctx.context_id)
        self.assertEqual(len(ctx_sums), 1)
        self.assertEqual(ctx_sums[0]["lifecycle_status"], "blocked")

        # Check pre-training elimination gate
        blocked = query_blocked_candidates(
            self.conn,
            context_id=self.ctx.context_id,
            candidate_names=["bad_exp_feat", "other_feat"],
        )
        self.assertEqual(blocked, {"bad_exp_feat"})

        # Verify that a different context (e.g. SENSEX) is NOT blocked
        ctx_sensex = build_dataset_context(
            market="SENSEX", sampling_interval_sec=1, sliding_window="standard", feature_project_id="all"
        )
        blocked_sensex = query_blocked_candidates(
            self.conn,
            context_id=ctx_sensex.context_id,
            candidate_names=["bad_exp_feat"],
        )
        self.assertEqual(blocked_sensex, set())

    def test_base_pipeline_and_registry_never_blocked(self) -> None:
        rows = [
            {
                "evidence_id": "ev_b1",
                "feature_name": "base_feat_1",
                "feature_source": "base_pipeline",
                "recommendation": "REMOVE",
                "validation_run_id": "run_b1",
                "model_name": "Model_B1",
                "run_timestamp": "2026-08-16T10:00:00Z",
            },
            {
                "evidence_id": "ev_b2",
                "feature_name": "base_feat_1",
                "feature_source": "base_pipeline",
                "recommendation": "REMOVE",
                "validation_run_id": "run_b2",
                "model_name": "Model_B2",
                "run_timestamp": "2026-08-16T11:00:00Z",
            },
            {
                "evidence_id": "ev_reg1",
                "feature_name": "reg_feat_1",
                "feature_source": "registry",
                "recommendation": "REMOVE",
                "validation_run_id": "run_r1",
                "model_name": "Model_R1",
                "run_timestamp": "2026-08-16T10:00:00Z",
            },
            {
                "evidence_id": "ev_reg2",
                "feature_name": "reg_feat_1",
                "feature_source": "registry",
                "recommendation": "REMOVE",
                "validation_run_id": "run_r2",
                "model_name": "Model_R2",
                "run_timestamp": "2026-08-16T11:00:00Z",
            },
            {
                "evidence_id": "ev_reg3",
                "feature_name": "reg_feat_1",
                "feature_source": "registry",
                "recommendation": "REMOVE",
                "validation_run_id": "run_r3",
                "model_name": "Model_R3",
                "run_timestamp": "2026-08-16T12:00:00Z",
            },
        ]
        append_validation_evidence(self.conn, context=self.ctx, evidence_rows=rows)

        ctx_sums = {r["feature_name"]: r for r in get_feature_context_summaries(self.conn, self.ctx.context_id)}
        self.assertEqual(ctx_sums["base_feat_1"]["lifecycle_status"], "alert")
        self.assertEqual(ctx_sums["reg_feat_1"]["lifecycle_status"], "alert")

        # Base and Registry features must NEVER appear in candidate blocking query
        blocked = query_blocked_candidates(
            self.conn,
            context_id=self.ctx.context_id,
            candidate_names=["base_feat_1", "reg_feat_1"],
        )
        self.assertEqual(blocked, set())

    def test_rebuild_all_projections_is_deterministic(self) -> None:
        rows = [
            {
                "evidence_id": "ev_1",
                "feature_name": "f1",
                "feature_source": "experimental",
                "pipeline_id": "PL_0001",
                "pipeline_snapshot_id": "snap_1",
                "recommendation": "KEEP",
                "validation_run_id": "r1",
                "model_name": "M1",
                "run_timestamp": "2026-08-16T10:00:00Z",
            },
            {
                "evidence_id": "ev_2",
                "feature_name": "f1",
                "feature_source": "experimental",
                "pipeline_id": "PL_0001",
                "pipeline_snapshot_id": "snap_1",
                "recommendation": "REMOVE",
                "validation_run_id": "r2",
                "model_name": "M2",
                "run_timestamp": "2026-08-16T11:00:00Z",
            },
        ]
        append_validation_evidence(self.conn, context=self.ctx, evidence_rows=rows)
        sums_before = get_feature_context_summaries(self.conn, self.ctx.context_id)
        lins_before = get_experimental_lineage_summaries(self.conn, self.ctx.context_id)

        # Run disaster recovery rebuild
        rebuild_all_projections(self.conn)
        sums_after = get_feature_context_summaries(self.conn, self.ctx.context_id)
        lins_after = get_experimental_lineage_summaries(self.conn, self.ctx.context_id)

        # Pop projection_rebuilt_at timestamp for deterministic content assertion
        for s in sums_before + sums_after:
            s.pop("projection_rebuilt_at", None)
        for l in lins_before + lins_after:
            l.pop("projection_rebuilt_at", None)

        self.assertEqual(sums_before, sums_after)
        self.assertEqual(lins_before, lins_after)


class TestRecommendationMigration(unittest.TestCase):
    def test_idempotent_migration(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            json_file = os.path.join(tmp, "feature_recommendation_history.json")
            legacy_data = {
                "version": 1,
                "entries": [
                    {
                        "id": 1,
                        "feature_name": "oi_pcr",
                        "model_name": "NonExistentModel",
                        "recommendation": "REMOVE",
                        "generated_date": "2026-08-16T10:00:00Z",
                        "production_validation_run_id": "run-legacy-1",
                    }
                ],
            }
            with open(json_file, "w", encoding="utf-8") as fh:
                json.dump(legacy_data, fh)

            # First migration
            res1 = migrate_legacy_recommendation_json(tmp)
            self.assertEqual(res1["status"], "completed")
            self.assertEqual(res1["migrated_entries"], 1)
            self.assertEqual(res1["legacy_unknown_count"], 1)

            conn = get_connection(tmp)
            self.assertTrue(is_migration_completed(conn))

            # Second migration -> Should be idempotent no-op
            res2 = migrate_legacy_recommendation_json(tmp)
            self.assertEqual(res2["status"], "already_completed")
            self.assertEqual(res2["migrated_entries"], 0)

            # Legacy unknown should NOT participate in candidate blocking
            blocked = query_blocked_candidates(
                conn,
                context_id=LEGACY_UNKNOWN_CONTEXT_ID,
                candidate_names=["oi_pcr"],
            )
            self.assertEqual(blocked, set())
            conn.close()


class TestAutoCandidateGenerationGate(unittest.TestCase):
    def test_gate_blocks_experimental_candidates(self) -> None:
        from chain_replay_ml.dataset_builder.pipeline_registry_store import (
            add_candidate_features,
            create_pipeline,
            ensure_default_existing_pipeline,
            load_store,
            save_store,
        )
        from master_dataset_tk.auto_candidate_generation import generate_pipeline_candidate_names

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            os.makedirs(data_dir, exist_ok=True)

            # Create pipeline PL_0002 with source candidate feature
            store = ensure_default_existing_pipeline(data_dir)
            create_pipeline(store, name="Test Pipeline 2", pipeline_type="manual")
            add_candidate_features(store, "PL_0002", ["source_a"])
            save_store(data_dir, store)

            # Block feature 'source_a_lag_6s' in NIFTY 3s context
            ctx = build_dataset_context(
                market="NIFTY", sampling_interval_sec=3, sliding_window="standard", feature_project_id="all"
            )
            conn = get_connection(data_dir)
            try:
                rows = [
                    {
                        "evidence_id": "b_1",
                        "feature_name": "source_a_lag_6s",
                        "feature_source": "experimental",
                        "pipeline_id": "PL_0001",
                        "pipeline_snapshot_id": "snap_1",
                        "recommendation": "REMOVE",
                        "validation_run_id": "run_1",
                        "model_name": "M1",
                        "run_timestamp": "2026-08-16T10:00:00Z",
                    },
                    {
                        "evidence_id": "b_2",
                        "feature_name": "source_a_lag_6s",
                        "feature_source": "experimental",
                        "pipeline_id": "PL_0001",
                        "pipeline_snapshot_id": "snap_1",
                        "recommendation": "REMOVE",
                        "validation_run_id": "run_2",
                        "model_name": "M2",
                        "run_timestamp": "2026-08-16T11:00:00Z",
                    },
                ]
                append_validation_evidence(conn, context=ctx, evidence_rows=rows)
            finally:
                conn.close()

            # Run candidate generation for PL_0002 on NIFTY 3s using pipeline source
            prefs = {
                "source": "pipeline",
                "market": "NIFTY",
                "sliding_window": "standard",
                "feature_project_id": "all",
                "transformations": {"lag": True},
                "horizons_sec": [6, 12],
            }
            report = generate_pipeline_candidate_names(
                chart_dir=tmp,
                pipeline_id="PL_0002",
                interval_sec=3,
                candidate_prefs=prefs,
            )

            # Verify that source_a_lag_6s was blocked by evidence gate
            self.assertIn("source_a_lag_6s", report.evidence_blocked_names)
            self.assertEqual(report.candidates_rejected_evidence_blocked, 1)
            self.assertNotIn("source_a_lag_6s", report.new_names)
            # source_a_lag_12s was NOT blocked and is admitted
            self.assertIn("source_a_lag_12s", report.new_names)

            # Verify that creating a brand new pipeline PL_0003 with source_a CANNOT bypass the block
            create_pipeline(store, name="Test Pipeline 3", pipeline_type="manual")
            add_candidate_features(store, "PL_0003", ["source_a"])
            save_store(data_dir, store)
            report_pl3 = generate_pipeline_candidate_names(
                chart_dir=tmp,
                pipeline_id="PL_0003",
                interval_sec=3,
                candidate_prefs=prefs,
            )
            self.assertIn("source_a_lag_6s", report_pl3.evidence_blocked_names)
            self.assertNotIn("source_a_lag_6s", report_pl3.new_names)


if __name__ == "__main__":
    unittest.main()
