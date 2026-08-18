"""Unit tests for Feature Studio Recommendation UI Integration & Three Feature Populations."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.production_validation.api import (
    build_dataset_context,
    get_population_recommendations,
    rebuild_all_projections,
)
from chain_replay_ml.production_validation.dataset_context import (
    LEGACY_UNKNOWN_CONTEXT_ID,
)
from chain_replay_ml.production_validation.evidence_store import (
    append_validation_evidence,
    get_connection,
)


class TestFeatureStudioRecommendationUI(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.ctx_nifty = build_dataset_context(
            market="NIFTY", sampling_interval_sec=3, sliding_window="standard", feature_project_id="all"
        )
        self.ctx_sensex = build_dataset_context(
            market="SENSEX", sampling_interval_sec=1, sliding_window="standard", feature_project_id="all"
        )
        self.ctx_legacy = build_dataset_context(
            market="UNKNOWN", sampling_interval_sec=1, sliding_window="unknown", feature_project_id="unknown"
        )

        conn = get_connection(self.tmp)
        try:
            # 1. Feature Registry evidence in NIFTY 3s
            reg_rows = [
                {
                    "evidence_id": "r_1",
                    "feature_name": "chain_pcr",
                    "feature_source": "registry",
                    "recommendation": "REMOVE",
                    "validation_run_id": "run_r1",
                    "model_name": "Model_R1",
                    "run_timestamp": "2026-08-16T10:00:00Z",
                },
                {
                    "evidence_id": "r_2",
                    "feature_name": "chain_pcr",
                    "feature_source": "registry",
                    "recommendation": "REMOVE",
                    "validation_run_id": "run_r2",
                    "model_name": "Model_R2",
                    "run_timestamp": "2026-08-16T11:00:00Z",
                },
                {
                    "evidence_id": "r_3",
                    "feature_name": "chain_pcr",
                    "feature_source": "registry",
                    "recommendation": "REMOVE",
                    "validation_run_id": "run_r3",
                    "model_name": "Model_R3",
                    "run_timestamp": "2026-08-16T12:00:00Z",
                },
                {
                    "evidence_id": "r_4",
                    "feature_name": "atm_pcr",
                    "feature_source": "registry",
                    "recommendation": "KEEP",
                    "validation_run_id": "run_r1",
                    "model_name": "Model_R1",
                    "run_timestamp": "2026-08-16T10:00:00Z",
                },
            ]
            append_validation_evidence(conn, context=self.ctx_nifty, evidence_rows=reg_rows)

            # 2. Base Pipeline evidence in NIFTY 3s
            base_rows = [
                {
                    "evidence_id": "b_1",
                    "feature_name": "base_high_perf",
                    "feature_source": "base_pipeline",
                    "recommendation": "KEEP",
                    "validation_run_id": "run_b1",
                    "model_name": "Model_B1",
                    "run_timestamp": "2026-08-16T10:00:00Z",
                },
                {
                    "evidence_id": "b_2",
                    "feature_name": "base_high_perf",
                    "feature_source": "base_pipeline",
                    "recommendation": "KEEP",
                    "validation_run_id": "run_b2",
                    "model_name": "Model_B2",
                    "run_timestamp": "2026-08-16T11:00:00Z",
                },
                {
                    "evidence_id": "b_3",
                    "feature_name": "base_low_perf",
                    "feature_source": "base_pipeline",
                    "recommendation": "REMOVE",
                    "validation_run_id": "run_b1",
                    "model_name": "Model_B1",
                    "run_timestamp": "2026-08-16T10:00:00Z",
                },
                {
                    "evidence_id": "b_4",
                    "feature_name": "base_low_perf",
                    "feature_source": "base_pipeline",
                    "recommendation": "REMOVE",
                    "validation_run_id": "run_b2",
                    "model_name": "Model_B2",
                    "run_timestamp": "2026-08-16T11:00:00Z",
                },
            ]
            append_validation_evidence(conn, context=self.ctx_nifty, evidence_rows=base_rows)

            # 3. Experimental features in NIFTY 3s
            # Feature 1: Promotion Candidate (3 consecutive KEEPs on unique models)
            # Feature 2: Blocked Candidate (2 consecutive REMOVEs on unique models)
            exp_rows = [
                {
                    "evidence_id": "e_k1",
                    "feature_name": "candidate_stellar",
                    "feature_source": "experimental",
                    "pipeline_id": "PL_0005",
                    "pipeline_snapshot_id": "snap_v1",
                    "recommendation": "KEEP",
                    "validation_run_id": "run_e1",
                    "model_name": "Model_E1",
                    "run_timestamp": "2026-08-16T10:00:00Z",
                },
                {
                    "evidence_id": "e_k2",
                    "feature_name": "candidate_stellar",
                    "feature_source": "experimental",
                    "pipeline_id": "PL_0005",
                    "pipeline_snapshot_id": "snap_v1",
                    "recommendation": "KEEP",
                    "validation_run_id": "run_e2",
                    "model_name": "Model_E2",
                    "run_timestamp": "2026-08-16T11:00:00Z",
                },
                {
                    "evidence_id": "e_k3",
                    "feature_name": "candidate_stellar",
                    "feature_source": "experimental",
                    "pipeline_id": "PL_0005",
                    "pipeline_snapshot_id": "snap_v1",
                    "recommendation": "KEEP",
                    "validation_run_id": "run_e3",
                    "model_name": "Model_E3",
                    "run_timestamp": "2026-08-16T12:00:00Z",
                },
                {
                    "evidence_id": "e_r1",
                    "feature_name": "candidate_failed",
                    "feature_source": "experimental",
                    "pipeline_id": "PL_0005",
                    "pipeline_snapshot_id": "snap_v1",
                    "recommendation": "REMOVE",
                    "validation_run_id": "run_e1",
                    "model_name": "Model_E1",
                    "run_timestamp": "2026-08-16T10:00:00Z",
                },
                {
                    "evidence_id": "e_r2",
                    "feature_name": "candidate_failed",
                    "feature_source": "experimental",
                    "pipeline_id": "PL_0005",
                    "pipeline_snapshot_id": "snap_v1",
                    "recommendation": "REMOVE",
                    "validation_run_id": "run_e2",
                    "model_name": "Model_E2",
                    "run_timestamp": "2026-08-16T11:00:00Z",
                },
            ]
            append_validation_evidence(conn, context=self.ctx_nifty, evidence_rows=exp_rows)

            # 4. SENSEX 1s context record (isolated)
            append_validation_evidence(
                conn,
                context=self.ctx_sensex,
                evidence_rows=[
                    {
                        "evidence_id": "sx_1",
                        "feature_name": "candidate_failed",
                        "feature_source": "experimental",
                        "pipeline_id": "PL_0009",
                        "pipeline_snapshot_id": "snap_sx1",
                        "recommendation": "KEEP",
                        "validation_run_id": "run_sx1",
                        "model_name": "Model_SX1",
                        "run_timestamp": "2026-08-16T10:00:00Z",
                    }
                ],
            )
        finally:
            conn.close()

    def test_feature_registry_population_evidence(self) -> None:
        rows = get_population_recommendations(
            self.tmp,
            population="registry",
            context_id=self.ctx_nifty.context_id,
        )
        self.assertEqual(len(rows), 2)
        by_name = {r["feature_name"]: r for r in rows}

        # chain_pcr had 3 REMOVEs -> Status should be ALERT (never BLOCKED)
        self.assertEqual(by_name["chain_pcr"]["lifecycle_status"], "alert")
        self.assertNotEqual(by_name["chain_pcr"]["lifecycle_status"], "blocked")
        self.assertEqual(by_name["chain_pcr"]["remove_runs"], 3)
        self.assertLess(by_name["chain_pcr"]["evidence_score"], 0)

        # atm_pcr had 1 KEEP -> Status should be ACTIVE
        self.assertEqual(by_name["atm_pcr"]["lifecycle_status"], "active")
        self.assertGreater(by_name["atm_pcr"]["evidence_score"], 0)

    def test_base_pipeline_population_ranking_and_evidence(self) -> None:
        rows = get_population_recommendations(
            self.tmp,
            population="base_pipeline",
            context_id=self.ctx_nifty.context_id,
        )
        self.assertEqual(len(rows), 2)
        # Strong KEEP should rank #1
        self.assertEqual(rows[0]["feature_name"], "base_high_perf")
        self.assertEqual(rows[0]["priority_rank"], 1)
        self.assertGreater(rows[0]["evidence_score"], 0)
        self.assertEqual(rows[0]["lifecycle_status"], "active")

        # Weak REMOVE should rank #2 (status is alert or active, never blocked)
        self.assertEqual(rows[1]["feature_name"], "base_low_perf")
        self.assertEqual(rows[1]["priority_rank"], 2)
        self.assertNotEqual(rows[1]["lifecycle_status"], "blocked")

    def test_selected_experimental_lineage_and_promotion(self) -> None:
        rows = get_population_recommendations(
            self.tmp,
            population="experimental",
            context_id=self.ctx_nifty.context_id,
        )
        self.assertEqual(len(rows), 2)
        by_name = {r["feature_name"]: r for r in rows}

        # candidate_stellar satisfies promotion policy
        stellar = by_name["candidate_stellar"]
        self.assertEqual(stellar["pipeline_id"], "PL_0005")
        self.assertEqual(stellar["pipeline_snapshot_id"], "snap_v1")
        self.assertEqual(stellar["lifecycle_status"], "promotion_candidate")
        self.assertEqual(stellar["context_status"], "active")
        self.assertEqual(stellar["consecutive_keep_count"], 3)
        self.assertGreaterEqual(stellar["lineage_evidence_score"], 75.0)

        # candidate_failed triggered context-level blocked gate
        failed = by_name["candidate_failed"]
        self.assertEqual(failed["lifecycle_status"], "blocked")
        self.assertEqual(failed["context_status"], "blocked")
        self.assertEqual(failed["consecutive_remove_count"], 2)

    def test_dataset_context_isolation(self) -> None:
        # Query SENSEX 1s context
        rows_sx = get_population_recommendations(
            self.tmp,
            population="experimental",
            context_id=self.ctx_sensex.context_id,
        )
        self.assertEqual(len(rows_sx), 1)
        # In SENSEX 1s, candidate_failed is KEEP on PL_0009 and is ACTIVE, NOT blocked!
        self.assertEqual(rows_sx[0]["feature_name"], "candidate_failed")
        self.assertEqual(rows_sx[0]["pipeline_id"], "PL_0009")
        self.assertEqual(rows_sx[0]["lifecycle_status"], "active")
        self.assertEqual(rows_sx[0]["context_status"], "active")

    def test_viewer_dialog_initialization_with_policy_tab(self) -> None:
        import tkinter as tk
        from master_dataset_tk.feature_recommendation_viewer import open_feature_recommendation_viewer

        root = tk.Tk()
        root.withdraw()
        try:
            dlg = open_feature_recommendation_viewer(
                root,
                chart_dir=self.tmp,
                initial_market="NIFTY",
                initial_interval_sec=3,
                initial_sliding_window="standard",
                initial_feature_project_id="all",
            )
            # Verify that all 5 tabs are present
            tab_texts = [dlg._notebook.tab(i, "text") for i in range(dlg._notebook.index("end"))]
            self.assertIn("1. Feature Registry", tab_texts)
            self.assertIn("2. Base Pipeline", tab_texts)
            self.assertIn("3. Selected Experimental", tab_texts)
            self.assertIn("4. Raw Evidence Log", tab_texts)
            self.assertIn("5. Policy Settings", tab_texts)

            # Verify policy fields populated
            self.assertEqual(dlg._exp_promo_keep_streak_var.get(), "3")
            self.assertEqual(dlg._weight_keep_var.get(), "25.0")
            dlg.destroy()
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
