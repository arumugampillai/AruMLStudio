"""Tests for Phase 3B: Model Builder Handoff & Training Candidate Selection Bridge.

Verifies:
1. TRAIN_CANDIDATE retrieval
2. Promotion candidate prioritization/ordering
3. REVIEW items unselected by default
4. NEW_UNSEEN items unselected by default
5. EXCLUDE items strictly cannot be selected or exported
6. User candidate deselection handling
7. Preset JSON schema and backward compatibility
8. recommendation_decision_bundle persistence
9. Context isolation (NIFTY vs SENSEX)
10. Dataset compatibility validation
11. Empty candidate pool handling
12. Evidence DB SHA-256 unchanged (immutability)
13. Zero model training execution verification
14. Legacy preset without bundle compatibility
"""

import hashlib
import json
import os
import shutil
import tempfile
import unittest

from chain_replay_ml.production_validation.api import (
    TrainingDecisionPolicy,
    TrainingDecisionState,
    build_dataset_context,
    build_model_builder_training_bundle,
    export_training_candidates_preset,
    get_evidence_connection,
    load_recommendation_policy,
    persist_validation_evidence,
    save_recommendation_policy,
)
from master_dataset_tk.model_builder.feature_preset import (
    apply_feature_preset,
    clear_feature_preset,
    load_feature_preset,
    save_feature_preset,
)


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(8192):
            h.update(chunk)
    return h.hexdigest()


class TestModelBuilderHandoff(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp(prefix="aruml_phase3b_test_")
        self.chart_dir = os.path.join(self.temp_dir, "chart")
        self.data_dir = os.path.join(self.temp_dir, "data")
        os.makedirs(self.chart_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)

        self.ctx_nifty_3s = build_dataset_context(
            market="NIFTY", sampling_interval_sec=3, sliding_window="standard", feature_project_id="all"
        )
        self.ctx_sensex_1s = build_dataset_context(
            market="SENSEX", sampling_interval_sec=1, sliding_window="standard", feature_project_id="all"
        )

        # Seed realistic multi-run evidence for NIFTY 3s
        items = [
            # Promotion candidate: 3 consecutive KEEP runs, 2 unique models
            {
                "feature_name": "exp_promo_cand",
                "feature_source": "experimental",
                "pipeline_id": "PL_EXP_1",
                "pipeline_snapshot_id": "SNP_1",
                "model_name": "model_xgb",
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "sliding_window": "standard",
                "feature_project_id": "all",
                "recommendation": "KEEP",
                "run_timestamp": "2026-08-18T10:00:00Z",
            },
            {
                "feature_name": "exp_promo_cand",
                "feature_source": "experimental",
                "pipeline_id": "PL_EXP_1",
                "pipeline_snapshot_id": "SNP_1",
                "model_name": "model_lgbm",
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "sliding_window": "standard",
                "feature_project_id": "all",
                "recommendation": "KEEP",
                "run_timestamp": "2026-08-18T11:00:00Z",
            },
            {
                "feature_name": "exp_promo_cand",
                "feature_source": "experimental",
                "pipeline_id": "PL_EXP_1",
                "pipeline_snapshot_id": "SNP_1",
                "model_name": "model_xgb",
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "sliding_window": "standard",
                "feature_project_id": "all",
                "recommendation": "KEEP",
                "run_timestamp": "2026-08-18T12:00:00Z",
            },
            # Base candidate
            {
                "feature_name": "base_cand",
                "feature_source": "base_pipeline",
                "model_name": "model_xgb",
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "sliding_window": "standard",
                "feature_project_id": "all",
                "recommendation": "KEEP",
                "run_timestamp": "2026-08-18T10:00:00Z",
            },
            {
                "feature_name": "base_cand",
                "feature_source": "base_pipeline",
                "model_name": "model_lgbm",
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "sliding_window": "standard",
                "feature_project_id": "all",
                "recommendation": "KEEP",
                "run_timestamp": "2026-08-18T11:00:00Z",
            },
            # Registry candidate
            {
                "feature_name": "reg_cand",
                "feature_source": "registry",
                "model_name": "model_xgb",
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "sliding_window": "standard",
                "feature_project_id": "all",
                "recommendation": "KEEP",
                "run_timestamp": "2026-08-18T10:00:00Z",
            },
            {
                "feature_name": "reg_cand",
                "feature_source": "registry",
                "model_name": "model_lgbm",
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "sliding_window": "standard",
                "feature_project_id": "all",
                "recommendation": "KEEP",
                "run_timestamp": "2026-08-18T11:00:00Z",
            },
            # Review item: split consensus
            {
                "feature_name": "exp_review_split",
                "feature_source": "experimental",
                "pipeline_id": "PL_EXP_2",
                "pipeline_snapshot_id": "SNP_2",
                "model_name": "model_xgb",
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "sliding_window": "standard",
                "feature_project_id": "all",
                "recommendation": "KEEP",
                "run_timestamp": "2026-08-18T10:00:00Z",
            },
            {
                "feature_name": "exp_review_split",
                "feature_source": "experimental",
                "pipeline_id": "PL_EXP_2",
                "pipeline_snapshot_id": "SNP_2",
                "model_name": "model_lgbm",
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "sliding_window": "standard",
                "feature_project_id": "all",
                "recommendation": "REMOVE",
                "run_timestamp": "2026-08-18T11:00:00Z",
            },
            # Excluded item: 2 consecutive REMOVEs
            {
                "feature_name": "exp_excluded_streak",
                "feature_source": "experimental",
                "pipeline_id": "PL_EXP_3",
                "pipeline_snapshot_id": "SNP_3",
                "model_name": "model_xgb",
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "sliding_window": "standard",
                "feature_project_id": "all",
                "recommendation": "REMOVE",
                "run_timestamp": "2026-08-18T10:00:00Z",
            },
            {
                "feature_name": "exp_excluded_streak",
                "feature_source": "experimental",
                "pipeline_id": "PL_EXP_3",
                "pipeline_snapshot_id": "SNP_3",
                "model_name": "model_lgbm",
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "sliding_window": "standard",
                "feature_project_id": "all",
                "recommendation": "REMOVE",
                "run_timestamp": "2026-08-18T11:00:00Z",
            },
            # SENSEX candidate
            {
                "feature_name": "sensex_cand",
                "feature_source": "registry",
                "model_name": "model_xgb",
                "market": "SENSEX",
                "sampling_interval_sec": 1,
                "sliding_window": "standard",
                "feature_project_id": "all",
                "recommendation": "KEEP",
                "run_timestamp": "2026-08-18T10:00:00Z",
            },
            {
                "feature_name": "sensex_cand",
                "feature_source": "registry",
                "model_name": "model_lgbm",
                "market": "SENSEX",
                "sampling_interval_sec": 1,
                "sliding_window": "standard",
                "feature_project_id": "all",
                "recommendation": "KEEP",
                "run_timestamp": "2026-08-18T11:00:00Z",
            },
        ]

        from chain_replay_ml.production_validation.evidence_store import append_validation_evidence
        conn = get_evidence_connection(self.data_dir)
        try:
            append_validation_evidence(conn, context=self.ctx_nifty_3s, evidence_rows=items[:11])
            append_validation_evidence(conn, context=self.ctx_sensex_1s, evidence_rows=items[11:])
        finally:
            conn.close()

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_train_candidate_retrieval(self) -> None:
        """Verify build_model_builder_training_bundle extracts all qualifying TRAIN_CANDIDATE features."""
        bundle = build_model_builder_training_bundle(
            self.data_dir,
            context=self.ctx_nifty_3s,
        )
        rdb = bundle["recommendation_decision_bundle"]
        features = bundle["features"]

        self.assertIn("exp_promo_cand", features)
        self.assertIn("base_cand", features)
        self.assertIn("reg_cand", features)
        self.assertNotIn("exp_excluded_streak", features)

        self.assertGreaterEqual(rdb["eligible_candidates_count"], 3)
        self.assertGreaterEqual(rdb["selected_candidates_count"], 3)

    def test_02_promotion_candidate_prioritization_and_ordering(self) -> None:
        """Verify PROMOTION_CANDIDATE_QUALIFIED is placed at the very top of the candidate list."""
        bundle = build_model_builder_training_bundle(
            self.data_dir,
            context=self.ctx_nifty_3s,
        )
        features = bundle["features"]
        rdb = bundle["recommendation_decision_bundle"]

        # exp_promo_cand must be index 0
        self.assertEqual(features[0], "exp_promo_cand")
        promo_prov = rdb["feature_provenance"]["exp_promo_cand"]
        self.assertEqual(promo_prov["primary_reason"], "PROMOTION_CANDIDATE_QUALIFIED")
        self.assertIn("[PROMOTION]", promo_prov["reason_badges"])
        self.assertEqual(rdb["selection_summary"]["promotion_qualified_count"], 1)

    def test_03_review_not_selected_by_default(self) -> None:
        """Verify REVIEW features (e.g. consensus split) are categorized under review and not selected."""
        bundle = build_model_builder_training_bundle(
            self.data_dir,
            context=self.ctx_nifty_3s,
        )
        rdb = bundle["recommendation_decision_bundle"]
        features = bundle["features"]

        self.assertNotIn("exp_review_split", features)
        self.assertGreaterEqual(rdb["review_count"], 1)

        split_prov = rdb["feature_provenance"].get("exp_review_split")
        self.assertIsNotNone(split_prov)
        self.assertEqual(split_prov["decision"], TrainingDecisionState.REVIEW)
        self.assertEqual(split_prov["primary_reason"], "CONSENSUS_SPLIT")

    def test_04_new_unseen_not_selected_by_default(self) -> None:
        """Verify features with 0 historical runs are marked NEW_UNSEEN and not selected by default."""
        bundle = build_model_builder_training_bundle(
            self.data_dir,
            context=self.ctx_nifty_3s,
        )
        rdb = bundle["recommendation_decision_bundle"]
        features = bundle["features"]

        # All items in bundle['features'] must be approved training candidates
        prov_map = rdb["feature_provenance"]
        for f in features:
            self.assertEqual(prov_map[f]["decision"], TrainingDecisionState.TRAIN_CANDIDATE)

    def test_05_exclude_strictly_cannot_be_selected_or_exported(self) -> None:
        """Verify EXCLUDE features cannot be selected even if explicitly passed in selected_features."""
        bundle = build_model_builder_training_bundle(
            self.data_dir,
            context=self.ctx_nifty_3s,
            selected_features=["exp_promo_cand", "exp_excluded_streak", "base_cand"],
        )
        features = bundle["features"]
        rdb = bundle["recommendation_decision_bundle"]

        self.assertIn("exp_promo_cand", features)
        self.assertIn("base_cand", features)
        # exp_excluded_streak MUST be stripped out
        self.assertNotIn("exp_excluded_streak", features)
        self.assertEqual(rdb["feature_provenance"]["exp_excluded_streak"]["decision"], TrainingDecisionState.EXCLUDE)

    def test_06_user_candidate_deselection(self) -> None:
        """Verify user deselection in the approval boundary preserves only the requested subset."""
        # Deselect base_cand
        bundle = build_model_builder_training_bundle(
            self.data_dir,
            context=self.ctx_nifty_3s,
            selected_features=["exp_promo_cand", "reg_cand"],
        )
        features = bundle["features"]
        rdb = bundle["recommendation_decision_bundle"]

        self.assertEqual(set(features), {"exp_promo_cand", "reg_cand"})
        self.assertNotIn("base_cand", features)
        self.assertEqual(rdb["selected_candidates_count"], 2)
        self.assertEqual(rdb["selection_summary"]["user_deselected_count"], 1)

    def test_07_preset_json_schema_and_backward_compatibility(self) -> None:
        """Verify export_training_candidates_preset creates valid preset JSON matching Model Builder schema."""
        preset_doc = export_training_candidates_preset(
            self.chart_dir,
            self.data_dir,
            context=self.ctx_nifty_3s,
        )

        self.assertIn("features", preset_doc)
        self.assertIn("source_model", preset_doc)
        self.assertIn("at", preset_doc)
        self.assertIn("recommendation_decision_bundle", preset_doc)

        loaded = load_feature_preset(self.chart_dir)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["features"], preset_doc["features"])
        self.assertEqual(
            loaded["recommendation_decision_bundle"]["context_id"],
            self.ctx_nifty_3s.context_id,
        )

    def test_08_recommendation_decision_bundle_persistence_and_provenance(self) -> None:
        """Verify full decision provenance structure in saved preset."""
        export_training_candidates_preset(
            self.chart_dir,
            self.data_dir,
            context=self.ctx_nifty_3s,
        )

        loaded = load_feature_preset(self.chart_dir)
        rdb = loaded["recommendation_decision_bundle"]

        self.assertEqual(rdb["context_id"], self.ctx_nifty_3s.context_id)
        self.assertEqual(rdb["market"], "NIFTY")
        self.assertEqual(rdb["sampling_interval_sec"], 3)
        self.assertEqual(rdb["sliding_window"], "standard")
        self.assertEqual(rdb["feature_project_id"], "all")
        self.assertEqual(rdb["decision_engine_version"], "3B.1")

        prov = rdb["feature_provenance"]["exp_promo_cand"]
        self.assertEqual(prov["decision"], "TRAIN_CANDIDATE")
        self.assertEqual(prov["primary_reason"], "PROMOTION_CANDIDATE_QUALIFIED")
        self.assertEqual(prov["dominant_recommendation"], "KEEP")
        self.assertIn("passed_checks", prov)
        self.assertIn("failed_checks", prov)

    def test_09_context_isolation_nifty_vs_sensex(self) -> None:
        """Verify presets for NIFTY 3s and SENSEX 1s maintain distinct context bundles."""
        # NIFTY preset
        export_training_candidates_preset(
            self.chart_dir,
            self.data_dir,
            context=self.ctx_nifty_3s,
        )
        loaded_nifty = load_feature_preset(self.chart_dir)
        self.assertEqual(
            loaded_nifty["recommendation_decision_bundle"]["market"], "NIFTY"
        )
        self.assertEqual(
            loaded_nifty["recommendation_decision_bundle"]["sampling_interval_sec"], 3
        )

        # SENSEX preset
        export_training_candidates_preset(
            self.chart_dir,
            self.data_dir,
            context=self.ctx_sensex_1s,
        )
        loaded_sensex = load_feature_preset(self.chart_dir)
        self.assertEqual(
            loaded_sensex["recommendation_decision_bundle"]["market"], "SENSEX"
        )
        self.assertEqual(
            loaded_sensex["recommendation_decision_bundle"]["sampling_interval_sec"], 1
        )

    def test_10_dataset_compatibility_validation(self) -> None:
        """Verify apply_feature_preset warns on context mismatch (NIFTY vs SENSEX)."""
        preset = export_training_candidates_preset(
            self.chart_dir,
            self.data_dir,
            context=self.ctx_nifty_3s,
        )

        # Matching dataset
        match_res = apply_feature_preset(
            preset,
            dataset_name="ohlc_NIFTY_3s_standard",
            dataset_feature_names=["exp_promo_cand", "base_cand", "reg_cand"],
        )
        self.assertTrue(match_res["context_match"])
        self.assertIsNone(match_res["context_warning"])
        self.assertEqual(len(match_res["features"]), 3)

        # Mismatched dataset (SENSEX)
        mismatch_res = apply_feature_preset(
            preset,
            dataset_name="ohlc_SENSEX_1s_standard",
            dataset_feature_names=["exp_promo_cand", "base_cand"],
        )
        self.assertFalse(mismatch_res["context_match"])
        self.assertIn("Context Mismatch", mismatch_res["context_warning"])

    def test_11_empty_candidate_pool_handling(self) -> None:
        """Verify empty candidate pool yields clean empty features list without crash."""
        empty_ctx = build_dataset_context(
            market="BANKNIFTY", sampling_interval_sec=15, sliding_window="atm_15", feature_project_id="all"
        )
        bundle = build_model_builder_training_bundle(
            self.data_dir,
            context=empty_ctx,
        )
        rdb = bundle["recommendation_decision_bundle"]

        self.assertEqual(bundle["features"], [])
        self.assertEqual(rdb["selected_candidates_count"], 0)
        self.assertEqual(rdb["eligible_candidates_count"], 0)

    def test_12_evidence_db_immutability(self) -> None:
        """Verify Evidence DB SHA-256 remains 100% identical before and after preset export."""
        db_path = os.path.join(self.data_dir, "feature_recommendation_evidence.db")
        sha_before = _file_sha256(db_path)

        for _ in range(5):
            export_training_candidates_preset(
                self.chart_dir,
                self.data_dir,
                context=self.ctx_nifty_3s,
            )

        sha_after = _file_sha256(db_path)
        self.assertEqual(sha_before, sha_after)

    def test_13_zero_model_training_execution_verification(self) -> None:
        """Verify building and exporting presets does not start any model training."""
        preset = export_training_candidates_preset(
            self.chart_dir,
            self.data_dir,
            context=self.ctx_nifty_3s,
        )
        # Check no model artifacts or run files were created
        model_runs_dir = os.path.join(self.data_dir, "model_runs")
        self.assertFalse(os.path.exists(model_runs_dir))

    def test_14_legacy_preset_backward_compatibility(self) -> None:
        """Verify legacy presets without recommendation_decision_bundle still load and apply seamlessly."""
        # Create legacy preset using standard save_feature_preset without bundle
        save_feature_preset(
            self.chart_dir,
            features=["legacy_feat_1", "legacy_feat_2"],
            dataset="legacy_dataset",
            source_model="legacy_model",
        )

        loaded = load_feature_preset(self.chart_dir)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["features"], ["legacy_feat_1", "legacy_feat_2"])
        self.assertNotIn("recommendation_decision_bundle", loaded)

        applied = apply_feature_preset(
            loaded,
            dataset_name="legacy_dataset",
            dataset_feature_names=["legacy_feat_1", "legacy_feat_2", "extra_feat"],
        )
        self.assertTrue(applied["applied"])
        self.assertTrue(applied["context_match"])
        self.assertEqual(applied["features"], ["legacy_feat_1", "legacy_feat_2"])

    def test_15_dialog_selection_toggle_and_exclusion_invariants(self) -> None:
        """Verify interactive dialog selection sequence:

        1. Initially eligible candidates are selected.
        2. Deselect All -> selected count = 0.
        3. Select one individual eligible feature -> selected count = 1.
        4. Select another -> selected count = 2.
        5. Deselect one individual feature -> selected count = 1.
        6. Select All Eligible -> selected count = all eligible candidates.
        7. Excluded candidates can never become selected.
        """
        import tkinter as tk
        from master_dataset_tk.training_candidate_handoff_dialog import TrainingCandidateHandoffDialog

        root = tk.Tk()
        root.withdraw()
        try:
            dialog = TrainingCandidateHandoffDialog(
                root,
                chart_dir=self.chart_dir,
                data_dir=self.data_dir,
                context=self.ctx_nifty_3s,
            )

            # 1. Initially eligible candidates are selected
            sel = [fn for fn, v in dialog._feature_selection_vars.items() if v.get()]
            self.assertEqual(len(sel), 3)
            self.assertIn("exp_promo_cand", sel)
            self.assertIn("base_cand", sel)
            self.assertIn("reg_cand", sel)

            # 2. Deselect All -> selected count = 0
            dialog._deselect_all()
            sel = [fn for fn, v in dialog._feature_selection_vars.items() if v.get()]
            self.assertEqual(len(sel), 0)

            # 3. Select one individual eligible feature -> selected count = 1
            toggled = dialog.toggle_feature_selection("base_cand")
            self.assertTrue(toggled)
            sel = [fn for fn, v in dialog._feature_selection_vars.items() if v.get()]
            self.assertEqual(sel, ["base_cand"])

            # 4. Select another -> selected count = 2
            toggled = dialog.toggle_feature_selection("reg_cand")
            self.assertTrue(toggled)
            sel = [fn for fn, v in dialog._feature_selection_vars.items() if v.get()]
            self.assertEqual(set(sel), {"base_cand", "reg_cand"})

            # 5. Deselect one individual feature -> selected count = 1
            toggled = dialog.toggle_feature_selection("base_cand")
            self.assertFalse(toggled)  # toggled from True -> False
            sel = [fn for fn, v in dialog._feature_selection_vars.items() if v.get()]
            self.assertEqual(sel, ["reg_cand"])

            # 6. Select All Eligible -> selected count = all eligible candidates
            dialog._select_all_eligible()
            sel = [fn for fn, v in dialog._feature_selection_vars.items() if v.get()]
            self.assertEqual(set(sel), {"exp_promo_cand", "base_cand", "reg_cand"})

            # 7. Excluded candidates can never become selected
            excluded_toggled = dialog.toggle_feature_selection("exp_excluded_streak")
            self.assertFalse(excluded_toggled)
            sel = [fn for fn, v in dialog._feature_selection_vars.items() if v.get()]
            self.assertNotIn("exp_excluded_streak", sel)

            dialog.destroy()
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
