"""Comprehensive Unit Tests for Phase 4D.6: Research Campaign Lifecycle & Champion Transition History."""

import concurrent.futures
import hashlib
import os
import shutil
import tempfile
import unittest

from chain_replay_ml.research_memory import (
    allocate_experiment_slot,
    cancel_campaign,
    complete_campaign,
    create_benchmark_run,
    create_campaign,
    fail_campaign,
    get_campaign,
    get_campaign_experiments,
    get_champion_history_for_context,
    get_latest_champion_transition,
    init_analysis_db,
    link_experiment_to_campaign,
    list_campaigns_for_context,
    pause_campaign,
    persist_context_rankings,
    rank_models_in_context,
    reconstruct_champion_at_timestamp,
    record_champion_transition,
    record_model_benchmark,
    register_or_get_experiment,
    resume_campaign,
    start_campaign,
)


class TestCampaignsAndChampionHistory(unittest.TestCase):
    """Test suite verifying campaign lifecycle, quota control, experiment linking, and immutable champion history."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_camp_champ_")
        init_analysis_db(self.tmp_dir)

        self.spec_trend = {
            "market": "NIFTY",
            "sampling_interval_sec": 3,
            "task_type": "DIRECTION_CLASSIFIER",
            "prediction_horizon": "5m",
            "regime_id": "R001",
            "regime_definition_hash": "def_hash_trend_v1",
            "dataset_snapshot_hash": "ds_hash_20260819",
            "features": ["adx_14", "rsi_14", "atm_iv_pctile"],
            "algorithm": "catboost",
            "hyperparameters": {"depth": 6},
            "walk_forward_config": {"folds": 5, "window_mode": "expanding"},
            "random_seed": 42,
        }

        _, self.rec_trend = register_or_get_experiment(self.tmp_dir, self.spec_trend, model_name="DIR_TREND_CAT_v1")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_campaign_creation_and_fields(self):
        """1. Verify research campaign record creation and initial state."""
        camp_id = create_campaign(
            self.tmp_dir,
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            campaign_name="Overnight Trend Feature Exploration",
            max_experiments_limit=50,
            max_duration_seconds=7200.0,
        )
        self.assertTrue(camp_id.startswith("CAMP_"))

        doc = get_campaign(self.tmp_dir, camp_id)
        self.assertIsNotNone(doc)
        self.assertEqual(doc["status"], "CREATED")
        self.assertEqual(doc["context_key"], "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        self.assertEqual(doc["max_experiments_limit"], 50)
        self.assertEqual(doc["completed_count"], 0)

    def test_campaign_lifecycle_transitions(self):
        """2. Verify valid campaign state progression (CREATED -> RUNNING -> PAUSED -> RUNNING -> COMPLETED)."""
        camp_id = create_campaign(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")

        # Start
        self.assertTrue(start_campaign(self.tmp_dir, camp_id))
        doc = get_campaign(self.tmp_dir, camp_id)
        self.assertEqual(doc["status"], "RUNNING")
        self.assertIsNotNone(doc["start_time"])

        # Pause
        self.assertTrue(pause_campaign(self.tmp_dir, camp_id))
        self.assertEqual(get_campaign(self.tmp_dir, camp_id)["status"], "PAUSED")

        # Resume
        self.assertTrue(resume_campaign(self.tmp_dir, camp_id))
        self.assertEqual(get_campaign(self.tmp_dir, camp_id)["status"], "RUNNING")

        # Complete
        self.assertTrue(complete_campaign(self.tmp_dir, camp_id, termination_reason="ALL_TRIALS_COMPLETED"))
        doc = get_campaign(self.tmp_dir, camp_id)
        self.assertEqual(doc["status"], "COMPLETED")
        self.assertEqual(doc["termination_reason"], "ALL_TRIALS_COMPLETED")
        self.assertIsNotNone(doc["end_time"])

    def test_invalid_lifecycle_transition_rejection(self):
        """3. Verify invalid state transitions raise ValueError (e.g. COMPLETED -> RUNNING)."""
        camp_id = create_campaign(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        start_campaign(self.tmp_dir, camp_id)
        complete_campaign(self.tmp_dir, camp_id)

        # Attempting to re-open COMPLETED campaign must fail
        with self.assertRaises(ValueError):
            start_campaign(self.tmp_dir, camp_id)

        with self.assertRaises(ValueError):
            pause_campaign(self.tmp_dir, camp_id)

    def test_campaign_quota_enforcement(self):
        """4. Verify quota allocation prevents trials beyond max_experiments_limit."""
        camp_id = create_campaign(
            self.tmp_dir,
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            max_experiments_limit=2,
        )
        start_campaign(self.tmp_dir, camp_id)

        # Slot 1
        ok1, idx1, err1 = allocate_experiment_slot(self.tmp_dir, camp_id)
        self.assertTrue(ok1)
        self.assertEqual(idx1, 1)
        link_experiment_to_campaign(
            self.tmp_dir, campaign_id=camp_id, trial_index=idx1, signature_hash=self.rec_trend["signature_hash"]
        )

        # Slot 2
        ok2, idx2, err2 = allocate_experiment_slot(self.tmp_dir, camp_id)
        self.assertTrue(ok2)
        self.assertEqual(idx2, 2)
        link_experiment_to_campaign(
            self.tmp_dir, campaign_id=camp_id, trial_index=idx2, signature_hash=self.rec_trend["signature_hash"]
        )

        # Slot 3 (Quota exhausted)
        ok3, _, err3 = allocate_experiment_slot(self.tmp_dir, camp_id)
        self.assertFalse(ok3)
        self.assertIn("quota exhausted", err3.lower())

    def test_concurrent_quota_consumption(self):
        """5. Verify multi-threaded workers allocate slots safely without index collisions."""
        camp_id = create_campaign(
            self.tmp_dir,
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            max_experiments_limit=10,
        )
        start_campaign(self.tmp_dir, camp_id)

        def worker_task(_):
            ok, idx, _ = allocate_experiment_slot(self.tmp_dir, camp_id)
            if ok:
                link_experiment_to_campaign(
                    self.tmp_dir,
                    campaign_id=camp_id,
                    trial_index=idx,
                    signature_hash=self.rec_trend["signature_hash"],
                )
                return idx
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(worker_task, range(10)))

        allocated = [r for r in results if r is not None]
        self.assertEqual(len(allocated), 10)
        self.assertEqual(len(set(allocated)), 10) # All indices unique

        camp = get_campaign(self.tmp_dir, camp_id)
        self.assertEqual(camp["completed_count"], 10)

    def test_campaign_failure_and_cancellation(self):
        """6. Verify FAILED and CANCELLED terminal states."""
        camp1 = create_campaign(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        start_campaign(self.tmp_dir, camp1)
        fail_campaign(self.tmp_dir, camp1, error_message="Worker Out of Memory")
        self.assertEqual(get_campaign(self.tmp_dir, camp1)["status"], "FAILED")
        self.assertIn("Out of Memory", get_campaign(self.tmp_dir, camp1)["termination_reason"])

        camp2 = create_campaign(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        cancel_campaign(self.tmp_dir, camp2, reason="USER_STOPPED_MANUALLY")
        self.assertEqual(get_campaign(self.tmp_dir, camp2)["status"], "CANCELLED")

    def test_champion_history_record_and_delta(self):
        """7. Verify champion transition record creation and score delta calculation."""
        tid = record_champion_transition(
            self.tmp_dir,
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            new_champion_name="DIR_TREND_CAT_v1",
            new_robustness_score=85.5,
            previous_champion_name="DIR_TREND_XGB_v0",
            previous_robustness_score=80.0,
            promoted_by="HUMAN_RESEARCHER",
            promotion_reason="Superior temporal stability and 5.5pt robustness improvement",
        )
        self.assertGreater(tid, 0)

        latest = get_latest_champion_transition(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        self.assertIsNotNone(latest)
        self.assertEqual(latest["new_champion_name"], "DIR_TREND_CAT_v1")
        self.assertEqual(latest["previous_champion_name"], "DIR_TREND_XGB_v0")
        self.assertEqual(latest["score_delta"], 5.5)
        self.assertEqual(latest["promoted_by"], "HUMAN_RESEARCHER")

    def test_champion_history_append_only_and_time_travel(self):
        """8. Verify append-only transitions and historical time-travel query."""
        key = "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001"

        # Transition 1 at 2026-08-01
        record_champion_transition(
            self.tmp_dir,
            context_key=key,
            new_champion_name="CHAMPION_V1",
            new_robustness_score=75.0,
            transition_timestamp="2026-08-01T10:00:00Z",
        )

        # Transition 2 at 2026-08-10
        record_champion_transition(
            self.tmp_dir,
            context_key=key,
            new_champion_name="CHAMPION_V2",
            new_robustness_score=80.0,
            previous_champion_name="CHAMPION_V1",
            previous_robustness_score=75.0,
            transition_timestamp="2026-08-10T10:00:00Z",
        )

        # Transition 3 at 2026-08-19
        record_champion_transition(
            self.tmp_dir,
            context_key=key,
            new_champion_name="CHAMPION_V3",
            new_robustness_score=85.0,
            previous_champion_name="CHAMPION_V2",
            previous_robustness_score=80.0,
            transition_timestamp="2026-08-19T10:00:00Z",
        )

        history = get_champion_history_for_context(self.tmp_dir, key)
        self.assertEqual(len(history), 3)

        # Time-travel reconstruction at 2026-08-05 -> CHAMPION_V1
        c_aug05 = reconstruct_champion_at_timestamp(self.tmp_dir, key, "2026-08-05T12:00:00Z")
        self.assertEqual(c_aug05["new_champion_name"], "CHAMPION_V1")

        # Time-travel reconstruction at 2026-08-15 -> CHAMPION_V2
        c_aug15 = reconstruct_champion_at_timestamp(self.tmp_dir, key, "2026-08-15T12:00:00Z")
        self.assertEqual(c_aug15["new_champion_name"], "CHAMPION_V2")

        # Latest -> CHAMPION_V3
        latest = get_latest_champion_transition(self.tmp_dir, key)
        self.assertEqual(latest["new_champion_name"], "CHAMPION_V3")

    def test_trend_and_sideways_champion_isolation(self):
        """9. Verify Trend (R001) and Sideways (R002) champion histories are strictly isolated."""
        record_champion_transition(
            self.tmp_dir,
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            new_champion_name="TREND_CHAMPION_V1",
            new_robustness_score=88.0,
        )

        record_champion_transition(
            self.tmp_dir,
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002",
            new_champion_name="SIDE_CHAMPION_V1",
            new_robustness_score=82.0,
        )

        trend_hist = get_champion_history_for_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        side_hist = get_champion_history_for_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002")

        self.assertEqual(len(trend_hist), 1)
        self.assertEqual(trend_hist[0]["new_champion_name"], "TREND_CHAMPION_V1")

        self.assertEqual(len(side_hist), 1)
        self.assertEqual(side_hist[0]["new_champion_name"], "SIDE_CHAMPION_V1")

    def test_campaign_to_ranking_to_champion_lineage(self):
        """10. Verify end-to-end lineage from Campaign -> Experiment -> Benchmark -> Ranking -> Candidate -> Champion History."""
        # 1. Create & Start Campaign
        camp_id = create_campaign(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001", max_experiments_limit=10)
        start_campaign(self.tmp_dir, camp_id)

        # 2. Allocate and link experiment
        _, trial_idx, _ = allocate_experiment_slot(self.tmp_dir, camp_id)
        link_experiment_to_campaign(
            self.tmp_dir, campaign_id=camp_id, trial_index=trial_idx, signature_hash=self.rec_trend["signature_hash"]
        )

        # 3. Create Benchmark Run linked to campaign
        run_id = create_benchmark_run(
            self.tmp_dir,
            campaign_id=camp_id,
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
        )

        # 4. Record Model Benchmark
        record_model_benchmark(
            self.tmp_dir,
            benchmark_run_id=run_id,
            signature_hash=self.rec_trend["signature_hash"],
            model_name="DIR_TREND_CAT_v1",
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            algorithm="catboost",
            dataset_name="trend_3s.parquet",
            feature_count=3,
            primary_metric_name="roc_auc",
            primary_metric_value=0.80,
            fold_metric_mean=0.80,
            fold_metric_std=0.01,
        )

        # 5. Rank Models
        ranked = rank_models_in_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001", benchmark_run_id=run_id)
        persist_context_rankings(self.tmp_dir, benchmark_run_id=run_id, ranked_dossiers=ranked)

        self.assertEqual(ranked[0]["recommendation_status"], "CHAMPION_CANDIDATE")
        cand_score = ranked[0]["robustness_score"]

        # 6. Human Reviewer Approves -> Record in Champion History
        record_champion_transition(
            self.tmp_dir,
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            new_champion_name=ranked[0]["model_name"],
            new_robustness_score=cand_score,
            promoted_by="HUMAN_RESEARCHER",
            promotion_reason=f"Campaign {camp_id} validated candidate",
        )

        # Verify full audit trail
        history = get_champion_history_for_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["new_champion_name"], "DIR_TREND_CAT_v1")

    def test_evidence_db_immutability(self):
        """11. Verify Phase 4D.6 operations do NOT touch or mutate feature_recommendation_evidence.db."""
        ev_path = "apps/feature_recommendation_evidence.db"
        self.assertTrue(os.path.isfile(ev_path))
        with open(ev_path, "rb") as fh:
            sha_before = hashlib.sha256(fh.read()).hexdigest()

        camp_id = create_campaign(self.tmp_dir, context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        start_campaign(self.tmp_dir, camp_id)
        record_champion_transition(
            self.tmp_dir,
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            new_champion_name="TEST_CHAMPION",
            new_robustness_score=80.0,
        )

        with open(ev_path, "rb") as fh:
            sha_after = hashlib.sha256(fh.read()).hexdigest()

        self.assertEqual(sha_before, sha_after)
        self.assertEqual(sha_after, "6f91afca94ec87a1210d8f4bcef356b9c16a6ef5a488268f41c5b1b81431ade2")


if __name__ == "__main__":
    unittest.main()
