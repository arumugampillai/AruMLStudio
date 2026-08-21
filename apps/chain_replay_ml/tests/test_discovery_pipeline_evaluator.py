"""Unit tests for Phase 4: Real-Data Discovery Feature Evaluator & Chronological Walk-Forward."""

from __future__ import annotations

import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from chain_replay_ml.discovery_pipeline.evaluator import (
    DiscoveryFeatureEvaluator,
    generate_chronological_splits,
)
from chain_replay_ml.discovery_pipeline.persistence import (
    init_discovery_pipeline_tables,
    load_discovered_features,
    persist_discovered_features,
    persist_discovery_pipeline,
)
from chain_replay_ml.discovery_pipeline.synthesizer import generate_discovery_features_from_dataset
from chain_replay_ml.discovery_pipeline.types import (
    DiscoveredFeatureSpec,
    DiscoveryLifecycleStatus,
    DiscoveryPipelineBudget,
    DiscoveryPipelineSpec,
    GeneratorStrategy,
    compute_formula_hash,
)


class TestDiscoveryFeatureEvaluator(unittest.TestCase):
    """Test suite for chronological walk-forward splits, target isolation, baseline comparison, and campaign isolation."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        np.random.seed(42)
        n = 600
        # Feature with true predictive signal
        signal = np.sin(np.linspace(0, 10, n))
        self.df = pd.DataFrame({
            "base_f1": np.random.normal(0, 1, n),
            "base_f2": np.random.normal(0, 1, n),
            "base_f3": np.random.normal(0, 1, n),
            "pred_signal": signal + np.random.normal(0, 0.2, n),
            "label_up_5pct_5m": (signal + np.random.normal(0, 0.5, n) > 0).astype(int),
            "target_unused": np.random.choice([0, 1], n),
        })

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_chronological_split_integrity(self):
        """Verify strict chronological forward splits without data leakage."""
        splits = generate_chronological_splits(600, n_splits=5, min_train_ratio=0.50)
        self.assertEqual(len(splits), 5)

        for i, (train_idx, val_idx) in enumerate(splits):
            # Check non-empty
            self.assertGreater(len(train_idx), 0)
            self.assertGreater(len(val_idx), 0)

            # Check strictly disjoint
            self.assertEqual(len(set(train_idx).intersection(set(val_idx))), 0)

            # Check chronological causality: max(train) < min(val)
            self.assertLess(train_idx.max(), val_idx.min())

            # Check expanding train window
            if i > 0:
                prev_train_idx, _ = splits[i - 1]
                self.assertGreater(len(train_idx), len(prev_train_idx))

    def test_target_isolation_and_error_handling(self):
        """Verify target column isolation and missing target handling."""
        pipe_id = "DP_CAMP_ERR_001"
        spec = DiscoveredFeatureSpec(
            feature_id="DF_1",
            pipeline_id=pipe_id,
            feature_name="f_test",
            formula_expression="col('base_f1') + 1",
            formula_hash=compute_formula_hash("col('base_f1') + 1"),
            generator_strategy=GeneratorStrategy.NONLINEAR,
            parent_features=["base_f1"],
            generation_discovered=1,
        )

        # Missing target should raise ValueError
        with self.assertRaises(ValueError):
            DiscoveryFeatureEvaluator.evaluate_features_on_dataset(
                self.df,
                data_dir=self.test_dir,
                pipeline_id=pipe_id,
                campaign_id="CAMP_ERR_001",
                base_feature_names=["base_f1", "base_f2"],
                discovery_features=[spec],
                target_column="non_existent_target",
            )

    def test_incremental_feature_evaluation(self):
        """Verify baseline vs augmented model evaluation and incremental metric calculation."""
        pipe_id = "DP_CAMP_EVAL_001"
        camp_id = "CAMP_EVAL_001"

        pipe = DiscoveryPipelineSpec(
            pipeline_id=pipe_id,
            campaign_id=camp_id,
            context_key="TEST_CONTEXT",
            dataset_name="test_ds",
            dataset_snapshot_hash="snap_123",
            base_feature_count=3,
            base_feature_names=["base_f1", "base_f2", "base_f3"],
        )
        persist_discovery_pipeline(self.test_dir, pipe)

        # 1. Feature with high signal (pred_signal * 2)
        feat_good = DiscoveredFeatureSpec(
            feature_id="DF_GOOD_01",
            pipeline_id=pipe_id,
            feature_name="synth_good",
            formula_expression="col('pred_signal') * 2.0",
            formula_hash=compute_formula_hash("col('pred_signal') * 2.0"),
            generator_strategy=GeneratorStrategy.NONLINEAR,
            parent_features=["pred_signal"],
            generation_discovered=1,
        )

        # 2. Pure noise feature
        feat_noise = DiscoveredFeatureSpec(
            feature_id="DF_NOISE_02",
            pipeline_id=pipe_id,
            feature_name="synth_noise",
            formula_expression="col('base_f1') * col('base_f2')",
            formula_hash=compute_formula_hash("col('base_f1') * col('base_f2')"),
            generator_strategy=GeneratorStrategy.INTERACTION,
            parent_features=["base_f1", "base_f2"],
            generation_discovered=1,
        )

        persist_discovered_features(self.test_dir, [feat_good, feat_noise])

        res = DiscoveryFeatureEvaluator.evaluate_features_on_dataset(
            self.df,
            data_dir=self.test_dir,
            pipeline_id=pipe_id,
            campaign_id=camp_id,
            base_feature_names=["base_f1", "base_f2", "base_f3"],
            discovery_features=[feat_good, feat_noise],
            target_column="label_up_5pct_5m",
            generation_number=1,
            n_splits=5,
        )

        self.assertEqual(res["evaluated_features_count"], 2)
        self.assertIn("baseline_metrics", res)
        self.assertGreater(res["baseline_metrics"]["mean_roc_auc"], 0.0)

        # Reload from DB and verify updated telemetry
        loaded = load_discovered_features(self.test_dir, pipe_id)
        self.assertEqual(len(loaded), 2)
        good_loaded = next(f for f in loaded if f.feature_id == "DF_GOOD_01")
        noise_loaded = next(f for f in loaded if f.feature_id == "DF_NOISE_02")

        # Telemetry verification
        self.assertEqual(good_loaded.total_evaluations, 1)
        self.assertIn("delta_auc", good_loaded.metadata)
        self.assertIn("fold_consistency", good_loaded.metadata)

        # Good signal feature should achieve higher score than noise
        self.assertGreater(good_loaded.evidence_score, noise_loaded.evidence_score)

    def test_campaign_isolation_during_evaluation(self):
        """Verify evaluating DP_CAMP_A cannot modify features in DP_CAMP_B."""
        pipe_a = "DP_CAMP_A"
        pipe_b = "DP_CAMP_B"

        feat_a = DiscoveredFeatureSpec(
            feature_id="DF_A",
            pipeline_id=pipe_a,
            feature_name="feat_a",
            formula_expression="col('base_f1') + 1",
            formula_hash=compute_formula_hash("col('base_f1') + 1"),
            generator_strategy=GeneratorStrategy.NONLINEAR,
            parent_features=["base_f1"],
            generation_discovered=1,
            evidence_score=10.0,
        )
        feat_b = DiscoveredFeatureSpec(
            feature_id="DF_B",
            pipeline_id=pipe_b,
            feature_name="feat_b",
            formula_expression="col('base_f2') + 2",
            formula_hash=compute_formula_hash("col('base_f2') + 2"),
            generator_strategy=GeneratorStrategy.NONLINEAR,
            parent_features=["base_f2"],
            generation_discovered=1,
            evidence_score=99.0,  # Untouched initial score
        )

        persist_discovered_features(self.test_dir, [feat_a])
        persist_discovered_features(self.test_dir, [feat_b])

        # Evaluate DP_CAMP_A only
        DiscoveryFeatureEvaluator.evaluate_features_on_dataset(
            self.df,
            data_dir=self.test_dir,
            pipeline_id=pipe_a,
            campaign_id="CAMP_A",
            base_feature_names=["base_f1", "base_f2"],
            discovery_features=[feat_a],
            target_column="label_up_5pct_5m",
        )

        # Verify DP_CAMP_B remains completely unchanged
        feats_b = load_discovered_features(self.test_dir, pipe_b)
        self.assertEqual(len(feats_b), 1)
        self.assertEqual(feats_b[0].evidence_score, 99.0)
        self.assertEqual(feats_b[0].total_evaluations, 0)


if __name__ == "__main__":
    unittest.main()
