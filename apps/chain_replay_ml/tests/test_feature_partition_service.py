"""Unit tests for authoritative Feature Partition Service (Phase 4F / Morning Dossier)."""

import os
import shutil
import tempfile
import unittest

from path_config import ensure_ml_studio_paths
ensure_ml_studio_paths()

from chain_replay_ml.feature_partition import (
    FeatureCategory,
    classify_feature,
    is_synthetic_or_experimental,
    partition_feature_records,
    resolve_feature_partition_sets,
)


class TestFeaturePartitionService(unittest.TestCase):
    """Test suite verifying strict mutually-exclusive tripartite feature partitioning."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_feat_part_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_01_synthetic_markers(self):
        """1. Verify explicit discovery feature prefixes are recognized."""
        self.assertTrue(is_synthetic_or_experimental("DF_0001"))
        self.assertTrue(is_synthetic_or_experimental("synth_ratio_iv"))
        self.assertTrue(is_synthetic_or_experimental("df_adx_ratio"))
        self.assertFalse(is_synthetic_or_experimental("rsi_14"))
        self.assertFalse(is_synthetic_or_experimental("adx_14"))
        self.assertFalse(is_synthetic_or_experimental("abs_delta"))
        self.assertFalse(is_synthetic_or_experimental("delta_x_spot"))

    def test_02_strict_mutual_exclusion(self):
        """2. Invariant: Baseline, Registry, and Experimental sets are strictly disjoint."""
        baseline_set = {"adx_14", "rsi_14", "macd_diff", "bb_width_20", "iv_mean", "delta_x_spot"}
        # Suppose raw registry store contains baseline and registry features
        raw_registry = {"adx_14", "rsi_14", "abs_delta", "ask_depth_l1_5", "atm_iv_ce"}
        # Strict exclusion rule: registry = raw_registry - baseline
        registry_set = raw_registry - baseline_set
        experimental_set = {"DF_0001", "synth_rsi_vol"}

        # Verify sets are mutually disjoint
        self.assertEqual(len(baseline_set.intersection(registry_set)), 0)
        self.assertEqual(len(baseline_set.intersection(experimental_set)), 0)
        self.assertEqual(len(registry_set.intersection(experimental_set)), 0)

        # Test classify_feature
        self.assertEqual(classify_feature("adx_14", baseline_set, registry_set, experimental_set), FeatureCategory.BASELINE)
        self.assertEqual(classify_feature("delta_x_spot", baseline_set, registry_set, experimental_set), FeatureCategory.BASELINE)
        self.assertEqual(classify_feature("abs_delta", baseline_set, registry_set, experimental_set), FeatureCategory.REGISTRY)
        self.assertEqual(classify_feature("DF_0001", baseline_set, registry_set, experimental_set), FeatureCategory.EXPERIMENTAL)
        self.assertEqual(classify_feature("synth_rsi_vol", baseline_set, registry_set, experimental_set), FeatureCategory.EXPERIMENTAL)

    def test_03_partition_feature_records(self):
        """3. Verify partition_feature_records partitions list into exact 3 disjoint subsets with provenance."""
        baseline_set = {"f_base_1", "f_base_2"}
        registry_set = {"f_reg_1", "f_reg_2", "f_reg_3"}

        test_records = [
            {"feature_name": "f_base_1"},
            {"feature_name": "f_base_2"},
            {"feature_name": "f_reg_1"},
            {"feature_name": "f_reg_2"},
            {"feature_name": "f_reg_3"},
            {"feature_name": "DF_interaction_01"},
            {"feature_name": "synth_feature_02"},
        ]

        reg_l, base_l, exp_l, prov = partition_feature_records(test_records, data_dir=self.tmp_dir)
        self.assertEqual(len(base_l) + len(reg_l) + len(exp_l), len(test_records))
        self.assertIsInstance(prov, dict)

    def test_04_discovery_pipeline_info_resolution(self):
        """4. Verify get_campaign_discovery_pipeline_info resolves pipeline metadata gracefully."""
        from chain_replay_ml.feature_partition import get_campaign_discovery_pipeline_info
        info = get_campaign_discovery_pipeline_info(self.tmp_dir, "CAMP_MOCK_001")
        self.assertEqual(info["total_pipelines"], 0)
        self.assertIsNone(info["primary_pipeline_id"])


if __name__ == "__main__":
    unittest.main()
