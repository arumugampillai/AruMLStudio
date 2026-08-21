"""Unit tests for Phase 7: Autonomous Discovery Evolutionary Loop Engine."""

from __future__ import annotations

import shutil
import tempfile
import unittest
import numpy as np
import pandas as pd

from chain_replay_ml.discovery_pipeline.loop import (
    run_autonomous_discovery_loop,
    run_discovery_generation,
)
from chain_replay_ml.discovery_pipeline.persistence import (
    load_discovered_features,
    load_discovery_pipeline,
    load_discovery_snapshots_for_pipeline,
)
from chain_replay_ml.discovery_pipeline.types import (
    DiscoveryLifecycleStatus,
    DiscoveryPipelineBudget,
)


class TestDiscoveryPipelineLoop(unittest.TestCase):
    """Test suite for Autonomous Discovery multi-generation evolutionary loop and snapshots."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.camp_id = "CAMP_LOOP_TEST_001"
        self.pipe_id = f"DP_{self.camp_id}"

        np.random.seed(42)
        n = 600
        signal = np.random.normal(0, 1, n)
        self.df = pd.DataFrame({
            "f1": signal + np.random.normal(0, 0.2, n),
            "f2": np.random.normal(0, 1, n),
            "f3": np.random.exponential(1.0, n) + 0.1,
            "label_up_5pct_5m": np.random.choice([0, 1], n, p=[0.5, 0.5]),
        })
        self.base_features = ["f1", "f2", "f3"]

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_single_discovery_generation(self):
        """Verify single generation execution, snapshot creation, and DB state."""
        budget = DiscoveryPipelineBudget(max_new_features_per_gen=4)
        gen_res = run_discovery_generation(
            self.df,
            data_dir=self.test_dir,
            pipeline_id=self.pipe_id,
            campaign_id=self.camp_id,
            generation_number=1,
            base_features=self.base_features,
            budget=budget,
        )

        self.assertEqual(gen_res["generation_number"], 1)
        self.assertTrue(gen_res["snapshot_hash"].startswith("DP_SNAP_"))
        self.assertGreater(gen_res["new_features_generated"], 0)

        # Check DB state
        pipe = load_discovery_pipeline(self.test_dir, self.pipe_id)
        self.assertIsNotNone(pipe)
        self.assertEqual(pipe.current_generation, 1)
        self.assertEqual(pipe.current_snapshot_hash, gen_res["snapshot_hash"])

        snaps = load_discovery_snapshots_for_pipeline(self.test_dir, self.pipe_id)
        self.assertEqual(len(snaps), 1)
        self.assertEqual(snaps[0].generation_number, 1)

    def test_multi_generation_evolution_and_pool_growth(self):
        """Verify Gen 1 -> Gen 2 multi-generation loop evolution and deduplication."""
        budget = DiscoveryPipelineBudget(max_new_features_per_gen=3)
        loop_res = run_autonomous_discovery_loop(
            self.df,
            data_dir=self.test_dir,
            campaign_id=self.camp_id,
            total_generations=2,
            base_features=self.base_features,
            budget=budget,
        )

        self.assertEqual(loop_res["total_generations_completed"], 2)
        self.assertEqual(loop_res["current_generation"], 2)

        # Verify snapshots for both generations
        snaps = load_discovery_snapshots_for_pipeline(self.test_dir, self.pipe_id)
        self.assertEqual(len(snaps), 2)
        self.assertEqual(snaps[0].generation_number, 1)
        self.assertEqual(snaps[1].generation_number, 2)

        # Verify unique formula hashes across generations
        all_feats = load_discovered_features(self.test_dir, self.pipe_id)
        hashes = [f.formula_hash for f in all_feats]
        self.assertEqual(len(hashes), len(set(hashes)), "All discovered features must have unique formula hashes!")


if __name__ == "__main__":
    unittest.main()
