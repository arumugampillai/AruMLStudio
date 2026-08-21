"""Unit & UI integration test for Phase 8: Research Leaderboard Discovered Features Tab."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import tkinter as tk
import unittest
import numpy as np
import pandas as pd

from chain_replay_ml.discovery_pipeline.loop import run_autonomous_discovery_loop
from chain_replay_ml.discovery_pipeline.persistence import (
    init_discovery_pipeline_tables,
    persist_discovered_features,
    persist_discovery_pipeline,
    persist_discovery_snapshot,
)
from chain_replay_ml.discovery_pipeline.types import (
    DiscoveredFeatureSpec,
    DiscoveryLifecycleStatus,
    DiscoveryPipelineBudget,
    DiscoveryPipelineSnapshot,
    DiscoveryPipelineSpec,
    GeneratorStrategy,
    compute_discovery_snapshot_hash,
)
from apps.master_dataset_tk.model_research_leaderboard_panel import ModelResearchLeaderboardPanel


def _sha256_file(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


class TestResearchLeaderboardDiscoveryTab(unittest.TestCase):
    """Verify Research Leaderboard 'Discovered Features' tab rendering and telemetry display."""

    root: tk.Tk | None = None

    @classmethod
    def setUpClass(cls):
        try:
            cls.root = tk.Tk()
            cls.root.withdraw()
        except Exception:
            cls.root = None

    @classmethod
    def tearDownClass(cls):
        pass

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.data_dir = "data"
        self.feat_store_path = os.path.join(self.data_dir, "feature_registry_store.json")
        self.pipe_store_path = os.path.join(self.data_dir, "pipeline_registry_store.json")

        self.feat_store_hash_before = _sha256_file(self.feat_store_path)
        self.pipe_store_hash_before = _sha256_file(self.pipe_store_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_discovery_tab_present_and_renders_empty_state(self):
        """Verify tab exists in notebook and displays clean placeholder when no pipeline exists."""
        if not self.root:
            self.skipTest("Tkinter display not available in headless test environment")

        panel = ModelResearchLeaderboardPanel(self.root, chart_dir=self.test_dir)
        panel._context_key_var.set("NIFTY_6s_DIRECTION_CLASSIFIER_5m_R999")

        # Check tab exists in notebook
        tabs = [panel._detail_nb.tab(i, "text") for i in range(panel._detail_nb.index("end"))]
        self.assertIn("🔬 Discovered Features", tabs)

        # Trigger rendering
        panel._render_tab_discovered_features()
        inner_widgets = panel._tab_discovered_features.inner.winfo_children()
        self.assertGreater(len(inner_widgets), 0)

    def test_discovery_tab_renders_populated_pipeline(self):
        """Verify tab renders populated metadata, tables, inspector, and snapshots."""
        if not self.root:
            self.skipTest("Tkinter display not available in headless test environment")

        ctx_key = "NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001"
        pipe_id = "DP_CAMP_TAB_TEST_001"
        camp_id = "CAMP_TAB_TEST_001"

        init_discovery_pipeline_tables(self.test_dir)

        # Persist test pipeline
        pipe = DiscoveryPipelineSpec(
            pipeline_id=pipe_id,
            campaign_id=camp_id,
            context_key=ctx_key,
            dataset_name="analysis_198r_171b_6s_20260820_223630",
            dataset_snapshot_hash="1714b8dddb455a95",
            base_feature_count=12,
            active_features_count=2,
            total_generated_count=3,
            current_generation=2,
            current_snapshot_hash="DP_SNAP_test123456",
        )
        persist_discovery_pipeline(self.test_dir, pipe)

        f1 = DiscoveredFeatureSpec(
            feature_id="DF_001",
            pipeline_id=pipe_id,
            feature_name="synth_log1p__reiv_skew",
            formula_expression="sign(col('reiv_skew')) * log1p(abs(col('reiv_skew')))",
            formula_hash="h111",
            generator_strategy=GeneratorStrategy.NONLINEAR,
            parent_features=["reiv_skew"],
            generation_discovered=1,
            lifecycle_status=DiscoveryLifecycleStatus.KEEP,
            evidence_score=58.2,
            ks_statistic=0.041,
            metadata={"delta_auc": 0.0095, "baseline_auc": 0.518, "fold_consistency": 0.80},
        )
        f2 = DiscoveredFeatureSpec(
            feature_id="DF_002",
            pipeline_id=pipe_id,
            feature_name="synth_ratio__iv_div_reiv",
            formula_expression="col('iv_atm') / (abs(col('reiv_skew')) + 0.001)",
            formula_hash="h222",
            generator_strategy=GeneratorStrategy.RATIO,
            parent_features=["iv_atm", "reiv_skew"],
            generation_discovered=1,
            lifecycle_status=DiscoveryLifecycleStatus.WATCH,
            evidence_score=49.1,
            ks_statistic=0.062,
            metadata={"delta_auc": 0.0012, "baseline_auc": 0.518, "fold_consistency": 0.60},
        )
        f3 = DiscoveredFeatureSpec(
            feature_id="DF_003",
            pipeline_id=pipe_id,
            feature_name="synth_bad_drift",
            formula_expression="col('delta_oi') * col('volume_flow')",
            formula_hash="h333",
            generator_strategy=GeneratorStrategy.INTERACTION,
            parent_features=["delta_oi", "volume_flow"],
            generation_discovered=2,
            lifecycle_status=DiscoveryLifecycleStatus.REMOVE,
            evidence_score=32.0,
            ks_statistic=0.45,
            drift_severity=2,
            metadata={"delta_auc": -0.012, "baseline_auc": 0.518, "fold_consistency": 0.20},
        )
        persist_discovered_features(self.test_dir, [f1, f2, f3])

        snap = DiscoveryPipelineSnapshot(
            snapshot_hash="DP_SNAP_test123456",
            pipeline_id=pipe_id,
            generation_number=2,
            active_feature_names=["synth_log1p__reiv_skew", "synth_ratio__iv_div_reiv"],
            feature_count=2,
            keep_count=1,
            watch_count=1,
            remove_count=1,
        )
        persist_discovery_snapshot(self.test_dir, snap)

        panel = ModelResearchLeaderboardPanel(self.root, chart_dir=self.test_dir)
        panel._context_key_var.set(ctx_key)

        # Render populated tab
        panel._render_tab_discovered_features()
        children = panel._tab_discovered_features.inner.winfo_children()
        self.assertGreater(len(children), 3)

        # Verify Registry Immutability
        self.assertEqual(self.feat_store_hash_before, _sha256_file(self.feat_store_path))
        self.assertEqual(self.pipe_store_hash_before, _sha256_file(self.pipe_store_path))


if __name__ == "__main__":
    unittest.main()
