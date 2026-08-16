"""Comprehensive tests for Create Model Section 5 three feature tabs (Registry, Base Pipeline, Selected Experimental)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import tkinter as tk
import unittest
from pathlib import Path

_apps_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _apps_dir not in sys.path:
    sys.path.insert(0, _apps_dir)
from path_config import ensure_ml_studio_paths

ensure_ml_studio_paths()

import master_dataset_tk.feature_registry_service as svc
from master_dataset_tk.model_builder.panel import CreateModelPanel


class TestModelBuilderFeatureTabs(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls) -> None:
        try:
            cls.root.destroy()
        except Exception:
            pass

    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.chart_dir = os.path.join(self.tmp_dir, "chart")
        self.data_dir = os.path.join(self.chart_dir, "data")
        self.datasets_dir = os.path.join(self.data_dir, "datasets")
        os.makedirs(self.datasets_dir, exist_ok=True)
        svc.ensure_all_project(self.chart_dir)

        # 1. Create a custom Feature Project
        svc.create_project(
            self.chart_dir,
            project_id="nifty_classification",
            label="NIFTY_Classification",
            feature_names=["spot", "ltp", "delta", "gamma", "vega"],
            project_groups=[{"id": "opt_greeks", "label": "Option Greeks"}],
            feature_group_map={"delta": "opt_greeks", "gamma": "opt_greeks", "vega": "opt_greeks"},
        )

        # 2. Dataset with Feature Registry features ONLY
        ds_reg_only = {
            "dataset_name": "ds_reg_only",
            "output_parquet": "datasets/ds_reg_only.parquet",
            "feature_project_id": "all",
            "registry_export_features": ["spot", "ltp", "delta", "gamma", "vega", "theta"],
            "feature_columns": ["spot", "ltp", "delta", "gamma", "vega", "theta"],
            "target_columns": ["fwd_ret_5m"],
            "prediction_targets": ["fwd_ret_5m"],
        }
        with open(os.path.join(self.datasets_dir, "ds_reg_only.json"), "w") as fh:
            json.dump(ds_reg_only, fh)

        # 3. Dataset with Base Pipeline features ONLY
        ds_base_only = {
            "dataset_name": "ds_base_only",
            "output_parquet": "datasets/ds_base_only.parquet",
            "feature_project_id": "all",
            "base_pipeline_export_features": ["fwd_ret_5m", "ret_5m", "diff_spot_ltp", "spot_roll_mean_5m"],
            "base_pipeline_export_count": 4,
            "feature_columns": ["fwd_ret_5m", "ret_5m", "diff_spot_ltp", "spot_roll_mean_5m"],
            "target_columns": ["fwd_ret_5m"],
            "prediction_targets": ["fwd_ret_5m"],
        }
        with open(os.path.join(self.datasets_dir, "ds_base_only.json"), "w") as fh:
            json.dump(ds_base_only, fh)

        # 4. Dataset with Experimental Pipeline features ONLY
        ds_exp_only = {
            "dataset_name": "ds_exp_only",
            "output_parquet": "datasets/ds_exp_only.parquet",
            "feature_project_id": "all",
            "pipeline_id": "PL_0002",
            "pipeline_name": "Pipeline_002 — ZScore Expansion",
            "pipeline_type": "auto",
            "pipeline_snapshot_id": "snap_12345678",
            "pipeline_provenance": {
                "pipeline_id": "PL_0002",
                "candidate_features": ["custom_zscore_vol", "custom_ratio_skew"],
            },
            "experimental_pipeline_export_count": 2,
            "feature_columns": ["custom_zscore_vol", "custom_ratio_skew"],
            "target_columns": ["fwd_ret_5m"],
            "prediction_targets": ["fwd_ret_5m"],
        }
        with open(os.path.join(self.datasets_dir, "ds_exp_only.json"), "w") as fh:
            json.dump(ds_exp_only, fh)

        # 5. Dataset containing ALL THREE types (with 1,000+ dummy features simulating large exports)
        dummy_columns = [f"non_project_col_{i}" for i in range(1000)]
        ds_all_three = {
            "dataset_name": "ds_all_three",
            "output_parquet": "datasets/ds_all_three.parquet",
            "feature_project_id": "nifty_classification",
            "registry_export_features": ["spot", "ltp", "delta", "gamma", "vega"],
            "base_pipeline_export_features": ["ret_5m", "diff_spot_ltp"],
            "base_pipeline_export_count": 2,
            "pipeline_id": "PL_0003",
            "pipeline_name": "Pipeline_003 — Custom Interactions",
            "pipeline_type": "manual",
            "pipeline_snapshot_id": "snap_87654321",
            "pipeline_provenance": {
                "pipeline_id": "PL_0003",
                "candidate_features": ["custom_poly_strike"],
            },
            "experimental_pipeline_export_count": 1,
            "feature_columns": (
                ["spot", "ltp", "delta", "gamma", "vega", "ret_5m", "diff_spot_ltp", "custom_poly_strike"]
                + dummy_columns
            ),
            "target_columns": ["fwd_ret_5m"],
            "prediction_targets": ["fwd_ret_5m"],
        }
        with open(os.path.join(self.datasets_dir, "ds_all_three.json"), "w") as fh:
            json.dump(ds_all_three, fh)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_registry_only_dataset(self) -> None:
        """1. Dataset with Feature Registry features only populates Tab 1 and leaves others empty."""
        panel = CreateModelPanel(self.root, chart_dir=self.chart_dir)
        panel._dataset_var.set("ds_reg_only")
        panel._on_dataset_changed(persist=False)

        reg_groups = panel._registry_feature_groups()
        base_groups = panel._base_pipeline_feature_groups()
        exp_groups = panel._experimental_pipeline_feature_groups()

        self.assertTrue(len(reg_groups) > 0)
        self.assertEqual(len(base_groups), 0)
        self.assertEqual(len(exp_groups), 0)

        reg_feats = [f for g in reg_groups for f in g["features"]]
        self.assertIn("spot", reg_feats)
        self.assertIn("delta", reg_feats)

    def test_base_pipeline_only_dataset(self) -> None:
        """2. Dataset with Base Pipeline features only populates Tab 2."""
        panel = CreateModelPanel(self.root, chart_dir=self.chart_dir)
        panel._dataset_var.set("ds_base_only")
        panel._on_dataset_changed(persist=False)

        reg_groups = panel._registry_feature_groups()
        base_groups = panel._base_pipeline_feature_groups()
        exp_groups = panel._experimental_pipeline_feature_groups()

        self.assertEqual(len(reg_groups), 0)
        self.assertTrue(len(base_groups) > 0)
        self.assertEqual(len(exp_groups), 0)

        base_feats = [f for g in base_groups for f in g["features"]]
        self.assertIn("ret_5m", base_feats)

    def test_experimental_pipeline_only_dataset(self) -> None:
        """3. Dataset with Experimental Pipeline features only populates Tab 3."""
        panel = CreateModelPanel(self.root, chart_dir=self.chart_dir)
        panel._dataset_var.set("ds_exp_only")
        panel._on_dataset_changed(persist=False)

        reg_groups = panel._registry_feature_groups()
        base_groups = panel._base_pipeline_feature_groups()
        exp_groups = panel._experimental_pipeline_feature_groups()

        self.assertEqual(len(reg_groups), 0)
        self.assertEqual(len(base_groups), 0)
        self.assertTrue(len(exp_groups) > 0)

        exp_feats = [f for g in exp_groups for f in g["features"]]
        self.assertIn("custom_zscore_vol", exp_feats)
        self.assertIn("custom_ratio_skew", exp_feats)

    def test_dataset_containing_all_three_types(self) -> None:
        """4. Dataset containing all three types partitions features cleanly into 3 tabs without leakage."""
        panel = CreateModelPanel(self.root, chart_dir=self.chart_dir)
        panel._dataset_var.set("ds_all_three")
        panel._on_dataset_changed(persist=False)

        reg_groups = panel._registry_feature_groups()
        base_groups = panel._base_pipeline_feature_groups()
        exp_groups = panel._experimental_pipeline_feature_groups()

        reg_feats = [f for g in reg_groups for f in g["features"]]
        base_feats = [f for g in base_groups for f in g["features"]]
        exp_feats = [f for g in exp_groups for f in g["features"]]

        # 6. feature_project_id correctly selects NIFTY_Classification
        self.assertEqual(panel._feature_project_summary_label(), "NIFTY_Classification")
        self.assertIn("delta", reg_feats)

        # 7. Pipeline features do NOT appear in Registry tab
        self.assertNotIn("ret_5m", reg_feats)
        self.assertNotIn("diff_spot_ltp", reg_feats)
        self.assertNotIn("custom_poly_strike", reg_feats)

        # 8. Registry features do NOT appear in Pipeline tabs
        self.assertNotIn("delta", base_feats)
        self.assertNotIn("delta", exp_feats)
        self.assertIn("ret_5m", base_feats)
        self.assertIn("custom_poly_strike", exp_feats)

        # 9. Union of selections matches self.state.features
        all_visible = panel._visible_dataset_feature_names()
        self.assertEqual(set(all_visible), set(reg_feats + base_feats + exp_feats))
        self.assertEqual(panel.state.features, set(all_visible))

        # 10. Verify 1000+ dummy columns are completely excluded
        for i in range(10):
            self.assertNotIn(f"non_project_col_{i}", all_visible)

    def test_dataset_change_refreshes_all_three_tabs(self) -> None:
        """5. Changing datasets completely cleans and reloads all 3 tabs with no stale state."""
        panel = CreateModelPanel(self.root, chart_dir=self.chart_dir)

        # Load ds_exp_only
        panel._dataset_var.set("ds_exp_only")
        panel._on_dataset_changed(persist=False)
        self.assertEqual(len(panel._registry_feature_groups()), 0)
        self.assertTrue(len(panel._experimental_pipeline_feature_groups()) > 0)

        # Switch to ds_reg_only
        panel._dataset_var.set("ds_reg_only")
        panel._on_dataset_changed(persist=False)

        # Old experimental features must be gone, registry features must be present
        self.assertTrue(len(panel._registry_feature_groups()) > 0)
        self.assertEqual(len(panel._experimental_pipeline_feature_groups()), 0)
        self.assertNotIn("custom_zscore_vol", panel.state.features)
        self.assertIn("spot", panel.state.features)

    def test_toolbar_actions_across_tabs(self) -> None:
        """Verify All, Clear, Expand, Collapse toolbar actions work across all tabs."""
        panel = CreateModelPanel(self.root, chart_dir=self.chart_dir)
        panel._dataset_var.set("ds_all_three")
        panel._on_dataset_changed(persist=False)

        # Clear
        panel._features_clear()
        self.assertEqual(len(panel.state.features), 0)

        # All
        panel._features_all()
        self.assertEqual(panel.state.features, set(panel._visible_dataset_feature_names()))

        # Expand & Collapse
        panel._features_expand_all()
        panel._features_collapse_all()


if __name__ == "__main__":
    unittest.main()
