"""Tests for Create Model (ModelBuilderPanel) feature_project_id binding and feature tree rendering."""

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


class TestModelBuilderFeatureProjectBinding(unittest.TestCase):
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

        # 1. Create a custom project
        svc.create_project(
            self.chart_dir,
            project_id="nifty_classification",
            label="NIFTY_Classification",
            feature_names=["spot", "ltp", "delta", "gamma", "vega"],
            project_groups=[{"id": "opt_greeks", "label": "Option Greeks"}],
            feature_group_map={"delta": "opt_greeks", "gamma": "opt_greeks", "vega": "opt_greeks"},
        )

        # 2. Create Dataset 1 with feature_project_id = "all"
        ds1_meta = {
            "dataset_name": "ds_all",
            "output_parquet": "datasets/ds_all.parquet",
            "feature_project_id": "all",
            "feature_columns": ["spot", "ltp", "delta", "gamma", "vega", "theta", "current_iv"],
            "target_columns": ["fwd_ret_5m"],
            "prediction_targets": ["fwd_ret_5m"],
        }
        with open(os.path.join(self.datasets_dir, "ds_all.json"), "w") as fh:
            json.dump(ds1_meta, fh)

        # 3. Create Dataset 2 with feature_project_id = "nifty_classification"
        ds2_meta = {
            "dataset_name": "ds_custom",
            "output_parquet": "datasets/ds_custom.parquet",
            "feature_project_id": "nifty_classification",
            "feature_columns": ["spot", "ltp", "delta", "gamma", "vega", "other_col_123"],
            "target_columns": ["fwd_ret_5m"],
            "prediction_targets": ["fwd_ret_5m"],
        }
        with open(os.path.join(self.datasets_dir, "ds_custom.json"), "w") as fh:
            json.dump(ds2_meta, fh)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_dataset_all_renders_all_feature_project_groups_and_summary(self) -> None:
        """Verify that selecting dataset with 'all' displays 'all' project groups and correct summary."""
        panel = CreateModelPanel(self.root, chart_dir=self.chart_dir)
        panel._dataset_var.set("ds_all")
        panel._on_dataset_changed(persist=False)

        # 1. Summary line should say "Feature project: all"
        summary_label = panel._feature_project_summary_label()
        self.assertEqual(summary_label, "all")

        # 2. Feature groups should come from the 'all' Feature Project
        groups = panel._feature_groups()
        group_labels = [g["label"] for g in groups]
        self.assertIn("Price & Premium", group_labels)
        self.assertIn("Greeks", group_labels)
        self.assertNotIn("Dataset (not in registry)", group_labels)
        self.assertNotIn("Other", group_labels)

    def test_dataset_custom_renders_custom_feature_project_groups_and_summary(self) -> None:
        """Verify that selecting dataset with 'nifty_classification' renders its feature tree."""
        panel = CreateModelPanel(self.root, chart_dir=self.chart_dir)
        panel._dataset_var.set("ds_custom")
        panel._on_dataset_changed(persist=False)

        # 1. Summary line should say "Feature project: NIFTY_Classification"
        summary_label = panel._feature_project_summary_label()
        self.assertEqual(summary_label, "NIFTY_Classification")

        # 2. Feature groups should only contain NIFTY_Classification groups & features
        groups = panel._feature_groups()
        group_labels = [g["label"] for g in groups]
        self.assertIn("Option Greeks", group_labels)
        self.assertNotIn("Dataset (not in registry)", group_labels)
        self.assertNotIn("Other", group_labels)

        # 3. Non-project column ('other_col_123') must be excluded
        all_rendered_feats = [f for g in groups for f in g["features"]]
        self.assertNotIn("other_col_123", all_rendered_feats)
        self.assertIn("delta", all_rendered_feats)
        self.assertIn("gamma", all_rendered_feats)


if __name__ == "__main__":
    unittest.main()
