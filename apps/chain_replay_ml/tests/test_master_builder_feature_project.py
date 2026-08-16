"""Tests for Master Dataset Feature Project integration."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import tkinter as tk

from chain_replay_ml.dataset_builder.feature_project_organization import (
    RESERVED_ALL_PROJECT_ID,
    build_default_all_project_doc,
    project_registry_feature_source,
)
from chain_replay_ml.dataset_builder.feature_ownership import (
    canonical_registry_features,
    is_canonical,
)
from chain_replay_ml.dataset_builder.feature_sources_catalog import (
    DATASET_SOURCE_BASE_PIPELINE,
    DATASET_SOURCE_FEATURE_REGISTRY,
    DATASET_SOURCE_OTHER_PIPELINE,
)
from chain_replay_ml.dataset_builder.orchestrator import _load_feature_registry
from master_dataset_tk import feature_registry_service as svc
from master_dataset_tk.feature_selection_picker import FeatureSelectionPicker
from master_dataset_tk.feature_selection_tab import FeatureSelectionTab
from master_dataset_tk.config_panel import BuildConfigPanel


class TestMasterBuilderFeatureProject(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.data_dir = self.tmp_dir.name
        self.registry = _load_feature_registry()
        try:
            self.root = tk.Tk()
            self.root.withdraw()
        except Exception:
            self.root = None

        # Create custom "chart" project
        svc.create_project(
            self.data_dir,
            project_id="chart",
            label="Chart Analysis",
            feature_names=["spot", "atm_iv_ce", "futures_ltp", "spot_ema9", "spot_ema20"],
            project_groups=[
                {
                    "id": "chart_core",
                    "label": "Chart Core Group",
                    "features": ["spot", "atm_iv_ce", "futures_ltp"],
                },
                {
                    "id": "chart_ema",
                    "label": "Chart EMA Group",
                    "features": ["spot_ema9", "spot_ema20"],
                },
            ],
            feature_group_map={
                "spot": "chart_core",
                "atm_iv_ce": "chart_core",
                "futures_ltp": "chart_core",
                "spot_ema9": "chart_ema",
                "spot_ema20": "chart_ema",
            },
        )

    def tearDown(self) -> None:
        if self.root is not None:
            try:
                self.root.destroy()
            except Exception:
                pass
        self.tmp_dir.cleanup()

    def test_selecting_project_all_shows_all_active_registry_features(self) -> None:
        picker = FeatureSelectionPicker(
            self.registry,
            chart_dir=self.data_dir,
            feature_project_id="all",
        )
        active_features = picker.active_project_features()
        self.assertEqual(len(active_features), 206)
        rows = picker._domain_rows()
        self.assertEqual(len(rows), 11)
        self.assertIn("Groups 0 / 11", picker.stats_text())

    def test_selecting_project_chart_shows_only_chart_features(self) -> None:
        picker = FeatureSelectionPicker(
            self.registry,
            chart_dir=self.data_dir,
            feature_project_id="chart",
        )
        active_features = picker.active_project_features()
        self.assertEqual(sorted(active_features), ["atm_iv_ce", "futures_ltp", "spot", "spot_ema20", "spot_ema9"])
        rows = picker._domain_rows()
        self.assertEqual(len(rows), 2)
        group_ids = [gid for gid, _, _ in rows]
        self.assertEqual(group_ids, ["chart_core", "chart_ema"])
        group_labels = [label for _, label, _ in rows]
        self.assertEqual(group_labels, ["Chart Core Group", "Chart EMA Group"])

    def test_changing_project_reloads_tree_and_reconciles_selection(self) -> None:
        picker = FeatureSelectionPicker(
            self.registry,
            chart_dir=self.data_dir,
            feature_project_id="all",
        )
        # Select all in project "all"
        picker._select_all()
        self.assertEqual(len(picker._enabled_features), 206)

        # Switch to "chart"
        picker.set_feature_project("chart", self.data_dir)
        chart_features = {"spot", "atm_iv_ce", "futures_ltp", "spot_ema9", "spot_ema20"}
        # Enabled features should be reconciled to only chart features
        self.assertEqual(picker._enabled_features, chart_features)
        self.assertEqual(picker.active_feature_total(), 5)
        self.assertIn("Groups 2 / 2", picker.stats_text())
        self.assertIn("Features 5 / 5", picker.stats_text())

    def test_ui_config_panel_and_tab_project_propagation(self) -> None:
        if self.root is None:
            return

        panel = BuildConfigPanel(self.root, chart_dir=self.data_dir)
        self.assertIsNotNone(panel._feature_tab)

        # Initially "all"
        self.assertEqual(panel.feature_project_id(), "all")
        self.assertEqual(panel._feature_tab._picker.active_feature_total(), 206)

        # Switch combobox to "chart"
        panel._set_feature_project_combo_value("chart")
        self.assertEqual(panel.feature_project_id(), "chart")
        self.assertEqual(panel._feature_tab._picker.active_feature_total(), 5)
        self.assertEqual(
            sorted(panel._feature_tab._picker.active_project_features()),
            ["atm_iv_ce", "futures_ltp", "spot", "spot_ema20", "spot_ema9"],
        )

        # Verify all features belong to Feature Registry and not Pipeline
        from chain_replay_ml.dataset_builder.feature_sources_catalog import classify_dataset_feature_source
        for feat in panel._feature_tab._picker.active_project_features():
            self.assertTrue(is_canonical(feat))
            src = classify_dataset_feature_source(feat, data_dir=self.data_dir)
            self.assertEqual(src, DATASET_SOURCE_FEATURE_REGISTRY)

    def test_master_dataset_build_metadata_preserves_feature_project_id_and_registry_exports(self) -> None:
        """Test master build configuration and schema write feature_project_id and registry_export_features."""
        from chain_replay_ml.dataset_builder.master_feature_project import (
            read_master_feature_project_id,
            set_master_feature_project_id,
        )
        from chain_replay_ml.dataset_builder.master_store import MasterStore
        from master_dataset_tk.build_service import chart_data_dir

        data_dir = chart_data_dir(self.data_dir)
        db_path = os.path.join(data_dir, "test_master.db")
        os.makedirs(data_dir, exist_ok=True)
        store = MasterStore(db_path)
        store.open()
        try:
            set_master_feature_project_id(store, data_dir, "chart")
            self.assertEqual(read_master_feature_project_id(store), "chart")

            # Check master_config dict
            cfg = {
                "market": "NIFTY",
                "feature_project_id": "chart",
                "registry_export_features": ["spot", "atm_iv_ce", "futures_ltp"],
                "registry_export_count": 3,
            }
            store.set_meta("master_config", cfg)
            read_cfg = store.get_meta("master_config")
            self.assertIsInstance(read_cfg, dict)
            self.assertEqual(read_cfg.get("feature_project_id"), "chart")
            self.assertEqual(read_cfg.get("registry_export_features"), ["spot", "atm_iv_ce", "futures_ltp"])
            self.assertEqual(read_cfg.get("registry_export_count"), 3)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()

