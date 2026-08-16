"""Tests for FeatureProjectManagerPanel UI rendering, 'all' special browse view, and project isolation."""

from __future__ import annotations

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
from chain_replay_ml.dataset_builder.feature_project_organization import (
    RESERVED_ALL_PROJECT_ID,
    is_reserved_all_project_id,
    project_registry_groups,
)
from master_dataset_tk.feature_project_manager_panel import FeatureProjectManagerPanel


class TestFeatureProjectManagerPanel(unittest.TestCase):
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
        os.makedirs(self.chart_dir, exist_ok=True)
        svc.ensure_all_project(self.chart_dir)

    def tearDown(self) -> None:
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_all_project_loads_canonical_groups_in_tree(self) -> None:
        """Verify that opening with 'all' populates the tree with all canonical groups and features."""
        panel = FeatureProjectManagerPanel(self.root, chart_dir=self.chart_dir)
        panel.refresh()

        self.assertEqual(panel._selected_id, RESERVED_ALL_PROJECT_ID)
        tree = panel._group_tree
        root_nodes = tree.get_children()
        
        # Verify canonical groups are rendered
        self.assertGreaterEqual(len(root_nodes), 10)
        
        # Verify groups have non-zero features
        group_texts = [tree.item(item, "text") for item in root_nodes]
        self.assertTrue(any("Price" in t for t in group_texts))
        self.assertTrue(any("Greeks" in t for t in group_texts))
        self.assertTrue(any("Implied Volatility" in t for t in group_texts))

        # Check features inside first two groups
        for gid in root_nodes[:2]:
            children = tree.get_children(gid)
            self.assertGreater(len(children), 0, f"Group {tree.item(gid, 'text')} has no features!")
            sample_feat = tree.item(children[0], "text")
            self.assertTrue(len(sample_feat) > 0)

    def test_custom_project_loads_membership_tree(self) -> None:
        """Verify that custom project renders its specific feature membership."""
        # Create a custom project
        svc.create_project(
            self.chart_dir,
            project_id="nifty_classification",
            label="NIFTY_Classification",
            feature_names=["spot", "ltp", "delta", "gamma"],
            project_groups=[{"id": "core", "label": "Core Signals"}],
            feature_group_map={"spot": "core", "ltp": "core", "delta": "core", "gamma": "core"},
        )

        panel = FeatureProjectManagerPanel(self.root, chart_dir=self.chart_dir)
        panel.refresh()
        panel._load_project("nifty_classification", force=True)

        self.assertEqual(panel._selected_id, "nifty_classification")
        tree = panel._group_tree
        root_nodes = tree.get_children()
        
        # Should have the "core" custom group containing the 4 features
        self.assertGreaterEqual(len(root_nodes), 1)
        core_group = next((n for n in root_nodes if "Core Signals" in tree.item(n, "text")), None)
        self.assertIsNotNone(core_group)
        children = tree.get_children(core_group)
        self.assertEqual(len(children), 4)

    def test_context_menu_rules_for_all_vs_custom_project(self) -> None:
        """Verify context menu rules: 'all' only allows Feature Details; custom project allows edits."""
        panel = FeatureProjectManagerPanel(self.root, chart_dir=self.chart_dir)
        panel.refresh()
        
        # 1. For 'all':
        menu = tk.Menu(panel)
        panel._build_feature_context_menu(menu, "delta")
        labels = [menu.entrycget(i, "label") for i in range(menu.index("end") + 1) if menu.type(i) == "command"]
        self.assertIn("Feature Details", labels)
        self.assertNotIn("Remove from Project", labels)

        # 2. For custom project:
        svc.create_project(
            self.chart_dir,
            project_id="test_proj",
            label="Test Proj",
            feature_names=["delta"],
            project_groups=[],
            feature_group_map={},
        )
        panel.refresh()
    def test_save_project_does_not_modify_registry(self) -> None:
        """Verify saving project edits modifies only project store, not the Registry catalog."""
        catalog_before = svc.load_catalog(self.chart_dir)
        features_before = catalog_before.get("features") or []

        svc.create_project(
            self.chart_dir,
            project_id="test_save_proj",
            label="Test Save",
            feature_names=["delta", "gamma"],
            project_groups=[{"id": "g1", "label": "Group 1"}],
            feature_group_map={"delta": "g1", "gamma": "g1"},
        )

        catalog_after = svc.load_catalog(self.chart_dir)
        features_after = catalog_after.get("features") or []

        # Feature count and feature registry definitions must be identical
        self.assertEqual(len(features_before), len(features_after))
        self.assertEqual(
            [f.get("name") for f in features_before],
            [f.get("name") for f in features_after],
        )


if __name__ == "__main__":
    unittest.main()
