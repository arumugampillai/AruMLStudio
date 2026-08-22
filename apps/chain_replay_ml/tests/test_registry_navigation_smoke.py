"""Focused smoke test for Registry navigation enhancement (Pipeline Features & Morning Dossier)."""

import os
import tkinter as tk
import unittest

from master_dataset_tk.app import MLResearchStudioApp, _NAV_SECTIONS, _PAGE_TITLES
from master_dataset_tk.pipeline_registry_panel import PipelineRegistryPanel
from master_dataset_tk.morning_research_dossier_panel import MorningResearchDossierPanel
from chain_replay_ml.dataset_builder.pipeline_registry_store import load_store as load_pl_store


class TestRegistryNavigationSmoke(unittest.TestCase):
    """Smoke test for ML Research Studio Registry navigation."""

    def test_nav_sections_structure_and_labels(self) -> None:
        """Verify the exact requested Registry menu items and order."""
        registry_section = next((items for sec, items in _NAV_SECTIONS if sec == "Registry"), None)
        self.assertIsNotNone(registry_section, "Registry section not found in _NAV_SECTIONS")

        expected_items = [
            ("registry.datasets", "Dataset Registry"),
            ("registry.features", "Feature Registry"),
            ("registry.pipeline_features", "Pipeline Features"),
            ("registry.autonomous_researches", "Autonomous Researches"),
            ("registry.models", "Models"),
        ]
        self.assertEqual(registry_section, expected_items)

        # Check page titles mapping
        self.assertEqual(_PAGE_TITLES.get("registry.pipeline_features"), "Pipeline Features")
        self.assertEqual(_PAGE_TITLES.get("registry.autonomous_researches"), "Autonomous Research Registry")

    def test_app_shell_registry_page_routing(self) -> None:
        """Instantiate app shell and test navigating to Pipeline Features and Autonomous Researches."""
        try:
            app = MLResearchStudioApp()
            app.withdraw()

            # 1. Verify Pipeline Features Panel
            self.assertIn("registry.pipeline_features", app._pages)
            pipe_panel = app._pages["registry.pipeline_features"]
            self.assertIsInstance(pipe_panel, PipelineRegistryPanel)

            app._show_page("registry.pipeline_features")
            self.assertEqual(app._current_page, "registry.pipeline_features")
            self.assertGreaterEqual(len(pipe_panel._pipelines), 1)

            # 2. Verify Autonomous Researches Panel
            self.assertIn("registry.autonomous_researches", app._pages)
            dossier_panel = app._pages["registry.autonomous_researches"]
            self.assertIsInstance(dossier_panel, MorningResearchDossierPanel)

            app._show_page("registry.autonomous_researches")
            self.assertEqual(app._current_page, "registry.autonomous_researches")
            # Verify autonomous research registry primary view loads records
            self.assertIsNotNone(dossier_panel.tree)
            self.assertGreaterEqual(len(dossier_panel._all_records), 1)

            # 3. Verify Existing Registry Pages Intact
            for key in ("registry.datasets", "registry.features", "registry.models"):
                self.assertIn(key, app._pages)
                app._show_page(key)
                self.assertEqual(app._current_page, key)

            app.destroy()
        except tk.TclError:
            self.skipTest("Headless environment without display server")

    def test_invariants_unaltered(self) -> None:
        """Verify PL_0001 remains 171 features."""
        pl_doc = load_pl_store()
        self.assertEqual(len(pl_doc["pipelines"]["PL_0001"]["candidate_features"]), 171)


if __name__ == "__main__":
    unittest.main()
