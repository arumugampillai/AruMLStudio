"""Tests for Phase 5: Complete Import Isolation of AruMLStudio from AruNeo."""

from __future__ import annotations

import importlib
import os
import sys
import unittest
from pathlib import Path

_apps_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _apps_dir not in sys.path:
    sys.path.insert(0, _apps_dir)
from path_config import ensure_ml_studio_paths

ensure_ml_studio_paths()


class TestImportIsolation(unittest.TestCase):
    def test_active_interpreter_is_arumlstudio_venv(self) -> None:
        exe = sys.executable.lower()
        self.assertIn("arumlstudio", exe)
        self.assertIn(".venv", exe)
        self.assertNotIn("aruneo\\.venv", exe)

    def test_apps_root_priority_on_sys_path(self) -> None:
        ensure_ml_studio_paths()
        self.assertEqual(sys.path[0], _apps_dir)
        # Verify no external chart/data directory is inserted before apps/
        for p in sys.path:
            norm = os.path.normpath(p).lower()
            if "angelone\\chart" in norm or "aruneo" in norm:
                # Should never be ahead of apps
                idx = sys.path.index(p)
                self.assertGreater(idx, 0, f"Unwanted AruNeo path {p} was placed at sys.path[0]")

    def test_core_production_modules_never_import_from_aruneo(self) -> None:
        core_module_names = [
            "master_dataset_tk",
            "master_dataset_tk.app",
            "master_dataset_tk.build_service",
            "master_dataset_tk.project_config",
            "master_dataset_tk.ui_state",
            "master_dataset_tk.create_dataset_panel",
            "master_dataset_tk.research_lab_panel",
            "chain_replay_ml",
            "chain_replay_ml.dataset_builder.orchestrator",
            "chain_replay_ml.dataset_builder.master_naming",
            "chain_replay_ml.dataset_builder.master_feature_project",
            "chain_replay_ml.dataset_builder.feature_domains",
            "chain_replay_ml.dataset_builder.pipeline_registry_store",
            "chain_replay_ml.model_lab.store",
            "chain_replay_ml.model_lab.paths",
            "chain_replay_ml.training.dataset_loader",
            "chain_replay_ml.training.training_monitor",
            "chain_replay_ml.training.load_backend",
            "chain_replay_ml.training.row_group_prune",
            "chain_replay_ml.performance.numba_utils",
            "chain_replay_ml.post_training.config",
            "feature_intelligence.core.paths",
            "path_config",
            "tick_data_paths",
        ]

        for mod_name in core_module_names:
            mod = importlib.import_module(mod_name)
            file_path = getattr(mod, "__file__", None)
            if file_path:
                norm_file = os.path.normpath(file_path).lower()
                self.assertIn("arumlstudio", norm_file, f"Module {mod_name} resolved to {file_path}")
                self.assertNotIn("aruneo", norm_file, f"Module {mod_name} resolved to legacy AruNeo: {file_path}")

    def test_no_angelone_module_in_sys_modules(self) -> None:
        # Import main UI and dataset orchestrator
        import master_dataset_tk.app
        import chain_replay_ml.dataset_builder.orchestrator
        
        # Verify angelone is NOT loaded into sys.modules
        loaded = set(sys.modules.keys())
        self.assertNotIn("angelone", loaded)
        self.assertNotIn("AruNeo", loaded)


if __name__ == "__main__":
    unittest.main()
