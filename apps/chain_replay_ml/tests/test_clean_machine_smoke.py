"""Smoke test verifying clean machine installation and startup of AruMLStudio."""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

_apps_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _apps_dir not in sys.path:
    sys.path.insert(0, _apps_dir)
from path_config import ensure_ml_studio_paths

ensure_ml_studio_paths()

import __version__


class TestCleanMachineSmoke(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_appdata = tempfile.mkdtemp()
        self._env_backup = os.environ.get("APPDATA")
        os.environ["APPDATA"] = self.tmp_appdata

    def tearDown(self) -> None:
        if self._env_backup is None:
            os.environ.pop("APPDATA", None)
        else:
            os.environ["APPDATA"] = self._env_backup
        import shutil
        shutil.rmtree(self.tmp_appdata, ignore_errors=True)

    def test_canonical_version_and_identity(self) -> None:
        self.assertEqual(__version__.__app_name__, "AruMLStudio")
        self.assertTrue(__version__.__version__.startswith("1."))
        
        import master_dataset_tk
        self.assertEqual(master_dataset_tk.__version__, __version__.__version__)

    def test_all_core_ml_packages_importable(self) -> None:
        import numpy
        import pandas
        import polars
        import pyarrow
        import scipy
        import sklearn
        import xgboost
        import lightgbm
        import catboost
        import numba
        import duckdb
        import peewee
        import pydantic
        import psutil

        self.assertIsNotNone(numpy.__version__)
        self.assertIsNotNone(xgboost.__version__)
        self.assertIsNotNone(duckdb.__version__)

    def test_appdata_initialization_on_clean_machine(self) -> None:
        from master_dataset_tk.project_config import config_path, load_project_config, save_project_config
        from master_dataset_tk.ui_state import default_settings_path, UIStateManager

        # Ensure no legacy folder created
        cfg_file = config_path()
        self.assertIn("AruMLStudio", cfg_file)
        self.assertFalse(os.path.exists(os.path.join(self.tmp_appdata, "AruNeo")))

        ui_file = default_settings_path()
        self.assertIn("AruMLStudio", ui_file)
        self.assertFalse(os.path.exists(os.path.join(self.tmp_appdata, "AruNeo")))

    def test_feature_registry_subsystem_startup(self) -> None:
        from chain_replay_ml.dataset_builder.orchestrator import _load_feature_registry
        reg = _load_feature_registry()
        self.assertIsInstance(reg, dict)

    def test_feature_project_subsystem_startup(self) -> None:
        from chain_replay_ml.dataset_builder.master_feature_project import (
            normalize_feature_project_id,
            project_exists,
        )
        self.assertEqual(normalize_feature_project_id("all"), "all")

    def test_feature_transformation_subsystem_startup(self) -> None:
        from chain_replay_ml.dataset_builder.transformations.registry import registered_transformation_count
        count = registered_transformation_count()
        self.assertGreaterEqual(count, 12)

    def test_dataset_registry_subsystem_startup(self) -> None:
        from chain_replay_ml.dataset_builder.pipeline_registry_store import load_store
        store = load_store(self.tmp_appdata)
        self.assertIsInstance(store, dict)

    def test_model_registry_and_model_lab_startup(self) -> None:
        from chain_replay_ml.model_lab.store import ModelLabStore
        from chain_replay_ml.model_lab.paths import resolve_model_research_dir
        
        research_dir = resolve_model_research_dir()
        self.assertTrue(os.path.isdir(research_dir))

    def test_research_lab_and_master_dataset_ui_instantiation(self) -> None:
        from master_dataset_tk.create_dataset_panel import CreateDatasetPanel
        from master_dataset_tk.research_lab_panel import ResearchLabPanel
        from master_dataset_tk.app import MLResearchStudioApp

        self.assertTrue(callable(CreateDatasetPanel))
        self.assertTrue(callable(ResearchLabPanel))
        self.assertTrue(callable(MLResearchStudioApp))


if __name__ == "__main__":
    unittest.main()
