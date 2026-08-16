"""Tests for Phase 3: AruMLStudio AppData isolation and safe migration."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_apps_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _apps_dir not in sys.path:
    sys.path.insert(0, _apps_dir)
from path_config import ensure_ml_studio_paths

ensure_ml_studio_paths()

from master_dataset_tk.project_config import (
    config_path,
    load_project_config,
    save_project_config,
    save_master_data_dir,
    resolve_master_data_dir,
)
from master_dataset_tk.ui_state import (
    default_settings_path,
    UIStateManager,
)
from feature_intelligence.core.paths import (
    default_data_dir as fic_default_data_dir,
    default_db_path as fic_default_db_path,
)


class TestAppDataIsolationAndMigration(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = tempfile.mkdtemp()
        self._env_backup = {
            "APPDATA": os.environ.get("APPDATA"),
            "ARUMLSTUDIO_UI_STATE_PATH": os.environ.get("ARUMLSTUDIO_UI_STATE_PATH"),
            "ARUNEO_UI_STATE_PATH": os.environ.get("ARUNEO_UI_STATE_PATH"),
            "ARUNEO_MASTER_DATA_DIR": os.environ.get("ARUNEO_MASTER_DATA_DIR"),
        }
        os.environ["APPDATA"] = self.tmp_root
        os.environ.pop("ARUMLSTUDIO_UI_STATE_PATH", None)
        os.environ.pop("ARUNEO_UI_STATE_PATH", None)
        os.environ.pop("ARUNEO_MASTER_DATA_DIR", None)

    def tearDown(self) -> None:
        for key, val in self._env_backup.items():
            if val is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = val
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_fresh_install_creates_arumlstudio_state_folder(self) -> None:
        cpath = config_path()
        self.assertIn("AruMLStudio", cpath)
        self.assertFalse(os.path.exists(os.path.join(self.tmp_root, "AruNeo")))
        
        save_project_config("D:/MyData/project")
        loaded = load_project_config()
        self.assertEqual(loaded.get("chart_dir"), os.path.abspath("D:/MyData/project"))
        
        # Verify written file is strictly inside AruMLStudio folder
        self.assertTrue(os.path.isfile(os.path.join(self.tmp_root, "AruMLStudio", "ml_research_studio.json")))
        self.assertFalse(os.path.exists(os.path.join(self.tmp_root, "AruNeo")))

    def test_migration_from_legacy_aruneo_config(self) -> None:
        # Create legacy AruNeo config
        legacy_dir = os.path.join(self.tmp_root, "AruNeo")
        os.makedirs(legacy_dir, exist_ok=True)
        legacy_cfg = os.path.join(legacy_dir, "ml_research_studio.json")
        with open(legacy_cfg, "w", encoding="utf-8") as fh:
            json.dump({"chart_dir": "D:/Legacy/chart", "master_data_dir": "D:/Legacy/master"}, fh)

        # Trigger resolution in AruMLStudio
        cpath = config_path()
        self.assertTrue(os.path.isfile(cpath))
        self.assertIn("AruMLStudio", cpath)

        loaded = load_project_config()
        self.assertEqual(loaded.get("chart_dir"), "D:/Legacy/chart")
        self.assertEqual(loaded.get("master_data_dir"), "D:/Legacy/master")

        # Verify old file was preserved and not deleted
        self.assertTrue(os.path.isfile(legacy_cfg))

        # Modifying config now writes ONLY to AruMLStudio
        save_project_config("D:/New/chart")
        with open(legacy_cfg, "r", encoding="utf-8") as fh:
            old_doc = json.load(fh)
        self.assertEqual(old_doc.get("chart_dir"), "D:/Legacy/chart")  # Untouched

    def test_existing_arumlstudio_config_ignores_legacy(self) -> None:
        # Create AruMLStudio config
        new_dir = os.path.join(self.tmp_root, "AruMLStudio")
        os.makedirs(new_dir, exist_ok=True)
        new_cfg = os.path.join(new_dir, "ml_research_studio.json")
        with open(new_cfg, "w", encoding="utf-8") as fh:
            json.dump({"chart_dir": "D:/Modern/chart"}, fh)

        # Create conflicting legacy AruNeo config
        legacy_dir = os.path.join(self.tmp_root, "AruNeo")
        os.makedirs(legacy_dir, exist_ok=True)
        legacy_cfg = os.path.join(legacy_dir, "ml_research_studio.json")
        with open(legacy_cfg, "w", encoding="utf-8") as fh:
            json.dump({"chart_dir": "D:/Old/chart"}, fh)

        loaded = load_project_config()
        self.assertEqual(loaded.get("chart_dir"), "D:/Modern/chart")

    def test_ui_state_migration_and_isolation(self) -> None:
        # Create legacy UI state
        legacy_dir = os.path.join(self.tmp_root, "AruNeo")
        os.makedirs(legacy_dir, exist_ok=True)
        legacy_ui = os.path.join(legacy_dir, "ui_state_tk.json")
        with open(legacy_ui, "w", encoding="utf-8") as fh:
            json.dump({"window_geometry": "1200x800+10+10", "active_tab": "builder.create"}, fh)

        mgr = UIStateManager()
        self.assertIn("AruMLStudio", mgr.path)
        self.assertEqual(mgr.get("window_geometry"), "1200x800+10+10")
        self.assertEqual(mgr.get("active_tab"), "builder.create")

        # New UI writes go to AruMLStudio
        mgr.set("active_tab", "registry.datasets")
        mgr.flush()

        new_ui = os.path.join(self.tmp_root, "AruMLStudio", "ui_state_tk.json")
        self.assertTrue(os.path.isfile(new_ui))
        with open(new_ui, "r", encoding="utf-8") as fh:
            new_doc = json.load(fh)
        self.assertEqual(new_doc.get("active_tab"), "registry.datasets")

        # Legacy file remains untouched
        with open(legacy_ui, "r", encoding="utf-8") as fh:
            old_doc = json.load(fh)
        self.assertEqual(old_doc.get("active_tab"), "builder.create")

    def test_feature_intelligence_state_migration(self) -> None:
        # Create legacy FIC folder with a dummy db
        legacy_fic = os.path.join(self.tmp_root, "AruNeo", "feature_intelligence")
        os.makedirs(legacy_fic, exist_ok=True)
        legacy_db = os.path.join(legacy_fic, "feature_intelligence.db")
        with open(legacy_db, "w", encoding="utf-8") as fh:
            fh.write("SQLITE_DUMMY_HEADER")

        data_dir = fic_default_data_dir()
        self.assertIn("AruMLStudio", str(data_dir))
        
        migrated_db = fic_default_db_path()
        self.assertTrue(migrated_db.is_file())
        with open(migrated_db, "r", encoding="utf-8") as fh:
            self.assertEqual(fh.read(), "SQLITE_DUMMY_HEADER")

    def test_data_path_separation_and_sys_path_safety(self) -> None:
        # Ensure external chart_dir is never placed on sys.path ahead of apps
        external_data_dir = os.path.join(self.tmp_root, "external_market_data")
        os.makedirs(os.path.join(external_data_dir, "data"), exist_ok=True)
        
        # Save external data dir as project config
        save_project_config(external_data_dir)
        cfg = load_project_config()
        self.assertEqual(cfg.get("chart_dir"), os.path.abspath(external_data_dir))

        # Check sys.path ordering rules: apps/ must always precede external data
        from master_dataset_tk.app import main as app_main
        # Ensure apps is at index 0
        ensure_ml_studio_paths()
        apps_idx = sys.path.index(_apps_dir) if _apps_dir in sys.path else -1
        self.assertEqual(apps_idx, 0)


if __name__ == "__main__":
    unittest.main()
