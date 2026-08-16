"""Tests for Phase 4: ARUMLSTUDIO_* environment variable precedence and isolation."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

_apps_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _apps_dir not in sys.path:
    sys.path.insert(0, _apps_dir)
from path_config import ensure_ml_studio_paths

ensure_ml_studio_paths()

from master_dataset_tk.project_config import resolve_master_data_dir
from master_dataset_tk.ui_state import default_settings_path
from master_dataset_tk.gil_monitor import gil_monitor_enabled
from tick_data_paths import resolve_tick_data_dir
from chain_replay_ml.model_lab.paths import resolve_model_research_dir
from chain_replay_ml.training.load_backend import resolve_training_load_backend
from chain_replay_ml.training.dataset_loader import _train_frame_bridge_via_polars
from chain_replay_ml.training.row_group_prune import row_group_prune_mode, premium_overread_factor
from chain_replay_ml.performance.numba_utils import env_numba_flag
from chain_replay_ml.post_training.config import env_master_enabled


class TestEnvVarIsolation(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.mkdtemp()
        self.env_keys = [
            "ARUMLSTUDIO_MASTER_DATA_DIR", "ARUNEO_MASTER_DATA_DIR",
            "ARUMLSTUDIO_UI_STATE_PATH", "ARUNEO_UI_STATE_PATH",
            "ARUMLSTUDIO_GIL_MONITOR", "ARUNEO_GIL_MONITOR",
            "ARUMLSTUDIO_TRAIN_FRAME_BRIDGE", "ARUNEO_TRAIN_FRAME_BRIDGE",
            "ARUMLSTUDIO_DATASET_ENGINE", "ARUNEO_DATASET_ENGINE",
            "ARUMLSTUDIO_TRAIN_ROW_GROUP_PRUNE", "ARUNEO_TRAIN_ROW_GROUP_PRUNE",
            "ARUMLSTUDIO_TRAIN_ROW_GROUP_OVERREAD", "ARUNEO_TRAIN_ROW_GROUP_OVERREAD",
            "ARUMLSTUDIO_TICK_DATA_DIR", "ARUNEO_TICK_DATA_DIR",
            "ARUMLSTUDIO_MODEL_RESEARCH_DIR", "ARUNEO_MODEL_RESEARCH_DIR",
            "ARUMLSTUDIO_FEATURE_NUMBA", "ARUNEO_FEATURE_NUMBA",
            "ARUMLSTUDIO_POST_TRAINING", "ARUNEO_POST_TRAINING",
            "APPDATA",
        ]
        self._env_backup = {k: os.environ.get(k) for k in self.env_keys}
        for k in self.env_keys:
            if k != "APPDATA":
                os.environ.pop(k, None)
        os.environ["APPDATA"] = self.tmp_dir

    def tearDown(self) -> None:
        for k, v in self._env_backup.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # 1. ARUMLSTUDIO_MASTER_DATA_DIR
    def test_master_data_dir_precedence(self) -> None:
        dir_new = os.path.join(self.tmp_dir, "master_new")
        dir_legacy = os.path.join(self.tmp_dir, "master_legacy")
        
        # A. Neither -> default
        self.assertIn("datasets", resolve_master_data_dir())
        
        # B. Legacy only -> legacy wins
        os.environ["ARUNEO_MASTER_DATA_DIR"] = dir_legacy
        self.assertEqual(resolve_master_data_dir(), os.path.abspath(dir_legacy))
        
        # C. Both -> new wins
        os.environ["ARUMLSTUDIO_MASTER_DATA_DIR"] = dir_new
        self.assertEqual(resolve_master_data_dir(), os.path.abspath(dir_new))
        
        # D. New only -> new wins
        os.environ.pop("ARUNEO_MASTER_DATA_DIR", None)
        self.assertEqual(resolve_master_data_dir(), os.path.abspath(dir_new))

    # 2. ARUMLSTUDIO_UI_STATE_PATH
    def test_ui_state_path_precedence(self) -> None:
        path_new = os.path.join(self.tmp_dir, "ui_new.json")
        path_legacy = os.path.join(self.tmp_dir, "ui_legacy.json")

        # A. Neither -> default AruMLStudio
        self.assertIn("AruMLStudio", default_settings_path())

        # B. Legacy only -> legacy wins
        os.environ["ARUNEO_UI_STATE_PATH"] = path_legacy
        self.assertEqual(default_settings_path(), path_legacy)

        # C. Both -> new wins
        os.environ["ARUMLSTUDIO_UI_STATE_PATH"] = path_new
        self.assertEqual(default_settings_path(), path_new)

        # D. New only -> new wins
        os.environ.pop("ARUNEO_UI_STATE_PATH", None)
        self.assertEqual(default_settings_path(), path_new)

    # 3. ARUMLSTUDIO_GIL_MONITOR
    def test_gil_monitor_precedence(self) -> None:
        # A. Neither -> default False
        self.assertFalse(gil_monitor_enabled())

        # B. Legacy only
        os.environ["ARUNEO_GIL_MONITOR"] = "1"
        self.assertTrue(gil_monitor_enabled())

        # C. Both (New=0, Legacy=1) -> New wins
        os.environ["ARUMLSTUDIO_GIL_MONITOR"] = "0"
        self.assertFalse(gil_monitor_enabled())

        # D. New only
        os.environ.pop("ARUNEO_GIL_MONITOR", None)
        os.environ["ARUMLSTUDIO_GIL_MONITOR"] = "1"
        self.assertTrue(gil_monitor_enabled())

    # 4. ARUMLSTUDIO_DATASET_ENGINE
    def test_dataset_engine_precedence(self) -> None:
        # A. Legacy only
        os.environ["ARUNEO_DATASET_ENGINE"] = "off"
        self.assertEqual(resolve_training_load_backend(), "pandas")

        # B. Both (New=on, Legacy=off) -> New wins
        os.environ["ARUMLSTUDIO_DATASET_ENGINE"] = "on"
        self.assertEqual(resolve_training_load_backend(), "dataset_engine")

        # C. New only (New=off)
        os.environ.pop("ARUNEO_DATASET_ENGINE", None)
        os.environ["ARUMLSTUDIO_DATASET_ENGINE"] = "off"
        self.assertEqual(resolve_training_load_backend(), "pandas")

    # 5. ARUMLSTUDIO_TRAIN_FRAME_BRIDGE
    def test_train_frame_bridge_precedence(self) -> None:
        # A. Default -> False (arrow)
        self.assertFalse(_train_frame_bridge_via_polars())

        # B. Legacy only -> True
        os.environ["ARUNEO_TRAIN_FRAME_BRIDGE"] = "polars"
        self.assertTrue(_train_frame_bridge_via_polars())

        # C. Both (New=arrow, Legacy=polars) -> New wins (False)
        os.environ["ARUMLSTUDIO_TRAIN_FRAME_BRIDGE"] = "arrow"
        self.assertFalse(_train_frame_bridge_via_polars())

        # D. New only -> True
        os.environ.pop("ARUNEO_TRAIN_FRAME_BRIDGE", None)
        os.environ["ARUMLSTUDIO_TRAIN_FRAME_BRIDGE"] = "polars"
        self.assertTrue(_train_frame_bridge_via_polars())

    # 6. ARUMLSTUDIO_TRAIN_ROW_GROUP_PRUNE & OVERREAD
    def test_row_group_prune_and_overread_precedence(self) -> None:
        # Prune Mode
        self.assertEqual(row_group_prune_mode(), "auto")
        os.environ["ARUNEO_TRAIN_ROW_GROUP_PRUNE"] = "off"
        self.assertEqual(row_group_prune_mode(), "off")
        os.environ["ARUMLSTUDIO_TRAIN_ROW_GROUP_PRUNE"] = "on"
        self.assertEqual(row_group_prune_mode(), "on")

        # Overread Factor
        self.assertEqual(premium_overread_factor(), 2.0)
        os.environ["ARUNEO_TRAIN_ROW_GROUP_OVERREAD"] = "3.5"
        self.assertEqual(premium_overread_factor(), 3.5)
        os.environ["ARUMLSTUDIO_TRAIN_ROW_GROUP_OVERREAD"] = "4.0"
        self.assertEqual(premium_overread_factor(), 4.0)

    # 7. ARUMLSTUDIO_TICK_DATA_DIR & ARUMLSTUDIO_MODEL_RESEARCH_DIR
    def test_tick_and_model_research_dir_precedence(self) -> None:
        d1 = os.path.join(self.tmp_dir, "t1")
        d2 = os.path.join(self.tmp_dir, "t2")
        os.environ["ARUNEO_TICK_DATA_DIR"] = d1
        self.assertEqual(resolve_tick_data_dir(), os.path.abspath(d1))
        os.environ["ARUMLSTUDIO_TICK_DATA_DIR"] = d2
        self.assertEqual(resolve_tick_data_dir(), os.path.abspath(d2))

        m1 = os.path.join(self.tmp_dir, "m1")
        m2 = os.path.join(self.tmp_dir, "m2")
        os.environ["ARUNEO_MODEL_RESEARCH_DIR"] = m1
        self.assertEqual(resolve_model_research_dir(), os.path.abspath(m1))
        os.environ["ARUMLSTUDIO_MODEL_RESEARCH_DIR"] = m2
        self.assertEqual(resolve_model_research_dir(), os.path.abspath(m2))

    # 8. Worker Process Inheritance Test
    def test_worker_process_inherits_arumlstudio_env(self) -> None:
        custom_master_dir = os.path.join(self.tmp_dir, "worker_master_target")
        os.environ["ARUMLSTUDIO_MASTER_DATA_DIR"] = custom_master_dir
        os.environ["ARUMLSTUDIO_DATASET_ENGINE"] = "off"

        script = """
import sys, os
sys.path.insert(0, 'apps')
from master_dataset_tk.project_config import resolve_master_data_dir
from chain_replay_ml.training.load_backend import resolve_training_load_backend

master_dir = resolve_master_data_dir()
engine = resolve_training_load_backend()
print(f"MASTER:{master_dir}")
print(f"ENGINE:{engine}")
"""
        cmd = [sys.executable, "-c", script]
        res = subprocess.run(cmd, capture_output=True, text=True, cwd=_apps_dir)
        self.assertEqual(res.returncode, 0, f"Worker failed: {res.stderr}")
        self.assertIn(f"MASTER:{os.path.abspath(custom_master_dir)}", res.stdout)
        self.assertIn("ENGINE:pandas", res.stdout)


if __name__ == "__main__":
    unittest.main()
