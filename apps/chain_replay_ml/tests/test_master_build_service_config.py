"""Tests for master dataset build_service configuration and feature_project_id propagation."""

from __future__ import annotations

import inspect
import os
import sys
import tempfile
import unittest

_apps_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _apps_dir not in sys.path:
    sys.path.insert(0, _apps_dir)
from path_config import ensure_ml_studio_paths

ensure_ml_studio_paths()

from master_dataset_tk.build_service import build_master_insert_config


class TestMasterBuildServiceConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.tmp, "data"), exist_ok=True)
        self.dummy_sources = [
            {
                "trading_day": "2026-07-30",
                "market": "NIFTY",
                "expiry": "2026-08-04",
            }
        ]

    def test_build_master_insert_config_signature_has_feature_project_id(self) -> None:
        sig = inspect.signature(build_master_insert_config)
        self.assertIn("feature_project_id", sig.parameters)
        param = sig.parameters["feature_project_id"]
        self.assertIsNone(param.default)

    def test_build_master_insert_config_with_project_all(self) -> None:
        cfg = build_master_insert_config(
            self.tmp,
            sources=self.dummy_sources,
            interval_sec=6,
            feature_project_id="all",
        )
        self.assertEqual(cfg.feature_project_id, "all")
        self.assertEqual(cfg.storage_backend, "master_sqlite")
        self.assertTrue(cfg.also_write_master_db)

    def test_build_master_insert_config_with_named_project(self) -> None:
        cfg = build_master_insert_config(
            self.tmp,
            sources=self.dummy_sources,
            interval_sec=6,
            feature_project_id="Alpha_Core",
        )
        self.assertEqual(cfg.feature_project_id, "alpha_core")

    def test_build_master_insert_config_default_none_becomes_all(self) -> None:
        cfg = build_master_insert_config(
            self.tmp,
            sources=self.dummy_sources,
            interval_sec=6,
            feature_project_id=None,
        )
        self.assertEqual(cfg.feature_project_id, "all")


if __name__ == "__main__":
    unittest.main()
