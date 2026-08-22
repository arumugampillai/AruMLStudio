"""Focused unit verification for Canonical Data Root Service (Phase 1, Doc 17)."""

import json
import os
import shutil
import tempfile
import unittest

from chain_replay_ml.core.data_root import (
    DEFAULT_CANONICAL_DATA_ROOT,
    DataRootService,
    get_data_root_service,
    normalize_storage_path,
    resolve_data_root,
    save_data_root,
)


class TestDataRootService(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.custom_root = os.path.join(self.tmp_dir, "custom_data_root")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_data_root_resolution_default_and_custom(self):
        """1. Verify DataRootService resolves default D:\\data and custom root paths."""
        svc_default = DataRootService()
        self.assertEqual(svc_default.data_root, os.path.abspath(r"D:\data"))
        self.assertEqual(svc_default.get_data_root(), os.path.abspath(r"D:\data"))

        svc_custom = DataRootService(self.custom_root)
        self.assertEqual(svc_custom.data_root, os.path.abspath(self.custom_root))
        self.assertEqual(svc_custom.get_data_root(), os.path.abspath(self.custom_root))

    def test_canonical_database_paths(self):
        """2. Verify database path resolutions match Doc 17 canonical hierarchy."""
        svc = DataRootService(r"D:\data")
        self.assertEqual(
            svc.get_database_path("analysis"),
            os.path.abspath(r"D:\data\databases\analysis.db"),
        )
        self.assertEqual(
            svc.get_database_path("feature_evidence"),
            os.path.abspath(r"D:\data\databases\feature_recommendation_evidence.db"),
        )
        self.assertEqual(
            svc.get_database_path("angel_historic"),
            os.path.abspath(r"D:\data\databases\angel_historic_bars.db"),
        )
        self.assertEqual(
            svc.get_database_path("predictions"),
            os.path.abspath(r"D:\data\databases\prediction_runs.db"),
        )
        self.assertEqual(
            svc.get_database_path("strategies"),
            os.path.abspath(r"D:\data\databases\strategy_runs.db"),
        )

    def test_canonical_registry_paths(self):
        """3. Verify registry path resolutions match Doc 17 canonical hierarchy."""
        svc = DataRootService(r"D:\data")
        self.assertEqual(
            svc.get_registry_path("feature"),
            os.path.abspath(r"D:\data\registries\feature_registry_store.json"),
        )
        self.assertEqual(
            svc.get_registry_path("pipeline"),
            os.path.abspath(r"D:\data\registries\pipeline_registry_store.json"),
        )
        self.assertEqual(
            svc.get_registry_path("model"),
            os.path.abspath(r"D:\data\registries\model_registry.db"),
        )

    def test_canonical_directory_subpaths(self):
        """4. Verify datasets, models, research, predictions, ticks, logs, cache paths."""
        svc = DataRootService(r"D:\data")
        self.assertEqual(svc.get_datasets_dir("master"), os.path.abspath(r"D:\data\datasets\master"))
        self.assertEqual(svc.get_datasets_dir("analysis"), os.path.abspath(r"D:\data\datasets\analysis"))
        self.assertEqual(svc.get_datasets_dir("labels"), os.path.abspath(r"D:\data\datasets\labels"))
        self.assertEqual(svc.get_datasets_dir("exports"), os.path.abspath(r"D:\data\datasets\exports"))

        self.assertEqual(svc.get_models_dir("production"), os.path.abspath(r"D:\data\models\production"))
        self.assertEqual(svc.get_models_dir("candidates"), os.path.abspath(r"D:\data\models\candidates"))
        self.assertEqual(svc.get_models_dir("research"), os.path.abspath(r"D:\data\models\research"))

        self.assertEqual(svc.get_research_dir("campaigns"), os.path.abspath(r"D:\data\research\campaigns"))
        self.assertEqual(svc.get_research_dir("discovery"), os.path.abspath(r"D:\data\research\discovery"))
        self.assertEqual(svc.get_research_dir("snapshots"), os.path.abspath(r"D:\data\research\snapshots"))
        self.assertEqual(svc.get_research_dir("dossiers"), os.path.abspath(r"D:\data\research\dossiers"))

        self.assertEqual(svc.get_predictions_dir("datasets"), os.path.abspath(r"D:\data\predictions\datasets"))
        self.assertEqual(svc.get_predictions_dir("artifacts"), os.path.abspath(r"D:\data\predictions\artifacts"))

        self.assertEqual(svc.get_ticks_dir(), os.path.abspath(r"D:\data\ticks"))
        self.assertEqual(svc.get_logs_dir(), os.path.abspath(r"D:\data\logs"))
        self.assertEqual(svc.get_cache_dir(), os.path.abspath(r"D:\data\cache"))

    def test_no_side_effects_on_path_resolution(self):
        """5. Invariant: Resolving a path must NEVER create files or directories implicitly."""
        non_existent_root = os.path.join(self.tmp_dir, "ghost_root")
        svc = DataRootService(non_existent_root)
        
        # Calling resolvers
        _ = svc.get_database_path("analysis")
        _ = svc.get_registry_path("pipeline")
        _ = svc.get_datasets_dir("analysis")

        # Invariant check: ghost_root must NOT exist on disk
        self.assertFalse(os.path.exists(non_existent_root))

    def test_explicit_ensure_layout(self):
        """6. Invariant: Directory creation occurs ONLY when ensure_layout() is explicitly invoked."""
        target_root = os.path.join(self.tmp_dir, "test_ensure_root")
        svc = DataRootService(target_root)
        self.assertFalse(os.path.exists(target_root))

        created = svc.ensure_layout()
        self.assertTrue(os.path.isdir(target_root))
        self.assertTrue(os.path.isdir(os.path.join(target_root, "databases")))
        self.assertTrue(os.path.isdir(os.path.join(target_root, "registries")))
        self.assertTrue(os.path.isdir(os.path.join(target_root, "datasets", "analysis")))
        self.assertTrue(os.path.isdir(os.path.join(target_root, "models", "production")))
        self.assertTrue(os.path.isdir(os.path.join(target_root, "ticks")))
        self.assertGreater(len(created), 10)

        # Idempotent secondary call
        created_again = svc.ensure_layout()
        self.assertEqual(len(created_again), 0)

    def test_cwd_independence_invariant(self):
        """7. Invariant: Resolver returns identical absolute paths regardless of current working directory."""
        svc = DataRootService(r"D:\data")
        expected_db = os.path.abspath(r"D:\data\databases\analysis.db")
        
        orig_cwd = os.getcwd()
        try:
            os.chdir(self.tmp_dir)
            self.assertEqual(svc.get_database_path("analysis"), expected_db)
            self.assertEqual(svc.get_registry_path("feature"), os.path.abspath(r"D:\data\registries\feature_registry_store.json"))
        finally:
            os.chdir(orig_cwd)

    def test_validate_layout_diagnostics(self):
        """8. Verify layout validation diagnostic telemetry."""
        target_root = os.path.join(self.tmp_dir, "diag_root")
        svc = DataRootService(target_root)
        val_before = svc.validate_layout()
        self.assertFalse(val_before["root_exists"])

        svc.ensure_layout()
        val_after = svc.validate_layout()
        self.assertTrue(val_after["root_exists"])
        self.assertTrue(val_after["is_valid"])
        self.assertTrue(val_after["subdirs"]["databases"])
        self.assertTrue(val_after["subdirs"]["registries"])


if __name__ == "__main__":
    unittest.main()
