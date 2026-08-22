"""Focused smoke test for Phase 4: Settings UI & Safe Migration Assistant."""

import os
import shutil
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from chain_replay_ml.core.data_root import (
    DEFAULT_CANONICAL_DATA_ROOT,
    DataRootService,
    get_data_root_service,
    resolve_data_root,
    save_data_root,
)
from chain_replay_ml.core.migration_service import (
    DataMigrationService,
    MigrationPlan,
    MigrationPlanItem,
)
from chain_replay_ml.dataset_builder.pipeline_registry_store import load_store as load_pl_store
from chain_replay_ml.discovery_dashboard.service import list_discovery_pipelines


class TestPhase4SettingsAndMigration(unittest.TestCase):
    """Smoke tests for Phase 4 canonical Data Root settings & migration engine."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.mkdtemp()
        self.orig_root = resolve_data_root()

    def tearDown(self) -> None:
        save_data_root(self.orig_root)
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_data_root_service_configuration(self) -> None:
        """Test DataRootService configuration and path derivation."""
        custom_root = os.path.join(self.temp_dir, "custom_data")
        svc = DataRootService(custom_root)
        self.assertEqual(svc.data_root, os.path.abspath(custom_root))
        self.assertEqual(svc.get_database_path("analysis"), os.path.join(svc.data_root, "databases", "analysis.db"))
        self.assertEqual(svc.get_registry_path("pipeline"), os.path.join(svc.data_root, "registries", "pipeline_registry_store.json"))
        self.assertEqual(svc.get_datasets_dir("analysis"), os.path.join(svc.data_root, "datasets", "analysis"))

        # Test ensure_layout
        svc.ensure_layout()
        val = svc.validate_layout()
        self.assertTrue(val["root_exists"])
        self.assertGreaterEqual(len(val["subdirs"]), 14)

    def test_migration_service_dry_run_and_conflicts(self) -> None:
        """Test migration service discovers files, computes sizes, and detects identical files vs conflicts."""
        src_dir = os.path.join(self.temp_dir, "legacy_source")
        target_root = os.path.join(self.temp_dir, "target_data_root")
        os.makedirs(os.path.join(src_dir, "datasets"), exist_ok=True)

        # Create a sample mock analysis db in source
        src_db = os.path.join(src_dir, "analysis.db")
        conn = sqlite3.connect(src_db)
        conn.execute("CREATE TABLE mock_tbl (id INTEGER PRIMARY KEY, val TEXT);")
        conn.execute("INSERT INTO mock_tbl VALUES (1, 'alpha'), (2, 'beta');")
        conn.commit()
        conn.close()

        # Create a sample parquet-like file in source datasets
        src_parquet = os.path.join(src_dir, "datasets", "sample.parquet")
        with open(src_parquet, "wb") as f:
            f.write(b"PARQUET_MOCK_DATA_BYTE_STREAM_12345")

        mig_svc = DataMigrationService(target_data_root=target_root)
        plan = mig_svc.build_plan(source_dir=src_dir)

        self.assertEqual(len(plan.items), 2)
        self.assertEqual(plan.ready_count, 2)
        self.assertEqual(plan.conflict_count, 0)
        self.assertTrue(plan.is_safe_to_execute)

        # Execute migration
        result = mig_svc.execute_migration(plan)
        self.assertTrue(result["success"])
        self.assertEqual(result["copied_count"], 2)

        # Verify destination files
        dst_db = os.path.join(target_root, "databases", "analysis.db")
        self.assertTrue(os.path.isfile(dst_db))
        conn_dst = sqlite3.connect(dst_db)
        cur = conn_dst.cursor()
        cur.execute("PRAGMA integrity_check;")
        self.assertEqual(cur.fetchone()[0], "ok")
        cur.execute("SELECT count(*) FROM mock_tbl;")
        self.assertEqual(cur.fetchone()[0], 2)
        conn_dst.close()

        # Test dry-run on already-migrated data (should detect identical status)
        plan2 = mig_svc.build_plan(source_dir=src_dir)
        self.assertEqual(plan2.identical_count, 2)
        self.assertEqual(plan2.ready_count, 0)
    def test_canonical_invariants_intact(self) -> None:
        """Verify PL_0001 (171 features) and Discovery Pipeline DP_CAMP_... remain intact."""
        pl_doc = load_pl_store()
        self.assertEqual(len(pl_doc["pipelines"]["PL_0001"]["candidate_features"]), 171)

        target_pid = "DP_CAMP_NIFTY_6s_DIRECTION_CLASSIFIER_5m_R001_20260822_002913"
        pipes = list_discovery_pipelines()
        pipe = next(p for p in pipes if p["pipeline_id"] == target_pid)
        self.assertEqual(pipe["current_generation"], 2)
        self.assertEqual(pipe["total_df_features_created"], 56)
        self.assertEqual(pipe["keep_count"], 9)
        self.assertEqual(pipe["watch_count"], 5)
        self.assertEqual(pipe["remove_count"], 42)
        self.assertEqual(pipe["active_discovery_pool"], 14)

    def test_settings_panel_and_migration_dialog_instantiation(self) -> None:
        """Verify SettingsPanel and MigrationAssistantDialog initialize without errors."""
        import tkinter as tk
        try:
            root = tk.Tk()
            root.withdraw()
            from master_dataset_tk.settings_panel import SettingsPanel
            from master_dataset_tk.migration_dialog import MigrationAssistantDialog

            panel = SettingsPanel(root)
            self.assertEqual(panel._data_root_var.get(), r"D:\data")
            self.assertIsNotNone(panel.layout_tree)

            dlg = MigrationAssistantDialog(root, target_data_root=r"D:\data")
            self.assertEqual(dlg.target_data_root, r"D:\data")
            dlg.destroy()
            root.destroy()
        except tk.TclError:
            # Headless environment without DISPLAY
            pass


if __name__ == "__main__":
    unittest.main()
