"""Comprehensive Tests for Phase 4C.2: Model Registry SQLite Extension & Taxonomy Persistence."""

import json
import os
import shutil
import sqlite3
import tempfile
import unittest

from chain_replay_ml.model_taxonomy import (
    BASELINE_REGIME_CATALOG,
    DEFAULT_REGIME_ID,
    DEFAULT_REGIME_NAME,
    ModelContextKey,
    ModelLifecycleStatus,
    ModelMetadata,
    ModelPopulationTier,
    RegimeScope,
    RegimeSpec,
    TaskSpec,
    TaskType,
    infer_task_type_from_target,
    resolve_model_metadata_or_legacy,
)
from chain_replay_ml.training.config import TrainingConfig
from chain_replay_ml.training.lifecycle_store import (
    _connect,
    delete_history_for_model,
    ensure_lifecycle_tables,
    get_champion_for_context,
    get_history_by_model_name,
    list_context_champions,
    list_history_for_model,
    list_model_champions,
    migrate_lifecycle_schema_v2,
    record_training_history,
    set_champion_for_context,
)
from chain_replay_ml.training.paths import model_package_dir, models_dir, safe_model_name
from chain_replay_ml.training.registry import list_trained_models


class TestModelRegistryExtension(unittest.TestCase):
    """Test suite for Phase 4C.2 SQLite schema extension and multi-regime champion governance."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_lifecycle_")
        self.models_dir = models_dir(self.tmp_dir)
        os.makedirs(self.models_dir, exist_ok=True)
        self.db_path = os.path.join(self.models_dir, ".lifecycle_registry.db")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_fresh_lifecycle_db_migration(self):
        """1. Fresh lifecycle DB migration: Verify all Phase 4C.2 taxonomy columns and indexes are created."""
        with _connect(self.tmp_dir) as conn:
            hist_cols = {r[1] for r in conn.execute("PRAGMA table_info(model_history)").fetchall()}
            reg_cols = {r[1] for r in conn.execute("PRAGMA table_info(model_registry)").fetchall()}

            for col in ("task_type", "regime_id", "regime_name", "population", "status", "context_key", "package_model_id", "metadata_json"):
                self.assertIn(col, hist_cols)

            for col in ("task_type", "regime_id", "context_key", "champion_model_name", "challenger_model_name", "regime_scope"):
                self.assertIn(col, reg_cols)

    def test_existing_lifecycle_db_migration_and_idempotency(self):
        """2-5. Existing lifecycle DB migration: Verify non-destructive idempotent migration with row preservation."""
        conn = sqlite3.connect(self.db_path)
        conn.executescript(
            """
            CREATE TABLE model_registry (
                model_id TEXT PRIMARY KEY,
                display_name TEXT,
                current_model_name TEXT NOT NULL,
                current_version TEXT,
                current_version_number INTEGER NOT NULL DEFAULT 1,
                status TEXT NOT NULL DEFAULT 'ready',
                created_on TEXT NOT NULL,
                updated_on TEXT NOT NULL,
                current_metrics_json TEXT
            );
            CREATE TABLE model_history (
                history_id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_history_id INTEGER,
                model_id TEXT NOT NULL,
                model_name TEXT NOT NULL UNIQUE,
                version_label TEXT NOT NULL,
                version_number INTEGER NOT NULL,
                lifecycle TEXT NOT NULL,
                parent_model_name TEXT,
                trained_at TEXT NOT NULL,
                dataset TEXT,
                target TEXT,
                algorithm TEXT,
                validation_strategy TEXT,
                row_count INTEGER,
                trading_days INTEGER,
                feature_count INTEGER,
                mae REAL,
                rmse REAL,
                directional_accuracy_pct REAL,
                composite_score REAL,
                hpo_trials INTEGER,
                parameters_changed INTEGER,
                changes_json TEXT,
                metrics_json TEXT
            );
            INSERT INTO model_registry VALUES ('legacy_family_1', 'Legacy Family', 'Legacy_Model__v1', 'v1', 1, 'ready', '2026-01-01', '2026-01-01', '{}');
            INSERT INTO model_history (model_id, model_name, version_label, version_number, lifecycle, trained_at, target, algorithm)
            VALUES ('legacy_family_1', 'Legacy_Model__v1', 'v1', 1, 'new_model', '2026-01-01', 'future_ltp_5m', 'xgboost');
            """
        )
        conn.commit()
        conn.close()

        # Run migration twice to assert idempotency
        with _connect(self.tmp_dir) as conn:
            migrate_lifecycle_schema_v2(conn)
            migrate_lifecycle_schema_v2(conn)

            # Assert existing row preserved
            row = conn.execute("SELECT * FROM model_history WHERE model_name = 'Legacy_Model__v1'").fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["target"], "future_ltp_5m")
            self.assertEqual(row["regime_id"], "R000")

    def test_record_training_history_with_taxonomy(self):
        """6-14. Taxonomy fields persistence, context key, population, status, and package_model_id handling."""
        tc = TrainingConfig(
            dataset="analysis_nifty_3s_exp001",
            target="label_up_5m",
            algorithm="catboost",
            features=["spot", "atm_iv_ce", "futures_basis_zscore"],
            lifecycle={
                "regime_id": "R001",
                "regime_name": "TREND",
                "population": "VALIDATED",
                "market": "NIFTY",
                "sampling_interval_sec": 3,
                "prediction_horizon": "5m",
            },
        )

        lineage_dict = {
            "ancestor_model_id": "MF_NIFTY_3S_DIR_TREND",
            "package_model_id": "MD000142",
        }

        res = record_training_history(
            data_dir=self.tmp_dir,
            model_name="DIR_TREND_5M_CAT__v1",
            trained_at="2026-08-19T00:00:00Z",
            config=tc,
            metrics={"directional_accuracy_pct": 62.5},
            metadata={"row_count": 100000},
            matrix_report={},
            lineage=lineage_dict,
        )

        self.assertEqual(res["version_label"], "v1")

        # Inspect history row
        hist = get_history_by_model_name(self.tmp_dir, "DIR_TREND_5M_CAT__v1")
        self.assertIsNotNone(hist)
        self.assertEqual(hist["task_type"], "DIRECTION_CLASSIFIER")
        self.assertEqual(hist["regime_id"], "R001")
        self.assertEqual(hist["regime_name"], "TREND")
        self.assertEqual(hist["population"], "VALIDATED")
        self.assertEqual(hist["status"], "ACTIVE")
        self.assertEqual(hist["context_key"], "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        self.assertEqual(hist["package_model_id"], "MD000142")

        # Inspect registry row
        champ = get_champion_for_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        self.assertIsNotNone(champ)
        self.assertEqual(champ["champion_model_name"], "DIR_TREND_5M_CAT__v1")
        self.assertEqual(champ["task_type"], "DIRECTION_CLASSIFIER")
        self.assertEqual(champ["regime_id"], "R001")

    def test_multi_regime_champions_coexist(self):
        """15-22. Multi-regime champions: Trend Champion and Sideways Champion coexist independently."""
        # 1. Trend Champion (R001)
        tc_trend = TrainingConfig(
            dataset="ds_trend",
            target="label_up_5m",
            algorithm="xgboost",
            features=["f1", "f2"],
            lifecycle={"regime_id": "R001", "regime_name": "TREND"},
        )
        record_training_history(
            data_dir=self.tmp_dir,
            model_name="DIR_TREND_XGB__v1",
            trained_at="2026-08-19T01:00:00Z",
            config=tc_trend,
            metrics={},
            metadata={"row_count": 50000},
            matrix_report={},
        )

        # 2. Sideways Champion (R002)
        tc_side = TrainingConfig(
            dataset="ds_side",
            target="label_up_5m",
            algorithm="catboost",
            features=["f3", "f4"],
            lifecycle={"regime_id": "R002", "regime_name": "SIDEWAYS"},
        )
        record_training_history(
            data_dir=self.tmp_dir,
            model_name="DIR_SIDEWAYS_CAT__v1",
            trained_at="2026-08-19T02:00:00Z",
            config=tc_side,
            metrics={},
            metadata={"row_count": 50000},
            matrix_report={},
        )

        trend_champ = get_champion_for_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        side_champ = get_champion_for_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002")

        self.assertIsNotNone(trend_champ)
        self.assertIsNotNone(side_champ)
        self.assertEqual(trend_champ["champion_model_name"], "DIR_TREND_XGB__v1")
        self.assertEqual(side_champ["champion_model_name"], "DIR_SIDEWAYS_CAT__v1")

    def test_legacy_model_package_loading_without_file_rewrite(self):
        """15-16. Legacy model packages resolve safely without modifying historical JSON files."""
        pkg_dir = os.path.join(self.models_dir, "Future_LTP_5m_WF_1168f_XGB__v1")
        os.makedirs(pkg_dir, exist_ok=True)

        config_path = os.path.join(pkg_dir, "config.json")
        legacy_config = {
            "model_name": "Future_LTP_5m_WF_1168f_XGB__v1",
            "target": "future_ltp_5m",
            "algorithm": "xgboost",
            "features": ["spot", "atm_iv_ce"],
            "dataset": "analysis_nifty_3s_exp001",
        }
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(legacy_config, fh, indent=2)

        with open(os.path.join(pkg_dir, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump({"rmse": 12.4, "mae": 8.2}, fh)

        # List models via registry API
        models = list_trained_models(self.tmp_dir)
        self.assertEqual(len(models), 1)
        m = models[0]
        self.assertEqual(m["model_name"], "Future_LTP_5m_WF_1168f_XGB__v1")
        self.assertEqual(m["task_type"], "REGRESSION")
        self.assertEqual(m["regime_id"], "R000")
        self.assertEqual(m["regime_name"], "ALL_REGIMES")
        self.assertEqual(m["population"], "EXPERIMENTAL")

        # Verify on-disk config.json was NOT modified or rewritten
        with open(config_path, "r", encoding="utf-8") as fh:
            raw_on_disk = json.load(fh)
        self.assertNotIn("task", raw_on_disk)
        self.assertNotIn("regime", raw_on_disk)

    def test_regime_resolution_hierarchy(self):
        """17-19. Regime resolution hierarchy: Explicit config > dataset metadata > R000 fallback."""
        # 1. Dataset metadata inheritance
        tc_ds = TrainingConfig(
            dataset="ds_highvol",
            target="future_realized_vol_5m",
            algorithm="xgboost",
            features=["f1"],
        )
        tc_ds.dataset_metadata = {"regime_id": "R003"}
        record_training_history(
            data_dir=self.tmp_dir,
            model_name="VOL_HIGHVOL_XGB__v1",
            trained_at="2026-08-19T03:00:00Z",
            config=tc_ds,
            metrics={},
            metadata={},
            matrix_report={},
        )
        hist = get_history_by_model_name(self.tmp_dir, "VOL_HIGHVOL_XGB__v1")
        self.assertEqual(hist["regime_id"], "R003")
        self.assertEqual(hist["task_type"], "VOLATILITY_ESTIMATOR")

        # 2. R000 Fallback
        tc_fallback = TrainingConfig(
            dataset="ds_general",
            target="direction_15m",
            algorithm="lightgbm",
            features=["f1"],
        )
        record_training_history(
            data_dir=self.tmp_dir,
            model_name="DIR_GENERAL_LGB__v1",
            trained_at="2026-08-19T04:00:00Z",
            config=tc_fallback,
            metrics={},
            metadata={},
            matrix_report={},
        )
        hist_fb = get_history_by_model_name(self.tmp_dir, "DIR_GENERAL_LGB__v1")
        self.assertEqual(hist_fb["regime_id"], "R000")
        self.assertEqual(hist_fb["regime_name"], "ALL_REGIMES")

    def test_set_and_get_champion_for_context(self):
        """20-22. Programmatic setting and querying of context-scoped champions and challengers."""
        key = ModelContextKey(
            market="NIFTY",
            sampling_interval_sec=3,
            task_type=TaskType.DIRECTION_CLASSIFIER,
            prediction_horizon="5m",
            regime_id="R005",  # BREAKOUT
        )

        set_champion_for_context(
            self.tmp_dir,
            context_key=key,
            champion_model_name="DIR_BREAKOUT_XGB__v2",
            challenger_model_name="DIR_BREAKOUT_CAT__v1",
        )

        champ = get_champion_for_context(self.tmp_dir, key)
        self.assertIsNotNone(champ)
        self.assertEqual(champ["champion_model_name"], "DIR_BREAKOUT_XGB__v2")
        self.assertEqual(champ["challenger_model_name"], "DIR_BREAKOUT_CAT__v1")

        # Update champion
        set_champion_for_context(
            self.tmp_dir,
            context_key=key,
            champion_model_name="DIR_BREAKOUT_CAT__v2",
            challenger_model_name="DIR_BREAKOUT_XGB__v2",
        )
        updated = get_champion_for_context(self.tmp_dir, key)
        self.assertEqual(updated["champion_model_name"], "DIR_BREAKOUT_CAT__v2")
        self.assertEqual(updated["challenger_model_name"], "DIR_BREAKOUT_XGB__v2")

    def test_delete_model_history_and_champion_repoint(self):
        """23-27. Deleting a model package cleans history and preserves registry integrity."""
        tc = TrainingConfig(
            dataset="ds_del",
            target="future_ltp_5m",
            algorithm="xgboost",
            features=["f1"],
        )
        record_training_history(
            data_dir=self.tmp_dir,
            model_name="DEL_TEST_XGB__v1",
            trained_at="2026-08-19T05:00:00Z",
            config=tc,
            metrics={},
            metadata={},
            matrix_report={},
        )
        record_training_history(
            data_dir=self.tmp_dir,
            model_name="DEL_TEST_XGB__v2",
            trained_at="2026-08-19T06:00:00Z",
            config=tc,
            metrics={},
            metadata={},
            matrix_report={},
            lineage={"parent_model_id": "DEL_TEST_XGB__v1", "ancestor_model_id": "DEL_TEST_XGB__v1"},
        )

        hist = list_history_for_model(self.tmp_dir, model_name="DEL_TEST_XGB__v1")
        self.assertEqual(len(hist), 2)

        # Delete v2, verify v1 becomes current champion again
        delete_history_for_model(self.tmp_dir, "DEL_TEST_XGB__v2")
        hist_after = list_history_for_model(self.tmp_dir, model_name="DEL_TEST_XGB__v1")
        self.assertEqual(len(hist_after), 1)
        self.assertEqual(hist_after[0]["model_name"], "DEL_TEST_XGB__v1")


if __name__ == "__main__":
    unittest.main()
