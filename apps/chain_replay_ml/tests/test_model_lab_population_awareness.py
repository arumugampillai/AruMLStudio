"""Comprehensive Test Suite for Phase 4C.4: Model Research Lab Population Awareness."""

import json
import os
import shutil
import tempfile
import unittest

from chain_replay_ml.model_taxonomy import (
    DEFAULT_REGIME_ID,
    DEFAULT_REGIME_NAME,
    ModelContextKey,
    ModelLifecycleStatus,
    ModelPopulationTier,
    RegimeScope,
    RegimeSpec,
    TaskSpec,
    TaskType,
    filter_model_records,
    format_model_taxonomy_display,
    get_context_champions_map,
    list_regimes,
    resolve_model_metadata_or_legacy,
)
from chain_replay_ml.research_memory.champion_history import (
    get_champion_for_context,
    set_champion_for_context,
)


class TestModelLabPopulationAwareness(unittest.TestCase):
    """Test suite verifying Model Research Lab taxonomy awareness, faceted filtering, and champion scoping."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_lab_pop_")

        # Create a rich population of test model records
        self.model_records = [
            # 1. Trend Direction Specialist (Champion)
            {
                "model_name": "DIR_TREND_XGB__v1",
                "target": "label_up_5m",
                "label_strategy": "fixed_horizon",
                "task_type": "DIRECTION_CLASSIFIER",
                "regime_id": "R001",
                "regime_name": "TREND",
                "population": "CHAMPION",
                "status": "ACTIVE",
                "sampling_interval_sec": 3,
                "dataset": "analysis_nifty_3s_trend.parquet",
            },
            # 2. Trend Direction Specialist (Challenger)
            {
                "model_name": "DIR_TREND_CAT__v2",
                "target": "label_up_5m",
                "label_strategy": "fixed_horizon",
                "task_type": "DIRECTION_CLASSIFIER",
                "regime_id": "R001",
                "regime_name": "TREND",
                "population": "CHALLENGER",
                "status": "ACTIVE",
                "sampling_interval_sec": 3,
                "dataset": "analysis_nifty_3s_trend.parquet",
            },
            # 3. Sideways Direction Specialist (Champion)
            {
                "model_name": "DIR_SIDEWAYS_LGBM__v1",
                "target": "label_up_5m",
                "label_strategy": "fixed_horizon",
                "task_type": "DIRECTION_CLASSIFIER",
                "regime_id": "R002",
                "regime_name": "SIDEWAYS",
                "population": "CHAMPION",
                "status": "ACTIVE",
                "sampling_interval_sec": 3,
                "dataset": "analysis_nifty_3s_sideways.parquet",
            },
            # 4. Universal Direction Baseline (Validated)
            {
                "model_name": "DIR_UNIVERSAL_XGB__v3",
                "target": "label_up_5m",
                "label_strategy": "fixed_horizon",
                "task_type": "DIRECTION_CLASSIFIER",
                "regime_id": "R000",
                "regime_name": "ALL_REGIMES",
                "population": "VALIDATED",
                "status": "ACTIVE",
                "sampling_interval_sec": 3,
                "dataset": "analysis_nifty_3s_all.parquet",
            },
            # 5. Volatility Estimator (Experimental)
            {
                "model_name": "VOL_HIGHVOL_RF__v1",
                "target": "realized_vol_5m",
                "label_strategy": "fixed_horizon",
                "task_type": "VOLATILITY_ESTIMATOR",
                "regime_id": "R003",
                "regime_name": "HIGH_VOLATILITY",
                "population": "EXPERIMENTAL",
                "status": "ACTIVE",
                "sampling_interval_sec": 3,
                "dataset": "analysis_nifty_3s_highvol.parquet",
            },
            # 6. Triple Barrier Specialist (Deprecated)
            {
                "model_name": "TB_BREAKOUT_XGB__v1",
                "target": "triple_barrier_signal",
                "label_strategy": "triple_barrier",
                "task_type": "TRIPLE_BARRIER",
                "regime_id": "R005",
                "regime_name": "BREAKOUT",
                "population": "EXPERIMENTAL",
                "status": "DEPRECATED",
                "sampling_interval_sec": 3,
                "dataset": "analysis_nifty_3s_tb.parquet",
            },
            # 7. Legacy Model (No explicit taxonomy metadata)
            {
                "model_name": "LEGACY_REG_MODEL__v0",
                "target": "future_ltp_5m",
                "sampling_interval_sec": 3,
                "dataset": "analysis_legacy.parquet",
            },
        ]

        # Register champions in lifecycle DB
        set_champion_for_context(
            self.tmp_dir,
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001",
            champion_model_name="DIR_TREND_XGB__v1",
        )
        set_champion_for_context(
            self.tmp_dir,
            context_key="NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002",
            champion_model_name="DIR_SIDEWAYS_LGBM__v1",
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_task_type_filtering(self):
        """1. Verify filtering by TaskType."""
        dir_models = filter_model_records(self.model_records, task_type="DIRECTION_CLASSIFIER")
        tb_models = filter_model_records(self.model_records, task_type="TRIPLE_BARRIER")
        vol_models = filter_model_records(self.model_records, task_type="VOLATILITY_ESTIMATOR")
        reg_models = filter_model_records(self.model_records, task_type="REGRESSION")

        self.assertEqual(len(dir_models), 4)  # 3 explicit + 1 universal
        self.assertEqual(len(tb_models), 1)
        self.assertEqual(len(vol_models), 1)
        self.assertEqual(len(reg_models), 1)  # Legacy model resolves to REGRESSION

    def test_regime_filtering(self):
        """2. Verify filtering by Market Regime."""
        trend_models = filter_model_records(self.model_records, regime_id="R001")
        sideways_models = filter_model_records(self.model_records, regime_id="R002")
        universal_models = filter_model_records(self.model_records, regime_id="R000")

        self.assertEqual(len(trend_models), 2)
        self.assertEqual(len(sideways_models), 1)
        self.assertEqual(len(universal_models), 2)  # DIR_UNIVERSAL_XGB__v3 + LEGACY_REG_MODEL__v0

    def test_population_filtering(self):
        """3. Verify filtering by Population Tier."""
        champions = filter_model_records(self.model_records, population="CHAMPION")
        challengers = filter_model_records(self.model_records, population="CHALLENGER")
        validated = filter_model_records(self.model_records, population="VALIDATED")
        experimental = filter_model_records(self.model_records, population="EXPERIMENTAL")

        self.assertEqual(len(champions), 2)
        self.assertEqual(len(challengers), 1)
        self.assertEqual(len(validated), 1)
        self.assertEqual(len(experimental), 3)  # RF + TB + Legacy default

    def test_lifecycle_filtering(self):
        """4. Verify filtering by Lifecycle Status."""
        active = filter_model_records(self.model_records, lifecycle_status="ACTIVE")
        deprecated = filter_model_records(self.model_records, lifecycle_status="DEPRECATED")

        self.assertEqual(len(active), 6)
        self.assertEqual(len(deprecated), 1)
        self.assertEqual(deprecated[0]["model_name"], "TB_BREAKOUT_XGB__v1")

    def test_combined_multi_dimensional_filtering(self):
        """5. Verify combined multi-dimensional filtering across all four dimensions."""
        matched = filter_model_records(
            self.model_records,
            task_type="DIRECTION_CLASSIFIER",
            regime_id="R001",
            population="CHAMPION",
            lifecycle_status="ACTIVE",
        )
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["model_name"], "DIR_TREND_XGB__v1")

    def test_empty_filter_match(self):
        """6. Verify empty filter match returns empty list safely."""
        matched = filter_model_records(
            self.model_records,
            task_type="REGRESSION",
            regime_id="R001",  # No regression models in R001
        )
        self.assertEqual(len(matched), 0)

    def test_trend_and_sideways_regime_separation(self):
        """7. Verify Trend and Sideways remain strictly distinct regimes."""
        trend_m = filter_model_records(self.model_records, regime_id="R001")
        side_m = filter_model_records(self.model_records, regime_id="R002")

        trend_names = {r["model_name"] for r in trend_m}
        side_names = {r["model_name"] for r in side_m}

        self.assertTrue(trend_names.isdisjoint(side_names))
        self.assertIn("DIR_TREND_XGB__v1", trend_names)
        self.assertIn("DIR_SIDEWAYS_LGBM__v1", side_names)

    def test_context_champions_scoping(self):
        """8. Verify Trend Champion cannot appear as Sideways Champion."""
        champs_map = get_context_champions_map(self.tmp_dir)

        r_trend = self.model_records[0]  # DIR_TREND_XGB__v1
        r_side = self.model_records[2]   # DIR_SIDEWAYS_LGBM__v1

        disp_trend = format_model_taxonomy_display(r_trend, champions_map=champs_map)
        disp_side = format_model_taxonomy_display(r_side, champions_map=champs_map)

        self.assertEqual(disp_trend["context_key"], "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        self.assertEqual(disp_side["context_key"], "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002")

        self.assertTrue(disp_trend["is_champion"])
        self.assertTrue(disp_side["is_champion"])
        self.assertEqual(disp_trend["population_badge"], "👑 CHAMPION")
        self.assertEqual(disp_side["population_badge"], "👑 CHAMPION")

        # Check that query for Trend context returns DIR_TREND_XGB__v1
        trend_champ_doc = get_champion_for_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")
        self.assertEqual(trend_champ_doc["champion_model_name"], "DIR_TREND_XGB__v1")

        # Check that query for Sideways context returns DIR_SIDEWAYS_LGBM__v1
        side_champ_doc = get_champion_for_context(self.tmp_dir, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R002")
        self.assertEqual(side_champ_doc["champion_model_name"], "DIR_SIDEWAYS_LGBM__v1")
        self.assertNotEqual(side_champ_doc["champion_model_name"], "DIR_TREND_XGB__v1")

    def test_legacy_model_resolution(self):
        """9. Verify legacy model without taxonomy metadata resolves safely to defaults."""
        r_legacy = self.model_records[6]
        disp_legacy = format_model_taxonomy_display(r_legacy)

        self.assertEqual(disp_legacy["task_type"], "REGRESSION")
        self.assertEqual(disp_legacy["regime_id"], "R000")
        self.assertEqual(disp_legacy["population"], "EXPERIMENTAL")
        self.assertEqual(disp_legacy["status"], "ACTIVE")
        self.assertEqual(disp_legacy["context_key"], "NIFTY_3s_REGRESSION_5m_R000")

    def test_dynamic_regime_loading(self):
        """10. Verify dynamic listing of regimes from registry store."""
        regimes = list_regimes(self.tmp_dir, include_retired=True)
        self.assertTrue(len(regimes) >= 8)
        ids = {r["regime_id"] for r in regimes}
        self.assertIn("R000", ids)
        self.assertIn("R001", ids)
        self.assertIn("R002", ids)


if __name__ == "__main__":
    unittest.main()
