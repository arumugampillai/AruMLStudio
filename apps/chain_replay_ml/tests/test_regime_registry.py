"""Comprehensive Tests for Phase 4C.3: Market Regime Registry Foundation."""

import json
import os
import shutil
import tempfile
import unittest

from chain_replay_ml.model_taxonomy import (
    BASELINE_REGIME_CATALOG,
    DEFAULT_REGIME_ID,
    DEFAULT_REGIME_NAME,
    ModelContextKey,
    RegimeScope,
    RegimeSpec,
    TaskType,
    compute_regime_definition_hash,
    get_regime_record,
    list_regimes,
    load_regime_registry,
    reactivate_regime,
    regime_registry_path,
    register_regime,
    retire_regime,
    save_regime_registry,
    update_regime_definition,
    validate_regime_id_format,
)


class TestRegimeRegistry(unittest.TestCase):
    """Test suite for Phase 4C.3 Regime Registry store, versioning, hashing, and hierarchy."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp(prefix="aruml_test_regimes_")

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_fresh_registry_initialization(self):
        """1. Verify that loading from a fresh directory initializes baseline regimes R000–R007."""
        store = load_regime_registry(self.tmp_dir)
        self.assertEqual(store.get("schema_version"), "1.0")
        self.assertEqual(store.get("default_regime_id"), "R000")

        regimes = store.get("regimes", {})
        self.assertEqual(len(regimes), 8)

        # Check all baseline IDs present
        for rid in ("R000", "R001", "R002", "R003", "R004", "R005", "R006", "R007"):
            self.assertIn(rid, regimes)
            rec = regimes[rid]
            self.assertEqual(rec["regime_id"], rid)
            self.assertEqual(rec["status"], "ACTIVE")
            self.assertEqual(rec["current_version"], 1)
            self.assertTrue(len(rec["definition_hash"]) == 64)
            self.assertEqual(len(rec["version_history"]), 1)

    def test_r000_universal_root_properties(self):
        """2. Verify R000 is correctly configured as the Universal Root regime."""
        r0 = get_regime_record(self.tmp_dir, "R000")
        self.assertIsNotNone(r0)
        self.assertEqual(r0["regime_name"], "ALL_REGIMES")
        self.assertEqual(r0["scope"], "ALL_REGIMES")
        self.assertIsNone(r0["parent_regime_id"])
        self.assertEqual(r0["detection_type"], "UNIVERSAL")
        self.assertEqual(r0["required_features"], [])

    def test_baseline_regimes_hierarchy_and_features(self):
        """3. Verify baseline specialized regimes R001-R007 inherit from R000 and have required features."""
        for rid in ("R001", "R002", "R003", "R004", "R005", "R006", "R007"):
            rec = get_regime_record(self.tmp_dir, rid)
            self.assertIsNotNone(rec)
            self.assertEqual(rec["parent_regime_id"], "R000")
            self.assertEqual(rec["scope"], "SPECIALIZED")
            self.assertTrue(len(rec["required_features"]) > 0)
            self.assertIn("spot", rec["required_features"])

    def test_deterministic_definition_hash(self):
        """4. Verify definition hash calculation is deterministic, order-invariant, and case-normalized."""
        h1 = compute_regime_definition_hash(
            detection_type="rule_based",
            detection_spec={"threshold": 25.0, "primary_indicator": "adx_14", "condition": "gte"},
            required_features=["adx_14", "spot"],
            parent_regime_id="r000",
        )
        h2 = compute_regime_definition_hash(
            detection_type="RULE_BASED",
            detection_spec={"primary_indicator": "adx_14", "condition": "gte", "threshold": 25.0},
            required_features=["spot", "adx_14"],
            parent_regime_id="R000",
        )
        self.assertEqual(h1, h2)

        # Different parameter -> Different hash
        h3 = compute_regime_definition_hash(
            detection_type="RULE_BASED",
            detection_spec={"primary_indicator": "adx_14", "condition": "gte", "threshold": 30.0},
            required_features=["spot", "adx_14"],
            parent_regime_id="R000",
        )
        self.assertNotEqual(h1, h3)

    def test_register_custom_regime(self):
        """5. Verify registering a new custom regime."""
        rec = register_regime(
            self.tmp_dir,
            regime_id="R008_GAMMA_SQUEEZE",
            regime_name="GAMMA_SQUEEZE",
            display_name="Extreme Gamma Squeeze",
            family="OPTION_GAMMA",
            description="Rapid delta expansion caused by market maker gamma hedging.",
            scope="SPECIALIZED",
            parent_regime_id="R000",
            detection_type="RULE_BASED",
            detection_spec={"primary_indicator": "net_gamma_imbalance", "condition": "gte", "threshold": 3.0},
            required_features=["spot", "net_gamma_imbalance"],
        )

        self.assertEqual(rec["regime_id"], "R008_GAMMA_SQUEEZE")
        self.assertEqual(rec["current_version"], 1)

        queried = get_regime_record(self.tmp_dir, "R008_GAMMA_SQUEEZE")
        self.assertIsNotNone(queried)
        self.assertEqual(queried["family"], "OPTION_GAMMA")

    def test_invalid_and_duplicate_regime_id(self):
        """6. Verify ID validation and duplicate rejection."""
        self.assertFalse(validate_regime_id_format("INVALID_ID"))
        self.assertFalse(validate_regime_id_format("123"))
        self.assertTrue(validate_regime_id_format("R001"))
        self.assertTrue(validate_regime_id_format("R008_CUSTOM_REGIME"))

        with self.assertRaises(ValueError):
            register_regime(
                self.tmp_dir,
                regime_id="INVALID_NAME",
                regime_name="TEST",
            )

        with self.assertRaises(ValueError):
            # Duplicate R001
            register_regime(
                self.tmp_dir,
                regime_id="R001",
                regime_name="DUPLICATE_TREND",
            )

    def test_update_regime_definition_versioning_and_history(self):
        """7. Verify definition updates bump version and archive historical version."""
        r1_orig = get_regime_record(self.tmp_dir, "R001")
        orig_hash = r1_orig["definition_hash"]
        self.assertEqual(r1_orig["current_version"], 1)

        # 1. Non-breaking update (description only) -> Version does NOT bump
        update_regime_definition(
            self.tmp_dir,
            "R001",
            description="Updated trend description without changing mathematical rules.",
        )
        r1_desc = get_regime_record(self.tmp_dir, "R001")
        self.assertEqual(r1_desc["current_version"], 1)
        self.assertEqual(r1_desc["definition_hash"], orig_hash)

        # 2. Mathematical rule change -> Version bumps to 2
        update_regime_definition(
            self.tmp_dir,
            "R001",
            detection_spec={"primary_indicator": "adx_14", "condition": "gte", "threshold": 30.0},
        )
        r1_v2 = get_regime_record(self.tmp_dir, "R001")
        self.assertEqual(r1_v2["current_version"], 2)
        self.assertNotEqual(r1_v2["definition_hash"], orig_hash)
        self.assertEqual(len(r1_v2["version_history"]), 2)

        # 3. Query historical version 1
        r1_v1 = get_regime_record(self.tmp_dir, "R001", version=1)
        self.assertIsNotNone(r1_v1)
        self.assertEqual(r1_v1["version"], 1)
        self.assertEqual(r1_v1["definition_hash"], orig_hash)
        self.assertEqual(r1_v1["detection_spec"]["threshold"], 25.0)

    def test_retire_and_reactivate_regime(self):
        """8. Verify retiring a regime and reactivating it."""
        with self.assertRaises(ValueError):
            retire_regime(self.tmp_dir, "R000")  # Cannot retire R000

        retire_regime(self.tmp_dir, "R007")

        active_regimes = list_regimes(self.tmp_dir, include_retired=False)
        all_regimes = list_regimes(self.tmp_dir, include_retired=True)

        active_ids = {r["regime_id"] for r in active_regimes}
        all_ids = {r["regime_id"] for r in all_regimes}

        self.assertNotIn("R007", active_ids)
        self.assertIn("R007", all_ids)

        # Reactivate
        reactivate_regime(self.tmp_dir, "R007")
        active_ids_after = {r["regime_id"] for r in list_regimes(self.tmp_dir, include_retired=False)}
        self.assertIn("R007", active_ids_after)

    def test_hierarchy_filtering(self):
        """9. Verify listing regimes filtered by parent hierarchy."""
        # Create a sub-regime under R001
        register_regime(
            self.tmp_dir,
            regime_id="R001_BULL_MOMENTUM",
            regime_name="BULL_MOMENTUM",
            parent_regime_id="R001",
            detection_spec={"primary_indicator": "supertrend_direction", "condition": "eq", "threshold": 1.0},
        )

        r0_children = list_regimes(self.tmp_dir, parent_id="R000")
        r1_children = list_regimes(self.tmp_dir, parent_id="R001")

        r0_ids = {r["regime_id"] for r in r0_children}
        r1_ids = {r["regime_id"] for r in r1_children}

        self.assertIn("R001", r0_ids)
        self.assertNotIn("R001_BULL_MOMENTUM", r0_ids)
        self.assertIn("R001_BULL_MOMENTUM", r1_ids)

    def test_model_context_key_invariant(self):
        """10. Verify ModelContextKey remains strictly the 5-dimensional operational routing key."""
        ctx = ModelContextKey(
            market="NIFTY",
            sampling_interval_sec=3,
            task_type=TaskType.DIRECTION_CLASSIFIER,
            prediction_horizon="5m",
            regime_id="R001",
        )
        self.assertEqual(ctx.canonical_key_str(), "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")


if __name__ == "__main__":
    unittest.main()
