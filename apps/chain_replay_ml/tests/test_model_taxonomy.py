"""Unit tests for Phase 4C.1 Model Taxonomy Foundation."""

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


class TestModelTaxonomyFoundation(unittest.TestCase):
    """Test suite for 4-dimensional model taxonomy and backward-compatibility adapter."""

    def test_task_type_enum_values(self):
        self.assertEqual(TaskType.DIRECTION_CLASSIFIER.value, "DIRECTION_CLASSIFIER")
        self.assertEqual(TaskType.REGIME_CLASSIFIER.value, "REGIME_CLASSIFIER")
        self.assertEqual(TaskType.REGRESSION.value, "REGRESSION")
        self.assertEqual(TaskType.TRIPLE_BARRIER.value, "TRIPLE_BARRIER")
        self.assertEqual(TaskType.CONFIDENCE_CLASSIFIER.value, "CONFIDENCE_CLASSIFIER")
        self.assertEqual(TaskType.VOLATILITY_ESTIMATOR.value, "VOLATILITY_ESTIMATOR")

    def test_task_type_classification_predicates(self):
        self.assertTrue(TaskType.DIRECTION_CLASSIFIER.is_classification())
        self.assertTrue(TaskType.REGIME_CLASSIFIER.is_classification())
        self.assertTrue(TaskType.TRIPLE_BARRIER.is_classification())
        self.assertTrue(TaskType.CONFIDENCE_CLASSIFIER.is_classification())
        self.assertFalse(TaskType.REGRESSION.is_classification())
        self.assertFalse(TaskType.VOLATILITY_ESTIMATOR.is_classification())

        self.assertTrue(TaskType.REGRESSION.is_regression())
        self.assertTrue(TaskType.VOLATILITY_ESTIMATOR.is_regression())
        self.assertFalse(TaskType.DIRECTION_CLASSIFIER.is_regression())

    def test_task_type_from_str_and_aliases(self):
        self.assertEqual(TaskType.from_str("DIRECTION_CLASSIFIER"), TaskType.DIRECTION_CLASSIFIER)
        self.assertEqual(TaskType.from_str("direction"), TaskType.DIRECTION_CLASSIFIER)
        self.assertEqual(TaskType.from_str("REGRESSION"), TaskType.REGRESSION)
        self.assertEqual(TaskType.from_str("tb"), TaskType.TRIPLE_BARRIER)
        self.assertEqual(TaskType.from_str("confidence"), TaskType.CONFIDENCE_CLASSIFIER)
        self.assertEqual(TaskType.from_str("volatility"), TaskType.VOLATILITY_ESTIMATOR)

    def test_task_type_rejects_regimes_and_invalids(self):
        with self.assertRaises(ValueError):
            TaskType.from_str("TREND")
        with self.assertRaises(ValueError):
            TaskType.from_str("SIDEWAYS")
        with self.assertRaises(ValueError):
            TaskType.from_str("INVALID_TASK_xyz")

    def test_population_tier_enum(self):
        self.assertEqual(ModelPopulationTier.from_str("EXPERIMENTAL"), ModelPopulationTier.EXPERIMENTAL)
        self.assertEqual(ModelPopulationTier.from_str("VALIDATED"), ModelPopulationTier.VALIDATED)
        self.assertEqual(ModelPopulationTier.from_str("CHALLENGER"), ModelPopulationTier.CHALLENGER)
        self.assertEqual(ModelPopulationTier.from_str("CHAMPION"), ModelPopulationTier.CHAMPION)
        self.assertEqual(ModelPopulationTier.from_str("unknown"), ModelPopulationTier.EXPERIMENTAL)

    def test_lifecycle_status_enum(self):
        self.assertEqual(ModelLifecycleStatus.from_str("CANDIDATE"), ModelLifecycleStatus.CANDIDATE)
        self.assertEqual(ModelLifecycleStatus.from_str("ACTIVE"), ModelLifecycleStatus.ACTIVE)
        self.assertEqual(ModelLifecycleStatus.from_str("DEGRADED"), ModelLifecycleStatus.DEGRADED)
        self.assertEqual(ModelLifecycleStatus.from_str("DEPRECATED"), ModelLifecycleStatus.DEPRECATED)
        self.assertEqual(ModelLifecycleStatus.from_str("RETIRED"), ModelLifecycleStatus.RETIRED)
        self.assertEqual(ModelLifecycleStatus.from_str("ready"), ModelLifecycleStatus.ACTIVE)

    def test_regime_spec_defaults_and_baseline_catalog(self):
        spec = RegimeSpec()
        self.assertEqual(spec.regime_id, DEFAULT_REGIME_ID)
        self.assertEqual(spec.regime_name, DEFAULT_REGIME_NAME)
        self.assertEqual(spec.regime_scope, RegimeScope.ALL_REGIMES.value)

        # Baseline catalog has 8 core regimes
        self.assertIn("R000", BASELINE_REGIME_CATALOG)
        self.assertIn("R001", BASELINE_REGIME_CATALOG)
        self.assertIn("R007", BASELINE_REGIME_CATALOG)
        self.assertEqual(BASELINE_REGIME_CATALOG["R001"]["name"], "TREND")
        self.assertEqual(BASELINE_REGIME_CATALOG["R002"]["name"], "SIDEWAYS")

    def test_regime_spec_custom_roundtrip(self):
        spec = RegimeSpec(
            regime_id="R001",
            regime_name="TREND",
            regime_version=2,
            regime_scope=RegimeScope.SPECIALIZED.value,
        )
        d = spec.to_dict()
        self.assertEqual(d["regime_id"], "R001")
        self.assertEqual(d["regime_name"], "TREND")
        loaded = RegimeSpec.from_dict(d)
        self.assertEqual(loaded, spec)

    def test_model_context_key(self):
        key = ModelContextKey(
            market="NIFTY",
            sampling_interval_sec=3,
            task_type=TaskType.DIRECTION_CLASSIFIER,
            prediction_horizon="5m",
            regime_id="R001",
        )
        canonical_str = key.canonical_key_str()
        self.assertEqual(canonical_str, "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")

        parsed = ModelContextKey.from_key_str(canonical_str)
        self.assertEqual(parsed.market, "NIFTY")
        self.assertEqual(parsed.sampling_interval_sec, 3)
        self.assertEqual(parsed.task_type, TaskType.DIRECTION_CLASSIFIER)
        self.assertEqual(parsed.prediction_horizon, "5m")
        self.assertEqual(parsed.regime_id, "R001")

    def test_model_metadata_roundtrip(self):
        meta = ModelMetadata(
            model_id="MD000142",
            model_name="DIR_TREND_5M_WF_XGB_1168F",
            version=1,
            model_family_id="MF_NIFTY_3S_DIR_TREND",
            task=TaskSpec(
                task_type=TaskType.DIRECTION_CLASSIFIER,
                target="label_up_5m",
                target_type="BINARY_CLASSIFICATION",
                prediction_horizon="5m",
            ),
            regime=RegimeSpec(
                regime_id="R001",
                regime_name="TREND",
                regime_version=1,
                regime_scope=RegimeScope.SPECIALIZED.value,
            ),
            market_context={"market": "NIFTY", "sampling_interval_sec": 3},
            population=ModelPopulationTier.CHAMPION,
            status=ModelLifecycleStatus.ACTIVE,
            algorithm="xgboost",
            feature_count=1168,
        )

        d = meta.to_dict()
        self.assertEqual(d["model_id"], "MD000142")
        self.assertEqual(d["population"], "CHAMPION")
        self.assertEqual(d["task"]["task_type"], "DIRECTION_CLASSIFIER")
        self.assertEqual(d["regime"]["regime_id"], "R001")
        self.assertEqual(d["context_key_str"], "NIFTY_3s_DIRECTION_CLASSIFIER_5m_R001")

        restored = ModelMetadata.from_dict(d)
        self.assertEqual(restored.model_id, meta.model_id)
        self.assertEqual(restored.population, ModelPopulationTier.CHAMPION)
        self.assertEqual(restored.task.task_type, TaskType.DIRECTION_CLASSIFIER)

    def test_legacy_model_adapter_regression(self):
        legacy_doc = {
            "model_name": "Future_LTP_5m_WF_1168f_XGB",
            "target": "future_ltp_5m",
            "algorithm": "xgboost",
            "sampling_interval_sec": 3,
            "feature_count": 1168,
        }
        resolved = resolve_model_metadata_or_legacy(legacy_doc)
        self.assertEqual(resolved.model_name, "Future_LTP_5m_WF_1168f_XGB")
        self.assertEqual(resolved.task.task_type, TaskType.REGRESSION)
        self.assertEqual(resolved.regime.regime_id, DEFAULT_REGIME_ID)
        self.assertEqual(resolved.regime.regime_name, DEFAULT_REGIME_NAME)
        self.assertEqual(resolved.population, ModelPopulationTier.EXPERIMENTAL)
        self.assertEqual(resolved.status, ModelLifecycleStatus.ACTIVE)

    def test_legacy_model_adapter_direction(self):
        legacy_doc = {
            "model_name": "Label_Up_5m_WF_CAT_800f",
            "target": "label_up_5m",
            "algorithm": "catboost",
            "sampling_interval_sec": 3,
        }
        resolved = resolve_model_metadata_or_legacy(legacy_doc)
        self.assertEqual(resolved.task.task_type, TaskType.DIRECTION_CLASSIFIER)
        self.assertEqual(resolved.regime.regime_id, DEFAULT_REGIME_ID)

    def test_legacy_model_adapter_triple_barrier(self):
        legacy_doc = {
            "model_name": "TB_Strategy_Model_v1",
            "config": {
                "strategy_id": "triple_barrier",
                "target": "label_id",
                "prediction_type": "classification",
            },
        }
        resolved = resolve_model_metadata_or_legacy(legacy_doc)
        self.assertEqual(resolved.task.task_type, TaskType.TRIPLE_BARRIER)

    def test_legacy_model_adapter_confidence(self):
        legacy_doc = {
            "model_name": "Conf_Filter_v1",
            "config": {
                "target": "target_hit",
                "prediction_type": "classification",
            },
        }
        resolved = resolve_model_metadata_or_legacy(legacy_doc)
        self.assertEqual(resolved.task.task_type, TaskType.CONFIDENCE_CLASSIFIER)

    def test_four_dimensions_vary_independently(self):
        """Verify that TaskType, Regime, Population, and Lifecycle can vary freely."""
        # 1. Direction Classifier in Trend Regime as Champion
        m1 = ModelMetadata(
            model_id="M1",
            model_name="DIR_TREND_CHAMP",
            task=TaskSpec(TaskType.DIRECTION_CLASSIFIER, "label_up_5m"),
            regime=RegimeSpec(regime_id="R001", regime_name="TREND"),
            population=ModelPopulationTier.CHAMPION,
            status=ModelLifecycleStatus.ACTIVE,
        )
        # 2. Direction Classifier in Sideways Regime as Challenger
        m2 = ModelMetadata(
            model_id="M2",
            model_name="DIR_SIDEWAYS_CHALL",
            task=TaskSpec(TaskType.DIRECTION_CLASSIFIER, "label_up_5m"),
            regime=RegimeSpec(regime_id="R002", regime_name="SIDEWAYS"),
            population=ModelPopulationTier.CHALLENGER,
            status=ModelLifecycleStatus.ACTIVE,
        )
        # 3. Regression in High Volatility as Experimental Candidate
        m3 = ModelMetadata(
            model_id="M3",
            model_name="REG_HIGHVOL_EXP",
            task=TaskSpec(TaskType.REGRESSION, "future_ltp_5m"),
            regime=RegimeSpec(regime_id="R003", regime_name="HIGH_VOLATILITY"),
            population=ModelPopulationTier.EXPERIMENTAL,
            status=ModelLifecycleStatus.CANDIDATE,
        )

        self.assertEqual(m1.task.task_type, TaskType.DIRECTION_CLASSIFIER)
        self.assertEqual(m1.regime.regime_name, "TREND")
        self.assertEqual(m1.population, ModelPopulationTier.CHAMPION)

        self.assertEqual(m2.task.task_type, TaskType.DIRECTION_CLASSIFIER)
        self.assertEqual(m2.regime.regime_name, "SIDEWAYS")
        self.assertEqual(m2.population, ModelPopulationTier.CHALLENGER)

        self.assertEqual(m3.task.task_type, TaskType.REGRESSION)
        self.assertEqual(m3.regime.regime_name, "HIGH_VOLATILITY")
        self.assertEqual(m3.population, ModelPopulationTier.EXPERIMENTAL)


if __name__ == "__main__":
    unittest.main()
