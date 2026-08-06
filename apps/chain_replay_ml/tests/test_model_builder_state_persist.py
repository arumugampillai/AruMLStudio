"""Tests for Create Model panel config persistence."""

from __future__ import annotations

import tempfile
import unittest

from master_dataset_tk.model_builder.state import (
    ModelBuilderState,
    load_persisted_state,
    save_persisted_state,
)


class ModelBuilderStatePersistTests(unittest.TestCase):
    def test_roundtrip_data_split_and_ui(self) -> None:
        state = ModelBuilderState(
            dataset="master_nifty_3s",
            target="future_ltp_5m",
            features={"delta", "gamma", "ltp"},
            split_train=60,
            split_val=20,
            split_test=20,
            validation_strategy="walk_forward",
            wf_folds=7,
            wf_window_mode="rolling",
            wf_train_window=4000,
            wf_val_window=800,
            wf_feature_selection="shap",
            wf_opt_metric="rmse",
            wf_hpo_enabled=False,
            wf_hpo_trials=40,
            global_hpo_enabled=True,
            global_hpo_trials=50,
            model_name="my_custom_model",
            model_name_manual=True,
            show_features=True,
            show_advanced_params=True,
            lifecycle_mode="feature_optimization",
            lifecycle_feature_mode="locked",
            lifecycle={"source_model": "baseline_v1"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            chart_dir = tmp
            save_persisted_state(chart_dir, state)
            loaded = load_persisted_state(chart_dir)
            self.assertIsNotNone(loaded)
            restored = ModelBuilderState()
            restored.apply_saved_dict(loaded or {})
        self.assertEqual(restored.dataset, "master_nifty_3s")
        self.assertEqual(restored.target, "future_ltp_5m")
        self.assertEqual(restored.features, {"delta", "gamma", "ltp"})
        self.assertEqual(restored.split_train, 60)
        self.assertEqual(restored.split_val, 20)
        self.assertEqual(restored.split_test, 20)
        self.assertEqual(restored.validation_strategy, "walk_forward")
        self.assertEqual(restored.wf_folds, 7)
        self.assertEqual(restored.wf_window_mode, "rolling")
        self.assertEqual(restored.wf_train_window, 4000)
        self.assertEqual(restored.wf_val_window, 800)
        self.assertEqual(restored.wf_feature_selection, "shap")
        self.assertEqual(restored.wf_opt_metric, "rmse")
        self.assertFalse(restored.wf_hpo_enabled)
        self.assertEqual(restored.wf_hpo_trials, 40)
        self.assertTrue(restored.global_hpo_enabled)
        self.assertEqual(restored.global_hpo_trials, 50)
        self.assertEqual(restored.model_name, "")
        self.assertFalse(restored.model_name_manual)
        self.assertTrue(restored.show_features)
        self.assertTrue(restored.show_advanced_params)
        self.assertIsNone(restored.lifecycle_mode)
        self.assertIsNone(restored.lifecycle)

    def test_lifecycle_not_persisted(self) -> None:
        state = ModelBuilderState(
            dataset="master_nifty_3s",
            lifecycle_mode="retrain",
            lifecycle_feature_mode="locked",
            lifecycle={"source_model": "baseline_v1"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            save_persisted_state(tmp, state)
            loaded = load_persisted_state(tmp)
        self.assertIsNotNone(loaded)
        self.assertNotIn("lifecycle", loaded or {})
        restored = ModelBuilderState()
        restored.apply_saved_dict(loaded or {})
        self.assertIsNone(restored.lifecycle_mode)
        self.assertIsNone(restored.lifecycle)

    def test_model_name_not_persisted(self) -> None:
        state = ModelBuilderState(
            dataset="master_nifty_3s",
            model_name="my_custom_model",
            model_name_manual=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            save_persisted_state(tmp, state)
            loaded = load_persisted_state(tmp)
        self.assertIsNotNone(loaded)
        self.assertNotIn("modelName", loaded or {})
        self.assertNotIn("modelNameManual", loaded or {})
        restored = ModelBuilderState()
        restored.apply_saved_dict(loaded or {})
        self.assertEqual(restored.model_name, "")
        self.assertFalse(restored.model_name_manual)

    def test_split_validation_alias(self) -> None:
        restored = ModelBuilderState()
        restored.apply_saved_dict({
            "split": {"train": 55, "validation": 25, "test": 20},
            "validationStrategy": "rolling_window",
        })
        self.assertEqual(restored.split_train, 55)
        self.assertEqual(restored.split_val, 25)
        self.assertEqual(restored.split_test, 20)
        self.assertEqual(restored.validation_strategy, "rolling_window")

    def test_premium_selection_roundtrip_and_training_config(self) -> None:
        state = ModelBuilderState(
            dataset="master_nifty_3s",
            target="future_ltp_5m",
            features={"delta"},
            premium_selection_enabled=True,
            premium_min=15.0,
            premium_max=100.0,
        )
        cfg = state.build_training_config()
        self.assertEqual(
            cfg.get("premium_selection"),
            {"enabled": True, "premium_min": 15.0, "premium_max": 100.0},
        )
        with tempfile.TemporaryDirectory() as tmp:
            save_persisted_state(tmp, state)
            loaded = load_persisted_state(tmp)
        restored = ModelBuilderState()
        restored.apply_saved_dict(loaded or {})
        self.assertTrue(restored.premium_selection_enabled)
        self.assertEqual(restored.premium_min, 15.0)
        self.assertEqual(restored.premium_max, 100.0)

        from chain_replay_ml.training.config import normalize_training_config

        tc = normalize_training_config(cfg)
        self.assertTrue(tc.premium_selection_enabled)
        self.assertEqual(tc.premium_min, 15.0)
        self.assertEqual(tc.premium_max, 100.0)


if __name__ == "__main__":
    unittest.main()
