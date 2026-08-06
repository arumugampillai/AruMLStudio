"""Verify selected_features == XGBoost model feature names (order + count)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from chain_replay_ml.dataset_builder.analysis_experiment_lifecycle import (
    assert_selected_features_match_model,
    extract_model_feature_names,
)


class FeatureNamesMatchTests(unittest.TestCase):
    def test_assert_match_ok(self) -> None:
        class _M:
            feature_names_in_ = np.array(["a", "b", "c"])

        names = assert_selected_features_match_model(["a", "b", "c"], _M())
        self.assertEqual(names, ["a", "b", "c"])

    def test_assert_length_mismatch(self) -> None:
        class _M:
            feature_names_in_ = np.array(["a", "b"])

        with self.assertRaises(ValueError) as ctx:
            assert_selected_features_match_model(["a", "b", "c"], _M())
        self.assertIn("len=", str(ctx.exception))

    def test_assert_order_mismatch(self) -> None:
        class _M:
            feature_names_in_ = np.array(["b", "a", "c"])

        with self.assertRaises(ValueError) as ctx:
            assert_selected_features_match_model(["a", "b", "c"], _M())
        self.assertIn("index 0", str(ctx.exception))

    def test_real_xgb_train_columns_match_persisted(self) -> None:
        try:
            from xgboost import XGBRegressor
        except ImportError:
            self.skipTest("xgboost not installed")

        rng = np.random.default_rng(0)
        feats = [f"f{i:02d}" for i in range(12)]
        n = 80
        X = pd.DataFrame(
            {c: rng.normal(size=n) for c in feats},
            columns=feats,
        )
        y = X["f00"] * 0.4 + rng.normal(scale=0.05, size=n)

        from chain_replay_ml.dataset_builder.analysis_train_device import (
            fit_xgb_regressor_gpu_first,
        )

        with mock.patch(
            "chain_replay_ml.dataset_builder.analysis_train_device."
            "resolve_experiment_xgb_plan",
            side_effect=lambda: mock.Mock(
                use_gpu=False,
                gpu_name=None,
                fallback_reason="test CPU",
                probe_notes=(),
                library_params={
                    "tree_method": "hist",
                    "device": "cpu",
                    "predictor": "cpu_predictor",
                },
            ),
        ):
            model, _info = fit_xgb_regressor_gpu_first(
                X,
                y,
                base_params={
                    "n_estimators": 8,
                    "max_depth": 2,
                    "verbosity": 0,
                },
            )
        model_names = extract_model_feature_names(model)
        self.assertEqual(len(feats), len(model_names))
        self.assertEqual(feats, model_names)
        assert_selected_features_match_model(feats, model)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.ubj"
            model.save_model(str(path))
            bst = XGBRegressor()
            bst.load_model(str(path))
            reloaded = extract_model_feature_names(bst)
            if reloaded:
                self.assertEqual(feats, reloaded)


if __name__ == "__main__":
    unittest.main()
