"""Tests for Math / Normalization / Regime transforms."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from chain_replay_ml.dataset_builder.transformations import (
    list_registered_transformations,
    run_transformation_pipeline,
)
from chain_replay_ml.dataset_builder.transformations.base import TransformContext
from chain_replay_ml.dataset_builder.transformations.math_transform import math_column_name
from chain_replay_ml.dataset_builder.transformations.math_ui import (
    build_math_transformation_config,
)
from chain_replay_ml.dataset_builder.transformations.normalization import (
    normalization_column_name,
)
from chain_replay_ml.dataset_builder.transformations.normalization_ui import (
    build_normalization_transformation_config,
)
from chain_replay_ml.dataset_builder.transformations.regime import regime_column_name
from chain_replay_ml.dataset_builder.transformations.regime_ui import (
    build_regime_transformation_config,
)


def _frame(vals: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "trading_day": ["2024-01-01"] * len(vals),
        "token": ["A"] * len(vals),
        "x": vals,
    })


class TestNewTransformFamilies(unittest.TestCase):
    def test_registered(self) -> None:
        ids = {t.id for t in list_registered_transformations()}
        self.assertIn("math", ids)
        self.assertIn("normalization", ids)
        self.assertIn("regime", ids)
        rolling = next(t for t in list_registered_transformations() if t.id == "rolling")
        self.assertEqual(rolling.name, "Rolling Statistics")

    def test_math_abs_log_clip(self) -> None:
        self.assertEqual(math_column_name("ltp", "abs"), "ltp_abs")
        df = _frame([1.0, -2.0, 4.0, 9.0])
        cfg = build_math_transformation_config(
            enabled=True,
            features=["x"],
            operations=["abs", "log", "square"],
        )
        result = run_transformation_pipeline(
            df, cfg, context=TransformContext(config=cfg, sample_interval_sec=3.0)
        )
        self.assertEqual(result.executed, 1)
        self.assertAlmostEqual(result.frame.loc[1, "x_abs"], 2.0)
        self.assertTrue(np.isnan(result.frame.loc[1, "x_log"]))
        self.assertAlmostEqual(result.frame.loc[2, "x_log"], np.log(4.0))
        self.assertAlmostEqual(result.frame.loc[2, "x_square"], 16.0)

    def test_normalization_rolling_zscore(self) -> None:
        self.assertEqual(
            normalization_column_name("spot", "zscore_rolling", window=3),
            "spot_zscore_3",
        )
        vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        df = _frame(vals)
        cfg = build_normalization_transformation_config(
            enabled=True,
            features=["x"],
            methods=["zscore_rolling", "zscore_expanding"],
            windows=[3],
        )
        result = run_transformation_pipeline(
            df, cfg, context=TransformContext(config=cfg, sample_interval_sec=3.0)
        )
        self.assertEqual(result.executed, 1)
        self.assertIn("x_zscore_3", result.created_columns)
        self.assertIn("x_zscore_exp", result.created_columns)
        # Window [1,2,3]: mean=2, std(ddof=0)=sqrt(2/3)
        mean = 2.0
        std = float(np.std([1.0, 2.0, 3.0], ddof=0))
        self.assertAlmostEqual(result.frame.loc[2, "x_zscore_3"], (3.0 - mean) / std)

    def test_regime_binary_ternary(self) -> None:
        self.assertEqual(regime_column_name("obi", "binary_threshold", threshold=0.0), "obi_bin_0")
        df = _frame([-2.0, -0.5, 0.0, 0.5, 2.0])
        cfg = build_regime_transformation_config(
            enabled=True,
            features=["x"],
            methods=["binary_threshold", "ternary_state"],
            windows=[],
            threshold=0.0,
            low=-1.0,
            high=1.0,
        )
        result = run_transformation_pipeline(
            df, cfg, context=TransformContext(config=cfg, sample_interval_sec=3.0)
        )
        self.assertEqual(result.executed, 1)
        self.assertEqual(result.frame.loc[0, "x_bin_0"], 0.0)
        self.assertEqual(result.frame.loc[2, "x_bin_0"], 1.0)
        self.assertEqual(result.frame.loc[0, "x_tern_-1_1"], -1.0)
        self.assertEqual(result.frame.loc[2, "x_tern_-1_1"], 0.0)
        self.assertEqual(result.frame.loc[4, "x_tern_-1_1"], 1.0)


if __name__ == "__main__":
    unittest.main()
