"""Training matrix sanitization for XGBoost float32 safety."""

from __future__ import annotations

import math
import unittest
import warnings

import numpy as np
import pandas as pd

from chain_replay_ml.training.feature_matrix import sanitize_training_features


class FeatureMatrixSanitizeTests(unittest.TestCase):
    def test_replaces_inf_and_clips_float32_overflow(self) -> None:
        X = pd.DataFrame(
            {
                "huge": [3.5e42, -4.0e42],
                "inf_col": [np.inf, -np.inf],
                "ok": [1.5, 2.5],
            }
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", RuntimeWarning)
            out = sanitize_training_features(X)
        overflow_warnings = [
            w for w in caught
            if issubclass(w.category, RuntimeWarning) and "overflow" in str(w.message).lower()
        ]
        self.assertEqual(overflow_warnings, [])
        arr = out.to_numpy(dtype=np.float32)
        self.assertEqual(int(np.isinf(arr).sum()), 0)
        f32_max = float(np.finfo(np.float32).max)
        self.assertLessEqual(float(out["huge"].max()), f32_max)
        self.assertGreaterEqual(float(out["huge"].min()), -f32_max)
        self.assertTrue(math.isnan(float(out["inf_col"].iloc[0])))


if __name__ == "__main__":
    unittest.main()
