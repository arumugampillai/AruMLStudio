"""Tests for prediction meta samples schema reader."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.prediction_meta.schema import read_prediction_samples_schema
from chain_replay_ml.prediction_meta.store import PredictionMetaStore


class TestPredictionSamplesSchema(unittest.TestCase):
    def test_read_groups_and_copy_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "pred.db")
            with PredictionMetaStore(db_path) as store:
                store.ensure_columns([
                    "current_ltp", "ensemble_mean", "model_1_pred", "model_1_delta_from_mean",
                    "actual_5m_ltp", "prediction_error",
                ])
            out = read_prediction_samples_schema(db_path)
            self.assertTrue(out["exists"])
            names = [c["name"] for c in out["columns"]]
            self.assertIn("prediction_id", names)
            self.assertIn("model_1_pred", names)
            self.assertIn("actual_5m_ltp", names)
            self.assertEqual(out["copy_text"], "\n".join(names))
            group_ids = {g["id"] for g in out["column_groups"]}
            self.assertIn("model_predictions", group_ids)
            self.assertIn("outcomes", group_ids)

    def test_missing_db(self) -> None:
        out = read_prediction_samples_schema("/nonexistent/pred.db")
        self.assertFalse(out["exists"])


if __name__ == "__main__":
    unittest.main()
