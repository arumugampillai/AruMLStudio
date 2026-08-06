"""Hit Confidence classifier helpers + binary eval."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np

from chain_replay_ml.model_lab.confidence import (
    confidence_band,
    compute_calibration_bins,
    confidence_sidecar_path,
    link_trained_confidence_model,
    read_confidence_link,
    write_confidence_link,
)
from chain_replay_ml.training.evaluator import evaluate_classification, evaluate_predictions


class TestConfidenceHelpers(unittest.TestCase):
    def test_confidence_band(self) -> None:
        self.assertEqual(confidence_band(0.95), "Very High")
        self.assertEqual(confidence_band(0.80), "High")
        self.assertEqual(confidence_band(0.65), "Medium")
        self.assertEqual(confidence_band(0.55), "Low")
        self.assertEqual(confidence_band(0.2), "Very Low")

    def test_calibration_bins(self) -> None:
        y_true = [0, 0, 1, 1, 1, 1]
        y_prob = [0.2, 0.4, 0.55, 0.7, 0.8, 0.95]
        rows = compute_calibration_bins(y_true, y_prob)
        self.assertEqual(len(rows), 5)
        self.assertEqual(rows[0]["band"], "<50%")
        self.assertEqual(rows[0]["rows"], 2)

    def test_sidecar_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            lab = os.path.join(tmp, "lab_v1.db")
            open(lab, "wb").close()
            path = write_confidence_link(lab, {"status": "pending_train", "dataset_name": "hit_x"})
            self.assertEqual(path, confidence_sidecar_path(lab))
            doc = read_confidence_link(lab)
            assert doc is not None
            self.assertEqual(doc["dataset_name"], "hit_x")
            link_trained_confidence_model(lab, model_name="hit_confidence_demo")
            ready = read_confidence_link(lab)
            assert ready is not None
            self.assertEqual(ready["status"], "ready")
            self.assertEqual(ready["model_name"], "hit_confidence_demo")
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
            self.assertEqual(raw["status"], "ready")


class TestEvaluateClassification(unittest.TestCase):
    def test_perfect_proba(self) -> None:
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0.1, 0.2, 0.9, 0.85])
        m = evaluate_classification(y_true, y_pred)
        self.assertEqual(m["accuracy_pct"], 100.0)
        self.assertEqual(m["f1_pct"], 100.0)
        self.assertEqual(m["specificity_pct"], 100.0)
        self.assertGreaterEqual(float(m["roc_auc"]), 0.99)
        self.assertIsNotNone(m.get("brier_score"))
        self.assertLess(float(m["brier_score"]), 0.1)
        self.assertEqual(m["confusion"]["tp"], 2)
        self.assertEqual(m["confusion"]["tn"], 2)

    def test_dispatch(self) -> None:
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0.2, 0.8, 0.3, 0.7])
        m = evaluate_predictions(y_true, y_pred, prediction_type="binary")
        self.assertIn("accuracy_pct", m)
        self.assertNotIn("rmse", m)


if __name__ == "__main__":
    unittest.main()
