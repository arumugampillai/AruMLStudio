"""Tests for trader-facing classification composite recipe."""

from __future__ import annotations

import unittest

from chain_replay_ml.training.objective_scoring import (
    CLASSIFICATION_COMPOSITE_WEIGHTS,
    classification_composite_breakdown,
    classification_composite_score,
)


class TestClassificationComposite(unittest.TestCase):
    def test_weights_match_overview_recipe(self) -> None:
        self.assertAlmostEqual(CLASSIFICATION_COMPOSITE_WEIGHTS["precision"], 0.40)
        self.assertAlmostEqual(CLASSIFICATION_COMPOSITE_WEIGHTS["f1"], 0.30)
        self.assertAlmostEqual(CLASSIFICATION_COMPOSITE_WEIGHTS["roc_auc"], 0.20)
        self.assertAlmostEqual(CLASSIFICATION_COMPOSITE_WEIGHTS["recall"], 0.10)

    def test_score_and_breakdown(self) -> None:
        metrics = {
            "precision_pct": 50.0,
            "f1_pct": 50.0,
            "recall_pct": 50.0,
            "roc_auc": 0.80,
        }
        score = classification_composite_score(metrics)
        # 0.4*0.5 + 0.3*0.5 + 0.2*0.8 + 0.1*0.5 = 0.2+0.15+0.16+0.05 = 0.56
        self.assertAlmostEqual(score, 0.56, places=6)
        rows = classification_composite_breakdown(metrics)
        self.assertEqual([r["label"] for r in rows], ["Precision", "F1", "ROC-AUC", "Recall"])
        self.assertEqual([r["weight_pct"] for r in rows], [40.0, 30.0, 20.0, 10.0])


if __name__ == "__main__":
    unittest.main()
