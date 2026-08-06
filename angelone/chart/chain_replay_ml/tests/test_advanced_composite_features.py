"""Tests for advanced weighted-ratio composite helpers (legacy emitters).

These composites are Interaction-pipeline products — not Feature Registry entries.
"""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.advanced_composite_features import enrich_advanced_composite_features
from chain_replay_ml.dataset_builder.feature_ownership import (
    evaluate_registry_admission,
    is_interaction_feature,
)
from chain_replay_ml.dataset_builder.feature_plugins import GROUP_FEATURE_SOURCES


class AdvancedCompositeFeatureTests(unittest.TestCase):
    def test_moneyness_composites(self) -> None:
        raw = {
            "ltp": 100.0,
            "moneyness": 1.05,
            "delta": 0.4,
            "weighted_spot_ema_to_ltp_ratio": 1.2,
        }
        out = enrich_advanced_composite_features(
            raw,
            active_features=frozenset(
                {
                    "weighted_spot_ema_to_ltp_ratio_x_moneyness",
                    "weighted_spot_ema_to_ltp_ratio_x_moneyness_x_delta",
                }
            ),
        )
        self.assertAlmostEqual(out["weighted_spot_ema_to_ltp_ratio_x_moneyness"], 1.26)
        self.assertAlmostEqual(out["weighted_spot_ema_to_ltp_ratio_x_moneyness_x_delta"], 0.504)

    def test_not_in_feature_registry(self) -> None:
        feats = GROUP_FEATURE_SOURCES.get("advanced") or {}
        self.assertNotIn("weighted_spot_ema_to_ltp_ratio_x_moneyness", feats)
        self.assertNotIn("weighted_spot_ema_to_ltp_ratio_x_moneyness_x_delta", feats)
        for name in (
            "weighted_spot_ema_to_ltp_ratio_x_moneyness",
            "weighted_spot_ema_to_ltp_ratio_x_moneyness_x_delta",
        ):
            self.assertTrue(is_interaction_feature(name))
            decision = evaluate_registry_admission(name)
            self.assertFalse(decision["allowed"])
            self.assertEqual(decision["category"], "interaction")


if __name__ == "__main__":
    unittest.main()
