"""Tests for IV z-score feature helpers.

IV z-score windows are Pipeline Owned. Weighted composites with ``_x_`` are
Interaction-pipeline products — not Feature Registry entries.
"""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.feature_ownership import (
    evaluate_registry_admission,
    is_interaction_feature,
)
from chain_replay_ml.dataset_builder.feature_plugins import GROUP_FEATURE_SOURCES
from chain_replay_ml.dataset_builder.iv_zscore_features import (
    enrich_iv_zscore_features,
    weighted_iv_zscore,
)


class IvZscoreFeatureTests(unittest.TestCase):
    def test_weighted_iv_zscore_formula(self) -> None:
        self.assertAlmostEqual(weighted_iv_zscore(3.0, 2.0, 1.0), (9.0 + 4.0 + 1.0) / 6.0)
        self.assertIsNone(weighted_iv_zscore(None, 2.0, 1.0))

    def test_enrich_composites(self) -> None:
        raw = {
            "ltp": 100.0,
            "delta": 0.5,
            "weighted_spot_ema_to_ltp_ratio": 1.2,
            "iv_zscore_1m": 3.0,
            "iv_zscore_5m": 2.0,
            "iv_zscore_15m": 1.0,
        }
        out = enrich_iv_zscore_features(
            raw,
            active_features=frozenset(
                {
                    "weighted_iv_zscore_x_weighted_spot_ema_to_ltp_ratio",
                    "weighted_spot_ema_to_ltp_ratio_x_delta",
                    "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_1m_x_delta",
                }
            ),
        )
        w_iv = (3.0 * 3 + 2.0 * 2 + 1.0) / 6.0
        self.assertAlmostEqual(
            out["weighted_iv_zscore_x_weighted_spot_ema_to_ltp_ratio"],
            w_iv * 1.2,
        )
        self.assertAlmostEqual(out["weighted_spot_ema_to_ltp_ratio_x_delta"], 0.6)
        self.assertAlmostEqual(out["weighted_spot_ema_to_ltp_ratio_x_iv_zscore_1m_x_delta"], 1.8)

    def test_iv_zscore_group_excludes_interactions(self) -> None:
        feats = GROUP_FEATURE_SOURCES.get("iv_zscore") or {}
        self.assertEqual(feats, {})
        for name in (
            "weighted_iv_zscore_x_weighted_spot_ema_to_ltp_ratio",
            "weighted_spot_ema_to_ltp_ratio_x_iv_zscore_15m_x_delta",
        ):
            self.assertTrue(is_interaction_feature(name))
            decision = evaluate_registry_admission(name)
            self.assertFalse(decision["allowed"])
            self.assertEqual(decision["category"], "interaction")


if __name__ == "__main__":
    unittest.main()
