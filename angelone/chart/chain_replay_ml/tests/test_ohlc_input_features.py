"""Tests for OHLC Aggregation independent input feature defaults."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.transformations.ohlc_aggregation_ui import (
    default_selected_ohlc_features,
)


class OhlcInputFeatureDefaultsTests(unittest.TestCase):
    def test_prefers_spot_and_ltp(self) -> None:
        self.assertEqual(
            default_selected_ohlc_features(
                ["iv", "delta", "gamma", "ltp", "spot", "volume", "theta"]
            ),
            ["spot", "ltp"],
        )

    def test_fallback_first_available(self) -> None:
        self.assertEqual(default_selected_ohlc_features(["iv", "delta"]), ["iv"])

    def test_empty(self) -> None:
        self.assertEqual(default_selected_ohlc_features([]), [])


if __name__ == "__main__":
    unittest.main()
