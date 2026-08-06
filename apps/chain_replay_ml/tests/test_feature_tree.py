"""Tests for Model Builder feature selection tree."""

from __future__ import annotations

import unittest

from master_dataset_tk.model_builder.feature_tree import feature_display_name


class FeatureTreeTests(unittest.TestCase):
    def test_feature_display_name_prefers_schema_label(self) -> None:
        cols = {"ltp_to_spot_ratio": {"display_name": "LTP / Spot"}}
        self.assertEqual(feature_display_name(cols, "ltp_to_spot_ratio"), "LTP / Spot")
        self.assertEqual(feature_display_name(cols, "unknown_feat"), "unknown_feat")


if __name__ == "__main__":
    unittest.main()
