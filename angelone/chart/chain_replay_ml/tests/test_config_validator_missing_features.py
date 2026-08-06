"""Unit tests for missing-feature messaging in training validation."""

from __future__ import annotations

import unittest

from chain_replay_ml.training.config_validator import _format_missing_features_detail


class TestMissingFeaturesDetail(unittest.TestCase):
    def test_lists_each_feature_when_few(self) -> None:
        text = _format_missing_features_detail(
            ["iv_rank", "delta_ce"],
            dataset="MS_256f_3s_0604",
        )
        self.assertIn("2 feature(s) missing in dataset MS_256f_3s_0604", text)
        self.assertIn("• iv_rank", text)
        self.assertIn("• delta_ce", text)

    def test_collapses_when_many(self) -> None:
        names = [f"feat_{i}" for i in range(35)]
        text = _format_missing_features_detail(names, dataset="DS")
        self.assertIn("35 feature(s) missing in dataset DS", text)
        self.assertIn("feat_0", text)
        self.assertIn("+5 more", text)
        self.assertNotIn("• feat_0", text)


if __name__ == "__main__":
    unittest.main()
