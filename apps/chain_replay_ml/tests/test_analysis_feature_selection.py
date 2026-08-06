"""Feature Selection Strategy builders and filters."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.analysis_feature_selection import (
    POLICY_TOP_2,
    STRATEGY_CORR_PERM,
    STRATEGY_HCA,
    build_selection_config,
    compare_strategy_rows,
    correlation_filter,
    format_selection_summary,
    normalize_policy,
    normalize_strategy,
    permutation_filter,
    required_modules_for_strategy,
)


class FeatureSelectionUnitTests(unittest.TestCase):
    def test_normalize_strategy_aliases(self) -> None:
        self.assertEqual(normalize_strategy("HCA"), STRATEGY_HCA)
        self.assertEqual(normalize_strategy("corr_perm"), STRATEGY_CORR_PERM)

    def test_normalize_policy_top_n(self) -> None:
        self.assertEqual(normalize_policy("top_2"), (POLICY_TOP_2, 2))
        p, n = normalize_policy("top_n", top_n=5)
        self.assertEqual(n, 5)
        self.assertEqual(p, "top_n")

    def test_correlation_filter_keeps_higher_score(self) -> None:
        feats = ["a", "b", "c"]
        pairs = [("a", "b", 0.99)]
        scores = {"a": 80.0, "b": 50.0, "c": 10.0}
        kept = correlation_filter(feats, pairs, scores, threshold=0.95)
        self.assertEqual(kept, ["a", "c"])

    def test_permutation_filter_threshold(self) -> None:
        feats = ["a", "b", "c"]
        perm = {"a": 0.01, "b": 0.0005, "c": 0.002}
        kept = permutation_filter(feats, perm, threshold=0.001)
        self.assertEqual(kept, ["a", "c"])

    def test_required_modules_differ_by_strategy(self) -> None:
        self.assertIn("hca", required_modules_for_strategy(STRATEGY_HCA))
        self.assertNotIn("hca", required_modules_for_strategy(STRATEGY_CORR_PERM))
        self.assertEqual(
            required_modules_for_strategy("perm_only"), ("permutation",)
        )

    def test_format_selection_summary_hca(self) -> None:
        cfg = build_selection_config(
            STRATEGY_HCA, representative_policy="top_2"
        )
        cfg["n_families"] = 57
        text = format_selection_summary(cfg, n_features=108)
        self.assertIn("Feature Selection", text)
        self.assertIn("HCA", text)
        self.assertIn("Top 2", text)
        self.assertIn("108", text)

    def test_format_selection_summary_corr_perm(self) -> None:
        cfg = build_selection_config(
            STRATEGY_CORR_PERM,
            correlation_threshold=0.95,
            permutation_threshold=0.001,
        )
        text = format_selection_summary(cfg, n_features=128)
        self.assertIn("Correlation Threshold", text)
        self.assertIn("0.950", text)
        self.assertIn("128", text)

    def test_compare_strategy_rows(self) -> None:
        rows = compare_strategy_rows(
            [
                {
                    "feature_selection": build_selection_config(
                        STRATEGY_HCA, representative_policy="top_1"
                    ),
                    "n_features": 57,
                    "holdout_score": 0.9,
                },
                {
                    "feature_selection": build_selection_config(
                        STRATEGY_CORR_PERM
                    ),
                    "count": 128,
                    "holdout_r2": 0.88,
                },
            ]
        )
        self.assertEqual(rows[0]["strategy_label"], "HCA Top 1")
        self.assertEqual(rows[0]["n_features"], 57)
        self.assertEqual(rows[1]["strategy_label"], "Corr+Perm")
        self.assertEqual(rows[1]["n_features"], 128)


if __name__ == "__main__":
    unittest.main()
