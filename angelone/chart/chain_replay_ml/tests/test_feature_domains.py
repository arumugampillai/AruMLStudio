"""Feature Registry primary-domain taxonomy (Auto Feature Generation prep)."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.feature_domains import (
    DOMAIN_LABELS,
    DOMAIN_ORDER,
    all_feature_domain_meta,
    primary_domain_of,
    validate_domain_coverage,
)
from chain_replay_ml.dataset_builder.feature_ownership import ownership_of


class TestFeatureDomains(unittest.TestCase):
    def test_coverage_206(self) -> None:
        result = validate_domain_coverage(expected_total=206)
        self.assertTrue(result["ok"], result)
        self.assertEqual(result["registry_count"], 206)
        self.assertEqual(sum(result["domain_counts_by_id"].values()), 206)
        self.assertEqual(result["missing"], [])
        self.assertEqual(result["extra"], [])
        self.assertEqual(result["duplicates"], [])

    def test_no_engineered_or_atm_domains(self) -> None:
        labels = set(DOMAIN_LABELS.values())
        self.assertNotIn("Engineered", labels)
        self.assertNotIn("ATM", labels)
        self.assertNotIn("OI / Volume", labels)

    def test_examples(self) -> None:
        self.assertEqual(primary_domain_of("atm_straddle"), "chain_analytics")
        self.assertEqual(primary_domain_of("atm_iv_ce"), "implied_volatility")
        self.assertEqual(primary_domain_of("atm_pcr"), "open_interest")
        self.assertEqual(primary_domain_of("otm_ce_volume"), "volume_liquidity")
        self.assertEqual(primary_domain_of("spot"), "spot_futures")
        self.assertEqual(primary_domain_of("ltp"), "price_premium")

    def test_meta_flags_present(self) -> None:
        meta = all_feature_domain_meta()
        self.assertEqual(len(meta), 206)
        sample = meta["ltp"]
        for key in (
            "primary_domain",
            "ownership",
            "data_type",
            "can_apply_lag",
            "can_apply_difference",
            "can_apply_return",
            "can_apply_rolling",
            "can_apply_zscore",
            "can_participate_in_interaction",
        ):
            self.assertIn(key, sample)
        self.assertEqual(sample["ownership"], ownership_of("ltp"))
        self.assertTrue(sample["can_apply_lag"])
        self.assertFalse(meta["is_call"]["can_apply_return"])

    def test_domain_order_stable(self) -> None:
        self.assertEqual(len(DOMAIN_ORDER), 11)
        self.assertEqual(DOMAIN_LABELS[DOMAIN_ORDER[0]], "Price & Premium")
        self.assertEqual(DOMAIN_LABELS[DOMAIN_ORDER[-1]], "Metadata")


if __name__ == "__main__":
    unittest.main()
