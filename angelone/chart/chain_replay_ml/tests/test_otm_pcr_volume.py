"""OTM volume PCR Registry wave (v29)."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.controller_registry import (
    CONTROLLER_FEATURES,
    FEATURE_REGISTRY_VERSION,
)
from chain_replay_ml.dataset_builder.feature_ownership import (
    OWNERSHIP_COMPUTED_BASE,
    ownership_of,
)
from chain_replay_ml.dataset_builder.feature_plugins import _REGISTRY_FEATURES
from chain_replay_ml.dataset_builder.transformations.lag_ui import classify_feature


class TestOtmPcrVolume(unittest.TestCase):
    def test_registry(self) -> None:
        self.assertEqual(FEATURE_REGISTRY_VERSION, 30)
        names = ("otm_ce_volume", "otm_pe_volume", "otm_pcr_volume")
        all_feats = {f for feats in _REGISTRY_FEATURES.values() for f in feats}
        for name in names:
            self.assertIn(name, all_feats)
            self.assertIn(name, CONTROLLER_FEATURES["token.chain"])
            self.assertEqual(ownership_of(name), OWNERSHIP_COMPUTED_BASE)
        self.assertIn("otm_ce_volume", _REGISTRY_FEATURES["volume"])
        self.assertIn("otm_pe_volume", _REGISTRY_FEATURES["volume"])
        self.assertIn("otm_pcr_volume", _REGISTRY_FEATURES["chain"])
        self.assertEqual(classify_feature("otm_pcr_volume"), "Volume & Liquidity")
        self.assertEqual(classify_feature("otm_ce_volume"), "Volume & Liquidity")

    def test_otm_filter_and_pcr(self) -> None:
        """CE OTM = K > spot; PE OTM = K < spot; PCR = pe/ce."""
        spot = 24500.0
        # Volumes: CE 100@24400 (ITM), 200@24500 (ATM), 300@24600 (OTM)
        #          PE 400@24400 (OTM), 50@24500 (ATM), 10@24600 (ITM)
        strike_vols_ce = {24400.0: 100, 24500.0: 200, 24600.0: 300}
        strike_vols_pe = {24400.0: 400, 24500.0: 50, 24600.0: 10}

        otm_ce = 0
        otm_pe = 0
        for k, v in strike_vols_ce.items():
            if k > spot:
                otm_ce += v
        for k, v in strike_vols_pe.items():
            if k < spot:
                otm_pe += v
        self.assertEqual(otm_ce, 300)
        self.assertEqual(otm_pe, 400)
        self.assertAlmostEqual(otm_pe / otm_ce, 400 / 300)

        # Mirror store path used in build_chain_maps.
        maps: dict[str, float] = {
            "otm_ce_volume": float(otm_ce),
            "otm_pe_volume": float(otm_pe),
            "otm_pcr_volume": float(otm_pe) / float(otm_ce),
        }
        self.assertAlmostEqual(maps["otm_pcr_volume"], 4.0 / 3.0)

    def test_pcr_null_when_no_otm_ce(self) -> None:
        otm_ce_volume = 0
        otm_pe_volume = 500
        stored = None
        if otm_ce_volume > 0:
            stored = float(otm_pe_volume) / float(otm_ce_volume)
        self.assertIsNone(stored)


if __name__ == "__main__":
    unittest.main()
