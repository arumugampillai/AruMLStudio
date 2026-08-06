"""Wave B: IV−RV spread Computed Base levels."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.controller_registry import (
    CONTROLLER_FEATURES,
    CONTROLLER_REGISTRY,
)
from chain_replay_ml.dataset_builder.feature_ownership import (
    OWNERSHIP_COMPUTED_BASE,
    ownership_of,
)
from chain_replay_ml.dataset_builder.feature_plugins import _REGISTRY_FEATURES
from chain_replay_ml.dataset_builder.iv_rv_spread_features import (
    IV_RV_SPREAD_FEATURES,
    compute_iv_rv_spread,
    enrich_iv_rv_spread_features,
    iv_as_percent,
)


class TestIvRvSpread(unittest.TestCase):
    def test_registry(self) -> None:
        all_feats = {f for feats in _REGISTRY_FEATURES.values() for f in feats}
        for name in IV_RV_SPREAD_FEATURES:
            self.assertIn(name, all_feats)
            self.assertIn(name, _REGISTRY_FEATURES["momentum"])
            self.assertEqual(ownership_of(name), OWNERSHIP_COMPUTED_BASE)
        self.assertIn("composite.iv_rv_spread", CONTROLLER_REGISTRY)
        self.assertEqual(
            CONTROLLER_FEATURES["composite.iv_rv_spread"],
            list(IV_RV_SPREAD_FEATURES),
        )

    def test_iv_as_percent(self) -> None:
        self.assertAlmostEqual(iv_as_percent(0.18), 18.0)
        self.assertAlmostEqual(iv_as_percent(18.0), 18.0)

    def test_compute(self) -> None:
        self.assertAlmostEqual(compute_iv_rv_spread(0.20, 2.5), 17.5)
        self.assertIsNone(compute_iv_rv_spread(None, 1.0))
        self.assertIsNone(compute_iv_rv_spread(0.2, None))

    def test_enrich_from_raw_and_cache(self) -> None:
        raw = {"iv": 0.15}
        cache = {1000.0: {"spot_rv_5m": 1.0, "spot_rv_10m": 0.8}}
        out = enrich_iv_rv_spread_features(
            raw, ts=1000.0, spot_rv_cache=cache, active_features=None
        )
        self.assertAlmostEqual(out["iv_rv_spread_5m"], 14.0)
        self.assertAlmostEqual(out["iv_rv_spread_10m"], 14.2)


if __name__ == "__main__":
    unittest.main()
