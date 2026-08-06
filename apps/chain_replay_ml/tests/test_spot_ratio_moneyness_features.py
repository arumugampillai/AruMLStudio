"""Tests for Spot EMA × moneyness (Wave 2).

Moneyness enricher is a no-op. Canonical ``spot_emaN`` levels are Computed Base
via plugins / controllers. Packaged crosses / ÷ltp ratios are Pipeline Owned.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from chain_replay_ml.dataset_builder.feature_ownership import (
    OWNERSHIP_COMPUTED_BASE,
    OWNERSHIP_PIPELINE_OWNED,
    evaluate_registry_admission,
    is_interaction_feature,
    ownership_of,
)
from chain_replay_ml.dataset_builder.feature_plugins import GROUP_FEATURE_SOURCES
from chain_replay_ml.dataset_builder.rolling_controllers import (
    SpotControllers,
    TokenControllers,
    update_token_ltp_controllers,
)
from chain_replay_ml.dataset_builder.schema_registry import load_schema_registry
from chain_replay_ml.dataset_builder.spot_ratio_moneyness_features import (
    SPOT_RATIO_MONEYNESS_FEATURES,
    enrich_spot_ratio_moneyness_features,
)

_SPOT_EMA_PERIODS = (9, 20, 50, 100, 200, 300)


class SpotRatioMoneynessFeatureTests(unittest.TestCase):
    def test_enricher_is_noop_and_features_empty(self) -> None:
        self.assertEqual(SPOT_RATIO_MONEYNESS_FEATURES, ())
        spot = SpotControllers()
        ctrl = TokenControllers()
        for i in range(50):
            spot.update(22000.0 + i * 0.1, ts=float(i))
            update_token_ltp_controllers(ctrl, 100.0 + i * 0.01, ts=float(i))
        raw = {"ltp": 100.0, "spot": 22000.0, "moneyness": 1.02}
        out = enrich_spot_ratio_moneyness_features(
            raw,
            spot_controllers=spot,
            opt_state=SimpleNamespace(controllers=ctrl),
            active_features=frozenset({"spot_ema300_to_ltp_ratio"}),
        )
        self.assertIs(out, raw)
        self.assertNotIn("spot_ema300_to_ltp_ratio", out)

    def test_spot_ema_levels_are_computed_base_in_registry(self) -> None:
        feats = GROUP_FEATURE_SOURCES.get("spot_and_other_ratio") or {}
        for period in _SPOT_EMA_PERIODS:
            name = f"spot_ema{period}"
            self.assertIn(name, feats)
            self.assertEqual(ownership_of(name), OWNERSHIP_COMPUTED_BASE)
        schema = load_schema_registry(use_cache=False)
        group = (schema.get("groups") or {}).get("spot_and_other_ratio") or {}
        features = group.get("features") or []
        for period in _SPOT_EMA_PERIODS:
            self.assertIn(f"spot_ema{period}", features)

    def test_packaged_crosses_and_ratios_are_pipeline_owned(self) -> None:
        packaged = [
            "spot_ema300_to_ltp_ratio",
            "spot_ema50_to_ltp_ratio",
            *[f"spot_ema{p}_to_ltp_ratio_x_moneyness" for p in _SPOT_EMA_PERIODS],
            *[f"ltp_ema{p}_to_spot_ratio_x_moneyness" for p in _SPOT_EMA_PERIODS],
        ]
        feats = GROUP_FEATURE_SOURCES.get("spot_and_other_ratio") or {}
        schema = load_schema_registry(use_cache=False)
        columns = schema.get("columns") or {}
        for name in packaged:
            self.assertEqual(ownership_of(name), OWNERSHIP_PIPELINE_OWNED)
            self.assertTrue(is_interaction_feature(name))
            self.assertNotIn(name, feats)
            self.assertFalse(evaluate_registry_admission(name)["allowed"])
            self.assertNotIn(name, columns)

    def test_hl_ema_levels_stay_in_spot_hl(self) -> None:
        spot_ratio = GROUP_FEATURE_SOURCES.get("spot_and_other_ratio") or {}
        spot_hl = GROUP_FEATURE_SOURCES.get("spot_hl") or {}
        for name in ("spot_high_ema300", "spot_low_ema300"):
            self.assertNotIn(name, spot_ratio)
            self.assertIn(name, spot_hl)
            self.assertEqual(ownership_of(name), OWNERSHIP_COMPUTED_BASE)


if __name__ == "__main__":
    unittest.main()
