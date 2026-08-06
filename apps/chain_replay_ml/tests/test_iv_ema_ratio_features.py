"""Tests for IV EMA level features (Wave 2).

Controllers emit canonical ``iv_emaN`` levels as Computed Base.
Packaged normalizations / crosses are Interaction / Pipeline Owned only.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from chain_replay_ml.dataset_builder.feature_ownership import (
    evaluate_registry_admission,
    is_interaction_feature,
    ownership_of,
    OWNERSHIP_COMPUTED_BASE,
)
from chain_replay_ml.dataset_builder.feature_plugins import GROUP_FEATURE_SOURCES
from chain_replay_ml.dataset_builder.iv_ema_ratio_features import (
    IV_EMA_LEVEL_FEATURES,
    IV_EMA_PERIODS,
    IV_EMA_RATIO_FEATURES,
    enrich_iv_ema_ratio_features,
)
from chain_replay_ml.dataset_builder.rolling_controllers import (
    TokenControllers,
    update_token_iv_controllers,
    update_token_ltp_controllers,
)
from chain_replay_ml.dataset_builder.schema_registry import load_schema_registry

# Packaged ratios / crosses — not registry, not emitted by enricher.
_PIPELINE_IV_EMA_PACKAGED = tuple(
    name
    for period in IV_EMA_PERIODS
    for name in (
        f"iv_ema{period}_to_ltp_ratio",
        f"iv_ema{period}_to_spot_ratio",
        f"ltp_ema{period}_to_spot_ratio_x_iv_ema{period}",
        f"spot_to_ltp_ratio_x_iv_ema{period}",
        f"spot_to_ltp_ratio_x_iv_ema{period}_x_moneyness",
    )
)


class IvEmaLevelFeatureTests(unittest.TestCase):
    def test_registry_keeps_only_iv_ema_levels(self) -> None:
        feats = GROUP_FEATURE_SOURCES.get("iv_ema_ratio") or {}
        self.assertEqual(len(IV_EMA_LEVEL_FEATURES), 6)
        self.assertEqual(IV_EMA_RATIO_FEATURES, IV_EMA_LEVEL_FEATURES)
        for period in IV_EMA_PERIODS:
            self.assertIn(f"iv_ema{period}", feats)
            self.assertEqual(ownership_of(f"iv_ema{period}"), OWNERSHIP_COMPUTED_BASE)
        for name in _PIPELINE_IV_EMA_PACKAGED:
            self.assertTrue(is_interaction_feature(name))
            self.assertNotIn(name, feats)
            self.assertFalse(evaluate_registry_admission(name)["allowed"])

    def test_schema_sync_iv_ema_levels_only(self) -> None:
        schema = load_schema_registry(use_cache=False)
        self.assertIn("iv_ema_ratio", schema.get("groupOrder") or [])
        group = (schema.get("groups") or {}).get("iv_ema_ratio") or {}
        features = group.get("features") or []
        self.assertEqual(len(features), 6)
        for period in IV_EMA_PERIODS:
            self.assertIn(f"iv_ema{period}", features)
        for name in _PIPELINE_IV_EMA_PACKAGED:
            self.assertNotIn(name, features)
            self.assertNotIn(name, schema.get("columns") or {})

    def test_enrich_emits_levels_after_warmup_not_ratios(self) -> None:
        ctrl = TokenControllers()
        for i in range(320):
            update_token_ltp_controllers(ctrl, 100.0 + i * 0.01, ts=float(i))
            update_token_iv_controllers(ctrl, 0.15 + i * 0.0001, ts=float(i))

        opt_state = SimpleNamespace(controllers=ctrl)
        raw = {
            "ltp": 100.0,
            "spot": 22000.0,
            "moneyness": 1.02,
            "current_iv": 0.18,
        }
        out = enrich_iv_ema_ratio_features(
            raw,
            opt_state=opt_state,
            active_features=frozenset(IV_EMA_LEVEL_FEATURES),
        )
        for period in IV_EMA_PERIODS:
            self.assertIsNotNone(out[f"iv_ema{period}"])
            self.assertNotIn(f"iv_ema{period}_to_ltp_ratio", out)
            self.assertNotIn(f"iv_ema{period}_to_spot_ratio", out)
            self.assertNotIn(f"ltp_ema{period}_to_spot_ratio_x_iv_ema{period}", out)


if __name__ == "__main__":
    unittest.main()
