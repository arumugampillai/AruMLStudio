"""Architectural review: greek/moneyness LTP-spot composites are pipeline-only.

These are generic registry-math products (Interaction), not Feature Registry entries.
Master enrich no longer materializes them.
"""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.extended_features import OptionFeatureState, enrich_dataset_features
from chain_replay_ml.dataset_builder.feature_migration import is_pipeline_owned
from chain_replay_ml.dataset_builder.feature_ownership import (
    evaluate_registry_admission,
)
from chain_replay_ml.dataset_builder.feature_plugins import GROUP_FEATURE_SOURCES
from chain_replay_ml.ticks import TickTimeline


_MOVED = (
    "current_to_atm6_flow_delta_ltp_to_spot_ratio",
    "delta_ltp_to_spot_ratio",
    "gamma_ltp_to_spot_ratio",
    "moneyness_delta_ltp_to_spot_ratio",
)


class TestGreekLtpSpotRatios(unittest.TestCase):
    def _enrich(
        self,
        *,
        ltp: float = 120.0,
        spot: float = 25000.0,
        strike: float = 25000.0,
    ) -> dict:
        index_tl = TickTimeline()
        opt_tl = TickTimeline()
        ts = 1000.0
        index_tl.append(ts, int(round(spot * 100)))
        opt_tl.append(ts, int(round(ltp * 100)))
        raw = {"spot": spot, "ltp": ltp, "iv": 0.18}
        return enrich_dataset_features(
            raw,
            ts=ts,
            option_timeline=opt_tl,
            index_timeline=index_tl,
            option_type="CE",
            strike_rupees=strike,
            atm_strike=int(strike),
            strike_step=50,
            expiry_ts=ts + 86400.0 * 3,
            open_ts=ts - 3600.0,
            close_ts=ts + 3600.0 * 6,
            trading_day="2026-05-27",
            expiry_norm="2026-05-29",
            opt_state=OptionFeatureState(),
        )

    def test_not_emitted_by_master_enrich(self) -> None:
        out = self._enrich(ltp=100.0, spot=25000.0)
        for name in (
            "delta_ltp_to_spot_ratio",
            "gamma_ltp_to_spot_ratio",
            "moneyness_delta_ltp_to_spot_ratio",
        ):
            self.assertNotIn(name, out)

    def test_removed_from_feature_registry(self) -> None:
        advanced = GROUP_FEATURE_SOURCES.get("advanced") or {}
        for name in _MOVED:
            self.assertNotIn(name, advanced)
            self.assertTrue(is_pipeline_owned(name))
            decision = evaluate_registry_admission(
                name,
                generic_registry_math=True,
            )
            self.assertFalse(decision["allowed"])

    def test_canonical_inputs_remain(self) -> None:
        """Market-state inputs stay in Registry; composites do not."""
        from chain_replay_ml.dataset_builder.feature_plugins import _REGISTRY_FEATURES

        all_feats = {f for feats in _REGISTRY_FEATURES.values() for f in feats}
        for name in ("delta", "gamma", "abs_delta", "ltp", "spot", "moneyness", "ltp_to_spot_ratio"):
            self.assertIn(name, all_feats)
        for name in _MOVED:
            self.assertNotIn(name, all_feats)


if __name__ == "__main__":
    unittest.main()
