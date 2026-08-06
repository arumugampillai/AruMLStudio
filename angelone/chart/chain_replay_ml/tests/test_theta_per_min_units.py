"""Unit consistency: theta (₹/day) and theta_per_min (₹/min)."""

from __future__ import annotations

import unittest

from chain_replay_ml import bs
from chain_replay_ml.dataset_builder.extended_features import (
    OptionFeatureState,
    enrich_dataset_features,
)
from chain_replay_ml.ticks import TickTimeline


class TestThetaPerMinUnits(unittest.TestCase):
    def _enrich_with_theta(self, theta: float) -> dict:
        index_tl = TickTimeline()
        ts = 1_700_000_000.0
        index_tl.append(ts, int(round(25_000 * 100)))
        raw = {
            "spot": 25_000.0,
            "ltp": 120.0,
            "theta": theta,
            "delta": 0.5,
            "gamma": 0.01,
        }
        return enrich_dataset_features(
            raw,
            ts=ts,
            option_timeline=None,
            index_timeline=index_tl,
            option_type="CE",
            strike_rupees=25_000.0,
            atm_strike=25_000,
            strike_step=50,
            expiry_ts=ts + 86400.0 * 3,
            open_ts=ts - 3600.0,
            close_ts=ts + 3600.0 * 6,
            trading_day="2026-05-27",
            expiry_norm="2026-05-29",
            opt_state=OptionFeatureState(),
        )

    def test_theta_per_min_from_per_day_theta(self) -> None:
        out = self._enrich_with_theta(-1440.0)
        self.assertAlmostEqual(out["theta_per_min"], -1.0)

    def test_theta_per_min_matches_greek_predicted_ltp_one_minute(self) -> None:
        theta = -12.34
        g = {"delta": 0.55, "gamma": 0.001, "theta": theta, "vega": 8.0}
        anchor = 150.0
        at_zero = bs.greek_predicted_ltp(
            anchor, g, spot_change_points=0.0, fwd_min=0.0, iv_change_pct=0.0
        )
        at_one_min = bs.greek_predicted_ltp(
            anchor, g, spot_change_points=0.0, fwd_min=1.0, iv_change_pct=0.0
        )
        expected = theta / 1440.0
        self.assertAlmostEqual(at_one_min - at_zero, expected)

        out = self._enrich_with_theta(theta)
        self.assertAlmostEqual(out["theta_per_min"], expected)

    def test_bs_greeks_theta_is_per_day(self) -> None:
        g = bs.greeks("CE", s=25_000.0, k=25_000.0, r=0.07, t=5 / 365.0, sigma=0.18)
        theta_day = g["theta"]
        out = self._enrich_with_theta(theta_day)
        self.assertAlmostEqual(out["theta_per_min"], theta_day / 1440.0)


if __name__ == "__main__":
    unittest.main()
