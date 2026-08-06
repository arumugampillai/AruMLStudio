"""Tests for current-to-ATM6 flow feature."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.current_to_atm6_flow import (
    compute_current_to_atm6_flow_delta_ltp_to_spot_ratio,
    strikes_toward_atm,
)
from chain_replay_ml.ticks import TickTimeline


def _tl(
    *,
    ltp: float = 100.0,
    volume_series: list[tuple[float, int]] | None = None,
    oi_series: list[tuple[float, int]] | None = None,
) -> TickTimeline:
    tl = TickTimeline()
    ts = 1000.0
    tl.append(ts, int(round(ltp * 100)))
    if volume_series:
        for t, vol in volume_series:
            tl.append(t, int(round(ltp * 100)), volume=vol)
    if oi_series:
        for t, oi in oi_series:
            tl.append(t, int(round(ltp * 100)), oi=oi)
    return tl


class TestStrikesTowardAtm(unittest.TestCase):
    def test_ce_walks_lower(self) -> None:
        strikes = strikes_toward_atm(25000.0, step=50, option_type="CE")
        self.assertEqual(strikes, [25000.0, 24950.0, 24900.0, 24850.0, 24800.0, 24750.0, 24700.0])

    def test_pe_walks_higher(self) -> None:
        strikes = strikes_toward_atm(25000.0, step=50, option_type="PE")
        self.assertEqual(strikes, [25000.0, 25050.0, 25100.0, 25150.0, 25200.0, 25250.0, 25300.0])


class TestCurrentToAtm6FlowFeature(unittest.TestCase):
    def test_computes_flow_ratio(self) -> None:
        step = 50
        ts = 2000.0
        past_ts = ts - 60.0
        strike_mapping = {}
        for i, strike in enumerate(strikes_toward_atm(25000.0, step=step, option_type="CE")):
            vol_now = 200 + i * 10
            vol_past = 100
            oi_now = 1000 + i * 50
            oi_past = 800
            tl = TickTimeline()
            tl.append(past_ts, 10000, volume=vol_past, oi=oi_past)
            tl.append(ts, 10000, volume=vol_now, oi=oi_now)
            strike_mapping[(strike, "CE")] = ("tok", "SYM", tl)

        result = compute_current_to_atm6_flow_delta_ltp_to_spot_ratio(
            strike_mapping=strike_mapping,
            ts=ts,
            current_strike=25000.0,
            step=step,
            option_type="CE",
            delta=0.35,
            ltp=120.0,
            spot=24800.0,
        )
        self.assertIsNotNone(result)
        # vol pct per strike: (200+i*10 - 100)/100 * 100 = 100 + i*10
        vol_avgs = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0]
        oi_avgs = [(1000 + i * 50 - 800) / 800 * 100 for i in range(7)]
        flow = (sum(vol_avgs) / 7 + sum(oi_avgs) / 7) / 2.0
        expected = flow * 0.35 * 120.0 / 24800.0
        self.assertAlmostEqual(result, expected, places=6)

    def test_uses_abs_delta_for_puts(self) -> None:
        step = 50
        ts = 2000.0
        past_ts = ts - 60.0
        strike_mapping = {}
        for strike in strikes_toward_atm(25000.0, step=step, option_type="PE"):
            tl = TickTimeline()
            tl.append(past_ts, 10000, volume=100, oi=800)
            tl.append(ts, 10000, volume=200, oi=1000)
            strike_mapping[(strike, "PE")] = ("tok", "SYM", tl)

        pos = compute_current_to_atm6_flow_delta_ltp_to_spot_ratio(
            strike_mapping=strike_mapping,
            ts=ts,
            current_strike=25000.0,
            step=step,
            option_type="PE",
            delta=0.35,
            ltp=120.0,
            spot=24800.0,
        )
        neg = compute_current_to_atm6_flow_delta_ltp_to_spot_ratio(
            strike_mapping=strike_mapping,
            ts=ts,
            current_strike=25000.0,
            step=step,
            option_type="PE",
            delta=-0.35,
            ltp=120.0,
            spot=24800.0,
        )
        self.assertIsNotNone(pos)
        self.assertAlmostEqual(pos, neg, places=9)

    def test_missing_strike_returns_none(self) -> None:
        result = compute_current_to_atm6_flow_delta_ltp_to_spot_ratio(
            strike_mapping={},
            ts=2000.0,
            current_strike=25000.0,
            step=50,
            option_type="CE",
            delta=0.35,
            ltp=120.0,
            spot=24800.0,
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
