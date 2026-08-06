"""Registry freeze Tier 1–2 + chain totals."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from chain_replay_ml.dataset_builder.chain_maps import chain_features_at, precompute_chain_maps
from chain_replay_ml.dataset_builder.controller_registry import CONTROLLER_FEATURES
from chain_replay_ml.dataset_builder.feature_ownership import (
    OWNERSHIP_BASE,
    OWNERSHIP_COMPUTED_BASE,
    ownership_of,
)
from chain_replay_ml.dataset_builder.feature_plugins import _REGISTRY_FEATURES
from chain_replay_ml.dataset_builder.option_tape_features import emit_option_tape_features
from chain_replay_ml.ticks import TickTimeline, load_tick_timelines


class TestFreezeRegistryAdmission(unittest.TestCase):
    def test_tier1_base(self) -> None:
        for name in (
            "option_oi",
            "option_day_volume",
            "futures_oi",
            "ltq",
            "total_buy_qty",
            "total_sell_qty",
            "option_bid",
            "option_ask",
        ):
            self.assertEqual(ownership_of(name), OWNERSHIP_BASE)
            self.assertTrue(any(name in feats for feats in _REGISTRY_FEATURES.values()))

    def test_tier2_and_chain_computed(self) -> None:
        for name in (
            "atm_iv_ce",
            "atm_iv_pe",
            "total_call_oi",
            "total_put_oi",
            "total_ce_volume",
            "total_pe_volume",
        ):
            self.assertEqual(ownership_of(name), OWNERSHIP_COMPUTED_BASE)
            self.assertIn(name, CONTROLLER_FEATURES["token.chain"])


class TestTapeLoaderAndEmit(unittest.TestCase):
    def test_load_ltq_and_totals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ticks.db"
            conn = sqlite3.connect(str(db))
            conn.execute(
                """
                CREATE TABLE ticks (
                    token TEXT, ts REAL, ltp INTEGER, day_volume INTEGER, oi INTEGER,
                    atp INTEGER, ltq INTEGER, total_buy INTEGER, total_sell INTEGER,
                    sequence_number INTEGER,
                    bid_prices TEXT, ask_prices TEXT,
                    bid_quantities TEXT, ask_quantities TEXT
                )
                """
            )
            conn.execute(
                "INSERT INTO ticks VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    "T1", 1000.0, 14500, 15240, 482100, 14255, 65, 421135, 398220, 1,
                    "[14250]", "[14300]", "[10]", "[12]",
                ),
            )
            conn.commit()
            tl = load_tick_timelines(conn, ["T1"], 900.0, 1100.0)["T1"]
            conn.close()
            self.assertEqual(tl.ltq_at(1000.0), 65)
            self.assertEqual(tl.total_buy_at(1000.0), 421135)
            self.assertEqual(tl.total_sell_at(1000.0), 398220)
            out = emit_option_tape_features({}, ts=1000.0, option_timeline=tl)
            self.assertAlmostEqual(out["option_oi"], 482100.0)
            self.assertAlmostEqual(out["option_day_volume"], 15240.0)
            self.assertAlmostEqual(out["ltq"], 65.0)
            self.assertAlmostEqual(out["total_buy_qty"], 421135.0)
            self.assertAlmostEqual(out["total_sell_qty"], 398220.0)
            self.assertAlmostEqual(out["option_bid"], 142.50)
            self.assertAlmostEqual(out["option_ask"], 143.00)

    def test_chain_totals_maps(self) -> None:
        ts = 1000.0
        index = TickTimeline()
        index.append(ts, 2450000)
        ce = TickTimeline()
        ce.append(ts, 10000, volume=100, oi=1000)
        pe = TickTimeline()
        pe.append(ts, 11000, volume=80, oi=1200)
        strike_mapping = {
            (24500.0, "CE"): ("C", "CE", ce),
            (24500.0, "PE"): ("P", "PE", pe),
        }
        maps = precompute_chain_maps(
            index_tl=index,
            strike_mapping=strike_mapping,
            timestamps=[ts],
            strike_step=50,
        )
        self.assertAlmostEqual(maps.total_call_oi[ts], 1000.0)
        self.assertAlmostEqual(maps.total_put_oi[ts], 1200.0)
        self.assertAlmostEqual(maps.total_ce_volume[ts], 100.0)
        self.assertAlmostEqual(maps.total_pe_volume[ts], 80.0)
        feats = chain_features_at(
            maps,
            ts,
            expiry_ts=ts + 86400,
            strike_mapping=strike_mapping,
            index_tl=index,
            atm_strike=24500,
        )
        self.assertAlmostEqual(feats["total_call_oi"], 1000.0)
        self.assertAlmostEqual(feats["total_pe_volume"], 80.0)


if __name__ == "__main__":
    unittest.main()
