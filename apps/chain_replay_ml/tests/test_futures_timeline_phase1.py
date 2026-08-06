"""Phase 1: Futures Timeline in DayContext + futures_ltp / futures_vwap Base."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from chain_replay_ml.dataset_builder.day_context import DayContext, SourceSpec
from chain_replay_ml.dataset_builder.feature_migration import is_pipeline_owned
from chain_replay_ml.dataset_builder.feature_ownership import OWNERSHIP_BASE, ownership_of
from chain_replay_ml.dataset_builder.feature_plugins import _REGISTRY_FEATURES
from chain_replay_ml.dataset_builder.futures_context import (
    emit_futures_timeline_features,
    resolve_front_month_futures,
)
from chain_replay_ml.ticks import TickTimeline


class TestFuturesRegistry(unittest.TestCase):
    def test_registry_base(self) -> None:
        all_feats = {f for feats in _REGISTRY_FEATURES.values() for f in feats}
        for name in (
            "futures_ltp",
            "futures_vwap",
            "futures_day_volume",
            "futures_bid",
            "futures_ask",
            "futures_spread",
        ):
            self.assertIn(name, all_feats)
            self.assertIn(name, _REGISTRY_FEATURES["price"])
            self.assertEqual(ownership_of(name), OWNERSHIP_BASE)
        self.assertNotIn("spot_minus_futures_ltp", all_feats)
        self.assertTrue(is_pipeline_owned("spot_minus_futures_ltp"))
        self.assertTrue(is_pipeline_owned("futures_ltp_minus_futures_vwap"))


class TestResolveFrontMonthFutures(unittest.TestCase):
    def test_prefers_current_month_and_soft_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ticks.db"
            conn = sqlite3.connect(str(db))
            conn.execute(
                """
                CREATE TABLE token_day_meta (
                    token TEXT, as_of_date TEXT, name TEXT,
                    trading_symbol TEXT, instrument_type TEXT, expiry_date TEXT,
                    option_type TEXT, strike_price INTEGER
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE ticks (
                    token TEXT, ts REAL, ltp INTEGER, day_volume INTEGER, oi INTEGER,
                    atp INTEGER, sequence_number INTEGER,
                    bid_prices TEXT, ask_prices TEXT,
                    bid_quantities TEXT, ask_quantities TEXT
                )
                """
            )
            # Two futures: Jul (current) and Aug — prefer Jul on 2026-07-15.
            conn.execute(
                "INSERT INTO token_day_meta VALUES (?,?,?,?,?,?,?,?)",
                ("FUT_JUL", "2026-07-15", "NIFTY", "NIFTY26JULFUT", "FUTIDX",
                 "2026-07-31", None, None),
            )
            conn.execute(
                "INSERT INTO token_day_meta VALUES (?,?,?,?,?,?,?,?)",
                ("FUT_AUG", "2026-07-15", "NIFTY", "NIFTY26AUGFUT", "FUTIDX",
                 "2026-08-28", None, None),
            )
            conn.execute(
                "INSERT INTO ticks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("FUT_JUL", 1000.0, 2451250, 100, 0, 2450820, 1, "[]", "[]", "[]", "[]"),
            )
            conn.commit()
            picked = resolve_front_month_futures(
                conn,
                underlying="NIFTY",
                trading_day="2026-07-15",
                normalize_index_name=lambda x: str(x).upper(),
                open_ts=900.0,
                close_ts=1100.0,
            )
            self.assertIsNotNone(picked)
            assert picked is not None
            self.assertEqual(picked["token"], "FUT_JUL")
            conn.close()

    def test_soft_fail_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ticks.db"
            conn = sqlite3.connect(str(db))
            conn.execute(
                """
                CREATE TABLE token_day_meta (
                    token TEXT, as_of_date TEXT, name TEXT,
                    trading_symbol TEXT, instrument_type TEXT, expiry_date TEXT
                )
                """
            )
            conn.execute("CREATE TABLE ticks (token TEXT, ts REAL, ltp INTEGER)")
            conn.commit()
            picked = resolve_front_month_futures(
                conn,
                underlying="NIFTY",
                trading_day="2026-07-15",
                normalize_index_name=lambda x: str(x).upper(),
            )
            self.assertIsNone(picked)
            conn.close()


class TestEmitFuturesFeatures(unittest.TestCase):
    def test_emit_from_timeline(self) -> None:
        tl = TickTimeline()
        tl.append(
            1000.0,
            2451250,
            volume=185420,
            atp_paise=2450820,
            spread_paise=100,
            bid_prices_paise=[2451100],
            ask_prices_paise=[2451200],
        )
        out = emit_futures_timeline_features({}, ts=1000.0, futures_tl=tl)
        self.assertAlmostEqual(out["futures_ltp"], 24512.50)
        self.assertAlmostEqual(out["futures_vwap"], 24508.20)
        self.assertAlmostEqual(out["futures_day_volume"], 185420.0)
        self.assertAlmostEqual(out["futures_bid"], 24511.0)
        self.assertAlmostEqual(out["futures_ask"], 24512.0)
        self.assertAlmostEqual(out["futures_spread"], 1.0)

    def test_emit_null_when_missing(self) -> None:
        out = emit_futures_timeline_features({"spot": 1.0}, ts=1000.0, futures_tl=None)
        for key in (
            "futures_ltp",
            "futures_vwap",
            "futures_day_volume",
            "futures_oi",
            "futures_bid",
            "futures_ask",
            "futures_spread",
        ):
            self.assertIsNone(out[key])


class TestDayContextAliases(unittest.TestCase):
    def test_spot_tl_alias(self) -> None:
        index = TickTimeline()
        index.append(1.0, 100)
        ctx = DayContext(
            source=SourceSpec("s", "2026-07-15", "NIFTY", "2026-07-31"),
            db_path="x",
            expiry_norm="2026-07-31",
            open_ts=0.0,
            close_ts=1.0,
            expiry_ts=2.0,
            index_tl=index,
            strike_mapping={},
            futures_tl=None,
        )
        self.assertIs(ctx.spot_tl, ctx.index_tl)
        self.assertIsNone(ctx.futures_tl)
        self.assertIsNone(ctx.history_tl)


if __name__ == "__main__":
    unittest.main()
