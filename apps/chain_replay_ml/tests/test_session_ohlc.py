"""Session OHLC Base features from token_day_meta."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from chain_replay_ml.dataset_builder.controller_registry import FEATURE_REGISTRY_VERSION
from chain_replay_ml.dataset_builder.feature_ownership import OWNERSHIP_BASE, ownership_of
from chain_replay_ml.dataset_builder.feature_plugins import _REGISTRY_FEATURES
from chain_replay_ml.dataset_builder.session_ohlc import (
    emit_session_ohlc_features,
    load_session_ohlc_by_token,
)


_SESSION_NAMES = (
    "spot_open",
    "spot_high",
    "spot_low",
    "spot_prev_close",
    "option_open",
    "option_high",
    "option_low",
    "option_prev_close",
)


class TestSessionOhlcRegistry(unittest.TestCase):
    def test_registered_as_base(self) -> None:
        self.assertEqual(FEATURE_REGISTRY_VERSION, 30)
        for name in _SESSION_NAMES:
            self.assertEqual(ownership_of(name), OWNERSHIP_BASE)
            self.assertTrue(any(name in feats for feats in _REGISTRY_FEATURES.values()))


class TestSessionOhlcLoadAndEmit(unittest.TestCase):
    def test_load_paise_to_rupees_and_emit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ticks.db"
            conn = sqlite3.connect(str(db))
            conn.execute(
                """
                CREATE TABLE token_day_meta (
                    token TEXT, as_of_date TEXT,
                    day_open INTEGER, day_high INTEGER, day_low INTEGER, prev_close INTEGER
                )
                """
            )
            conn.execute(
                "INSERT INTO token_day_meta VALUES (?,?,?,?,?,?)",
                ("IDX", "2026-07-15", 2_450_000, 2_458_000, 2_442_000, 2_448_500),
            )
            conn.execute(
                "INSERT INTO token_day_meta VALUES (?,?,?,?,?,?)",
                ("OPT1", "2026-07-15", 14_200, 16_850, 11_800, 13_525),
            )
            conn.commit()
            loaded = load_session_ohlc_by_token(
                conn, ["IDX", "OPT1", "MISSING"], as_of_date="2026-07-15",
            )
            conn.close()

            self.assertEqual(loaded["IDX"]["open"], 24500.0)
            self.assertEqual(loaded["IDX"]["high"], 24580.0)
            self.assertEqual(loaded["IDX"]["low"], 24420.0)
            self.assertEqual(loaded["IDX"]["prev_close"], 24485.0)
            self.assertEqual(loaded["OPT1"]["open"], 142.0)
            self.assertEqual(loaded["OPT1"]["high"], 168.5)
            self.assertEqual(loaded["OPT1"]["low"], 118.0)
            self.assertEqual(loaded["OPT1"]["prev_close"], 135.25)
            self.assertIsNone(loaded["MISSING"]["open"])

            out = emit_session_ohlc_features(
                {"ltp": 150.0},
                spot_session=loaded["IDX"],
                option_session=loaded["OPT1"],
            )
            self.assertEqual(out["spot_open"], 24500.0)
            self.assertEqual(out["spot_high"], 24580.0)
            self.assertEqual(out["spot_low"], 24420.0)
            self.assertEqual(out["spot_prev_close"], 24485.0)
            self.assertEqual(out["option_open"], 142.0)
            self.assertEqual(out["option_high"], 168.5)
            self.assertEqual(out["option_low"], 118.0)
            self.assertEqual(out["option_prev_close"], 135.25)

    def test_emit_nulls_when_missing(self) -> None:
        out = emit_session_ohlc_features({}, spot_session=None, option_session=None)
        for name in _SESSION_NAMES:
            if name == "option_low":
                self.assertEqual(out[name], 0.0)
            else:
                self.assertIsNone(out[name])

    def test_day_low_zero_emits_option_low_zero(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ticks.db"
            conn = sqlite3.connect(str(db))
            conn.execute(
                """
                CREATE TABLE token_day_meta (
                    token TEXT, as_of_date TEXT,
                    day_open INTEGER, day_high INTEGER, day_low INTEGER, prev_close INTEGER
                )
                """
            )
            conn.execute(
                "INSERT INTO token_day_meta VALUES (?,?,?,?,?,?)",
                ("OPT1", "2026-07-15", 14_200, 16_850, 0, 13_525),
            )
            conn.commit()
            loaded = load_session_ohlc_by_token(
                conn, ["OPT1"], as_of_date="2026-07-15",
            )
            conn.close()
            self.assertEqual(loaded["OPT1"]["low"], 0.0)
            self.assertEqual(loaded["OPT1"]["open"], 142.0)
            out = emit_session_ohlc_features(
                {},
                spot_session=None,
                option_session=loaded["OPT1"],
            )
            self.assertEqual(out["option_low"], 0.0)
            self.assertEqual(out["option_open"], 142.0)


if __name__ == "__main__":
    unittest.main()
