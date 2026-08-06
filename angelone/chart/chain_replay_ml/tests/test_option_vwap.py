"""Option VWAP from exchange ATP (Registry Base); distances via Interaction."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from chain_replay_ml.dataset_builder.extended_features import (
    OptionFeatureState,
    enrich_dataset_features,
)
from chain_replay_ml.dataset_builder.feature_migration import is_pipeline_owned
from chain_replay_ml.dataset_builder.feature_ownership import OWNERSHIP_BASE, ownership_of
from chain_replay_ml.dataset_builder.feature_plugins import _REGISTRY_FEATURES
from chain_replay_ml.dataset_builder.transformations import run_transformation_pipeline
from chain_replay_ml.dataset_builder.transformations.base import TransformContext
from chain_replay_ml.dataset_builder.transformations.lag_ui import classify_feature
from chain_replay_ml.ticks import TickTimeline, load_tick_timelines


class TestOptionVwapRegistry(unittest.TestCase):
    def test_registry_base(self) -> None:
        all_feats = {f for feats in _REGISTRY_FEATURES.values() for f in feats}
        self.assertIn("option_vwap", all_feats)
        self.assertIn("option_vwap", _REGISTRY_FEATURES["price"])
        self.assertEqual(ownership_of("option_vwap"), OWNERSHIP_BASE)
        self.assertEqual(classify_feature("option_vwap"), "Price & Premium")
        self.assertNotIn("ltp_minus_option_vwap", all_feats)
        self.assertTrue(is_pipeline_owned("ltp_minus_option_vwap"))
        self.assertTrue(is_pipeline_owned("ltp_minus_option_vwap_div_option_vwap"))


class TestOptionVwapLoader(unittest.TestCase):
    def test_load_atp_and_asof(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "ticks.db"
            conn = sqlite3.connect(str(db))
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
            conn.execute(
                "INSERT INTO ticks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("T1", 1000.0, 14500, 100, 50, 14255, 1, "[]", "[]", "[]", "[]"),
            )
            conn.execute(
                "INSERT INTO ticks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                ("T1", 1060.0, 14800, 120, 52, 14310, 2, "[]", "[]", "[]", "[]"),
            )
            conn.commit()
            tls = load_tick_timelines(conn, ["T1"], 900.0, 1100.0)
            conn.close()
            tl = tls["T1"]
            self.assertEqual(tl.atp_paise_at(1000.0), 14255)
            self.assertAlmostEqual(tl.atp_rupees_at(1000.0) or 0.0, 142.55)
            self.assertEqual(tl.atp_paise_at(1050.0), 14255)
            self.assertEqual(tl.atp_paise_at(1060.0), 14310)
            self.assertIsNone(tl.atp_paise_at(900.0))

    def test_zero_atp_is_missing(self) -> None:
        tl = TickTimeline()
        tl.append(1000.0, 14500, volume=10, oi=1, atp_paise=0)
        self.assertIsNone(tl.atp_rupees_at(1000.0))


class TestOptionVwapEmitAndDistancePipeline(unittest.TestCase):
    def test_enrich_emits_option_vwap(self) -> None:
        ts = 1_700_000_000.0
        opt = TickTimeline()
        opt.append(ts, 14500, volume=10, oi=1, atp_paise=14255)
        index = TickTimeline()
        index.append(ts, int(round(25_000 * 100)))
        out = enrich_dataset_features(
            {"spot": 25_000.0, "ltp": 145.0, "delta": 0.5, "gamma": 0.01, "theta": -1.0},
            ts=ts,
            option_timeline=opt,
            index_timeline=index,
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
        self.assertAlmostEqual(out.get("option_vwap"), 142.55)

    def test_distance_via_interaction(self) -> None:
        df = pd.DataFrame(
            {
                "trading_day": ["2026-07-24"] * 3,
                "token": ["T"] * 3,
                "ltp": [145.0, 148.0, 140.0],
                "option_vwap": [142.55, 143.10, 143.50],
            }
        )
        cfg = {
            "transformation_pipeline_version": 1,
            "transformations": [
                {
                    "id": "interaction",
                    "enabled": True,
                    "params": {
                        "pairs": [
                            {
                                "left": "ltp",
                                "right": "option_vwap",
                                "op": "subtract",
                                "output": "ltp_minus_option_vwap",
                            },
                            {
                                "left": "ltp_minus_option_vwap",
                                "right": "option_vwap",
                                "op": "divide",
                                "output": "ltp_minus_option_vwap_div_option_vwap",
                                "eps": 1.0e-9,
                            },
                        ]
                    },
                },
            ],
        }
        result = run_transformation_pipeline(
            df, cfg, context=TransformContext(sample_interval_sec=3)
        )
        out = result.frame
        self.assertAlmostEqual(float(out["ltp_minus_option_vwap"].iloc[0]), 145.0 - 142.55)
        self.assertAlmostEqual(
            float(out["ltp_minus_option_vwap_div_option_vwap"].iloc[0]),
            (145.0 - 142.55) / 142.55,
        )


if __name__ == "__main__":
    unittest.main()
