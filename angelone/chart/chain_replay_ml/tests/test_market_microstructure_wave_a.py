"""Wave A: Market Microstructure Controller + tick book loader."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from chain_replay_ml.dataset_builder.controller_registry import (
    CONTROLLER_FEATURES,
    CONTROLLER_REGISTRY,
)
from chain_replay_ml.dataset_builder.feature_migration import is_pipeline_owned
from chain_replay_ml.dataset_builder.feature_ownership import (
    OWNERSHIP_COMPUTED_BASE,
    ownership_of,
)
from chain_replay_ml.dataset_builder.feature_plugins import _REGISTRY_FEATURES
from chain_replay_ml.dataset_builder.market_microstructure import (
    MARKET_MICROSTRUCTURE_FEATURES,
    SPREAD_NORMALIZED_LTP_STEP,
    compute_microstructure_levels,
    enrich_market_microstructure_features,
)
from chain_replay_ml.dataset_builder.transformations import run_transformation_pipeline
from chain_replay_ml.dataset_builder.transformations.base import TransformContext
from chain_replay_ml.ticks import BookSnapshot, TickTimeline, load_tick_timelines


class TestMicrostructureLevels(unittest.TestCase):
    def test_formulas(self) -> None:
        book = BookSnapshot(
            bid_prices_paise=(10000, 9950, 9900, 9850, 9800),
            ask_prices_paise=(10050, 10100, 10150, 10200, 10250),
            bid_quantities=(100, 80, 60, 40, 20),
            ask_quantities=(50, 40, 30, 20, 10),
            spread_paise=50,
        )
        levels = compute_microstructure_levels(book)
        self.assertAlmostEqual(levels["mid_price"], 100.25)
        # micro = (100.50*100 + 100.00*50) / 150 = 100.333...
        self.assertAlmostEqual(levels["microprice"], (100.50 * 100 + 100.00 * 50) / 150)
        # bias = (micro - mid) / spread; spread = 0.50
        self.assertAlmostEqual(
            levels["microprice_bias"],
            (levels["microprice"] - levels["mid_price"]) / 0.50,
        )
        self.assertAlmostEqual(levels["book_imbalance_l1"], (100 - 50) / 150)
        self.assertAlmostEqual(levels["bid_depth_l1_5"], 300.0)
        self.assertAlmostEqual(levels["ask_depth_l1_5"], 150.0)
        self.assertAlmostEqual(levels["book_imbalance_l1_5"], (300 - 150) / 450)
        # Bid qty 100,80,60,40,20 vs index 0..4 → slope = -20
        self.assertAlmostEqual(levels["book_depth_slope_bid"], -20.0)
        # Ask qty 50,40,30,20,10 → slope = -10
        self.assertAlmostEqual(levels["book_depth_slope_ask"], -10.0)

    def test_enrich_from_timeline(self) -> None:
        tl = TickTimeline()
        tl.append(
            1000.0,
            10025,
            spread_paise=50,
            bid_prices_paise=(10000, 0, 0, 0, 0),
            ask_prices_paise=(10050, 0, 0, 0, 0),
            bid_quantities=(200, 0, 0, 0, 0),
            ask_quantities=(100, 0, 0, 0, 0),
        )
        raw = enrich_market_microstructure_features({}, ts=1000.0, option_timeline=tl)
        self.assertAlmostEqual(raw["mid_price"], 100.25)
        self.assertAlmostEqual(raw["book_imbalance_l1"], (200 - 100) / 300)

    def test_registry_and_controller(self) -> None:
        all_feats = {f for feats in _REGISTRY_FEATURES.values() for f in feats}
        for name in MARKET_MICROSTRUCTURE_FEATURES:
            self.assertIn(name, all_feats)
            self.assertIn(name, _REGISTRY_FEATURES["market_microstructure"])
            self.assertEqual(ownership_of(name), OWNERSHIP_COMPUTED_BASE)
        self.assertIn("token.book", CONTROLLER_REGISTRY)
        self.assertEqual(
            CONTROLLER_FEATURES["token.book"],
            list(MARKET_MICROSTRUCTURE_FEATURES),
        )
        self.assertNotIn("bid_ask_spread", CONTROLLER_FEATURES["token.book"])
        self.assertTrue(is_pipeline_owned(SPREAD_NORMALIZED_LTP_STEP))


class TestTickBookLoader(unittest.TestCase):
    def test_load_parses_l1_l5_qty(self) -> None:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            db = Path(tmp) / "ticks.db"
            conn = sqlite3.connect(str(db))
            conn.execute(
                """
                CREATE TABLE ticks (
                    token TEXT, ts REAL, ltp INTEGER, day_volume INTEGER, oi INTEGER,
                    sequence_number INTEGER,
                    bid_prices TEXT, ask_prices TEXT,
                    bid_quantities TEXT, ask_quantities TEXT
                )
                """
            )
            conn.execute(
                """
                INSERT INTO ticks VALUES (?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    "T1",
                    1000.0,
                    10025,
                    10,
                    100,
                    1,
                    json.dumps([10000, 9950, 9900, 9850, 9800]),
                    json.dumps([10050, 10100, 10150, 10200, 10250]),
                    json.dumps([100, 80, 60, 40, 20]),
                    json.dumps([50, 40, 30, 20, 10]),
                ),
            )
            conn.commit()
            tls = load_tick_timelines(conn, ["T1"], 900.0, 1100.0)
            conn.close()
            book = tls["T1"].book_at(1000.0)
            self.assertIsNotNone(book)
            assert book is not None
            self.assertEqual(book.bid_quantities[0], 100)
            self.assertEqual(book.ask_prices_paise[0], 10050)
            self.assertEqual(book.spread_paise, 50)
            levels = compute_microstructure_levels(book)
            self.assertAlmostEqual(levels["bid_depth_l1_5"], 300.0)


class TestSpreadNormalizedStepPipeline(unittest.TestCase):
    def test_difference_then_divide_by_spread(self) -> None:
        df = pd.DataFrame(
            {
                "trading_day": ["2026-07-24"] * 4,
                "token": ["T"] * 4,
                "ltp": [100.0, 100.5, 100.2, 101.0],
                "bid_ask_spread": [0.5, 0.5, 0.5, 0.5],
            }
        )
        cfg = {
            "transformation_pipeline_version": 1,
            "transformations": [
                {
                    "id": "difference",
                    "enabled": True,
                    "params": {
                        "features": ["ltp"],
                        "horizons_sec": [3],
                    },
                },
                {
                    "id": "interaction",
                    "enabled": True,
                    "params": {
                        "pairs": [
                            {
                                "left": "ltp_diff_3s",
                                "right": "bid_ask_spread",
                                "op": "divide",
                                "output": SPREAD_NORMALIZED_LTP_STEP,
                                "eps": 1.0e-6,
                            }
                        ]
                    },
                },
            ],
        }
        result = run_transformation_pipeline(
            df, cfg, context=TransformContext(sample_interval_sec=3)
        )
        out = result.frame
        self.assertIn(SPREAD_NORMALIZED_LTP_STEP, out.columns)
        # Row 1: (100.5-100.0)/0.5 = 1.0
        self.assertAlmostEqual(float(out[SPREAD_NORMALIZED_LTP_STEP].iloc[1]), 1.0)


if __name__ == "__main__":
    unittest.main()
