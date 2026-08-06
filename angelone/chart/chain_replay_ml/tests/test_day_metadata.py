"""Trading-day metadata — compute once, persist, UI reads."""



from __future__ import annotations



import os

import tempfile

import unittest





class DayMetadataTests(unittest.TestCase):

    def test_compute_and_persist_roundtrip(self) -> None:

        from chain_replay_ml.dataset_builder.day_metadata import (

            build_and_persist_day_metadata,

            load_column_metadata,

            load_day_overview,

            load_gap_metadata,

        )



        base = 1_700_000_000.0

        rows = []

        for i in range(20):

            ts = base + i * 3.0

            if i >= 10:

                ts = base + 9 * 3.0 + 45.0 + (i - 10) * 3.0

            rows.append(

                {

                    "trading_day": "2026-07-21",

                    "token": "T1",

                    "timestamp": ts,

                    "expiry": "2026-07-21",

                    "spot": 100.0 + i,

                    "current_iv": 0.12,

                    "bid_ask_spread": 0.5,

                    "ema200": None if i < 5 else 100.0,

                    "empty_feat": None,

                }

            )



        with tempfile.TemporaryDirectory() as tmp:

            import sqlite3



            path = os.path.join(tmp, "m.db")

            conn = sqlite3.connect(path)

            try:

                payload = build_and_persist_day_metadata(

                    conn,

                    rows,

                    trading_day="2026-07-21",

                    registry_features=["spot", "ema200", "empty_feat", "missing_reg"],

                    meta_columns=("trading_day", "token", "timestamp", "expiry"),

                    gap_max_sec=20.0,

                    sampling_interval_sec=3.0,

                    build_version="test_ds",

                    family_by_name={"spot": "Base", "ema200": "Derived"},

                    ingestion={

                        "dataset_version": "test_ds",

                        "registry_version": "6",

                        "feature_engine_version": "6",

                        "gap_policy_version": "1",

                    },

                )

                overview = load_day_overview(conn, "2026-07-21")

                cols = load_column_metadata(conn, "2026-07-21")

                gaps = load_gap_metadata(conn, "2026-07-21")

            finally:

                conn.close()



        self.assertIsNotNone(overview)

        assert overview is not None

        self.assertEqual(overview["total_rows"], 20)

        self.assertEqual(overview["registry_missing"], 1)

        self.assertGreaterEqual(overview["gap_triggered"], 1)

        self.assertEqual(overview["dataset_version"], "test_ds")

        self.assertEqual(overview["registry_version"], "6")

        self.assertIsNotNone(overview.get("spot_min"))

        self.assertGreaterEqual(int(overview.get("unexpected_empty_features") or 0), 1)

        self.assertTrue(any(c["feature"] == "spot" and c["status"] == "Healthy" for c in cols))

        ema = next(c for c in cols if c["feature"] == "ema200")

        self.assertEqual(ema["status"], "Warm-up")

        self.assertEqual(ema["feature_family"], "Derived")

        empty = next(c for c in cols if c["feature"] == "empty_feat")

        self.assertEqual(empty["status"], "Unexpected Empty")

        self.assertEqual(empty.get("availability"), "Unavailable")

        self.assertIn("Unexpected", str(empty.get("reason") or ""))



        futures_rows = [

            {

                "trading_day": "2026-07-21",

                "token": "T1",

                "timestamp": 1.0 + i,

                "spot": 100.0,

                "futures_ltp": None,

            }

            for i in range(5)

        ]

        with tempfile.TemporaryDirectory() as tmp2:

            import sqlite3



            conn2 = sqlite3.connect(os.path.join(tmp2, "m2.db"))

            try:

                build_and_persist_day_metadata(

                    conn2,

                    futures_rows,

                    trading_day="2026-07-21",

                    registry_features=["spot", "futures_ltp"],

                    meta_columns=("trading_day", "token", "timestamp"),

                )

                cols2 = load_column_metadata(conn2, "2026-07-21")

                overview2 = load_day_overview(conn2, "2026-07-21")

            finally:

                conn2.close()

        fut = next(c for c in cols2 if c["feature"] == "futures_ltp")

        self.assertEqual(fut["status"], "Expected Empty")

        self.assertIn("Futures", str(fut.get("reason") or ""))

        self.assertEqual(fut.get("source"), "Futures Feed")

        self.assertEqual(fut.get("availability"), "Optional")

        self.assertTrue(bool(fut.get("expected_empty")))

        self.assertGreaterEqual(int(overview2.get("expected_empty_features") or 0), 1)

        missing = next(c for c in cols if c["feature"] == "missing_reg")

        self.assertEqual(missing["status"], "Registry Missing")

        self.assertTrue(any(g.get("action") == "Gap policy triggered" for g in gaps))

        self.assertIn("coverage", payload["day"]["health_components"])



    def test_delete_day_metadata(self) -> None:

        from chain_replay_ml.dataset_builder.day_metadata import (

            build_and_persist_day_metadata,

            delete_day_metadata,

            load_day_overview,

        )

        import sqlite3



        rows = [

            {

                "trading_day": "2026-07-21",

                "token": "T1",

                "timestamp": 1.0 + i,

                "spot": 1.0,

            }

            for i in range(5)

        ]

        with tempfile.TemporaryDirectory() as tmp:

            conn = sqlite3.connect(os.path.join(tmp, "m.db"))

            try:

                build_and_persist_day_metadata(

                    conn,

                    rows,

                    trading_day="2026-07-21",

                    registry_features=["spot"],

                    meta_columns=("trading_day", "token", "timestamp"),

                )

                self.assertIsNotNone(load_day_overview(conn, "2026-07-21"))

                delete_day_metadata(conn, "2026-07-21")

                conn.commit()

                self.assertIsNone(load_day_overview(conn, "2026-07-21"))

            finally:

                conn.close()





if __name__ == "__main__":

    unittest.main()


