"""Tests for chain source selection persistence."""

from __future__ import annotations

import json
import os
import tempfile
import unittest


class ChainSourcePrefsTests(unittest.TestCase):
    def test_save_and_load_chain_source_prefs(self) -> None:
        import sys
        from pathlib import Path

        chart_root = Path(__file__).resolve().parents[2]
        if str(chart_root) not in sys.path:
            sys.path.insert(0, str(chart_root))
        from master_dataset_tk.build_config_prefs import (
            load_chain_source_prefs,
            save_chain_source_prefs,
            storage_path,
        )

        with tempfile.TemporaryDirectory() as tmp:
            chart_dir = os.path.join(tmp, "chart")
            os.makedirs(os.path.join(chart_dir, "data"), exist_ok=True)
            save_chain_source_prefs(
                chart_dir,
                market="NIFTY",
                selected_source_ids=[
                    "2026-07-01|NIFTY|2026-07-07",
                    "2026-07-02|NIFTY|2026-07-10",
                ],
            )
            self.assertTrue(os.path.isfile(storage_path(chart_dir)))
            loaded = load_chain_source_prefs(chart_dir)
            self.assertEqual(loaded["market"], "NIFTY")
            self.assertEqual(
                loaded["selected_source_ids"],
                ["2026-07-01|NIFTY|2026-07-07", "2026-07-02|NIFTY|2026-07-10"],
            )

    def test_inventory_rows_include_db_path(self) -> None:
        import sys
        from pathlib import Path

        chart_root = Path(__file__).resolve().parents[2]
        if str(chart_root) not in sys.path:
            sys.path.insert(0, str(chart_root))
        from master_dataset_tk.inventory import load_chain_inventory_rows

        with tempfile.TemporaryDirectory() as tmp:
            chart_dir = os.path.join(tmp, "chart")
            data_dir = os.path.join(chart_dir, "data")
            os.makedirs(data_dir, exist_ok=True)
            cache = {
                "last_updated": "2026-07-01T10:00:00",
                "databases": {
                    "2026-07-01": {
                        "db_file": "2026-07-01.db",
                        "db_path": "replay/2026-07-01.db",
                        "rows": [
                            {"kind": "spot", "index": "NIFTY"},
                            {
                                "kind": "chain",
                                "index": "NIFTY",
                                "expiry": "2026-07-07",
                                "tick_count": 100,
                            },
                        ],
                    },
                },
            }
            with open(os.path.join(data_dir, "market_db_inventory.json"), "w", encoding="utf-8") as fh:
                json.dump(cache, fh)
            rows, _meta = load_chain_inventory_rows(chart_dir)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["db_file"], "2026-07-01.db")
            self.assertEqual(rows[0]["db_path"], "replay/2026-07-01.db")

    def test_selected_sources_respect_market_filter(self) -> None:
        import sys
        from pathlib import Path

        chart_root = Path(__file__).resolve().parents[2]
        if str(chart_root) not in sys.path:
            sys.path.insert(0, str(chart_root))
        from master_dataset_tk.inventory import filter_inventory_rows

        rows = [
            {"source_id": "2026-07-03|NIFTY|2026-07-10", "market": "NIFTY", "spot_available": True},
            {"source_id": "2026-07-07|SENSEX|2026-07-09", "market": "SENSEX", "spot_available": True},
        ]
        selected_ids = {r["source_id"] for r in rows}
        visible = filter_inventory_rows(rows, "NIFTY")
        visible_ids = {str(r["source_id"]) for r in visible}
        picked = [
            r for r in rows
            if r.get("source_id") in selected_ids and str(r.get("source_id")) in visible_ids
        ]
        self.assertEqual(len(picked), 1)
        self.assertEqual(picked[0]["market"], "NIFTY")


if __name__ == "__main__":
    unittest.main()
