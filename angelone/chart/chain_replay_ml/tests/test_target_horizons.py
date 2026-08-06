"""Tests for Tk target horizon helpers."""

from __future__ import annotations

import unittest

from master_dataset_tk.target_horizons import (
    ALL_HORIZON_SEC,
    DEFAULT_HORIZON_SEC,
    compact_target_label,
    default_horizon_selection,
    horizons_summary_labels,
    target_horizon_columns,
    target_horizon_rows,
)


class TargetHorizonsTests(unittest.TestCase):
    def test_all_horizons_match_web(self) -> None:
        self.assertEqual(ALL_HORIZON_SEC, [3, 5, 10, 30, 60, 180, 300])
        self.assertEqual(DEFAULT_HORIZON_SEC, ALL_HORIZON_SEC)

    def test_target_rows_from_schema(self) -> None:
        rows = target_horizon_rows()
        self.assertEqual(len(rows), 7)
        self.assertEqual(rows[0]["name"], "future_ltp_3s")
        self.assertEqual(rows[0]["display_name"], "Future LTP (3 s)")
        self.assertEqual(rows[-2]["name"], "future_ltp_3m")
        self.assertEqual(rows[-2]["display_name"], "Future LTP (3 m)")

    def test_target_columns_split(self) -> None:
        seconds, minutes = target_horizon_columns()
        self.assertEqual([r["sec"] for r in seconds], [3, 5, 10, 30])
        self.assertEqual([r["sec"] for r in minutes], [60, 180, 300])

    def test_default_selection_all(self) -> None:
        defaults = default_horizon_selection()
        for sec in ALL_HORIZON_SEC:
            self.assertTrue(defaults[sec])

    def test_summary_labels(self) -> None:
        text = horizons_summary_labels([10, 60])
        self.assertIn("Future LTP (10 s)", text)
        self.assertIn("Future LTP (1 m)", text)

    def test_compact_target_label(self) -> None:
        self.assertEqual(compact_target_label(3), "Future LTP (3 s)")
        self.assertEqual(compact_target_label(60), "Future LTP (1 m)")
        self.assertEqual(compact_target_label(180), "Future LTP (3 m)")


if __name__ == "__main__":
    unittest.main()
