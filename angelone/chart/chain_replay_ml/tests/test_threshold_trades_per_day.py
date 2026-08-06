"""Trades/Day on Threshold Analysis rows."""

from __future__ import annotations

import unittest

from chain_replay_ml.training.evaluator import (
    attach_trades_per_day,
    normalize_threshold_analysis_rows,
    threshold_row_from_confusion,
)


class TradesPerDayTests(unittest.TestCase):
    def test_row_includes_trades_per_day(self) -> None:
        row = threshold_row_from_confusion(
            threshold=0.5, tp=100, fp=50, fn=10, tn=40, n_days=10
        )
        self.assertEqual(row["buy_signals"], 150)
        self.assertEqual(row["n_days"], 10)
        self.assertEqual(row["trades_per_day"], 15.0)

    def test_attach_enriches_legacy_rows(self) -> None:
        raw = [{"threshold": 0.7, "tp": 20, "fp": 10, "fn": 5, "tn": 65}]
        rows = attach_trades_per_day(normalize_threshold_analysis_rows(raw), 5)
        self.assertEqual(rows[0]["buy_signals"], 30)
        self.assertEqual(rows[0]["trades_per_day"], 6.0)


if __name__ == "__main__":
    unittest.main()
