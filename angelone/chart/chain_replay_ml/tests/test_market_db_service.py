"""Tests for market_db_service helpers."""

from __future__ import annotations

import unittest

from master_dataset_tk.market_db_service import format_size_bytes, format_tick_count


class MarketDbServiceFormatTests(unittest.TestCase):
    def test_format_size_bytes(self) -> None:
        self.assertEqual(format_size_bytes(512), "512 B")
        self.assertEqual(format_size_bytes(2048), "2.0 KB")
        self.assertIn("MB", format_size_bytes(5 * 1024 * 1024))

    def test_format_tick_count(self) -> None:
        self.assertEqual(format_tick_count(42), "42")
        self.assertEqual(format_tick_count(1500), "1.5k")
        self.assertIn("M", format_tick_count(2_500_000))


if __name__ == "__main__":
    unittest.main()
