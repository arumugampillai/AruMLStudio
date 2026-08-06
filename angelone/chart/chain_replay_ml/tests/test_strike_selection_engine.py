"""Tests for strike selection helpers."""

from __future__ import annotations

import unittest

from master_dataset_tk.strike_selection_engine import (
    MASTER_DATASET_ATM_BAND,
    atm_band_hint_text,
    default_strike_config,
    normalize_strike_config,
    strike_selection_for_master,
    strike_summary_label,
)


class StrikeSelectionEngineTests(unittest.TestCase):
    def test_default_config(self) -> None:
        cfg = default_strike_config()
        self.assertEqual(cfg["mode"], "atm_band")
        self.assertEqual(cfg["atmBand"], MASTER_DATASET_ATM_BAND)

    def test_atm_hint(self) -> None:
        self.assertIn("31", atm_band_hint_text(15))
        self.assertIn("No band limit", atm_band_hint_text("all"))

    def test_master_preserves_band(self) -> None:
        cfg = strike_selection_for_master({"mode": "atm_band", "atmBand": 10})
        self.assertEqual(cfg["mode"], "atm_band")
        self.assertEqual(cfg["atmBand"], 10)

    def test_atm_band_from_config(self) -> None:
        from master_dataset_tk.strike_selection_engine import atm_band_from_strike_config

        self.assertEqual(atm_band_from_strike_config({"mode": "atm_band", "atmBand": 10}), 10)

    def test_summary_label(self) -> None:
        text = strike_summary_label(normalize_strike_config({"mode": "atm_band", "atmBand": 10}))
        self.assertIn("±10", text)


if __name__ == "__main__":
    unittest.main()
