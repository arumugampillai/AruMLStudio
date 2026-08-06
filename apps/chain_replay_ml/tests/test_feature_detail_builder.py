"""Tests for unified feature detail builder."""

from __future__ import annotations

import unittest

from master_dataset_tk.feature_detail_builder import build_feature_detail
from master_dataset_tk.feature_detail_format import format_feature_detail_text


class FeatureDetailBuilderTests(unittest.TestCase):
    def test_chain_pcr_detail(self) -> None:
        detail = build_feature_detail("chain_pcr")
        self.assertTrue(detail.get("ok"))
        self.assertIn("put", detail.get("formula_doc", "").lower())
        self.assertIn("chain_pcr", detail.get("python_code", ""))
        text = format_feature_detail_text(detail)
        self.assertIn("Formula", text)
        self.assertIn("Python", text)

    def test_ema_ratio_detail(self) -> None:
        detail = build_feature_detail("ltp_ema20_to_spot_ratio")
        self.assertTrue(detail.get("ok"))
        code = detail.get("python_code") or ""
        self.assertIn("rolling_ema", code)
        self.assertIn("ema20", code.lower())

    def test_context_in_format(self) -> None:
        detail = build_feature_detail(
            "roll_age_min",
            context={"sample": 401, "time": "10:19:38", "display": "5.0000", "status": "VALUE"},
        )
        text = format_feature_detail_text(detail)
        self.assertIn("Sample Context", text)
        self.assertIn("401", text)
        self.assertIn("5.0000", text)


if __name__ == "__main__":
    unittest.main()
