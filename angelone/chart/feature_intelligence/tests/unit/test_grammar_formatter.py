"""Unit tests for TL canonical formatter."""

from __future__ import annotations

import unittest

from feature_intelligence.grammar.formatter import format_expression


class TestGrammarFormatter(unittest.TestCase):
    def test_idempotent(self) -> None:
        raw = "OP_EMA(  source=PR_SPOT,period=20 )"
        once = format_expression(raw)
        twice = format_expression(once)
        self.assertEqual(once, twice)

    def test_layout(self) -> None:
        out = format_expression("OP_EMA(source=PR_SPOT,period=20)")
        expected = (
            "OP_EMA(\n"
            "    period = 20,\n"
            "    source = PR_SPOT\n"
            ")"
        )
        # period is schema-required first; source is alpha remainder
        self.assertEqual(out, expected)

    def test_nested_ratio(self) -> None:
        raw = "OP_RATIO(right=PR_SPOT,left=OP_EMA(period=20,source=PR_SPOT))"
        out = format_expression(raw)
        self.assertIn("left = OP_EMA(", out)
        self.assertIn("right = PR_SPOT", out)
        self.assertEqual(format_expression(out), out)

    def test_empty_call(self) -> None:
        self.assertEqual(format_expression("OP_DELTA()"), "OP_DELTA()")


if __name__ == "__main__":
    unittest.main()
