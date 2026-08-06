"""Unit tests for TL import/export envelopes."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from feature_intelligence.grammar.formatter import format_expression
from feature_intelligence.grammar.import_export import export_expression, import_expression
from feature_intelligence.grammar.pack import FORMATTER_VERSION, GRAMMAR_VERSION


class TestGrammarImportExport(unittest.TestCase):
    def test_json_round_trip(self) -> None:
        raw = "OP_EMA(source=PR_SPOT,period=20)"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            exported = root / "out.json"
            export_expression(raw, exported, fmt="json")
            data = json.loads(exported.read_text(encoding="utf-8"))
            self.assertEqual(data["grammar_version"], GRAMMAR_VERSION)
            self.assertEqual(data["formatter_version"], FORMATTER_VERSION)
            self.assertEqual(data["expression"], format_expression(raw))

            out_tl = root / "round.tl"
            canonical = import_expression(exported, out_tl, fmt="json")
            self.assertEqual(canonical, format_expression(raw))
            again = root / "again.json"
            export_expression(out_tl.read_text(encoding="utf-8"), again, fmt="json")
            data2 = json.loads(again.read_text(encoding="utf-8"))
            self.assertEqual(data2["expression"], data["expression"])

    def test_text_round_trip(self) -> None:
        raw = "OP_RATIO(left=PR_SPOT,right=PR_SPOT)"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "a.tl"
            export_expression(raw, src, fmt="text")
            dst = root / "b.tl"
            canonical = import_expression(src, dst, fmt="text")
            self.assertEqual(canonical, format_expression(raw))

    def test_yaml_export_import(self) -> None:
        raw = "OP_EMA(period=20)"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            yml = root / "e.yaml"
            export_expression(raw, yml, fmt="yaml")
            out = root / "e.tl"
            canonical = import_expression(yml, out, fmt="yaml")
            self.assertEqual(canonical, format_expression(raw))


if __name__ == "__main__":
    unittest.main()
