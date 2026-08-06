"""Regression tests for grammar/examples corpus."""

from __future__ import annotations

import re
import tempfile
import unittest
from pathlib import Path

from feature_intelligence.grammar.formatter import format_expression
from feature_intelligence.grammar.pack import EXAMPLES_INVALID, EXAMPLES_VALID
from feature_intelligence.grammar.validator import primary_error_code, validate_text
from feature_intelligence.migrations.runner import MigrationRunner

_EXPECT_RE = re.compile(r"#\s*expect:\s*([A-Z_]+)")
_BOUND_CODES = frozenset({"UNKNOWN_OPERATOR", "MISSING_PARAM", "UNKNOWN_PARAM"})


def _expected_code(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    m = _EXPECT_RE.search(text)
    if m:
        return m.group(1)
    sidecar = path.with_suffix(".expected")
    if sidecar.is_file():
        return sidecar.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    raise AssertionError(f"No expected error code for {path}")


class TestGrammarExamples(unittest.TestCase):
    def test_valid_examples(self) -> None:
        self.assertTrue(EXAMPLES_VALID.is_dir())
        files = sorted(EXAMPLES_VALID.glob("*.tl"))
        self.assertGreaterEqual(len(files), 2)
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "feature_intelligence.db"
            MigrationRunner(db).upgrade()
            for path in files:
                text = path.read_text(encoding="utf-8")
                self.assertNotIn("\u2192", text)
                self.assertNotIn("->", text.replace("-->", ""))
                report = validate_text(text, mode="bound", db_path=db)
                self.assertTrue(report.passed, f"{path.name}: {report.failed_rules}")
                canonical = format_expression(text)
                self.assertEqual(format_expression(canonical), canonical)

    def test_invalid_examples(self) -> None:
        self.assertTrue(EXAMPLES_INVALID.is_dir())
        files = sorted(EXAMPLES_INVALID.glob("*.tl"))
        self.assertGreaterEqual(len(files), 5)
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "feature_intelligence.db"
            MigrationRunner(db).upgrade()
            for path in files:
                code = _expected_code(path)
                text = path.read_text(encoding="utf-8")
                mode = "bound" if code in _BOUND_CODES else "syntax_only"
                report = validate_text(
                    text,
                    mode=mode,
                    db_path=db if mode == "bound" else None,
                )
                self.assertFalse(report.passed, path.name)
                self.assertEqual(primary_error_code(report), code, path.name)


if __name__ == "__main__":
    unittest.main()
