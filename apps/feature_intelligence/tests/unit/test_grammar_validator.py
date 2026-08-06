"""Unit tests for TL syntax validator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from feature_intelligence.grammar.pack import (
    EXPECTED_GRAMMAR_CHECKSUM,
    compute_grammar_pack_checksum,
)
from feature_intelligence.grammar.validator import (
    MAX_NESTING_DEPTH,
    primary_error_code,
    validate_text,
)
from feature_intelligence.migrations.runner import MigrationRunner


class TestGrammarValidator(unittest.TestCase):
    def test_checksum_locked(self) -> None:
        self.assertEqual(compute_grammar_pack_checksum(), EXPECTED_GRAMMAR_CHECKSUM)
        self.assertTrue(EXPECTED_GRAMMAR_CHECKSUM)

    def test_syntax_only_accepts_ema(self) -> None:
        report = validate_text(
            "OP_EMA(source=PR_SPOT, period=20)", mode="syntax_only"
        )
        self.assertTrue(report.passed, report.failed_rules)
        self.assertEqual(report.validated_objects, "1 expression")

    def test_syntax_only_vs_bound_unknown_op(self) -> None:
        text = "OP_NOTREAL(period=20)"
        syn = validate_text(text, mode="syntax_only")
        self.assertTrue(syn.passed, syn.failed_rules)

        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "feature_intelligence.db"
            MigrationRunner(db).upgrade()
            bound = validate_text(text, mode="bound", db_path=db)
            self.assertFalse(bound.passed)
            self.assertEqual(primary_error_code(bound), "UNKNOWN_OPERATOR")

    def test_bound_ema_with_source_slot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "feature_intelligence.db"
            MigrationRunner(db).upgrade()
            report = validate_text(
                "OP_EMA(source=PR_SPOT, period=20)",
                mode="bound",
                db_path=db,
            )
            self.assertTrue(report.passed, report.failed_rules)

    def test_bound_missing_param(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "feature_intelligence.db"
            MigrationRunner(db).upgrade()
            report = validate_text("OP_EMA()", mode="bound", db_path=db)
            self.assertFalse(report.passed)
            self.assertEqual(primary_error_code(report), "MISSING_PARAM")

    def test_feat_pattern(self) -> None:
        bad = validate_text(
            "OP_RATIO(left=FEAT_nothex, right=PR_SPOT)", mode="syntax_only"
        )
        self.assertFalse(bad.passed)
        good_id = "FEAT_" + ("A" * 32)
        ok = validate_text(
            f"OP_RATIO(left={good_id}, right=PR_SPOT)", mode="syntax_only"
        )
        self.assertTrue(ok.passed, ok.failed_rules)

    def test_nesting_depth(self) -> None:
        # Build OP_ABS(source=OP_ABS(source=...)) deeper than max
        expr = "PR_SPOT"
        for _ in range(MAX_NESTING_DEPTH + 1):
            expr = f"OP_ABS(source={expr})"
        report = validate_text(expr, mode="syntax_only")
        self.assertFalse(report.passed)
        self.assertEqual(primary_error_code(report), "NESTING_DEPTH")

    def test_error_codes(self) -> None:
        cases = [
            ("OP_EMA(period=20,)", "TRAILING_COMMA"),
            ("OP_EMA(20)", "POSITIONAL_ARG"),
            ("OP_EMA(period: 20)", "FORBIDDEN_COLON"),
            ("OP_EMA(period=20, period=30)", "DUPLICATE_PARAM"),
            ("OP_EMA(period=20", "UNBALANCED_PAREN"),
        ]
        for text, code in cases:
            report = validate_text(text, mode="syntax_only")
            self.assertFalse(report.passed, text)
            self.assertEqual(primary_error_code(report), code, text)


if __name__ == "__main__":
    unittest.main()
