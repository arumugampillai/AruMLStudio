"""Transformation Language grammar — validator + formatter (Sprint 4)."""

from __future__ import annotations

from feature_intelligence.grammar.formatter import format_expression, format_file
from feature_intelligence.grammar.import_export import export_expression, import_expression
from feature_intelligence.grammar.models import GrammarRegistryRecord
from feature_intelligence.grammar.pack import (
    EXPECTED_GRAMMAR_CHECKSUM,
    FORMATTER_VERSION,
    GRAMMAR_PACK_VERSION,
    GRAMMAR_VERSION,
    TOKEN_PACK_VERSION,
    compute_grammar_pack_checksum,
)
from feature_intelligence.grammar.store import GrammarStore
from feature_intelligence.grammar.validator import (
    MAX_NESTING_DEPTH,
    grammar_pack_report,
    primary_error_code,
    validate_expression,
    validate_file,
    validate_text,
)

__all__ = [
    "EXPECTED_GRAMMAR_CHECKSUM",
    "FORMATTER_VERSION",
    "GRAMMAR_PACK_VERSION",
    "GRAMMAR_VERSION",
    "MAX_NESTING_DEPTH",
    "TOKEN_PACK_VERSION",
    "GrammarRegistryRecord",
    "GrammarStore",
    "compute_grammar_pack_checksum",
    "export_expression",
    "format_expression",
    "format_file",
    "grammar_pack_report",
    "import_expression",
    "primary_error_code",
    "validate_expression",
    "validate_file",
    "validate_text",
]
