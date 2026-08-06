"""Grammar pack constants and checksum (Sprint 4)."""

from __future__ import annotations

import hashlib
from pathlib import Path

GRAMMAR_VERSION = "1.0"
GRAMMAR_PACK_VERSION = "1.0.0"
TOKEN_PACK_VERSION = "1.0.0"
FORMATTER_VERSION = "1.0.0"

PACKAGE_ROOT = Path(__file__).resolve().parent
EBNF_PATH = PACKAGE_ROOT / "ebnf" / "tl_v1.ebnf"
TOKENS_PATH = PACKAGE_ROOT / "tokens.json"
COMPAT_PATH = PACKAGE_ROOT / "grammar_compatibility.json"
EXAMPLES_VALID = PACKAGE_ROOT / "examples" / "valid"
EXAMPLES_INVALID = PACKAGE_ROOT / "examples" / "invalid"


def compute_grammar_pack_checksum(
    *,
    ebnf: Path | None = None,
    compatibility: Path | None = None,
    tokens: Path | None = None,
) -> str:
    """SHA-256 over UTF-8 bytes: ebnf || compatibility || tokens."""
    parts = [
        (ebnf or EBNF_PATH).read_bytes(),
        (compatibility or COMPAT_PATH).read_bytes(),
        (tokens or TOKENS_PATH).read_bytes(),
    ]
    return hashlib.sha256(b"".join(parts)).hexdigest()


# Locked after first compute — update only with intentional freeze bump.
EXPECTED_GRAMMAR_CHECKSUM = (
    "4f6972fc7f8c94f57afbe340a0138a1d7751e9c86febf06ebc467c7d1e1b6f3e"
)
