"""Vocabulary type maps and ID pattern helpers (Sprint 6)."""

from __future__ import annotations

import re

VOCAB_TYPE_PREFIX: dict[str, str] = {
    "DOMAIN": "DOM_",
    "SIGNAL_TYPE": "SIG_",
    "MATH_FAMILY": "MATH_",
    "HORIZON": "HOR_",
    "OUTPUT_TYPE": "OUT_",
    "FREQUENCY": "FREQ_",
    "STABILITY": "STAB_",
}

VOCAB_ID_PATTERNS: dict[str, re.Pattern[str]] = {
    "DOMAIN": re.compile(r"^DOM_[A-Z][A-Z0-9_]*$"),
    "SIGNAL_TYPE": re.compile(r"^SIG_[A-Z][A-Z0-9_]*$"),
    "MATH_FAMILY": re.compile(r"^MATH_[A-Z][A-Z0-9_]*$"),
    "HORIZON": re.compile(r"^HOR_[A-Z][A-Z0-9_]*$"),
    "OUTPUT_TYPE": re.compile(r"^OUT_[A-Z][A-Z0-9_]*$"),
    "FREQUENCY": re.compile(r"^FREQ_[A-Z0-9_]+$"),
    "STABILITY": re.compile(r"^STAB_[A-Z][A-Z0-9_]*$"),
}

# Field → expected vocabulary_type
FIELD_VOCAB_TYPE: dict[str, str] = {
    "domain": "DOMAIN",
    "signal_type": "SIGNAL_TYPE",
    "mathematical_family": "MATH_FAMILY",
    "horizon": "HORIZON",
    "output_type": "OUTPUT_TYPE",
    "frequency": "FREQUENCY",
    "stability": "STABILITY",
}

OBJECT_REF_PATTERN = re.compile(
    r"^(PR_[A-Z][A-Z0-9_]*|OP_[A-Z][A-Z0-9_]*|TR_[0-9A-F]{32}|FEAT_[0-9A-F]{32})$"
)


def vocab_type_for_id(vocabulary_id: str) -> str | None:
    for vtype, prefix in VOCAB_TYPE_PREFIX.items():
        if vocabulary_id.startswith(prefix):
            return vtype
    return None


def is_valid_vocab_id(vocabulary_id: str, vocabulary_type: str) -> bool:
    pat = VOCAB_ID_PATTERNS.get(vocabulary_type)
    if pat is None:
        return False
    return bool(pat.match(vocabulary_id))
