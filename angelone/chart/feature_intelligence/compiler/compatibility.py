"""Compiler compatibility matrix enforcement (Sprint 5)."""

from __future__ import annotations

import json
from pathlib import Path

from feature_intelligence.compiler.models import COMPILER_VERSION
from feature_intelligence.grammar.pack import GRAMMAR_VERSION
from feature_intelligence.operators.catalog import OPERATOR_PACK_VERSION

COMPAT_PATH = Path(__file__).resolve().parent / "compiler_compatibility.json"


def load_compatibility(path: Path | None = None) -> dict:
    return json.loads((path or COMPAT_PATH).read_text(encoding="utf-8"))


def is_supported(
    *,
    grammar_version: str = GRAMMAR_VERSION,
    compiler_version: str = COMPILER_VERSION,
    operator_pack_version: str = OPERATOR_PACK_VERSION,
    matrix: dict | None = None,
) -> bool:
    data = matrix if matrix is not None else load_compatibility()
    if str(data.get("compiler_version")) != compiler_version:
        # Matrix file may be for a different compiler release
        if compiler_version != COMPILER_VERSION:
            return False
    for row in data.get("supported") or []:
        if (
            str(row.get("grammar_version")) == grammar_version
            and str(row.get("operator_pack_version")) == operator_pack_version
            and bool(row.get("supported"))
        ):
            return True
    return False


def require_supported(
    *,
    grammar_version: str = GRAMMAR_VERSION,
    compiler_version: str = COMPILER_VERSION,
    operator_pack_version: str = OPERATOR_PACK_VERSION,
) -> None:
    if not is_supported(
        grammar_version=grammar_version,
        compiler_version=compiler_version,
        operator_pack_version=operator_pack_version,
    ):
        raise CompatibilityError(
            f"unsupported triple grammar={grammar_version} "
            f"compiler={compiler_version} pack={operator_pack_version}"
        )


class CompatibilityError(Exception):
    """Raised when grammar/compiler/operator-pack combination is unsupported."""

    code = "COMPAT_MATRIX"
