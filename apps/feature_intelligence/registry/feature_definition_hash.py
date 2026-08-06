"""Definition hash for Feature Registry (Sprint 2)."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def compute_definition_hash(
    *,
    canonical_name: str,
    warmup_periods: int,
    gap_policy: str,
    memory_model: str,
    primitive_ids: Iterable[str],
) -> str:
    """SHA-256 (lowercase hex) over UTF-8 canonical definition line."""
    ids = ",".join(sorted(primitive_ids))
    line = f"{canonical_name}|{int(warmup_periods)}|{gap_policy}|{memory_model}|{ids}"
    return hashlib.sha256(line.encode("utf-8")).hexdigest()
