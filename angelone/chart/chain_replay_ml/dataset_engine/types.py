"""Public types for the Dataset Engine consumer contract."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# pyarrow is optional at import time until Phase 1 skeleton runs end-to-end.
try:
    import pyarrow as pa

    ArrowTable = pa.Table
except ImportError:  # pragma: no cover
    pa = None  # type: ignore[assignment]
    ArrowTable = Any  # type: ignore[misc,assignment]


@dataclass(frozen=True)
class SampleSpec:
    """Shared sampling policy (ADR §7)."""

    max_rows: int | None = None
    seed: int = 42
    # Future: stratify_by_day: bool = False


@dataclass
class ExecutionStats:
    """Best-effort execution metadata (ADR §3.3). Missing fields stay None."""

    rows_scanned: int | None = None
    rows_returned: int | None = None
    partitions_scanned: int | None = None
    partitions_pruned: int | None = None
    columns_read: tuple[str, ...] | None = None
    execution_time_sec: float | None = None
    bytes_read: int | None = None
    backend: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryResult:
    """Materialized query: Arrow table + execution stats."""

    table: ArrowTable
    stats: ExecutionStats
