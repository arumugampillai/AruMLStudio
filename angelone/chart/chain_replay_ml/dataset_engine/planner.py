"""Query planning — normalize filters, select partitions, minimize columns (ADR §5–6)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .types import SampleSpec


@dataclass
class QueryPlan:
    """Backend-agnostic plan produced before I/O."""

    dataset_id: str
    columns: tuple[str, ...] | None
    filters: dict[str, Any]
    sample: SampleSpec | None
    schema_version: str | None = None
    # Partition keys selected after pruning (e.g. trading_day list).
    partitions: tuple[str, ...] | None = None
    partitions_pruned: int = 0
    # Optional deterministic ORDER BY columns (training time-series stability).
    order_by: tuple[str, ...] | None = None
    # Optional Parquet row-group indices (Create Model chronological tail prune).
    row_groups: tuple[int, ...] | None = None
    notes: list[str] = field(default_factory=list)


def plan_query(
    dataset_id: str,
    *,
    columns: Sequence[str] | None = None,
    filters: Mapping[str, Any] | None = None,
    sample: SampleSpec | None = None,
    schema_version: str | None = None,
    available_partitions: Sequence[str] | None = None,
    order_by: Sequence[str] | None = None,
    row_groups: Sequence[int] | None = None,
) -> QueryPlan:
    """Compile a request into a QueryPlan.

    Phase-1 behaviour:
    - Normalize filters to a plain dict.
    - If ``filters`` contains ``trading_day`` / ``trading_days``, prune
      ``available_partitions`` when provided.
    - Preserve requested column list (None = all).
    - Optional ``order_by`` for deterministic row sequences (Create Model).
    - Optional ``row_groups`` to materialize only selected Parquet groups.
    """
    filt = dict(filters or {})
    col_tuple = tuple(columns) if columns is not None else None
    order_tuple = tuple(str(c) for c in order_by) if order_by else None
    rg_tuple = tuple(int(i) for i in row_groups) if row_groups is not None else None

    partitions: tuple[str, ...] | None = None
    pruned = 0
    notes: list[str] = []

    day_filter = filt.get("trading_days")
    if day_filter is None and "trading_day" in filt:
        day_filter = [filt["trading_day"]]

    if available_partitions is not None:
        available = tuple(str(p) for p in available_partitions)
        if day_filter is not None:
            wanted = {str(d) for d in day_filter}
            selected = tuple(p for p in available if p in wanted)
            pruned = len(available) - len(selected)
            partitions = selected
            notes.append(f"partition_prune trading_day wanted={len(wanted)} kept={len(selected)}")
        else:
            partitions = available
            notes.append("no day filter; all available partitions selected")

    if order_tuple:
        notes.append(f"order_by={list(order_tuple)}")
    if rg_tuple is not None:
        notes.append(f"row_groups={list(rg_tuple)}")

    return QueryPlan(
        dataset_id=str(dataset_id),
        columns=col_tuple,
        filters=filt,
        sample=sample,
        schema_version=schema_version,
        partitions=partitions,
        partitions_pruned=pruned,
        order_by=order_tuple,
        row_groups=rg_tuple,
        notes=notes,
    )
