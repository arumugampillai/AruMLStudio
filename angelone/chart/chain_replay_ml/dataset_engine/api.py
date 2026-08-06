"""Dataset Engine public API — query_dataset / stream_dataset."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from .backends.duckdb_backend import DuckDbBackend
from .planner import plan_query
from .types import QueryResult, SampleSpec


def query_dataset(
    dataset_id: str,
    *,
    columns: Sequence[str] | None = None,
    filters: Mapping[str, Any] | None = None,
    sample: SampleSpec | None = None,
    schema_version: str | None = None,
    available_partitions: Sequence[str] | None = None,
    order_by: Sequence[str] | None = None,
    row_groups: Sequence[int] | None = None,
    backend: str = "duckdb",
    parquet_path: str | None = None,
) -> QueryResult:
    """Retrieve a filtered/pruned/sampled slice as Arrow + execution stats.

    Read-only. Does not generate or mutate Master features (ADR §0).
    """
    plan = plan_query(
        dataset_id,
        columns=columns,
        filters=filters,
        sample=sample,
        schema_version=schema_version,
        available_partitions=available_partitions,
        order_by=order_by,
        row_groups=row_groups,
    )
    if schema_version is not None:
        # Full registry stamping lands later; refuse silent mismatch hooks here.
        plan.notes.append(f"schema_version requested={schema_version}")

    if backend != "duckdb":
        raise ValueError(f"Unsupported Dataset Engine backend: {backend!r}")

    path = parquet_path
    if path is None and str(dataset_id).lower().endswith(".parquet"):
        path = str(dataset_id)
    executor = DuckDbBackend(parquet_path=path)
    return executor.execute(plan)


def stream_dataset(
    dataset_id: str,
    *,
    columns: Sequence[str] | None = None,
    filters: Mapping[str, Any] | None = None,
    sample: SampleSpec | None = None,
    schema_version: str | None = None,
    batch_size: int | None = None,
    **kwargs: Any,
) -> Iterator[QueryResult]:
    """Incremental batches (ADR §3.2). Not implemented in Phase-1 skeleton."""
    raise NotImplementedError(
        "stream_dataset is a future extension; use query_dataset for Phase 1."
    )