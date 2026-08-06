"""Dataset Engine — read-optimized retrieval over Registry / Analysis Parquet.

ADR: docs/architecture/DATASET_ENGINE.md
"""

from __future__ import annotations

from .api import query_dataset, stream_dataset
from .types import ExecutionStats, QueryResult, SampleSpec

__all__ = [
    "ExecutionStats",
    "QueryResult",
    "SampleSpec",
    "query_dataset",
    "stream_dataset",
]
