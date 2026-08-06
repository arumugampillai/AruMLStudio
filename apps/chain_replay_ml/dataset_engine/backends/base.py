"""Backend protocol for Dataset Engine adapters."""

from __future__ import annotations

from typing import Protocol

from ..planner import QueryPlan
from ..types import QueryResult


class DatasetBackend(Protocol):
    """Execute a QueryPlan and return Arrow + stats."""

    name: str

    def execute(self, plan: QueryPlan) -> QueryResult:
        ...
