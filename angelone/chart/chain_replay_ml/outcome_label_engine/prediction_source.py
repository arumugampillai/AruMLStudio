"""Prediction Dataset day-chunk adapters for OLE (read-only; no Prediction DB writes)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass
class InMemoryPredictionDaySource:
    """Day-chunked Prediction samples already in memory (tests / small runs)."""

    days: dict[str, list[dict[str, Any]]]
    source_kind: str = "prediction"
    session_close_by_day: dict[str, float] = field(default_factory=dict)

    def iter_days(self) -> Iterable[str]:
        return sorted(self.days.keys())

    def load_day(self, day: str) -> list[dict[str, Any]]:
        rows = list(self.days.get(day) or [])
        close_ts = self.session_close_by_day.get(day)
        if close_ts is None:
            return rows
        out: list[dict[str, Any]] = []
        for row in rows:
            r = dict(row)
            r.setdefault("session_close_ts", close_ts)
            out.append(r)
        return out


@dataclass
class CallablePredictionDaySource:
    """Lazy day loader — call ``load_fn(day)`` per chunk (bounded RAM)."""

    days: list[str]
    load_fn: Callable[[str], list[dict[str, Any]]]
    source_kind: str = "prediction"

    def iter_days(self) -> Iterable[str]:
        return list(self.days)

    def load_day(self, day: str) -> list[dict[str, Any]]:
        return list(self.load_fn(day) or [])
