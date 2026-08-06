"""Rolling PredictionSnapshot history for momentum meta features."""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass
from typing import Any

from .snapshot import PredictionSnapshot


@dataclass
class _HistoryEntry:
    timestamp: float
    snapshot: PredictionSnapshot
    meta: dict[str, Any]


class PredictionSnapshotHistory:
    """In-memory ring buffer of prediction snapshots + derived meta scalars."""

    def __init__(self, *, maxlen: int = 30) -> None:
        self._entries: deque[_HistoryEntry] = deque(maxlen=maxlen)

    def append(self, snapshot: PredictionSnapshot, meta: dict[str, Any]) -> None:
        self._entries.append(_HistoryEntry(
            timestamp=float(snapshot.timestamp),
            snapshot=snapshot,
            meta=dict(meta),
        ))

    def latest_meta_value(self, key: str) -> float | None:
        if not self._entries:
            return None
        val = self._entries[-1].meta.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def meta_value_at_offset(self, key: str, offset_sec: float) -> float | None:
        if not self._entries:
            return None
        target_ts = self._entries[-1].timestamp - float(offset_sec)
        candidate: _HistoryEntry | None = None
        for entry in reversed(self._entries):
            if entry.timestamp <= target_ts + 0.05:
                candidate = entry
                break
        if candidate is None:
            return None
        val = candidate.meta.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    def meta_series(self, key: str, window_sec: float) -> list[tuple[float, float]]:
        if not self._entries:
            return []
        end_ts = self._entries[-1].timestamp
        start_ts = end_ts - float(window_sec)
        out: list[tuple[float, float]] = []
        for entry in self._entries:
            if entry.timestamp < start_ts - 0.05:
                continue
            val = entry.meta.get(key)
            if val is None:
                continue
            try:
                out.append((entry.timestamp, float(val)))
            except (TypeError, ValueError):
                continue
        return out

    def rolling_std(self, key: str, window_sec: float) -> float | None:
        series = self.meta_series(key, window_sec)
        if len(series) < 2:
            return None
        vals = [v for _t, v in series]
        return statistics.pstdev(vals)

    def __len__(self) -> int:
        return len(self._entries)
