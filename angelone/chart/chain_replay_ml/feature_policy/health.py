"""Per-feature health statistics from policy engine runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FeatureHealthRow:
    name: str
    rows: int = 0
    ready_count: int = 0
    warmup_count: int = 0
    gap_reset_count: int = 0
    missing_count: int = 0

    @property
    def ready_pct(self) -> float:
        return round(self.ready_count / max(self.rows, 1) * 100.0, 2)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rows": self.rows,
            "ready_pct": self.ready_pct,
            "warmup_pct": round(self.warmup_count / max(self.rows, 1) * 100.0, 2),
            "gap_reset_pct": round(self.gap_reset_count / max(self.rows, 1) * 100.0, 2),
            "missing_pct": round(self.missing_count / max(self.rows, 1) * 100.0, 2),
        }


@dataclass
class FeatureHealthTracker:
    _rows: dict[str, FeatureHealthRow] = field(default_factory=dict)

    def record(
        self,
        name: str,
        *,
        ready: bool,
        warmup: bool = False,
        gap_reset: bool = False,
        missing: bool = False,
    ) -> None:
        row = self._rows.setdefault(name, FeatureHealthRow(name=name))
        row.rows += 1
        if ready:
            row.ready_count += 1
        if warmup:
            row.warmup_count += 1
        if gap_reset:
            row.gap_reset_count += 1
        if missing:
            row.missing_count += 1

    def summary(self) -> list[dict[str, Any]]:
        return [r.as_dict() for r in sorted(self._rows.values(), key=lambda x: x.name)]

    def feature_report(self, name: str) -> dict[str, Any] | None:
        row = self._rows.get(name)
        return row.as_dict() if row else None
