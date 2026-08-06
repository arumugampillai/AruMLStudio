"""Shared types for correlation compute backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

BackendPreference = Literal["auto", "cpu", "gpu"]
BackendUsed = Literal["cpu", "gpu"]

VALID_PREFERENCES: frozenset[str] = frozenset({"auto", "cpu", "gpu"})


def normalize_preference(value: str | None) -> BackendPreference:
    raw = str(value or "auto").strip().lower()
    if raw in VALID_PREFERENCES:
        return raw  # type: ignore[return-value]
    return "auto"


@dataclass
class CorrelationTiming:
    """Wall-clock timings in seconds for a single Pearson matrix compute."""

    cpu_compute_sec: float | None = None
    gpu_transfer_sec: float | None = None
    gpu_compute_sec: float | None = None
    total_sec: float = 0.0
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CorrelationComputeResult:
    """Pearson correlation matrix plus backend metadata.

    ``matrix`` is always a pandas DataFrame (UI / Analysis Lab contract).
    """

    matrix: Any
    backend_used: BackendUsed
    preference: BackendPreference
    timing: CorrelationTiming = field(default_factory=CorrelationTiming)
    n_rows: int = 0
    n_features: int = 0
    gpu_available: bool = False

    def to_meta(self) -> dict[str, Any]:
        return {
            "backend_used": self.backend_used,
            "preference": self.preference,
            "gpu_available": bool(self.gpu_available),
            "n_rows": int(self.n_rows),
            "n_features": int(self.n_features),
            "timing": self.timing.to_dict(),
        }
