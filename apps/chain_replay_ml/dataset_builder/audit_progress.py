"""Audit stage timing for WebSocket progress events."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any


def system_metrics() -> dict[str, Any]:
    """Best-effort CPU and memory for live audit dashboard."""
    try:
        import psutil

        proc = psutil.Process()
        return {
            "memory_mb": round(proc.memory_info().rss / (1024 * 1024), 1),
            "cpu_percent": round(proc.cpu_percent(interval=None), 1),
        }
    except Exception:
        return {}


class AuditStageTracker:
    """Track per-stage durations and emit structured progress payloads."""

    def __init__(self, on_progress: Callable[[dict[str, Any]], None] | None) -> None:
        self._on_progress = on_progress
        self._starts: dict[str, float] = {}
        self.durations: dict[str, float] = {}
        self._t0 = time.monotonic()

    def elapsed_total(self) -> float:
        return time.monotonic() - self._t0

    @property
    def progress_callback(self) -> Callable[[dict[str, Any]], None] | None:
        return self._on_progress

    def emit(self, step: str, status: str = "running", *, phase: str = "audit", **extra: Any) -> None:
        if not self._on_progress:
            return
        now = time.monotonic()
        stage_start = self._starts.get(step, now)
        payload: dict[str, Any] = {
            "phase": phase,
            "step": step,
            "status": status,
            "elapsed_sec": round(now - self._t0, 2),
            "stage_timings": {k: round(v, 2) for k, v in self.durations.items()},
        }
        if status == "running":
            self._starts[step] = now
            payload["stage_elapsed_sec"] = 0.0
        elif status == "done":
            dur = now - stage_start
            self.durations[step] = dur
            payload["duration_sec"] = round(dur, 2)
            payload["stage_elapsed_sec"] = round(dur, 2)
            payload["stage_timings"] = {k: round(v, 2) for k, v in self.durations.items()}
        else:
            payload["stage_elapsed_sec"] = round(now - stage_start, 2)
        self._on_progress({**payload, **system_metrics(), **extra})

    def emit_dashboard(self, **extra: Any) -> None:
        if not self._on_progress:
            return
        self._on_progress({
            "phase": "dashboard",
            "elapsed_sec": round(time.monotonic() - self._t0, 2),
            **system_metrics(),
            **extra,
        })
