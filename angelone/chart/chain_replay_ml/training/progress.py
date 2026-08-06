"""Training stage timing for WebSocket progress events."""

from __future__ import annotations

import time
import threading
from collections.abc import Callable
from typing import Any

from chain_replay_ml.dataset_builder.audit_progress import system_metrics

TRAIN_STEP_ORDER = [
    ("preparing_dataset", "Preparing Dataset"),
    ("preparing_matrix", "Preparing Matrix"),
    ("training", "Training"),
    ("evaluation", "Evaluation"),
    ("saving", "Saving"),
]


class TrainStageTracker:
    def __init__(self, on_progress: Callable[[dict[str, Any]], None] | None) -> None:
        self._on_progress = on_progress
        self._starts: dict[str, float] = {}
        self.durations: dict[str, float] = {}
        self._t0 = time.monotonic()
        self.dashboard: dict[str, Any] = {}
        self._lock = threading.Lock()

    def elapsed_total(self) -> float:
        return time.monotonic() - self._t0

    def emit(self, step: str, status: str = "running", **extra: Any) -> None:
        if not self._on_progress:
            return
        now = time.monotonic()
        with self._lock:
            stage_start = self._starts.get(step, now)
        payload: dict[str, Any] = {
            "phase": "train",
            "step": step,
            "status": status,
            "elapsed_sec": round(now - self._t0, 2),
            "stage_timings": {},
            **system_metrics(),
        }
        with self._lock:
            payload["stage_timings"] = {k: round(v, 2) for k, v in self.durations.items()}
        if status == "running":
            with self._lock:
                self._starts[step] = now
            payload["stage_elapsed_sec"] = 0.0
        elif status == "done":
            dur = now - stage_start
            with self._lock:
                self.durations[step] = dur
            payload["duration_sec"] = round(dur, 2)
            payload["stage_elapsed_sec"] = round(dur, 2)
        else:
            payload["stage_elapsed_sec"] = round(now - stage_start, 2)
        with self._lock:
            dashboard = dict(self.dashboard)
        self._on_progress({**payload, **dashboard, **extra})

    def update_dashboard(self, **extra: Any) -> None:
        with self._lock:
            self.dashboard.update(extra)
            dashboard = dict(self.dashboard)
        if not self._on_progress:
            return
        self._on_progress({
            "phase": "dashboard",
            "elapsed_sec": round(time.monotonic() - self._t0, 2),
            **system_metrics(),
            **dashboard,
            **extra,
        })
