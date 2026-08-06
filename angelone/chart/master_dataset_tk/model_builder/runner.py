"""Background model training — mirrors ml_training/ws.py without WebSocket."""

from __future__ import annotations

import threading
from typing import Any, Callable

from chain_replay_ml.training.orchestrator import train_model

from ..build_service import chart_data_dir

ProgressCallback = Callable[[dict[str, Any]], None]
DoneCallback = Callable[[dict[str, Any]], None]


class ModelTrainingRunner:
    def __init__(self, chart_dir: str) -> None:
        self.chart_dir = chart_dir
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._lock = threading.Lock()

    @property
    def running(self) -> bool:
        t = self._thread
        return bool(t and t.is_alive())

    def cancel(self) -> None:
        self._cancel.set()

    def start(
        self,
        config: dict[str, Any],
        *,
        on_progress: ProgressCallback,
        on_done: DoneCallback,
    ) -> None:
        if self.running:
            raise RuntimeError("Training already running")
        self._cancel.clear()
        data_dir = chart_data_dir(self.chart_dir)

        def _worker() -> None:
            result: dict[str, Any]
            try:
                result = train_model(
                    data_dir=data_dir,
                    raw_config=config,
                    on_progress=on_progress,
                    cancel_check=self._cancel.is_set,
                )
                if self._cancel.is_set():
                    result = {"ok": False, "cancelled": True}
            except Exception as exc:
                result = {"ok": False, "error": str(exc)}
            on_done(result)

        self._thread = threading.Thread(target=_worker, name="tk-model-train", daemon=True)
        self._thread.start()
