"""Background experiment job execution for the Experiment Planner UI."""

from __future__ import annotations

import threading
from typing import Any, Callable

DoneCallback = Callable[[dict[str, Any]], None]


class ExperimentJobRunner:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._job_id: str | None = None

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive())

    @property
    def job_id(self) -> str | None:
        return self._job_id

    def reset(self) -> None:
        self._thread = None
        self._job_id = None

    def start(
        self,
        data_dir: str,
        template_id: str,
        *,
        overrides: dict[str, Any] | None = None,
        on_done: DoneCallback,
    ) -> dict[str, Any]:
        if self.running:
            return {"ok": False, "error": "An experiment job is already running"}

        from chain_replay_ml.fold_research.experiment_pipeline import (
            create_template_job,
            execute_job_pipeline,
        )

        created = create_template_job(data_dir, template_id, overrides=overrides)
        if not created.get("ok"):
            return created

        job = created.get("job") or {}
        self._job_id = str(job.get("job_id") or "")

        def _worker() -> None:
            result = execute_job_pipeline(data_dir, self._job_id or "")
            on_done(result)

        self._thread = threading.Thread(
            target=_worker,
            name="tk-experiment-job",
            daemon=True,
        )
        self._thread.start()
        return created
