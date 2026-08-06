"""Background campaign job execution for Research Programs UI."""

from __future__ import annotations

import threading
from typing import Any, Callable

DoneCallback = Callable[[dict[str, Any]], None]


class CampaignJobRunner:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._job_id: str | None = None
        self._campaign_id: str | None = None

    @property
    def running(self) -> bool:
        thread = self._thread
        return bool(thread and thread.is_alive())

    @property
    def job_id(self) -> str | None:
        return self._job_id

    @property
    def campaign_id(self) -> str | None:
        return self._campaign_id

    def reset(self) -> None:
        self._thread = None
        self._job_id = None
        self._campaign_id = None

    def start_job(
        self,
        data_dir: str,
        job_id: str,
        *,
        campaign_id: str,
        on_done: DoneCallback,
    ) -> dict[str, Any]:
        if self.running:
            return {"ok": False, "error": "A campaign job is already running"}

        from chain_replay_ml.fold_research.experiment_pipeline import execute_job_pipeline

        self._job_id = job_id
        self._campaign_id = campaign_id

        def _worker() -> None:
            result = execute_job_pipeline(data_dir, job_id)
            on_done(result)

        self._thread = threading.Thread(
            target=_worker,
            name="tk-campaign-job",
            daemon=True,
        )
        self._thread.start()
        return {"ok": True, "job_id": job_id}

    def start_next(
        self,
        data_dir: str,
        campaign_id: str,
        *,
        on_done: DoneCallback,
    ) -> dict[str, Any]:
        if self.running:
            return {"ok": False, "error": "A campaign job is already running"}

        from chain_replay_ml.fold_research.campaign_scheduler import run_next_campaign_experiment

        created = run_next_campaign_experiment(data_dir, campaign_id)
        if not created.get("ok"):
            return created

        job = created.get("job") or {}
        job_id = str(job.get("job_id") or "")
        return self.start_job(data_dir, job_id, campaign_id=campaign_id, on_done=on_done)
