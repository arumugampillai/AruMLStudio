"""Global research campaign execution — shared by Model Registry and Research Programs."""

from __future__ import annotations

import time
from typing import Any, Callable

from .build_progress_manager import BuildProgressManager, get_build_progress_manager
from .campaign_job_runner import CampaignJobRunner

ScheduleFn = Callable[[Callable[[], None], int], None]
Listener = Callable[[], None]


class ResearchCampaignCoordinator:
    """Runs campaign experiment jobs, chains auto-run, and publishes global progress."""

    def __init__(self, progress_manager: BuildProgressManager | None = None) -> None:
        self._progress = progress_manager or get_build_progress_manager()
        self._runner = CampaignJobRunner()
        self._schedule: ScheduleFn | None = None
        self._listeners: list[Listener] = []
        self._data_dir: str | None = None
        self._watched_campaigns: set[str] = set()
        self._campaign_meta: dict[str, dict[str, str]] = {}
        self._job_started_at: float | None = None
        self._progress_active = False
        self._kick_blocked: set[str] = set()

    @property
    def running(self) -> bool:
        return self._runner.running

    @property
    def active_campaign_id(self) -> str | None:
        return self._runner.campaign_id

    def bind_ui(self, schedule: ScheduleFn) -> None:
        self._schedule = schedule

    def subscribe(self, listener: Listener) -> None:
        if listener not in self._listeners:
            self._listeners.append(listener)

    def unsubscribe(self, listener: Listener) -> None:
        if listener in self._listeners:
            self._listeners.remove(listener)

    def watch_campaign(
        self,
        campaign_id: str,
        *,
        model_id: str = "",
        program_name: str = "",
        campaign_name: str = "",
        run_id: str = "",
    ) -> None:
        cid = str(campaign_id or "").strip()
        if not cid:
            return
        self._watched_campaigns.add(cid)
        meta = dict(self._campaign_meta.get(cid) or {})
        if model_id:
            meta["model_id"] = model_id
        if program_name:
            meta["program_name"] = program_name
        if campaign_name:
            meta["campaign_name"] = campaign_name
        if run_id:
            meta["run_id"] = run_id
        self._campaign_meta[cid] = meta

    def watch_program_run(self, run: dict[str, Any], *, model_id: str = "") -> None:
        manifest = run.get("manifest") or {}
        program_name = str(manifest.get("program_name") or "")
        run_id = str(run.get("run_id") or "")
        for entry in manifest.get("campaigns") or []:
            if not isinstance(entry, dict):
                continue
            self.watch_campaign(
                str(entry.get("campaign_id") or ""),
                model_id=model_id or str(run.get("model_id") or ""),
                program_name=program_name,
                campaign_name=str(entry.get("name") or ""),
                run_id=run_id,
            )

    def start_program_on_model(
        self,
        data_dir: str,
        *,
        model_id: str,
        program_id: str,
        research_report_id: str | None = None,
    ) -> dict[str, Any]:
        from chain_replay_ml.fold_research import start_program_on_model as _start

        self._data_dir = data_dir
        out = _start(
            data_dir,
            model_id=model_id,
            program_id=program_id,
            research_report_id=research_report_id,
        )
        if not out.get("ok"):
            return out

        run = out.get("run") or {}
        self.watch_program_run(run, model_id=model_id)
        manifest = run.get("manifest") or {}
        program_name = str(manifest.get("program_name") or program_id)
        first = out.get("first_campaign") or {}
        cid = str(first.get("campaign_id") or "")
        kick: dict[str, Any] = {"ok": False, "error": "no campaign started"}
        if cid:
            kick = self.kick_campaign(data_dir, cid)
        out["kick"] = kick

        if kick.get("ok"):
            self._begin_progress(program_name=program_name, model_id=model_id, campaign_id=cid)
        elif not self._runner.running:
            self._progress.publish({
                "status": "failed",
                "_done": True,
                "job_kind": "research_campaign",
                "job_title": "Research Program",
                "message": kick.get("error") or "Failed to start first experiment",
            })
        return out

    def kick_campaign(self, data_dir: str, campaign_id: str) -> dict[str, Any]:
        """Queue and run the next pending experiment for a running campaign."""
        if self._runner.running:
            return {"ok": False, "error": "A research job is already running"}

        from chain_replay_ml.fold_research.campaign_scheduler import run_next_campaign_experiment
        from chain_replay_ml.fold_research.research_program import get_research_campaign, update_research_campaign

        self._data_dir = data_dir
        self.watch_campaign(campaign_id)

        camp = get_research_campaign(data_dir, campaign_id)
        if not camp:
            return {"ok": False, "error": "campaign not found"}
        if camp.get("status") != "running":
            return {"ok": False, "error": f"campaign is not running (status={camp.get('status')})"}

        memory = dict(camp.get("memory") or {})
        pending = str(memory.get("pending_job_id") or "").strip()
        if pending:
            return self.execute_job(data_dir, pending, campaign_id)

        nxt = run_next_campaign_experiment(data_dir, campaign_id)
        if not nxt.get("ok"):
            self._kick_blocked.add(campaign_id)
            return nxt

        job_id = str((nxt.get("job") or {}).get("job_id") or "")
        if not job_id:
            return {"ok": False, "error": "no experiment job created"}

        memory["pending_job_id"] = job_id
        memory["active_job_id"] = job_id
        update_research_campaign(data_dir, campaign_id, memory=memory)
        self._kick_blocked.discard(campaign_id)
        return self.execute_job(data_dir, job_id, campaign_id)

    def execute_job(self, data_dir: str, job_id: str, campaign_id: str) -> dict[str, Any]:
        if self._runner.running:
            return {"ok": False, "error": "A research job is already running"}

        from chain_replay_ml.fold_research.research_program import get_research_campaign, update_research_campaign

        self._data_dir = data_dir
        self.watch_campaign(campaign_id)

        camp = get_research_campaign(data_dir, campaign_id)
        if camp:
            mem = dict(camp.get("memory") or {})
            mem.pop("pending_job_id", None)
            mem["active_job_id"] = job_id
            update_research_campaign(data_dir, campaign_id, memory=mem)
            meta = self._campaign_meta.get(campaign_id) or {}
            if not meta.get("program_name"):
                pid = str(camp.get("program_id") or "")
                if pid:
                    from chain_replay_ml.fold_research import get_research_program

                    prog = get_research_program(data_dir, pid)
                    if prog:
                        meta["program_name"] = str(prog.get("name") or pid)
            if not meta.get("campaign_name"):
                meta["campaign_name"] = str(camp.get("name") or "")
            if not meta.get("model_id"):
                meta["model_id"] = str((camp.get("memory") or {}).get("model_id") or "")
            self._campaign_meta[campaign_id] = meta

        meta = self._campaign_meta.get(campaign_id) or {}
        self._begin_progress(
            program_name=str(meta.get("program_name") or "Research Program"),
            model_id=str(meta.get("model_id") or ""),
            campaign_id=campaign_id,
            campaign_name=str(meta.get("campaign_name") or ""),
        )
        self._job_started_at = time.time()

        return self._runner.start_job(
            data_dir,
            job_id,
            campaign_id=campaign_id,
            on_done=lambda result: self._on_job_done(campaign_id, result),
        )

    def tick(self, data_dir: str) -> None:
        """Poll job progress and resume pending work (call from app main loop)."""
        self._data_dir = data_dir
        if self._runner.running:
            self._poll_running_job(data_dir)
            return
        self._scan_for_pending_work(data_dir)

    def _begin_progress(
        self,
        *,
        program_name: str,
        model_id: str = "",
        campaign_id: str = "",
        campaign_name: str = "",
    ) -> None:
        title = program_name
        if model_id:
            title = f"{program_name} · {model_id}"
        self._progress.begin_job("research_campaign", title=title)
        self._progress_active = True
        detail_parts = [p for p in (campaign_name, campaign_id[:8] if campaign_id else "") if p]
        self._progress.publish({
            "status": "running",
            "job_kind": "research_campaign",
            "job_title": title,
            "message": "Starting research experiment…",
            "stage_name": " · ".join(detail_parts) if detail_parts else "Research experiment",
            "percent": 0.0,
        })

    def _poll_running_job(self, data_dir: str) -> None:
        job_id = self._runner.job_id
        campaign_id = self._runner.campaign_id or ""
        if not job_id:
            return

        from chain_replay_ml.fold_research import get_experiment_job

        doc = get_experiment_job(data_dir, job_id)
        if not doc:
            return

        progress = doc.get("progress") if isinstance(doc.get("progress"), dict) else {}
        step_idx = int(progress.get("step_index") or 0)
        total = int(progress.get("total_steps") or 1)
        pct = round(100.0 * step_idx / max(1, total), 1)
        msg = str(progress.get("message") or doc.get("current_step") or "Running")
        meta = self._campaign_meta.get(campaign_id) or {}
        title = meta.get("program_name") or "Research Program"
        model_id = meta.get("model_id") or ""
        if model_id:
            title = f"{title} · {model_id}"

        elapsed = None
        if self._job_started_at is not None:
            elapsed = time.time() - self._job_started_at

        self._progress.publish({
            "status": "running",
            "job_kind": "research_campaign",
            "job_title": title,
            "message": msg,
            "stage_name": msg,
            "percent": pct,
            "current": step_idx,
            "total": total,
            "elapsed_sec": elapsed,
            "substage": meta.get("campaign_name") or f"Job #{doc.get('job_number') or '—'}",
        })

    def _scan_for_pending_work(self, data_dir: str) -> None:
        from chain_replay_ml.fold_research.experiment_pipeline_store import ExperimentPipelineStore
        from chain_replay_ml.fold_research.program_execution_store import ProgramExecutionStore
        from chain_replay_ml.fold_research.research_program import get_research_campaign

        for cid in list(self._watched_campaigns):
            camp = get_research_campaign(data_dir, cid)
            if not camp or camp.get("status") != "running":
                continue
            memory = camp.get("memory") or {}
            pending = str(memory.get("pending_job_id") or "").strip()
            active = str(memory.get("active_job_id") or "").strip()
            if pending:
                self.execute_job(data_dir, pending, cid)
                return
            if active:
                with ExperimentPipelineStore(data_dir) as store:
                    job = store._load_job(active)
                if job and job.get("status") == "running":
                    self.execute_job(data_dir, active, cid)
                    return

        with ProgramExecutionStore(data_dir) as store:
            runs = store.list_runs(status="running", limit=10)
        for run in runs:
            self.watch_program_run(run, model_id=str(run.get("model_id") or ""))
            manifest = run.get("manifest") or {}
            checkpoint = run.get("checkpoint") or {}
            idx = int(checkpoint.get("current_campaign_index") or 0)
            campaigns = manifest.get("campaigns") or []
            if idx < len(campaigns):
                cid = str(campaigns[idx].get("campaign_id") or "")
                if cid:
                    camp = get_research_campaign(data_dir, cid)
                    if camp and camp.get("status") == "running":
                        memory = camp.get("memory") or {}
                        if memory.get("auto_run") and not memory.get("active_job_id"):
                            if cid in self._kick_blocked:
                                continue
                            kick = self.kick_campaign(data_dir, cid)
                            if kick.get("ok"):
                                return

    def _on_job_done(self, campaign_id: str, result: dict[str, Any]) -> None:
        self._runner.reset()
        data_dir = self._data_dir
        ok = bool(result.get("ok"))
        job = result.get("job") or {}
        job_num = job.get("job_number") or "—"
        verdict = ((job.get("results") or {}).get("verdict") or {}).get("verdict") or ""

        if self._progress_active:
            self._progress.publish({
                "status": "completed" if ok else "failed",
                "_done": True,
                "job_kind": "research_campaign",
                "job_title": "Research Program",
                "message": (
                    f"Experiment #{job_num} complete{f' — {verdict}' if verdict else ''}"
                    if ok
                    else str(result.get("error") or job.get("error") or "Research experiment failed")
                ),
                "percent": 100.0 if ok else 0.0,
            })
            self._progress_active = False

        def _continue() -> None:
            if not data_dir:
                self._notify_listeners()
                return

            next_job_id = ""
            sched = result.get("scheduler") or {}
            if sched.get("action") == "auto_queued":
                next_job_id = str((sched.get("next") or {}).get("job", {}).get("job_id") or "")

            if not next_job_id:
                from chain_replay_ml.fold_research.research_program import get_research_campaign

                camp = get_research_campaign(data_dir, campaign_id)
                memory = dict((camp or {}).get("memory") or {})
                if memory.get("auto_run"):
                    next_job_id = str(memory.get("pending_job_id") or "")

            if next_job_id and ok:
                from chain_replay_ml.fold_research.research_program import update_research_campaign

                camp = get_research_campaign(data_dir, campaign_id)
                memory = dict((camp or {}).get("memory") or {})
                memory["pending_job_id"] = next_job_id
                update_research_campaign(data_dir, campaign_id, memory=memory)
                self.execute_job(data_dir, next_job_id, campaign_id)
            else:
                self._scan_for_pending_work(data_dir)

            self._notify_listeners()

        self._schedule_main(_continue)

    def _schedule_main(self, fn: Callable[[], None]) -> None:
        if self._schedule is not None:
            self._schedule(fn, 0)
        else:
            fn()

    def _notify_listeners(self) -> None:
        for listener in list(self._listeners):
            try:
                listener()
            except Exception:
                pass


_coordinator: ResearchCampaignCoordinator | None = None


def get_research_campaign_coordinator() -> ResearchCampaignCoordinator:
    global _coordinator
    if _coordinator is None:
        _coordinator = ResearchCampaignCoordinator()
    return _coordinator
