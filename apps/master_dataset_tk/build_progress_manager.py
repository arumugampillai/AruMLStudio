"""Global build progress hub — event-driven, thread-safe, UI-agnostic."""

from __future__ import annotations

import queue
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .progress_panel import live_pipeline_payload

Subscriber = Callable[["BuildStatusSnapshot"], None]
CancelFn = Callable[[], None]

_STATUS_ICONS = {
    "idle": "⚪",
    "running": "🟢",
    "completed": "✅",
    "failed": "🔴",
    "cancelled": "🟡",
}


def _fmt_duration_mmss(sec: float | None) -> str:
    if sec is None:
        return "—"
    total = max(0, int(sec))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _fmt_num(n: int | float | None) -> str:
    if n is None:
        return "—"
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "—"


def _ram_label() -> str:
    try:
        import psutil

        proc = psutil.Process()
        used_gb = proc.memory_info().rss / (1024**3)
        total_gb = psutil.virtual_memory().total / (1024**3)
        return f"{used_gb:.1f}/{total_gb:.0f} GB"
    except Exception:
        return "—"


def compute_overall_percent(payload: dict[str, Any]) -> float:
    if payload.get("backfill_active") and payload.get("backfill_percent") is not None:
        try:
            return max(0.0, min(100.0, float(payload["backfill_percent"])))
        except (TypeError, ValueError):
            pass
    raw = payload.get("percent")
    if raw is not None:
        try:
            return max(0.0, min(100.0, float(raw)))
        except (TypeError, ValueError):
            pass
    stage = int(payload.get("stage") or 0)
    if stage <= 0:
        cur = int(payload.get("current") or 0)
        tot = int(payload.get("total") or 0)
        if tot > 0:
            return round(100.0 * cur / tot, 1)
        return 0.0
    total_stages = 8
    stage_frac = (stage - 1) / total_stages
    intra = 0.0
    sub_total = payload.get("sub_total")
    sub_current = payload.get("sub_current")
    if sub_total and sub_current is not None:
        try:
            intra = min(1.0, float(sub_current) / float(sub_total)) / total_stages
        except (TypeError, ValueError, ZeroDivisionError):
            intra = 0.0
    elif payload.get("substage_percent") is not None:
        try:
            intra = float(payload["substage_percent"]) / 100.0 / total_stages
        except (TypeError, ValueError):
            intra = 0.0
    return round(min(99.9, (stage_frac + intra) * 100), 1)


def _task_label(payload: dict[str, Any]) -> str:
    kind = str(payload.get("job_kind") or "dataset_build")
    if kind == "research_campaign":
        substage = str(payload.get("substage") or "").strip()
        if substage:
            return substage
        return str(payload.get("message") or payload.get("stage_name") or "Research experiment")
    fg = str(payload.get("feature_group_current") or "").strip()
    if fg:
        return f"Building {fg}"
    for key in ("stage_name", "message", "substage"):
        text = str(payload.get(key) or "").strip()
        if text:
            return text
    if kind == "registry_export":
        return "Exporting to registry"
    if kind == "feature_migration":
        return "Migrating features"
    return "Building dataset"


def _day_label(payload: dict[str, Any]) -> str:
    if payload.get("backfill_active") and payload.get("backfill_days_total"):
        cur = int(payload.get("backfill_days_current") or 0)
        tot = int(payload.get("backfill_days_total") or 0)
        return f"Backfill day {cur}/{tot}"
    idx = payload.get("source_day_index")
    tot = payload.get("source_day_total")
    if idx is not None and tot:
        return f"Day {int(idx)}/{int(tot)}"
    cur = payload.get("current")
    total = payload.get("total")
    if cur and total and int(total) > 1:
        return f"Day {int(cur)}/{int(total)}"
    td = str(payload.get("trading_day") or "").strip()
    if td:
        return td
    return "—"


def _samples_label(payload: dict[str, Any]) -> str:
    if payload.get("backfill_active") and payload.get("backfill_rows_total"):
        cur = int(payload.get("backfill_rows_current") or 0)
        tot = int(payload.get("backfill_rows_total") or 0)
        return f"{cur:,} / {tot:,} (backfill)"
    rows = payload.get("rows")
    total = payload.get("total")
    if rows is not None and total:
        try:
            return f"{int(rows):,} / {int(total):,}"
        except (TypeError, ValueError):
            pass
    if rows is not None:
        return f"{_fmt_num(rows)} samples"
    ticks = payload.get("ticks_in_memory")
    if ticks:
        return f"{_fmt_num(ticks)} ticks"
    sp = payload.get("sample_points")
    if sp:
        return f"{_fmt_num(sp)} points"
    return "—"


def _speed_label(payload: dict[str, Any], pl: dict[str, Any]) -> str:
    rate = pl.get("rows_per_sec")
    if rate:
        stage = int(payload.get("stage") or 0)
        unit = "steps/s" if stage == 8 else "rows/s"
        return f"{int(rate):,} {unit}"
    return "—"


@dataclass(frozen=True)
class BuildStatusSnapshot:
    active: bool = False
    status: str = "idle"
    job_kind: str = ""
    job_title: str = "Ready"
    task_label: str = "—"
    percent: float = 0.0
    elapsed_sec: float | None = None
    elapsed_label: str = "—"
    eta_label: str = "—"
    speed_label: str = "—"
    worker_count: int = 1
    day_label: str = "—"
    samples_label: str = "—"
    ram_label: str = "—"
    status_icon: str = "⚪"
    cancellable: bool = False
    message: str = ""

    @classmethod
    def idle(cls) -> BuildStatusSnapshot:
        return cls(
            active=False,
            status="idle",
            job_title="Ready",
            task_label="—",
            ram_label=_ram_label(),
        )


class BuildProgressManager:
    """Collects build events from background workers; subscribers update UI."""

    def __init__(self) -> None:
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._subscribers: list[Subscriber] = []
        self._cancel_fn: CancelFn | None = None
        self._snapshot = BuildStatusSnapshot.idle()
        self._last_payload: dict[str, Any] | None = None
        self._received_at: float = 0.0
        self._job_kind: str = ""

    @property
    def snapshot(self) -> BuildStatusSnapshot:
        return self._snapshot

    def subscribe(self, callback: Subscriber) -> None:
        if callback not in self._subscribers:
            self._subscribers.append(callback)
        callback(self._snapshot)

    def unsubscribe(self, callback: Subscriber) -> None:
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def begin_job(
        self,
        kind: str,
        *,
        title: str | None = None,
        cancel_fn: CancelFn | None = None,
    ) -> None:
        self._job_kind = str(kind or "dataset_build")
        self._cancel_fn = cancel_fn
        self.publish({
            "status": "running",
            "job_kind": self._job_kind,
            "job_title": title or "Building Dataset",
            "message": "Starting…",
            "stage": 1,
            "stage_name": "Load Database",
        })

    def register_cancel(self, cancel_fn: CancelFn | None) -> None:
        self._cancel_fn = cancel_fn

    def request_cancel(self) -> None:
        if self._cancel_fn is not None:
            self._cancel_fn()

    def publish(self, payload: dict[str, Any]) -> None:
        """Thread-safe — call from background build workers."""
        self._queue.put(dict(payload))

    def tick(self) -> None:
        """Drain queued events and bump live elapsed time (main/UI thread only)."""
        changed = False
        while True:
            try:
                payload = self._queue.get_nowait()
            except queue.Empty:
                break
            self._apply_payload(payload)
            changed = True

        if self._snapshot.status == "running" and self._last_payload:
            live = live_pipeline_payload(self._last_payload, self._received_at)
            snap = self._snapshot_from_payload(live)
            if snap != self._snapshot:
                self._snapshot = snap
                changed = True

        if changed:
            self._notify()

    def _apply_payload(self, payload: dict[str, Any]) -> None:
        if payload.pop("_done", False):
            status = str(payload.get("status") or "completed").lower()
            if status not in ("completed", "failed", "cancelled"):
                status = "completed"
            payload["status"] = status
            self._cancel_fn = None
        if payload.get("job_kind") is None and self._job_kind:
            payload = {**payload, "job_kind": self._job_kind}
        self._last_payload = dict(payload)
        self._received_at = time.time()
        self._snapshot = self._snapshot_from_payload(payload)
        if self._snapshot.status in ("completed", "failed", "cancelled", "idle"):
            self._cancel_fn = None

    def _snapshot_from_payload(self, payload: dict[str, Any]) -> BuildStatusSnapshot:
        status = str(payload.get("status") or "idle").lower()
        if status not in _STATUS_ICONS:
            status = "running" if status == "running" else "idle"
        pl = payload.get("pipeline") if isinstance(payload.get("pipeline"), dict) else {}
        elapsed_sec = pl.get("total_elapsed_sec")
        if elapsed_sec is None and payload.get("elapsed_sec") is not None:
            elapsed_sec = payload.get("elapsed_sec")
        eta_label = str(pl.get("eta_label") or "—")
        job_kind = str(payload.get("job_kind") or self._job_kind or "dataset_build")
        job_title = str(
            payload.get("job_title")
            or {
                "dataset_build": "Building Dataset",
                "registry_export": "Exporting Dataset",
                "analysis_dataset_build": "Analysis Dataset",
                "research_campaign": "Research Program",
                "model_lab_prediction": "Prediction Dataset",
                "feature_migration": "Feature Migration",
            }.get(job_kind, "Building Dataset")
        )
        workers = payload.get("worker_count")
        try:
            worker_count = max(1, int(workers)) if workers is not None else 1
        except (TypeError, ValueError):
            worker_count = 1
        active = status == "running"
        return BuildStatusSnapshot(
            active=active,
            status=status if active or status in _STATUS_ICONS else "idle",
            job_kind=job_kind,
            job_title=job_title if active else {
                "completed": {
                    "dataset_build": "Build complete",
                    "registry_export": "Export complete",
                    "research_campaign": "Research complete",
                    "model_lab_prediction": "Prediction ready",
                    "feature_migration": "Migration complete",
                }.get(job_kind, "Build complete"),
                "failed": {
                    "dataset_build": "Build failed",
                    "registry_export": "Export failed",
                    "research_campaign": "Research failed",
                    "model_lab_prediction": "Prediction failed",
                    "feature_migration": "Migration failed",
                }.get(job_kind, "Build failed"),
                "cancelled": "Build cancelled",
            }.get(status, "Ready"),
            task_label=_task_label(payload) if active else "—",
            percent=compute_overall_percent(payload) if active else (
                100.0 if status == "completed" else 0.0
            ),
            elapsed_sec=float(elapsed_sec) if elapsed_sec is not None else None,
            elapsed_label=str(pl.get("total_elapsed_label") or _fmt_duration_mmss(elapsed_sec)),
            eta_label=eta_label if active else "—",
            speed_label=_speed_label(payload, pl) if active else "—",
            worker_count=worker_count,
            day_label=_day_label(payload) if active else "—",
            samples_label=_samples_label(payload) if active else "—",
            ram_label=_ram_label(),
            status_icon=_STATUS_ICONS.get(status, "⚪"),
            cancellable=active and self._cancel_fn is not None,
            message=str(payload.get("message") or ""),
        )

    def _notify(self) -> None:
        snap = self._snapshot
        for cb in list(self._subscribers):
            try:
                cb(snap)
            except Exception:
                pass


_manager: BuildProgressManager | None = None


def get_build_progress_manager() -> BuildProgressManager:
    global _manager
    if _manager is None:
        _manager = BuildProgressManager()
    return _manager
