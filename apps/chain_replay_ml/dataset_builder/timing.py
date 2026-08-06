"""Live per-stage timing for the dataset build pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .progress import STAGE_NAMES, TOTAL_STAGES


def _fmt_duration(sec: float | None) -> str | None:
    if sec is None or sec < 0:
        return None
    if sec >= 3600:
        h = int(sec // 3600)
        m = int((sec % 3600) // 60)
        s = sec % 60
        return f"{h}h {m}m"
    if sec >= 60:
        m = int(sec // 60)
        s = sec % 60
        return f"{m}m {s:04.1f}s"
    return f"{sec:.2f} s"


@dataclass
class SubstageTiming:
    id: str
    label: str
    status: str = "waiting"
    started_at: float | None = None
    ended_at: float | None = None

    def elapsed(self, now: float) -> float | None:
        if self.started_at is None:
            return None
        end = self.ended_at if self.ended_at is not None else now
        return max(0.0, end - self.started_at)


@dataclass
class StageTiming:
    stage: int
    name: str
    status: str = "waiting"
    started_at: float | None = None
    ended_at: float | None = None
    progress_current: int = 0
    progress_total: int = 0
    progress_unit: str = "rows"
    substages: list[SubstageTiming] = field(default_factory=list)

    def elapsed(self, now: float) -> float | None:
        if self.started_at is None:
            return None
        end = self.ended_at if self.ended_at is not None else now
        return max(0.0, end - self.started_at)


class PipelineTimer:
    def __init__(self, feature_group_labels: list[tuple[str, str]] | None = None) -> None:
        self._t0 = time.monotonic()
        self._stages: dict[int, StageTiming] = {
            i: StageTiming(i, STAGE_NAMES.get(i, f"Stage {i}"))
            for i in range(1, TOTAL_STAGES + 1)
        }
        labels = feature_group_labels or []
        self._stages[6].substages = [
            SubstageTiming(id=gid, label=label) for gid, label in labels
        ]
        self._rows_at_stage4_start = 0
        self._stage4_started_at: float | None = None
        self._rows_at_row_build_start = 0

    def start_stage(self, stage: int, *, rows: int = 0) -> None:
        now = time.monotonic()
        for st in self._stages.values():
            if st.status == "running" and st.stage != stage:
                self.end_stage(st.stage)
        cur = self._stages[stage]
        if cur.status != "done":
            cur.status = "running"
            if cur.started_at is None:
                cur.started_at = now
        if stage == 4:
            self._stage4_started_at = now
            self._rows_at_row_build_start = max(0, int(rows))
            self._rows_at_stage4_start = self._rows_at_row_build_start
        elif stage == 6 and self._stages[4].status == "skipped" and self._stages[5].status == "skipped":
            if self._stage4_started_at is None:
                self._stage4_started_at = now
                self._rows_at_row_build_start = max(0, int(rows))

    def set_stage_name(self, stage: int, name: str | None) -> None:
        """Override a stage's display name (e.g. backfill borrowing stage 1's
        timer slot), or pass None to restore the default STAGE_NAMES label."""
        self._stages[stage].name = name if name else STAGE_NAMES.get(stage, f"Stage {stage}")

    def end_stage(self, stage: int) -> None:
        now = time.monotonic()
        cur = self._stages[stage]
        if cur.ended_at is None:
            cur.ended_at = now
        cur.status = "done"

    def skip_stage(self, stage: int) -> None:
        now = time.monotonic()
        cur = self._stages[stage]
        if cur.started_at is None:
            cur.started_at = now
        if cur.ended_at is None:
            cur.ended_at = now
        cur.status = "skipped"

    def set_stage_progress(
        self,
        stage: int,
        *,
        current: int,
        total: int,
        unit: str = "rows",
    ) -> None:
        cur = self._stages[stage]
        cur.progress_current = max(0, int(current))
        cur.progress_total = max(0, int(total))
        cur.progress_unit = unit

    def start_substage(self, stage: int, sub_id: str) -> None:
        now = time.monotonic()
        cur = self._stages[stage]
        for sub in cur.substages:
            if sub.status == "running" and sub.id != sub_id:
                self.end_substage(stage, sub.id)
        for sub in cur.substages:
            if sub.id == sub_id:
                sub.status = "running"
                if sub.started_at is None:
                    sub.started_at = now
                return

    def end_substage(self, stage: int, sub_id: str) -> None:
        now = time.monotonic()
        for sub in self._stages[stage].substages:
            if sub.id == sub_id and sub.ended_at is None:
                sub.ended_at = now
                sub.status = "done"

    def snapshot(
        self,
        *,
        rows: int = 0,
        estimated_total_rows: int = 0,
    ) -> dict[str, Any]:
        now = time.monotonic()
        total_elapsed = now - self._t0
        stages_out: list[dict[str, Any]] = []
        substages_out: list[dict[str, Any]] = []

        for i in range(1, TOTAL_STAGES + 1):
            st = self._stages[i]
            elapsed = st.elapsed(now)
            entry: dict[str, Any] = {
                "id": st.stage,
                "name": st.name,
                "status": st.status,
                "elapsed_sec": round(elapsed, 2) if elapsed is not None else None,
                "elapsed_label": _fmt_duration(elapsed),
            }
            if st.progress_total > 0 and st.status in ("running", "done", "skipped"):
                entry["progress_current"] = st.progress_current
                entry["progress_total"] = st.progress_total
                entry["progress_unit"] = st.progress_unit
            stages_out.append(entry)

            if i == 6:
                for sub in st.substages:
                    sub_elapsed = sub.elapsed(now)
                    substages_out.append({
                        "parent_stage": 6,
                        "id": sub.id,
                        "label": sub.label,
                        "status": sub.status,
                        "elapsed_sec": round(sub_elapsed, 2) if sub_elapsed is not None else None,
                        "elapsed_label": _fmt_duration(sub_elapsed),
                    })

        rows_per_sec: float | None = None
        eta_sec: float | None = None
        active_stage: dict[str, Any] | None = None
        row_building = (
            self._stages[4].status == "running"
            or self._stages[5].status == "running"
            or self._stages[6].status == "running"
        )
        if row_building and self._stage4_started_at is not None:
            base_rows = self._rows_at_row_build_start
            if rows > base_rows:
                dt = now - self._stage4_started_at
                if dt > 0.1:
                    rows_per_sec = (rows - base_rows) / dt
        elif self._stages[4].status == "done":
            st4 = self._stages[4]
            elapsed = st4.elapsed(now)
            if elapsed and elapsed > 0 and st4.progress_current:
                rows_per_sec = st4.progress_current / elapsed

        for stage_id in (7, 8):
            st = self._stages[stage_id]
            if st.status != "running":
                continue
            elapsed = st.elapsed(now)
            if st.progress_total > 0 and st.progress_current > 0 and elapsed and elapsed > 0.1:
                rate = st.progress_current / elapsed
                if st.progress_unit == "steps":
                    rows_per_sec = rate
                    if st.progress_total > st.progress_current:
                        eta_sec = (st.progress_total - st.progress_current) / rate
                else:
                    rows_per_sec = rate
                    if st.progress_total > st.progress_current:
                        eta_sec = (st.progress_total - st.progress_current) / rate
            active_stage = {
                "id": stage_id,
                "name": st.name,
                "elapsed_sec": round(elapsed, 2) if elapsed is not None else None,
                "elapsed_label": _fmt_duration(elapsed),
                "progress_current": st.progress_current,
                "progress_total": st.progress_total,
                "progress_unit": st.progress_unit,
            }
            break

        if rows_per_sec and rows_per_sec > 0 and eta_sec is None and estimated_total_rows > rows and row_building:
            eta_sec = (estimated_total_rows - rows) / rows_per_sec

        done_stages = [
            s for s in stages_out
            if s.get("status") == "done" and (s.get("elapsed_sec") or 0) > 0
        ]
        stage_time_sum = sum(float(s.get("elapsed_sec") or 0) for s in done_stages)
        slowest: dict[str, Any] | None = None
        if done_stages:
            slowest_entry = max(done_stages, key=lambda s: float(s.get("elapsed_sec") or 0))
            for entry in stages_out:
                elapsed = float(entry.get("elapsed_sec") or 0)
                if entry.get("status") == "done" and stage_time_sum > 0 and elapsed > 0:
                    entry["pct"] = round(100.0 * elapsed / stage_time_sum)
                if entry.get("id") == slowest_entry.get("id"):
                    entry["is_slowest"] = True
            slowest = {
                "id": slowest_entry.get("id"),
                "name": slowest_entry.get("name"),
                "elapsed_sec": slowest_entry.get("elapsed_sec"),
                "elapsed_label": slowest_entry.get("elapsed_label"),
            }

        build_complete = self._stages[8].status == "done"
        avg_rows_per_sec: float | None = None
        if build_complete and rows > 0 and total_elapsed > 0.1:
            avg_rows_per_sec = rows / total_elapsed

        if active_stage is None:
            for entry in stages_out:
                if entry.get("status") == "running":
                    active_stage = {
                        "id": entry.get("id"),
                        "name": entry.get("name"),
                        "elapsed_sec": entry.get("elapsed_sec"),
                        "elapsed_label": entry.get("elapsed_label"),
                        "progress_current": entry.get("progress_current"),
                        "progress_total": entry.get("progress_total"),
                        "progress_unit": entry.get("progress_unit"),
                    }
                    break

        return {
            "stages": stages_out,
            "substages": substages_out,
            "total_elapsed_sec": round(total_elapsed, 2),
            "total_elapsed_label": _fmt_duration(total_elapsed),
            "rows_per_sec": round(rows_per_sec) if rows_per_sec else None,
            "avg_rows_per_sec": round(avg_rows_per_sec) if avg_rows_per_sec else None,
            "eta_sec": round(eta_sec) if eta_sec else None,
            "eta_label": _fmt_duration(eta_sec) if eta_sec else None,
            "estimated_total_rows": estimated_total_rows,
            "current_rows": rows,
            "slowest_stage": slowest,
            "active_stage": active_stage,
            "build_complete": build_complete,
        }
