"""Measure GIL hold times and main-thread scheduling lag during dataset builds.

Diagnostic only — off by default (heavy sys.setprofile overhead when enabled).
Enable via Settings, or ARUNEO_GIL_MONITOR=1 for a single session.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

YIELD_GAP_MS = 2.0
HOLD_THRESHOLD_MS = 5.0
LONG_HOLD_MS = 50.0
PROBE_INTERVAL_S = 0.001


def gil_monitor_disabled(chart_dir: str | None) -> bool:
    """User preference: diagnostics off (default True)."""
    if not chart_dir:
        return True
    try:
        from .build_config_prefs import load_build_config_prefs

        studio = (load_build_config_prefs(chart_dir) or {}).get("studio") or {}
        return bool(studio.get("disable_gil_monitor", True))
    except Exception:
        return True


def gil_monitor_enabled(chart_dir: str | None = None) -> bool:
    env = os.environ.get("ARUMLSTUDIO_GIL_MONITOR") or os.environ.get("ARUNEO_GIL_MONITOR")
    if env is not None:
        return env.strip().lower() not in ("0", "false", "no", "off")
    if chart_dir and gil_monitor_disabled(chart_dir):
        return False
    return False


def _frame_label(frame: Any) -> str:
    code = frame.f_code
    return f"{code.co_filename}:{code.co_name}:{frame.f_lineno}"


def _stack_snippet(thread_id: int, *, max_frames: int = 10) -> str:
    frame = sys._current_frames().get(thread_id)
    if frame is None:
        return "<thread not in Python>"
    lines = traceback.format_stack(frame)
    return "".join(lines[-max_frames:]).strip()


@dataclass
class HoldEvent:
    hold_ms: float
    stack: str


@dataclass
class BurstEvent:
    duration_ms: float
    location: str


class WorkerBurstTracker:
    """Track longest continuous Python execution bursts on the build worker thread."""

    def __init__(self) -> None:
        self._thread_id = threading.get_ident()
        self._burst_start: float | None = None
        self._burst_location = ""
        self._last_event: float | None = None
        self.longest_burst_ms = 0.0
        self.longest_burst_location = ""
        self.bursts_over_threshold: list[BurstEvent] = []
        self.yield_count = 0
        self._active = False

    def activate(self) -> None:
        self._active = True
        self._thread_id = threading.get_ident()
        sys.setprofile(self._profile)

    def deactivate(self) -> None:
        if not self._active:
            return
        sys.setprofile(None)
        self._close_burst(time.perf_counter())
        self._active = False

    def _profile(self, frame: Any, event: str, arg: Any) -> Callable[..., Any] | None:
        if threading.get_ident() != self._thread_id:
            return self._profile
        if event not in ("call", "c_call"):
            return self._profile
        now = time.perf_counter()
        if self._last_event is not None:
            gap_ms = (now - self._last_event) * 1000.0
            if gap_ms >= YIELD_GAP_MS:
                self._close_burst(now)
                self.yield_count += 1
                self._burst_start = now
                self._burst_location = _frame_label(frame)
        if self._burst_start is None:
            self._burst_start = now
            self._burst_location = _frame_label(frame)
        self._last_event = now
        return self._profile

    def _close_burst(self, now: float) -> None:
        if self._burst_start is None:
            return
        dur_ms = (now - self._burst_start) * 1000.0
        loc = self._burst_location
        if dur_ms > self.longest_burst_ms:
            self.longest_burst_ms = dur_ms
            self.longest_burst_location = loc
        if dur_ms >= LONG_HOLD_MS:
            self.bursts_over_threshold.append(BurstEvent(duration_ms=dur_ms, location=loc))
        self._burst_start = None

    def to_dict(self) -> dict[str, Any]:
        bursts = sorted(self.bursts_over_threshold, key=lambda b: b.duration_ms, reverse=True)
        return {
            "longest_continuous_python_ms": round(self.longest_burst_ms, 2),
            "longest_burst_location": self.longest_burst_location,
            "voluntary_yield_gaps_detected": self.yield_count,
            "bursts_over_50ms": [
                {"duration_ms": round(b.duration_ms, 2), "location": b.location}
                for b in bursts[:25]
            ],
        }


class GILHoldProbe:
    """Estimate GIL hold duration by measuring probe-thread scheduling overruns."""

    def __init__(self) -> None:
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._watch_id: int | None = None
        self._holds_ms: list[float] = []
        self._long_holds: list[HoldEvent] = []
        self.longest_hold_ms = 0.0

    def start(self, *, watch_thread_id: int) -> None:
        self._watch_id = watch_thread_id
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gil-hold-probe", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            t0 = time.perf_counter()
            time.sleep(PROBE_INTERVAL_S)
            actual = time.perf_counter() - t0
            overrun_ms = max(0.0, (actual - PROBE_INTERVAL_S) * 1000.0)
            if overrun_ms < HOLD_THRESHOLD_MS:
                continue
            self._holds_ms.append(overrun_ms)
            if overrun_ms > self.longest_hold_ms:
                self.longest_hold_ms = overrun_ms
            if overrun_ms >= LONG_HOLD_MS and self._watch_id is not None:
                self._long_holds.append(
                    HoldEvent(
                        hold_ms=overrun_ms,
                        stack=_stack_snippet(self._watch_id),
                    )
                )

    def to_dict(self) -> dict[str, Any]:
        holds = sorted(self._holds_ms)
        long_holds = sorted(self._long_holds, key=lambda h: h.hold_ms, reverse=True)

        def _pct(p: float) -> float | None:
            if not holds:
                return None
            idx = min(len(holds) - 1, int(p * len(holds)))
            return round(holds[idx], 2)

        return {
            "probe_interval_ms": PROBE_INTERVAL_S * 1000.0,
            "hold_samples_over_5ms": len(holds),
            "longest_estimated_gil_hold_ms": round(self.longest_hold_ms, 2),
            "estimated_hold_p50_ms": _pct(0.50),
            "estimated_hold_p95_ms": _pct(0.95),
            "estimated_hold_p99_ms": _pct(0.99),
            "holds_over_50ms": [
                {"hold_ms": round(h.hold_ms, 2), "worker_stack": h.stack}
                for h in long_holds[:15]
            ],
        }


class MainThreadLagTracker:
    """Measure gaps between Tk poll callbacks (proxy for event-loop starvation)."""

    _instance: MainThreadLagTracker | None = None

    def __init__(self) -> None:
        self._last_tick: float | None = None
        self._expected_ms = 200.0
        self.gaps_ms: list[float] = []
        self.longest_gap_ms = 0.0

    @classmethod
    def instance(cls) -> MainThreadLagTracker:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def reset(self, *, expected_poll_ms: float = 200.0) -> None:
        self._expected_ms = expected_poll_ms
        self._last_tick = None
        self.gaps_ms.clear()
        self.longest_gap_ms = 0.0

    def tick(self) -> None:
        now = time.perf_counter()
        if self._last_tick is not None:
            gap_ms = (now - self._last_tick) * 1000.0
            overrun = gap_ms - self._expected_ms
            if overrun >= HOLD_THRESHOLD_MS:
                self.gaps_ms.append(gap_ms)
                if gap_ms > self.longest_gap_ms:
                    self.longest_gap_ms = gap_ms
        self._last_tick = now

    def to_dict(self) -> dict[str, Any]:
        gaps = sorted(self.gaps_ms)
        return {
            "expected_poll_interval_ms": self._expected_ms,
            "longest_poll_gap_ms": round(self.longest_gap_ms, 2),
            "poll_gaps_over_50ms": sum(1 for g in gaps if g >= LONG_HOLD_MS),
            "poll_gap_p95_ms": round(gaps[int(0.95 * (len(gaps) - 1))], 2) if gaps else None,
        }


class GILBuildSession:
    """Context manager for a single dataset build GIL measurement session."""

    def __init__(self, *, chart_dir: str, on_progress: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.chart_dir = chart_dir
        self._on_progress = on_progress
        self._probe = GILHoldProbe()
        self._burst = WorkerBurstTracker()
        self._progress_yields = 0
        self._wrapped_progress: Callable[[dict[str, Any]], None] | None = None
        self._worker_thread: threading.Thread | None = None

    def wrap_progress(self, on_progress: Callable[[dict[str, Any]], None]) -> Callable[[dict[str, Any]], None]:
        def wrapped(payload: dict[str, Any]) -> None:
            self._progress_yields += 1
            on_progress(payload)

        self._wrapped_progress = wrapped
        return wrapped

    def worker_enter(self) -> None:
        self._burst.activate()

    def worker_exit(self) -> None:
        self._burst.deactivate()
        self._probe.stop()

    def attach_worker_thread(self, thread: threading.Thread) -> None:
        self._worker_thread = thread
        if thread.ident is not None:
            self._probe.start(watch_thread_id=thread.ident)

    def build_report(self, *, build_status: str) -> dict[str, Any]:
        burst = self._burst.to_dict()
        probe = self._probe.to_dict()
        main_lag = MainThreadLagTracker.instance().to_dict()
        gil_contention_likely = (
            probe.get("longest_estimated_gil_hold_ms", 0) >= LONG_HOLD_MS
            or burst.get("longest_continuous_python_ms", 0) >= LONG_HOLD_MS
            or main_lag.get("longest_poll_gap_ms", 0) >= LONG_HOLD_MS + main_lag.get("expected_poll_interval_ms", 200)
        )
        return {
            "build_status": build_status,
            "gil_contention_likely": gil_contention_likely,
            "progress_callback_yields": self._progress_yields,
            "worker_burst_tracker": burst,
            "gil_hold_probe": probe,
            "main_thread_lag": main_lag,
            "interpretation": _interpret(burst, probe, main_lag, self._progress_yields),
        }

    def write_report(self, report: dict[str, Any]) -> str:
        out_dir = os.path.join(self.chart_dir, "data", "gil_reports")
        os.makedirs(out_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(out_dir, f"gil_build_{stamp}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
        return path


def _interpret(
    burst: dict[str, Any],
    probe: dict[str, Any],
    main_lag: dict[str, Any],
    progress_yields: int,
) -> str:
    lines: list[str] = []
    longest_burst = float(burst.get("longest_continuous_python_ms") or 0)
    longest_hold = float(probe.get("longest_estimated_gil_hold_ms") or 0)
    longest_gap = float(main_lag.get("longest_poll_gap_ms") or 0)
    expected = float(main_lag.get("expected_poll_interval_ms") or 200)

    if longest_burst >= LONG_HOLD_MS:
        lines.append(
            f"Worker ran Python continuously for up to {longest_burst:.0f} ms without yielding "
            f"({burst.get('longest_burst_location', '?')})."
        )
    if longest_hold >= LONG_HOLD_MS:
        lines.append(
            f"GIL probe estimated holds up to {longest_hold:.0f} ms "
            f"({probe.get('hold_samples_over_5ms', 0)} samples > 5 ms)."
        )
    if longest_gap >= expected + LONG_HOLD_MS:
        lines.append(
            f"Main/Tk poll loop stalled up to {longest_gap:.0f} ms "
            f"(expected ~{expected:.0f} ms) — mouse events would lag."
        )
    if not lines:
        lines.append("No GIL holds or main-thread gaps exceeded 50 ms during this build.")
    lines.append(f"Progress callbacks (voluntary yields via on_progress): {progress_yields}.")
    return " ".join(lines)


_active_session: GILBuildSession | None = None
_session_lock = threading.Lock()


def begin_gil_session(*, chart_dir: str) -> GILBuildSession | None:
    global _active_session
    if not gil_monitor_enabled():
        return None
    with _session_lock:
        MainThreadLagTracker.instance().reset()
        session = GILBuildSession(chart_dir=chart_dir)
        _active_session = session
        return session


def get_gil_session() -> GILBuildSession | None:
    return _active_session


def finish_gil_session(*, build_status: str) -> dict[str, Any] | None:
    global _active_session
    with _session_lock:
        session = _active_session
        _active_session = None
    if session is None:
        return None
    report = session.build_report(build_status=build_status)
    try:
        path = session.write_report(report)
        report["report_path"] = path
    except OSError:
        pass
    return report
