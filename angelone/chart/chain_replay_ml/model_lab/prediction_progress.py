"""Pipeline stages + heartbeat progress for Prediction Dataset builds."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Any, Callable

ProgressFn = Callable[[dict[str, Any]], None]

# Explicit pipeline stages (stable ids + labels for UI)
STAGE_LOAD_MODEL = 1
STAGE_READ_METADATA = 2
STAGE_CATALOG_DAYS = 3
STAGE_LOAD_DAY = 4
STAGE_ENRICH = 5
STAGE_PREDICT = 6
STAGE_WRITE = 7
STAGE_FINISHED = 8

STAGE_TOTAL = 8

STAGE_LABELS: dict[int, str] = {
    STAGE_LOAD_MODEL: "Loading model",
    STAGE_READ_METADATA: "Reading parquet metadata",
    STAGE_CATALOG_DAYS: "Cataloging trading days",
    STAGE_LOAD_DAY: "Loading trading day",
    STAGE_ENRICH: "Enriching path outcomes",
    STAGE_PREDICT: "Predicting",
    STAGE_WRITE: "Writing SQLite",
    STAGE_FINISHED: "Finished",
}

HEARTBEAT_INTERVAL_SEC = 1.0
STALL_WARN_SEC = 30.0

PARQUET_BATCH_ROWS = 5_000


def stage_label(stage: int) -> str:
    return STAGE_LABELS.get(int(stage), f"Stage {stage}")


class ProgressHub:
    """
    Thread-safe progress state with a 1 Hz heartbeat emitter.

    Workers call update(); a daemon thread re-emits the latest snapshot every second
    so the GUI never freezes during long parquet / predict work.
    """

    def __init__(
        self,
        on_progress: ProgressFn | None,
        *,
        started_at: float | None = None,
        heartbeat_sec: float = HEARTBEAT_INTERVAL_SEC,
    ) -> None:
        self._on_progress = on_progress
        self._lock = threading.Lock()
        self._started_at = float(started_at if started_at is not None else time.perf_counter())
        self._state: dict[str, Any] = {
            "phase": "building",
            "status": "building",
            "stage": 0,
            "stage_total": STAGE_TOTAL,
            "stage_label": "",
            "stage_detail": "",
            "message": "",
            "current_day": "",
            "rows_loaded": 0,
            "rows_day_total": 0,
            "samples_done": 0,
            "samples_total": 0,
            "trading_days_done": 0,
            "trading_days_total": 0,
            "predictions_written": 0,
            "worker_count": 1,
            "workers": [],
            "heartbeat": False,
            "last_update": "",
            "last_update_mono": time.monotonic(),
        }
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._heartbeat_sec = float(heartbeat_sec)

    def start(self) -> None:
        if self._on_progress is None or self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name="pred-progress-heartbeat",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        th = self._thread
        if th is not None and th.is_alive():
            th.join(timeout=2.0)
        self._thread = None

    def update(self, **fields: Any) -> None:
        """Merge fields and emit immediately (not only on heartbeat)."""
        with self._lock:
            self._state.update({k: v for k, v in fields.items() if v is not None})
            stage = int(self._state.get("stage") or 0)
            if stage and not fields.get("stage_label"):
                self._state["stage_label"] = stage_label(stage)
            self._state["heartbeat"] = False
            self._state["last_update"] = datetime.now().strftime("%H:%M:%S")
            self._state["last_update_mono"] = time.monotonic()
            self._state["elapsed_sec"] = round(time.perf_counter() - self._started_at, 1)
            payload = dict(self._state)
            payload["eta_sec"] = self._eta(payload)
            payload["percent"] = self._percent(payload)
            payload["message"] = self._compose_message(payload)
        self._emit(payload)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            payload = dict(self._state)
        payload["elapsed_sec"] = round(time.perf_counter() - self._started_at, 1)
        payload["eta_sec"] = self._eta(payload)
        payload["percent"] = self._percent(payload)
        return payload

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self._heartbeat_sec):
            with self._lock:
                payload = dict(self._state)
                payload["heartbeat"] = True
                payload["last_update"] = datetime.now().strftime("%H:%M:%S")
                # Do not refresh last_update_mono on heartbeat — stall detection needs real work updates
                payload["elapsed_sec"] = round(time.perf_counter() - self._started_at, 1)
                payload["eta_sec"] = self._eta(payload)
                payload["percent"] = self._percent(payload)
                payload["message"] = self._compose_message(payload)
            self._emit(payload)

    def _emit(self, payload: dict[str, Any]) -> None:
        if self._on_progress:
            try:
                self._on_progress(payload)
            except Exception:
                pass

    @staticmethod
    def _compose_message(p: dict[str, Any]) -> str:
        stage = int(p.get("stage") or 0)
        label = str(p.get("stage_label") or stage_label(stage) or "Working")
        detail = str(p.get("stage_detail") or "").strip()
        day = str(p.get("current_day") or "").strip()
        parts = [f"Stage {stage}/{STAGE_TOTAL}", label]
        if day:
            parts.append(day)
        if detail:
            parts.append(detail)
        return " · ".join(parts)

    @staticmethod
    def _percent(p: dict[str, Any]) -> float:
        # Prefer trading-day progress (stable unit of work)
        try:
            days_done = float(p.get("trading_days_done") or p.get("days_completed") or 0)
            days_total = float(p.get("trading_days_total") or 0)
            if days_total > 0:
                day_pct = p.get("day_progress_pct")
                if day_pct is None:
                    loaded = p.get("rows_loaded")
                    day_tot = p.get("rows_day_total")
                    if loaded is not None and day_tot and float(day_tot) > 0:
                        day_pct = 100.0 * min(1.0, float(loaded) / float(day_tot))
                day_frac = 0.0
                if day_pct is not None:
                    day_frac = max(0.0, min(1.0, float(day_pct) / 100.0))
                overall = (days_done + day_frac) / days_total
                return max(0.0, min(99.9, overall * 100.0))
        except (TypeError, ValueError):
            pass
        # Stage-based fallback so prep isn't stuck at 0%
        stage = int(p.get("stage") or 0)
        if stage <= 0:
            return 0.0
        return max(0.0, min(95.0, (stage - 1) / STAGE_TOTAL * 100.0))

    @staticmethod
    def _eta(p: dict[str, Any]) -> float | None:
        """Estimate remaining time from completed trading days (not rows)."""
        elapsed = float(p.get("elapsed_sec") or 0.0)
        if elapsed <= 0:
            return None
        try:
            days_done = float(p.get("trading_days_done") or p.get("days_completed") or 0)
            rem = p.get("days_remaining")
            if rem is None:
                days_total = float(p.get("trading_days_total") or 0)
                rem = max(0.0, days_total - days_done) if days_total else 0.0
            else:
                rem = float(rem)
            if days_done > 0 and rem > 0:
                return (elapsed / days_done) * rem
        except (TypeError, ValueError):
            pass
        return None
