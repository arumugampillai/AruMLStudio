"""Shared cancel / pause control for Prediction Dataset builds."""

from __future__ import annotations

import threading
from typing import Any


class BuildControl:
    """Thread-safe Start/Pause/Resume/Cancel signals for the day job runner."""

    def __init__(self) -> None:
        self.cancel = threading.Event()
        self.pause = threading.Event()
        self._lock = threading.Lock()
        self._meta: dict[str, Any] = {}

    def reset(self) -> None:
        self.cancel.clear()
        self.pause.clear()
        with self._lock:
            self._meta.clear()

    def request_cancel(self) -> None:
        self.cancel.set()
        self.pause.clear()

    def request_pause(self) -> None:
        """Finish the current day, then stop (do not start the next day)."""
        self.pause.set()

    def request_resume(self) -> None:
        self.pause.clear()
        self.cancel.clear()

    @property
    def should_stop_after_day(self) -> bool:
        return self.cancel.is_set() or self.pause.is_set()

    @property
    def is_cancel(self) -> bool:
        return self.cancel.is_set()

    @property
    def is_pause(self) -> bool:
        return self.pause.is_set() and not self.cancel.is_set()


# Process-wide active control for the UI buttons
_ACTIVE: BuildControl | None = None
_ACTIVE_LOCK = threading.Lock()


def get_active_build_control() -> BuildControl | None:
    with _ACTIVE_LOCK:
        return _ACTIVE


def set_active_build_control(ctrl: BuildControl | None) -> None:
    global _ACTIVE
    with _ACTIVE_LOCK:
        _ACTIVE = ctrl
