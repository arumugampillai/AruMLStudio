"""Timestamped training log written to training_log.txt."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

_IST = ZoneInfo("Asia/Kolkata")


class TrainingLog:
    def __init__(self) -> None:
        self._lines: list[str] = []

    def log(self, message: str) -> None:
        now = datetime.now(_IST)
        stamp = now.strftime("%H:%M:%S")
        self._lines.append(f"{stamp} {message}")

    def lines(self) -> list[str]:
        return list(self._lines)

    def text(self) -> str:
        return "\n".join(self._lines) + ("\n" if self._lines else "")
