"""Shared UI helpers for master dataset Tk app."""

from __future__ import annotations

import os
import subprocess
import sys


def open_path(path: str) -> None:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if sys.platform == "win32":
        os.startfile(path if os.path.isdir(path) else os.path.dirname(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)
