"""Tests for VS Code-style lazy panel loading."""

from __future__ import annotations

import sys
import threading
import time
import unittest
from pathlib import Path

import tkinter as tk

_CHART_DIR = Path(__file__).resolve().parents[2]
if str(_CHART_DIR) not in sys.path:
    sys.path.insert(0, str(_CHART_DIR))

from master_dataset_tk.lazy_panel import LazyLoadMixin


class _TestPanel(tk.Frame, LazyLoadMixin):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self._lazy_init()

    def after(self, ms, func=None, *args):  # type: ignore[no-untyped-def]
        if func is not None and ms == 0:
            func(*args)
            return ""
        return super().after(ms, func, *args)


def _pump_events(root: tk.Tk, *, timeout: float = 3.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        root.update()
        time.sleep(0.02)


class TestLazyPanelStaleGeneration(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tk.Tk()
        self.root.withdraw()
        self.panel = _TestPanel(self.root)

    def tearDown(self) -> None:
        self.panel.cancel_lazy_load()
        try:
            _pump_events(self.root, timeout=0.5)
        except tk.TclError:
            pass
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def test_stale_generation_skips_apply(self) -> None:
        applied: list[str] = []
        load_done = threading.Event()
        allow_finish = threading.Event()

        def load() -> str:
            load_done.set()
            allow_finish.wait(timeout=2)
            return "ok"

        def apply(result: str) -> None:
            applied.append(result)

        self.panel.lazy_load(load=load, apply=apply, show_overlay=False)
        self.assertTrue(load_done.wait(timeout=2))
        self.panel.cancel_lazy_load()
        allow_finish.set()
        _pump_events(self.root, timeout=1.0)
        self.assertEqual(applied, [])

    def test_fresh_generation_applies(self) -> None:
        applied: list[str] = []

        def load() -> str:
            time.sleep(0.05)
            return "fresh"

        def apply(result: str) -> None:
            applied.append(result)

        self.panel.lazy_load(load=load, apply=apply, show_overlay=False)

        deadline = time.time() + 3
        while time.time() < deadline and not applied:
            self.root.update()
            time.sleep(0.02)

        self.assertEqual(applied, ["fresh"])

    def test_cancel_lazy_load_hides_overlay(self) -> None:
        overlay = self.panel._ensure_loading_overlay()
        overlay.show("Loading…")
        self.panel.cancel_lazy_load()
        self.assertFalse(overlay.winfo_ismapped())


if __name__ == "__main__":
    unittest.main()
