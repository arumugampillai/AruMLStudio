"""VS Code-style lazy panel loading — show shell immediately, fetch in background."""

from __future__ import annotations

import threading
from typing import Any, Callable, TypeVar

import tkinter as tk
from tkinter import ttk

T = TypeVar("T")
Loader = Callable[[], T]
Applier = Callable[[T], None]
ErrorHandler = Callable[[Exception], None]


class PanelLoadingOverlay(ttk.Frame):
    """Centered loading hint placed over a panel host."""

    def __init__(self, host: tk.Misc, *, message: str = "Loading…") -> None:
        super().__init__(host)
        self._message_var = tk.StringVar(value=message)
        inner = ttk.Frame(self, padding=16)
        inner.place(relx=0.5, rely=0.45, anchor="center")
        ttk.Label(
            inner,
            textvariable=self._message_var,
            font=("Segoe UI", 10),
            foreground="#666666",
        ).pack()
        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.place_forget()

    def show(self, message: str | None = None) -> None:
        if message:
            self._message_var.set(message)
        self.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.lift()

    def hide(self) -> None:
        self.place_forget()


class LazyLoadMixin:
    """Attach to ttk.Frame panels; call _lazy_init() from __init__."""

    _lazy_generation: int
    _loading_overlay: PanelLoadingOverlay | None

    def _lazy_init(self) -> None:
        self._lazy_generation = 0
        self._loading_overlay = None

    def _ensure_loading_overlay(self, *, message: str = "Loading…") -> PanelLoadingOverlay:
        if self._loading_overlay is None:
            self._loading_overlay = PanelLoadingOverlay(self, message=message)
        return self._loading_overlay

    def lazy_load(
        self,
        *,
        load: Loader[T],
        apply: Applier[T],
        message: str = "Loading…",
        on_error: ErrorHandler | None = None,
        show_overlay: bool = True,
        status_var: tk.StringVar | None = None,
    ) -> None:
        """Run ``load`` on a worker thread; ``apply`` on the Tk main thread."""
        self._lazy_generation = getattr(self, "_lazy_generation", 0) + 1
        generation = self._lazy_generation
        if status_var is not None:
            status_var.set(message)
        overlay: PanelLoadingOverlay | None = None
        if show_overlay:
            overlay = self._ensure_loading_overlay(message=message)
            overlay.show(message)

        def worker() -> None:
            err: Exception | None = None
            result: Any = None
            try:
                result = load()
            except Exception as exc:
                err = exc

            def finish() -> None:
                if generation != getattr(self, "_lazy_generation", 0):
                    return
                if overlay is not None:
                    overlay.hide()
                if err is not None:
                    if on_error is not None:
                        on_error(err)
                    elif status_var is not None:
                        status_var.set(f"Error: {err}")
                    else:
                        from tkinter import messagebox

                        messagebox.showerror("Load failed", str(err))
                    return
                try:
                    apply(result)
                except Exception as exc:
                    if on_error is not None:
                        on_error(exc)
                    elif status_var is not None:
                        status_var.set(f"Error: {exc}")
                    else:
                        from tkinter import messagebox

                        messagebox.showerror("Load failed", str(exc))

            try:
                self.after(0, finish)
            except tk.TclError:
                pass

        threading.Thread(
            target=worker,
            name=f"lazy-{self.__class__.__name__}",
            daemon=True,
        ).start()

    def cancel_lazy_load(self) -> None:
        """Invalidate in-flight lazy loads (e.g. on chart_dir change)."""
        self._lazy_generation = getattr(self, "_lazy_generation", 0) + 1
        if self._loading_overlay is not None:
            self._loading_overlay.hide()
