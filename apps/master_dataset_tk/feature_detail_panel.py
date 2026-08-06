"""Unified feature detail window — formula, inputs, and Python pseudocode."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Callable

from . import feature_detail_format as fmt
from .feature_detail_builder import build_feature_detail
from .fold_replay_widgets import place_toplevel_over_main, resolve_main_app_root


def resolve_chart_dir(widget: tk.Misc) -> str | None:
    w: tk.Misc | None = widget
    while w is not None:
        if hasattr(w, "chart_dir"):
            return str(getattr(w, "chart_dir"))
        w = w.master  # type: ignore[assignment]
    return None


def _load_features_by_name(chart_dir: str | None) -> dict[str, dict[str, Any]]:
    if not chart_dir:
        return {}
    try:
        from . import feature_registry_service as reg_svc

        catalog = reg_svc.load_catalog(chart_dir)
        return {
            str(f.get("name")): f
            for f in (catalog.get("features") or [])
            if f.get("name")
        }
    except Exception:
        return {}


def _window_alive(win: tk.Toplevel | None) -> bool:
    if win is None:
        return False
    try:
        return bool(win.winfo_exists())
    except tk.TclError:
        return False


def _fill_feature_detail_window(
    win: tk.Toplevel,
    detail: dict[str, Any],
    *,
    feature_name: str,
    title_suffix: str = "",
) -> None:
    name = str(feature_name)
    suffix = f" — {title_suffix}" if title_suffix else ""
    win.title(f"Feature: {detail.get('display_name') or name}{suffix}")
    win._feature_detail_payload = detail  # type: ignore[attr-defined]

    title_lbl: ttk.Label = win._feature_detail_title  # type: ignore[attr-defined]
    sub_lbl: ttk.Label = win._feature_detail_subtitle  # type: ignore[attr-defined]
    body: scrolledtext.ScrolledText = win._feature_detail_body  # type: ignore[attr-defined]

    title_lbl.configure(text=str(detail.get("display_name") or name))
    sub_lbl.configure(text=f"{name}  ·  double-click features to update · click source path to open code")
    body.configure(state="normal")
    body.delete("1.0", tk.END)
    fmt.render_feature_detail_widget(body, detail)


def _create_feature_detail_window(
    master: tk.Misc,
    *,
    on_destroy: Callable[[], None] | None = None,
) -> tk.Toplevel:
    root = resolve_main_app_root(master)
    win = tk.Toplevel(root)
    win.transient(root)
    win._feature_detail_payload = {}  # type: ignore[attr-defined]

    hdr = ttk.Frame(win, padding=8)
    hdr.pack(fill="x")
    title_lbl = ttk.Label(hdr, text="", font=("Segoe UI", 11, "bold"))
    title_lbl.pack(anchor="w")
    sub_lbl = ttk.Label(hdr, text="", foreground="#666", font=("Segoe UI", 8))
    sub_lbl.pack(anchor="w")
    win._feature_detail_title = title_lbl  # type: ignore[attr-defined]
    win._feature_detail_subtitle = sub_lbl  # type: ignore[attr-defined]

    body = scrolledtext.ScrolledText(
        win, wrap="word", font=("Consolas", 10), padx=8, pady=8,
    )
    body.pack(fill="both", expand=True, padx=8, pady=(0, 8))
    body.bind("<Key>", lambda _e: "break")
    win._feature_detail_body = body  # type: ignore[attr-defined]

    btn_row = ttk.Frame(win, padding=(8, 0, 8, 8))
    btn_row.pack(fill="x")

    def _copy() -> None:
        payload = getattr(win, "_feature_detail_payload", {}) or {}
        text = fmt.format_feature_detail_text(payload)
        win.clipboard_clear()
        win.clipboard_append(text)
        messagebox.showinfo("Copy", "Feature detail copied to clipboard.", parent=win)

    def _open_source() -> None:
        payload = getattr(win, "_feature_detail_payload", {}) or {}
        loc = payload.get("source_location") or {}
        if not loc.get("ok"):
            messagebox.showwarning(
                "Open Source",
                loc.get("error") or "Source file not found for this feature.",
                parent=win,
            )
            return
        from .source_navigation import open_source_location

        if not open_source_location(loc):
            messagebox.showwarning(
                "Open Source",
                f"Could not open editor for:\n{loc.get('path')}",
                parent=win,
            )

    def _close() -> None:
        if on_destroy is not None:
            on_destroy()
        win.destroy()

    ttk.Button(btn_row, text="Open Source", command=_open_source).pack(side="left", padx=(0, 8))
    ttk.Button(btn_row, text="Copy", command=_copy).pack(side="left")
    ttk.Button(btn_row, text="Close", command=_close).pack(side="left", padx=8)
    win.protocol("WM_DELETE_WINDOW", _close)

    return win


def open_feature_detail_window(
    master: tk.Misc,
    feature_name: str,
    *,
    chart_dir: str | None = None,
    features_by_name: dict[str, dict[str, Any]] | None = None,
    context: dict[str, Any] | None = None,
    title_suffix: str = "",
    reuse_window: tk.Toplevel | None = None,
    on_destroy: Callable[[], None] | None = None,
) -> tk.Toplevel | None:
    """Open or update the standard feature detail panel for *feature_name*."""
    name = str(feature_name or "").strip()
    if not name:
        return None

    chart_dir = chart_dir or resolve_chart_dir(master)
    fb = features_by_name if features_by_name is not None else _load_features_by_name(chart_dir)
    detail = build_feature_detail(name, features_by_name=fb, context=context)
    if not detail.get("ok"):
        messagebox.showwarning("Feature Detail", detail.get("error") or "Unknown feature", parent=master)
        return None

    win = reuse_window if _window_alive(reuse_window) else None
    created = False
    if win is None:
        win = _create_feature_detail_window(master, on_destroy=on_destroy)
        created = True

    try:
        _fill_feature_detail_window(win, detail, feature_name=name, title_suffix=title_suffix)
    except tk.TclError:
        win = _create_feature_detail_window(master, on_destroy=on_destroy)
        created = True
        _fill_feature_detail_window(win, detail, feature_name=name, title_suffix=title_suffix)

    if created:
        win.update_idletasks()
    place_toplevel_over_main(win, master, width_scale=0.75, height_scale=0.70)
    try:
        win.deiconify()
        win.lift()
        win.focus_force()
    except tk.TclError:
        pass
    return win
