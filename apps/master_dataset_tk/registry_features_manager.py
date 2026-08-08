"""Registry Features export selection — checkboxes only (no delete/edit)."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from chain_replay_ml.dataset_builder.feature_sources_catalog import registry_feature_source
from chain_replay_ml.dataset_builder.registry_features_prefs import (
    MODE_ALL,
    load_registry_export_mode,
    load_registry_export_selected_names,
    save_registry_export_selection,
)


class RegistryFeaturesSelectionDialog(tk.Toplevel):
    """Choose which Registry Features are written into the Analysis Dataset."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        data_dir: str,
        on_changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.title("Registry Features — Export Selection")
        self.transient(master.winfo_toplevel())
        self.geometry("680x720")
        self._data_dir = data_dir
        self._on_changed = on_changed
        self._filter_var = tk.StringVar(value="")
        self._count_var = tk.StringVar(value="")
        self._feature_vars: dict[str, tk.BooleanVar] = {}
        self._group_frames: list[ttk.LabelFrame] = []
        self._source = registry_feature_source(data_dir=data_dir)

        self._build_ui()
        self._load_selection()
        self._refresh_count()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        try:
            self.grab_set()
        except tk.TclError:
            pass

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill="both", expand=True)
        top.rowconfigure(2, weight=1)
        top.columnconfigure(0, weight=1)

        ttk.Label(
            top,
            text=(
                "Selected features are included in the Analysis Dataset. "
                "Unselected features remain available as sources for Pipeline "
                "Feature Transformations."
            ),
            wraplength=640,
        ).grid(row=0, column=0, sticky="w")

        toolbar = ttk.Frame(top)
        toolbar.grid(row=1, column=0, sticky="ew", pady=(8, 4))
        ttk.Label(toolbar, text="Search").pack(side="left")
        entry = ttk.Entry(toolbar, textvariable=self._filter_var)
        entry.pack(side="left", fill="x", expand=True, padx=(6, 12))
        entry.bind("<KeyRelease>", lambda _e: self._apply_filter())
        ttk.Button(toolbar, text="Select All", command=self._select_all).pack(side="left", padx=2)
        ttk.Button(toolbar, text="Unselect All", command=self._unselect_all).pack(side="left", padx=2)
        ttk.Label(toolbar, textvariable=self._count_var, font=("Segoe UI", 9, "bold")).pack(
            side="right", padx=(8, 0)
        )

        scroll_wrap = ttk.Frame(top)
        scroll_wrap.grid(row=2, column=0, sticky="nsew")
        scroll_wrap.rowconfigure(0, weight=1)
        scroll_wrap.columnconfigure(0, weight=1)
        canvas = tk.Canvas(scroll_wrap, highlightthickness=0)
        sb = ttk.Scrollbar(scroll_wrap, orient="vertical", command=canvas.yview)
        self._inner = ttk.Frame(canvas)
        inner_id = canvas.create_window((0, 0), window=self._inner, anchor="nw")
        canvas.configure(yscrollcommand=sb.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        def _sync_scroll(_event: tk.Event | None = None) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _sync_width(event: tk.Event) -> None:
            canvas.itemconfigure(inner_id, width=max(1, int(event.width)))

        self._inner.bind("<Configure>", _sync_scroll)
        canvas.bind("<Configure>", _sync_width)

        def _on_wheel(event: tk.Event) -> str | None:
            if event.delta:
                canvas.yview_scroll(int(-event.delta / 120), "units")
            elif getattr(event, "num", None) == 4:
                canvas.yview_scroll(-3, "units")
            elif getattr(event, "num", None) == 5:
                canvas.yview_scroll(3, "units")
            return "break"

        for w in (canvas, self._inner):
            w.bind("<Enter>", lambda _e, widget=w: widget.bind_all("<MouseWheel>", _on_wheel), add="+")
            w.bind("<Leave>", lambda _e, widget=w: widget.unbind_all("<MouseWheel>"), add="+")

        self._build_groups()

        actions = ttk.Frame(top)
        actions.grid(row=3, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(actions, text="Save", command=self._save).pack(side="right")
        ttk.Button(actions, text="Close", command=self._on_close).pack(side="right", padx=(0, 6))

    def _build_groups(self) -> None:
        for child in self._inner.winfo_children():
            child.destroy()
        self._group_frames.clear()
        for group in self._source.get("groups") or []:
            label = str(group.get("label") or group.get("id") or "Group")
            frame = ttk.LabelFrame(self._inner, text=label, padding=6)
            frame.pack(fill="x", pady=(0, 6))
            self._group_frames.append(frame)
            for feat in group.get("features") or []:
                name = str(feat)
                if name not in self._feature_vars:
                    var = tk.BooleanVar(value=True)
                    var.trace_add("write", lambda *_a: self._refresh_count())
                    self._feature_vars[name] = var
                row = ttk.Frame(frame)
                row.pack(anchor="w", fill="x")
                row._feature_name = name  # type: ignore[attr-defined]
                ttk.Checkbutton(row, text=name, variable=self._feature_vars[name]).pack(anchor="w")

    def _load_selection(self) -> None:
        all_names = {str(f) for f in (self._source.get("features") or [])}
        mode = load_registry_export_mode(self._data_dir)
        if mode == MODE_ALL:
            selected = all_names
        else:
            selected = set(load_registry_export_selected_names(self._data_dir))
            if not selected:
                selected = all_names
        for name, var in self._feature_vars.items():
            var.set(name in selected)

    def _selected_names(self) -> list[str]:
        return sorted(name for name, var in self._feature_vars.items() if var.get())

    def _refresh_count(self) -> None:
        total = len(self._feature_vars)
        n = len(self._selected_names())
        self._count_var.set(f"{n} / {total} Selected")

    def _apply_filter(self) -> None:
        needle = str(self._filter_var.get() or "").strip().lower()
        for frame in self._group_frames:
            any_visible = False
            for child in frame.winfo_children():
                if not isinstance(child, ttk.Frame):
                    continue
                name = str(getattr(child, "_feature_name", "") or "")
                show = not needle or needle in name.lower()
                if show:
                    child.pack(anchor="w", fill="x")
                    any_visible = True
                else:
                    child.pack_forget()
            if any_visible:
                frame.pack(fill="x", pady=(0, 6))
            else:
                frame.pack_forget()

    def _select_all(self) -> None:
        needle = str(self._filter_var.get() or "").strip().lower()
        for name, var in self._feature_vars.items():
            if not needle or needle in name.lower():
                var.set(True)

    def _unselect_all(self) -> None:
        needle = str(self._filter_var.get() or "").strip().lower()
        for name, var in self._feature_vars.items():
            if not needle or needle in name.lower():
                var.set(False)

    def _save(self) -> None:
        selected = self._selected_names()
        save_registry_export_selection(self._data_dir, selected=selected)
        self._refresh_count()
        if self._on_changed:
            self._on_changed()
        self.destroy()

    def _on_close(self) -> None:
        self.destroy()


def open_registry_features_selection(
    master: tk.Misc,
    *,
    data_dir: str,
    on_changed: Callable[[], None] | None = None,
) -> RegistryFeaturesSelectionDialog:
    return RegistryFeaturesSelectionDialog(master, data_dir=data_dir, on_changed=on_changed)


__all__ = [
    "RegistryFeaturesSelectionDialog",
    "open_registry_features_selection",
]
