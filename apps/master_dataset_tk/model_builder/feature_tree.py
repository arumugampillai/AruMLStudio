"""Collapsible feature-group tree — parity with web Model Builder feature tree."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

from ..model_registry_widgets import ACCENT, COL_MUTED


def feature_display_name(columns: dict[str, Any], fname: str) -> str:
    col = columns.get(fname) if isinstance(columns, dict) else None
    if isinstance(col, dict):
        display = col.get("display_name") or col.get("displayName")
        if display:
            return str(display)
    return fname


class FeatureSelectionTree:
    """Expandable group tree with per-group and per-feature selection."""

    def __init__(
        self,
        host: tk.Misc,
        *,
        on_structure_change: Callable[[], None],
        on_selection_change: Callable[[str, list[str], bool] | tuple[str, bool], None],
        on_preview: Callable[[str], None],
    ) -> None:
        self._host = host
        self._on_structure_change = on_structure_change
        self._on_selection_change = on_selection_change
        self._on_preview = on_preview
        self._expanded: set[str] = set()
        self._building = False

    def expand_all(self, group_ids: list[str]) -> None:
        self._expanded.update(group_ids)

    def collapse_all(self) -> None:
        self._expanded.clear()

    def ensure_expanded(self, gid: str) -> None:
        self._expanded.add(gid)

    def render(
        self,
        *,
        groups: list[dict[str, Any]],
        selected: set[str],
        columns: dict[str, Any] | None = None,
        query: str = "",
        preview_feature: str | None = None,
        read_only: bool = False,
        inspect_only: bool = False,
    ) -> None:
        if self._building:
            return
        self._building = True
        cols = columns or {}
        q = (query or "").strip().lower()
        for child in self._host.winfo_children():
            child.destroy()

        filtered: list[dict[str, Any]] = []
        for group in groups:
            feats = list(group.get("features") or [])
            label = str(group.get("label") or group.get("id") or "")
            if q:
                feats = [
                    f for f in feats
                    if q in f.lower()
                    or q in feature_display_name(cols, f).lower()
                    or q in label.lower()
                ]
            registry_total = int(
                group.get("total_features")
                or len(group.get("registry_features") or [])
                or len(feats)
            )
            if feats or (not inspect_only and not q and registry_total > 0):
                filtered.append({**group, "features": feats, "total_features": registry_total})

        if not filtered:
            ttk.Label(
                self._host,
                text="No features match your search query.",
                foreground=COL_MUTED,
            ).pack(anchor="w", pady=8)
            self._building = False
            return

        if q:
            self._expanded.update(str(g["id"]) for g in filtered)

        for group in filtered:
            gid = str(group["id"])
            feats = list(group["features"])
            label = str(group.get("label") or gid)
            total = int(group.get("total_features") or len(feats))
            n_sel = sum(1 for f in feats if f in selected)
            expanded = gid in self._expanded

            shell = ttk.Frame(self._host, padding=(0, 2))
            shell.pack(fill="x", pady=1)

            hdr = ttk.Frame(shell)
            hdr.pack(fill="x")

            chevron = "▼" if expanded else "▶"
            ttk.Button(
                hdr,
                text=chevron,
                width=2,
                command=lambda g=gid: self._toggle_expand(g),
            ).pack(side="left")

            if not inspect_only:
                gv = tk.StringVar()
                if not feats:
                    gv.set("off")
                elif n_sel == len(feats):
                    gv.set("on")
                elif n_sel == 0:
                    gv.set("off")
                else:
                    gv.set("")
                cb = tk.Checkbutton(
                    hdr,
                    variable=gv,
                    onvalue="on",
                    offvalue="off",
                    tristatevalue="",
                    state="disabled" if (read_only or not feats) else "normal",
                    command=lambda g=gid, fl=feats, var=gv: self._group_checked(g, fl, var),
                )
                cb.pack(side="left")

            title = ttk.Label(hdr, text=label, font=("Segoe UI", 9, "bold"), cursor="hand2")
            title.pack(side="left", padx=(2, 4))
            title.bind("<Button-1>", lambda _e, g=gid: self._toggle_expand(g))

            ttk.Label(
                hdr,
                text=f"{n_sel}/{total}",
                foreground=COL_MUTED,
                font=("Segoe UI", 8),
            ).pack(side="left")

            if not expanded:
                continue

            body = ttk.Frame(shell, padding=(18, 0, 0, 4))
            body.pack(fill="x")
            if not feats:
                ttk.Label(
                    body,
                    text="Not in selected dataset — rebuild with this group enabled.",
                    foreground=COL_MUTED,
                    wraplength=460,
                    font=("Segoe UI", 8),
                ).pack(anchor="w", pady=4)
                continue

            for feat in feats:
                row = ttk.Frame(body)
                row.pack(fill="x")
                display = feature_display_name(cols, feat)
                active = feat == preview_feature
                if inspect_only:
                    lbl = ttk.Label(
                        row,
                        text=display,
                        foreground=ACCENT if active else "#333333",
                        cursor="hand2",
                    )
                    lbl.pack(anchor="w", fill="x", padx=(8, 0))
                    lbl.bind("<Button-1>", lambda _e, fn=feat: self._on_preview(fn))
                    continue

                fv = tk.BooleanVar(value=feat in selected)
                ttk.Checkbutton(
                    row,
                    variable=fv,
                    state="disabled" if read_only else "normal",
                    command=lambda fn=feat, var=fv: self._feature_checked(fn, var),
                ).pack(side="left")
                lbl = ttk.Label(
                    row,
                    text=display,
                    foreground=ACCENT if active else "#333333",
                    cursor="hand2",
                )
                lbl.pack(side="left", padx=(2, 0))
                lbl.bind("<Button-1>", lambda _e, fn=feat: self._on_preview(fn))

        self._building = False

    def _toggle_expand(self, gid: str) -> None:
        if gid in self._expanded:
            self._expanded.discard(gid)
        else:
            self._expanded.add(gid)
        self._on_structure_change()

    def _group_checked(self, gid: str, feats: list[str], var: tk.StringVar) -> None:
        if self._building:
            return
        self._on_selection_change(("group", gid, feats, var.get() == "on"))

    def _feature_checked(self, feat: str, var: tk.BooleanVar) -> None:
        if self._building:
            return
        self._on_selection_change(("feature", feat, bool(var.get())))
