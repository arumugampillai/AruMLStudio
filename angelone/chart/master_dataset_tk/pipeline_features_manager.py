"""Pipeline Features manager — browse + permanently delete selected features."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable

from chain_replay_ml.dataset_builder.feature_sources_catalog import (
    GENERATOR_FAMILY_LABELS,
    pipeline_family_of,
    pipeline_feature_names,
)
from chain_replay_ml.dataset_builder.pipeline_features_prefs import (
    load_retired_pipeline_features,
    retire_pipeline_features,
)


class PipelineFeaturesManagerDialog(tk.Toplevel):
    """List active Pipeline Features; permanently delete the selection."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        data_dir: str,
        on_changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.title("Pipeline Features")
        self.transient(master.winfo_toplevel())
        self.geometry("640x720")
        self._data_dir = data_dir
        self._on_changed = on_changed
        self._filter_var = tk.StringVar(value="")
        self._status_var = tk.StringVar(value="")
        self._rec_status_var = tk.StringVar(value="")
        self._iid_to_name: dict[str, str] = {}
        self._rec_iid_to_name: dict[str, str] = {}

        self._build_ui()
        self._reload()
        self._reload_recommendations()
        self.protocol("WM_DELETE_WINDOW", self.destroy)
        try:
            self.grab_set()
        except tk.TclError:
            pass

    def _build_ui(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill="both", expand=True)
        top.rowconfigure(1, weight=1)
        top.columnconfigure(0, weight=1)

        ttk.Label(
            top,
            text="Select Pipeline Features to remove permanently from Auto builds.",
            wraplength=600,
        ).grid(row=0, column=0, sticky="w")

        panes = ttk.Panedwindow(top, orient="vertical")
        panes.grid(row=1, column=0, sticky="nsew", pady=(6, 0))

        # ── Active catalogue ──────────────────────────────────────────────
        catalog = ttk.Frame(panes, padding=(0, 0, 0, 4))
        catalog.columnconfigure(0, weight=1)
        catalog.rowconfigure(1, weight=1)
        panes.add(catalog, weight=3)

        filt = ttk.Frame(catalog)
        filt.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        ttk.Label(filt, text="Filter").pack(side="left")
        entry = ttk.Entry(filt, textvariable=self._filter_var)
        entry.pack(side="left", fill="x", expand=True, padx=(6, 0))
        entry.bind("<KeyRelease>", lambda _e: self._reload())

        tree_frame = ttk.Frame(catalog)
        tree_frame.grid(row=1, column=0, sticky="nsew")
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        self._tree = ttk.Treeview(
            tree_frame,
            columns=("family",),
            show="tree headings",
            selectmode="extended",
            height=12,
        )
        self._tree.heading("#0", text="Feature")
        self._tree.heading("family", text="Family")
        self._tree.column("#0", width=400, stretch=True)
        self._tree.column("family", width=140, anchor="w")
        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")

        ttk.Label(catalog, textvariable=self._status_var, foreground="#555").grid(
            row=2, column=0, sticky="w", pady=(4, 0)
        )

        cat_actions = ttk.Frame(catalog)
        cat_actions.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(cat_actions, text="Select all", command=self._select_all).pack(
            side="left"
        )
        ttk.Button(
            cat_actions, text="Clear selection", command=self._clear_selection
        ).pack(side="left", padx=(6, 0))
        ttk.Button(
            cat_actions,
            text="Delete selected permanently",
            command=self._delete_selected,
        ).pack(side="right")

        # ── Recommended for Removal ───────────────────────────────────────
        rec_panel = ttk.LabelFrame(panes, text="Recommended for Removal", padding=6)
        rec_panel.columnconfigure(0, weight=1)
        rec_panel.rowconfigure(1, weight=1)
        panes.add(rec_panel, weight=2)

        ttk.Label(
            rec_panel,
            text=(
                "From Production Validation history. Nothing is removed automatically — "
                "select features and confirm."
            ),
            foreground="#666",
            wraplength=600,
        ).grid(row=0, column=0, sticky="w", pady=(0, 4))

        rec_tree_frame = ttk.Frame(rec_panel)
        rec_tree_frame.grid(row=1, column=0, sticky="nsew")
        rec_tree_frame.columnconfigure(0, weight=1)
        rec_tree_frame.rowconfigure(0, weight=1)
        self._rec_tree = ttk.Treeview(
            rec_tree_frame,
            columns=("feature", "remove_runs", "last_date"),
            show="headings",
            selectmode="extended",
            height=8,
        )
        self._rec_tree.heading("feature", text="Feature Name")
        self._rec_tree.heading("remove_runs", text="REMOVE Runs")
        self._rec_tree.heading("last_date", text="Last Date")
        self._rec_tree.column("feature", width=300, anchor="w", stretch=True)
        self._rec_tree.column("remove_runs", width=100, anchor="e")
        self._rec_tree.column("last_date", width=160, anchor="w")
        rec_sb = ttk.Scrollbar(
            rec_tree_frame, orient="vertical", command=self._rec_tree.yview
        )
        self._rec_tree.configure(yscrollcommand=rec_sb.set)
        self._rec_tree.grid(row=0, column=0, sticky="nsew")
        rec_sb.grid(row=0, column=1, sticky="ns")

        ttk.Label(rec_panel, textvariable=self._rec_status_var, foreground="#555").grid(
            row=2, column=0, sticky="w", pady=(4, 0)
        )

        rec_actions = ttk.Frame(rec_panel)
        rec_actions.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(
            rec_actions,
            text="Refresh Recommendations",
            command=self._reload_recommendations,
        ).pack(side="left")
        ttk.Button(
            rec_actions,
            text="Ignore Recommendation",
            command=self._ignore_selected_recommendations,
        ).pack(side="left", padx=(6, 0))
        ttk.Button(
            rec_actions,
            text="Remove Selected",
            command=self._remove_selected_recommendations,
        ).pack(side="right")

        # ── Dialog footer ─────────────────────────────────────────────────
        footer = ttk.Frame(top)
        footer.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(footer, text="Close", command=self.destroy).pack(side="right")

    def _reload(self) -> None:
        self._tree.delete(*self._tree.get_children())
        self._iid_to_name.clear()
        needle = str(self._filter_var.get() or "").strip().lower()
        names = pipeline_feature_names(data_dir=self._data_dir)
        retired_n = len(load_retired_pipeline_features(self._data_dir))
        shown = 0
        by_family: dict[str, list[str]] = {}
        for name in names:
            if needle and needle not in name.lower():
                continue
            fam = pipeline_family_of(name)
            by_family.setdefault(fam, []).append(name)
        for fam, feats in sorted(by_family.items(), key=lambda kv: kv[0]):
            label = GENERATOR_FAMILY_LABELS.get(fam, fam.title())
            parent = self._tree.insert(
                "", "end", text=f"{label} ({len(feats)})", open=True
            )
            for name in feats:
                iid = self._tree.insert(parent, "end", text=name, values=(label,))
                self._iid_to_name[iid] = name
                shown += 1
        self._status_var.set(
            f"Active: {len(names)} · Showing: {shown}"
            + (f" · Permanently deleted: {retired_n}" if retired_n else "")
        )

    def _format_last_date(self, value: Any) -> str:
        text = str(value or "").strip()
        if not text:
            return "—"
        # Show date portion when ISO timestamps are stored.
        if "T" in text:
            return text.split("T", 1)[0]
        return text[:19]

    def _reload_recommendations(self) -> None:
        self._rec_tree.delete(*self._rec_tree.get_children())
        self._rec_iid_to_name.clear()
        try:
            from chain_replay_ml.production_validation import recommended_for_removal
        except Exception as exc:
            self._rec_status_var.set(f"Could not load recommendations: {exc}")
            return

        active = set(pipeline_feature_names(data_dir=self._data_dir))
        try:
            rows = recommended_for_removal(self._data_dir)
        except Exception as exc:
            self._rec_status_var.set(f"Could not load recommendations: {exc}")
            return

        shown = 0
        for row in rows:
            if not isinstance(row, dict):
                continue
            name = str(row.get("feature_name") or "").strip()
            if not name or name not in active:
                continue
            iid = self._rec_tree.insert(
                "",
                "end",
                values=(
                    name,
                    str(int(row.get("remove_runs") or 0)),
                    self._format_last_date(row.get("last_date")),
                ),
            )
            self._rec_iid_to_name[iid] = name
            shown += 1

        total = len(rows)
        skipped = total - shown
        parts = [f"{shown} recommendation(s)"]
        if skipped:
            parts.append(f"{skipped} not in active Pipeline catalogue")
        self._rec_status_var.set(" · ".join(parts))

    def _selected_feature_names(self) -> list[str]:
        out: list[str] = []
        for iid in self._tree.selection():
            name = self._iid_to_name.get(iid)
            if name:
                out.append(name)
        return sorted(set(out))

    def _selected_recommendation_names(self) -> list[str]:
        out: list[str] = []
        for iid in self._rec_tree.selection():
            name = self._rec_iid_to_name.get(iid)
            if name:
                out.append(name)
        return sorted(set(out))

    def _select_all(self) -> None:
        self._tree.selection_set(tuple(self._iid_to_name.keys()))

    def _clear_selection(self) -> None:
        self._tree.selection_remove(self._tree.selection())

    def _notify_changed(self) -> None:
        if self._on_changed:
            try:
                self._on_changed()
            except Exception:
                pass

    def _delete_selected(self) -> None:
        names = self._selected_feature_names()
        if not names:
            messagebox.showinfo(
                "Pipeline Features",
                "Select one or more features to delete.",
                parent=self,
            )
            return
        preview = "\n".join(f"  • {n}" for n in names[:20])
        extra = f"\n  … and {len(names) - 20} more" if len(names) > 20 else ""
        ok = messagebox.askyesno(
            "Delete permanently?",
            f"Permanently remove {len(names)} Pipeline Feature(s) from Auto builds?\n\n"
            f"{preview}{extra}\n\n"
            "They will not appear in the catalogue and will not be regenerated.",
            parent=self,
        )
        if not ok:
            return
        try:
            retired = retire_pipeline_features(self._data_dir, names)
        except Exception as exc:
            messagebox.showerror("Pipeline Features", str(exc), parent=self)
            return
        self._reload()
        self._reload_recommendations()
        self._notify_changed()
        messagebox.showinfo(
            "Pipeline Features",
            f"Deleted {len(names)} feature(s).\n"
            f"Active catalogue now excludes {len(retired)} permanently deleted name(s).",
            parent=self,
        )

    def _remove_selected_recommendations(self) -> None:
        names = self._selected_recommendation_names()
        if not names:
            messagebox.showinfo(
                "Recommended for Removal",
                "Select one or more recommended features to remove.",
                parent=self,
            )
            return
        preview = "\n".join(f"  • {n}" for n in names[:20])
        extra = f"\n  … and {len(names) - 20} more" if len(names) > 20 else ""
        ok = messagebox.askyesno(
            "Remove Selected?",
            f"Permanently remove {len(names)} recommended Pipeline Feature(s)?\n\n"
            f"{preview}{extra}\n\n"
            "They will be excluded from the catalogue and will not be regenerated.",
            parent=self,
        )
        if not ok:
            return
        try:
            retire_pipeline_features(self._data_dir, names)
        except Exception as exc:
            messagebox.showerror("Recommended for Removal", str(exc), parent=self)
            return
        self._reload()
        self._reload_recommendations()
        self._notify_changed()
        messagebox.showinfo(
            "Recommended for Removal",
            f"Removed {len(names)} feature(s) from Pipeline Features.",
            parent=self,
        )

    def _ignore_selected_recommendations(self) -> None:
        names = self._selected_recommendation_names()
        if not names:
            messagebox.showinfo(
                "Recommended for Removal",
                "Select one or more recommendations to ignore.",
                parent=self,
            )
            return
        ok = messagebox.askyesno(
            "Ignore Recommendation?",
            f"Ignore removal recommendations for {len(names)} feature(s)?\n\n"
            "History is kept; they will no longer appear in this list.",
            parent=self,
        )
        if not ok:
            return
        try:
            from chain_replay_ml.production_validation import ignore_recommendation

            for name in names:
                ignore_recommendation(
                    self._data_dir,
                    name,
                    scope="pipeline",
                    reason="Ignored from Pipeline Features manager",
                )
        except Exception as exc:
            messagebox.showerror("Recommended for Removal", str(exc), parent=self)
            return
        self._reload_recommendations()
        messagebox.showinfo(
            "Recommended for Removal",
            f"Ignored {len(names)} recommendation(s).",
            parent=self,
        )


def open_pipeline_features_manager(
    master: tk.Misc,
    *,
    data_dir: str,
    on_changed: Callable[[], None] | None = None,
) -> PipelineFeaturesManagerDialog:
    return PipelineFeaturesManagerDialog(master, data_dir=data_dir, on_changed=on_changed)


__all__ = [
    "PipelineFeaturesManagerDialog",
    "open_pipeline_features_manager",
]
