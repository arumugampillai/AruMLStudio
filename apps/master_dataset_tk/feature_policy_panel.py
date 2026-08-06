"""Feature Policy inspector — side panel for feature selection workflows."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Callable

from chain_replay_ml.feature_policy import DEFAULT_GAP_MAX_SEC, load_feature_policy_registry

from . import feature_policy_format as fmt
from .build_config_prefs import load_build_config_prefs, save_build_config_prefs
from .fold_replay_widgets import place_toplevel_beside_main

_WARMUP_TAGS = {
    "low": "warmup_low",
    "mid": "warmup_mid",
    "high": "warmup_high",
    "none": "warmup_none",
}


class FeaturePolicyInspectorPanel(ttk.Frame):
    """List + detail panel for feature policy metadata."""

    # Set True to open Feature Detail on single click (disabled — use double-click).
    _OPEN_FEATURE_DETAIL_ON_CLICK = False

    def __init__(
        self,
        master: tk.Misc,
        *,
        sampling_interval_sec: float = 10.0,
        gap_max_sec: float = DEFAULT_GAP_MAX_SEC,
        on_selection_change: Callable[[str | None], None] | None = None,
        chart_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(master, padding=6, **kwargs)
        self._interval = float(sampling_interval_sec)
        self._gap_max = float(gap_max_sec)
        self._on_selection_change = on_selection_change
        self._chart_dir = chart_dir
        self._features_by_name: dict[str, dict[str, Any]] = {}
        self._feature_names: list[str] = []
        self._used_by_index: dict[str, list[str]] = {}
        self._selected_feature: str | None = None
        self._feature_detail_win: tk.Toplevel | None = None
        self._summary_var = tk.StringVar(value="")
        self._category_var = tk.StringVar(value="all|All")
        self._feature_search_var = tk.StringVar()
        self._build_ui()

    def _build_ui(self) -> None:
        hdr = ttk.Frame(self)
        hdr.pack(fill="x", pady=(0, 4))
        ttk.Label(hdr, text="Dataset Feature Policy", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Label(hdr, textvariable=self._summary_var, foreground="#666").pack(side="right")

        self._root_notebook = ttk.Notebook(self)
        self._root_notebook.pack(fill="both", expand=True)

        policy_tab = ttk.Frame(self._root_notebook, padding=0)
        self._root_notebook.add(policy_tab, text="Feature Policy")

        paned = ttk.Panedwindow(policy_tab, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True)
        self._policy_paned = paned

        left = ttk.LabelFrame(paned, text="Features", padding=4)
        paned.add(left, weight=60)
        filter_row = ttk.Frame(left)
        filter_row.pack(fill="x", pady=(0, 4))
        ttk.Label(filter_row, text="Category").pack(side="left")
        self._category_combo = ttk.Combobox(
            filter_row, textvariable=self._category_var, width=16, state="readonly",
        )
        self._category_combo["values"] = [
            f"{key}|{label}" for key, label in fmt.CATEGORY_FILTER_OPTIONS
        ]
        self._category_combo.pack(side="left", padx=(4, 0))

        search_row = ttk.Frame(left)
        search_row.pack(fill="x", pady=(0, 4))
        ttk.Label(search_row, text="Search").pack(side="left")
        search_entry = ttk.Entry(search_row, textvariable=self._feature_search_var, width=24)
        search_entry.pack(side="left", fill="x", expand=True, padx=(4, 0))

        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill="both", expand=True)
        cols = ("feature", "category", "warmup", "dependencies", "used_by")
        self._tree = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="browse", height=14)
        for col, w, title in (
            ("feature", 180, "Feature"),
            ("category", 68, "Category"),
            ("warmup", 150, "Warm-up"),
            ("dependencies", 120, "Depends On"),
            ("used_by", 56, "Used By"),
        ):
            self._tree.heading(col, text=title)
            self._tree.column(col, width=w, minwidth=40)
        self._tree.tag_configure("warmup_low", foreground="#2e7d32")
        self._tree.tag_configure("warmup_mid", foreground="#e65100")
        self._tree.tag_configure("warmup_high", foreground="#c62828")
        self._tree.tag_configure("warmup_none", foreground="#666666")
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self._tree.bind("<ButtonRelease-1>", self._on_tree_click)
        self._tree.bind("<Double-1>", self._on_tree_add_data_filter_click)
        detail_btn_row = ttk.Frame(left)
        detail_btn_row.pack(fill="x", pady=(4, 0))
        ttk.Label(
            detail_btn_row,
            text="Double-click row → Data Filter",
            foreground="#888",
            font=("Segoe UI", 8),
        ).pack(side="left")
        ttk.Button(detail_btn_row, text="Download CSV", command=self._download_features_csv).pack(
            side="right",
            padx=(6, 0),
        )
        ttk.Button(detail_btn_row, text="Feature Detail…", command=self._on_tree_open_detail_click).pack(
            side="right",
        )

        right = ttk.LabelFrame(paned, text="Policy Detail", padding=4)
        paned.add(right, weight=40)
        self._policy_paned_split_done = False
        paned.bind("<Configure>", self._apply_policy_paned_split)
        self.after_idle(self._apply_policy_paned_split)
        self._detail_notebook = ttk.Notebook(right)
        self._detail_notebook.pack(fill="both", expand=True)

        summary_frame = ttk.Frame(self._detail_notebook, padding=2)
        self._detail_notebook.add(summary_frame, text="Summary")
        self._summary_text = scrolledtext.ScrolledText(
            summary_frame, wrap="word", font=("Consolas", 9), height=22,
        )
        self._summary_text.pack(fill="both", expand=True)

        detail_frame = ttk.Frame(self._detail_notebook, padding=2)
        self._detail_notebook.add(detail_frame, text="Detail")
        self._detail = scrolledtext.ScrolledText(
            detail_frame, wrap="word", font=("Consolas", 9), height=22,
        )
        self._detail.pack(fill="both", expand=True)
        self._set_readonly_text(self._detail, "Select a feature in the list for full policy detail.")

        sim_tab = ttk.Frame(self._root_notebook, padding=0)
        self._root_notebook.add(sim_tab, text="Warm-up Simulator")
        from .warmup_simulator_panel import WarmupSimulatorPanel

        self._simulator = WarmupSimulatorPanel(
            sim_tab,
            chart_dir=self._chart_dir,
            feature_names=self._feature_names,
            features_by_name=self._features_by_name,
            sampling_interval_sec=self._interval,
            gap_max_sec=self._gap_max,
            get_selected_feature=lambda: self._selected_feature,
        )
        self._simulator.pack(fill="both", expand=True)

        self._category_var.trace_add("write", lambda *_a: self._on_category_filter())
        self._feature_search_var.trace_add("write", lambda *_a: self._on_feature_search())
        self._load_policy_prefs()

    def _apply_policy_paned_split(self, _event: tk.Event | None = None) -> None:
        if getattr(self, "_policy_paned_split_done", False):
            return
        paned = getattr(self, "_policy_paned", None)
        if paned is None:
            return
        width = paned.winfo_width()
        if width <= 1:
            return
        paned.sashpos(0, max(200, int(width * 0.60)))
        self._policy_paned_split_done = True

    def _policy_prefs(self) -> dict[str, Any]:
        if not self._chart_dir:
            return {}
        doc = load_build_config_prefs(self._chart_dir) or {}
        fp = doc.get("feature_policy")
        return fp if isinstance(fp, dict) else {}

    def _save_policy_prefs(self, patch: dict[str, Any]) -> None:
        if not self._chart_dir:
            return
        current = self._policy_prefs()
        save_build_config_prefs(self._chart_dir, {
            "feature_policy": {**current, **patch},
        })

    def _load_policy_prefs(self) -> None:
        prefs = self._policy_prefs()
        cat = str(prefs.get("category_filter") or "").strip()
        if cat and "|" in cat:
            self._category_var.set(cat)
        search = str(prefs.get("feature_search") or "").strip()
        if search:
            self._feature_search_var.set(search)

    def set_sampling_interval(self, sec: float) -> None:
        self._interval = max(0.001, float(sec))
        if hasattr(self, "_simulator"):
            self._simulator.set_sampling_interval(sec)

    def set_chart_dir(self, chart_dir: str | None) -> None:
        self._chart_dir = chart_dir
        if hasattr(self, "_simulator"):
            self._simulator.set_chart_dir(chart_dir)

    def _merge_registry_metadata(self) -> None:
        if not self._feature_names:
            return
        try:
            reg = load_feature_policy_registry(feature_names=self._feature_names)
            for name in self._feature_names:
                meta = reg.get(name)
                if not meta:
                    continue
                existing = self._features_by_name.get(name) or {}
                merged = {**existing, **meta.as_dict(), "name": name}
                self._features_by_name[name] = merged
        except Exception:
            pass

    def _rebuild_used_by_index(self) -> None:
        self._used_by_index = fmt.build_used_by_index(
            self._feature_names, self._features_by_name,
        )

    def set_features(
        self,
        feature_names: list[str],
        *,
        features_by_name: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        names = list(dict.fromkeys(feature_names))
        meta = dict(features_by_name or {})
        if self._chart_dir:
            from .feature_registry_service import filter_active_registry_names

            names = filter_active_registry_names(self._chart_dir, names)
            meta = {k: v for k, v in meta.items() if k in set(names)}
        self._feature_names = names
        self._features_by_name = meta
        self._merge_registry_metadata()
        self._rebuild_used_by_index()
        restore = self._policy_prefs().get("selected_feature")
        self._selected_feature = str(restore).strip() if restore else None
        if self._selected_feature and self._selected_feature not in self._feature_names:
            self._selected_feature = None
        self._render_list()
        self._show_summary()
        if self._selected_feature:
            self.select_feature(self._selected_feature)
        if hasattr(self, "_simulator"):
            self._simulator.set_features(
                self._feature_names,
                features_by_name=self._features_by_name,
            )
            self._simulator.load_prefs()

    def _policy_for(self, name: str) -> dict[str, Any]:
        feat = self._features_by_name.get(name) or {"name": name}
        return fmt.normalize_policy({**feat, "name": name})

    def _feature_category_id(self, name: str) -> str:
        return str(self._policy_for(name).get("feature_category") or "raw").lower()

    def _filtered_feature_names(self) -> list[str]:
        cat_key = fmt.category_filter_key(self._category_var.get())
        names = list(self._feature_names)
        if cat_key != "all":
            names = [n for n in names if self._feature_category_id(n) == cat_key]
        query = self._feature_search_var.get().strip().lower()
        if query:
            names = [n for n in names if query in n.lower()]
        return names

    def _on_feature_search(self) -> None:
        self._save_policy_prefs({"feature_search": self._feature_search_var.get().strip()})
        self._render_list()
        self._show_summary()

    def _show_summary(self) -> None:
        names = self._filtered_feature_names()
        total = len(self._feature_names)
        shown = len(names)
        if shown == total:
            self._summary_var.set(f"{total} features")
        else:
            self._summary_var.set(f"{shown} of {total} features")
        text = fmt.format_policy_summary_tab(
            names,
            self._features_by_name,
            sampling_interval_sec=self._interval,
            gap_max_sec=self._gap_max,
        )
        self._set_readonly_text(self._summary_text, text)
        if self._selected_feature and self._selected_feature in names:
            self._show_feature(self._selected_feature)
            return
        self._detail_notebook.select(0)

    def _on_category_filter(self) -> None:
        self._save_policy_prefs({"category_filter": self._category_var.get()})
        self._render_list()
        self._show_summary()

    def select_feature(self, name: str | None) -> None:
        if not name:
            return
        self._selected_feature = name
        for item in self._tree.get_children():
            if self._tree.item(item, "values")[0] == name:
                self._tree.selection_set(item)
                self._tree.focus(item)
                self._tree.see(item)
                self._show_feature(name)
                if self._OPEN_FEATURE_DETAIL_ON_CLICK:
                    self._on_tree_open_detail(name)
                return

    def _used_by_cell(self, name: str, pol: dict[str, Any]) -> str:
        users = fmt.resolve_used_by(name, pol, self._used_by_index)
        if not users:
            return "—"
        if len(users) == 1:
            return users[0][:18]
        return str(len(users))

    def _row_for_feature(self, name: str) -> tuple[str, str, str, str, str, str, int]:
        pol = self._policy_for(name)
        cat = fmt.category_label(pol.get("feature_category"))
        samples = fmt.warmup_samples(pol)
        warmup = fmt.format_warmup_cell(
            pol,
            feature_name=name,
            sampling_interval_sec=self._interval,
            features_by_name=self._features_by_name,
        )
        deps = ", ".join(fmt.display_dependencies(pol)[:3])
        if len(fmt.display_dependencies(pol)) > 3:
            deps += "…"
        used_by = self._used_by_cell(name, pol)
        tier = fmt.warmup_tier(samples)
        return name, cat, warmup, deps or "—", used_by, _WARMUP_TAGS[tier], samples

    def _render_list(self) -> None:
        tree = getattr(self, "_tree", None)
        if tree is None:
            return
        for item in tree.get_children():
            tree.delete(item)
        rows: list[tuple[str, str, str, str, str, str, int]] = []
        for name in self._filtered_feature_names():
            rows.append(self._row_for_feature(name))
        rows.sort(key=lambda r: (-r[6], r[0]))
        for name, cat, warmup, deps, used_by, tag, _ in rows:
            self._tree.insert(
                "", "end", iid=name,
                values=(name, cat, warmup, deps, used_by),
                tags=(tag,),
            )
        total = len(self._feature_names)
        shown = len(rows)
        if shown == total:
            self._summary_var.set(f"{total} features")
        else:
            self._summary_var.set(f"{shown} of {total} features")

    def _on_tree_click(self, event: tk.Event | None = None) -> None:
        if event is not None:
            iid = self._tree.identify_row(event.y)
            if iid:
                self._tree.selection_set(iid)
                self._tree.focus(iid)
                name = str(iid)
                self._selected_feature = name
                self._save_policy_prefs({"selected_feature": name})
                self._show_feature(name)
                if self._on_selection_change:
                    self._on_selection_change(name)
                if self._OPEN_FEATURE_DETAIL_ON_CLICK:
                    self._on_tree_open_detail(name)
                return
        sel = self._tree.selection()
        if sel:
            self._on_tree_select()

    def _on_tree_add_data_filter_click(self, event: tk.Event | None = None) -> None:
        name: str | None = None
        if event is not None:
            iid = self._tree.identify_row(event.y)
            if iid:
                name = str(iid)
        if not name:
            sel = self._tree.selection()
            if sel:
                name = str(sel[0])
        if not name:
            return
        if hasattr(self, "_simulator"):
            self._simulator.add_data_filter_feature(name)
            self._root_notebook.select(1)
            self._simulator.focus_data_filter_tab()

    def _on_tree_open_detail_click(self, event: tk.Event | None = None) -> None:
        if event is not None:
            iid = self._tree.identify_row(event.y)
            if iid:
                self._on_tree_open_detail(str(iid))
                return
        sel = self._tree.selection()
        if sel:
            self._on_tree_open_detail(str(sel[0]))

    def _download_features_csv(self) -> None:
        names = self._filtered_feature_names()
        if not names:
            messagebox.showinfo(
                "Download CSV",
                "No features to download.\n\nClear the category/search filter or load features first.",
                parent=self.winfo_toplevel(),
            )
            return
        csv_text = fmt.feature_policy_csv_text(
            names,
            self._features_by_name,
            sampling_interval_sec=self._interval,
            gap_max_sec=self._gap_max,
        )
        n = len(names)
        total = len(self._feature_names)
        suffix = f"_{n}of{total}" if n != total else f"_{n}"
        path = filedialog.asksaveasfilename(
            parent=self.winfo_toplevel(),
            title="Save Feature Policy CSV",
            defaultextension=".csv",
            initialfile=f"feature_policy{suffix}.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                fh.write(csv_text)
            messagebox.showinfo(
                "Download CSV",
                f"Saved {n:,} feature(s) to\n{path}",
                parent=self.winfo_toplevel(),
            )
        except OSError as exc:
            messagebox.showerror(
                "Download CSV",
                f"Could not save file:\n{exc}",
                parent=self.winfo_toplevel(),
            )

    def _clear_feature_detail_win(self) -> None:
        self._feature_detail_win = None

    def _on_tree_open_detail(self, name: str | None = None) -> None:
        feature = name or self._selected_feature
        if not feature:
            return
        from .feature_detail_panel import open_feature_detail_window

        if not self._feature_detail_win:
            pass
        else:
            try:
                if not self._feature_detail_win.winfo_exists():
                    self._feature_detail_win = None
            except tk.TclError:
                self._feature_detail_win = None

        self._feature_detail_win = open_feature_detail_window(
            self,
            feature,
            chart_dir=self._chart_dir,
            features_by_name=self._features_by_name,
            reuse_window=self._feature_detail_win,
            on_destroy=self._clear_feature_detail_win,
        )

    def _on_tree_select(self, _event: tk.Event | None = None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        name = str(sel[0])
        self._selected_feature = name
        self._save_policy_prefs({"selected_feature": name})
        self._show_feature(name)
        if self._on_selection_change:
            self._on_selection_change(name)
        if self._OPEN_FEATURE_DETAIL_ON_CLICK:
            self._on_tree_open_detail(name)

    def _show_feature(self, name: str) -> None:
        feat = self._features_by_name.get(name) or {"name": name}
        text = fmt.format_feature_policy_detail(
            {**feat, "name": name},
            sampling_interval_sec=self._interval,
            gap_max_sec=self._gap_max,
            features_by_name=self._features_by_name,
            used_by_index=self._used_by_index,
        )
        self._set_readonly_text(self._detail, text)
        self._detail_notebook.select(1)

    @staticmethod
    def _set_readonly_text(widget: scrolledtext.ScrolledText, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state="disabled")


def open_feature_policy_window(
    master: tk.Misc,
    *,
    title: str,
    feature_names: list[str],
    features_by_name: dict[str, dict[str, Any]] | None = None,
    sampling_interval_sec: float = 10.0,
    gap_max_sec: float = DEFAULT_GAP_MAX_SEC,
    select_feature: str | None = None,
    chart_dir: str | None = None,
) -> tk.Toplevel:
    """Open a companion window beside the main app with policy inspector."""
    if chart_dir is None:
        w: tk.Misc | None = master
        while w is not None:
            if hasattr(w, "chart_dir"):
                chart_dir = str(getattr(w, "chart_dir"))
                break
            w = w.master  # type: ignore[assignment]

    win = tk.Toplevel(master)
    win.title(title)
    win.transient(master.winfo_toplevel())
    panel = FeaturePolicyInspectorPanel(
        win,
        sampling_interval_sec=sampling_interval_sec,
        gap_max_sec=gap_max_sec,
        chart_dir=chart_dir,
    )
    panel.pack(fill="both", expand=True)
    panel.set_features(feature_names, features_by_name=features_by_name)
    if select_feature:
        panel.select_feature(select_feature)
    win.update_idletasks()
    place_toplevel_beside_main(win, master)
    panel._policy_paned_split_done = False
    panel.after_idle(panel._apply_policy_paned_split)
    win.lift()
    win.focus_force()
    return win
