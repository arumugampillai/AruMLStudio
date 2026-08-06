"""Feature Registry page — mirrors web Feature Registry tab (standalone)."""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any

from . import feature_registry_format as fmt
from . import feature_policy_format as pol_fmt
from . import feature_registry_service as svc
from .lazy_panel import LazyLoadMixin


class FeatureRegistryPanel(ttk.Frame, LazyLoadMixin):
    """Browse/edit feature catalog — search, filters, detail, import/export, projects."""

    def __init__(self, master: tk.Misc, *, chart_dir: str) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._catalog: dict[str, Any] | None = None
        self._selected_name: str | None = None
        self._search_var = tk.StringVar()
        self._project_var = tk.StringVar(value="all")
        self._status_var = tk.StringVar(value="all")
        self._group_var = tk.StringVar(value="all")
        self._category_var = tk.StringVar(value="all|All")
        self._show_disabled_var = tk.BooleanVar(value=False)
        self._meta_var = tk.StringVar(value="—")
        self._list_meta_var = tk.StringVar(value="—")
        self._job_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        self._parity_busy = False
        self._pm_window: tk.Toplevel | None = None
        self._build_ui()
        self._lazy_init()
        self._search_var.trace_add("write", lambda *_: self._render_list())
        self._project_var.trace_add("write", lambda *_: self._on_project_changed())
        self._status_var.trace_add("write", lambda *_: self._render_list())
        self._group_var.trace_add("write", lambda *_: self._render_list())
        self._category_var.trace_add("write", lambda *_: self._render_list())
        self._show_disabled_var.trace_add("write", lambda *_: self._render_list())

    def set_chart_dir(self, chart_dir: str) -> None:
        self.chart_dir = chart_dir

    def on_show(self) -> None:
        self.refresh(lazy=True)

    def poll_jobs(self) -> None:
        try:
            while True:
                msg = self._job_queue.get_nowait()
                self._handle_job(msg)
        except queue.Empty:
            pass

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=6)
        toolbar.pack(fill="x")

        row1 = ttk.Frame(toolbar)
        row1.pack(fill="x", pady=(0, 4))
        ttk.Label(row1, text="Search").pack(side="left")
        ttk.Entry(row1, textvariable=self._search_var, width=28).pack(side="left", padx=(4, 12))
        ttk.Label(row1, text="Project").pack(side="left")
        self._project_combo = ttk.Combobox(row1, textvariable=self._project_var, width=18, state="readonly")
        self._project_combo.pack(side="left", padx=4)
        ttk.Button(row1, text="Manage Projects", command=self._open_project_manager).pack(
            side="left", padx=4
        )
        ttk.Label(row1, textvariable=self._meta_var, foreground="#666").pack(side="right", padx=4)
        ttk.Button(row1, text="Refresh", command=self.refresh).pack(side="right")

        row2 = ttk.Frame(toolbar)
        row2.pack(fill="x", pady=(0, 4))
        ttk.Label(row2, text="Status").pack(side="left")
        self._status_combo = ttk.Combobox(row2, textvariable=self._status_var, width=16, state="readonly")
        self._status_combo.pack(side="left", padx=(4, 12))
        ttk.Label(row2, text="Domain").pack(side="left")
        self._group_combo = ttk.Combobox(row2, textvariable=self._group_var, width=28, state="readonly")
        self._group_combo.pack(side="left", padx=4)
        ttk.Label(row2, text="Category").pack(side="left", padx=(8, 0))
        self._category_combo = ttk.Combobox(row2, textvariable=self._category_var, width=14, state="readonly")
        self._category_combo.pack(side="left", padx=4)

        row3 = ttk.Frame(toolbar)
        row3.pack(fill="x")
        for label, cmd in (
            ("+ Add Feature", self._open_add_dialog),
            ("Import JSON", self._open_import_dialog),
            ("Export JSON", self._export_json),
            ("Audit", self._run_parity_audit),
            ("Rules", self._show_parity_rules),
        ):
            ttk.Button(row3, text=label, command=cmd).pack(side="left", padx=2)
        self._toggle_active_btn = ttk.Button(
            row3, text="Disable", command=self._toggle_selected_active, state="disabled",
        )
        self._toggle_active_btn.pack(side="left", padx=(8, 2))
        ttk.Checkbutton(
            row3,
            text="Show disabled",
            variable=self._show_disabled_var,
        ).pack(side="left", padx=4)
        self._delete_btn = ttk.Button(row3, text="Delete Feature", command=self._open_delete_dialog, state="disabled")
        self._delete_btn.pack(side="left", padx=8)
        ttk.Button(
            row3,
            text="Recommended Features for Retirement",
            command=self._open_retirement_recommendations,
        ).pack(side="left", padx=8)

        paned = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.LabelFrame(paned, text="Features", padding=4)
        paned.add(left, weight=2)
        hdr = ttk.Frame(left)
        hdr.pack(fill="x")
        ttk.Label(hdr, textvariable=self._list_meta_var, foreground="#888").pack(side="right")
        tree_frame = ttk.Frame(left)
        tree_frame.pack(fill="both", expand=True)
        cols = ("id", "feature", "domain", "category", "active", "status")
        self._tree = ttk.Treeview(tree_frame, columns=cols, show="headings", selectmode="browse")
        for col, w, title in (
            ("id", 72, "ID"),
            ("feature", 200, "Feature"),
            ("domain", 140, "Domain"),
            ("category", 72, "Category"),
            ("active", 56, "Active"),
            ("status", 120, "Status"),
        ):
            self._tree.heading(col, text=title)
            self._tree.column(col, width=w, minwidth=48)
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tree_frame.rowconfigure(0, weight=1)
        tree_frame.columnconfigure(0, weight=1)
        self._tree.bind("<<TreeviewSelect>>", self._on_select)
        self._tree.bind("<Double-1>", lambda _e: self._open_feature_ref_from_row())

        right = ttk.LabelFrame(paned, text="Feature Detail", padding=4)
        paned.add(right, weight=3)
        self._detail_text = scrolledtext.ScrolledText(right, wrap="word", font=("Consolas", 9))
        self._detail_text.pack(fill="both", expand=True)
        self._detail_text.configure(state="disabled")
        self._set_detail("Select a feature to view details, dependencies, and usage.")

    def _set_detail(self, text: str) -> None:
        self._detail_text.configure(state="normal")
        self._detail_text.delete("1.0", tk.END)
        self._detail_text.insert("1.0", text)
        self._detail_text.configure(state="disabled")

    def refresh(self, *, lazy: bool = True) -> None:
        if lazy:
            self.lazy_load(
                load=lambda: svc.load_catalog(self.chart_dir),
                apply=self._apply_catalog,
                message="Loading feature registry…",
                status_var=self._meta_var,
                on_error=self._on_load_error,
            )
            return
        try:
            catalog = svc.load_catalog(self.chart_dir)
        except Exception as exc:
            messagebox.showerror("Feature Registry", f"Failed to load:\n{exc}")
            self._catalog = None
            return
        self._apply_catalog(catalog)

    def _on_load_error(self, exc: Exception) -> None:
        self._catalog = None
        self._meta_var.set(f"Error: {exc}")
        messagebox.showerror("Feature Registry", f"Failed to load:\n{exc}")

    def _apply_catalog(self, catalog: dict[str, Any] | None) -> None:
        self._catalog = catalog
        if catalog is None:
            return
        self._populate_filters()
        self._meta_var.set(fmt.format_registry_meta(self._catalog))
        self._render_list()

    def _populate_filters(self) -> None:
        cat = self._catalog or {}
        projects = cat.get("projects") or []
        self._project_combo["values"] = ["all"] + [
            f"{p.get('id')}|{p.get('label') or p.get('id')}" for p in projects
        ]
        cur = self._project_var.get()
        if cur not in self._project_combo["values"]:
            self._project_var.set("all")

        status_opts = ["all"] + [
            f"{o.get('id')}|{o.get('label')}" for o in (cat.get("status_options") or [])
        ]
        if len(status_opts) == 1:
            status_opts += [
                "implemented|Implemented",
                "planned|Planned",
                "in_progress|In Progress",
                "not_implemented|Not Implemented",
            ]
        self._status_combo["values"] = status_opts
        if self._status_var.get() not in status_opts:
            self._status_var.set("all")

        domains = cat.get("domains") or []
        if domains:
            group_opts = ["all|All"] + [
                f"{d.get('id')}|{d.get('chip_label') or d.get('label')}"
                for d in domains
            ]
        else:
            groups = self._visible_groups()
            group_opts = ["all|All"] + [
                f"{g.get('filter') or g.get('id')}|{g.get('filter') or g.get('label')}"
                for g in groups
            ]
        seen: set[str] = set()
        deduped = []
        for g in group_opts:
            key = g.split("|", 1)[0]
            if key in seen:
                continue
            seen.add(key)
            deduped.append(g)
        self._group_combo["values"] = deduped
        if self._group_var.get() not in deduped:
            self._group_var.set("all|All")

        cat_opts = [f"{k}|{v}" for k, v in pol_fmt.CATEGORY_FILTER_OPTIONS]
        self._category_combo["values"] = cat_opts
        if self._category_var.get() not in cat_opts:
            self._category_var.set("all|All")

    def _filter_key(self, combo_val: str) -> str:
        return combo_val.split("|", 1)[0] if "|" in combo_val else combo_val

    def _current_project(self) -> dict[str, Any] | None:
        key = self._filter_key(self._project_var.get())
        if key == "all":
            return None
        for p in (self._catalog or {}).get("projects") or []:
            if p.get("id") == key:
                return p
        return None

    def _project_feature_names(self) -> set[str] | None:
        proj = self._current_project()
        if not proj:
            return None
        if "feature_names" in proj or "enabled_features" in proj:
            return {
                str(n)
                for n in (proj.get("feature_names") or proj.get("enabled_features") or [])
            }
        return None

    def _project_group_ids(self) -> set[str] | None:
        proj = self._current_project()
        if not proj:
            return None
        return {str(g) for g in (proj.get("group_ids") or [])}

    def _visible_groups(self) -> list[dict[str, Any]]:
        all_groups = (self._catalog or {}).get("groups") or []
        names = self._project_feature_names()
        if names is not None:
            if not names:
                return []
            group_ids: set[str] = set()
            for f in (self._catalog or {}).get("features") or []:
                if str(f.get("name") or "") not in names:
                    continue
                gid = f.get("group_id") or f.get("group_filter")
                if gid:
                    group_ids.add(str(gid))
            return [g for g in all_groups if g.get("id") in group_ids]
        ids = self._project_group_ids()
        if not ids:
            return all_groups
        return [g for g in all_groups if g.get("id") in ids]

    def _feature_in_project(self, f: dict[str, Any]) -> bool:
        names = self._project_feature_names()
        if names is not None:
            return str(f.get("name") or "") in names
        ids = self._project_group_ids()
        if not ids:
            return True
        return f.get("group_id") in ids or f.get("group_filter") in ids

    def _feature_is_active(self, f: dict[str, Any]) -> bool:
        return bool(f.get("registry_active", True))

    def _filtered_features(self) -> list[dict[str, Any]]:
        rows = (self._catalog or {}).get("features") or []
        q = self._search_var.get().strip().lower()
        status = self._filter_key(self._status_var.get())
        group = self._filter_key(self._group_var.get())
        category = pol_fmt.category_filter_key(self._category_var.get())
        show_disabled = bool(self._show_disabled_var.get())
        out: list[dict[str, Any]] = []
        for f in rows:
            if not self._feature_in_project(f):
                continue
            active = self._feature_is_active(f)
            # Checkbox on → disabled only; off → active only (unless Group=disabled).
            if show_disabled:
                if active:
                    continue
            elif not active and group != "disabled":
                continue
            if active and group == "disabled":
                continue
            if status != "all" and f.get("implementation_status") != status:
                continue
            if group != "all":
                if group == "disabled":
                    if active:
                        continue
                elif (
                    f.get("primary_domain") != group
                    and f.get("domain_filter") != group
                    and f.get("group_filter") != group
                    and f.get("group_id") != group
                ):
                    continue
            if category != "all" and str(f.get("feature_category") or "") != category:
                continue
            if q:
                hay = " ".join(
                    str(x or "")
                    for x in (
                        f.get("name"),
                        f.get("display_name"),
                        f.get("group"),
                        f.get("domain"),
                        f.get("primary_domain_label"),
                        f.get("category"),
                        f.get("description"),
                        f.get("why_needed"),
                        f.get("formula"),
                        " ".join(f.get("tags") or []),
                    )
                ).lower()
                if q not in hay:
                    continue
            out.append(f)
        return out

    def _render_list(self) -> None:
        if not self._catalog:
            return
        filtered = self._filtered_features()
        project_ids = self._project_group_ids()
        all_rows = (self._catalog.get("features") or [])
        show_disabled = bool(self._show_disabled_var.get())
        scope_total = sum(
            1
            for f in all_rows
            if self._feature_in_project(f)
            and (
                (not self._feature_is_active(f))
                if show_disabled
                else self._feature_is_active(f)
            )
        )
        if len(filtered) == scope_total:
            self._list_meta_var.set(f"{scope_total} features")
        else:
            self._list_meta_var.set(f"{len(filtered)} of {scope_total}")

        self._tree.delete(*self._tree.get_children())
        names: list[str] = []
        for f in filtered:
            name = f.get("name") or ""
            if not name:
                continue
            names.append(name)
            disp = f.get("display_name") or ""
            feat_col = f"{name}\n{disp}" if disp else name
            self._tree.insert(
                "",
                tk.END,
                iid=name,
                values=(
                    f.get("feature_id") or "—",
                    feat_col,
                    f.get("primary_domain_label") or f.get("domain") or f.get("group_filter") or "—",
                    f.get("feature_category") or "—",
                    "Yes" if self._feature_is_active(f) else "No",
                    f.get("implementation_label") or f.get("implementation_status") or "—",
                ),
            )

        if self._selected_name and self._selected_name in names:
            self._tree.selection_set(self._selected_name)
        elif names:
            self._selected_name = names[0]
            self._tree.selection_set(names[0])
            self._show_detail(names[0])
        else:
            self._selected_name = None
            self._set_detail("No features match filters.")
        self._delete_btn.configure(state="normal" if self._selected_name else "disabled")
        self._update_toggle_active_btn()

    def _update_toggle_active_btn(self) -> None:
        if not hasattr(self, "_toggle_active_btn"):
            return
        if not self._selected_name:
            self._toggle_active_btn.configure(state="disabled", text="Disable")
            return
        feat = next(
            (r for r in (self._catalog or {}).get("features") or [] if r.get("name") == self._selected_name),
            None,
        )
        if not feat:
            self._toggle_active_btn.configure(state="disabled", text="Disable")
            return
        if self._feature_is_active(feat):
            self._toggle_active_btn.configure(state="normal", text="Disable")
        else:
            self._toggle_active_btn.configure(state="normal", text="Enable")

    def _toggle_selected_active(self) -> None:
        if not self._selected_name:
            return
        feat = next(
            (r for r in (self._catalog or {}).get("features") or [] if r.get("name") == self._selected_name),
            None,
        )
        if not feat:
            return
        currently_active = self._feature_is_active(feat)
        home_group = str(feat.get("group_id") or "") if currently_active else None
        try:
            svc.set_feature_registry_active(
                self.chart_dir,
                self._selected_name,
                active=not currently_active,
                home_group_id=home_group,
            )
            self.refresh()
            self._render_list()
            self._show_detail(self._selected_name)
        except Exception as exc:
            messagebox.showerror("Feature Registry", str(exc))

    def _open_retirement_recommendations(self) -> None:
        from .feature_registry_retirement_dialog import open_recommended_retirement_dialog

        open_recommended_retirement_dialog(
            self,
            chart_dir=self.chart_dir,
            on_changed=self.refresh,
        )
    def _on_select(self, _event: tk.Event | None = None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        name = sel[0]
        if name == self._selected_name:
            return
        self._selected_name = name
        self._show_detail(name)
        self._delete_btn.configure(state="normal")
        self._update_toggle_active_btn()

    def _show_detail(self, name: str) -> None:
        f = next((r for r in (self._catalog or {}).get("features") or [] if r.get("name") == name), None)
        self._set_detail(fmt.format_feature_detail(f or {}, self._catalog))

    def _feature_by_ref(self, ref: str) -> dict[str, Any] | None:
        if not ref or not self._catalog:
            return None
        index = self._catalog.get("feature_id_index") or {}
        for f in self._catalog.get("features") or []:
            if f.get("feature_id") == ref or f.get("name") == ref:
                return f
        name = index.get(ref)
        if name:
            return next((f for f in self._catalog.get("features") or [] if f.get("name") == name), None)
        return None

    def _open_feature_ref_from_row(self) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        item = self._tree.item(sel[0])
        fid = (item.get("values") or [None])[0]
        if fid and fid != "—":
            f = self._feature_by_ref(str(fid))
            if f and f.get("name"):
                self._selected_name = f["name"]
                self._tree.selection_set(f["name"])
                self._show_detail(f["name"])
                from .feature_detail_panel import open_feature_detail_window

                open_feature_detail_window(
                    self,
                    f["name"],
                    chart_dir=self.chart_dir,
                    features_by_name={
                        str(r.get("name")): r
                        for r in (self._catalog or {}).get("features") or []
                        if r.get("name")
                    },
                )

    def _on_project_changed(self) -> None:
        self._group_var.set("all|All")
        self._populate_filters()
        self._render_list()

    def _open_add_dialog(self) -> None:
        dlg = tk.Toplevel(self)
        dlg.title("Add Planned Feature")
        dlg.geometry("520x620")
        dlg.transient(self.winfo_toplevel())
        dlg.grab_set()

        fields: dict[str, Any] = {}
        body = ttk.Frame(dlg, padding=8)
        body.pack(fill="both", expand=True)

        def add_row(label: str, widget: tk.Widget) -> None:
            row = ttk.Frame(body)
            row.pack(fill="x", pady=3)
            ttk.Label(row, text=label, width=18).pack(side="left", anchor="n")
            widget.pack(side="left", fill="x", expand=True)

        name_var = tk.StringVar()
        add_row("Feature Name", ttk.Entry(body, textvariable=name_var))
        group_var = tk.StringVar()
        groups = self._visible_groups() or (self._catalog or {}).get("groups") or []
        group_combo = ttk.Combobox(
            body,
            textvariable=group_var,
            values=[g.get("id") or "" for g in groups],
            state="readonly",
        )
        if groups:
            group_var.set(groups[0].get("id") or "")
        add_row("Group", group_combo)

        desc = scrolledtext.ScrolledText(body, height=3, font=("Segoe UI", 9))
        add_row("Description", desc)
        why = scrolledtext.ScrolledText(body, height=2, font=("Segoe UI", 9))
        add_row("Why needed", why)
        formula = scrolledtext.ScrolledText(body, height=2, font=("Segoe UI", 9))
        add_row("Formula", formula)
        inputs_var = tk.StringVar()
        add_row("Inputs (csv)", ttk.Entry(body, textvariable=inputs_var))
        deps_var = tk.StringVar()
        add_row("Dependencies (csv)", ttk.Entry(body, textvariable=deps_var))
        dtype_var = tk.StringVar(value="float")
        add_row("Data type", ttk.Combobox(body, textvariable=dtype_var, values=["float", "int", "bool", "string"], state="readonly"))
        range_var = tk.StringVar()
        add_row("Expected range", ttk.Entry(body, textvariable=range_var))
        status_var = tk.StringVar(value="planned")
        add_row("Status", ttk.Combobox(
            body, textvariable=status_var,
            values=["planned", "in_progress", "not_implemented"], state="readonly",
        ))
        priority_var = tk.StringVar(value="medium")
        add_row("Priority", ttk.Combobox(body, textvariable=priority_var, values=["low", "medium", "high"], state="readonly"))
        owner_var = tk.StringVar()
        add_row("Owner", ttk.Entry(body, textvariable=owner_var))
        notes = scrolledtext.ScrolledText(body, height=2, font=("Segoe UI", 9))
        add_row("Notes", notes)

        def save() -> None:
            name = name_var.get().strip()
            if not name:
                messagebox.showerror("Add Feature", "Feature name is required.", parent=dlg)
                return
            payload = {
                "name": name,
                "group": group_var.get() or "advanced",
                "description": desc.get("1.0", tk.END).strip(),
                "why_needed": why.get("1.0", tk.END).strip(),
                "formula": formula.get("1.0", tk.END).strip(),
                "inputs_required": svc.split_csv(inputs_var.get()),
                "dependencies": svc.split_csv(deps_var.get()),
                "expected_data_type": dtype_var.get() or "float",
                "expected_range": svc.parse_expected_range(range_var.get()),
                "implementation_status": status_var.get() or "planned",
                "priority": priority_var.get() or "medium",
                "owner": owner_var.get().strip(),
                "notes": notes.get("1.0", tk.END).strip(),
            }
            try:
                result = svc.save_planned_feature(self.chart_dir, payload)
                saved = (result.get("feature") or {}).get("name") or name
                dlg.destroy()
                self.refresh()
                self._selected_name = saved
                self._render_list()
                self._show_detail(saved)
                fid = (result.get("feature") or {}).get("feature_id")
                if fid:
                    messagebox.showinfo("Add Feature", f"Saved with stable ID {fid}.")
            except Exception as exc:
                messagebox.showerror("Add Feature", str(exc), parent=dlg)

        btns = ttk.Frame(dlg, padding=8)
        btns.pack(fill="x")
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="right", padx=4)
        ttk.Button(btns, text="Save to Backlog", command=save).pack(side="right")

    def _open_project_manager(self) -> None:
        from .feature_project_manager_panel import open_feature_project_manager_window

        win = self._pm_window
        if win is not None:
            try:
                if win.winfo_exists():
                    win.lift()
                    win.focus_force()
                    panel = getattr(win, "_project_manager_panel", None)
                    if panel is not None:
                        panel.set_chart_dir(self.chart_dir)
                        panel.refresh()
                    return
            except tk.TclError:
                self._pm_window = None

        self._pm_window = open_feature_project_manager_window(
            self,
            chart_dir=self.chart_dir,
            on_changed=self._on_projects_changed,
        )

    def _on_projects_changed(self) -> None:
        prev = self._project_var.get()
        self.refresh(lazy=False)
        values = list(self._project_combo["values"] or [])
        if prev in values:
            self._project_var.set(prev)

    def _open_import_dialog(self) -> None:
        dlg = tk.Toplevel(self)
        dlg.title("Import Feature Registry")
        dlg.geometry("640x520")
        dlg.transient(self.winfo_toplevel())
        dlg.grab_set()

        import_type_var = tk.StringVar(value="preview_only")
        conflict_var = tk.StringVar(value="skip")
        file_var = tk.StringVar()
        new_group_var = tk.StringVar()
        target_group_var = tk.StringVar()
        preview_state: dict[str, Any] = {"payload": None, "result": None}

        body = ttk.Frame(dlg, padding=8)
        body.pack(fill="both", expand=True)

        ttk.Label(body, text="Import type").pack(anchor="w")
        type_row = ttk.Frame(body)
        type_row.pack(anchor="w")

        file_row = ttk.Frame(body)
        file_row.pack(fill="x", pady=8)
        ttk.Entry(file_row, textvariable=file_var).pack(side="left", fill="x", expand=True)
        def pick_file() -> None:
            path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
            if path:
                file_var.set(path)
        ttk.Button(file_row, text="Browse…", command=pick_file).pack(side="left", padx=4)

        ng_row = ttk.Frame(body)
        ng_row.pack(fill="x", pady=2)
        ttk.Label(ng_row, text="New group id", width=16).pack(side="left")
        ttk.Entry(ng_row, textvariable=new_group_var).pack(side="left", fill="x", expand=True)

        tg_row = ttk.Frame(body)
        tg_row.pack(fill="x", pady=2)
        ttk.Label(tg_row, text="Target group", width=16).pack(side="left")
        tg_combo = ttk.Combobox(
            tg_row, textvariable=target_group_var,
            values=[g.get("id") or "" for g in (self._catalog or {}).get("groups") or []],
            state="readonly",
        )
        tg_combo.pack(side="left", fill="x", expand=True)

        cf_row = ttk.Frame(body)
        cf_row.pack(fill="x", pady=4)
        ttk.Label(cf_row, text="Conflict policy").pack(side="left")
        ttk.Radiobutton(cf_row, text="Skip", variable=conflict_var, value="skip").pack(side="left", padx=4)
        ttk.Radiobutton(cf_row, text="Overwrite", variable=conflict_var, value="overwrite").pack(side="left")

        preview = scrolledtext.ScrolledText(body, height=14, font=("Consolas", 9))
        preview.pack(fill="both", expand=True, pady=8)
        preview.insert("1.0", "Select a JSON file and click Preview.")
        preview.configure(state="disabled")

        def toggle_fields() -> None:
            t = import_type_var.get()
            if t == "new_group":
                ng_row.pack(fill="x", pady=2)
            else:
                ng_row.pack_forget()
            if t == "existing_group":
                tg_row.pack(fill="x", pady=2)
            else:
                tg_row.pack_forget()
            if t not in ("preview_only", "merge_registry"):
                cf_row.pack(fill="x", pady=4)
            else:
                cf_row.pack_forget()

        toggle_fields()
        for val, label in (
            ("new_group", "New Feature Group"),
            ("existing_group", "Existing Group"),
            ("merge_registry", "Merge Registry"),
            ("preview_only", "Preview Only"),
        ):
            ttk.Radiobutton(
                type_row, text=label, variable=import_type_var, value=val, command=toggle_fields,
            ).pack(anchor="w")

        def set_preview(text: str) -> None:
            preview.configure(state="normal")
            preview.delete("1.0", tk.END)
            preview.insert("1.0", text)
            preview.configure(state="disabled")

        def do_preview() -> None:
            path = file_var.get().strip()
            if not path:
                messagebox.showerror("Import", "Choose a JSON file first.", parent=dlg)
                return
            try:
                payload = svc.load_json_file(path)
                preview_state["payload"] = payload
                result = svc.preview_import(
                    self.chart_dir,
                    payload=payload,
                    import_type=import_type_var.get(),
                    target_group=target_group_var.get() or None,
                    new_group={"id": new_group_var.get().strip()} if new_group_var.get().strip() else None,
                    conflict_policy=conflict_var.get(),
                )
                preview_state["result"] = result
                set_preview(fmt.format_import_preview(result))
                apply_btn.configure(state="normal" if result.get("can_apply") and import_type_var.get() != "preview_only" else "disabled")
            except Exception as exc:
                set_preview(f"Error:\n{exc}")
                apply_btn.configure(state="disabled")

        def do_apply() -> None:
            payload = preview_state.get("payload")
            if not payload:
                return
            itype = import_type_var.get()
            if itype == "preview_only":
                return
            if not messagebox.askyesno("Import", f"Apply import ({itype})?", parent=dlg):
                return
            try:
                svc.apply_import(
                    self.chart_dir,
                    payload=payload,
                    import_type=itype,
                    target_group=target_group_var.get() or None,
                    new_group={"id": new_group_var.get().strip()} if new_group_var.get().strip() else None,
                    conflict_policy=conflict_var.get(),
                )
                dlg.destroy()
                self.refresh()
                messagebox.showinfo("Import", "Import applied.")
            except Exception as exc:
                messagebox.showerror("Import", str(exc), parent=dlg)

        btns = ttk.Frame(dlg, padding=8)
        btns.pack(fill="x")
        apply_btn = ttk.Button(btns, text="Apply", command=do_apply, state="disabled")
        apply_btn.pack(side="right", padx=4)
        ttk.Button(btns, text="Preview", command=do_preview).pack(side="right", padx=4)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="right")

    def _export_json(self) -> None:
        if not self._catalog:
            self.refresh()
        if not self._catalog:
            return
        filtered = self._filtered_features()
        ctx = {
            "search": self._search_var.get(),
            "status_filter": self._filter_key(self._status_var.get()),
            "group_filter": self._filter_key(self._group_var.get()),
            "project_filter": self._filter_key(self._project_var.get()),
            "selected_feature": self._selected_name,
            "filtered_count": len(filtered),
            "total_count": self._catalog.get("feature_count") or len(self._catalog.get("features") or []),
        }
        payload = svc.export_catalog_json(self._catalog, filtered_features=filtered, context=ctx)
        import datetime as dt

        date = dt.date.today().isoformat()
        h = str(self._catalog.get("schema_registry_hash") or "registry")[:8]
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile=f"feature_registry_{date}_{h}.json",
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        messagebox.showinfo("Export", f"Saved to\n{path}")

    def _open_delete_dialog(self) -> None:
        if not self._selected_name:
            return
        dlg = tk.Toplevel(self)
        dlg.title(f"Delete — {self._selected_name}")
        dlg.geometry("520x400")
        dlg.transient(self.winfo_toplevel())
        dlg.grab_set()
        body = scrolledtext.ScrolledText(dlg, wrap="word", font=("Consolas", 9))
        body.pack(fill="both", expand=True, padx=8, pady=8)
        body.insert("1.0", "Scanning dependencies…")
        body.configure(state="disabled")
        report_holder: dict[str, Any] = {}

        def set_body(text: str) -> None:
            body.configure(state="normal")
            body.delete("1.0", tk.END)
            body.insert("1.0", text)
            body.configure(state="disabled")

        def load_preview() -> None:
            try:
                report = svc.delete_preview(self.chart_dir, self._selected_name or "")
                report_holder["report"] = report
                set_body(fmt.format_delete_preview(report))
                confirm_btn.configure(state="normal" if report.get("can_delete") else "disabled")
            except Exception as exc:
                set_body(f"Error:\n{exc}")

        btns = ttk.Frame(dlg, padding=8)
        btns.pack(fill="x")
        confirm_btn = ttk.Button(btns, text="Delete", state="disabled", command=lambda: None)
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="right", padx=4)

        def confirm() -> None:
            report = report_holder.get("report") or {}
            if not report.get("can_delete"):
                return
            try:
                svc.delete_feature(
                    self.chart_dir,
                    feature_id=report.get("feature_id"),
                    name=report.get("name"),
                )
                dlg.destroy()
                self._selected_name = None
                self.refresh()
                messagebox.showinfo("Delete", "Feature deleted.")
            except Exception as exc:
                messagebox.showerror("Delete", str(exc), parent=dlg)

        confirm_btn.configure(command=confirm)
        confirm_btn.pack(side="right")
        dlg.after(50, load_preview)

    def _show_parity_rules(self) -> None:
        try:
            data = svc.parity_rules()
        except Exception as exc:
            messagebox.showerror("Parity rules", str(exc))
            return
        self._show_json_dialog("Feature Parity Rules", data)

    def _run_parity_audit(self) -> None:
        if self._parity_busy:
            messagebox.showwarning("Audit", "Parity audit is already running.")
            return
        self._parity_busy = True
        dlg = tk.Toplevel(self)
        dlg.title("Feature Pipeline Parity")
        dlg.geometry("720x520")
        dlg.transient(self.winfo_toplevel())
        body = scrolledtext.ScrolledText(dlg, wrap="word", font=("Consolas", 9))
        body.pack(fill="both", expand=True, padx=8, pady=8)
        body.insert("1.0", "Running pipeline parity audit…")
        body.configure(state="disabled")

        def worker() -> None:
            try:
                data = svc.run_pipeline_parity_audit(self.chart_dir, include_parquet=True, trading_day="2026-07-03")
                self._job_queue.put({"kind": "parity_done", "dlg": dlg, "body": body, "data": data})
            except Exception as exc:
                self._job_queue.put({"kind": "parity_error", "dlg": dlg, "body": body, "message": str(exc)})

        threading.Thread(target=worker, daemon=True, name="fr-parity-audit").start()

        btn_row = ttk.Frame(dlg, padding=8)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Close", command=dlg.destroy).pack(side="right")

    def _show_json_dialog(self, title: str, data: Any) -> None:
        dlg = tk.Toplevel(self)
        dlg.title(title)
        dlg.geometry("720x560")
        dlg.transient(self.winfo_toplevel())
        txt = scrolledtext.ScrolledText(dlg, wrap="none", font=("Consolas", 9))
        txt.pack(fill="both", expand=True, padx=8, pady=8)
        txt.insert("1.0", fmt.format_json(data))
        txt.configure(state="disabled")
        row = ttk.Frame(dlg, padding=8)
        row.pack(fill="x")

        def save() -> None:
            path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
            if path:
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2, default=str)

        ttk.Button(row, text="Save JSON", command=save).pack(side="left")
        ttk.Button(row, text="Close", command=dlg.destroy).pack(side="right")

    def _handle_job(self, msg: dict[str, Any]) -> None:
        kind = msg.get("kind")
        if kind == "parity_done":
            self._parity_busy = False
            body = msg.get("body")
            data = msg.get("data") or {}
            if body and body.winfo_exists():
                body.configure(state="normal")
                body.delete("1.0", tk.END)
                body.insert("1.0", fmt.format_parity_audit(data))
                body.configure(state="disabled")
            dlg = msg.get("dlg")
            if dlg and dlg.winfo_exists():
                btn_row = ttk.Frame(dlg, padding=8)
                btn_row.pack(fill="x")

                def save() -> None:
                    path = filedialog.asksaveasfilename(
                        defaultextension=".json",
                        filetypes=[("JSON", "*.json")],
                        initialfile="feature_pipeline_parity.json",
                    )
                    if path:
                        with open(path, "w", encoding="utf-8") as fh:
                            json.dump(data, fh, indent=2, default=str)

                ttk.Button(btn_row, text="Save JSON", command=save).pack(side="left")
            return
        if kind == "parity_error":
            self._parity_busy = False
            body = msg.get("body")
            if body and body.winfo_exists():
                body.configure(state="normal")
                body.delete("1.0", tk.END)
                body.insert("1.0", f"Error:\n{msg.get('message')}")
                body.configure(state="disabled")
