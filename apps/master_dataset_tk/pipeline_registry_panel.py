"""Pipeline Feature Registry tab — list pipelines, membership, candidates."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Callable

from .pipeline_registry_service import (
    create_pipeline,
    delete_pipeline,
    get_pipeline,
    load_pipelines,
    registry_catalog_features,
    set_pipeline_registry_members,
)


def _prepare_modal_dialog(
    win: tk.Toplevel,
    anchor: tk.Misc,
    *,
    width: int | None = None,
    height: int | None = None,
    min_width: int = 320,
    min_height: int = 180,
    padding: int = 16,
) -> None:
    """Center a modal dialog over the Feature Transformations panel."""
    from .fold_replay_widgets import center_toplevel_on_widget

    win.transient(anchor.winfo_toplevel())
    win.update_idletasks()
    ww = max(int(width or 0), int(win.winfo_reqwidth()) + padding, min_width)
    wh = max(int(height or 0), int(win.winfo_reqheight()) + padding, min_height)
    win.minsize(min_width, min_height)
    win.geometry(f"{ww}x{wh}")
    center_toplevel_on_widget(win, anchor)
    win.lift()
    try:
        win.focus_force()
    except tk.TclError:
        pass
    win.grab_set()


class PipelineRegistryFeatureDialog(tk.Toplevel):
    """Select master registry features (by feature_id) for a pipeline."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        selected_ids: list[str],
        on_apply: Callable[[list[str]], None],
    ) -> None:
        super().__init__(master)
        self.title("Select Registry Features")
        _prepare_modal_dialog(self, master, width=640, height=560, min_width=640, min_height=560)
        self._features = registry_catalog_features(chart_dir)
        self._on_apply = on_apply
        self._filter_var = tk.StringVar(value="")
        self._count_var = tk.StringVar(value="")
        self._vars: dict[str, tk.BooleanVar] = {}
        selected = {str(x).strip().upper() for x in selected_ids}

        top = ttk.Frame(self, padding=8)
        top.pack(fill="both", expand=True)
        toolbar = ttk.Frame(top)
        toolbar.pack(fill="x", pady=(0, 6))
        ttk.Label(toolbar, text="Search").pack(side="left")
        ent = ttk.Entry(toolbar, textvariable=self._filter_var)
        ent.pack(side="left", fill="x", expand=True, padx=(6, 8))
        ent.bind("<KeyRelease>", lambda _e: self._apply_filter())
        ttk.Label(toolbar, textvariable=self._count_var, font=("Segoe UI", 9, "bold")).pack(side="right")

        list_frame = ttk.Frame(top)
        list_frame.pack(fill="both", expand=True)
        sb = ttk.Scrollbar(list_frame, orient="vertical")
        self._list = tk.Listbox(list_frame, selectmode=tk.MULTIPLE, height=18)
        self._list.configure(yscrollcommand=sb.set)
        sb.config(command=self._list.yview)
        self._list.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._rows: list[dict[str, Any]] = []

        for feat in self._features:
            fid = feat["feature_id"]
            var = tk.BooleanVar(value=fid in selected)
            self._vars[fid] = var

        self._rebuild_list()
        self._list.bind("<<ListboxSelect>>", self._on_list_select)

        actions = ttk.Frame(self, padding=8)
        actions.pack(fill="x")
        ttk.Button(actions, text="Apply", command=self._apply).pack(side="right")
        ttk.Button(actions, text="Cancel", command=self.destroy).pack(side="right", padx=(0, 8))

    def _rebuild_list(self) -> None:
        q = str(self._filter_var.get() or "").strip().lower()
        self._list.delete(0, tk.END)
        self._rows = []
        for feat in self._features:
            blob = f"{feat.get('name')} {feat.get('label')} {feat.get('group')}".lower()
            if q and q not in blob:
                continue
            self._rows.append(feat)
            mark = "☑" if self._vars[feat["feature_id"]].get() else "☐"
            self._list.insert(tk.END, f"{mark}  {feat['name']}")
        self._count_var.set(f"{sum(v.get() for v in self._vars.values())} selected")

    def _apply_filter(self) -> None:
        self._rebuild_list()

    def _on_list_select(self, _event: Any = None) -> None:
        for i in self._list.selection():
            feat = self._rows[i]
            fid = feat["feature_id"]
            self._vars[fid].set(not self._vars[fid].get())
        self._rebuild_list()

    def _apply(self) -> None:
        selected = [fid for fid, var in self._vars.items() if var.get()]
        self._on_apply(selected)
        self.destroy()


class CreatePipelineDialog(tk.Toplevel):
    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_create: Callable[[str, str], None],
    ) -> None:
        super().__init__(master)
        self.title("Create Pipeline")
        self.resizable(False, False)
        self._chart_dir = chart_dir
        self._on_create = on_create

        from chain_replay_ml.dataset_builder.pipeline_registry_store import format_display_name

        outer = ttk.Frame(self, padding=12)
        outer.pack(fill="both", expand=True)

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)

        ttk.Label(body, text="Pipeline Name").grid(row=0, column=0, sticky="w")
        self._name_var = tk.StringVar(value=format_display_name(999))  # placeholder
        name_row = ttk.Frame(body)
        name_row.grid(row=0, column=1, sticky="ew", pady=(0, 10))
        ttk.Entry(name_row, textvariable=self._name_var, width=24).pack(side="left", fill="x", expand=True)
        ttk.Button(name_row, text="Auto Generate Name", command=self._auto_name).pack(
            side="left", padx=(8, 0)
        )

        ttk.Label(body, text="Pipeline Type").grid(row=1, column=0, sticky="nw", pady=(4, 0))
        self._type_var = tk.StringVar(value="auto")
        types = ttk.Frame(body)
        types.grid(row=1, column=1, sticky="w", pady=(4, 0))
        ttk.Radiobutton(types, text="Manual", value="manual", variable=self._type_var).pack(anchor="w")
        ttk.Radiobutton(types, text="Auto", value="auto", variable=self._type_var).pack(anchor="w")

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(16, 0))
        ttk.Button(footer, text="Create", command=self._create).pack(side="right")
        ttk.Button(footer, text="Cancel", command=self.destroy).pack(side="right", padx=(0, 8))

        self._auto_name()
        _prepare_modal_dialog(self, master, min_width=400, min_height=200)

    def _auto_name(self) -> None:
        from chain_replay_ml.dataset_builder.pipeline_registry_store import (
            format_display_name,
            load_store,
        )
        from .build_service import chart_data_dir

        data_dir = chart_data_dir(self._chart_dir)
        doc = load_store(data_dir)
        seq = int(doc.get("next_display_seq") or 1)
        self._name_var.set(format_display_name(seq))

    def _create(self) -> None:
        name = str(self._name_var.get() or "").strip()
        ptype = str(self._type_var.get() or "manual")
        if not name:
            messagebox.showwarning("Create Pipeline", "Pipeline name is required.", parent=self)
            return
        self._on_create(name, ptype)
        self.destroy()


class PipelineRegistryPanel(ttk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_pipelines_changed: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master, padding=8)
        self.chart_dir = chart_dir
        self._on_pipelines_changed = on_pipelines_changed
        self._pipelines: list[dict[str, Any]] = []
        self._selected_id: str | None = None
        self._detail_var = tk.StringVar(value="Select a pipeline")
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        header = ttk.Frame(self)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text="Pipeline Feature Registry", font=("Segoe UI", 12, "bold")).pack(
            side="left"
        )
        ttk.Button(header, text="Create Pipeline", command=self._open_create).pack(side="right")
        ttk.Button(header, text="Refresh", command=self.refresh).pack(side="right", padx=(0, 8))

        cols = ("name", "type", "features", "status")
        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill="x")
        self._tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=6)
        for c, label, w in (
            ("name", "Pipeline Name", 160),
            ("type", "Type", 90),
            ("features", "Features", 70),
            ("status", "Status", 80),
        ):
            self._tree.heading(c, text=label)
            self._tree.column(c, width=w, anchor="center" if c != "name" else "w")
        sb = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.pack(side="left", fill="x", expand=True)
        sb.pack(side="right", fill="y")
        self._tree.bind("<<TreeviewSelect>>", self._on_select)

        detail = ttk.LabelFrame(self, text="Pipeline Detail", padding=8)
        detail.pack(fill="both", expand=True, pady=(8, 0))
        ttk.Label(detail, textvariable=self._detail_var, justify="left").pack(anchor="w")

        btn_row = ttk.Frame(detail)
        btn_row.pack(fill="x", pady=(8, 0))
        ttk.Button(btn_row, text="Select Features", command=self._select_features).pack(side="left")
        ttk.Label(btn_row, text="(from Master Feature Registry — membership only)", foreground="#666").pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(btn_row, text="Delete Pipeline", command=self._delete_pipeline).pack(side="right")

        cand_frame = ttk.LabelFrame(detail, text="Candidate Features", padding=4)
        cand_frame.pack(fill="both", expand=True, pady=(8, 0))
        self._cand_list = tk.Listbox(cand_frame, height=12)
        cand_sb = ttk.Scrollbar(cand_frame, orient="vertical", command=self._cand_list.yview)
        self._cand_list.configure(yscrollcommand=cand_sb.set)
        self._cand_list.pack(side="left", fill="both", expand=True)
        cand_sb.pack(side="right", fill="y")

    def refresh(self) -> None:
        self._pipelines = load_pipelines(self.chart_dir)
        self._tree.delete(*self._tree.get_children())
        for row in self._pipelines:
            pid = row["pipeline_id"]
            self._tree.insert(
                "",
                tk.END,
                iid=pid,
                values=(
                    row.get("name"),
                    row.get("type_label"),
                    row.get("feature_count"),
                    row.get("status_label"),
                ),
            )
        if self._selected_id and self._tree.exists(self._selected_id):
            self._tree.selection_set(self._selected_id)
            self._show_detail(self._selected_id)
        elif self._pipelines:
            first = self._pipelines[0]["pipeline_id"]
            self._tree.selection_set(first)
            self._show_detail(first)

    def _on_select(self, _event: Any = None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        self._show_detail(sel[0])

    def _show_detail(self, pipeline_id: str) -> None:
        self._selected_id = pipeline_id
        row = get_pipeline(self.chart_dir, pipeline_id)
        if not row:
            self._detail_var.set("Pipeline not found")
            self._cand_list.delete(0, tk.END)
            return
        self._detail_var.set(
            f"ID: {row['pipeline_id']}  ·  Name: {row['name']}  ·  "
            f"Type: {row['type_label']}  ·  Status: {row['status_label']}\n"
            f"Registry members: {row['registry_feature_count']}  ·  "
            f"Candidates: {row['candidate_count']}"
        )
        self._cand_list.delete(0, tk.END)
        for name in row.get("candidate_features") or []:
            self._cand_list.insert(tk.END, name)

    def _open_create(self) -> None:
        CreatePipelineDialog(self, chart_dir=self.chart_dir, on_create=self._create_pipeline)

    def _notify_pipelines_changed(self, pipeline_id: str | None = None) -> None:
        if callable(self._on_pipelines_changed):
            try:
                self._on_pipelines_changed(select_pipeline_id=pipeline_id)
            except TypeError:
                try:
                    self._on_pipelines_changed()
                except Exception:
                    pass
            except Exception:
                pass

    def _create_pipeline(self, name: str, pipeline_type: str) -> None:
        try:
            row = create_pipeline(self.chart_dir, name=name, pipeline_type=pipeline_type)
            self.refresh()
            pid = row.get("pipeline_id")
            if pid:
                self._tree.selection_set(pid)
                self._show_detail(pid)
                self._notify_pipelines_changed(str(pid))
            else:
                self._notify_pipelines_changed()
        except Exception as exc:
            messagebox.showerror("Create Pipeline", str(exc), parent=self)

    def _delete_pipeline(self) -> None:
        pid = self._selected_id
        if not pid:
            messagebox.showinfo("Delete Pipeline", "Select a pipeline first.", parent=self)
            return
        row = get_pipeline(self.chart_dir, pid)
        if not row:
            messagebox.showerror("Delete Pipeline", "Pipeline not found.", parent=self)
            return
        if str(row.get("type") or "") == "existing":
            messagebox.showinfo(
                "Delete Pipeline",
                "The existing default pipeline cannot be deleted.",
                parent=self,
            )
            return
        name = str(row.get("name") or pid)
        if not messagebox.askyesno(
            "Delete Pipeline",
            f"Delete pipeline \"{name}\" ({pid})?\n\n"
            "Registry membership and candidate features for this pipeline will be removed.\n"
            "The Master Feature Registry will not be changed.\n\n"
            "This cannot be undone.",
            parent=self,
            icon="warning",
        ):
            return
        try:
            delete_pipeline(self.chart_dir, pid)
        except Exception as exc:
            messagebox.showerror("Delete Pipeline", str(exc), parent=self)
            return
        self._selected_id = None
        self.refresh()
        self._notify_pipelines_changed()

    def _select_features(self) -> None:
        pid = self._selected_id
        if not pid:
            messagebox.showinfo("Select Features", "Select a pipeline first.", parent=self)
            return
        row = get_pipeline(self.chart_dir, pid)
        if not row:
            return
        if str(row.get("type") or "") == "existing":
            messagebox.showinfo(
                "Select Features",
                "The existing default pipeline uses legacy pipeline features, not registry membership.",
                parent=self,
            )
            return

        def _apply(feature_ids: list[str]) -> None:
            try:
                set_pipeline_registry_members(self.chart_dir, pid, feature_ids)
                self.refresh()
                self._show_detail(pid)
            except Exception as exc:
                messagebox.showerror("Select Features", str(exc), parent=self)

        PipelineRegistryFeatureDialog(
            self,
            chart_dir=self.chart_dir,
            selected_ids=list(row.get("registry_feature_ids") or []),
            on_apply=_apply,
        )