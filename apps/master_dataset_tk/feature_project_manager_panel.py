"""Feature Project Manager — list + editor workspace (refs only, no feature ownership)."""

from __future__ import annotations

import json
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk
from typing import Any, Callable

from chain_replay_ml.dataset_builder.orchestrator import _load_feature_registry

from .fold_replay_widgets import place_toplevel_beside_main
from . import feature_registry_service as svc
from .feature_selection_engine import (
    all_group_ids,
    group_meta,
    sync_enabled_groups_from_features,
)


def _fmt_ts(value: Any) -> str:
    if not value:
        return "—"
    text = str(value)
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        dt = datetime.fromisoformat(text)
        return dt.astimezone().strftime("%Y-%m-%d %H:%M")
    except Exception:
        return text[:16]


def _warmup_label(minutes: Any) -> str:
    if minutes is None or str(minutes).strip() == "":
        return "—"
    try:
        n = int(minutes)
    except (TypeError, ValueError):
        return str(minutes)
    if n <= 0:
        return "—"
    if n == 1:
        return "1 minute"
    if n % 60 == 0 and n >= 60:
        hours = n // 60
        return f"{hours} hour" if hours == 1 else f"{hours} hours"
    return f"{n} minutes"


class FeatureProjectManagerPanel(ttk.Frame):
    """Split Project Manager: project table (left) + editor (right)."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_changed: Callable[[], None] | None = None,
        on_close: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._on_changed = on_changed
        self._on_close = on_close
        self._catalog: dict[str, Any] | None = None
        self._registry: dict[str, Any] = {}
        self._projects: list[dict[str, Any]] = []
        self._selected_id: str | None = None
        self._selected_features: set[str] = set()
        self._dirty = False
        self._loading = False
        self._group_filter: str | None = None
        self._build_ui()

    def set_chart_dir(self, chart_dir: str) -> None:
        self.chart_dir = chart_dir

    def refresh(self) -> None:
        try:
            self._catalog = svc.load_catalog(self.chart_dir)
            self._registry = _load_feature_registry() or {}
            self._projects = list((self._catalog or {}).get("projects") or [])
        except Exception as exc:
            messagebox.showerror("Project Manager", f"Failed to load projects:\n{exc}", parent=self)
            return
        self._render_project_list()
        if self._selected_id and any(p.get("id") == self._selected_id for p in self._projects):
            self._load_project(self._selected_id, force=True)
        elif self._projects:
            self._load_project(str(self._projects[0].get("id") or ""), force=True)
        else:
            self._clear_editor()

    def _notify_changed(self) -> None:
        if self._on_changed:
            self._on_changed()

    def _build_ui(self) -> None:
        hdr = ttk.Frame(self, padding=(8, 6, 8, 0))
        hdr.pack(fill="x")
        ttk.Label(hdr, text="Project Manager", font=("Segoe UI", 12, "bold")).pack(side="left")
        ttk.Label(
            hdr,
            text="Feature projects store references only — Feature Registry is source of truth.",
            foreground="#888",
        ).pack(side="left", padx=(10, 0))
        if self._on_close:
            ttk.Button(hdr, text="Close", command=self._on_close).pack(side="right")

        body = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        body.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.Frame(body)
        body.add(left, weight=2)
        right = ttk.Frame(body)
        body.add(right, weight=3)

        self._build_list_side(left)
        self._build_editor_side(right)

    def _build_list_side(self, parent: ttk.Frame) -> None:
        tools = ttk.Frame(parent)
        tools.pack(fill="x", pady=(0, 6))
        for text, cmd in (
            ("New Project", self._wizard_new),
            ("Clone Project", self._clone_selected),
            ("Delete Project", self._delete_selected),
            ("Rename Project", self._rename_selected),
            ("Export Project", self._export_selected),
            ("Import Project", self._import_project),
            ("Refresh", self.refresh),
        ):
            ttk.Button(tools, text=text, command=cmd).pack(side="left", padx=(0, 4))

        cols = ("name", "id", "features", "warmup", "modified", "models", "status")
        self._tree = ttk.Treeview(parent, columns=cols, show="headings", selectmode="browse", height=18)
        headings = {
            "name": ("Project Name", 140),
            "id": ("Project ID", 110),
            "features": ("Features", 70),
            "warmup": ("Warmup", 90),
            "modified": ("Last Modified", 120),
            "models": ("Models", 60),
            "status": ("Status", 70),
        }
        for c, (title, width) in headings.items():
            self._tree.heading(c, text=title)
            self._tree.column(c, width=width, anchor="w")
        sb = ttk.Scrollbar(parent, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=sb.set)
        self._tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)

    def _build_editor_side(self, parent: ttk.Frame) -> None:
        self._editor = ttk.Frame(parent)
        self._editor.pack(fill="both", expand=True)

        meta = ttk.LabelFrame(self._editor, text="Project", padding=8)
        meta.pack(fill="x", pady=(0, 8))

        grid = ttk.Frame(meta)
        grid.pack(fill="x")
        self._name_var = tk.StringVar()
        self._id_var = tk.StringVar()
        self._warmup_var = tk.StringVar()
        self._sampling_var = tk.StringVar()
        rows = (
            ("Project Name", self._name_var, False),
            ("Project ID", self._id_var, True),
            ("Warmup Period (minutes)", self._warmup_var, False),
            ("Default Sampling (future)", self._sampling_var, False),
        )
        for i, (label, var, readonly) in enumerate(rows):
            ttk.Label(grid, text=label).grid(row=i, column=0, sticky="w", pady=2, padx=(0, 8))
            ent = ttk.Entry(grid, textvariable=var, width=36, state="readonly" if readonly else "normal")
            ent.grid(row=i, column=1, sticky="ew", pady=2)
            if not readonly:
                var.trace_add("write", lambda *_: self._mark_dirty())
        grid.columnconfigure(1, weight=1)

        ttk.Label(meta, text="Description").pack(anchor="w", pady=(8, 2))
        self._desc = scrolledtext.ScrolledText(meta, height=2, font=("Segoe UI", 9), wrap="word")
        self._desc.pack(fill="x")
        self._desc.bind("<<Modified>>", self._on_desc_modified)

        sel = ttk.LabelFrame(self._editor, text="Feature Selection", padding=8)
        sel.pack(fill="both", expand=True, pady=(0, 8))

        top = ttk.Frame(sel)
        top.pack(fill="x", pady=(0, 6))
        ttk.Label(top, text="Feature Groups").pack(side="left", anchor="n")
        self._group_list = tk.Listbox(top, height=10, exportselection=False, width=28)
        self._group_list.pack(side="left", padx=(8, 12))
        self._group_list.bind("<<ListboxSelect>>", lambda _e: self._refresh_available())

        search_col = ttk.Frame(top)
        search_col.pack(side="left", fill="y", anchor="n")
        ttk.Label(search_col, text="Search").pack(anchor="w")
        self._search_var = tk.StringVar()
        self._search_var.trace_add("write", lambda *_: self._refresh_available())
        ttk.Entry(search_col, textvariable=self._search_var, width=24).pack(anchor="w", pady=(2, 6))
        ttk.Button(
            search_col, text="Select Entire Group", command=self._select_entire_group,
        ).pack(anchor="w", pady=(0, 4))
        ttk.Button(
            search_col, text="Unselect Entire Group", command=self._unselect_entire_group,
        ).pack(anchor="w")

        lists = ttk.Frame(sel)
        lists.pack(fill="both", expand=True)
        left_box = ttk.Frame(lists)
        left_box.pack(side="left", fill="both", expand=True)
        ttk.Label(left_box, text="Available Features").pack(anchor="w")
        self._avail = tk.Listbox(left_box, selectmode="extended", exportselection=False)
        self._avail.pack(fill="both", expand=True)
        mid = ttk.Frame(lists)
        mid.pack(side="left", padx=8, fill="y")
        ttk.Button(mid, text="Add →", command=self._add_features).pack(pady=4)
        ttk.Button(mid, text="← Remove", command=self._remove_features).pack(pady=4)
        right_box = ttk.Frame(lists)
        right_box.pack(side="left", fill="both", expand=True)
        ttk.Label(right_box, text="Selected Features").pack(anchor="w")
        self._selected_list = tk.Listbox(right_box, selectmode="extended", exportselection=False)
        self._selected_list.pack(fill="both", expand=True)

        bottom = ttk.Frame(self._editor)
        bottom.pack(fill="x")
        summary = ttk.LabelFrame(bottom, text="Project Summary", padding=8)
        summary.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self._summary_var = tk.StringVar(value="Select a project.")
        ttk.Label(summary, textvariable=self._summary_var, justify="left").pack(anchor="w")

        future = ttk.LabelFrame(bottom, text="Future settings", padding=8)
        future.pack(side="left", fill="both", expand=True, padx=(0, 8))
        ttk.Label(
            future,
            text=(
                "Reserved for:\n"
                "• Default Build Configuration\n"
                "• Default Label Configuration\n"
                "• Feature Validation Rules\n"
                "• Experiment History / Model Associations"
            ),
            foreground="#888",
            justify="left",
        ).pack(anchor="w")

        actions = ttk.Frame(bottom)
        actions.pack(side="right", fill="y")
        ttk.Button(actions, text="Save Project", command=self._save_current).pack(pady=2)
        self._dirty_var = tk.StringVar(value="")
        ttk.Label(actions, textvariable=self._dirty_var, foreground="#B45309").pack()

    def _mark_dirty(self) -> None:
        if self._loading:
            return
        self._dirty = True
        self._dirty_var.set("Unsaved changes")

    def _on_desc_modified(self, _event: tk.Event | None = None) -> None:
        if self._desc.edit_modified():
            self._desc.edit_modified(False)
            self._mark_dirty()

    def _render_project_list(self) -> None:
        selected = self._selected_id
        self._tree.delete(*self._tree.get_children())
        for p in self._projects:
            pid = str(p.get("id") or "")
            feats = p.get("feature_names") or []
            status = "ready" if feats else "empty"
            self._tree.insert(
                "",
                "end",
                iid=pid,
                values=(
                    p.get("label") or pid,
                    pid,
                    f"{len(feats):,}",
                    _warmup_label(p.get("warmup_minutes")),
                    _fmt_ts(p.get("updated_at") or p.get("created_at")),
                    "—",
                    status,
                ),
            )
        if selected and self._tree.exists(selected):
            self._tree.selection_set(selected)
            self._tree.see(selected)

    def _on_tree_select(self, _event: tk.Event | None = None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        pid = sel[0]
        if pid == self._selected_id and not self._dirty:
            return
        if self._dirty and not self._confirm_discard():
            if self._selected_id and self._tree.exists(self._selected_id):
                self._tree.selection_set(self._selected_id)
            return
        self._load_project(pid, force=True)

    def _confirm_discard(self) -> bool:
        return bool(
            messagebox.askyesno(
                "Unsaved changes",
                "Discard unsaved project edits?",
                parent=self,
            )
        )

    def _project_by_id(self, pid: str) -> dict[str, Any] | None:
        for p in self._projects:
            if p.get("id") == pid:
                return p
        return None

    def _clear_editor(self) -> None:
        self._loading = True
        self._selected_id = None
        self._selected_features.clear()
        self._name_var.set("")
        self._id_var.set("")
        self._warmup_var.set("")
        self._sampling_var.set("")
        self._desc.delete("1.0", tk.END)
        self._group_list.delete(0, tk.END)
        self._avail.delete(0, tk.END)
        self._selected_list.delete(0, tk.END)
        self._summary_var.set("No projects yet. Click New Project.")
        self._dirty = False
        self._dirty_var.set("")
        self._loading = False

    def _load_project(self, pid: str, *, force: bool = False) -> None:
        proj = self._project_by_id(pid)
        if not proj:
            self._clear_editor()
            return
        self._loading = True
        self._selected_id = pid
        self._selected_features = {
            str(n) for n in (proj.get("feature_names") or proj.get("enabled_features") or [])
        }
        self._name_var.set(str(proj.get("label") or ""))
        self._id_var.set(pid)
        warm = proj.get("warmup_minutes")
        self._warmup_var.set("" if warm is None else str(warm))
        self._sampling_var.set(str(proj.get("default_sampling") or ""))
        self._desc.delete("1.0", tk.END)
        self._desc.insert("1.0", str(proj.get("description") or ""))
        self._desc.edit_modified(False)
        self._populate_groups()
        self._refresh_available()
        self._refresh_selected_list()
        self._update_summary(proj)
        if self._tree.exists(pid):
            self._tree.selection_set(pid)
            self._tree.see(pid)
        self._dirty = False
        self._dirty_var.set("")
        self._loading = False

    def _populate_groups(self) -> None:
        self._group_list.delete(0, tk.END)
        self._group_list.insert(tk.END, "(all groups)")
        self._group_ids_order: list[str | None] = [None]
        for gid in all_group_ids(self._registry):
            meta = group_meta(self._registry, gid) or {}
            label = str(meta.get("label") or gid)
            self._group_list.insert(tk.END, f"{label} ({gid})")
            self._group_ids_order.append(gid)
        self._group_list.selection_set(0)
        self._group_filter = None

    def _current_group_id(self) -> str | None:
        sel = self._group_list.curselection()
        if not sel:
            return self._group_filter
        idx = int(sel[0])
        if 0 <= idx < len(getattr(self, "_group_ids_order", [])):
            self._group_filter = self._group_ids_order[idx]
        return self._group_filter

    def _features_in_group(self, gid: str | None) -> list[str]:
        if gid is None:
            names: list[str] = []
            for g in all_group_ids(self._registry):
                meta = group_meta(self._registry, g) or {}
                names.extend(str(f) for f in (meta.get("features") or []))
            # unique preserve order
            seen: set[str] = set()
            out: list[str] = []
            for n in names:
                if n not in seen:
                    seen.add(n)
                    out.append(n)
            return out
        meta = group_meta(self._registry, gid) or {}
        return [str(f) for f in (meta.get("features") or [])]

    def _refresh_available(self) -> None:
        q = self._search_var.get().strip().lower()
        gid = self._current_group_id()
        self._avail.delete(0, tk.END)
        for name in self._features_in_group(gid):
            if name in self._selected_features:
                continue
            if q and q not in name.lower():
                continue
            self._avail.insert(tk.END, name)

    def _refresh_selected_list(self) -> None:
        self._selected_list.delete(0, tk.END)
        for name in sorted(self._selected_features):
            self._selected_list.insert(tk.END, name)

    def _update_summary(self, proj: dict[str, Any] | None = None) -> None:
        groups = sync_enabled_groups_from_features(self._registry, self._selected_features)
        disabled = 0
        try:
            disabled_set = svc.disabled_registry_features(self.chart_dir)
            disabled = sum(1 for n in self._selected_features if n in disabled_set)
        except Exception:
            disabled = 0
        warm = self._warmup_var.get().strip()
        self._summary_var.set(
            f"Feature Groups: {len(groups)}\n"
            f"Selected Features: {len(self._selected_features)}\n"
            f"Warmup: {_warmup_label(warm if warm else None)}\n"
            f"Disabled Features: {disabled}\n"
            f"Estimated Build Time: — (future)\n"
            f"Models Using Project: — (future)\n"
            f"Version: {(proj or {}).get('version') or '1'}"
        )

    def _add_features(self) -> None:
        picked = [self._avail.get(i) for i in self._avail.curselection()]
        if not picked:
            return
        self._selected_features.update(picked)
        self._refresh_available()
        self._refresh_selected_list()
        self._update_summary()
        self._mark_dirty()

    def _remove_features(self) -> None:
        picked = [self._selected_list.get(i) for i in self._selected_list.curselection()]
        if not picked:
            return
        for name in picked:
            self._selected_features.discard(name)
        self._refresh_available()
        self._refresh_selected_list()
        self._update_summary()
        self._mark_dirty()

    def _select_entire_group(self) -> None:
        gid = self._current_group_id()
        if gid is None:
            messagebox.showinfo("Feature Selection", "Select a feature group first.", parent=self)
            return
        self._selected_features.update(self._features_in_group(gid))
        self._refresh_available()
        self._refresh_selected_list()
        self._update_summary()
        self._mark_dirty()

    def _unselect_entire_group(self) -> None:
        gid = self._current_group_id()
        if gid is None:
            messagebox.showinfo("Feature Selection", "Select a feature group first.", parent=self)
            return
        for name in self._features_in_group(gid):
            self._selected_features.discard(name)
        self._refresh_available()
        self._refresh_selected_list()
        self._update_summary()
        self._mark_dirty()

    def _parse_warmup(self) -> int | None:
        raw = self._warmup_var.get().strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError as exc:
            raise ValueError("Warmup must be an integer number of minutes.") from exc

    def _save_current(self) -> None:
        if not self._selected_id:
            messagebox.showinfo("Project Manager", "Select or create a project first.", parent=self)
            return
        try:
            warmup = self._parse_warmup()
            groups = sorted(sync_enabled_groups_from_features(self._registry, self._selected_features))
            svc.update_project(
                self.chart_dir,
                self._selected_id,
                label=self._name_var.get().strip(),
                description=self._desc.get("1.0", tk.END).strip(),
                group_ids=groups,
                feature_names=sorted(self._selected_features),
                warmup_minutes=warmup,
                default_sampling=self._sampling_var.get().strip(),
            )
        except Exception as exc:
            messagebox.showerror("Project Manager", str(exc), parent=self)
            return
        self._dirty = False
        self._dirty_var.set("")
        self._notify_changed()
        self.refresh()
        messagebox.showinfo("Project Manager", "Project saved.", parent=self)

    def _wizard_new(self) -> None:
        if self._dirty and not self._confirm_discard():
            return
        dlg = tk.Toplevel(self)
        dlg.title("New Feature Project")
        dlg.geometry("480x420")
        dlg.transient(self.winfo_toplevel())
        dlg.grab_set()

        body = ttk.Frame(dlg, padding=12)
        body.pack(fill="both", expand=True)
        name_var = tk.StringVar()
        id_var = tk.StringVar()
        mode_var = tk.StringVar(value="empty")
        clone_var = tk.StringVar(value="")

        ttk.Label(body, text="Project Name").pack(anchor="w")
        ttk.Entry(body, textvariable=name_var).pack(fill="x", pady=(0, 8))
        ttk.Label(body, text="Project ID (snake_case)").pack(anchor="w")
        ttk.Entry(body, textvariable=id_var).pack(fill="x", pady=(0, 8))
        ttk.Label(body, text="Description").pack(anchor="w")
        desc = scrolledtext.ScrolledText(body, height=3, font=("Segoe UI", 9))
        desc.pack(fill="x", pady=(0, 8))

        ttk.Label(body, text="Create From").pack(anchor="w")
        ttk.Radiobutton(body, text="Empty Project", variable=mode_var, value="empty").pack(anchor="w")
        ttk.Radiobutton(body, text="Clone Existing Project", variable=mode_var, value="clone").pack(anchor="w")
        clone_vals = [f"{p.get('id')}|{p.get('label')}" for p in self._projects]
        clone_combo = ttk.Combobox(body, textvariable=clone_var, values=clone_vals, state="readonly")
        clone_combo.pack(fill="x", pady=(4, 12))

        def create() -> None:
            label = name_var.get().strip()
            if not label:
                messagebox.showerror("New Project", "Project Name is required.", parent=dlg)
                return
            pid = id_var.get().strip() or None
            description = desc.get("1.0", tk.END).strip()
            try:
                if mode_var.get() == "clone":
                    src = (clone_var.get() or "").split("|", 1)[0].strip()
                    if not src:
                        messagebox.showerror("New Project", "Select a project to clone.", parent=dlg)
                        return
                    result = svc.clone_project(
                        self.chart_dir, src, label=label, project_id=pid,
                    )
                    # apply description override
                    new_id = (result.get("project") or {}).get("id")
                    if new_id and description:
                        svc.update_project(self.chart_dir, new_id, description=description)
                else:
                    result = svc.create_project(
                        self.chart_dir,
                        label=label,
                        project_id=pid,
                        description=description,
                        group_ids=[],
                        feature_names=[],
                    )
                    new_id = (result.get("project") or {}).get("id")
            except Exception as exc:
                messagebox.showerror("New Project", str(exc), parent=dlg)
                return
            dlg.destroy()
            self._notify_changed()
            self.refresh()
            if new_id:
                self._load_project(str(new_id), force=True)

        btns = ttk.Frame(body)
        btns.pack(fill="x")
        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="right", padx=4)
        ttk.Button(btns, text="Create", command=create).pack(side="right")

    def _clone_selected(self) -> None:
        if not self._selected_id:
            messagebox.showinfo("Clone Project", "Select a project first.", parent=self)
            return
        if self._dirty and not self._confirm_discard():
            return
        src = self._project_by_id(self._selected_id) or {}
        default_name = f"{src.get('label') or self._selected_id} Copy"
        name = simpledialog.askstring("Clone Project", "New project name:", initialvalue=default_name, parent=self)
        if not name:
            return
        try:
            result = svc.clone_project(self.chart_dir, self._selected_id, label=name.strip())
            new_id = (result.get("project") or {}).get("id")
        except Exception as exc:
            messagebox.showerror("Clone Project", str(exc), parent=self)
            return
        self._notify_changed()
        self.refresh()
        if new_id:
            self._load_project(str(new_id), force=True)

    def _delete_selected(self) -> None:
        if not self._selected_id:
            messagebox.showinfo("Delete Project", "Select a project first.", parent=self)
            return
        proj = self._project_by_id(self._selected_id) or {}
        label = proj.get("label") or self._selected_id
        if not messagebox.askyesno(
            "Delete Project",
            f'Delete project "{label}" ({self._selected_id})?\n\n'
            "This removes project metadata and feature references only.\n"
            "Feature Registry entries are NOT deleted.",
            parent=self,
        ):
            return
        try:
            svc.delete_project(self.chart_dir, self._selected_id)
        except Exception as exc:
            messagebox.showerror("Delete Project", str(exc), parent=self)
            return
        self._selected_id = None
        self._dirty = False
        self._notify_changed()
        self.refresh()

    def _rename_selected(self) -> None:
        if not self._selected_id:
            messagebox.showinfo("Rename Project", "Select a project first.", parent=self)
            return
        proj = self._project_by_id(self._selected_id) or {}
        name = simpledialog.askstring(
            "Rename Project",
            "Project name:",
            initialvalue=str(proj.get("label") or ""),
            parent=self,
        )
        if not name or not name.strip():
            return
        try:
            svc.update_project(self.chart_dir, self._selected_id, label=name.strip())
        except Exception as exc:
            messagebox.showerror("Rename Project", str(exc), parent=self)
            return
        self._notify_changed()
        self.refresh()
        self._load_project(self._selected_id, force=True)

    def _export_selected(self) -> None:
        if not self._selected_id:
            messagebox.showinfo("Export Project", "Select a project first.", parent=self)
            return
        proj = self._project_by_id(self._selected_id)
        if not proj:
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Export Feature Project",
            defaultextension=".json",
            initialfile=f"{self._selected_id}.json",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        payload = {
            "kind": "feature_project",
            "version": proj.get("version") or "1",
            "project": {
                "id": proj.get("id"),
                "label": proj.get("label"),
                "description": proj.get("description") or "",
                "group_ids": list(proj.get("group_ids") or []),
                "feature_names": list(proj.get("feature_names") or []),
                "warmup_minutes": proj.get("warmup_minutes"),
                "default_sampling": proj.get("default_sampling") or "",
                "notes": proj.get("notes") or "",
                "version": proj.get("version") or "1",
            },
        }
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except Exception as exc:
            messagebox.showerror("Export Project", str(exc), parent=self)
            return
        messagebox.showinfo("Export Project", f"Exported to:\n{path}", parent=self)

    def _import_project(self) -> None:
        if self._dirty and not self._confirm_discard():
            return
        path = filedialog.askopenfilename(
            parent=self,
            title="Import Feature Project",
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as fh:
                payload = json.load(fh)
            proj = payload.get("project") if isinstance(payload, dict) else None
            if not isinstance(proj, dict):
                proj = payload if isinstance(payload, dict) else None
            if not isinstance(proj, dict):
                raise ValueError("Invalid project JSON")
            label = str(proj.get("label") or "").strip()
            if not label:
                raise ValueError("Project label is required in import file")
            result = svc.create_project(
                self.chart_dir,
                label=label,
                project_id=str(proj.get("id") or "").strip() or None,
                description=str(proj.get("description") or ""),
                group_ids=list(proj.get("group_ids") or []),
                feature_names=list(proj.get("feature_names") or proj.get("enabled_features") or []),
                warmup_minutes=proj.get("warmup_minutes"),
                default_sampling=str(proj.get("default_sampling") or ""),
                notes=str(proj.get("notes") or ""),
                version=str(proj.get("version") or "1"),
            )
            new_id = (result.get("project") or {}).get("id")
        except Exception as exc:
            messagebox.showerror("Import Project", str(exc), parent=self)
            return
        self._notify_changed()
        self.refresh()
        if new_id:
            self._load_project(str(new_id), force=True)


def open_feature_project_manager_window(
    master: tk.Misc,
    *,
    chart_dir: str,
    on_changed: Callable[[], None] | None = None,
) -> tk.Toplevel:
    """Open Project Manager beside the main app (same pattern as Feature Policy)."""
    win = tk.Toplevel(master)
    win.title("Feature Project Manager")
    win.transient(master.winfo_toplevel())

    def _close() -> None:
        try:
            win.destroy()
        except tk.TclError:
            pass

    panel = FeatureProjectManagerPanel(
        win,
        chart_dir=chart_dir,
        on_changed=on_changed,
        on_close=_close,
    )
    panel.pack(fill="both", expand=True)
    win._project_manager_panel = panel  # type: ignore[attr-defined]
    win.protocol("WM_DELETE_WINDOW", _close)
    win.update_idletasks()
    place_toplevel_beside_main(win, master)
    panel.refresh()
    win.lift()
    win.focus_force()
    return win
