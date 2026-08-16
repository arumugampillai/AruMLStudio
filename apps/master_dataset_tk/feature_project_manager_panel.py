"""Feature Project Manager — project picker, editor, project-specific group organization."""

from __future__ import annotations

import json
import re
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk
from typing import Any, Callable

from chain_replay_ml.dataset_builder.feature_project_organization import (
    backfill_feature_group_map,
    canonical_group_for_feature,
    is_canonical_domain_id,
    is_reserved_all_project_id,
    migrate_project_organization,
    normalize_custom_project_groups,
    project_group_tree,
    project_registry_groups,
    RESERVED_ALL_PROJECT_ID,
    sync_project_group_ids,
)

from .build_config_prefs import active_feature_project_id, set_active_feature_project_id
from .fold_replay_widgets import place_toplevel_beside_main
from . import feature_registry_format as fmt
from . import feature_registry_service as svc


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


def _slug_group_id(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(label or "").strip().lower()).strip("_")
    return slug or "group"


_NEW_PROJECT_LABEL = "(new project — not saved)"
_ALL_PROJECTS_LABEL = RESERVED_ALL_PROJECT_ID


class FeatureProjectManagerPanel(ttk.Frame):
    """Feature Project Manager — references only; Registry remains source of truth."""

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
        self._group_tree_rows: list[dict[str, Any]] = []
        self._projects: list[dict[str, Any]] = []
        self._selected_id: str | None = RESERVED_ALL_PROJECT_ID
        self._is_new_project = False
        self._id_user_edited = False
        self._selected_features: set[str] = set()
        self._project_groups: list[dict[str, str]] = []
        self._feature_group_map: dict[str, str] = {}
        self._feature_meta: dict[str, dict[str, str]] = {}
        self._project_label_to_id: dict[str, str] = {}
        self._group_label_to_id: dict[str, str] = {}
        self._dirty = False
        self._loading = False
        self._project_var = tk.StringVar(value=_ALL_PROJECTS_LABEL)
        self._build_ui()

    def set_chart_dir(self, chart_dir: str) -> None:
        self.chart_dir = chart_dir

    def refresh(self) -> None:
        try:
            svc.ensure_all_project(self.chart_dir)
            self._catalog = svc.load_catalog(self.chart_dir)
            data_dir = svc.data_dir_for(self.chart_dir)
            self._group_tree_rows = project_group_tree(
                project_groups=self._project_groups,
                feature_group_map=self._feature_group_map,
                data_dir=data_dir,
            )
            self._projects = list((self._catalog or {}).get("projects") or [])
            if not self._projects:
                self._projects = list(svc.list_projects(self.chart_dir))
            self._build_feature_meta()
        except Exception as exc:
            messagebox.showerror("Feature Project Manager", f"Failed to load projects:\n{exc}", parent=self)
            return
        self._populate_project_combo()
        if self._is_new_project:
            self._reload_group_tree_rows()
            self._render_group_tree()
            self._refresh_group_combos()
            self._update_editing_state()
            return
        if not self._dirty:
            self._selected_id = active_feature_project_id(self.chart_dir)
        load_id = self._selected_id or RESERVED_ALL_PROJECT_ID
        if any(p.get("id") == load_id for p in self._projects):
            self._load_project(load_id, force=True)
        else:
            self._load_project(RESERVED_ALL_PROJECT_ID, force=True)

    def _data_dir(self) -> str:
        return svc.data_dir_for(self.chart_dir)

    def _reload_group_tree_rows(self) -> None:
        self._group_tree_rows = project_group_tree(
            project_groups=self._project_groups,
            feature_group_map=self._feature_group_map,
            data_dir=self._data_dir(),
        )

    def _notify_changed(self) -> None:
        if self._on_changed:
            self._on_changed()

    def _build_feature_meta(self) -> None:
        self._feature_meta = {}
        for row in (self._catalog or {}).get("features") or []:
            if not isinstance(row, dict):
                continue
            name = str(row.get("name") or "").strip()
            if not name:
                continue
            self._feature_meta[name] = {
                "feature_id": str(row.get("feature_id") or "").strip(),
                "label": str(row.get("display_name") or row.get("label") or name).strip(),
            }

    def _build_ui(self) -> None:
        hdr = ttk.Frame(self, padding=(8, 6, 8, 0))
        hdr.pack(fill="x")
        ttk.Label(hdr, text="Feature Project Manager", font=("Segoe UI", 12, "bold")).pack(side="left")
        ttk.Label(
            hdr,
            text="Projects organize Registry features — moving groups does not change the Registry.",
            foreground="#888",
        ).pack(side="left", padx=(10, 0))
        if self._on_close:
            ttk.Button(hdr, text="Close", command=self._on_close).pack(side="right")

        outer = ttk.Frame(self, padding=8)
        outer.pack(fill="both", expand=True)
        scroll = ttk.Scrollbar(outer, orient=tk.VERTICAL)
        canvas = tk.Canvas(outer, highlightthickness=0)
        scroll.config(command=canvas.yview)
        canvas.config(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill="both", expand=True)
        body = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=body, anchor="nw")

        def _on_configure(_event: tk.Event) -> None:
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfigure(canvas_window, width=event.width)

        body.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>", _on_canvas_configure)

        top = ttk.Frame(body)
        top.pack(fill="x", pady=(0, 8))
        ttk.Label(top, text="Project:").pack(side="left")
        self._project_cb = ttk.Combobox(
            top,
            textvariable=self._project_var,
            width=36,
            state="readonly",
        )
        self._project_cb.pack(side="left", padx=(8, 12))
        self._project_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_project_selected())
        self._project_action_buttons: list[ttk.Button] = []
        self._btn_delete: ttk.Button | None = None
        for text, cmd in (
            ("New Project", self._new_project),
            ("Clone", self._clone_selected),
            ("Delete", self._delete_selected),
            ("Rename", self._rename_selected),
            ("Save Project", self._save_current),
        ):
            btn = ttk.Button(top, text=text, command=cmd)
            btn.pack(side="left", padx=(0, 4))
            if text == "Delete":
                self._btn_delete = btn
            if text != "New Project":
                self._project_action_buttons.append(btn)
        ttk.Button(top, text="Refresh", command=self.refresh).pack(side="left", padx=(8, 0))

        ttk.Separator(body, orient=tk.HORIZONTAL).pack(fill="x", pady=(0, 8))

        meta = ttk.LabelFrame(body, text="Project Details", padding=8)
        meta.pack(fill="x", pady=(0, 8))
        grid = ttk.Frame(meta)
        grid.pack(fill="x")
        self._name_var = tk.StringVar()
        self._id_var = tk.StringVar()
        self._warmup_var = tk.StringVar()
        self._sampling_var = tk.StringVar()
        self._editable_entries: list[ttk.Entry] = []
        rows = (
            ("Name", self._name_var, False),
            ("ID", self._id_var, False),
            ("Warmup", self._warmup_var, False),
            ("Sampling", self._sampling_var, False),
        )
        for i, (label, var, readonly) in enumerate(rows):
            ttk.Label(grid, text=label).grid(row=i, column=0, sticky="w", pady=2, padx=(0, 8))
            state = "readonly" if readonly else "normal"
            ent = ttk.Entry(grid, textvariable=var, width=40, state=state)
            ent.grid(row=i, column=1, sticky="ew", pady=2)
            self._editable_entries.append(ent)
            if label == "ID":
                self._id_entry = ent
            if not readonly:
                var.trace_add("write", lambda *_: self._mark_dirty())
        self._name_var.trace_add("write", lambda *_: self._on_name_changed())
        self._id_var.trace_add("write", lambda *_: self._on_id_changed())
        grid.columnconfigure(1, weight=1)
        ttk.Label(meta, text="Description").pack(anchor="w", pady=(8, 2))
        self._desc = scrolledtext.ScrolledText(meta, height=2, font=("Segoe UI", 9), wrap="word")
        self._desc.pack(fill="x")
        self._desc.bind("<<Modified>>", self._on_desc_modified)

        ttk.Separator(body, orient=tk.HORIZONTAL).pack(fill="x", pady=(0, 8))

        groups_box = ttk.LabelFrame(body, text="Feature Groups", padding=8)
        groups_box.pack(fill="both", expand=True, pady=(0, 8))
        grp_tools = ttk.Frame(groups_box)
        grp_tools.pack(fill="x", pady=(0, 6))
        self._btn_new_group = ttk.Button(grp_tools, text="+ New Group", command=self._new_group)
        self._btn_new_group.pack(side="left")
        self._group_tree = self._make_group_tree(groups_box)

        feat_box = ttk.Frame(body)
        feat_box.pack(fill="x", pady=(0, 8))
        lists = ttk.Frame(feat_box)
        lists.pack(fill="x")
        left_box = ttk.Frame(lists)
        left_box.pack(side="left", fill="both", expand=True, padx=(0, 4))
        ttk.Label(left_box, text="Available Features").pack(anchor="w")
        self._avail_search_var = tk.StringVar()
        self._avail_search_var.trace_add("write", lambda *_: self._refresh_available())
        ttk.Entry(left_box, textvariable=self._avail_search_var).pack(fill="x", pady=(2, 4))
        self._avail = tk.Listbox(left_box, selectmode="extended", exportselection=False, height=8)
        self._avail.pack(fill="both", expand=True)

        mid = ttk.Frame(lists)
        mid.pack(side="left", padx=8, fill="y")
        self._btn_add = ttk.Button(mid, text="Add →", command=self._add_features)
        self._btn_add.pack(pady=4)
        self._btn_remove = ttk.Button(mid, text="← Remove", command=self._remove_features)
        self._btn_remove.pack(pady=4)

        move_row = ttk.Frame(feat_box)
        move_row.pack(fill="x", pady=(8, 0))
        ttk.Label(move_row, text="Move to Group:").pack(side="left")
        self._move_group_var = tk.StringVar()
        self._move_group_cb = ttk.Combobox(move_row, textvariable=self._move_group_var, width=28, state="readonly")
        self._move_group_cb.pack(side="left", padx=(6, 8))
        self._btn_move = ttk.Button(move_row, text="Move", command=self._move_selected_to_group)
        self._btn_move.pack(side="left")

        ttk.Separator(body, orient=tk.HORIZONTAL).pack(fill="x", pady=(8, 8))

        summary = ttk.LabelFrame(body, text="Project Summary", padding=8)
        summary.pack(fill="x")
        self._summary_var = tk.StringVar(value="Select or create a project.")
        ttk.Label(summary, textvariable=self._summary_var, justify="left").pack(anchor="w")
        self._dirty_var = tk.StringVar(value="")
        ttk.Label(summary, textvariable=self._dirty_var, foreground="#B45309").pack(anchor="w", pady=(6, 0))

    def _mark_dirty(self) -> None:
        if self._loading:
            return
        self._dirty = True
        self._dirty_var.set("Unsaved changes")

    def _on_desc_modified(self, _event: tk.Event | None = None) -> None:
        if self._desc.edit_modified():
            self._desc.edit_modified(False)
            self._mark_dirty()

    def _on_name_changed(self) -> None:
        if self._loading or not self._is_new_project or self._id_user_edited:
            return
        label = self._name_var.get().strip()
        if label:
            self._loading = True
            try:
                self._id_var.set(svc.suggest_project_id(self.chart_dir, label))
            finally:
                self._loading = False

    def _on_id_changed(self) -> None:
        if self._loading:
            return
        if self._is_new_project:
            self._id_user_edited = True

    def _populate_project_combo(self) -> None:
        labels: list[str] = [_ALL_PROJECTS_LABEL]
        self._project_label_to_id = {_ALL_PROJECTS_LABEL: RESERVED_ALL_PROJECT_ID}
        for p in self._projects:
            pid = str(p.get("id") or "").strip()
            if not pid or is_reserved_all_project_id(pid):
                continue
            name = str(p.get("label") or pid)
            label = f"{name} ({pid})"
            labels.append(label)
            self._project_label_to_id[label] = pid
        if self._is_new_project:
            labels.insert(1, _NEW_PROJECT_LABEL)
            self._project_label_to_id[_NEW_PROJECT_LABEL] = ""
        self._project_cb["values"] = labels
        if self._is_new_project:
            self._project_var.set(_NEW_PROJECT_LABEL)
            return
        want = self._selected_id
        if want:
            for label, pid in self._project_label_to_id.items():
                if pid == want:
                    self._project_var.set(label)
                    return
        self._project_var.set(_ALL_PROJECTS_LABEL)

    def _on_project_selected(self) -> None:
        label = str(self._project_var.get() or "").strip()
        if label == _NEW_PROJECT_LABEL:
            return
        if label == _ALL_PROJECTS_LABEL:
            pid = RESERVED_ALL_PROJECT_ID
        else:
            pid = self._project_label_to_id.get(label, "")
        if not pid:
            self._project_var.set(_ALL_PROJECTS_LABEL)
            return
        if pid == self._selected_id and not self._dirty and not self._is_new_project:
            return
        if self._dirty and not self._confirm_discard():
            self._populate_project_combo()
            return
        self._is_new_project = False
        self._id_user_edited = False
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

    def _clear_editor(self, *, summary: str | None = None) -> None:
        self._loading = True
        self._selected_id = None
        self._is_new_project = False
        self._id_user_edited = False
        self._selected_features.clear()
        self._project_groups = []
        self._feature_group_map = {}
        self._name_var.set("")
        self._id_var.set("")
        self._warmup_var.set("")
        self._sampling_var.set("")
        self._desc.delete("1.0", tk.END)
        self._reload_group_tree_rows()
        self._render_group_tree()
        self._avail.delete(0, tk.END)
        if summary is None:
            summary = (
                "No projects yet. Click New Project."
                if not self._projects
                else "Browse mode — all Registry groups shown.\n"
                "Select a project from the list, or click New Project to edit membership."
            )
        self._summary_var.set(summary)
        self._dirty = False
        self._dirty_var.set("")
        self._refresh_available()
        self._refresh_group_combos()
        try:
            self._id_entry.configure(state="normal")
        except tk.TclError:
            pass
        self._loading = False
        self._update_editing_state()

    def _load_project(self, pid: str, *, force: bool = False) -> None:
        if not force and pid == self._selected_id and not self._dirty:
            return
        proj = self._project_by_id(pid)
        if not proj:
            self._load_project(RESERVED_ALL_PROJECT_ID, force=True)
            return
        migrated = migrate_project_organization(dict(proj), data_dir=self._data_dir())
        self._loading = True
        self._is_new_project = False
        self._id_user_edited = False
        self._selected_id = pid
        self._selected_features = {
            str(n) for n in (migrated.get("feature_names") or migrated.get("enabled_features") or [])
        }
        self._project_groups = normalize_custom_project_groups(migrated.get("project_groups"))
        self._feature_group_map = backfill_feature_group_map(
            self._selected_features,
            project_groups=self._project_groups,
            feature_group_map=dict(migrated.get("feature_group_map") or {}),
        )
        self._name_var.set(str(migrated.get("label") or ""))
        self._id_var.set(pid)
        try:
            self._id_entry.configure(state="readonly")
        except tk.TclError:
            pass
        self._warmup_var.set("" if migrated.get("warmup_minutes") is None else str(migrated.get("warmup_minutes")))
        self._sampling_var.set(str(migrated.get("default_sampling") or ""))
        self._desc.delete("1.0", tk.END)
        self._desc.insert("1.0", str(migrated.get("description") or ""))
        self._desc.edit_modified(False)
        self._reload_group_tree_rows()
        self._render_group_tree()
        self._refresh_group_combos()
        self._refresh_available()
        self._update_summary(migrated)
        self._populate_project_combo()
        self._project_var.set(
            next(
                (lbl for lbl, p in self._project_label_to_id.items() if p == pid),
                _ALL_PROJECTS_LABEL,
            )
        )
        self._dirty = False
        self._dirty_var.set("")
        self._loading = False
        set_active_feature_project_id(self.chart_dir, pid)
        self._update_editing_state()

    def _feature_display(self, name: str) -> str:
        meta = self._feature_meta.get(name) or {}
        fid = meta.get("feature_id") or "—"
        return f"{fid}  {name}"

    def _all_registry_feature_names(self) -> list[str]:
        names: list[str] = []
        for row in (self._catalog or {}).get("features") or []:
            if isinstance(row, dict) and row.get("name"):
                names.append(str(row["name"]))
        return list(dict.fromkeys(names))

    def _make_group_tree(self, parent: tk.Misc) -> ttk.Treeview:
        frame = ttk.Frame(parent, height=280)
        frame.pack(fill="x", expand=False)
        frame.pack_propagate(False)
        tree = ttk.Treeview(frame, columns=("count",), show="tree headings", height=14)
        tree.heading("#0", text="Group / Feature")
        tree.heading("count", text="#")
        tree.column("#0", width=280, stretch=True)
        tree.column("count", width=48, anchor="e")
        sb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        tree.bind("<Button-3>", self._on_group_tree_context_menu)
        tree.bind("<Button-2>", self._on_group_tree_context_menu)
        return tree

    def _update_editing_state(self) -> None:
        reserved = is_reserved_all_project_id(self._selected_id or "")
        if self._btn_delete is not None:
            self._btn_delete.configure(state="disabled" if reserved else "normal")
        for ent in self._editable_entries:
            if ent is self._id_entry:
                continue
            ent.configure(state="normal")
        self._desc.configure(state="normal")
        try:
            if self._is_new_project:
                self._id_entry.configure(state="normal")
            else:
                self._id_entry.configure(state="readonly")
        except tk.TclError:
            pass

    def _is_project_active(self) -> bool:
        return self._is_new_project or self._selected_id is not None

    def _group_tree_options(self) -> list[dict[str, str]]:
        return [
            {"id": str(g.get("id") or ""), "label": str(g.get("label") or g.get("id") or "")}
            for g in self._group_tree_rows
            if str(g.get("id") or "").strip()
        ]

    def _render_group_tree(self) -> None:
        tree = self._group_tree
        tree.delete(*tree.get_children())
        pid = self._selected_id or RESERVED_ALL_PROJECT_ID
        data_dir = self._data_dir()
        is_all = is_reserved_all_project_id(pid)

        doc = {
            "feature_names": sorted(self._selected_features),
            "project_groups": self._project_groups,
            "feature_group_map": self._feature_group_map,
        }
        groups = project_registry_groups(doc, data_dir=data_dir)
        inserted_groups = 0
        for g in groups:
            gid = str(g.get("id") or "").strip()
            if not gid:
                continue
            label = str(g.get("label") or gid)
            feats = list(g.get("features") or [])
            parent_iid = tree.insert(
                "",
                "end",
                iid=f"group:{gid}",
                text=label,
                values=(len(feats),),
                open=False,
            )
            inserted_groups += 1
            for feat in feats:
                tree.insert(parent_iid, "end", iid=f"feat:{feat}", text=feat, values=("",))

        # Diagnostic logging for Feature Project Manager
        import logging
        logger = logging.getLogger("AruMLStudio.FeatureProjectManager")
        logger.info(
            f"[FeatureProjectManager] project_id='{pid}', is_all={is_all}, "
            f"data_dir='{data_dir}', registry_features_loaded={len(self._selected_features)}, "
            f"canonical_groups_loaded={len(groups)}, tree_root_nodes_inserted={inserted_groups}"
        )

    def _tree_feature_name(self, iid: str) -> str | None:
        if str(iid).startswith("feat:"):
            return str(iid)[5:]
        return None

    def _tree_group_id(self, iid: str) -> str | None:
        if str(iid).startswith("group:"):
            return str(iid)[6:]
        return None

    def _on_group_tree_context_menu(self, event: tk.Event) -> None:
        tree = self._group_tree
        iid = tree.identify_row(event.y)
        if not iid:
            return
        tree.selection_set(iid)
        feature_name = self._tree_feature_name(iid)
        group_id = self._tree_group_id(iid)

        menu = tk.Menu(self, tearoff=0)
        if feature_name:
            self._build_feature_context_menu(menu, feature_name)
        elif group_id and self._is_project_active() and not is_reserved_all_project_id(self._selected_id or ""):
            self._build_group_context_menu(menu, group_id)
        else:
            return

        try:
            menu.tk_popup(event.x_root, event.y_root)
        finally:
            menu.grab_release()

    def _build_feature_context_menu(self, menu: tk.Menu, feature_name: str) -> None:
        picked = self._tree_selected_features()
        if feature_name not in picked:
            picked = [feature_name]

        menu.add_command(
            label="Feature Details",
            command=lambda name=feature_name: self._show_feature_details(name),
        )

        reserved = is_reserved_all_project_id(self._selected_id or "")
        if not reserved and self._is_project_active() and any(n in self._selected_features for n in picked):
            menu.add_separator()
            move_sub = tk.Menu(menu, tearoff=0)
            self._populate_move_to_group_menu(move_sub, picked)
            menu.add_cascade(label="Move to Group", menu=move_sub)
            menu.add_command(
                label="Remove from Project",
                command=lambda names=picked: self._remove_features_from_project(names),
            )

    def _build_group_context_menu(self, menu: tk.Menu, group_id: str) -> None:
        menu.add_command(label="Create Group", command=self._new_group)
        if is_canonical_domain_id(group_id):
            return
        if not any(str(g.get("id") or "") == group_id for g in self._project_groups):
            return
        menu.add_separator()
        menu.add_command(
            label="Rename Custom Group",
            command=lambda gid=group_id: self._rename_custom_group(gid),
        )
        menu.add_command(
            label="Delete Custom Group",
            command=lambda gid=group_id: self._delete_custom_group(gid),
        )

    def _rename_custom_group(self, group_id: str) -> None:
        if not self._is_project_active() or is_canonical_domain_id(group_id):
            return
        entry = next(
            (g for g in self._project_groups if str(g.get("id") or "") == group_id),
            None,
        )
        if not entry:
            return
        current = str(entry.get("label") or group_id)
        new_label = simpledialog.askstring(
            "Rename Custom Group",
            "Group name:",
            initialvalue=current,
            parent=self,
        )
        if not new_label or not str(new_label).strip():
            return
        text = str(new_label).strip()
        entry["label"] = text
        self._reload_group_tree_rows()
        self._render_group_tree()
        self._refresh_group_combos()
        if self._move_group_var.get() == current:
            self._move_group_var.set(text)
        self._mark_dirty()

    def _delete_custom_group(self, group_id: str) -> None:
        if not self._is_project_active() or is_canonical_domain_id(group_id):
            return
        label = next(
            (str(g.get("label") or group_id) for g in self._project_groups if g.get("id") == group_id),
            group_id,
        )
        if not messagebox.askyesno(
            "Delete Custom Group",
            f"Delete custom group \"{label}\"?\n\n"
            "Features in this group will return to their canonical Registry groups.",
            parent=self,
        ):
            return
        self._project_groups = [
            g for g in self._project_groups if str(g.get("id") or "") != group_id
        ]
        for name, gid in list(self._feature_group_map.items()):
            if str(gid) == group_id:
                self._feature_group_map[name] = canonical_group_for_feature(name)
        self._reload_group_tree_rows()
        self._render_group_tree()
        self._refresh_group_combos()
        self._update_summary()
        self._mark_dirty()

    def _populate_move_to_group_menu(self, menu: tk.Menu, feature_names: list[str]) -> None:
        canonical: list[dict[str, str]] = []
        custom: list[dict[str, str]] = []
        for g in self._group_tree_options():
            if is_canonical_domain_id(g["id"]):
                canonical.append(g)
            else:
                custom.append(g)
        for g in canonical:
            label = str(g.get("label") or g.get("id") or "")
            gid = str(g.get("id") or "")
            menu.add_command(
                label=label,
                command=lambda target=gid, names=feature_names: self._move_features_to_group(names, target),
            )
        if custom:
            menu.add_separator()
            for g in custom:
                label = str(g.get("label") or g.get("id") or "")
                gid = str(g.get("id") or "")
                menu.add_command(
                    label=label,
                    command=lambda target=gid, names=feature_names: self._move_features_to_group(names, target),
                )

    def _move_features_to_group(self, feature_names: list[str], target_gid: str) -> None:
        if not self._is_project_active() or not target_gid:
            return
        moved = False
        for name in feature_names:
            if name in self._selected_features:
                self._feature_group_map[name] = target_gid
                moved = True
        if not moved:
            return
        self._reload_group_tree_rows()
        self._render_group_tree()
        self._refresh_group_combos()
        self._update_summary()
        self._mark_dirty()

    def _remove_features_from_project(self, feature_names: list[str]) -> None:
        if not self._is_project_active():
            return
        removed = False
        for name in feature_names:
            if name in self._selected_features:
                self._selected_features.discard(name)
                self._feature_group_map.pop(name, None)
                removed = True
        if not removed:
            return
        self._refresh_available()
        self._reload_group_tree_rows()
        self._render_group_tree()
        self._refresh_group_combos()
        self._update_summary()
        self._mark_dirty()

    def _show_feature_details(self, name: str) -> None:
        row: dict[str, Any] | None = None
        for feat in (self._catalog or {}).get("features") or []:
            if isinstance(feat, dict) and str(feat.get("name") or "") == name:
                row = feat
                break
        text = fmt.format_feature_detail(row or {"name": name}, self._catalog)
        win = tk.Toplevel(self)
        win.title(f"Feature Details — {name}")
        win.transient(self.winfo_toplevel())
        win.geometry("520x420")
        body = ttk.Frame(win, padding=8)
        body.pack(fill="both", expand=True)
        detail = scrolledtext.ScrolledText(body, wrap="word", font=("Consolas", 9))
        detail.pack(fill="both", expand=True)
        detail.insert("1.0", text)
        detail.configure(state="disabled")
        ttk.Button(body, text="Close", command=win.destroy).pack(pady=(8, 0))
        win.update_idletasks()
        place_toplevel_beside_main(win, self)

    def _tree_selected_features(self) -> list[str]:
        out: list[str] = []
        for iid in self._group_tree.selection():
            if str(iid).startswith("feat:"):
                out.append(str(iid)[5:])
        return out

    def _tree_selected_group_id(self) -> str | None:
        for iid in self._group_tree.selection():
            if str(iid).startswith("group:"):
                return str(iid)[6:]
        return None

    def _refresh_group_combos(self) -> None:
        self._group_label_to_id = {}
        move_labels: list[str] = []
        used_labels: set[str] = set()
        for g in self._group_tree_options():
            label = str(g.get("label") or g.get("id") or "")
            gid = str(g.get("id") or "")
            if not gid:
                continue
            combo_label = label
            if combo_label in used_labels:
                combo_label = f"{label} ({gid})"
            used_labels.add(combo_label)
            self._group_label_to_id[combo_label] = gid
            move_labels.append(combo_label)
        self._move_group_cb["values"] = move_labels
        if move_labels and self._move_group_var.get() not in move_labels:
            self._move_group_var.set(move_labels[0])

    def _refresh_available(self) -> None:
        q = self._avail_search_var.get().strip().lower()
        self._avail.delete(0, tk.END)
        for name in self._all_registry_feature_names():
            if name in self._selected_features:
                continue
            if q and q not in name.lower():
                fid = (self._feature_meta.get(name) or {}).get("feature_id") or ""
                if q not in str(fid).lower():
                    continue
            self._avail.insert(tk.END, self._feature_display(name))

    def _update_summary(self, proj: dict[str, Any] | None = None) -> None:
        group_count = len(self._group_tree_options())
        warm = self._warmup_var.get().strip()
        status = "ready" if self._selected_features else "empty"
        self._summary_var.set(
            f"Features: {len(self._selected_features)}\n"
            f"Groups: {group_count}\n"
            f"Models: — (future)\n"
            f"Status: {status}\n"
            f"Warmup: {_warmup_label(warm if warm else None)}\n"
            f"Version: {(proj or {}).get('version') or '1'}"
        )

    def _default_assign_group(self, feature_name: str | None = None) -> str | None:
        tree_gid = self._tree_selected_group_id()
        if tree_gid:
            return tree_gid
        if feature_name:
            return canonical_group_for_feature(feature_name)
        options = self._group_tree_options()
        if options:
            return str(options[0]["id"])
        return None

    def _add_features(self) -> None:
        if not self._is_project_active():
            return
        picked: list[str] = []
        for i in self._avail.curselection():
            line = self._avail.get(i)
            parts = str(line).split()
            if len(parts) >= 2:
                picked.append(parts[-1])
        if not picked:
            return
        for name in picked:
            self._selected_features.add(name)
            gid = self._default_assign_group(name)
            if gid:
                self._feature_group_map[name] = gid
        self._refresh_available()
        self._reload_group_tree_rows()
        self._render_group_tree()
        self._refresh_group_combos()
        self._update_summary()
        self._mark_dirty()

    def _remove_features(self) -> None:
        if not self._is_project_active():
            return
        picked = self._tree_selected_features()
        if not picked:
            picked = []
            for i in self._avail.curselection():
                line = self._avail.get(i)
                parts = str(line).split()
                if len(parts) >= 2:
                    picked.append(parts[-1])
        if not picked:
            return
        for name in picked:
            self._selected_features.discard(name)
            self._feature_group_map.pop(name, None)
        self._refresh_available()
        self._reload_group_tree_rows()
        self._render_group_tree()
        self._refresh_group_combos()
        self._update_summary()
        self._mark_dirty()

    def _move_selected_to_group(self) -> None:
        if not self._is_project_active():
            return
        target_label = str(self._move_group_var.get() or "").strip()
        target_gid = self._group_label_to_id.get(target_label)
        if not target_gid:
            messagebox.showinfo("Move to Group", "Select a destination group.", parent=self)
            return
        picked = self._tree_selected_features()
        if not picked:
            messagebox.showinfo("Move to Group", "Select one or more features in the group tree.", parent=self)
            return
        self._move_features_to_group(picked, target_gid)

    def _new_group(self) -> None:
        if not self._is_project_active():
            return
        label = simpledialog.askstring("New Group", "Group name:", parent=self)
        if not label or not str(label).strip():
            return
        text = str(label).strip()
        base = _slug_group_id(text)
        if is_canonical_domain_id(base):
            messagebox.showerror(
                "New Group",
                f'"{text}" matches a canonical Registry group. Use the existing group in the tree.',
                parent=self,
            )
            return
        existing = {str(g.get("id") or "") for g in self._project_groups}
        gid = base
        if gid in existing:
            n = 2
            while f"{base}_{n}" in existing:
                n += 1
            gid = f"{base}_{n}"
        self._project_groups.append({"id": gid, "label": text})
        self._reload_group_tree_rows()
        self._render_group_tree()
        self._refresh_group_combos()
        self._move_group_var.set(text)
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
        label = self._name_var.get().strip()
        if not label:
            messagebox.showinfo("Save Project", "Project name is required.", parent=self)
            return
        pid = self._id_var.get().strip().lower()
        if not pid:
            messagebox.showinfo("Save Project", "Project ID is required.", parent=self)
            return
        try:
            warmup = self._parse_warmup()
            feature_names = sorted(self._selected_features)
            map_out = backfill_feature_group_map(
                feature_names,
                project_groups=self._project_groups,
                feature_group_map=self._feature_group_map,
            )
            group_ids = sync_project_group_ids(feature_names, map_out)
            kwargs = {
                "label": label,
                "description": self._desc.get("1.0", tk.END).strip(),
                "group_ids": group_ids,
                "feature_names": feature_names,
                "project_groups": self._project_groups,
                "feature_group_map": map_out,
                "warmup_minutes": warmup,
                "default_sampling": self._sampling_var.get().strip(),
            }
            if self._is_new_project:
                svc.create_project(self.chart_dir, project_id=pid, **kwargs)
                self._is_new_project = False
                self._selected_id = pid
                try:
                    self._id_entry.configure(state="readonly")
                except tk.TclError:
                    pass
            else:
                if not self._selected_id:
                    messagebox.showinfo("Save Project", "Select or create a project first.", parent=self)
                    return
                svc.update_project(self.chart_dir, self._selected_id, **kwargs)
        except Exception as exc:
            messagebox.showerror("Save Project", str(exc), parent=self)
            return
        self._dirty = False
        self._dirty_var.set("")
        self._notify_changed()
        set_active_feature_project_id(self.chart_dir, self._selected_id or RESERVED_ALL_PROJECT_ID)
        self.refresh()
        messagebox.showinfo("Save Project", "Project saved.", parent=self)

    def _new_project(self) -> None:
        if self._dirty and not self._confirm_discard():
            return
        self._loading = True
        self._is_new_project = True
        self._id_user_edited = False
        self._selected_id = None
        self._selected_features.clear()
        self._project_groups = []
        self._feature_group_map = {}
        suggested = svc.suggest_project_id(self.chart_dir, "")
        self._name_var.set("")
        self._id_var.set(suggested)
        try:
            self._id_entry.configure(state="normal")
        except tk.TclError:
            pass
        self._warmup_var.set("")
        self._sampling_var.set("")
        self._desc.delete("1.0", tk.END)
        self._avail_search_var.set("")
        self._reload_group_tree_rows()
        self._render_group_tree()
        self._avail.delete(0, tk.END)
        self._refresh_group_combos()
        self._update_summary()
        self._populate_project_combo()
        self._project_var.set(_NEW_PROJECT_LABEL)
        self._dirty = False
        self._dirty_var.set("")
        self._loading = False
        self._update_editing_state()

    def _clone_selected(self) -> None:
        if self._is_new_project:
            messagebox.showinfo("Clone Project", "Save or discard the new project first.", parent=self)
            return
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
        if self._is_new_project:
            self._new_project()
            return
        if not self._selected_id:
            messagebox.showinfo("Delete Project", "Select a project first.", parent=self)
            return
        if is_reserved_all_project_id(self._selected_id):
            messagebox.showinfo(
                "Delete Project",
                "The reserved project 'all' cannot be deleted.",
                parent=self,
            )
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
        self._selected_id = RESERVED_ALL_PROJECT_ID
        self._dirty = False
        self._notify_changed()
        self.refresh()
        self._load_project(RESERVED_ALL_PROJECT_ID, force=True)

    def _rename_selected(self) -> None:
        if self._is_new_project:
            messagebox.showinfo("Rename Project", "Enter the name in Project Details and Save.", parent=self)
            return
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
        self._name_var.set(name.strip())
        self._mark_dirty()

    def _export_selected(self) -> None:
        if self._is_new_project or not self._selected_id:
            messagebox.showinfo("Export Project", "Save the project before exporting.", parent=self)
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
                "project_groups": list(proj.get("project_groups") or []),
                "feature_group_map": dict(proj.get("feature_group_map") or {}),
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
                project_groups=list(proj.get("project_groups") or []),
                feature_group_map=dict(proj.get("feature_group_map") or {}),
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
