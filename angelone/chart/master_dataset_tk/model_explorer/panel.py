"""XGBoost Model Explorer — Tk panel (Phase 1, read-only)."""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .loader import LoadError, load_booster
from .registry_source import (
    format_registry_summary_text,
    list_xgboost_registry_models,
    load_registry_summary_for_model,
    registry_model_labels,
    resolve_registry_xgboost_artifact,
)
from .render import check_graphviz
from .tree_stats import (
    NodeDetails,
    build_feature_usage,
    build_model_summary,
    build_tree_statistics,
    list_nodes_for_tree,
    tree_count,
)
from .node_viewer import (
    NodeViewerPanel,
    format_node_details_text,
    open_node_viewer_window,
)
from .tree_viewer import TreeViewerPanel, open_tree_viewer_window


class ModelExplorerPanel(ttk.Frame):
    """Standalone Model Builder page: load booster, inspect trees, open Tree Viewer."""

    def __init__(self, master: tk.Misc, *, chart_dir: str = "") -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._booster: Any | None = None
        self._source_path: str = ""
        self._source_label: str = ""
        self._registry_models: list[dict[str, Any]] = []
        self._nodes: list[NodeDetails] = []
        self._suppress_registry_select = False
        self._external_visible = False
        self._viewer_win: tk.Toplevel | None = None
        self._viewer_panel: TreeViewerPanel | None = None
        self._node_viewer_win: tk.Toplevel | None = None
        self._node_viewer_panel: NodeViewerPanel | None = None

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._registry_var = tk.StringVar()
        self._path_var = tk.StringVar()
        self._status_var = tk.StringVar(
            value="Select an XGBoost model from the Model Registry, or load an external file."
        )
        self._gv_var = tk.StringVar(value=check_graphviz().message)

        self._build_toolbar()
        self._build_body()
        self.after(100, self._refresh_graphviz_status)
        self.after(150, self.refresh_registry_models)

    def _data_dir(self) -> str:
        from ..build_service import chart_data_dir

        return chart_data_dir(self.chart_dir) if self.chart_dir else ""

    # ------------------------------------------------------------------ UI
    def _build_toolbar(self) -> None:
        bar = ttk.Frame(self, padding=0)
        bar.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        bar.columnconfigure(0, weight=1)

        # Primary: Model Registry
        reg = ttk.LabelFrame(bar, text="Model Registry (XGBoost)", padding=6)
        reg.grid(row=0, column=0, sticky="ew")

        ttk.Label(reg, text="Model").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self._registry_combo = ttk.Combobox(
            reg,
            textvariable=self._registry_var,
            state="readonly",
            width=24,
        )
        self._registry_combo.grid(row=0, column=1, sticky="w")
        self._registry_combo.bind("<<ComboboxSelected>>", self._on_registry_selected)
        ttk.Button(reg, text="Load", command=self._load_registry_model).grid(
            row=0, column=2, padx=4
        )
        ttk.Button(reg, text="Refresh", command=self.refresh_registry_models).grid(
            row=0, column=3, padx=4
        )
        ttk.Button(reg, text="Tree Viewer…", command=self._open_tree_viewer).grid(
            row=0, column=4, padx=4
        )
        ttk.Button(reg, text="Node Viewer…", command=self._open_node_viewer).grid(
            row=0, column=5, padx=4
        )

        # Read-only Model Registry summary strip (metadata — not booster dump)
        self._registry_summary = tk.Text(
            reg,
            height=14,
            wrap="word",
            state="disabled",
            font=("Consolas", 9),
            relief="solid",
            borderwidth=1,
        )
        self._registry_summary.grid(
            row=1, column=0, columnspan=6, sticky="ew", pady=(8, 0)
        )
        self._clear_registry_summary()

        # Secondary: external file (collapsible)
        self._external_toggle_btn = ttk.Button(
            bar,
            text="▸ Load External (standalone file)",
            command=self._toggle_external,
        )
        self._external_toggle_btn.grid(row=1, column=0, sticky="w", pady=(6, 0))

        self._external_frame = ttk.LabelFrame(
            bar, text="Load External (standalone)", padding=6
        )
        self._external_frame.columnconfigure(1, weight=1)
        ttk.Label(self._external_frame, text="Path").grid(
            row=0, column=0, sticky="w", padx=(0, 6)
        )
        ttk.Entry(self._external_frame, textvariable=self._path_var).grid(
            row=0, column=1, sticky="ew"
        )
        ttk.Button(self._external_frame, text="Browse…", command=self._browse).grid(
            row=0, column=2, padx=4
        )
        ttk.Button(
            self._external_frame, text="Load Model", command=self._load_external_model
        ).grid(row=0, column=3, padx=4)
        # Hidden by default — secondary path
        self._external_frame.grid_remove()

        ttk.Label(bar, textvariable=self._status_var, wraplength=900).grid(
            row=3, column=0, sticky="ew", pady=(6, 0)
        )
        ttk.Label(bar, textvariable=self._gv_var, foreground="#888", wraplength=900).grid(
            row=4, column=0, sticky="ew", pady=(2, 0)
        )

    def _toggle_external(self) -> None:
        self._external_visible = not self._external_visible
        if self._external_visible:
            self._external_frame.grid(row=2, column=0, sticky="ew", pady=(4, 0))
            self._external_toggle_btn.configure(text="▾ Load External (standalone file)")
        else:
            self._external_frame.grid_remove()
            self._external_toggle_btn.configure(text="▸ Load External (standalone file)")

    def _build_body(self) -> None:
        body = ttk.Panedwindow(self, orient="horizontal")
        body.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)

        left = ttk.Frame(body, padding=2)
        right = ttk.Frame(body, padding=2)
        body.add(left, weight=1)
        body.add(right, weight=2)

        left.columnconfigure(0, weight=1)
        left.rowconfigure(1, weight=1)
        left.rowconfigure(3, weight=1)

        ttk.Label(left, text="Trees", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky="w")
        self._tree_list = tk.Listbox(left, exportselection=False, height=12)
        self._tree_list.grid(row=1, column=0, sticky="nsew", pady=(2, 6))
        self._tree_list.bind("<<ListboxSelect>>", self._on_tree_select)

        ttk.Label(left, text="Nodes (select for details)", font=("Segoe UI", 10, "bold")).grid(
            row=2, column=0, sticky="w"
        )
        self._node_list = tk.Listbox(left, exportselection=False, height=16)
        self._node_list.grid(row=3, column=0, sticky="nsew", pady=(2, 0))
        self._node_list.bind("<<ListboxSelect>>", self._on_node_select)

        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)

        details_nb = ttk.Notebook(right)
        details_nb.grid(row=0, column=0, sticky="nsew")

        tab_stats = ttk.Frame(details_nb, padding=4)
        tab_details = ttk.Frame(details_nb, padding=4)
        tab_summary = ttk.Frame(details_nb, padding=4)
        tab_usage = ttk.Frame(details_nb, padding=4)
        details_nb.add(tab_stats, text="Tree Statistics")
        details_nb.add(tab_details, text="Node Details")
        details_nb.add(tab_summary, text="Model Summary")
        details_nb.add(tab_usage, text="Feature Usage")

        for tab in (tab_stats, tab_details, tab_summary, tab_usage):
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)

        self._tree_stats = tk.Text(tab_stats, height=9, wrap="word", state="disabled")
        self._tree_stats.grid(row=0, column=0, sticky="nsew")
        self._set_text(self._tree_stats, "Select a tree to view statistics.")

        self._details = tk.Text(tab_details, height=10, wrap="word", state="disabled")
        self._details.grid(row=0, column=0, sticky="nsew")

        self._summary = tk.Text(tab_summary, height=8, wrap="word", state="disabled")
        self._summary.grid(row=0, column=0, sticky="nsew")

        cols = ("feature", "occurrences", "avg_gain", "avg_cover", "first_tree", "max_depth")
        self._usage = ttk.Treeview(tab_usage, columns=cols, show="headings", height=10)
        headings = {
            "feature": "Feature",
            "occurrences": "Occ.",
            "avg_gain": "Avg gain",
            "avg_cover": "Avg cover",
            "first_tree": "First tree",
            "max_depth": "Max depth",
        }
        widths = {
            "feature": 120,
            "occurrences": 50,
            "avg_gain": 80,
            "avg_cover": 80,
            "first_tree": 70,
            "max_depth": 70,
        }
        for c in cols:
            self._usage.heading(c, text=headings[c])
            self._usage.column(c, width=widths[c], minwidth=40, stretch=(c == "feature"))
        self._usage.grid(row=0, column=0, sticky="nsew")

    # ------------------------------------------------------------------ actions
    def on_show(self) -> None:
        """Called by the Studio shell when this page becomes visible."""
        self._refresh_graphviz_status()
        self.refresh_registry_models()

    def refresh(self, *, lazy: bool = True) -> None:
        """Refresh registry list (same trigger pattern as other Model Builder panels)."""
        del lazy  # sync refresh is cheap (list + path resolve)
        self.refresh_registry_models()

    def _refresh_graphviz_status(self) -> None:
        self._gv_var.set(check_graphviz().message)

    def refresh_registry_models(self) -> None:
        """Repopulate the registry dropdown (filter: XGBoost + valid artifact)."""
        data_dir = self._data_dir()
        current = str(self._registry_var.get() or "").strip()
        try:
            models = list_xgboost_registry_models(data_dir) if data_dir else []
        except Exception as exc:
            self._registry_models = []
            self._suppress_registry_select = True
            try:
                self._registry_combo["values"] = ()
                self._registry_var.set("")
            finally:
                self._suppress_registry_select = False
            self._status_var.set(f"Registry refresh failed: {exc}")
            self._clear_registry_summary()
            return

        labels = registry_model_labels(models)
        self._registry_models = models
        self._suppress_registry_select = True
        try:
            self._registry_combo["values"] = labels
            if current and current in labels:
                self._registry_var.set(current)
            elif labels:
                # Preserve selection only; do not auto-load on refresh.
                if not current:
                    self._registry_var.set("")
            else:
                self._registry_var.set("")
        finally:
            self._suppress_registry_select = False

        if not labels:
            self._clear_registry_summary()
            if not self._booster:
                self._status_var.set(
                    "No XGBoost models with valid artifacts in the Model Registry. "
                    "Train one, or use Load External."
                )
        else:
            selected = str(self._registry_var.get() or "").strip()
            if selected:
                self._update_registry_summary(selected)
            else:
                self._clear_registry_summary()
            if not self._booster:
                self._status_var.set(
                    f"{len(labels)} XGBoost model(s) in registry — select one to load."
                )

    def _on_registry_selected(self, _event: Any = None) -> None:
        if self._suppress_registry_select:
            return
        name = str(self._registry_var.get() or "").strip()
        if name:
            self._update_registry_summary(name)
        else:
            self._clear_registry_summary()
        self._load_registry_model()

    def _clear_registry_summary(self) -> None:
        self._set_text(self._registry_summary, "")

    def _update_registry_summary(self, model_name: str) -> None:
        """Populate the strip from Model Registry metadata for ``model_name``."""
        name = str(model_name or "").strip()
        if not name:
            self._clear_registry_summary()
            return
        cached = next(
            (m for m in self._registry_models if m.get("model_name") == name),
            None,
        )
        try:
            view = load_registry_summary_for_model(
                self._data_dir(),
                name,
                cached_row=cached,
            )
            text = format_registry_summary_text(view)
        except Exception:
            text = ""
        self._set_text(self._registry_summary, text)

    def _load_registry_model(self) -> None:
        name = str(self._registry_var.get() or "").strip()
        if not name:
            self._clear_registry_summary()
            messagebox.showinfo(
                "Model Explorer",
                "Select a model from the Model Registry dropdown.",
                parent=self,
            )
            return
        self._update_registry_summary(name)
        data_dir = self._data_dir()
        # Prefer cached artifact_path from last refresh; re-resolve for safety.
        cached = next(
            (m for m in self._registry_models if m.get("model_name") == name),
            None,
        )
        path = str((cached or {}).get("artifact_path") or "").strip()
        if not path or not os.path.isfile(path):
            resolved = resolve_registry_xgboost_artifact(
                data_dir,
                name,
                row_algorithm=(cached or {}).get("algorithm"),
            )
            if not resolved.get("ok"):
                messagebox.showerror(
                    "Model Explorer",
                    str(resolved.get("error") or f"Could not resolve artifact for '{name}'."),
                    parent=self,
                )
                return
            path = str(resolved["artifact_path"])

        self._apply_loaded_booster(
            path,
            status_label=f"Registry: {name}",
            source_label=name,
        )

    def _browse(self) -> None:
        initial = ""
        if self.chart_dir:
            models = os.path.join(self.chart_dir, "data", "models")
            initial = models if os.path.isdir(models) else self.chart_dir
        path = filedialog.askopenfilename(
            parent=self,
            title="Load XGBoost model",
            initialdir=initial or None,
            filetypes=[
                ("XGBoost models", "*.pkl *.pickle *.json *.bst *.ubj *.model"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._path_var.set(path)

    def _load_external_model(self) -> None:
        """Backward-compatible direct file load (.pkl / .json / .bst / .ubj)."""
        path = self._path_var.get().strip()
        if not path:
            messagebox.showinfo("Model Explorer", "Choose a model file path first.", parent=self)
            return
        self._apply_loaded_booster(
            path,
            status_label=os.path.basename(path),
            source_label=os.path.basename(path),
        )

    # Alias kept for any callers / tests that used the old name.
    def _load_model(self) -> None:
        self._load_external_model()

    def _apply_loaded_booster(
        self,
        path: str,
        *,
        status_label: str,
        source_label: str,
    ) -> None:
        try:
            bst = load_booster(path)
        except LoadError as exc:
            messagebox.showerror("Model Explorer", str(exc), parent=self)
            return
        except Exception as exc:
            messagebox.showerror("Model Explorer", f"Load failed: {exc}", parent=self)
            return

        self._booster = bst
        self._source_path = path
        self._source_label = source_label
        n = tree_count(bst)
        self._status_var.set(f"Loaded {status_label} — {n} tree(s).")
        self._populate_summary()
        self._populate_feature_usage()
        self._tree_list.delete(0, tk.END)
        for i in range(n):
            self._tree_list.insert(tk.END, f"Tree {i}")
        if n > 0:
            self._tree_list.selection_set(0)
            self._tree_list.activate(0)
            self._on_tree_select()
        else:
            self._set_text(self._tree_stats, "Select a tree to view statistics.")
            self._set_text(self._details, "")
            self._node_list.delete(0, tk.END)
            self._nodes = []
            self._sync_tree_viewer()
            self._sync_node_viewer()

    def _selected_tree_id(self) -> int | None:
        sel = self._tree_list.curselection()
        if not sel:
            return None
        return int(sel[0])

    def _selected_node(self) -> NodeDetails | None:
        sel = self._node_list.curselection()
        if not sel:
            return None
        idx = int(sel[0])
        if idx < 0 or idx >= len(self._nodes):
            return None
        return self._nodes[idx]

    def _on_tree_select(self, _event: Any = None) -> None:
        if self._booster is None:
            return
        tid = self._selected_tree_id()
        if tid is None:
            return
        self._nodes = list_nodes_for_tree(self._booster, tid)
        self._node_list.delete(0, tk.END)
        for node in self._nodes:
            if node.is_leaf:
                label = f"Node {node.node_id}  leaf={node.leaf_value:.6g}" if node.leaf_value is not None else f"Node {node.node_id}  leaf"
            else:
                thr = f"{node.threshold:.6g}" if node.threshold is not None else "?"
                label = f"Node {node.node_id}  {node.feature}<{thr}"
            self._node_list.insert(tk.END, label)
        self._populate_tree_statistics(tid)
        self._set_text(self._details, "Select a node to view details.")
        self._sync_tree_viewer()
        self._sync_node_viewer()

    def _on_node_select(self, _event: Any = None) -> None:
        node = self._selected_node()
        if node is None:
            return
        self._show_node_details(node)
        self._sync_node_viewer()

    def _show_node_details(self, node: NodeDetails) -> None:
        self._set_text(self._details, format_node_details_text(node))

    def _populate_tree_statistics(self, tree_id: int) -> None:
        if self._booster is None:
            self._set_text(self._tree_stats, "Select a tree to view statistics.")
            return
        stats = build_tree_statistics(self._booster, tree_id)
        feats = ", ".join(stats.features_used) if stats.features_used else "—"
        root = stats.root_feature if stats.root_feature else "—"
        text = "\n".join([
            f"Tree Index: {stats.tree_index}",
            f"Number of Nodes: {stats.n_nodes}",
            f"Number of Leaves: {stats.n_leaves}",
            f"Maximum Depth: {stats.max_depth}",
            f"Root Feature: {root}",
            f"Total Gain: {stats.total_gain:.6g}",
            f"Features Used ({len(stats.features_used)}): {feats}",
        ])
        self._set_text(self._tree_stats, text)

    def _populate_summary(self) -> None:
        if self._booster is None:
            return
        s = build_model_summary(self._booster)
        lr = f"{s.learning_rate}" if s.learning_rate is not None else "—"
        feats = ", ".join(s.features_used) if s.features_used else "—"
        label = self._source_label or os.path.basename(self._source_path)
        text = "\n".join([
            f"Model: {label}",
            f"Source: {self._source_path}",
            f"Trees: {s.n_trees}",
            f"Max depth: {s.max_depth}",
            f"Objective: {s.objective or '—'}",
            f"Learning rate: {lr}",
            f"Total nodes: {s.total_nodes}",
            f"Leaves: {s.n_leaves}",
            f"Features used ({len(s.features_used)}): {feats}",
        ])
        self._set_text(self._summary, text)

    def _populate_feature_usage(self) -> None:
        for item in self._usage.get_children():
            self._usage.delete(item)
        if self._booster is None:
            return
        for row in build_feature_usage(self._booster):
            self._usage.insert(
                "",
                tk.END,
                values=(
                    row.feature,
                    row.occurrences,
                    f"{row.avg_gain:.6g}",
                    f"{row.avg_cover:.6g}",
                    row.first_tree,
                    row.max_depth,
                ),
            )

    # ------------------------------------------------------------------ Tree Viewer side window
    def _viewer_is_open(self) -> bool:
        win = self._viewer_win
        if win is None:
            return False
        try:
            return bool(win.winfo_exists())
        except tk.TclError:
            return False

    def _open_tree_viewer(self) -> None:
        """Open Graphviz Tree Viewer beside the main app (Feature Policy pattern)."""
        if self._viewer_is_open():
            assert self._viewer_win is not None
            self._sync_tree_viewer()
            self._viewer_win.lift()
            self._viewer_win.focus_force()
            return

        tid = self._selected_tree_id()
        win, panel = open_tree_viewer_window(
            self,
            title="XGBoost Tree Viewer",
            booster=self._booster,
            tree_id=tid,
        )
        self._viewer_win = win
        self._viewer_panel = panel
        win.protocol("WM_DELETE_WINDOW", self._on_tree_viewer_close)

    def _on_tree_viewer_close(self) -> None:
        win = self._viewer_win
        self._viewer_win = None
        self._viewer_panel = None
        if win is not None:
            try:
                win.destroy()
            except tk.TclError:
                pass

    def _sync_tree_viewer(self) -> None:
        """Update side-window content when the selected tree changes (if open)."""
        if not self._viewer_is_open() or self._viewer_panel is None:
            return
        if self._booster is None:
            self._viewer_panel.clear("Load a model and select a tree in Model Explorer.")
            return
        tid = self._selected_tree_id()
        if tid is None:
            self._viewer_panel.clear("Select a tree in Model Explorer.")
            return
        self._viewer_panel.show_tree(self._booster, tid)

    # ------------------------------------------------------------------ Node Viewer side window
    def _node_viewer_is_open(self) -> bool:
        win = self._node_viewer_win
        if win is None:
            return False
        try:
            return bool(win.winfo_exists())
        except tk.TclError:
            return False

    def _open_node_viewer(self) -> None:
        """Open Node Viewer beside the main app (Tree Viewer / Feature Policy pattern)."""
        node = self._selected_node()
        if node is None:
            msg = "Select a node in the Nodes list first."
            self._status_var.set(msg)
            messagebox.showinfo("Node Viewer", msg, parent=self)
            return

        if self._node_viewer_is_open():
            assert self._node_viewer_win is not None
            self._sync_node_viewer()
            self._node_viewer_win.lift()
            self._node_viewer_win.focus_force()
            return

        win, panel = open_node_viewer_window(
            self,
            title="XGBoost Node Viewer",
            node=node,
        )
        self._node_viewer_win = win
        self._node_viewer_panel = panel
        win.protocol("WM_DELETE_WINDOW", self._on_node_viewer_close)

    def _on_node_viewer_close(self) -> None:
        win = self._node_viewer_win
        self._node_viewer_win = None
        self._node_viewer_panel = None
        if win is not None:
            try:
                win.destroy()
            except tk.TclError:
                pass

    def _sync_node_viewer(self) -> None:
        """Update side-window content when the selected node changes (if open)."""
        if not self._node_viewer_is_open() or self._node_viewer_panel is None:
            return
        node = self._selected_node()
        if node is None:
            self._node_viewer_panel.clear("Select a node in Model Explorer.")
            return
        self._node_viewer_panel.show_node(node)

    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", tk.END)
        widget.insert("1.0", text)
        widget.configure(state="disabled")
