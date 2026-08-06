"""Node Viewer — selected-node details in a companion Toplevel beside the main app.

Follows the same open/place pattern as Tree Viewer / Feature Policy
(``open_tree_viewer_window`` + ``place_toplevel_beside_main``).
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ..fold_replay_widgets import place_toplevel_beside_main
from .tree_stats import NodeDetails


def format_node_details_text(node: NodeDetails) -> str:
    """Plain-text dump of a node for the details pane / Node Viewer."""
    lines = [
        f"ID: {node.id_label}",
        f"Tree: {node.tree_id}",
        f"Node: {node.node_id}",
        f"Depth: {node.depth}",
        f"Parent: {node.parent_id if node.parent_id is not None else '—'}",
        f"Left (yes): {node.left_id if node.left_id is not None else '—'}",
        f"Right (no): {node.right_id if node.right_id is not None else '—'}",
    ]
    if node.is_leaf:
        lines.extend([
            "Type: leaf",
            f"Leaf value: {node.leaf_value}",
            f"Cover: {node.cover}",
        ])
    else:
        lines.extend([
            "Type: split",
            f"Feature: {node.feature}",
            f"Threshold: {node.threshold}",
            f"Gain: {node.gain}",
            f"Cover: {node.cover}",
            f"Missing direction: {node.missing_direction or '—'}",
        ])
    return "\n".join(lines)


class NodeViewerPanel(ttk.Frame):
    """Read-only node details text in a side window."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self._title_var = tk.StringVar(value="Node Viewer")
        ttk.Label(bar, textvariable=self._title_var, font=("Segoe UI", 10, "bold")).pack(
            side="left"
        )

        self._text = tk.Text(self, wrap="word", state="disabled", font=("Consolas", 10))
        self._text.grid(row=1, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(self, orient="vertical", command=self._text.yview)
        yscroll.grid(row=1, column=1, sticky="ns")
        self._text.configure(yscrollcommand=yscroll.set)

        self.clear("Select a node in Model Explorer, then open Node Viewer.")

    def show_node(self, node: NodeDetails) -> None:
        """Display details for ``node``."""
        self._title_var.set(f"Node Viewer — {node.id_label}")
        self._set_text(format_node_details_text(node))

    def clear(self, message: str) -> None:
        self._title_var.set("Node Viewer")
        self._set_text(message)

    def _set_text(self, text: str) -> None:
        self._text.configure(state="normal")
        self._text.delete("1.0", tk.END)
        self._text.insert("1.0", text)
        self._text.configure(state="disabled")


def open_node_viewer_window(
    master: tk.Misc,
    *,
    title: str = "XGBoost Node Viewer",
    node: NodeDetails | None = None,
) -> tuple[tk.Toplevel, NodeViewerPanel]:
    """Open a companion window beside the main app with node details.

    Same placement pattern as ``open_tree_viewer_window`` /
    ``open_feature_policy_window``.
    """
    win = tk.Toplevel(master)
    win.title(title)
    win.transient(master.winfo_toplevel())
    panel = NodeViewerPanel(win)
    panel.pack(fill="both", expand=True, padx=8, pady=8)
    if node is not None:
        panel.show_node(node)
    else:
        panel.clear("Select a node in Model Explorer.")
    win.update_idletasks()
    # Compact side window — node details are text-only
    try:
        win.geometry("420x480")
    except tk.TclError:
        pass
    place_toplevel_beside_main(win, master)
    win.lift()
    win.focus_force()
    return win, panel
