"""Tree Viewer — Graphviz canvas in a companion Toplevel beside the main app.

Follows the same open/place pattern as Feature Policy
(``open_feature_policy_window`` + ``place_toplevel_beside_main``).
"""

from __future__ import annotations

import os
import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from ..fold_replay_widgets import place_toplevel_beside_main
from .render import RenderError, check_graphviz, export_tree, render_tree_png


class TreeViewerPanel(ttk.Frame):
    """Graphviz tree canvas with zoom, pan, and export."""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self._booster: Any | None = None
        self._tree_id: int | None = None
        self._photo: Any | None = None
        self._png_path: str | None = None
        self._zoom: float = 1.0
        self._render_tmpdir = tempfile.mkdtemp(prefix="xgb_explorer_viewer_")

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        bar = ttk.Frame(self)
        bar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self._title_var = tk.StringVar(value="Tree Viewer (xgboost.to_graphviz)")
        ttk.Label(bar, textvariable=self._title_var, font=("Segoe UI", 10, "bold")).pack(
            side="left"
        )
        ttk.Button(bar, text="Zoom −", command=lambda: self._set_zoom(self._zoom / 1.2)).pack(
            side="right", padx=2
        )
        ttk.Button(bar, text="Zoom +", command=lambda: self._set_zoom(self._zoom * 1.2)).pack(
            side="right", padx=2
        )
        ttk.Button(bar, text="Reset", command=self._reset_view).pack(side="right", padx=2)
        ttk.Button(bar, text="Export PDF", command=lambda: self._export("pdf")).pack(
            side="right", padx=2
        )
        ttk.Button(bar, text="Export SVG", command=lambda: self._export("svg")).pack(
            side="right", padx=2
        )
        ttk.Button(bar, text="Export PNG", command=lambda: self._export("png")).pack(
            side="right", padx=2
        )

        canvas_wrap = ttk.Frame(self)
        canvas_wrap.grid(row=1, column=0, sticky="nsew")
        canvas_wrap.columnconfigure(0, weight=1)
        canvas_wrap.rowconfigure(0, weight=1)
        self._canvas = tk.Canvas(canvas_wrap, bg="#1e1e1e", highlightthickness=0)
        self._canvas.grid(row=0, column=0, sticky="nsew")
        yscroll = ttk.Scrollbar(canvas_wrap, orient="vertical", command=self._canvas.yview)
        xscroll = ttk.Scrollbar(canvas_wrap, orient="horizontal", command=self._canvas.xview)
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self._canvas.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self._canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self._canvas.bind("<B1-Motion>", self._on_pan_move)
        self._canvas.bind("<MouseWheel>", self._on_wheel_zoom)

        self.clear("Select a tree in Model Explorer, then open Tree Viewer.")

    def show_tree(self, booster: Any, tree_id: int) -> None:
        """Render ``tree_id`` from ``booster`` into the canvas."""
        self._booster = booster
        self._tree_id = int(tree_id)
        self._title_var.set(f"Tree Viewer — Tree {self._tree_id} (xgboost.to_graphviz)")
        self._zoom = 1.0
        self._render_tree(self._tree_id)

    def clear(self, message: str) -> None:
        self._booster = None
        self._tree_id = None
        self._canvas.delete("all")
        self._photo = None
        self._png_path = None
        self._title_var.set("Tree Viewer (xgboost.to_graphviz)")
        self._canvas.create_text(
            12,
            12,
            anchor="nw",
            fill="#cccccc",
            width=480,
            text=message,
            font=("Segoe UI", 10),
        )

    def _render_tree(self, tree_id: int) -> None:
        status = check_graphviz()
        if not status.available:
            self._clear_viewer(status.message)
            return
        assert self._booster is not None
        try:
            png = render_tree_png(self._booster, tree_id, out_dir=self._render_tmpdir)
        except RenderError as exc:
            self._clear_viewer(str(exc))
            return
        except Exception as exc:
            self._clear_viewer(f"Render failed: {exc}")
            return
        self._png_path = png
        self._draw_image()

    def _draw_image(self) -> None:
        self._canvas.delete("all")
        if not self._png_path or not os.path.isfile(self._png_path):
            return
        try:
            from PIL import Image, ImageTk
        except ImportError:
            try:
                self._photo = tk.PhotoImage(file=self._png_path)
            except tk.TclError as exc:
                self._clear_viewer(f"Cannot display PNG (install Pillow): {exc}")
                return
            self._canvas.create_image(0, 0, anchor="nw", image=self._photo)
            self._canvas.configure(scrollregion=self._canvas.bbox("all"))
            return

        img = Image.open(self._png_path)
        if self._zoom != 1.0:
            w = max(1, int(img.width * self._zoom))
            h = max(1, int(img.height * self._zoom))
            img = img.resize((w, h), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(img)
        self._canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self._canvas.configure(scrollregion=(0, 0, img.width, img.height))

    def _clear_viewer(self, message: str) -> None:
        self._canvas.delete("all")
        self._photo = None
        self._png_path = None
        self._canvas.create_text(
            12,
            12,
            anchor="nw",
            fill="#cccccc",
            width=480,
            text=message,
            font=("Segoe UI", 10),
        )

    def _set_zoom(self, zoom: float) -> None:
        self._zoom = max(0.2, min(5.0, float(zoom)))
        if self._png_path:
            self._draw_image()

    def _reset_view(self) -> None:
        self._zoom = 1.0
        self._canvas.xview_moveto(0)
        self._canvas.yview_moveto(0)
        if self._png_path:
            self._draw_image()

    def _on_pan_start(self, event: tk.Event) -> None:
        self._canvas.scan_mark(event.x, event.y)

    def _on_pan_move(self, event: tk.Event) -> None:
        self._canvas.scan_dragto(event.x, event.y, gain=1)

    def _on_wheel_zoom(self, event: tk.Event) -> None:
        if event.delta > 0:
            self._set_zoom(self._zoom * 1.1)
        else:
            self._set_zoom(self._zoom / 1.1)

    def _export(self, fmt: str) -> None:
        if self._booster is None or self._tree_id is None:
            messagebox.showinfo(
                "Tree Viewer",
                "Select a tree in Model Explorer first.",
                parent=self,
            )
            return
        status = check_graphviz()
        if not status.available:
            messagebox.showerror("Tree Viewer", status.message, parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            defaultextension=f".{fmt}",
            filetypes=[(fmt.upper(), f"*.{fmt}"), ("All files", "*.*")],
            initialfile=f"tree_{self._tree_id}.{fmt}",
        )
        if not path:
            return
        try:
            out = export_tree(self._booster, self._tree_id, path, fmt=fmt)
        except RenderError as exc:
            messagebox.showerror("Tree Viewer", str(exc), parent=self)
            return
        except Exception as exc:
            messagebox.showerror("Tree Viewer", f"Export failed: {exc}", parent=self)
            return
        messagebox.showinfo("Tree Viewer", f"Exported:\n{out}", parent=self)


def open_tree_viewer_window(
    master: tk.Misc,
    *,
    title: str = "XGBoost Tree Viewer",
    booster: Any | None = None,
    tree_id: int | None = None,
) -> tuple[tk.Toplevel, TreeViewerPanel]:
    """Open a companion window beside the main app with the Graphviz tree viewer.

    Same placement pattern as ``open_feature_policy_window``:
    ``Toplevel`` + ``transient`` + ``place_toplevel_beside_main`` + lift/focus.
    """
    win = tk.Toplevel(master)
    win.title(title)
    win.transient(master.winfo_toplevel())
    panel = TreeViewerPanel(win)
    panel.pack(fill="both", expand=True, padx=8, pady=8)
    if booster is not None and tree_id is not None:
        panel.show_tree(booster, tree_id)
    elif booster is None:
        panel.clear("Load a model and select a tree in Model Explorer.")
    else:
        panel.clear("Select a tree in Model Explorer.")
    win.update_idletasks()
    place_toplevel_beside_main(win, master)
    win.lift()
    win.focus_force()
    return win, panel
