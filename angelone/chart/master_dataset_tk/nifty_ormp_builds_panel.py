"""NIFTY ORMP — Builds (create / list versioned immutable ORMP DBs)."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from .ormp_service import (
    candle_coverage_dates,
    format_size,
    list_ormp_builds,
    run_ormp_build,
)
from .ui_util import open_path


class NiftyOrmpBuildsPanel(ttk.Frame):
    """Historical Data → NIFTY ORMP → ORMP Builds."""

    def __init__(self, master: tk.Misc, *, chart_dir: str = "") -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._status_var = tk.StringVar(value="Ready.")
        self._progress_var = tk.StringVar(value="")
        self._busy = False
        self._band_var = tk.StringVar(value="0.05")
        self._price_var = tk.StringVar(value="close")
        self._path_var = tk.StringVar(value="snapshot")
        self._from_var = tk.StringVar(value="")
        self._to_var = tk.StringVar(value="")
        self._coverage_from: str | None = None
        self._coverage_to: str | None = None
        self._build_ui()

    def set_chart_dir(self, chart_dir: str) -> None:
        self.chart_dir = chart_dir
        self._apply_coverage_defaults(force=True)

    def on_show(self) -> None:
        self.refresh_list()

    def _apply_coverage_defaults(self, *, force: bool = False) -> None:
        """Fill From/To from 1m NIFTY historic coverage."""
        try:
            cov_from, cov_to = candle_coverage_dates(self.chart_dir)
        except Exception:  # noqa: BLE001
            cov_from, cov_to = None, None
        prev_from, prev_to = self._coverage_from, self._coverage_to
        self._coverage_from = cov_from
        self._coverage_to = cov_to
        if not cov_from or not cov_to:
            return
        cur_from = self._from_var.get().strip()
        cur_to = self._to_var.get().strip()
        # Preserve user edits; refresh when empty or still matching last coverage defaults.
        if force or not cur_from or cur_from == (prev_from or ""):
            self._from_var.set(cov_from)
        if force or not cur_to or cur_to == (prev_to or ""):
            self._to_var.set(cov_to)

    def _build_ui(self) -> None:
        wrap = ttk.Frame(self, padding=16)
        wrap.pack(fill="both", expand=True)

        ttk.Label(
            wrap,
            text=(
                "Create versioned immutable ORMP feature DBs from 1m NIFTY history. "
                "Existing builds are never overwritten. Dataset Builder will consume "
                "these artifacts only — no rebuild at dataset time."
            ),
            foreground="#888",
            wraplength=900,
        ).pack(anchor="w", pady=(0, 12))

        form = ttk.LabelFrame(wrap, text="New build", padding=10)
        form.pack(fill="x", pady=(0, 10))

        r1 = ttk.Frame(form)
        r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="Band size %", width=14).pack(side="left")
        ttk.Entry(r1, textvariable=self._band_var, width=10).pack(side="left")
        ttk.Label(r1, text="  Price source", width=14).pack(side="left")
        ttk.Combobox(
            r1,
            textvariable=self._price_var,
            values=["close", "hlc3", "ohlc4", "typical_price"],
            state="readonly",
            width=16,
        ).pack(side="left")
        ttk.Label(r1, text="  Path mode", width=12).pack(side="left")
        ttk.Combobox(
            r1,
            textvariable=self._path_var,
            values=["snapshot", "continuous"],
            state="readonly",
            width=14,
        ).pack(side="left")

        r2 = ttk.Frame(form)
        r2.pack(fill="x", pady=(6, 2))
        ttk.Label(r2, text="From date", width=14).pack(side="left")
        ttk.Entry(r2, textvariable=self._from_var, width=14).pack(side="left")
        ttk.Label(r2, text="  To date", width=10).pack(side="left")
        ttk.Entry(r2, textvariable=self._to_var, width=14).pack(side="left")
        ttk.Label(
            r2, text="  (defaults = historic coverage)", foreground="#888"
        ).pack(side="left", padx=(8, 0))

        r3 = ttk.Frame(form)
        r3.pack(fill="x", pady=(10, 0))
        self._build_btn = ttk.Button(r3, text="Build ORMP", command=self._start_build)
        self._build_btn.pack(side="left")
        ttk.Button(r3, text="Refresh list", command=self.refresh_list).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(r3, text="Open outputs", command=self._open_outputs).pack(
            side="left", padx=(8, 0)
        )

        ttk.Label(wrap, textvariable=self._progress_var, foreground="#58a6ff").pack(
            anchor="w", pady=(0, 4)
        )
        ttk.Label(wrap, textvariable=self._status_var, foreground="#888").pack(
            anchor="w", pady=(0, 8)
        )

        list_fr = ttk.LabelFrame(wrap, text="Existing builds", padding=8)
        list_fr.pack(fill="both", expand=True)

        cols = ("name", "params", "coverage", "rows", "size", "built")
        self._tree = ttk.Treeview(
            list_fr,
            columns=cols,
            show="headings",
            height=14,
        )
        headings = {
            "name": ("Build", 220),
            "params": ("Parameters", 200),
            "coverage": ("Coverage", 160),
            "rows": ("Rows / days", 120),
            "size": ("Size", 80),
            "built": ("Built", 120),
        }
        for c, (title, width) in headings.items():
            self._tree.heading(c, text=title)
            self._tree.column(c, width=width, stretch=True)
        scroll = ttk.Scrollbar(list_fr, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=scroll.set)
        self._tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self._tree.bind("<Double-1>", self._on_double_click)
        self._paths: dict[str, str] = {}

    def refresh_list(self) -> None:
        self._apply_coverage_defaults()
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._paths.clear()
        try:
            builds = list_ormp_builds(self.chart_dir)
        except Exception as exc:  # noqa: BLE001
            self._status_var.set(f"Failed to list builds: {exc}")
            return
        for b in builds:
            bs = f"{b.band_size_pct:g}%" if b.band_size_pct is not None else "?"
            params = f"{bs} · {b.price_source} · {b.path_mode}"
            cov = "—"
            if b.from_date or b.to_date:
                cov = f"{b.from_date or '?'} → {b.to_date or '?'}"
            rows = "—"
            if b.rows is not None or b.days is not None:
                r = f"{b.rows:,}" if b.rows is not None else "?"
                d = f"{b.days:,}" if b.days is not None else "?"
                rows = f"{r} / {d}"
            iid = self._tree.insert(
                "",
                "end",
                values=(
                    b.display_name,
                    params,
                    cov,
                    rows,
                    format_size(b.file_size_bytes),
                    b.built_at_label,
                ),
            )
            self._paths[iid] = b.path
        self._status_var.set(f"{len(builds)} build(s). Double-click to open folder.")

    def _start_build(self) -> None:
        if self._busy:
            return
        try:
            band = float(self._band_var.get().strip())
        except ValueError:
            messagebox.showerror("ORMP Build", "Band size % must be a number (e.g. 0.05).")
            return
        if band <= 0:
            messagebox.showerror("ORMP Build", "Band size % must be > 0.")
            return
        from_date = self._from_var.get().strip() or None
        to_date = self._to_var.get().strip() or None
        price = self._price_var.get().strip()
        path_mode = self._path_var.get().strip()

        self._busy = True
        self._build_btn.state(["disabled"])
        self._status_var.set("Building ORMP… this can take several minutes.")
        self._progress_var.set("Starting…")

        def worker() -> None:
            err: str | None = None
            result: dict[str, Any] | None = None

            def on_progress(event: str, payload: dict[str, Any]) -> None:
                if event != "day":
                    return
                msg = (
                    f"Day {payload.get('i')}/{payload.get('n')}  "
                    f"{payload.get('trading_day')}  "
                    f"rows={payload.get('rows_written')}  "
                    f"ok={payload.get('days_ok')} fail={payload.get('days_fail')}"
                )
                self.after(0, lambda m=msg: self._progress_var.set(m))

            try:
                result = run_ormp_build(
                    self.chart_dir,
                    band_size_pct=band,
                    price_source=price,
                    path_mode=path_mode,
                    from_date=from_date,
                    to_date=to_date,
                    on_progress=on_progress,
                )
            except Exception as exc:  # noqa: BLE001
                err = str(exc)

            def done() -> None:
                self._busy = False
                self._build_btn.state(["!disabled"])
                if err:
                    self._status_var.set(f"Build failed: {err}")
                    self._progress_var.set("")
                    messagebox.showerror("ORMP Build", err)
                else:
                    assert result is not None
                    out = (result.get("build") or {}).get("output_path") or ""
                    ok = result.get("ok")
                    elapsed = result.get("elapsed_sec")
                    self._progress_var.set("")
                    self._status_var.set(
                        f"{'OK' if ok else 'Completed with issues'} · "
                        f"{elapsed}s · {out}"
                    )
                    self.refresh_list()
                    if ok:
                        messagebox.showinfo(
                            "ORMP Build",
                            f"Build complete.\n\n{out}",
                        )
                    else:
                        messagebox.showwarning(
                            "ORMP Build",
                            f"Build finished with validation issues.\n\n{out}",
                        )

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _open_outputs(self) -> None:
        from .ormp_service import ormp_outputs_dir

        open_path(ormp_outputs_dir(self.chart_dir))

    def _on_double_click(self, _event: object = None) -> None:
        sel = self._tree.selection()
        if not sel:
            return
        path = self._paths.get(sel[0])
        if path:
            import os

            open_path(os.path.dirname(path))
