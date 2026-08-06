"""Debug side panel — tick DB inventory and load probes beside the main app."""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Callable

from .fold_replay_widgets import place_toplevel_beside_main
from .inventory import filter_inventory_rows, load_chain_inventory_rows
from .market_db_service import (
    data_dir,
    format_size_bytes,
    format_tick_count,
    load_inventory,
    load_inventory_disk_only,
    probe_load_day_context,
)
from .ui_util import open_path


class DebugSidePanel:
    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        get_market: Callable[[], str] | None = None,
        get_interval_sec: Callable[[], int] | None = None,
    ) -> None:
        self._chart_dir = chart_dir
        self._get_market = get_market
        self._get_interval_sec = get_interval_sec
        self._payload: dict[str, Any] | None = None
        self._rows: list[dict[str, Any]] = []
        self._busy = False

        self._market_var = tk.StringVar(value="NIFTY")
        self._status_var = tk.StringVar(value="Open Debug — refresh to load tick DB inventory.")
        self._meta_var = tk.StringVar(value="")

        self._host = master
        self._build_ui(master)
        self._sync_market_filter()
        self.refresh(force=False)

    def set_chart_dir(self, chart_dir: str) -> None:
        self._chart_dir = chart_dir
        self.refresh(force=False)

    def _sync_market_filter(self) -> None:
        if self._get_market is not None:
            try:
                m = str(self._get_market() or "NIFTY").strip().upper()
                if m:
                    self._market_var.set(m)
            except Exception:
                pass

    def _interval_sec(self) -> int:
        if self._get_interval_sec is not None:
            try:
                return max(3, int(self._get_interval_sec()))
            except (TypeError, ValueError):
                pass
        return 10

    def _build_ui(self, parent: ttk.Frame) -> None:
        hdr = ttk.Frame(parent)
        hdr.pack(fill="x", pady=(0, 6))
        ttk.Label(
            hdr,
            text="Dataset Debug",
            font=("Segoe UI", 11, "bold"),
        ).pack(anchor="w")
        ttk.Label(
            hdr,
            text="Tick DB inventory, load probes, and DB folder access (same data as View DB).",
            foreground="#666",
            wraplength=520,
        ).pack(anchor="w", pady=(2, 0))

        ctrl = ttk.Frame(parent)
        ctrl.pack(fill="x", pady=(0, 4))
        ttk.Label(ctrl, text="Index").pack(side="left")
        market_cb = ttk.Combobox(
            ctrl,
            textvariable=self._market_var,
            values=["NIFTY", "BANKNIFTY", "SENSEX", "BOTH"],
            width=12,
            state="readonly",
        )
        market_cb.pack(side="left", padx=(6, 12))
        market_cb.bind("<<ComboboxSelected>>", lambda _e: self._apply_filter())
        ttk.Button(ctrl, text="Refresh", command=lambda: self.refresh(force=False)).pack(side="left", padx=2)
        ttk.Button(ctrl, text="Force scan", command=lambda: self.refresh(force=True)).pack(side="left", padx=2)

        meta_row = ttk.Frame(parent)
        meta_row.pack(fill="x", pady=(0, 4))
        ttk.Label(meta_row, textvariable=self._meta_var, foreground="#555").pack(side="left")

        table_frame = ttk.LabelFrame(parent, text="Chain sources", padding=4)
        table_frame.pack(fill="both", expand=True, pady=(0, 6))

        cols = ("day", "market", "expiry", "ticks", "db")
        self._tree = ttk.Treeview(table_frame, columns=cols, show="headings", height=16)
        for c, w, label in (
            ("day", 96, "Day"),
            ("market", 72, "Market"),
            ("expiry", 88, "Expiry"),
            ("ticks", 80, "Ticks"),
            ("db", 180, "DB file"),
        ):
            self._tree.heading(c, text=label)
            self._tree.column(c, width=w, anchor="w")
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=scroll.set)
        self._tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        self._tree.bind("<<TreeviewSelect>>", lambda _e: self._render_detail())

        btn_row = ttk.Frame(parent)
        btn_row.pack(fill="x", pady=(0, 6))
        ttk.Button(btn_row, text="Probe load_day_context", command=self._probe_load).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="Open DB folder", command=self._open_db_folder).pack(side="left")

        detail_frame = ttk.LabelFrame(parent, text="Details", padding=4)
        detail_frame.pack(fill="both", expand=True)
        self._detail = scrolledtext.ScrolledText(detail_frame, height=12, wrap="word", font=("Consolas", 9))
        self._detail.pack(fill="both", expand=True)
        self._detail.configure(state="disabled")

        ttk.Label(parent, textvariable=self._status_var, foreground="#444").pack(anchor="w", pady=(4, 0))

    def refresh(self, *, force: bool) -> None:
        if self._busy:
            return
        self._busy = True
        self._status_var.set("Scanning tick databases…" if force else "Loading inventory cache…")

        def worker() -> None:
            err: str | None = None
            payload: dict[str, Any] | None = None
            try:
                if force:
                    payload = load_inventory(self._chart_dir, force=True)
                else:
                    payload = load_inventory_disk_only(self._chart_dir)
                    if not payload:
                        payload = load_inventory(self._chart_dir, force=False)
            except Exception as exc:
                err = str(exc)
            self._host.after(0, lambda: self._on_refresh_done(payload, err))

        threading.Thread(target=worker, name="debug-panel-inventory", daemon=True).start()

    def _on_refresh_done(self, payload: dict[str, Any] | None, err: str | None) -> None:
        self._busy = False
        if err:
            self._status_var.set(f"Inventory failed: {err}")
            return
        self._payload = payload
        rows, meta = load_chain_inventory_rows(self._chart_dir)
        self._rows = rows
        updated = str((payload or {}).get("last_updated") or meta.get("last_updated") or "—")
        stale = " (stale cache)" if (payload or {}).get("stale") else ""
        self._meta_var.set(
            f"Last updated: {updated}{stale} · {len(rows)} chain row(s) in cache",
        )
        self._apply_filter()
        self._status_var.set("Inventory ready.")

    def _apply_filter(self) -> None:
        market = self._market_var.get()
        filtered = filter_inventory_rows(self._rows, market)
        self._tree.delete(*self._tree.get_children())
        for row in filtered:
            iid = str(row.get("source_id") or "")
            self._tree.insert(
                "",
                "end",
                iid=iid,
                values=(
                    row.get("trading_day") or row.get("date"),
                    row.get("market"),
                    row.get("expiry"),
                    format_tick_count(row.get("chain_ticks")),
                    row.get("db_file") or "—",
                ),
            )
        self._render_detail()

    def _selected_row(self) -> dict[str, Any] | None:
        sel = self._tree.selection()
        if not sel:
            return None
        iid = sel[0]
        for row in self._rows:
            if str(row.get("source_id") or "") == iid:
                return row
        return None

    def _render_detail(self) -> None:
        row = self._selected_row()
        lines: list[str] = []
        if not row:
            lines.append("Select a chain row to inspect DB metadata and run probes.")
        else:
            lines.extend([
                f"Trading day:  {row.get('trading_day')}",
                f"Market:       {row.get('market')}",
                f"Expiry:       {row.get('expiry')}",
                f"Chain ticks:  {format_tick_count(row.get('chain_ticks'))}",
                f"Spot OK:      {row.get('spot_available')}",
                f"DB file:      {row.get('db_file') or '—'}",
                f"DB path:      {row.get('db_path') or '—'}",
            ])
            day = str(row.get("trading_day") or "")
            for entry in (self._payload or {}).get("databases") or []:
                if str(entry.get("trading_day") or "") == day:
                    lines.append(f"DB size:      {format_size_bytes(entry.get('size_bytes'))}")
                    lines.append(f"DB ticks:     {format_tick_count(entry.get('total_ticks'))}")
                    if entry.get("error"):
                        lines.append(f"DB error:     {entry['error']}")
                    break
            lines.extend([
                "",
                "Probe load_day_context to time loading this day into memory",
                "(same path as Build stage 1 + feature prep).",
            ])
        self._detail.configure(state="normal")
        self._detail.delete("1.0", "end")
        self._detail.insert("1.0", "\n".join(lines))
        self._detail.configure(state="disabled")

    def _probe_load(self) -> None:
        row = self._selected_row()
        if not row:
            messagebox.showinfo("Probe", "Select a chain row first.")
            return
        if self._busy:
            return
        self._busy = True
        self._status_var.set("Probing load_day_context…")

        def worker() -> None:
            err: str | None = None
            result: dict[str, Any] | None = None
            try:
                result = probe_load_day_context(
                    self._chart_dir,
                    trading_day=str(row.get("trading_day") or ""),
                    market=str(row.get("market") or "NIFTY"),
                    expiry=str(row.get("expiry") or ""),
                    interval_sec=self._interval_sec(),
                )
            except Exception as exc:
                err = str(exc)
            self._host.after(0, lambda: self._on_probe_done(result, err))

        threading.Thread(target=worker, name="debug-panel-probe", daemon=True).start()

    def _on_probe_done(self, result: dict[str, Any] | None, err: str | None) -> None:
        self._busy = False
        if err:
            self._status_var.set(f"Probe failed: {err}")
            self._detail.configure(state="normal")
            self._detail.insert("end", f"\n\nProbe error:\n{err}")
            self._detail.configure(state="disabled")
            return
        assert result is not None
        lines = [
            "",
            "— Probe result —",
            f"Elapsed:      {result.get('elapsed_sec')}s",
            f"Source ticks: {format_tick_count(result.get('source_ticks'))}",
            f"Spot ticks:   {format_tick_count(result.get('spot_ticks'))}",
            f"Chain ticks:  {format_tick_count(result.get('chain_ticks'))}",
            f"Strikes:      {result.get('strikes')}",
            f"DB path:      {result.get('db_path')}",
        ]
        for ln in result.get("validation_lines") or []:
            lines.append(f"  {ln}")
        self._detail.configure(state="normal")
        self._detail.insert("end", "\n".join(lines))
        self._detail.configure(state="disabled")
        self._status_var.set(
            f"Probe done in {result.get('elapsed_sec')}s — "
            f"{format_tick_count(result.get('source_ticks'))} ticks loaded",
        )

    def _open_db_folder(self) -> None:
        row = self._selected_row()
        if row and row.get("db_path"):
            folder = os.path.dirname(os.path.join(data_dir(self._chart_dir), str(row["db_path"])))
            open_path(folder)
            return
        open_path(data_dir(self._chart_dir))


def open_debug_panel(
    master: tk.Misc,
    *,
    chart_dir: str,
    get_market: Callable[[], str] | None = None,
    get_interval_sec: Callable[[], int] | None = None,
) -> tk.Toplevel:
    """Open debug panel beside the main window."""
    win = tk.Toplevel(master)
    win.title("Dataset Debug")
    win.transient(master.winfo_toplevel())

    body = ttk.Frame(win, padding=10)
    body.pack(fill="both", expand=True)

    panel = DebugSidePanel(
        body,
        chart_dir=chart_dir,
        get_market=get_market,
        get_interval_sec=get_interval_sec,
    )
    win._debug_panel = panel  # type: ignore[attr-defined]

    win.protocol("WM_DELETE_WINDOW", win.destroy)
    win.update_idletasks()
    place_toplevel_beside_main(win, master)
    win.lift()
    win.focus_force()
    return win
