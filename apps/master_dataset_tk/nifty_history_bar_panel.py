"""Nifty History Bar — angel_historic_bars.db coverage + fetch (Neo historic)."""

from __future__ import annotations

import os
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import ttk
from typing import Any

from .ui_util import open_path

_NIFTY_TOKEN = "99926000"
_FETCH_GAP_SEC = 0.2  # 200ms between interval fetches
_HISTORIC_INTERVAL_ORDER: tuple[tuple[str, int], ...] = (
    ("1m", 60),
    ("3m", 180),
    ("5m", 300),
    ("15m", 900),
    ("30m", 1800),
    ("60m", 3600),
    ("1d", 86400),
)
# Intervals updated by "Fetch all up to date" (forward-fill to today).
# Dedicated "3m hist" / "15m history" buttons still extend *older* history backward.
_UP_TO_DATE_INTERVALS: tuple[tuple[str, int], ...] = (
    ("1m", 60),
    ("3m", 180),
    ("5m", 300),
    ("15m", 900),
    ("30m", 1800),
    ("60m", 3600),
    ("1d", 86400),
)
# (button label, interval_sec, short name, months per click)
_HIST_BUTTONS: tuple[tuple[str, int, str, int], ...] = (
    ("3m hist", 180, "3m", 3),
    ("15m history", 900, "15m", 12),  # 1 year per click
)


def _fmt_count(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "—"


def _fmt_ohlc(v: Any) -> str:
    try:
        return f"{float(v):.2f}"
    except (TypeError, ValueError):
        return "—"


class NiftyHistoryBarPanel(ttk.Frame):
    """Settings → Nifty History Bar: angel_historic_bars.db (Neo historic candles)."""

    def __init__(self, master: tk.Misc, *, chart_dir: str = "") -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._status_var = tk.StringVar(value="Open this page to load historic DB coverage.")
        self._db_path_var = tk.StringVar(value="")
        self._refresh_busy = False
        self._refresh_again = False
        self._fetch_busy = False
        self._applying = False
        self._selected_interval_sec = 60
        self._fetch_btns: list[ttk.Button] = []
        self._build_ui()

    def set_chart_dir(self, chart_dir: str) -> None:
        self.chart_dir = chart_dir

    def on_show(self) -> None:
        self.refresh(force=True)

    def refresh(self, *, force: bool = False) -> None:
        self._schedule_refresh(force=force)

    def _build_ui(self) -> None:
        wrap = ttk.Frame(self, padding=16)
        wrap.pack(fill="both", expand=True)

        ttk.Label(wrap, text="Nifty History Bar", font=("Segoe UI", 14, "bold")).pack(
            anchor="w", pady=(0, 4)
        )
        ttk.Label(
            wrap,
            text=(
                "Historic candles (Angel → DB) — token 99926000 · angel_historic_bars.db. "
                "Fetch all up to date: 1m/3m/5m/15m/30m/60m/1d → today. "
                "3m hist: ~3 months older history per click. "
                "15m history: ~1 year older history per click "
                "(first = latest older window, again = next older window)."
            ),
            foreground="#888",
            wraplength=900,
        ).pack(anchor="w", pady=(0, 12))

        path_fr = ttk.LabelFrame(wrap, text="Database", padding=8)
        path_fr.pack(fill="x", pady=(0, 10))
        ttk.Label(path_fr, textvariable=self._db_path_var, font=("Consolas", 9)).pack(
            side="left", fill="x", expand=True
        )
        self._fetch_btns = []
        for btn_text, sec, _label, months in reversed(_HIST_BUTTONS):
            btn = ttk.Button(
                path_fr,
                text=btn_text,
                command=lambda s=sec, t=btn_text, m=months: self._start_hist_months(s, t, m),
            )
            btn.pack(side="right", padx=(8, 0))
            self._fetch_btns.append(btn)
        fetch_all = ttk.Button(
            path_fr,
            text="Fetch all up to date",
            command=self._start_fetch_all,
        )
        fetch_all.pack(side="right", padx=(8, 0))
        self._fetch_btns.append(fetch_all)
        ttk.Button(path_fr, text="Open folder", command=self._open_db_folder).pack(
            side="right", padx=(8, 0)
        )
        ttk.Button(path_fr, text="Refresh", command=lambda: self.refresh(force=True)).pack(
            side="right"
        )

        ttk.Label(wrap, textvariable=self._status_var, foreground="#666").pack(
            anchor="w", pady=(0, 8)
        )

        cov = ttk.LabelFrame(wrap, text="DB coverage", padding=8)
        cov.pack(fill="x", pady=(0, 10))
        cols = ("interval", "bars", "oldest", "newest", "status")
        self._cov_tree = ttk.Treeview(cov, columns=cols, show="headings", height=8)
        headings = {
            "interval": ("Interval", 80),
            "bars": ("Bars", 100),
            "oldest": ("Oldest", 160),
            "newest": ("Newest", 160),
            "status": ("Status", 120),
        }
        for c, (label, width) in headings.items():
            self._cov_tree.heading(c, text=label)
            self._cov_tree.column(c, width=width, anchor="w")
        self._cov_tree.pack(side="left", fill="x", expand=True)
        sb = ttk.Scrollbar(cov, orient="vertical", command=self._cov_tree.yview)
        sb.pack(side="right", fill="y")
        self._cov_tree.configure(yscrollcommand=sb.set)
        self._cov_tree.tag_configure("ok", foreground="#2E7D32")
        self._cov_tree.tag_configure("missing", foreground="#B45309")
        self._cov_tree.bind("<<TreeviewSelect>>", self._on_interval_selected)

        sample = ttk.LabelFrame(wrap, text="Recent candles (selected interval)", padding=8)
        sample.pack(fill="both", expand=True)
        scols = ("time", "open", "high", "low", "close", "volume")
        self._bar_tree = ttk.Treeview(sample, columns=scols, show="headings", height=14)
        for c, label, width in (
            ("time", "Time IST", 160),
            ("open", "Open", 90),
            ("high", "High", 90),
            ("low", "Low", 90),
            ("close", "Close", 90),
            ("volume", "Volume", 90),
        ):
            self._bar_tree.heading(c, text=label)
            self._bar_tree.column(c, width=width, anchor="w")
        self._bar_tree.pack(side="left", fill="both", expand=True)
        sb2 = ttk.Scrollbar(sample, orient="vertical", command=self._bar_tree.yview)
        sb2.pack(side="right", fill="y")
        self._bar_tree.configure(yscrollcommand=sb2.set)

    def _store(self):
        from storage.angel_historic_store import AngelHistoricStore, default_db_path

        chart_dir = str(self.chart_dir or "").strip() or None
        return AngelHistoricStore(default_db_path(chart_dir))

    def _set_fetch_enabled(self, enabled: bool) -> None:
        for btn in self._fetch_btns:
            try:
                btn.state(["!disabled"] if enabled else ["disabled"])
            except tk.TclError:
                pass

    def _open_db_folder(self) -> None:
        path = str(self._db_path_var.get() or "").strip()
        if path and os.path.isfile(path):
            path = os.path.dirname(path)
        if not path:
            try:
                from storage.angel_historic_store import default_db_path

                path = os.path.dirname(default_db_path(self.chart_dir or None))
            except Exception:
                path = os.path.join(str(self.chart_dir or ""), "data")
        open_path(path)

    def _schedule_refresh(self, *, force: bool = False) -> None:
        if self._refresh_busy:
            self._refresh_again = True
            return
        self._refresh_busy = True
        if not self._fetch_busy:
            self._status_var.set("Refreshing historic DB…")

        def _worker() -> None:
            payload = self._load_payload()

            def _apply() -> None:
                self._refresh_busy = False
                self._apply_payload(payload)
                if self._refresh_again:
                    self._refresh_again = False
                    self._schedule_refresh(force=True)

            try:
                self.after(0, _apply)
            except tk.TclError:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def _load_payload(self) -> dict[str, Any]:
        try:
            store = self._store()
            availability = store.availability(_NIFTY_TOKEN, months=6)
            bars = store.fetch_bars(
                _NIFTY_TOKEN,
                int(self._selected_interval_sec),
                limit=80,
            )
            return {"ok": True, "availability": availability, "bars": bars}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _apply_payload(self, payload: dict[str, Any]) -> None:
        self._applying = True
        try:
            for item in self._cov_tree.get_children():
                self._cov_tree.delete(item)
            for item in self._bar_tree.get_children():
                self._bar_tree.delete(item)

            if not payload.get("ok"):
                self._status_var.set(str(payload.get("error") or "Failed to load historic DB."))
                return

            avail = payload.get("availability") or {}
            db_path = str(avail.get("db_path") or "")
            self._db_path_var.set(db_path)
            db_mb = avail.get("db_size_mb", "—")
            if not self._fetch_busy:
                self._status_var.set(
                    f"DB {db_path} · {db_mb} MB · "
                    f"updated {datetime.now().strftime('%H:%M:%S')}"
                )

            intervals = avail.get("intervals") or {}
            for label, sec in _HISTORIC_INTERVAL_ORDER:
                row = intervals.get(label) if isinstance(intervals, dict) else None
                if not isinstance(row, dict):
                    row = {}
                stored = int(row.get("stored_bars") or row.get("stored_total") or 0)
                oldest = row.get("oldest") or "—"
                newest = row.get("newest") or "—"
                if stored == 0:
                    status = "empty"
                    tag = "missing"
                else:
                    status = "OK"
                    tag = "ok"
                self._cov_tree.insert(
                    "",
                    "end",
                    iid=str(sec),
                    values=(
                        label,
                        _fmt_count(stored),
                        oldest,
                        newest,
                        status,
                    ),
                    tags=(tag,),
                )

            iid = str(self._selected_interval_sec)
            if self._cov_tree.exists(iid):
                self._cov_tree.selection_set(iid)
                self._cov_tree.see(iid)
            self._fill_sample_bars(payload.get("bars") or [])
        finally:
            self._applying = False

    def _fill_sample_bars(self, bars: list[dict[str, Any]]) -> None:
        from zoneinfo import ZoneInfo

        ist = ZoneInfo("Asia/Kolkata")
        for item in self._bar_tree.get_children():
            self._bar_tree.delete(item)
        for bar in reversed(list(bars)):
            ts = bar.get("bucket_start")
            if ts is None:
                ts = bar.get("time")
            time_txt = "—"
            if ts is not None:
                try:
                    time_txt = datetime.fromtimestamp(float(ts), tz=ist).strftime(
                        "%Y-%m-%d %H:%M"
                    )
                except (TypeError, ValueError, OSError, OverflowError):
                    time_txt = "—"
            self._bar_tree.insert(
                "",
                "end",
                values=(
                    time_txt,
                    _fmt_ohlc(bar.get("open")),
                    _fmt_ohlc(bar.get("high")),
                    _fmt_ohlc(bar.get("low")),
                    _fmt_ohlc(bar.get("close")),
                    _fmt_count(bar.get("volume")),
                ),
            )

    def _on_interval_selected(self, _event: tk.Event | None = None) -> None:
        if self._applying or self._fetch_busy:
            return
        sel = self._cov_tree.selection()
        if not sel:
            return
        try:
            interval_sec = int(sel[0])
        except (TypeError, ValueError):
            return
        if interval_sec == self._selected_interval_sec:
            return
        self._selected_interval_sec = interval_sec
        self._schedule_refresh(force=True)

    def _start_fetch_all(self) -> None:
        if self._fetch_busy:
            return
        self._fetch_busy = True
        self._set_fetch_enabled(False)
        self._status_var.set("Fetching all intervals up to today…")

        def _worker() -> None:
            summary = self._fetch_all_intervals()

            def _done() -> None:
                self._fetch_busy = False
                self._set_fetch_enabled(True)
                self._status_var.set(summary)
                self._schedule_refresh(force=True)

            try:
                self.after(0, _done)
            except tk.TclError:
                self._fetch_busy = False

        threading.Thread(target=_worker, daemon=True).start()

    def _fetch_all_intervals(self) -> str:
        from storage.angel_historic_fetch import (
            fetch_historic_range,
            is_newest_up_to_date,
        )

        store = self._store()
        lines: list[str] = []
        max_rounds = 25

        for idx, (label, sec) in enumerate(_UP_TO_DATE_INTERVALS):
            before = store.get_bounds(_NIFTY_TOKEN, sec)
            total_added = 0
            rounds = 0
            last_error: str | None = None
            newest_before = before.get("newest_ts")

            while rounds < max_rounds:
                bounds = store.get_bounds(_NIFTY_TOKEN, sec)
                if is_newest_up_to_date(bounds.get("newest_ts"), interval_sec=sec):
                    break

                try:
                    self.after(
                        0,
                        lambda L=label, R=rounds + 1: self._status_var.set(
                            f"Fetching {L} up to date (batch {R})…"
                        ),
                    )
                except tk.TclError:
                    pass

                mode = "extend_after" if int(bounds.get("bar_count") or 0) > 0 else "initial"
                result = fetch_historic_range(
                    _NIFTY_TOKEN,
                    sec,
                    months=6,
                    mode=mode,
                    store=store,
                )
                rounds += 1

                if result.get("already_current"):
                    break

                if result.get("error") and not result.get("bars"):
                    last_error = str(result.get("error"))
                    break

                stored = 0
                if result.get("bars"):
                    stored = store.upsert_bars(
                        _NIFTY_TOKEN,
                        str(result.get("exchange") or "NSE"),
                        sec,
                        result["bars"],
                    )

                after = store.get_bounds(_NIFTY_TOKEN, sec)
                newest_after = after.get("newest_ts")
                if (
                    newest_after == newest_before
                    and stored == 0
                    and not result.get("more_batches")
                ):
                    break
                if newest_after is not None:
                    newest_before = newest_after

                if is_newest_up_to_date(after.get("newest_ts"), interval_sec=sec):
                    break
                if not result.get("more_batches") and not result.get("bars"):
                    break

                time.sleep(_FETCH_GAP_SEC)

            after = store.get_bounds(_NIFTY_TOKEN, sec)
            total_added = int(after.get("bar_count") or 0) - int(before.get("bar_count") or 0)
            newest_txt = after.get("newest_time") or "—"
            if last_error:
                lines.append(f"{label}: fail ({last_error})")
            elif is_newest_up_to_date(after.get("newest_ts"), interval_sec=sec):
                lines.append(f"{label}: OK newest {newest_txt} (+{total_added})")
            else:
                lines.append(
                    f"{label}: partial newest {newest_txt} (+{total_added}, {rounds} batches)"
                )

            if idx < len(_UP_TO_DATE_INTERVALS) - 1:
                time.sleep(_FETCH_GAP_SEC)

        return "Up to date · " + " · ".join(lines)

    def _start_hist_months(
        self, interval_sec: int, button_label: str, months: int
    ) -> None:
        if self._fetch_busy:
            return
        self._fetch_busy = True
        self._set_fetch_enabled(False)
        self._selected_interval_sec = int(interval_sec)
        span = "1 year" if int(months) >= 12 else f"~{int(months)} months"
        self._status_var.set(f"{button_label}: fetching {span}…")

        def _worker() -> None:
            summary = self._fetch_hist_months_batch(
                int(interval_sec), button_label, int(months)
            )

            def _done() -> None:
                self._fetch_busy = False
                self._set_fetch_enabled(True)
                self._status_var.set(summary)
                self._schedule_refresh(force=True)

            try:
                self.after(0, _done)
            except tk.TclError:
                self._fetch_busy = False

        threading.Thread(target=_worker, daemon=True).start()

    def _fetch_hist_months_batch(
        self, interval_sec: int, button_label: str, months: int
    ) -> str:
        from storage.angel_historic_fetch import fetch_months_history_batch
        from storage.angel_historic_store import INTERVAL_LABEL

        store = self._store()
        label = INTERVAL_LABEL.get(interval_sec, f"{interval_sec}s")
        before = store.get_bounds(_NIFTY_TOKEN, interval_sec)
        mode_hint = "older" if int(before.get("bar_count") or 0) > 0 else "latest"
        span = "1y" if months >= 12 else f"{months}m"
        try:
            self.after(
                0,
                lambda: self._status_var.set(
                    f"{button_label}: {mode_hint} {span} for {label}…"
                ),
            )
        except tk.TclError:
            pass

        result = fetch_months_history_batch(
            _NIFTY_TOKEN,
            interval_sec,
            months=months,
            store=store,
        )
        stored = 0
        if result.get("bars"):
            stored = store.upsert_bars(
                _NIFTY_TOKEN,
                str(result.get("exchange") or "NSE"),
                interval_sec,
                result["bars"],
            )
        after = store.get_bounds(_NIFTY_TOKEN, interval_sec)
        added = int(after.get("bar_count") or 0) - int(before.get("bar_count") or 0)
        if result.get("error") and not result.get("ok") and stored == 0:
            return f"{button_label}: fail ({result.get('error')})"
        return (
            f"{button_label}: {mode_hint} {span} · "
            f"+{added} bars (stored {stored}) · "
            f"oldest {after.get('oldest_time') or '—'} → "
            f"newest {after.get('newest_time') or '—'} · "
            f"click again for next older {span}"
        )
