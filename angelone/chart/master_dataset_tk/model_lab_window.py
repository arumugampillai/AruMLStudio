"""Model Lab Toplevel window — Phase 1 Overview + Phase 2 Prediction Dataset."""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import Any, Callable

from .build_service import chart_data_dir
from .fold_replay_widgets import place_toplevel_beside_main
from .model_registry_widgets import ACCENT, COL_MUTED, COL_OK, COL_WARN, SECTION_FONT, ScrollableFrame, fmt_val
from .ui_state import get_ui_state_manager

_KPI_CARD_BG = "#F5F7FA"
_KPI_VALUE_FG = "#1a1a1a"
_KPI_LABEL_FG = "#666666"
_RESEARCH_SCOPE_BG = "#EEF2F7"
_SEEN_BADGE_BG = "#1565C0"
_UNSEEN_BADGE_BG = "#C62828"
_BOTH_BADGE_BG = "#2E7D32"
_BADGE_FG = "#FFFFFF"

_PURPOSE_CHOICES = (
    "General Research",
    "Gamma Research",
    "Expiry Investigation",
    "Production Candidate",
    "Other",
)

_EXPLORER_COLS = (
    "trading_day",
    "timestamp",
    "exit_at",
    "current_ltp",
    "predicted_future_ltp",
    "actual_future_ltp",
    "expected_move",
    "actual_move",
    "predicted_trend",
    "actual_trend",
    "direction_correct",
    "maximum_profit",
    "maximum_drawdown",
    "dd_before_target",
    "max_profit_at",
    "max_drawdown_at",
    "time_to_target",
    "target_reached",
    "target_reached_at",
    "time_to_max_profit",
    "time_to_max_drawdown",
    "time_to_dd_before_target",
    "prediction_error",
    "absolute_error",
    "premium_error_pct",
    "pred_prob_up_2pct_5m",
    "pred_prob_up_3pct_5m",
    "pred_prob_up_4pct_5m",
    "pred_prob_up_5pct_5m",
    "pred_prob_up_6pct_5m",
    "pred_prob_up_gt6pct_5m",
)

# Trends are stored in DB; keep virtual only as fallback if missing
_EXPLORER_VIRTUAL = frozenset()

_EXPLORER_HEADINGS = {
    "trading_day": "Day",
    "timestamp": "Entry Ts",
    "exit_at": "Exit Ts",
    "current_ltp": "Current",
    "predicted_future_ltp": "Predicted",
    "actual_future_ltp": "Actual",
    "expected_move": "Exp Move",
    "actual_move": "Act Move",
    "predicted_trend": "Pred Trend",
    "actual_trend": "Act Trend",
    "direction_correct": "Dir ✓",
    "maximum_profit": "Max Profit",
    "maximum_drawdown": "Max DD",
    "dd_before_target": "DD Before Target",
    "max_profit_at": "MFE Ts",
    "max_drawdown_at": "MAE Ts",
    "time_to_target": "T→Target (s)",
    "target_reached": "Hit ✓",
    "target_reached_at": "Target Ts",
    "time_to_max_profit": "T→Max Profit (s)",
    "time_to_max_drawdown": "T→Max DD (s)",
    "time_to_dd_before_target": "T→DD Before Target (s)",
    "prediction_error": "Pred Err",
    "absolute_error": "Abs Err",
    "premium_error_pct": "Prem Err%",
    "pred_prob_up_2pct_5m": "P(+2%)",
    "pred_prob_up_3pct_5m": "P(+3%)",
    "pred_prob_up_4pct_5m": "P(+4%)",
    "pred_prob_up_5pct_5m": "P(+5%)",
    "pred_prob_up_6pct_5m": "P(+6%)",
    "pred_prob_up_gt6pct_5m": "P(>6%)",
}


def tb_model_names_from_registry_rows(rows: list[dict[str, Any]]) -> list[str]:
    """Triple Barrier model names from ``list_trained_models`` rows.

    Uses the same ``resolve_model_registry_family`` classification as Model
    Registry (persisted ``label_strategy`` / ``label_strategy_id``, falling
    back to ``target == "label_id"``) so the TB scorer dropdown stays in
    sync with what Model Registry / Create Model actually produces.

    Preserves the incoming row order (expected newest-first per the global
    Model sort standard — see ``selection_lists.get_sorted_models``) rather
    than re-sorting alphabetically, so the newest Triple Barrier model stays
    at the top of the dropdown.
    """
    from chain_replay_ml.training.registry import resolve_model_registry_family

    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        if resolve_model_registry_family(row) != "triple_barrier":
            continue
        name = str(row.get("model_name") or "").strip()
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return names


def _ltp_trend(current: Any, future: Any) -> str:
    try:
        c = float(current)
        f = float(future)
    except (TypeError, ValueError):
        return "—"
    if f > c:
        return "▲ Up"
    if f < c:
        return "▼ Down"
    return "— Flat"


def _fmt_check(val: Any) -> str:
    if val is None or val == "":
        return "—"
    try:
        return "✓" if int(val) == 1 else "✗"
    except (TypeError, ValueError):
        return str(val)


def _fmt_time_metric(val: Any) -> str:
    """Format seconds metrics; -1 means target never reached."""
    if val is None or val == "":
        return "—"
    try:
        f = float(val)
    except (TypeError, ValueError):
        return str(val)
    if f < 0:
        return "miss"
    if f == int(f):
        return str(int(f))
    return f"{f:.1f}"

_DAY_TABLE_COLS = (
    "trading_day",
    "dataset_type",
    "dataset_rows",
    "pred_rows",
    "status",
    "time_taken",
    "note",
    "dashboard",
)

_OUTCOME_CHUNK_CHOICES = (1, 2, 4, 6, 8, 10, 12)

_STATUS_LABEL = {
    "waiting": "Waiting",
    "running": "▶ Running",
    "completed": "✅ Complete",
    "partial": "⚠ Partial",
    "failed": "Failed",
    "cancelled": "Cancelled",
    "skipped": "Skipped",
}

# Prediction Dataset Metadata tab (Phase 1.5/1.6) — coverage color/emoji.
# Stage list itself is NOT hardcoded here — the UI reads stage order, labels,
# status text, and notes straight from the metadata payload, which in turn is
# driven by the registry in chain_replay_ml.model_lab.prediction_metadata_stages.
# These two dicts mirror that module's COVERAGE_BUCKET_COLOR / _EMOJI exactly
# (kept as plain literals here so building this pane never needs to import the
# heavier chain_replay_ml package eagerly).
_PRED_META_COVERAGE_COLORS = {
    "full": "#2E7D32",   # 🟢 ~100%
    "high": "#B8860B",   # 🟡 partial, high (>=50%)
    "low": "#C77800",    # 🟠 partial, low (<50%)
    "empty": "#C62828",  # 🔴 0% / no data
}
_PRED_META_COVERAGE_EMOJI = {
    "full": "🟢",
    "high": "🟡",
    "low": "🟠",
    "empty": "🔴",
}


def _fmt_int(n: Any) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "—"


def _fmt_day_meta_cell(value: Any, *, ready: bool, blank: str = "…") -> str:
    """Placeholder for unloaded Trading Days metadata columns."""
    if not ready:
        return blank
    if value is None or value == "":
        return "—"
    return str(value)


def _day_row_meta_ready(row: dict) -> bool:
    if row.get("ui_meta_ready") is False:
        return False
    if row.get("ui_meta_ready") is True:
        return True
    if row.get("rows_expected") is not None:
        return True
    if int(row.get("row_count") or row.get("pred_rows") or 0) > 0:
        return True
    st = str(row.get("status") or "")
    return st not in ("", "waiting")


def _fmt_clock(iso: Any) -> str:
    """ISO timestamp → HH:MM local-ish display."""
    if not iso:
        return "—"
    s = str(iso)
    # Prefer embedded time portion
    if "T" in s:
        try:
            part = s.split("T", 1)[1]
            return part[:5]
        except Exception:
            pass
    if len(s) >= 16 and s[10] == " ":
        return s[11:16]
    return s[:16]


def _fmt_unix_time(val: Any) -> str:
    """Unix epoch seconds → HH:MM:SS (local)."""
    if val is None or val == "":
        return "—"
    try:
        from datetime import datetime

        ts = float(val)
        if ts > 1e12:  # ms accidentally stored
            ts = ts / 1000.0
        return datetime.fromtimestamp(ts).strftime("%H:%M:%S")
    except (TypeError, ValueError, OSError, OverflowError):
        return str(val)


def _fmt_sec(sec: Any) -> str:
    try:
        s = float(sec)
    except (TypeError, ValueError):
        return "—"
    if s < 0:
        return "—"
    m, rem = divmod(int(s), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {rem}s"
    if m:
        return f"{m}m {rem}s"
    return f"{rem}s"


def _parse_iso_ts(raw: Any) -> datetime | None:
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(s[:19], fmt)
        except ValueError:
            continue
    return None


def _fmt_day_duration(started_at: Any, finished_at: Any) -> str:
    """Human duration from build day timestamps, e.g. 50s / 1m 2s."""
    start = _parse_iso_ts(started_at)
    end = _parse_iso_ts(finished_at)
    if start is None or end is None:
        return "—"
    sec = (end - start).total_seconds()
    if sec < 0:
        return "—"
    return _fmt_sec(sec)


def _short_metric_blob(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, dict):
        keys = ("mae", "rmse", "r2", "directional_accuracy_pct", "composite_score", "mape")
        parts = []
        for k in keys:
            if k in value and value[k] is not None:
                parts.append(f"{k}={value[k]}")
        if parts:
            return " · ".join(parts)
        return json.dumps(value, ensure_ascii=False, default=str)[:240]
    return str(value)[:240]


def prompt_create_lab(
    parent: tk.Misc,
    *,
    model_name: str,
    default_lab_name: str,
) -> dict[str, str] | None:
    """Modal create dialog. Returns {lab_name, description, purpose} or None."""
    dlg = tk.Toplevel(parent)
    dlg.title("Research Lab")
    dlg.transient(parent.winfo_toplevel())
    dlg.grab_set()
    dlg.resizable(False, False)

    result: dict[str, str] | None = None

    frm = ttk.Frame(dlg, padding=16)
    frm.pack(fill="both", expand=True)

    ttk.Label(frm, text="Research Lab", font=SECTION_FONT).grid(row=0, column=0, columnspan=2, sticky="w")
    ttk.Label(frm, text="Parent Model", foreground=COL_MUTED).grid(row=1, column=0, sticky="w", pady=(12, 2))
    ttk.Label(frm, text=model_name, wraplength=420).grid(row=1, column=1, sticky="w", pady=(12, 2), padx=(12, 0))

    ttk.Label(frm, text="Lab Name", foreground=COL_MUTED).grid(row=2, column=0, sticky="nw", pady=(10, 2))
    name_var = tk.StringVar(value=default_lab_name)
    name_entry = ttk.Entry(frm, textvariable=name_var, width=48)
    name_entry.grid(row=2, column=1, sticky="ew", pady=(10, 2), padx=(12, 0))

    ttk.Label(frm, text="Description", foreground=COL_MUTED).grid(row=3, column=0, sticky="nw", pady=(10, 2))
    desc = scrolledtext.ScrolledText(frm, width=46, height=4, font=("Segoe UI", 9), wrap="word")
    desc.grid(row=3, column=1, sticky="ew", pady=(10, 2), padx=(12, 0))
    ttk.Label(frm, text="(Optional)", foreground=COL_MUTED).grid(row=4, column=1, sticky="w", padx=(12, 0))

    ttk.Label(frm, text="Purpose", foreground=COL_MUTED).grid(row=5, column=0, sticky="w", pady=(10, 2))
    purpose_var = tk.StringVar(value=_PURPOSE_CHOICES[0])
    purpose_cb = ttk.Combobox(
        frm, textvariable=purpose_var, values=_PURPOSE_CHOICES, state="readonly", width=45,
    )
    purpose_cb.grid(row=5, column=1, sticky="w", pady=(10, 2), padx=(12, 0))

    btns = ttk.Frame(frm)
    btns.grid(row=6, column=0, columnspan=2, sticky="e", pady=(18, 0))

    def _cancel() -> None:
        nonlocal result
        result = None
        dlg.destroy()

    def _create() -> None:
        nonlocal result
        name = name_var.get().strip()
        if not name:
            messagebox.showwarning("Research Lab", "Lab Name is required.", parent=dlg)
            return
        result = {
            "lab_name": name,
            "description": desc.get("1.0", "end").strip(),
            "purpose": purpose_var.get().strip() or _PURPOSE_CHOICES[0],
        }
        dlg.destroy()

    ttk.Button(btns, text="Cancel", command=_cancel).pack(side="right", padx=(8, 0))
    ttk.Button(btns, text="Create", command=_create).pack(side="right")

    dlg.protocol("WM_DELETE_WINDOW", _cancel)
    name_entry.focus_set()
    dlg.update_idletasks()
    dlg.wait_window()
    return result


class ModelLabWindow(tk.Toplevel):
    """Second-window Model Lab shell (research workspace for one frozen model)."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        model_name: str,
        detail_doc: dict[str, Any] | None = None,
        on_created: Callable[[], None] | None = None,
        initial_tab: str | None = None,
        defer_refresh: bool = False,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self.model_name = model_name
        self._detail_doc = detail_doc
        self._on_created = on_created
        self._initial_tab = (initial_tab or "").strip().lower() or None
        self._lab = None
        self._build_thread: threading.Thread | None = None
        self._open_thread: threading.Thread | None = None
        self._day_sync_thread: threading.Thread | None = None
        self._pred_manager = None
        self._pred_job_id: str | None = None
        self._progress: dict[str, Any] = {}
        self._explorer_order = "timestamp ASC"
        self._explorer_offset = 0
        self._explorer_page = 200
        self._explorer_where_sql = ""
        self._explorer_where_args: list[Any] = []
        self._explorer_filter_spec: dict[str, Any] | None = None
        self._explorer_filter_feature = ""
        self._explorer_filter_desc = tk.StringVar(value="")
        self._explorer_filter_count = tk.StringVar(value="")
        self._feature_analysis: dict[str, Any] | None = None
        self._feature_list_rows: list[dict[str, Any]] = []
        self._program_result: dict[str, Any] | None = None
        self._program_cohorts: list[dict[str, Any]] = []
        self._pred_refresh_gen = 0
        self._explorer_load_gen = 0
        self._explorer_total_cache: int | None = None
        self._outcome_busy = False
        self._mi_busy = False
        self._mi_thread: threading.Thread | None = None
        self._outcome_stop_log = threading.Event()
        self._day_inv: list[tuple[str, int, int, int]] = []
        self._registry_rows_by_day: dict[str, int] = {}
        self._parent_dataset: str | None = None
        self._row_limit = tk.StringVar(value="")
        self._build_workers = tk.StringVar(value="1")
        self._enrich_path = tk.BooleanVar(value=True)
        self._overwrite = tk.BooleanVar(value=False)
        self._pool_size = tk.StringVar(value="2")
        self._outcome_status = tk.StringVar(value="Ready")
        self._worker_vars = {
            w: tk.BooleanVar(value=(w == 1)) for w in _OUTCOME_CHUNK_CHOICES
        }
        # Local fall-back duration when DB started/finished timestamps are missing.
        self._day_local_timing: dict[str, tuple[str, str]] = {}
        self._research_filter_day: str | None = None
        self._research_confidence_filter = "disabled"
        self._research_conf_classifier = "target_hit"
        self._research_conf_prediction = "disabled"
        self._research_evaluation_set = "all"
        self._ui_state = get_ui_state_manager()

        self.title(f"Research Lab — {model_name}")
        self.transient(master.winfo_toplevel())

        self._status = tk.StringVar(value="")
        hdr = ttk.Frame(self, padding=(12, 10))
        hdr.pack(fill="x")
        ttk.Label(hdr, text="Research Lab", font=SECTION_FONT).pack(side="left")
        ttk.Label(hdr, textvariable=self._status, foreground=COL_MUTED).pack(side="right")

        # Status bar first (bottom dock), then notebook fills remaining space
        self._build_status_bar()

        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill="both", expand=True, padx=10, pady=(0, 4))
        self._notebook.bind("<<NotebookTabChanged>>", self._on_notebook_tab_changed)

        self._empty_tab = ttk.Frame(self._notebook, padding=16)
        self._overview_tab = ttk.Frame(self._notebook, padding=10)
        self._prediction_tab = ttk.Frame(self._notebook, padding=10)
        self._research_tab = ttk.Frame(self._notebook, padding=10)
        self._strategy_sim_tab = ttk.Frame(self._notebook, padding=10)
        self._feature_research_tab = ttk.Frame(self._notebook, padding=10)
        self._research_programs_tab = ttk.Frame(self._notebook, padding=10)
        self._model_improvement_tab = ttk.Frame(self._notebook, padding=10)
        self._confidence_tab = ttk.Frame(self._notebook, padding=10)
        self._rr_validation_tab = ttk.Frame(self._notebook, padding=10)
        self._strike_dashboard_tab = ttk.Frame(self._notebook, padding=10)

        self._build_empty_tab()
        self._build_overview_tab()
        self._build_prediction_tab()
        self._build_research_tab()
        self._build_strategy_sim_tab()
        self._build_strike_dashboard_tab()
        self._build_feature_research_tab()
        self._build_research_programs_tab()
        self._build_model_improvement_tab()
        self._build_confidence_tab()
        self._build_rr_validation_tab()

        if defer_refresh:
            self._reset_tabs(has_lab=False)
            self._status.set("Opening…")
            self._empty_status.set("Status\nOpening…")
            self._empty_detail.set(
                f"Model: {self.model_name}\n\nLoading research workspace…"
            )
            self._set_status_bar(
                status="Opening…",
                detail="Preparing Research Lab (this stays responsive)…",
                active=True,
            )
        else:
            self.refresh()
            self._apply_initial_tab()
        self.update_idletasks()
        place_toplevel_beside_main(self, master)
        self.lift()
        self.focus_force()

    def _begin_async_open(self, *, ensure_lab: bool = False) -> None:
        """Create lab (optional) + load tabs without freezing the main app."""
        if self._open_thread is not None and self._open_thread.is_alive():
            return
        self._status.set("Opening…")
        self._set_status_bar(
            status="Opening…",
            detail=(
                "Creating research lab…"
                if ensure_lab
                else "Loading research lab…"
            ),
            active=True,
        )

        def work() -> None:
            err: Exception | None = None
            created = False
            try:
                if ensure_lab:
                    created = ensure_default_research_lab(
                        self.chart_dir,
                        self.model_name,
                        detail_doc=self._detail_doc,
                    )
            except Exception as exc:  # noqa: BLE001
                err = exc

            def done() -> None:
                if err is not None:
                    self._set_status_bar(status="Error", detail=str(err), active=False)
                    messagebox.showerror("Research", str(err), parent=self)
                    self.refresh()
                    return
                if created and self._on_created:
                    try:
                        self._on_created()
                    except Exception:
                        pass
                self.refresh_async()

            try:
                self.after(0, done)
            except tk.TclError:
                pass

        self._open_thread = threading.Thread(target=work, daemon=True, name="lab-open")
        self._open_thread.start()

    def refresh_async(self) -> None:
        """Load lab metadata on a worker, then paint tabs on the UI thread."""
        self._set_status_bar(
            status="Loading…",
            detail="Loading Research Lab…",
            active=True,
        )

        def work() -> None:
            from chain_replay_ml.model_lab import find_latest_lab

            lab = None
            err: Exception | None = None
            try:
                lab = find_latest_lab(self.model_name)
            except Exception as exc:  # noqa: BLE001
                err = exc

            def done() -> None:
                if err is not None:
                    self._lab = None
                    self._reset_tabs(has_lab=False)
                    self._status.set("Status: Error")
                    self._empty_status.set("Status\nError")
                    self._empty_detail.set(f"Failed to open Research Lab:\n{err}")
                    self._set_status_bar(status="Error", detail=str(err), active=False)
                    return
                self._apply_lab_after_load(lab)

            try:
                self.after(0, done)
            except tk.TclError:
                pass

        threading.Thread(target=work, daemon=True, name="lab-refresh").start()

    def _apply_lab_after_load(self, lab: Any) -> None:
        self._lab = lab
        if self._lab is None:
            self._reset_tabs(has_lab=False)
            self._status.set("Status: Not Created")
            self._empty_status.set("Status\nNot Created")
            self._empty_detail.set(self._empty_state_detail())
            self.title(f"Research Lab — {self.model_name}")
            self._set_status_bar(status="Ready", detail="No research lab yet", active=False)
            return

        self._reset_tabs(has_lab=True)
        self._pred_meta_loaded = False
        self._status.set(
            f"{self._lab.status} · Phase {self._lab.phase} · v{self._lab.version}"
        )
        self.title(f"Research Lab — {self.model_name} · v{self._lab.version}")
        self._render_overview(self._lab)
        self._try_reattach_prediction_job()
        # Trading Days UI reads prediction_day_metadata only (no parent discovery).
        self._refresh_prediction_tab(defer_explorer=True)
        self._apply_initial_tab()
        # Heavier tabs later so Prediction Dataset paints first
        self.after(50, self._refresh_secondary_tabs)

    def _refresh_secondary_tabs(self) -> None:
        """Deferred; heavy scans wait until user opens those tabs (see notebook bind)."""
        return

    def _on_notebook_tab_changed(self, _event: object | None = None) -> None:
        if self._lab is None:
            return
        try:
            current = self._notebook.select()
            tab = self._notebook.nametowidget(current)
        except (tk.TclError, KeyError):
            return
        try:
            text = str(self._notebook.tab(current, "text") or "")
            if text:
                self._ui_state.set("model_lab.tab", text, widget=self)
        except (tk.TclError, KeyError):
            pass
        try:
            if tab is self._research_tab:
                self._refresh_research_dashboard()
            elif tab is self._strategy_sim_tab:
                self._refresh_strategy_sim_tab()
            elif tab is self._strike_dashboard_tab:
                self._refresh_strike_dashboard_tab()
            elif tab is self._feature_research_tab:
                self._refresh_feature_research_list()
            elif tab is self._research_programs_tab:
                self._refresh_research_programs_list()
            elif tab is self._confidence_tab:
                self._refresh_confidence_tab()
            elif tab is self._rr_validation_tab:
                self._refresh_rr_validation_tab()
            # Model Improvement is heavy — wait for explicit Start Analysis (no auto-run).
        except Exception as exc:  # noqa: BLE001
            self._set_status_bar(
                status="Ready",
                detail=f"Tab refresh: {exc}",
                active=False,
            )

    def select_prediction_tab(self) -> None:
        """Switch to Prediction Dataset when a lab is open."""
        if self._lab is None:
            return
        try:
            self._notebook.select(self._prediction_tab)
        except tk.TclError:
            pass

    def select_research_dashboard_tab(self) -> None:
        """Switch to Research Dashboard tab."""
        if self._lab is None:
            return
        try:
            self._notebook.select(self._research_tab)
        except tk.TclError:
            pass

    def select_strategy_sim_tab(self) -> None:
        """Switch to Strategy Simulator tab."""
        if self._lab is None:
            return
        try:
            self._notebook.select(self._strategy_sim_tab)
        except tk.TclError:
            pass

    def _clear_research_day_filter(self) -> None:
        self._research_filter_day = None
        self._refresh_research_dashboard()

    def _open_day_research_dashboard(self, trading_day: str) -> None:
        day = str(trading_day or "").strip()
        if not day or self._lab is None:
            return
        row = self._day_rows.get(day) or {}
        pred_n = int(row.get("pred_rows") or 0)
        if pred_n <= 0:
            messagebox.showinfo(
                "Research Dashboard",
                f"No predictions for {day} yet.\n\n"
                "Build predictions for this day first, then open the dashboard.",
                parent=self,
            )
            return
        self._research_filter_day = day
        self.select_research_dashboard_tab()
        self._refresh_research_dashboard()

    def _apply_initial_tab(self) -> None:
        if self._initial_tab in ("prediction", "prediction_dataset", "pred"):
            self.select_prediction_tab()
            self._initial_tab = None
        elif self._initial_tab in ("strategy", "strategy_simulator", "simulator"):
            self.select_strategy_sim_tab()
            self._initial_tab = None
        elif self._initial_tab in ("strike", "strike_dashboard", "strike_prediction"):
            self.select_strike_dashboard_tab()
            self._initial_tab = None

    def _data_dir(self) -> str:
        return chart_data_dir(self.chart_dir)

    def _build_status_bar(self) -> None:
        """Bottom status bar — status text + determinate percent while generating."""
        ttk.Separator(self, orient="horizontal").pack(fill="x", side="bottom")
        bar = ttk.Frame(self, padding=(8, 4))
        bar.pack(fill="x", side="bottom")

        self._bar_status = tk.StringVar(value="Ready")
        ttk.Label(bar, textvariable=self._bar_status, width=22).pack(side="left", padx=(0, 8))

        self._bar_progress = ttk.Progressbar(bar, length=160, mode="determinate", maximum=100)
        self._bar_progress.pack(side="left", padx=(0, 6))

        self._bar_percent = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self._bar_percent, width=5, font=("Consolas", 9)).pack(
            side="left", padx=(0, 8),
        )

        self._bar_detail = tk.StringVar(value="")
        ttk.Label(
            bar,
            textvariable=self._bar_detail,
            foreground=COL_MUTED,
            font=("Consolas", 9),
            anchor="w",
        ).pack(side="left", fill="x", expand=True)

    def _set_status_bar(
        self,
        *,
        status: str,
        percent: float | None = None,
        detail: str = "",
        active: bool = False,
    ) -> None:
        self._bar_status.set(status)
        self._bar_detail.set(detail)
        if not active or percent is None:
            self._bar_progress.configure(value=float(percent or 0.0) if percent is not None else 0.0)
            self._bar_percent.set("" if percent is None else f"{float(percent):.0f}%")
        else:
            pct = max(0.0, min(100.0, float(percent)))
            self._bar_progress.configure(value=pct)
            self._bar_percent.set(f"{pct:.0f}%")

    def _build_empty_tab(self) -> None:
        ttk.Label(self._empty_tab, text="Research Lab", font=SECTION_FONT).pack(anchor="w", pady=(8, 4))
        self._empty_status = tk.StringVar(value="Status\nNot Created")
        ttk.Label(
            self._empty_tab,
            textvariable=self._empty_status,
            foreground=COL_MUTED,
            justify="left",
        ).pack(anchor="w", pady=(0, 8))
        self._empty_detail = tk.StringVar(
            value="This model has no research workspace yet."
        )
        ttk.Label(
            self._empty_tab,
            textvariable=self._empty_detail,
            foreground=COL_MUTED,
            justify="left",
            wraplength=520,
        ).pack(anchor="w", pady=(0, 16))
        row = ttk.Frame(self._empty_tab)
        row.pack(anchor="w")
        ttk.Button(row, text="Start Research", command=self._create_lab).pack(side="left")
        ttk.Button(row, text="Refresh", command=self.refresh).pack(side="left", padx=8)

    def _build_overview_tab(self) -> None:
        toolbar = ttk.Frame(self._overview_tab)
        toolbar.pack(fill="x", pady=(0, 8))
        ttk.Button(toolbar, text="Refresh", command=self.refresh).pack(side="left")
        ttk.Button(toolbar, text="Create Another Lab Version", command=self._create_lab).pack(
            side="left", padx=8,
        )

        self._overview_body = scrolledtext.ScrolledText(
            self._overview_tab, font=("Consolas", 10), wrap="word", height=28,
        )
        self._overview_body.pack(fill="both", expand=True)

    def _build_strategy_sim_tab(self) -> None:
        """Sibling to Research Dashboard — trading outcomes from strategy_simulator."""
        from .model_lab_strategy_sim_panel import ModelLabStrategySimPanel

        self._strategy_sim_panel = ModelLabStrategySimPanel(
            self._strategy_sim_tab,
            chart_dir=self.chart_dir,
        )
        self._strategy_sim_panel.pack(fill="both", expand=True)

    def _refresh_strategy_sim_tab(self) -> None:
        panel = getattr(self, "_strategy_sim_panel", None)
        if panel is None:
            return
        lab_path = self._lab.db_path if self._lab is not None else None
        panel.refresh_for_lab(lab_db_path=lab_path, model_name=self.model_name)

    def _build_strike_dashboard_tab(self) -> None:
        """Per-strike prediction charts (LTP, confidence, error, gap, regression)."""
        from .strike_prediction_dashboard_panel import ModelLabStrikeDashboardPanel

        self._strike_dashboard_panel = ModelLabStrikeDashboardPanel(
            self._strike_dashboard_tab,
            chart_dir=self.chart_dir,
            on_status=lambda s: self._set_status_bar(status=str(s or ""), detail=""),
            get_day_filter=lambda: getattr(self, "_research_filter_day", None),
        )
        self._strike_dashboard_panel.pack(fill="both", expand=True)

    def _refresh_strike_dashboard_tab(self) -> None:
        panel = getattr(self, "_strike_dashboard_panel", None)
        if panel is None:
            return
        lab_path = self._lab.db_path if self._lab is not None else None
        panel.refresh_for_lab(
            lab_db_path=lab_path,
            model_name=self.model_name,
            trading_day=getattr(self, "_research_filter_day", None),
        )

    def select_strike_dashboard_tab(self) -> None:
        """Switch to Strike Dashboard tab."""
        if self._lab is None:
            return
        try:
            self._notebook.select(self._strike_dashboard_tab)
        except tk.TclError:
            pass

    def _build_research_tab(self) -> None:
        toolbar = ttk.Frame(self._research_tab)
        toolbar.pack(fill="x", pady=(0, 6))
        ttk.Button(toolbar, text="Refresh", command=self._refresh_research_dashboard).pack(
            side="left",
        )
        ttk.Button(
            toolbar,
            text="Rebuild Stats",
            command=lambda: self._refresh_research_dashboard(force=True),
        ).pack(side="left", padx=6)
        self._research_day_filter_var = tk.StringVar(value="")
        ttk.Label(
            toolbar,
            textvariable=self._research_day_filter_var,
            foreground=ACCENT,
        ).pack(side="left", padx=(12, 4))
        self._btn_research_show_all = ttk.Button(
            toolbar,
            text="Show all days",
            command=self._clear_research_day_filter,
        )
        self._research_note = tk.StringVar(value="")
        ttk.Label(
            toolbar,
            textvariable=self._research_note,
            foreground=COL_MUTED,
        ).pack(side="left", padx=12)

        conf_box = ttk.LabelFrame(self._research_tab, text="Confidence Filter", padding=8)
        conf_box.pack(fill="x", pady=(0, 6))

        conf_inner = ttk.Frame(conf_box)
        conf_inner.pack(fill="x")

        conf_left = ttk.Frame(conf_inner)
        conf_left.pack(side="left", fill="x", expand=True)

        # Evaluation Set badge — rightmost inside Confidence Filter (no description text)
        self._research_dataset_badge = tk.Label(
            conf_inner,
            text="SEEN + UNSEEN",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=6,
            bg=_BOTH_BADGE_BG,
            fg=_BADGE_FG,
        )
        self._research_dataset_badge.pack(side="right", padx=(12, 0), anchor="n")
        # Keep a no-op frame handle for older refresh paths that hide/show scope banner
        self._research_scope_frame = conf_box
        self._research_scope_detail = None

        row_cls = ttk.Frame(conf_left)
        row_cls.pack(fill="x", pady=(0, 4))
        ttk.Label(row_cls, text="Classifier:", foreground=COL_MUTED).pack(
            side="left", padx=(0, 6)
        )
        self._research_conf_classifier_var = tk.StringVar(value="Path Touch")
        self._research_conf_classifier_keys = {
            "Path Touch": "target_hit",
            "RR 1:1": "rr_1_1",
            "RR 2:3": "rr_2_3",
            "RR 1:2": "rr_1_2",
            "RR 1:3": "rr_1_3",
            "RR 1:4": "rr_1_4",
        }
        self._research_conf_classifier_labels = {
            v: k for k, v in self._research_conf_classifier_keys.items()
        }
        self._research_conf_classifier_cb = ttk.Combobox(
            row_cls,
            textvariable=self._research_conf_classifier_var,
            values=list(self._research_conf_classifier_keys.keys()),
            state="readonly",
            width=14,
        )
        self._research_conf_classifier_cb.pack(side="left", padx=(0, 16))
        self._research_conf_classifier_cb.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._on_research_confidence_filter_changed(),
        )

        ttk.Label(row_cls, text="Prediction:", foreground=COL_MUTED).pack(
            side="left", padx=(0, 6)
        )
        self._research_conf_prediction_var = tk.StringVar(
            value=getattr(self, "_research_conf_prediction", "disabled")
        )
        self._research_conf_radios: dict[str, ttk.Radiobutton] = {}
        for text, val in (
            ("Disabled", "disabled"),
            ("= 1", "1"),
            ("= 0", "0"),
        ):
            rb = ttk.Radiobutton(
                row_cls,
                text=text,
                value=val,
                variable=self._research_conf_prediction_var,
                command=self._on_research_confidence_filter_changed,
            )
            rb.pack(side="left", padx=(0, 10))
            self._research_conf_radios[val] = rb

        row_eval = ttk.Frame(conf_left)
        row_eval.pack(fill="x", pady=(2, 0))
        ttk.Label(row_eval, text="Evaluation Set:", foreground=COL_MUTED).pack(
            side="left", padx=(0, 6)
        )
        self._research_evaluation_set_var = tk.StringVar(
            value=getattr(self, "_research_evaluation_set", "all")
        )
        self._research_eval_radios: dict[str, ttk.Radiobutton] = {}
        for text, val in (
            ("Seen + Unseen", "all"),
            ("Seen Only", "seen"),
            ("Unseen Only", "unseen"),
        ):
            rb = ttk.Radiobutton(
                row_eval,
                text=text,
                value=val,
                variable=self._research_evaluation_set_var,
                command=self._on_research_evaluation_set_changed,
            )
            rb.pack(side="left", padx=(0, 12))
            self._research_eval_radios[val] = rb

        self._research_conf_status_var = tk.StringVar(value="")
        ttk.Label(
            conf_left,
            textvariable=self._research_conf_status_var,
            foreground=ACCENT,
        ).pack(anchor="w", pady=(6, 0))

        # Legacy alias used by older refresh paths
        self._research_conf_filter_var = self._research_conf_prediction_var

        # Sub-tabs: Overview dashboard + Confidence Filter Comparison
        self._research_sub_nb = ttk.Notebook(self._research_tab)
        self._research_sub_nb.pack(fill="both", expand=True)
        self._research_overview_pane = ttk.Frame(self._research_sub_nb, padding=0)
        self._research_compare_pane = ttk.Frame(self._research_sub_nb, padding=4)
        self._research_sub_nb.add(self._research_overview_pane, text="Overview")
        self._research_sub_nb.add(
            self._research_compare_pane, text="Confidence Filter Comparison"
        )
        self._research_sub_nb.bind(
            "<<NotebookTabChanged>>",
            lambda _e: self._on_research_subtab_changed(),
        )

        self._research_scroll = ScrollableFrame(self._research_overview_pane)
        self._research_scroll.pack(fill="both", expand=True)
        body = self._research_scroll.inner
        self._research_body = body

        # —— KPI strip ——
        kpi_wrap = ttk.Frame(body)
        kpi_wrap.pack(fill="x", pady=(0, 10))
        self._kpi_vars: dict[str, tk.StringVar] = {}
        self._kpi_frames: list[tk.Frame] = []
        for key, label in (
            ("target_hit_rate", "Path Touch Rate"),
            ("direction_accuracy", "Direction Accuracy"),
            ("mae", "MAE"),
            ("premium_rmse", "Premium RMSE"),
        ):
            card = tk.Frame(
                kpi_wrap,
                bg=_KPI_CARD_BG,
                highlightbackground="#D0D7DE",
                highlightthickness=1,
                padx=14,
                pady=10,
            )
            card.pack(side="left", fill="x", expand=True, padx=(0, 8))
            self._kpi_frames.append(card)
            tk.Label(
                card,
                text=label,
                bg=_KPI_CARD_BG,
                fg=_KPI_LABEL_FG,
                font=("Segoe UI", 9),
                anchor="w",
            ).pack(fill="x")
            var = tk.StringVar(value="—")
            self._kpi_vars[key] = var
            tk.Label(
                card,
                textvariable=var,
                bg=_KPI_CARD_BG,
                fg=_KPI_VALUE_FG,
                font=("Segoe UI", 16, "bold"),
                anchor="w",
            ).pack(fill="x", pady=(2, 0))

        # —— Row: Quality | Risk | Confidence Filter Effect (+35% width) ——
        row1 = ttk.Frame(body)
        row1.pack(fill="x", pady=(0, 8))
        row1.columnconfigure(0, weight=100)
        row1.columnconfigure(1, weight=100)
        row1.columnconfigure(2, weight=135)  # Confidence Filter Effect ~+35%
        self._tv_quality = self._make_metric_panel(
            row1, "Prediction Quality", height=8, grid_col=0
        )
        self._tv_risk = self._make_metric_panel(row1, "Risk", height=8, grid_col=1)
        self._build_confidence_filter_effect_panel(row1, grid_col=2)

        # —— Row: Error | Distribution ——
        row2 = ttk.Frame(body)
        row2.pack(fill="x", pady=(0, 8))
        self._tv_error = self._make_metric_panel(row2, "Error", height=7)
        self._tv_distribution = self._make_metric_panel(row2, "Prediction Distribution", height=7)

        # —— Premium bands ——
        self._tv_premium = self._make_table_panel(
            body,
            "Premium Band Performance",
            columns=(
                ("band", "Premium Band", 100),
                ("rows", "Rows", 70),
                ("hit", "Hit %", 70),
                ("dir", "Dir %", 70),
                ("mae", "MAE", 80),
                ("dd", "Avg DD", 80),
            ),
            height=6,
        )

        # —— Trading days ——
        self._tv_days = self._make_table_panel(
            body,
            "Trading Day Summary",
            columns=(
                ("day", "Day", 100),
                ("rows", "Rows", 70),
                ("hit", "Hit %", 70),
                ("dir", "Dir %", 70),
                ("mae", "MAE", 80),
                ("dd", "DD", 80),
            ),
            height=8,
        )
        ttk.Label(
            body,
            text="Deep per-feature laboratory is on the Feature Research tab.",
            foreground=COL_MUTED,
        ).pack(anchor="w", pady=(4, 8))

        self._build_confidence_filter_comparison_pane()

    def _build_confidence_filter_comparison_pane(self) -> None:
        """Side-by-side confidence filter impact on the current Evaluation Set."""
        pane = self._research_compare_pane
        hdr = ttk.Frame(pane)
        hdr.pack(fill="x", pady=(0, 6))
        self._compare_note_var = tk.StringVar(
            value=(
                "Compares every inferred confidence model on the same Evaluation Set "
                "(SQL only — no XGBoost). Baseline = None."
            )
        )
        ttk.Label(
            hdr,
            textvariable=self._compare_note_var,
            foreground=COL_MUTED,
            wraplength=900,
            justify="left",
        ).pack(side="left", fill="x", expand=True)
        ttk.Button(
            hdr,
            text="Refresh Comparison",
            command=self._refresh_confidence_filter_comparison,
        ).pack(side="right", padx=(8, 0))

        cols = (
            ("filter", "Filter", 100),
            ("rows_left", "Rows Left", 90),
            ("rows_removed", "Rows Removed", 100),
            ("hit", "Hit %", 75),
            ("avg_dd", "Avg DD", 80),
            ("avg_mfe", "Avg Max Profit", 105),
            ("profit_dd", "Profit/DD", 85),
            ("mae", "MAE", 70),
            ("prem_rmse", "Premium RMSE", 100),
            ("d_hit", "Δ Hit %", 80),
            ("d_dd", "Δ Avg DD", 85),
            ("d_profit", "Δ Profit", 85),
            ("d_pd", "Δ Profit/DD", 95),
        )
        wrap = ttk.Frame(pane)
        wrap.pack(fill="both", expand=True)
        ids = tuple(c[0] for c in cols)
        tv = ttk.Treeview(wrap, columns=ids, show="headings", height=14)
        for cid, heading, width in cols:
            tv.heading(cid, text=heading)
            tv.column(cid, width=width, minwidth=60, stretch=True, anchor="center")
        tv.column("filter", anchor="w")
        ys = ttk.Scrollbar(wrap, orient="vertical", command=tv.yview)
        xs = ttk.Scrollbar(wrap, orient="horizontal", command=tv.xview)
        tv.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)
        tv.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        tv.tag_configure("baseline", foreground=COL_MUTED)
        tv.tag_configure("improved", foreground=COL_OK)
        tv.tag_configure("degraded", foreground=COL_WARN)
        tv.tag_configure("neutral", foreground="#333333")
        self._tv_conf_compare = tv

        ttk.Label(
            pane,
            text=(
                "Green row = Profit/DD improved vs None · "
                "Red row = Profit/DD degraded · "
                "Models appear automatically after Confidence Inference."
            ),
            foreground=COL_MUTED,
        ).pack(anchor="w", pady=(6, 0))

    def _on_research_subtab_changed(self) -> None:
        try:
            cur = self._research_sub_nb.select()
            tab = self._research_sub_nb.nametowidget(cur) if cur else None
            if tab is self._research_compare_pane:
                self._refresh_confidence_filter_comparison()
        except Exception:
            pass

    def _fmt_compare_pct(self, rate: float | None, *, digits: int = 2) -> str:
        if rate is None:
            return "—"
        try:
            return f"{100.0 * float(rate):.{digits}f}%"
        except (TypeError, ValueError):
            return "—"

    def _fmt_compare_num(self, value: float | None, *, digits: int = 3) -> str:
        if value is None:
            return "—"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "—"

    def _fmt_compare_delta(
        self,
        value: float | None,
        *,
        unit: str = "%",
        invert_good: bool = False,
    ) -> tuple[str, bool | None]:
        """Return (display, improved?) for a delta value."""
        if value is None:
            return "—", None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "—", None
        sign = "+" if v > 0 else ""
        improved = (v < 0) if invert_good else (v > 0)
        if abs(v) < 1e-9:
            return f"0.0{unit}", None
        return f"{sign}{v:.1f}{unit}", improved

    def _refresh_confidence_filter_comparison(self) -> None:
        if self._lab is None or not hasattr(self, "_tv_conf_compare"):
            return
        from chain_replay_ml.model_lab.research_dashboard import (
            compute_confidence_filter_comparison,
        )

        eval_set = str(
            getattr(self, "_research_evaluation_set_var", tk.StringVar(value="all")).get()
            or "all"
        )
        day_filter = self._research_filter_day
        try:
            result = compute_confidence_filter_comparison(
                self._lab.db_path,
                evaluation_set=eval_set,
                trading_day=day_filter,
            )
        except Exception as exc:
            self._compare_note_var.set(f"Comparison failed: {exc}")
            return

        tv = self._tv_conf_compare
        for item in tv.get_children():
            tv.delete(item)

        if result.get("error"):
            self._compare_note_var.set(f"Comparison failed: {result['error']}")
            return

        eval_label = result.get("evaluation_set_label") or "Seen + Unseen"
        n_models = int(result.get("models_compared") or 0)
        n0 = int(result.get("baseline_rows") or 0)
        scope = f"Day {day_filter}" if day_filter else eval_label
        if n_models == 0:
            self._compare_note_var.set(
                f"Evaluation Set: {scope} · {n0:,} baseline rows · "
                "No confidence inference columns yet — run Inference on Confidence Model."
            )
        else:
            self._compare_note_var.set(
                f"Evaluation Set: {scope} · {n0:,} baseline rows · "
                f"{n_models} confidence model(s) compared (pred = 1)."
            )

        for row in result.get("rows") or []:
            d_hit_s, _ = self._fmt_compare_delta(row.get("delta_hit_pp"), unit=" pp")
            d_dd_s, _ = self._fmt_compare_delta(
                row.get("delta_avg_dd_pct"), unit="%", invert_good=True
            )
            d_mfe_s, _ = self._fmt_compare_delta(row.get("delta_avg_max_profit_pct"))
            d_pd_s, _ = self._fmt_compare_delta(row.get("delta_profit_dd_pct"))

            removed = row.get("rows_removed")
            removed_pct = row.get("rows_removed_pct")
            if row.get("is_baseline"):
                removed_disp = "—"
            elif removed is None:
                removed_disp = "—"
            elif removed_pct is not None:
                removed_disp = f"{int(removed):,} ({float(removed_pct):.1f}%)"
            else:
                removed_disp = f"{int(removed):,}"

            if row.get("is_baseline"):
                tag = "baseline"
            else:
                improved_pd = row.get("improved_profit_dd")
                if improved_pd is True:
                    tag = "improved"
                elif improved_pd is False:
                    tag = "degraded"
                else:
                    tag = "neutral"

            tv.insert(
                "",
                "end",
                values=(
                    str(row.get("label") or "—"),
                    f"{int(row.get('rows_left') or 0):,}",
                    removed_disp,
                    self._fmt_compare_pct(row.get("hit_rate")),
                    self._fmt_compare_num(row.get("avg_dd")),
                    self._fmt_compare_num(row.get("avg_max_profit")),
                    self._fmt_compare_num(row.get("profit_dd"), digits=3),
                    self._fmt_compare_num(row.get("mae"), digits=3),
                    self._fmt_compare_num(row.get("premium_rmse"), digits=2),
                    d_hit_s if not row.get("is_baseline") else "—",
                    d_dd_s if not row.get("is_baseline") else "—",
                    d_mfe_s if not row.get("is_baseline") else "—",
                    d_pd_s if not row.get("is_baseline") else "—",
                ),
                tags=(tag,),
            )

    def _build_confidence_filter_effect_panel(
        self, parent: tk.Misc, *, grid_col: int | None = None
    ) -> None:
        """Before → after summary when Confidence Filter is active (same row as Risk)."""
        box = ttk.LabelFrame(parent, text="Confidence Filter Effect", padding=8)
        if grid_col is not None:
            box.grid(row=0, column=grid_col, sticky="nsew", padx=(0, 0))
        else:
            box.pack(side="left", fill="both", expand=True, padx=(0, 0))
        self._conf_effect_box = box
        self._conf_effect_vars = {
            "rows": tk.StringVar(value="—"),
            "hit": tk.StringVar(value="—"),
            "hit_delta": tk.StringVar(value=""),
            "dd": tk.StringVar(value="—"),
            "dd_delta": tk.StringVar(value=""),
            "mfe": tk.StringVar(value="—"),
            "mfe_delta": tk.StringVar(value=""),
            "hint": tk.StringVar(
                value="Enable Path Touch = 1 (or 0) to compare before → after."
            ),
        }
        ttk.Label(box, text="Rows Removed", foreground=COL_MUTED).grid(
            row=0, column=0, sticky="w", pady=(0, 2)
        )
        ttk.Label(
            box,
            textvariable=self._conf_effect_vars["rows"],
            font=("Segoe UI", 11, "bold"),
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))

        ttk.Label(box, text="Path Touch", foreground=COL_MUTED).grid(
            row=2, column=0, sticky="w"
        )
        ttk.Label(
            box, textvariable=self._conf_effect_vars["hit"], font=("Segoe UI", 10, "bold")
        ).grid(row=3, column=0, sticky="w")
        self._conf_effect_hit_delta_lbl = ttk.Label(
            box, textvariable=self._conf_effect_vars["hit_delta"]
        )
        self._conf_effect_hit_delta_lbl.grid(row=3, column=1, sticky="w", padx=(8, 0))

        ttk.Label(box, text="Avg DD", foreground=COL_MUTED).grid(
            row=4, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Label(
            box, textvariable=self._conf_effect_vars["dd"], font=("Segoe UI", 10, "bold")
        ).grid(row=5, column=0, sticky="w")
        self._conf_effect_dd_delta_lbl = ttk.Label(
            box, textvariable=self._conf_effect_vars["dd_delta"]
        )
        self._conf_effect_dd_delta_lbl.grid(row=5, column=1, sticky="w", padx=(8, 0))

        ttk.Label(box, text="Avg Max Profit", foreground=COL_MUTED).grid(
            row=6, column=0, sticky="w", pady=(6, 0)
        )
        ttk.Label(
            box, textvariable=self._conf_effect_vars["mfe"], font=("Segoe UI", 10, "bold")
        ).grid(row=7, column=0, sticky="w")
        self._conf_effect_mfe_delta_lbl = ttk.Label(
            box, textvariable=self._conf_effect_vars["mfe_delta"]
        )
        self._conf_effect_mfe_delta_lbl.grid(row=7, column=1, sticky="w", padx=(8, 0))

        ttk.Label(
            box,
            textvariable=self._conf_effect_vars["hint"],
            foreground=COL_MUTED,
            wraplength=300,
            justify="left",
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _fmt_conf_effect_rate(self, v: Any) -> str:
        if v is None:
            return "—"
        try:
            return f"{100.0 * float(v):.2f}%"
        except (TypeError, ValueError):
            return "—"

    def _fmt_conf_effect_num(self, v: Any) -> str:
        if v is None:
            return "—"
        try:
            return f"{float(v):.3f}"
        except (TypeError, ValueError):
            return "—"

    def _fmt_conf_effect_delta_pp(self, v: Any) -> str:
        if v is None:
            return ""
        try:
            x = float(v)
        except (TypeError, ValueError):
            return ""
        sign = "+" if x >= 0 else ""
        return f"({sign}{x:.2f}%)"

    def _fmt_conf_effect_delta_pct(self, v: Any) -> str:
        if v is None:
            return ""
        try:
            x = float(v)
        except (TypeError, ValueError):
            return ""
        sign = "+" if x >= 0 else ""
        return f"({sign}{x:.1f}%)"

    def _update_confidence_filter_effect_panel(self, dash: dict[str, Any]) -> None:
        if not hasattr(self, "_conf_effect_vars"):
            return
        effect = dash.get("confidence_filter_effect") or {}
        if not effect.get("available"):
            self._conf_effect_vars["rows"].set("—")
            self._conf_effect_vars["hit"].set("—")
            self._conf_effect_vars["hit_delta"].set("")
            self._conf_effect_vars["dd"].set("—")
            self._conf_effect_vars["dd_delta"].set("")
            self._conf_effect_vars["mfe"].set("—")
            self._conf_effect_vars["mfe_delta"].set("")
            self._conf_effect_vars["hint"].set(
                "Enable Path Touch = 1 (or 0) to compare before → after."
            )
            return

        removed = int(effect.get("rows_removed") or 0)
        removed_pct = effect.get("rows_removed_pct")
        if removed_pct is not None:
            self._conf_effect_vars["rows"].set(
                f"{removed:,} ({float(removed_pct):.1f}%)"
            )
        else:
            self._conf_effect_vars["rows"].set(f"{removed:,}")

        before_h = self._fmt_conf_effect_rate(effect.get("target_hit_before"))
        after_h = self._fmt_conf_effect_rate(effect.get("target_hit_after"))
        self._conf_effect_vars["hit"].set(f"{before_h} → {after_h}")
        self._conf_effect_vars["hit_delta"].set(
            self._fmt_conf_effect_delta_pp(effect.get("target_hit_delta_pp"))
        )
        self._color_conf_effect_delta(
            self._conf_effect_hit_delta_lbl, effect.get("target_hit_delta_pp"), higher_better=True
        )

        before_dd = self._fmt_conf_effect_num(effect.get("avg_dd_before"))
        after_dd = self._fmt_conf_effect_num(effect.get("avg_dd_after"))
        self._conf_effect_vars["dd"].set(f"{before_dd} → {after_dd}")
        self._conf_effect_vars["dd_delta"].set(
            self._fmt_conf_effect_delta_pct(effect.get("avg_dd_delta_pct"))
        )
        self._color_conf_effect_delta(
            self._conf_effect_dd_delta_lbl, effect.get("avg_dd_delta_pct"), higher_better=False
        )

        before_m = self._fmt_conf_effect_num(effect.get("avg_max_profit_before"))
        after_m = self._fmt_conf_effect_num(effect.get("avg_max_profit_after"))
        self._conf_effect_vars["mfe"].set(f"{before_m} → {after_m}")
        self._conf_effect_vars["mfe_delta"].set(
            self._fmt_conf_effect_delta_pct(effect.get("avg_max_profit_delta_pct"))
        )
        self._color_conf_effect_delta(
            self._conf_effect_mfe_delta_lbl,
            effect.get("avg_max_profit_delta_pct"),
            higher_better=True,
        )
        self._conf_effect_vars["hint"].set(
            "Baseline = unfiltered set · Filtered = Confidence Path Touch filter"
        )

    def _color_conf_effect_delta(
        self, lbl: ttk.Label, delta: Any, *, higher_better: bool
    ) -> None:
        try:
            x = float(delta)
        except (TypeError, ValueError):
            lbl.configure(foreground=COL_MUTED)
            return
        good = (x >= 0) if higher_better else (x <= 0)
        lbl.configure(foreground=COL_OK if good else COL_WARN)

    def _make_metric_panel(
        self,
        parent: tk.Misc,
        title: str,
        *,
        height: int = 8,
        grid_col: int | None = None,
    ) -> ttk.Treeview:
        box = ttk.LabelFrame(parent, text=title, padding=8)
        if grid_col is not None:
            box.grid(row=0, column=grid_col, sticky="nsew", padx=(0, 8))
        else:
            box.pack(side="left", fill="both", expand=True, padx=(0, 8))
        wrap = ttk.Frame(box)
        wrap.pack(fill="both", expand=True)
        tv = ttk.Treeview(
            wrap,
            columns=("metric", "value"),
            show="headings",
            height=height,
        )
        tv.heading("metric", text="Metric")
        tv.heading("value", text="Value")
        tv.column("metric", width=220, stretch=True, anchor="w")
        tv.column("value", width=120, stretch=True, anchor="e")
        ys = ttk.Scrollbar(wrap, orient="vertical", command=tv.yview)
        tv.configure(yscrollcommand=ys.set)
        tv.pack(side="left", fill="both", expand=True)
        ys.pack(side="right", fill="y")
        return tv

    def _make_table_panel(
        self,
        parent: tk.Misc,
        title: str,
        *,
        columns: tuple[tuple[str, str, int], ...],
        height: int = 6,
    ) -> ttk.Treeview:
        box = ttk.LabelFrame(parent, text=title, padding=8)
        box.pack(fill="x", pady=(0, 8))
        wrap = ttk.Frame(box)
        wrap.pack(fill="both", expand=True)
        ids = tuple(c[0] for c in columns)
        tv = ttk.Treeview(wrap, columns=ids, show="headings", height=height)
        for cid, heading, width in columns:
            tv.heading(cid, text=heading)
            tv.column(cid, width=width, minwidth=50, stretch=True, anchor="center")
        ys = ttk.Scrollbar(wrap, orient="vertical", command=tv.yview)
        xs = ttk.Scrollbar(wrap, orient="horizontal", command=tv.xview)
        tv.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)
        wrap.rowconfigure(0, weight=1)
        wrap.columnconfigure(0, weight=1)
        tv.grid(row=0, column=0, sticky="nsew")
        ys.grid(row=0, column=1, sticky="ns")
        xs.grid(row=1, column=0, sticky="ew")
        return tv

    def _fmt_dash_value(self, key: str, value: Any) -> str:
        if value is None:
            return "—"
        if key in (
            "total_predictions",
            "target_hits",
            "target_misses",
            "predicted_up",
            "predicted_down",
            "actual_up",
            "actual_down",
            "rows",
        ):
            try:
                return f"{int(value):,}"
            except (TypeError, ValueError):
                return str(value)
        if key.endswith("_rate") or key in (
            "direction_accuracy",
            "target_hit_rate",
            "target_miss_rate",
            "hit_rate",
        ):
            try:
                return f"{100.0 * float(value):.2f}%"
            except (TypeError, ValueError):
                return str(value)
        if "time_to_target" in key or key.endswith("_ttt"):
            try:
                return f"{float(value):.2f}"
            except (TypeError, ValueError):
                return str(value)
        if "premium" in key:
            try:
                return f"{float(value):.2f}%"
            except (TypeError, ValueError):
                return str(value)
        try:
            return f"{float(value):.4g}"
        except (TypeError, ValueError):
            return str(value)

    def _fill_metric_tree(
        self,
        tv: ttk.Treeview,
        rows: tuple[tuple[str, str], ...],
        data: dict[str, Any],
    ) -> None:
        for item in tv.get_children():
            tv.delete(item)
        for key, label in rows:
            tv.insert("", "end", values=(label, self._fmt_dash_value(key, data.get(key))))

    def _research_dashboard_dataset_scope(
        self,
        *,
        day_filter: str | None,
        dash: dict[str, Any] | None = None,
    ) -> tuple[str, str, str]:
        """Return (badge_text, badge_bg, detail_line) for Seen / Unseen / Both."""
        from chain_replay_ml.model_lab.prediction_schema import (
            DATASET_TYPE_SEEN,
            DATASET_TYPE_UNSEEN,
            normalize_dataset_type,
        )

        day = str(day_filter or "").strip()
        if day:
            dtype = normalize_dataset_type(
                (self._day_rows.get(day) or {}).get("dataset_type")
            )
            if dtype == DATASET_TYPE_UNSEEN:
                return (
                    "UNSEEN",
                    _UNSEEN_BADGE_BG,
                    f"Trading day {day} · not used in this model's training export",
                )
            return (
                "SEEN",
                _SEEN_BADGE_BG,
                f"Trading day {day} · used in this model's training / walk-forward",
            )

        # Prefer explicit Evaluation Set control when no single-day filter
        eval_set = None
        if isinstance(dash, dict):
            eval_set = str(dash.get("evaluation_set") or "").strip().lower()
        if not eval_set:
            eval_set = str(
                getattr(self, "_research_evaluation_set", None)
                or (
                    getattr(self, "_research_evaluation_set_var", None).get()
                    if getattr(self, "_research_evaluation_set_var", None)
                    else "all"
                )
                or "all"
            ).strip().lower()
        if eval_set == "seen":
            return (
                "SEEN ONLY",
                _SEEN_BADGE_BG,
                "Evaluation Set: Seen Only (training-export days)",
            )
        if eval_set == "unseen":
            return (
                "UNSEEN ONLY",
                _UNSEEN_BADGE_BG,
                "Evaluation Set: Unseen Only (holdout / Master-only days)",
            )

        types: set[str] = set()
        dash_days = dash.get("trading_days") if isinstance(dash, dict) else None
        if isinstance(dash_days, list) and dash_days:
            for item in dash_days:
                if not isinstance(item, dict):
                    continue
                td = str(item.get("trading_day") or "").strip()
                if not td:
                    continue
                types.add(
                    normalize_dataset_type(
                        (self._day_rows.get(td) or {}).get("dataset_type")
                    )
                )
        if not types and self._day_rows:
            for row in self._day_rows.values():
                if int(row.get("pred_rows") or 0) > 0:
                    types.add(normalize_dataset_type(row.get("dataset_type")))

        if types == {DATASET_TYPE_SEEN}:
            return (
                "SEEN ONLY",
                _SEEN_BADGE_BG,
                "All prediction rows are from training-export (Seen) days",
            )
        if types == {DATASET_TYPE_UNSEEN}:
            return (
                "UNSEEN ONLY",
                _UNSEEN_BADGE_BG,
                "All prediction rows are from Master-only (Unseen) days",
            )
        if DATASET_TYPE_SEEN in types and DATASET_TYPE_UNSEEN in types:
            return (
                "SEEN + UNSEEN",
                _BOTH_BADGE_BG,
                "Evaluation Set: Seen + Unseen (full lab prediction scope)",
            )
        return (
            "SEEN + UNSEEN",
            _BOTH_BADGE_BG,
            "Evaluation Set: Seen + Unseen (full lab prediction scope)",
        )

    def _update_research_scope_banner(
        self,
        *,
        day_filter: str | None,
        dash: dict[str, Any] | None = None,
        visible: bool = True,
    ) -> None:
        """Update Evaluation Set badge inside Confidence Filter (label only, no description)."""
        _ = visible  # Badge stays in Confidence Filter; never hide the panel
        if not hasattr(self, "_research_dataset_badge"):
            return
        badge, bg, _detail = self._research_dashboard_dataset_scope(
            day_filter=day_filter,
            dash=dash,
        )
        self._research_dataset_badge.configure(text=badge, bg=bg, fg=_BADGE_FG)

    def _research_selected_classifier_key(self) -> str:
        label = str(
            getattr(self, "_research_conf_classifier_var", tk.StringVar(value="Path Touch")).get()
            or "Path Touch"
        )
        keys = getattr(self, "_research_conf_classifier_keys", None) or {
            "Path Touch": "target_hit"
        }
        return str(keys.get(label) or "target_hit")

    def _on_research_evaluation_set_changed(self) -> None:
        self._research_evaluation_set = str(
            self._research_evaluation_set_var.get() or "all"
        )
        self._refresh_research_dashboard()

    def _on_research_confidence_filter_changed(self) -> None:
        self._research_conf_classifier = self._research_selected_classifier_key()
        self._research_conf_prediction = str(
            self._research_conf_prediction_var.get() or "disabled"
        )
        pred = self._research_conf_prediction
        if pred in ("0", "1"):
            self._research_confidence_filter = f"{self._research_conf_classifier}_{pred}"
        else:
            self._research_confidence_filter = "disabled"
        self._refresh_research_dashboard()

    def _update_research_confidence_filter_gate(self) -> None:
        """Enable =1/=0 only when the selected classifier has completed inference."""
        if not hasattr(self, "_research_conf_radios"):
            return
        available = False
        reason = "Run Confidence Inference before using this filter."
        lab_path = getattr(self._lab, "db_path", None) if self._lab else None
        model_key = self._research_selected_classifier_key()
        label = (
            getattr(self, "_research_conf_classifier_labels", {}) or {}
        ).get(model_key) or model_key
        if lab_path:
            try:
                from chain_replay_ml.model_lab.confidence import (
                    confidence_filter_available,
                )

                gate = confidence_filter_available(lab_path, model_key)
                available = bool(gate.get("available"))
                reason = str(
                    gate.get("reason")
                    or f"Run Confidence Inference for {label} before using this filter."
                )
            except Exception:
                available = False
        state = "normal" if available else "disabled"
        for val, rb in self._research_conf_radios.items():
            if val == "disabled":
                rb.configure(state="normal")
            else:
                rb.configure(state=state)
        if not available:
            cur = str(self._research_conf_prediction_var.get() or "disabled")
            if cur != "disabled":
                self._research_conf_prediction_var.set("disabled")
                self._research_conf_prediction = "disabled"
                self._research_confidence_filter = "disabled"
            self._research_conf_status_var.set(reason)
        elif str(self._research_conf_prediction_var.get() or "") == "disabled":
            if not str(self._research_conf_status_var.get() or "").startswith("Filter:"):
                self._research_conf_status_var.set("")

    def _refresh_research_dashboard(self, force: bool = False) -> None:
        if self._lab is None:
            return
        from chain_replay_ml.model_lab.research_dashboard import (
            DISTRIBUTION_ROWS,
            ERROR_ROWS,
            QUALITY_ROWS,
            RISK_ROWS,
            compute_research_dashboard,
        )

        self._update_research_confidence_filter_gate()

        day_filter = self._research_filter_day
        if day_filter:
            self._research_day_filter_var.set(f"Day: {day_filter}")
            self._btn_research_show_all.pack(side="left", padx=(0, 8))
        else:
            self._research_day_filter_var.set("")
            self._btn_research_show_all.pack_forget()

        classifier = self._research_selected_classifier_key()
        pred = str(
            getattr(self, "_research_conf_prediction_var", tk.StringVar(value="disabled")).get()
            or "disabled"
        )
        if hasattr(self, "_research_conf_prediction_var"):
            pred = str(self._research_conf_prediction_var.get() or "disabled")
        self._research_conf_classifier = classifier
        self._research_conf_prediction = pred
        if pred in ("0", "1"):
            conf_filter = f"{classifier}_{pred}"
        else:
            conf_filter = "disabled"
        self._research_confidence_filter = conf_filter

        eval_set = str(
            getattr(self, "_research_evaluation_set_var", tk.StringVar(value="all")).get()
            or "all"
        )
        if hasattr(self, "_research_evaluation_set_var"):
            eval_set = str(self._research_evaluation_set_var.get() or "all")
        self._research_evaluation_set = eval_set

        dash = compute_research_dashboard(
            self._lab.db_path,
            force_recompute=bool(force),
            data_dir=self._data_dir(),
            trading_day=day_filter,
            confidence_classifier=classifier,
            confidence_prediction=pred if pred in ("0", "1") else None,
            evaluation_set=eval_set,
        )
        if hasattr(self, "_research_conf_status_var"):
            meta = dash.get("confidence_filter_meta") or {}
            if pred in ("0", "1") and meta:
                thr = meta.get("threshold")
                thr_s = f"{float(thr):.2f}" if thr is not None else "—"
                rem = meta.get("rows_remaining")
                total = meta.get("total_rows")
                model_label = meta.get("model") or meta.get("label") or classifier
                bits = [
                    f"Filter: {model_label} = {pred}",
                    f"Model: {model_label}",
                    f"Threshold: {thr_s}",
                ]
                eval_label = dash.get("evaluation_set_label") or meta.get(
                    "evaluation_set_label"
                )
                if eval_label:
                    bits.append(f"Eval: {eval_label}")
                if rem is not None and total is not None:
                    bits.append(f"Rows remaining: {int(rem):,} / {int(total):,}")
                self._research_conf_status_var.set(" · ".join(bits))
            elif pred in ("0", "1"):
                self._research_conf_status_var.set(
                    "Confidence filter active — run Confidence Inference if empty."
                )
            else:
                # Evaluation Set is shown by the badge — no duplicate status text
                self._research_conf_status_var.set("")

        if dash.get("error"):
            self._research_note.set(f"Could not load dashboard: {dash['error']}")
            self._update_research_scope_banner(day_filter=day_filter, visible=False)
            self._update_confidence_filter_effect_panel({})
            self._refresh_confidence_filter_comparison()
            return
        if not dash.get("available"):
            if day_filter:
                self._research_note.set(
                    f"No prediction rows for {day_filter} — build predictions for this day first."
                )
            elif pred in ("0", "1"):
                self._research_note.set(
                    "No rows match the Confidence Filter — run Inference or change filter."
                )
            elif eval_set != "all":
                self._research_note.set(
                    "No prediction rows for this Evaluation Set — check Seen/Unseen day labels."
                )
            else:
                self._research_note.set(
                    "No prediction rows yet — build the Prediction Dataset first "
                    "(Test 1k or Start)."
                )
            self._update_research_scope_banner(day_filter=day_filter, visible=False)
            self._update_confidence_filter_effect_panel(dash)
            for var in self._kpi_vars.values():
                var.set("—")
            for tv in (
                self._tv_quality,
                self._tv_risk,
                self._tv_error,
                self._tv_distribution,
                self._tv_premium,
                self._tv_days,
            ):
                for item in tv.get_children():
                    tv.delete(item)
            self._refresh_confidence_filter_comparison()
            return

        n = int(dash.get("total_predictions") or 0)
        computed_at = str(dash.get("computed_at") or "")
        if pred in ("0", "1"):
            cache_note = "confidence filter · live"
        elif eval_set != "all":
            cache_note = "evaluation set · live"
        elif day_filter:
            cache_note = "day filter · live"
        elif dash.get("rebuilt"):
            cache_note = "stats rebuilt"
        elif dash.get("cached"):
            cache_note = "cached"
        else:
            cache_note = "live"
        when = f" · {computed_at}" if computed_at else ""
        scope = f" · {day_filter}" if day_filter else ""
        note = (
            f"{n:,} rows · dashboard stats · {cache_note}{scope}{when}"
            " · Feature Research is separate"
        )
        self._research_note.set(note)
        self._update_research_scope_banner(
            day_filter=day_filter,
            dash=dash,
            visible=True,
        )

        kpi = dash.get("kpi") or {}
        self._kpi_vars["target_hit_rate"].set(
            self._fmt_dash_value("target_hit_rate", kpi.get("target_hit_rate"))
        )
        self._kpi_vars["direction_accuracy"].set(
            self._fmt_dash_value("direction_accuracy", kpi.get("direction_accuracy"))
        )
        mae = kpi.get("mae")
        self._kpi_vars["mae"].set(
            f"₹{float(mae):.2f}" if mae is not None else "—"
        )
        prem = kpi.get("premium_rmse")
        self._kpi_vars["premium_rmse"].set(
            f"{float(prem):.2f}%" if prem is not None else "—"
        )

        self._fill_metric_tree(self._tv_quality, QUALITY_ROWS, dash.get("quality") or {})
        self._fill_metric_tree(self._tv_risk, RISK_ROWS, dash.get("risk") or {})
        self._update_confidence_filter_effect_panel(dash)
        self._fill_metric_tree(self._tv_error, ERROR_ROWS, dash.get("error_metrics") or {})
        self._fill_metric_tree(
            self._tv_distribution, DISTRIBUTION_ROWS, dash.get("distribution") or {},
        )

        for item in self._tv_premium.get_children():
            self._tv_premium.delete(item)
        for band in dash.get("premium_bands") or []:
            self._tv_premium.insert(
                "",
                "end",
                values=(
                    band.get("band"),
                    self._fmt_dash_value("rows", band.get("rows")),
                    self._fmt_dash_value("hit_rate", band.get("hit_rate")),
                    self._fmt_dash_value("direction_accuracy", band.get("direction_accuracy")),
                    self._fmt_dash_value("mae", band.get("mae")),
                    self._fmt_dash_value("avg_dd_before_target", band.get("avg_dd_before_target")),
                ),
            )

        for item in self._tv_days.get_children():
            self._tv_days.delete(item)
        for day in dash.get("trading_days") or []:
            self._tv_days.insert(
                "",
                "end",
                values=(
                    day.get("trading_day"),
                    self._fmt_dash_value("rows", day.get("rows")),
                    self._fmt_dash_value("hit_rate", day.get("hit_rate")),
                    self._fmt_dash_value("direction_accuracy", day.get("direction_accuracy")),
                    self._fmt_dash_value("mae", day.get("mae")),
                    self._fmt_dash_value("avg_dd_before_target", day.get("avg_dd_before_target")),
                ),
            )

        # Keep Comparison sub-tab in sync with Evaluation Set / day filter
        self._refresh_confidence_filter_comparison()

    def _build_feature_research_tab(self) -> None:
        """Dedicated Feature Research laboratory (list + analysis)."""
        paned = ttk.Panedwindow(self._feature_research_tab, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned, padding=4)
        right = ttk.Frame(paned, padding=4)
        paned.add(left, weight=1)
        paned.add(right, weight=3)

        ttk.Label(left, text="Feature Research", font=SECTION_FONT).pack(anchor="w")
        ttk.Label(
            left,
            text="Double-click or Open → Feature Analysis lab",
            foreground=COL_MUTED,
        ).pack(anchor="w", pady=(0, 6))
        toolbar = ttk.Frame(left)
        toolbar.pack(fill="x", pady=(0, 6))
        ttk.Button(toolbar, text="Refresh", command=self._refresh_feature_research_list).pack(
            side="left",
        )
        ttk.Button(toolbar, text="Open", command=self._open_selected_feature_analysis).pack(
            side="left", padx=6,
        )
        self._fr_list_note = tk.StringVar(value="")
        ttk.Label(left, textvariable=self._fr_list_note, foreground=COL_MUTED).pack(anchor="w")

        list_wrap = ttk.Frame(left)
        list_wrap.pack(fill="both", expand=True, pady=(4, 0))
        self._fr_list = ttk.Treeview(
            list_wrap,
            columns=("research_rank", "feature", "model_rank"),
            show="headings",
            height=22,
        )
        self._fr_list.heading("research_rank", text="Research Rank")
        self._fr_list.heading("feature", text="Feature")
        self._fr_list.heading("model_rank", text="Model Rank")
        self._fr_list.column("research_rank", width=96, anchor="center", stretch=False)
        self._fr_list.column("feature", width=240, anchor="w", stretch=True)
        self._fr_list.column("model_rank", width=88, anchor="center", stretch=False)
        ys = ttk.Scrollbar(list_wrap, orient="vertical", command=self._fr_list.yview)
        self._fr_list.configure(yscrollcommand=ys.set)
        self._fr_list.pack(side="left", fill="both", expand=True)
        ys.pack(side="right", fill="y")
        self._fr_list.bind("<Double-1>", lambda _e: self._open_selected_feature_analysis())

        # —— Analysis lab (scrollable) ——
        self._fr_analysis_scroll = ScrollableFrame(right)
        self._fr_analysis_scroll.pack(fill="both", expand=True)
        lab = self._fr_analysis_scroll.inner

        self._fr_title = tk.StringVar(value="Feature Analysis")
        ttk.Label(lab, textvariable=self._fr_title, font=SECTION_FONT).pack(anchor="w")
        self._fr_subtitle = tk.StringVar(value="Select a feature and click Open.")
        ttk.Label(
            lab,
            textvariable=self._fr_subtitle,
            font=("Consolas", 10),
            foreground=COL_MUTED,
            justify="left",
        ).pack(anchor="w", pady=(0, 8))

        conclusion_box = ttk.LabelFrame(lab, text="Research Conclusion", padding=8)
        conclusion_box.pack(fill="x", pady=(0, 8))
        self._fr_conclusion_var = tk.StringVar(value="Open a feature to generate a conclusion.")
        ttk.Label(
            conclusion_box,
            textvariable=self._fr_conclusion_var,
            justify="left",
            font=("Segoe UI", 10),
        ).pack(anchor="w")

        stats_box = ttk.LabelFrame(lab, text="Distribution Stats", padding=8)
        stats_box.pack(fill="x", pady=(0, 8))
        self._fr_stats_var = tk.StringVar(value="—")
        ttk.Label(
            stats_box,
            textvariable=self._fr_stats_var,
            font=("Consolas", 10),
            justify="left",
        ).pack(anchor="w")

        self._fr_tert = self._make_table_panel(
            lab,
            "Low / Medium / High",
            columns=(
                ("band", "Range", 80),
                ("rows", "Rows", 60),
                ("hit", "Hit %", 70),
                ("dir", "Dir %", 70),
                ("mae", "MAE", 70),
                ("dd", "DD", 70),
            ),
            height=4,
        )

        gold = ttk.Frame(lab)
        gold.pack(fill="x", pady=(0, 8))
        best_box = ttk.LabelFrame(gold, text="Best Range", padding=8)
        best_box.pack(side="left", fill="both", expand=True, padx=(0, 6))
        self._fr_best_var = tk.StringVar(value="—")
        ttk.Label(best_box, textvariable=self._fr_best_var, justify="left").pack(anchor="w")
        ttk.Button(
            best_box,
            text="Show rows",
            command=lambda: self._show_feature_filter_rows("best"),
        ).pack(anchor="w", pady=(6, 0))

        worst_box = ttk.LabelFrame(gold, text="Worst Range", padding=8)
        worst_box.pack(side="left", fill="both", expand=True)
        self._fr_worst_var = tk.StringVar(value="—")
        ttk.Label(worst_box, textvariable=self._fr_worst_var, justify="left").pack(anchor="w")
        ttk.Button(
            worst_box,
            text="Show rows",
            command=lambda: self._show_feature_filter_rows("worst"),
        ).pack(anchor="w", pady=(6, 0))

        hist_box = ttk.LabelFrame(lab, text="Feature Histogram", padding=8)
        hist_box.pack(fill="x", pady=(0, 8))
        self._fr_hist_note = tk.StringVar(value="")
        ttk.Label(hist_box, textvariable=self._fr_hist_note, foreground=COL_MUTED).pack(
            anchor="w",
        )
        self._fr_hist_canvas = tk.Canvas(hist_box, height=120, highlightthickness=0)
        self._fr_hist_canvas.pack(fill="x", pady=(4, 0))

        cmp_box = ttk.LabelFrame(lab, text="Compare Low vs High", padding=8)
        cmp_box.pack(fill="x", pady=(0, 8))
        self._fr_compare_var = tk.StringVar(value="—")
        ttk.Label(cmp_box, textvariable=self._fr_compare_var, justify="left").pack(anchor="w")
        cmp_btns = ttk.Frame(cmp_box)
        cmp_btns.pack(anchor="w", pady=(6, 0))
        ttk.Button(
            cmp_btns,
            text="Show Low rows",
            command=lambda: self._show_feature_filter_rows("low"),
        ).pack(side="left")
        ttk.Button(
            cmp_btns,
            text="Show High rows",
            command=lambda: self._show_feature_filter_rows("high"),
        ).pack(side="left", padx=6)

    def _refresh_feature_research_list(self) -> None:
        if self._lab is None:
            return
        from chain_replay_ml.model_lab.feature_research import list_research_features

        for item in self._fr_list.get_children():
            self._fr_list.delete(item)
        result = list_research_features(self._lab.db_path, data_dir=self._data_dir())
        if result.get("error"):
            self._fr_list_note.set(str(result["error"]))
            return
        feats = list(result.get("features") or [])
        self._feature_list_rows = feats
        if not feats:
            self._fr_list_note.set("No selected feature columns in prediction dataset yet.")
            return
        self._fr_list_note.set(
            f"{len(feats)} features · Research Rank by Hit% tertile spread"
        )
        for row in feats:
            mr = row.get("model_rank")
            if mr is None:
                mr = row.get("feature_rank")
            iid = self._fr_list.insert(
                "",
                "end",
                values=(
                    row.get("research_rank") or row.get("rank"),
                    row.get("feature"),
                    mr if mr is not None else "—",
                ),
            )
            # stash feature name
            self._fr_list.item(iid, tags=(str(row.get("feature") or ""),))

    def _selected_feature_name(self) -> str | None:
        sel = self._fr_list.selection()
        if not sel:
            return None
        vals = self._fr_list.item(sel[0], "values")
        if vals and len(vals) >= 2:
            return str(vals[1])
        tags = self._fr_list.item(sel[0], "tags")
        return str(tags[0]) if tags else None

    def _open_selected_feature_analysis(self) -> None:
        name = self._selected_feature_name()
        if not name:
            messagebox.showinfo("Feature Research", "Select a feature first.", parent=self)
            return
        self._load_feature_analysis(name)

    def _fmt_fr_pct(self, v: Any) -> str:
        if v is None:
            return "—"
        try:
            return f"{100.0 * float(v):.1f}%"
        except (TypeError, ValueError):
            return str(v)

    def _fmt_fr_num(self, v: Any, digits: int = 4) -> str:
        if v is None:
            return "—"
        try:
            return f"{float(v):.{digits}g}"
        except (TypeError, ValueError):
            return str(v)

    def _fmt_fr_meta_int(self, v: Any) -> str:
        if v is None:
            return "—"
        try:
            return f"{int(v):,}"
        except (TypeError, ValueError):
            return str(v)

    def _fmt_fr_coverage(self, v: Any) -> str:
        if v is None:
            return "—"
        try:
            return f"{100.0 * float(v):.1f}%"
        except (TypeError, ValueError):
            return str(v)

    def _feature_analysis_header_text(self, analysis: dict[str, Any]) -> str:
        model_rank = analysis.get("model_rank")
        if model_rank is None:
            model_rank = analysis.get("feature_rank")
        rows = analysis.get("rows_analyzed")
        if rows is None:
            rows = analysis.get("rows")
        missing = analysis.get("missing_values")
        if missing is None:
            total = analysis.get("total_rows")
            try:
                missing = max(0, int(total or 0) - int(rows or 0))
            except (TypeError, ValueError):
                missing = None
        return (
            f"Model Rank       : {self._fmt_fr_meta_int(model_rank)}\n"
            f"Research Rank    : {self._fmt_fr_meta_int(analysis.get('research_rank'))}\n"
            f"Rows Analyzed    : {self._fmt_fr_meta_int(rows)}\n"
            f"Missing Values   : {self._fmt_fr_meta_int(missing)}"
        )

    def _load_feature_analysis(self, feature_name: str) -> None:
        if self._lab is None:
            return
        from chain_replay_ml.model_lab.feature_research import analyze_feature

        analysis = analyze_feature(self._lab.db_path, feature_name)
        self._feature_analysis = analysis
        self._fr_title.set(f"Feature Analysis — {feature_name}")
        if analysis.get("error") or not analysis.get("available"):
            self._fr_subtitle.set(str(analysis.get("error") or "Unavailable"))
            self._fr_conclusion_var.set("Not enough data for a research conclusion.")
            return

        stats = analysis.get("stats") or {}
        self._fr_subtitle.set(self._feature_analysis_header_text(analysis))
        conclusion = analysis.get("conclusion") or {}
        self._fr_conclusion_var.set(
            str(conclusion.get("text") or "Not enough Low vs High contrast to form a conclusion.")
        )
        self._fr_stats_var.set(
            f"Minimum   {self._fmt_fr_num(stats.get('minimum'))}\n"
            f"Maximum   {self._fmt_fr_num(stats.get('maximum'))}\n"
            f"Average   {self._fmt_fr_num(stats.get('average'))}\n"
            f"Median    {self._fmt_fr_num(stats.get('median'))}\n"
            f"Std Dev   {self._fmt_fr_num(stats.get('std_dev'))}"
        )

        tert = analysis.get("tertiles") or {}
        thr = tert.get("thresholds") or {}
        for item in self._fr_tert.get_children():
            self._fr_tert.delete(item)
        for key, label in (
            ("low", "Low"),
            ("medium", "Medium"),
            ("high", "High"),
        ):
            b = tert.get(key) or {}
            self._fr_tert.insert(
                "",
                "end",
                values=(
                    label,
                    b.get("rows"),
                    self._fmt_fr_pct(b.get("hit_rate")),
                    self._fmt_fr_pct(b.get("direction_accuracy")),
                    self._fmt_fr_num(b.get("mae")),
                    self._fmt_fr_num(b.get("avg_dd_before_target")),
                ),
            )
        thr_txt = (
            f"Cutoffs: Low ≤ {self._fmt_fr_num(thr.get('low_max'))} · "
            f"High > {self._fmt_fr_num(thr.get('high_min'))}"
        )

        best = analysis.get("best_range")
        if best:
            self._fr_best_var.set(
                f"Hit {self._fmt_fr_pct(best.get('hit_rate'))}\n"
                f"Coverage {self._fmt_fr_coverage(best.get('coverage'))}\n"
                f"when {self._fmt_fr_num(best.get('lo'))} ~ {self._fmt_fr_num(best.get('hi'))}\n"
                f"Rows {self._fmt_fr_meta_int(best.get('rows'))} · "
                f"Dir {self._fmt_fr_pct(best.get('direction_accuracy'))} · "
                f"DD {self._fmt_fr_num(best.get('avg_dd_before_target'))}"
            )
        else:
            self._fr_best_var.set("—")

        worst = analysis.get("worst_range")
        if worst:
            self._fr_worst_var.set(
                f"Hit {self._fmt_fr_pct(worst.get('hit_rate'))}\n"
                f"Coverage {self._fmt_fr_coverage(worst.get('coverage'))}\n"
                f"when {self._fmt_fr_num(worst.get('lo'))} ~ {self._fmt_fr_num(worst.get('hi'))}\n"
                f"Rows {self._fmt_fr_meta_int(worst.get('rows'))} · "
                f"Dir {self._fmt_fr_pct(worst.get('direction_accuracy'))} · "
                f"DD {self._fmt_fr_num(worst.get('avg_dd_before_target'))}"
            )
        else:
            self._fr_worst_var.set("—")

        peak = analysis.get("peak_bin") or {}
        if peak:
            self._fr_hist_note.set(
                f"Most values around {self._fmt_fr_num(peak.get('center'))} "
                f"({peak.get('count')} rows in bin) · {thr_txt}"
            )
        else:
            self._fr_hist_note.set(thr_txt)
        self._draw_feature_histogram(analysis.get("histogram") or [])

        cmp = analysis.get("compare") or {}
        lo = cmp.get("low") or {}
        hi = cmp.get("high") or {}
        self._fr_compare_var.set(
            f"Low   Hit {self._fmt_fr_pct(lo.get('hit_rate'))}  "
            f"({self._fmt_fr_meta_int(lo.get('rows'))} rows, "
            f"cov {self._fmt_fr_coverage(lo.get('coverage'))})\n"
            f"      Dir {self._fmt_fr_pct(lo.get('direction_accuracy'))}  "
            f"DD {self._fmt_fr_num(lo.get('avg_dd_before_target'))}  "
            f"MAE {self._fmt_fr_num(lo.get('mae'))}\n"
            f"High  Hit {self._fmt_fr_pct(hi.get('hit_rate'))}  "
            f"({self._fmt_fr_meta_int(hi.get('rows'))} rows, "
            f"cov {self._fmt_fr_coverage(hi.get('coverage'))})\n"
            f"      Dir {self._fmt_fr_pct(hi.get('direction_accuracy'))}  "
            f"DD {self._fmt_fr_num(hi.get('avg_dd_before_target'))}  "
            f"MAE {self._fmt_fr_num(hi.get('mae'))}\n"
            f"Δ Hit {self._fmt_fr_delta_pct(cmp.get('delta_hit_rate'))}  "
            f"Δ Dir {self._fmt_fr_delta_pct(cmp.get('delta_dir'))}  "
            f"Δ DD {self._fmt_fr_num(cmp.get('delta_dd'))}"
        )

    def _fmt_fr_delta_pct(self, v: Any) -> str:
        if v is None:
            return "—"
        try:
            return f"{100.0 * float(v):+.1f}pp"
        except (TypeError, ValueError):
            return str(v)

    def _draw_feature_histogram(self, bins: list[dict[str, Any]]) -> None:
        canvas = self._fr_hist_canvas
        canvas.delete("all")
        canvas.update_idletasks()
        w = max(int(canvas.winfo_width() or 400), 320)
        h = 120
        canvas.configure(width=w, height=h)
        if not bins:
            return
        max_c = max(int(b.get("count") or 0) for b in bins) or 1
        pad = 4
        bar_w = max(2.0, (w - 2 * pad) / len(bins))
        for i, b in enumerate(bins):
            c = int(b.get("count") or 0)
            bh = (h - 16) * (c / max_c)
            x0 = pad + i * bar_w
            x1 = x0 + bar_w * 0.85
            y1 = h - 4
            y0 = y1 - bh
            canvas.create_rectangle(x0, y0, x1, y1, fill="#4C8BF5", outline="")

    def _show_feature_filter_rows(self, which: str) -> None:
        analysis = self._feature_analysis or {}
        filters = analysis.get("filters") or {}
        spec = filters.get(which)
        if not spec:
            messagebox.showinfo(
                "Show rows",
                "No filter available for this band yet.",
                parent=self,
            )
            return
        from chain_replay_ml.model_lab.feature_research import (
            filter_label_from_spec,
            filter_sql_from_spec,
        )
        from chain_replay_ml.model_lab.prediction_feature_store import PredictionFeatureStore
        from chain_replay_ml.model_lab.store import ModelLabStore

        referenced = False
        try:
            with ModelLabStore(self._lab.db_path) as store:
                referenced = PredictionFeatureStore.from_store(store).is_referenced()
        except Exception:
            referenced = False
        where_sql, where_args = filter_sql_from_spec(spec, referenced=referenced)
        if not where_sql:
            messagebox.showerror("Show rows", "Invalid filter.", parent=self)
            return
        feat = str(analysis.get("feature") or "")
        self._explorer_where_sql = where_sql
        self._explorer_where_args = list(where_args)
        self._explorer_filter_spec = dict(spec)
        self._explorer_filter_feature = feat
        self._explorer_filter_desc.set(filter_label_from_spec(spec, feature=feat))
        self._explorer_offset = 0
        self._show_applied_filter_banner()
        self._notebook.select(self._prediction_tab)
        self._reload_explorer()

    def _show_applied_filter_banner(self) -> None:
        if getattr(self, "_applied_filter_frame", None) is None:
            return
        self._applied_filter_frame.pack(fill="x", pady=(0, 6), before=self._explorer_toolbar)

    def _hide_applied_filter_banner(self) -> None:
        if getattr(self, "_applied_filter_frame", None) is None:
            return
        self._applied_filter_frame.pack_forget()
        self._explorer_filter_desc.set("")
        self._explorer_filter_count.set("")

    def _clear_explorer_feature_filter(self) -> None:
        self._explorer_where_sql = ""
        self._explorer_where_args = []
        self._explorer_filter_spec = None
        self._explorer_filter_feature = ""
        self._hide_applied_filter_banner()
        self._explorer_offset = 0
        self._reload_explorer()

    def _build_research_programs_tab(self) -> None:
        """Hypothesis playbooks — cohorts over the prediction dataset."""
        paned = ttk.Panedwindow(self._research_programs_tab, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left = ttk.Frame(paned, padding=4)
        right = ttk.Frame(paned, padding=4)
        paned.add(left, weight=1)
        paned.add(right, weight=3)

        ttk.Label(left, text="Research Programs", font=SECTION_FONT).pack(anchor="w")
        ttk.Label(
            left,
            text="Each program answers a research question",
            foreground=COL_MUTED,
        ).pack(anchor="w", pady=(0, 6))
        toolbar = ttk.Frame(left)
        toolbar.pack(fill="x", pady=(0, 4))
        ttk.Button(toolbar, text="Refresh", command=self._refresh_research_programs_list).pack(
            side="left",
        )
        ttk.Button(toolbar, text="Run", command=self._run_selected_research_program).pack(
            side="left", padx=6,
        )

        cfg = ttk.Frame(left)
        cfg.pack(fill="x", pady=(0, 6))
        ttk.Label(cfg, text="Top N").pack(side="left")
        self._rp_limit_var = tk.StringVar(value="1000")
        self._rp_limit_combo = ttk.Combobox(
            cfg,
            textvariable=self._rp_limit_var,
            values=("100", "500", "1000", "5000"),
            width=8,
            state="readonly",
        )
        self._rp_limit_combo.pack(side="left", padx=6)
        ttk.Label(cfg, text="Compare").pack(side="left", padx=(8, 0))
        self._rp_compare_var = tk.StringVar(value="dataset")
        self._rp_compare_combo = ttk.Combobox(
            cfg,
            textvariable=self._rp_compare_var,
            values=("dataset", "worst"),
            width=12,
            state="readonly",
        )
        self._rp_compare_combo.pack(side="left", padx=6)

        self._rp_list_note = tk.StringVar(value="")
        ttk.Label(left, textvariable=self._rp_list_note, foreground=COL_MUTED).pack(anchor="w")

        list_wrap = ttk.Frame(left)
        list_wrap.pack(fill="both", expand=True, pady=(4, 0))
        self._rp_list = ttk.Treeview(
            list_wrap,
            columns=("title", "kind"),
            show="headings",
            height=20,
        )
        self._rp_list.heading("title", text="Program")
        self._rp_list.heading("kind", text="Kind")
        self._rp_list.column("title", width=220, anchor="w", stretch=True)
        self._rp_list.column("kind", width=72, anchor="center", stretch=False)
        ys = ttk.Scrollbar(list_wrap, orient="vertical", command=self._rp_list.yview)
        self._rp_list.configure(yscrollcommand=ys.set)
        self._rp_list.pack(side="left", fill="both", expand=True)
        ys.pack(side="right", fill="y")
        self._rp_list.bind("<Double-1>", lambda _e: self._run_selected_research_program())
        self._rp_list.bind("<<TreeviewSelect>>", lambda _e: self._on_research_program_select())

        self._rp_result_scroll = ScrollableFrame(right)
        self._rp_result_scroll.pack(fill="both", expand=True)
        lab = self._rp_result_scroll.inner

        self._rp_title = tk.StringVar(value="Research Program")
        ttk.Label(lab, textvariable=self._rp_title, font=SECTION_FONT).pack(anchor="w")
        self._rp_answer = tk.StringVar(value="")
        ttk.Label(
            lab,
            textvariable=self._rp_answer,
            font=("Segoe UI", 10, "bold"),
            justify="left",
            wraplength=520,
        ).pack(anchor="w", pady=(2, 2))
        self._rp_hypothesis = tk.StringVar(value="Select a program and click Run.")
        ttk.Label(
            lab,
            textvariable=self._rp_hypothesis,
            foreground=COL_MUTED,
            wraplength=520,
            justify="left",
        ).pack(anchor="w", pady=(0, 8))

        self._rp_summary_var = tk.StringVar(value="")
        ttk.Label(
            lab,
            textvariable=self._rp_summary_var,
            font=("Consolas", 10),
            justify="left",
        ).pack(anchor="w", pady=(0, 8))

        self._rp_cohorts_host = ttk.Frame(lab)
        self._rp_cohorts_host.pack(fill="both", expand=True)

        ttk.Label(
            lab,
            text="Feature Importance Evolution (cross-model) — coming next.",
            foreground=COL_MUTED,
        ).pack(anchor="w", pady=(12, 0))

    def _refresh_research_programs_list(self) -> None:
        from chain_replay_ml.model_lab.research_programs import list_research_programs

        for item in self._rp_list.get_children():
            self._rp_list.delete(item)
        programs = list_research_programs()
        self._rp_list_note.set(f"{len(programs)} hypothesis programs")
        for p in programs:
            iid = self._rp_list.insert(
                "",
                "end",
                values=(p.get("title"), p.get("kind")),
            )
            self._rp_list.item(iid, tags=(str(p.get("id") or ""),))

    def _selected_research_program_id(self) -> str | None:
        sel = self._rp_list.selection()
        if not sel:
            return None
        tags = self._rp_list.item(sel[0], "tags")
        return str(tags[0]) if tags else None

    def _on_research_program_select(self) -> None:
        from chain_replay_ml.model_lab.research_programs import list_research_programs

        pid = self._selected_research_program_id()
        meta = next((p for p in list_research_programs() if p.get("id") == pid), None)
        if not meta:
            return
        opts = list(meta.get("compare_options") or ["dataset"])
        self._rp_compare_combo.configure(values=opts)
        if self._rp_compare_var.get() not in opts:
            self._rp_compare_var.set(opts[0])
        if meta.get("configurable_limit"):
            self._rp_limit_combo.configure(state="readonly")
        else:
            self._rp_limit_combo.configure(state="disabled")

    def _run_selected_research_program(self) -> None:
        if self._lab is None:
            return
        program_id = self._selected_research_program_id()
        if not program_id:
            messagebox.showinfo("Research Programs", "Select a program first.", parent=self)
            return
        from chain_replay_ml.model_lab.research_programs import run_research_program

        try:
            limit = int(self._rp_limit_var.get() or 1000)
        except (TypeError, ValueError):
            limit = 1000
        compare_to = str(self._rp_compare_var.get() or "dataset")
        result = run_research_program(
            self._lab.db_path,
            program_id,
            limit=limit,
            compare_to=compare_to,
        )
        self._program_result = result
        self._program_cohorts = list(result.get("cohorts") or [])
        self._render_research_program_result(result)

    def _render_research_program_result(self, result: dict[str, Any]) -> None:
        for child in self._rp_cohorts_host.winfo_children():
            child.destroy()

        prog = result.get("program") or {}
        title = str(prog.get("title") or result.get("program_id") or "Research Program")
        self._rp_title.set(title)
        if result.get("error") or not result.get("available"):
            self._rp_answer.set("")
            self._rp_hypothesis.set(str(result.get("error") or "Unavailable"))
            self._rp_summary_var.set("")
            return

        answer = str(result.get("answer") or prog.get("answer") or "")
        self._rp_answer.set(f"Answer: {answer}" if answer else "")
        self._rp_hypothesis.set(str(result.get("hypothesis") or ""))
        cohorts = list(result.get("cohorts") or [])
        cmp = result.get("compare_to") or "dataset"
        self._rp_summary_var.set(
            f"{len(cohorts)} cohort(s) · Top N={result.get('limit') or 1000} · Compare={cmp}"
        )

        for i, cohort in enumerate(cohorts):
            self._render_program_cohort_card(self._rp_cohorts_host, cohort, index=i)

    def _render_program_cohort_card(
        self,
        host: ttk.Frame,
        cohort: dict[str, Any],
        *,
        index: int,
    ) -> None:
        box = ttk.LabelFrame(host, text=str(cohort.get("title") or f"Cohort {index + 1}"), padding=8)
        box.pack(fill="x", pady=(0, 8))
        m = cohort.get("executive_summary") or cohort.get("metrics") or {}

        # 1. Executive Summary
        exec_box = ttk.LabelFrame(box, text="Executive Summary", padding=6)
        exec_box.pack(fill="x", pady=(0, 8))
        ttk.Label(
            exec_box,
            text=(
                f"Rows: {self._fmt_fr_meta_int(m.get('rows'))}"
                f" ({self._fmt_fr_coverage(m.get('coverage'))})\n"
                f"Path Touch Rate: {self._fmt_fr_pct(m.get('hit_rate'))}\n"
                f"Direction Accuracy: {self._fmt_fr_pct(m.get('direction_accuracy'))}\n"
                f"Average MAE: {self._fmt_fr_num(m.get('mae'))}\n"
                f"Average DD: {self._fmt_fr_num(m.get('avg_dd_before_target'))}\n"
                f"Average Time To Target: {self._fmt_fr_num(m.get('avg_time_to_target'))} s"
            ),
            justify="left",
            font=("Consolas", 10),
        ).pack(anchor="w")

        # 2. Why table
        why_cols = cohort.get("why_columns") or {}
        col_a = str(why_cols.get("cohort") or "Cohort")
        col_b = str(why_cols.get("compare") or "Overall")
        why_title = str(
            (self._program_result or {}).get("answer")
            or cohort.get("label")
            or "Why?"
        )
        why_box = ttk.LabelFrame(box, text=why_title, padding=6)
        why_box.pack(fill="x", pady=(0, 8))
        ttk.Label(
            why_box,
            text=f"vs {cohort.get('compare_label') or col_b}",
            foreground=COL_MUTED,
        ).pack(anchor="w")
        why_tree = ttk.Treeview(
            why_box,
            columns=("feature", "cohort", "compare", "diff", "coverage", "effect", "confidence"),
            show="headings",
            height=8,
        )
        why_tree.heading("feature", text="Feature")
        why_tree.heading("cohort", text=col_a)
        why_tree.heading("compare", text=col_b)
        why_tree.heading("diff", text="Difference")
        why_tree.heading("coverage", text="Coverage")
        why_tree.heading("effect", text="Effect")
        why_tree.heading("confidence", text="Confidence")
        why_tree.column("feature", width=160, anchor="w")
        why_tree.column("cohort", width=72, anchor="e")
        why_tree.column("compare", width=72, anchor="e")
        why_tree.column("diff", width=72, anchor="center")
        why_tree.column("coverage", width=70, anchor="center")
        why_tree.column("effect", width=70, anchor="center")
        why_tree.column("confidence", width=80, anchor="center")
        why_tree.pack(fill="x", pady=(4, 0))
        for row in list(cohort.get("why_rows") or [])[:12]:
            why_tree.insert(
                "",
                "end",
                values=(
                    row.get("feature"),
                    self._fmt_fr_num(row.get("cohort_mean")),
                    self._fmt_fr_num(row.get("compare_mean")),
                    row.get("difference"),
                    self._fmt_fr_coverage(row.get("coverage")),
                    row.get("effect") or "—",
                    row.get("confidence") or "—",
                ),
            )

        # 3. Conclusions
        conc = cohort.get("conclusions") or {}
        conc_box = ttk.LabelFrame(box, text="Research Conclusions", padding=6)
        conc_box.pack(fill="x", pady=(0, 8))
        ttk.Label(
            conc_box,
            text=str(conc.get("text") or "—"),
            justify="left",
            font=("Segoe UI", 10),
        ).pack(anchor="w")

        btns = ttk.Frame(box)
        btns.pack(anchor="w", pady=(4, 0))
        ttk.Button(
            btns,
            text="Show rows",
            command=lambda c=cohort: self._show_program_cohort_rows(c),
        ).pack(side="left")
        ttk.Button(
            btns,
            text="Re-run Compare",
            command=self._run_selected_research_program,
        ).pack(side="left", padx=6)

    def _show_program_cohort_rows(self, cohort: dict[str, Any]) -> None:
        where_sql = str(cohort.get("where_sql") or "").strip()
        if not where_sql:
            messagebox.showinfo("Show rows", "No filter for this cohort.", parent=self)
            return
        self._explorer_where_sql = where_sql
        self._explorer_where_args = list(cohort.get("where_args") or [])
        self._explorer_filter_spec = None
        self._explorer_filter_feature = ""
        self._explorer_filter_desc.set(str(cohort.get("label") or cohort.get("title") or where_sql))
        self._explorer_offset = 0
        self._show_applied_filter_banner()
        self._notebook.select(self._prediction_tab)
        self._reload_explorer()

    def _build_model_improvement_tab(self) -> None:
        """Suggestions for the next train — not facts."""
        root = self._model_improvement_tab
        ttk.Label(root, text="Model Improvement", font=SECTION_FONT).pack(anchor="w")
        ttk.Label(
            root,
            text="Suggestions for the next experiment · Research Score vs Model Rank",
            foreground=COL_MUTED,
        ).pack(anchor="w", pady=(0, 6))

        toolbar = ttk.Frame(root)
        toolbar.pack(fill="x", pady=(0, 6))
        self._mi_start_btn = ttk.Button(
            toolbar,
            text="Start Analysis",
            command=self._refresh_model_improvement,
        )
        self._mi_start_btn.pack(side="left")
        self._mi_refresh_btn = ttk.Button(
            toolbar,
            text="Refresh",
            command=self._refresh_model_improvement,
        )
        self._mi_refresh_btn.pack(side="left", padx=(6, 0))

        self._mi_answers_var = tk.StringVar(value="")
        ttk.Label(
            root,
            textvariable=self._mi_answers_var,
            justify="left",
            font=("Consolas", 10),
        ).pack(anchor="w", pady=(0, 8))

        self._mi_summary_var = tk.StringVar(
            value="Click Start Analysis after the prediction dataset is ready "
            "(heavy scan — does not run on tab open)."
        )
        ttk.Label(root, textvariable=self._mi_summary_var, foreground=COL_MUTED).pack(anchor="w")

        paned = ttk.Panedwindow(root, orient="horizontal")
        paned.pack(fill="both", expand=True, pady=(8, 0))
        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=3)
        paned.add(right, weight=2)

        cols = ("feature", "score", "model_rank", "research_rank", "evidence", "rec")
        self._mi_tree = ttk.Treeview(left, columns=cols, show="headings", height=16)
        self._mi_tree.heading("feature", text="Feature")
        self._mi_tree.heading("score", text="Research Score")
        self._mi_tree.heading("model_rank", text="Model Rank")
        self._mi_tree.heading("research_rank", text="Research Rank")
        self._mi_tree.heading("evidence", text="Evidence")
        self._mi_tree.heading("rec", text="Suggestion")
        self._mi_tree.column("feature", width=160, anchor="w")
        self._mi_tree.column("score", width=96, anchor="center")
        self._mi_tree.column("model_rank", width=80, anchor="center")
        self._mi_tree.column("research_rank", width=88, anchor="center")
        self._mi_tree.column("evidence", width=72, anchor="center")
        self._mi_tree.column("rec", width=120, anchor="center")
        ys = ttk.Scrollbar(left, orient="vertical", command=self._mi_tree.yview)
        self._mi_tree.configure(yscrollcommand=ys.set)
        self._mi_tree.pack(side="left", fill="both", expand=True)
        ys.pack(side="right", fill="y")
        self._mi_tree.bind("<<TreeviewSelect>>", lambda _e: self._on_mi_select())

        ev_box = ttk.LabelFrame(right, text="Structured Evidence", padding=8)
        ev_box.pack(fill="both", expand=True)
        self._mi_evidence_var = tk.StringVar(value="Select a feature.")
        ttk.Label(
            ev_box,
            textvariable=self._mi_evidence_var,
            justify="left",
            font=("Consolas", 10),
        ).pack(anchor="nw")
        self._mi_rows_by_feature: dict[str, dict[str, Any]] = {}

    def _build_confidence_tab(self) -> None:
        """Confidence subsystem: Models + Labels sub-tabs."""
        root = self._confidence_tab
        for child in root.winfo_children():
            child.destroy()

        ttk.Label(root, text="Confidence", font=SECTION_FONT).pack(anchor="w")
        ttk.Label(
            root,
            text=(
                "Lab-owned classifiers · Market Outcomes + Replay-Based Outcomes · "
                "Seen data only · not Model Registry"
            ),
            foreground=COL_MUTED,
        ).pack(anchor="w", pady=(0, 8))

        conf_nb = ttk.Notebook(root)
        conf_nb.pack(fill="both", expand=True)
        models_root = ttk.Frame(conf_nb, padding=4)
        labels_root = ttk.Frame(conf_nb, padding=4)
        conf_nb.add(models_root, text="Models")
        conf_nb.add(labels_root, text="Labels")
        self._conf_sub_nb = conf_nb

        from .model_lab_confidence_labels_panel import ModelLabConfidenceLabelsPanel

        self._conf_labels_panel = ModelLabConfidenceLabelsPanel(
            labels_root,
            chart_dir=self.chart_dir,
            on_status=lambda s: self._set_status_bar(status=str(s or "")),
        )
        self._conf_labels_panel.pack(fill="both", expand=True)

        self._build_confidence_models_pane(models_root)

    def _build_confidence_models_pane(self, root: ttk.Frame) -> None:
        """Existing Confidence Models / Evaluate / Inference UI."""
        # Reserve Actions + Inference at the bottom first so expand=True
        # sections above cannot squeeze buttons to zero height.
        bottom = ttk.Frame(root)
        bottom.pack(side="bottom", fill="x")

        actions = ttk.LabelFrame(bottom, text="Actions", padding=8)
        actions.pack(fill="x")
        btn_row = ttk.Frame(actions)
        btn_row.pack(fill="x")
        ttk.Button(btn_row, text="Refresh", command=self._refresh_confidence_tab).pack(
            side="left", padx=(0, 6)
        )
        self._conf_create_ds_btn = ttk.Button(
            btn_row,
            text="Build Confidence Dataset",
            command=self._on_create_confidence_dataset,
        )
        self._conf_create_ds_btn.pack(side="left", padx=(0, 6))
        self._conf_eval_btn = ttk.Button(
            btn_row, text="Evaluate", command=self._on_evaluate_confidence_model
        )
        self._conf_eval_btn.pack(side="left", padx=(0, 6))
        self._conf_inf_run_btn = ttk.Button(
            btn_row,
            text="Run Inference",
            command=self._on_run_confidence_inference,
        )
        self._conf_inf_run_btn.pack(side="left", padx=(0, 6))
        self._conf_link_btn = ttk.Button(
            btn_row,
            text="Link Active Model",
            command=self._on_link_active_confidence_model,
        )
        self._conf_link_btn.pack(side="left", padx=(0, 6))
        self._conf_validate_btn = ttk.Button(
            btn_row,
            text="Validate Mapping",
            command=self._on_validate_confidence_mapping,
        )
        self._conf_validate_btn.pack(side="left", padx=(0, 6))
        self._conf_delete_btn = ttk.Button(
            btn_row, text="Delete", command=self._on_delete_confidence_model
        )
        self._conf_delete_btn.pack(side="left", padx=(0, 6))

        prog_row = ttk.Frame(actions)
        prog_row.pack(fill="x", pady=(8, 0))
        self._conf_progress = ttk.Progressbar(
            prog_row, length=220, mode="determinate", maximum=100
        )
        self._conf_progress.pack(side="left", padx=(0, 8))
        self._conf_progress_pct = tk.StringVar(value="")
        ttk.Label(
            prog_row, textvariable=self._conf_progress_pct, width=5, font=("Consolas", 9)
        ).pack(side="left", padx=(0, 8))
        self._conf_action_status = tk.StringVar(value="")
        ttk.Label(
            prog_row, textvariable=self._conf_action_status, foreground=COL_MUTED
        ).pack(side="left", fill="x", expand=True)

        self._conf_create_btn = self._conf_create_ds_btn
        self._conf_progress_queue: queue.Queue[dict[str, Any]] | None = None

        self._conf_inf_title_var = tk.StringVar(value="Inference · select a model")
        inf_box = ttk.LabelFrame(bottom, text="", padding=8)
        inf_box.pack(fill="x", pady=(8, 0))
        ttk.Label(
            inf_box,
            textvariable=self._conf_inf_title_var,
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(0, 4))
        self._conf_inf_vars = {
            "status": tk.StringVar(value="Not Run"),
            "rows": tk.StringVar(value="—"),
            "pos_neg": tk.StringVar(value="—"),
            "model": tk.StringVar(value="—"),
            "threshold": tk.StringVar(value="—"),
            "completed": tk.StringVar(value="—"),
            "validation": tk.StringVar(value="—"),
            "detail": tk.StringVar(value=""),
        }
        inf_grid = ttk.Frame(inf_box)
        inf_grid.pack(fill="x")
        for i, (label, key) in enumerate(
            (
                ("Status", "status"),
                ("Rows Scored", "rows"),
                ("Hit=1 / Hit=0", "pos_neg"),
                ("Model", "model"),
                ("Operating Threshold", "threshold"),
                ("Completed", "completed"),
                ("Validation", "validation"),
            )
        ):
            r, c = divmod(i, 4)
            ttk.Label(inf_grid, text=f"{label}:", foreground=COL_MUTED).grid(
                row=r, column=c * 2, sticky="w", padx=(0, 4), pady=2
            )
            ttk.Label(inf_grid, textvariable=self._conf_inf_vars[key]).grid(
                row=r, column=c * 2 + 1, sticky="w", padx=(0, 16), pady=2
            )
        ttk.Label(
            inf_box,
            textvariable=self._conf_inf_vars["detail"],
            foreground=COL_MUTED,
            wraplength=720,
            justify="left",
        ).pack(anchor="w", pady=(4, 4))
        inf_btns = ttk.Frame(inf_box)
        inf_btns.pack(fill="x")
        self._conf_inf_clear_btn = ttk.Button(
            inf_btns,
            text="Clear Inference",
            command=self._on_clear_confidence_inference,
        )
        self._conf_inf_clear_btn.pack(side="left", padx=(0, 6))
        self._conf_inf_progress = ttk.Progressbar(
            inf_btns, length=180, mode="determinate", maximum=100
        )
        self._conf_inf_progress.pack(side="left", padx=(8, 6))
        self._conf_inf_progress_var = tk.StringVar(value="")
        ttk.Label(
            inf_btns,
            textvariable=self._conf_inf_progress_var,
            width=36,
            font=("Consolas", 9),
        ).pack(side="left")
        self._conf_inf_queue: queue.Queue[dict[str, Any]] | None = None
        self._conf_inf_run_btn_actions = self._conf_inf_run_btn

        # —— Pipeline + Context on one row ——
        top_row = ttk.Frame(root)
        top_row.pack(fill="x", pady=(0, 8))
        top_row.columnconfigure(0, weight=1, minsize=280)
        top_row.columnconfigure(1, weight=2, minsize=420)

        pipe = ttk.LabelFrame(top_row, text="Pipeline", padding=8)
        pipe.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._conf_pipeline_var = tk.StringVar(value="—")
        ttk.Label(
            pipe,
            textvariable=self._conf_pipeline_var,
            justify="left",
            font=("Consolas", 9),
        ).pack(anchor="nw", fill="both", expand=True)

        ctx = ttk.LabelFrame(top_row, text="Context", padding=8)
        ctx.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        self._conf_ctx_vars = {
            "regression_model": tk.StringVar(value="—"),
            "training_dataset": tk.StringVar(value="—"),
            "prediction_rows": tk.StringVar(value="—"),
            "prediction_version": tk.StringVar(value="—"),
            "dataset_status": tk.StringVar(value="Not Created"),
            "dataset_rows": tk.StringVar(value="—"),
            "seen_days": tk.StringVar(value="—"),
            "feature_source": tk.StringVar(value="—"),
            "features": tk.StringVar(value="—"),
            "rr_labels": tk.StringVar(value="—"),
            "dataset_created": tk.StringVar(value="—"),
        }
        rows = (
            ("Regression Model", "regression_model"),
            ("Training Dataset", "training_dataset"),
            ("Prediction Dataset Rows", "prediction_rows"),
            ("Prediction Lab Version", "prediction_version"),
            ("Confidence Dataset Status", "dataset_status"),
            ("Confidence Dataset Rows", "dataset_rows"),
            ("Seen Trading Days", "seen_days"),
            ("Feature Source", "feature_source"),
            ("Selected Features", "features"),
            ("Available Classifier Labels", "rr_labels"),
            ("Confidence Dataset Created", "dataset_created"),
        )
        for i, (label, key) in enumerate(rows):
            ttk.Label(ctx, text=f"{label}:", foreground=COL_MUTED).grid(
                row=i // 2, column=(i % 2) * 2, sticky="w", padx=(0, 6), pady=2
            )
            ttk.Label(ctx, textvariable=self._conf_ctx_vars[key]).grid(
                row=i // 2, column=(i % 2) * 2 + 1, sticky="w", padx=(0, 16), pady=2
            )

        self._conf_status_var = self._conf_ctx_vars["dataset_status"]
        self._conf_detail_var = self._conf_action_status

        # —— Confidence Models ——
        models_box = ttk.LabelFrame(root, text="Confidence Models", padding=8)
        models_box.pack(fill="both", expand=True, pady=(0, 8))
        cols = (
            "target",
            "type",
            "status",
            "train",
            "f1",
            "precision",
            "recall",
            "roc_auc",
            "op_thr",
            "created",
            "active",
        )
        tree_wrap = ttk.Frame(models_box)
        tree_wrap.pack(fill="both", expand=True)
        self._conf_models_tree = ttk.Treeview(
            tree_wrap, columns=cols, show="headings", height=6, selectmode="browse"
        )
        headings = {
            "target": ("Target", 120),
            "type": ("Type", 90),
            "status": ("Status", 80),
            "train": ("Train", 88),
            "f1": ("F1", 70),
            "precision": ("Precision", 80),
            "recall": ("Recall", 70),
            "roc_auc": ("ROC AUC", 80),
            "op_thr": ("Op. Thr", 70),
            "created": ("Created", 130),
            "active": ("Active", 55),
        }
        for c, (title, width) in headings.items():
            self._conf_models_tree.heading(c, text=title)
            self._conf_models_tree.column(
                c,
                width=width,
                minwidth=width if c == "train" else 40,
                anchor="center" if c not in ("target", "type") else "w",
                stretch=(c != "train"),
            )
        yscroll = ttk.Scrollbar(
            tree_wrap, orient="vertical", command=self._on_conf_models_scroll
        )
        self._conf_models_tree.configure(yscrollcommand=self._on_conf_models_yscroll_set)
        self._conf_models_tree.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")
        self._conf_models_tree.bind(
            "<Double-1>", self._on_conf_models_tree_double_click
        )
        self._conf_models_tree.bind(
            "<<TreeviewSelect>>", lambda _e: self._on_confidence_model_selected()
        )
        self._conf_models_tree.bind(
            "<Configure>", lambda _e: self.after_idle(self._place_conf_train_buttons)
        )
        self._conf_models_tree.bind(
            "<MouseWheel>", lambda _e: self.after_idle(self._place_conf_train_buttons)
        )
        self._conf_trainable_keys: set[str] = set()
        self._conf_train_row_meta: dict[str, str] = {}
        self._conf_train_buttons: list[ttk.Button] = []
        self._conf_train_busy = False
        self._conf_models_yscroll = yscroll

        eval_cal_row = ttk.Frame(root)
        eval_cal_row.pack(fill="both", expand=True, pady=(0, 8))
        eval_cal_row.columnconfigure(0, weight=3, minsize=360)
        eval_cal_row.columnconfigure(1, weight=2, minsize=260)

        eval_box = ttk.LabelFrame(eval_cal_row, text="Evaluate", padding=8)
        eval_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        self._conf_eval_text = scrolledtext.ScrolledText(
            eval_box, height=8, wrap="word", font=("Consolas", 9)
        )
        self._conf_eval_text.pack(fill="both", expand=True)
        self._conf_eval_text.insert("1.0", "Select a model and click Evaluate.")
        self._conf_eval_text.configure(state="disabled")

        cal_box = ttk.LabelFrame(eval_cal_row, text="Calibration", padding=4)
        cal_box.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        cal_cols = ("band", "rows", "hit_pct")
        self._conf_cal_tree = ttk.Treeview(cal_box, columns=cal_cols, show="headings", height=8)
        self._conf_cal_tree.heading("band", text="Band")
        self._conf_cal_tree.heading("rows", text="Rows")
        self._conf_cal_tree.heading("hit_pct", text="Actual Hit %")
        self._conf_cal_tree.column("band", width=110, anchor="w")
        self._conf_cal_tree.column("rows", width=70, anchor="center")
        self._conf_cal_tree.column("hit_pct", width=100, anchor="center")
        self._conf_cal_tree.pack(fill="both", expand=True)

    def _format_confidence_pipeline(self, st: dict[str, Any]) -> str:
        pipe = st.get("pipeline") if isinstance(st.get("pipeline"), dict) else {}
        reg = pipe.get("regression_model") or {}
        pred = pipe.get("prediction_dataset") or {}
        cds = pipe.get("confidence_dataset") or {}
        models = pipe.get("confidence_models") or []

        def mark(ok: bool, *, stale: bool = False) -> str:
            if stale:
                return "⟳"
            return "✓" if ok else "—"

        lines = [
            f"Regression Model     {mark(bool(reg.get('ok')))}  {reg.get('label') or '—'}",
            "        │",
            f"Prediction Dataset   {mark(bool(pred.get('ok')))}"
            + (f"  {int(pred.get('rows') or 0):,} rows" if pred.get("ok") else ""),
            "        │",
            f"Confidence Dataset   {mark(bool(cds.get('ok')), stale=bool(cds.get('stale')))}"
            + (
                f"  {int(cds.get('rows') or 0):,} rows"
                if cds.get("ok") or cds.get("stale")
                else ""
            ),
            "        │",
            "Confidence Models",
        ]
        for m in models:
            ok = bool(m.get("ok"))
            stale = str(m.get("status") or "").lower() == "stale"
            lines.append(
                f"   {str(m.get('label') or m.get('key')):<12} "
                f"{mark(ok, stale=stale)}"
            )
        return "\n".join(lines)

    def _build_rr_validation_tab(self) -> None:
        """Read-only validation of persisted Reward/Risk classifier labels."""
        root = self._rr_validation_tab
        ttk.Label(root, text="RR Validation", font=SECTION_FONT).pack(anchor="w")
        ttk.Label(
            root,
            text="Pre-flight: are persisted RR labels valid before Build Confidence Dataset?",
            foreground=COL_MUTED,
        ).pack(anchor="w", pady=(0, 8))

        toolbar = ttk.Frame(root)
        toolbar.pack(fill="x", pady=(0, 8))
        ttk.Button(toolbar, text="Refresh", command=self._refresh_rr_validation_tab).pack(
            side="left",
        )
        self._rr_val_note_var = tk.StringVar(value="Open tab or click Refresh to validate.")
        ttk.Label(toolbar, textvariable=self._rr_val_note_var, foreground=COL_MUTED).pack(
            side="left", padx=12,
        )

        scroll = ScrollableFrame(root)
        scroll.pack(fill="both", expand=True)
        body = scroll.inner

        self._rr_val_consistency_var = tk.StringVar(value="—")
        self._rr_val_window_var = tk.StringVar(value="—")
        self._rr_val_consistency_lbl = ttk.Label(
            body,
            textvariable=self._rr_val_consistency_var,
            font=("Segoe UI", 10, "bold"),
        )
        self._rr_val_consistency_lbl.pack(anchor="w", pady=(0, 4))
        self._rr_val_window_lbl = ttk.Label(
            body,
            textvariable=self._rr_val_window_var,
            font=("Segoe UI", 10, "bold"),
        )
        self._rr_val_window_lbl.pack(anchor="w", pady=(0, 10))

        summary_box = ttk.LabelFrame(body, text="Label Summary", padding=8)
        summary_box.pack(fill="x", pady=(0, 8))
        cols = ("metric", "count", "pct")
        self._rr_val_summary_tree = ttk.Treeview(
            summary_box, columns=cols, show="headings", height=6,
        )
        for cid, head, w in (
            ("metric", "Metric", 180),
            ("count", "Count", 100),
            ("pct", "% of Total", 100),
        ):
            self._rr_val_summary_tree.heading(cid, text=head)
            self._rr_val_summary_tree.column(cid, width=w, anchor="center" if cid != "metric" else "w")
        self._rr_val_summary_tree.pack(fill="x")

        balance_box = ttk.LabelFrame(
            body,
            text="Class Balance (Positive % — classifier target decision)",
            padding=8,
        )
        balance_box.pack(fill="x", pady=(0, 8))
        bcols = ("label", "positive", "negative", "positive_pct")
        self._rr_val_balance_tree = ttk.Treeview(
            balance_box, columns=bcols, show="headings", height=4,
        )
        for cid, head, w in (
            ("label", "Label", 100),
            ("positive", "Positive", 90),
            ("negative", "Negative", 90),
            ("positive_pct", "Positive %", 100),
        ):
            self._rr_val_balance_tree.heading(cid, text=head)
            self._rr_val_balance_tree.column(cid, width=w, anchor="center")
        self._rr_val_balance_tree.pack(fill="x")

        rr_box = ttk.LabelFrame(body, text="Reward / Risk Ratio", padding=8)
        rr_box.pack(fill="x", pady=(0, 8))
        self._rr_val_ratio_avg_var = tk.StringVar(value="Average Reward / Risk Ratio : —")
        self._rr_val_ratio_med_var = tk.StringVar(value="Median Reward / Risk Ratio  : —")
        self._rr_val_ratio_p95_var = tk.StringVar(value="95th Percentile             : —")
        self._rr_val_ratio_n_var = tk.StringVar(value="")
        ttk.Label(rr_box, textvariable=self._rr_val_ratio_avg_var).pack(anchor="w")
        ttk.Label(rr_box, textvariable=self._rr_val_ratio_med_var).pack(anchor="w")
        ttk.Label(rr_box, textvariable=self._rr_val_ratio_p95_var).pack(anchor="w")
        ttk.Label(rr_box, textvariable=self._rr_val_ratio_n_var, foreground=COL_MUTED).pack(
            anchor="w", pady=(4, 0),
        )

        prem_box = ttk.LabelFrame(body, text="Premium Band Breakdown", padding=8)
        prem_box.pack(fill="x", pady=(0, 8))
        pcols = ("band", "rows", "rr11", "rr23", "rr12", "rr13", "rr14")
        self._rr_val_premium_tree = ttk.Treeview(
            prem_box, columns=pcols, show="headings", height=6,
        )
        for cid, head, w in (
            ("band", "Premium Band", 100),
            ("rows", "Rows", 60),
            ("rr11", "RR 1:1", 70),
            ("rr23", "RR 2:3", 70),
            ("rr12", "RR 1:2", 70),
            ("rr13", "RR 1:3", 70),
            ("rr14", "RR 1:4", 70),
        ):
            self._rr_val_premium_tree.heading(cid, text=head)
            self._rr_val_premium_tree.column(cid, width=w, anchor="center")
        self._rr_val_premium_tree.pack(fill="x")

        day_box = ttk.LabelFrame(body, text="Trading Day Breakdown", padding=8)
        day_box.pack(fill="both", expand=True, pady=(0, 8))
        dcols = ("day", "rows", "target", "rr11", "rr23", "rr12", "rr13", "rr14")
        self._rr_val_day_tree = ttk.Treeview(
            day_box, columns=dcols, show="headings", height=8,
        )
        for cid, head, w in (
            ("day", "Day", 100),
            ("rows", "Rows", 55),
            ("target", "Path Touch", 80),
            ("rr11", "RR 1:1", 65),
            ("rr23", "RR 2:3", 65),
            ("rr12", "RR 1:2", 65),
            ("rr13", "RR 1:3", 65),
            ("rr14", "RR 1:4", 65),
        ):
            self._rr_val_day_tree.heading(cid, text=head)
            self._rr_val_day_tree.column(cid, width=w, anchor="center")
        day_scroll = ttk.Scrollbar(day_box, orient="vertical", command=self._rr_val_day_tree.yview)
        self._rr_val_day_tree.configure(yscrollcommand=day_scroll.set)
        self._rr_val_day_tree.pack(side="left", fill="both", expand=True)
        day_scroll.pack(side="right", fill="y")

    def _fmt_rr_pct(self, value: Any) -> str:
        if value is None:
            return "—"
        try:
            return f"{float(value):.2f}%"
        except (TypeError, ValueError):
            return "—"

    def _fmt_rr_ratio(self, value: Any) -> str:
        if value is None:
            return "—"
        try:
            return f"{float(value):.2f}"
        except (TypeError, ValueError):
            return "—"

    def _rr_val_trees(self) -> tuple[Any, ...]:
        return (
            self._rr_val_summary_tree,
            self._rr_val_balance_tree,
            self._rr_val_premium_tree,
            self._rr_val_day_tree,
        )

    def _clear_rr_validation_tables(self) -> None:
        for tree in self._rr_val_trees():
            for item in tree.get_children():
                tree.delete(item)
        self._rr_val_ratio_avg_var.set("Average Reward / Risk Ratio : —")
        self._rr_val_ratio_med_var.set("Median Reward / Risk Ratio  : —")
        self._rr_val_ratio_p95_var.set("95th Percentile             : —")
        self._rr_val_ratio_n_var.set("")

    def _refresh_rr_validation_tab(self) -> None:
        from chain_replay_ml.model_lab.rr_validation import load_rr_validation_report

        if self._lab is None:
            self._rr_val_note_var.set("No Research Lab open.")
            return

        report = load_rr_validation_report(self._lab.db_path)
        if report.get("error"):
            self._rr_val_note_var.set(f"Error: {report['error']}")
            return

        if not report.get("available"):
            self._rr_val_note_var.set("No prediction rows — build the Prediction Dataset first.")
            self._clear_rr_validation_tables()
            self._rr_val_consistency_var.set("—")
            self._rr_val_window_var.set("—")
            return

        horizon = report.get("horizon_sec")
        target = report.get("target_column") or "—"
        labeled = int(report.get("labeled_rows") or 0)
        total = int(report.get("total_rows") or 0)
        self._rr_val_note_var.set(
            f"{total:,} predictions · {labeled:,} with RR labels · "
            f"target={target} · horizon={horizon}s"
        )

        cons = report.get("consistency") or {}
        win = report.get("outcome_window") or {}
        win_msg = str(win.get("message") or "")
        self._rr_val_consistency_var.set(
            (
                "✓ RR label consistency passed"
                if cons.get("ok")
                else "❌ RR label consistency failed\n"
                + str(cons.get("message") or "—")
                + "\nPrediction labels are inconsistent."
            )
        )
        self._rr_val_consistency_lbl.configure(
            foreground=COL_OK if cons.get("ok") else COL_WARN,
        )
        self._rr_val_window_var.set(
            (
                "✓ Outcome window validated"
                if win.get("ok")
                else (
                    f"❌ {win_msg}"
                    if "not computed yet" in win_msg.lower()
                    else (
                        "❌ Outcome window mismatch\n"
                        "RR labels are using metrics from different evaluation windows.\n"
                        "Labels may be invalid."
                    )
                )
            )
        )
        self._rr_val_window_lbl.configure(
            foreground=COL_OK if win.get("ok") else COL_WARN,
        )

        self._clear_rr_validation_tables()

        for row in report.get("summary") or []:
            self._rr_val_summary_tree.insert(
                "",
                "end",
                values=(
                    row.get("metric"),
                    f"{int(row.get('count') or 0):,}",
                    self._fmt_rr_pct(row.get("pct")),
                ),
            )

        for row in report.get("class_balance") or []:
            self._rr_val_balance_tree.insert(
                "",
                "end",
                values=(
                    row.get("label"),
                    f"{int(row.get('positive') or 0):,}",
                    f"{int(row.get('negative') or 0):,}",
                    self._fmt_rr_pct(row.get("positive_pct")),
                ),
            )

        rr = report.get("reward_risk") or {}
        self._rr_val_ratio_avg_var.set(
            f"Average Reward / Risk Ratio : {self._fmt_rr_ratio(rr.get('avg'))}"
        )
        self._rr_val_ratio_med_var.set(
            f"Median Reward / Risk Ratio  : {self._fmt_rr_ratio(rr.get('median'))}"
        )
        self._rr_val_ratio_p95_var.set(
            f"95th Percentile             : {self._fmt_rr_ratio(rr.get('p95'))}"
        )
        n_rr = int(rr.get("n") or 0)
        self._rr_val_ratio_n_var.set(
            f"Based on {n_rr:,} rows with maximum_profit / maximum_drawdown "
            "(drawdown > 0)"
            if n_rr
            else "No rows with valid reward/risk ratio"
        )

        for row in report.get("premium_bands") or []:
            self._rr_val_premium_tree.insert(
                "",
                "end",
                values=(
                    row.get("band"),
                    f"{int(row.get('rows') or 0):,}",
                    self._fmt_rr_pct(row.get("rr_1_1_pct")),
                    self._fmt_rr_pct(row.get("rr_2_3_pct")),
                    self._fmt_rr_pct(row.get("rr_1_2_pct")),
                    self._fmt_rr_pct(row.get("rr_1_3_pct")),
                    self._fmt_rr_pct(row.get("rr_1_4_pct")),
                ),
            )

        for row in report.get("trading_days") or []:
            self._rr_val_day_tree.insert(
                "",
                "end",
                values=(
                    row.get("trading_day"),
                    f"{int(row.get('rows') or 0):,}",
                    self._fmt_rr_pct(row.get("target_hit_pct")),
                    self._fmt_rr_pct(row.get("rr_1_1_pct")),
                    self._fmt_rr_pct(row.get("rr_2_3_pct")),
                    self._fmt_rr_pct(row.get("rr_1_2_pct")),
                    self._fmt_rr_pct(row.get("rr_1_3_pct")),
                    self._fmt_rr_pct(row.get("rr_1_4_pct")),
                ),
            )

    def _studio_app(self) -> Any | None:
        w: Any = self.master
        while w is not None:
            if hasattr(w, "create_model_panel") and hasattr(w, "_show_page"):
                return w
            w = getattr(w, "master", None)
        return None

    def _selected_confidence_model_key(self) -> str | None:
        tree = getattr(self, "_conf_models_tree", None)
        if tree is None:
            return None
        sel = tree.selection()
        if not sel:
            return None
        iid = str(sel[0] or "")
        from chain_replay_ml.model_lab.confidence_manifest import TARGET_BY_KEY

        if iid in TARGET_BY_KEY:
            return iid
        vals = tree.item(sel[0], "values")
        if not vals:
            return None
        label = str(vals[0])
        from chain_replay_ml.model_lab.confidence_manifest import CONFIDENCE_TARGETS

        for spec in CONFIDENCE_TARGETS:
            if spec["label"] == label:
                return spec["key"]
        return None

    def _on_confidence_model_selected(self) -> None:
        key = self._selected_confidence_model_key()
        if not key:
            self._refresh_confidence_inference_panel({})
            return
        lab_path = getattr(self._lab, "db_path", None) if self._lab else None
        if not lab_path:
            return
        from chain_replay_ml.model_lab.confidence import inference_status

        try:
            inf = inference_status(lab_path, key)
        except Exception as exc:
            inf = {"status": "not_run", "error": str(exc), "can_run": False, "model_key": key}
        self._refresh_confidence_inference_panel(inf)

    def _refresh_confidence_tab(self) -> None:
        from chain_replay_ml.model_lab.confidence import confidence_status

        if hasattr(self, "_conf_labels_panel"):
            lab_path = getattr(self._lab, "db_path", None) if self._lab else None
            try:
                self._conf_labels_panel.refresh_for_lab(
                    lab_db_path=lab_path,
                    model_name=getattr(self, "model_name", "") or "",
                )
            except Exception:
                pass

        if not hasattr(self, "_conf_ctx_vars"):
            return
        lab_path = getattr(self._lab, "db_path", None) if self._lab else None
        if not lab_path:
            for v in self._conf_ctx_vars.values():
                v.set("—")
            self._conf_ctx_vars["dataset_status"].set("No Research Lab open")
            if hasattr(self, "_conf_pipeline_var"):
                self._conf_pipeline_var.set("Open a Research Lab to see pipeline status.")
            for iid in self._conf_models_tree.get_children():
                self._conf_models_tree.delete(iid)
            self._conf_trainable_keys = set()
            self._conf_train_row_meta = {}
            self._clear_conf_train_buttons()
            self._conf_create_ds_btn.configure(state="disabled")
            if hasattr(self, "_conf_validate_btn"):
                self._conf_validate_btn.configure(state="disabled")
            if hasattr(self, "_conf_inf_vars"):
                self._refresh_confidence_inference_panel({})
            return

        st = confidence_status(lab_path, data_dir=chart_data_dir(self.chart_dir))
        if hasattr(self, "_conf_pipeline_var"):
            self._conf_pipeline_var.set(self._format_confidence_pipeline(st))

        self._conf_ctx_vars["regression_model"].set(str(st.get("regression_model") or "—"))
        self._conf_ctx_vars["training_dataset"].set(str(st.get("training_dataset") or "—"))
        self._conf_ctx_vars["prediction_rows"].set(f"{int(st.get('prediction_rows') or 0):,}")
        ver = st.get("prediction_lab_version")
        self._conf_ctx_vars["prediction_version"].set(f"v{ver}" if ver is not None else "—")
        ds_status = str(st.get("confidence_dataset_status") or "not_created")
        status_map = {
            "ready": "Ready",
            "stale": "Stale — rebuild required",
            "not_created": "Not Created",
        }
        self._conf_ctx_vars["dataset_status"].set(status_map.get(ds_status, ds_status))
        ds_ready = ds_status in ("ready", "stale")
        self._conf_ctx_vars["dataset_rows"].set(
            f"{int(st.get('confidence_dataset_rows') or 0):,}" if ds_ready else "—"
        )
        seen_days = st.get("seen_trading_days")
        self._conf_ctx_vars["seen_days"].set(
            str(int(seen_days)) if seen_days else "—"
        )
        self._conf_ctx_vars["feature_source"].set(
            str(st.get("feature_source") or "—")
        )
        feat_n = st.get("feature_count") or st.get("selected_features")
        self._conf_ctx_vars["features"].set(
            str(int(feat_n)) if feat_n else "—"
        )
        n_labels = st.get("available_classifier_labels")
        if ds_ready and n_labels:
            labels = st.get("rr_labels") or {}
            bits = []
            if labels.get("target_reached"):
                bits.append("Path Touch")
            for key, mark in (
                ("rr_1_1_hit", "RR 1:1"),
                ("rr_2_3_hit", "RR 2:3"),
                ("rr_1_2_hit", "RR 1:2"),
                ("rr_1_3_hit", "RR 1:3"),
                ("rr_1_4_hit", "RR 1:4"),
            ):
                if labels.get(key):
                    bits.append(mark)
            self._conf_ctx_vars["rr_labels"].set(
                f"{int(n_labels)} · " + ", ".join(bits) if bits else str(n_labels)
            )
        elif st.get("rr_labels_available"):
            self._conf_ctx_vars["rr_labels"].set("Available in Prediction Dataset")
        else:
            self._conf_ctx_vars["rr_labels"].set("—")
        self._conf_ctx_vars["dataset_created"].set(
            str(st.get("confidence_dataset_created_at") or "—") if ds_ready else "—"
        )

        for iid in self._conf_models_tree.get_children():
            self._conf_models_tree.delete(iid)

        def _pct(v: Any) -> str:
            if v is None:
                return "—"
            try:
                return f"{float(v):.1f}%"
            except (TypeError, ValueError):
                return "—"

        def _auc(v: Any) -> str:
            if v is None:
                return "—"
            try:
                return f"{float(v):.3f}"
            except (TypeError, ValueError):
                return "—"

        available_cols: set[str] | None = None
        ds_ok = ds_status in ("ready", "stale")
        if ds_ok and lab_path:
            try:
                from chain_replay_ml.model_lab.confidence_dataset import (
                    confidence_dataset_paths,
                )
                from chain_replay_ml.training.dataset_loader import parquet_column_names

                pq = confidence_dataset_paths(lab_path)["parquet"]
                if os.path.isfile(pq):
                    available_cols = set(parquet_column_names(pq))
            except Exception:
                available_cols = None

        trainable: set[str] = set()
        train_meta: dict[str, str] = {}
        for row in st.get("models") or []:
            created = str(row.get("created_at") or "—")
            if created != "—" and len(created) > 19:
                created = created[:19].replace("T", " ")
            readyish = str(row.get("status") or "") in ("ready", "stale")
            thr = row.get("operating_threshold")
            thr_s = "—"
            if readyish and thr is not None:
                try:
                    thr_s = f"{float(thr):.2f}"
                except (TypeError, ValueError):
                    thr_s = "—"
            key = str(row.get("key") or "")
            col_name = str(row.get("column") or "")
            label_ok = ds_ok and (
                available_cols is None
                or not col_name
                or col_name in available_cols
            )
            if label_ok and key:
                trainable.add(key)
                train_meta[key] = "retrain" if readyish else "train"
            elif ds_ok and key:
                train_meta[key] = "need_ds"
            else:
                train_meta[key] = "blocked"
            # Cell left blank — real ttk.Button is placed over this column.
            self._conf_models_tree.insert(
                "",
                "end",
                iid=key,
                values=(
                    row.get("label") or row.get("key"),
                    row.get("type") or row.get("label") or "—",
                    row.get("status_display")
                    or str(row.get("status") or "Missing").replace("_", " ").title(),
                    "",
                    _pct(row.get("f1_pct")) if readyish else "—",
                    _pct(row.get("precision_pct")) if readyish else "—",
                    _pct(row.get("recall_pct")) if readyish else "—",
                    _auc(row.get("roc_auc")) if readyish else "—",
                    thr_s,
                    created if readyish else "—",
                    "●" if row.get("active") and row.get("status") == "ready" else "",
                ),
            )
        self._conf_trainable_keys = trainable
        self._conf_train_row_meta = train_meta
        self.after_idle(self._place_conf_train_buttons)

        has_pred = bool(st.get("has_prediction_dataset"))
        can_train = ds_status == "ready"
        self._conf_create_ds_btn.configure(state="normal" if has_pred else "disabled")
        self._conf_eval_btn.configure(state="normal" if can_train or ds_status == "stale" else "disabled")
        self._conf_link_btn.configure(state="normal" if can_train else "disabled")
        self._conf_validate_btn.configure(
            state="normal" if can_train or ds_status == "stale" else "disabled"
        )
        self._conf_delete_btn.configure(state="normal" if can_train or ds_status == "stale" else "disabled")
        if st.get("regression_stale"):
            self._conf_action_status.set(
                "Regression model changed — rebuild Confidence Dataset."
            )
        elif st.get("message") and not has_pred:
            self._conf_action_status.set(str(st["message"]))
        else:
            self._conf_action_status.set("")

        # Keep selection if possible; otherwise select Target Hit when ready
        prev = self._selected_confidence_model_key()
        if prev and self._conf_models_tree.exists(prev):
            self._conf_models_tree.selection_set(prev)
            self._conf_models_tree.focus(prev)
        else:
            active_key = st.get("active_model_key") or "target_hit"
            if self._conf_models_tree.exists(active_key):
                self._conf_models_tree.selection_set(active_key)
                self._conf_models_tree.focus(active_key)
        self._on_confidence_model_selected()

    def _refresh_confidence_inference_panel(self, inf: dict[str, Any]) -> None:
        if not hasattr(self, "_conf_inf_vars"):
            return
        label = str(inf.get("model_label") or "")
        if label:
            self._conf_inf_title_var.set(f"Inference · {label}")
        else:
            self._conf_inf_title_var.set("Inference · select a model")
        status = str(inf.get("status_display") or inf.get("status") or "Not Run")
        self._conf_inf_vars["status"].set(status)
        rows = inf.get("rows")
        pred_n = inf.get("prediction_rows")
        if rows is not None and pred_n is not None:
            self._conf_inf_vars["rows"].set(f"{int(rows):,} / {int(pred_n):,}")
        elif rows is not None:
            self._conf_inf_vars["rows"].set(f"{int(rows):,}")
        else:
            self._conf_inf_vars["rows"].set("—")
        pos = inf.get("positive")
        neg = inf.get("negative")
        if pos is not None and neg is not None:
            self._conf_inf_vars["pos_neg"].set(f"{int(pos):,} / {int(neg):,}")
        else:
            self._conf_inf_vars["pos_neg"].set("—")
        self._conf_inf_vars["model"].set(label or "—")
        thr = inf.get("threshold")
        if thr is not None and inf.get("operating_threshold_set"):
            try:
                self._conf_inf_vars["threshold"].set(f"{float(thr):.2f}")
            except (TypeError, ValueError):
                self._conf_inf_vars["threshold"].set("—")
        else:
            self._conf_inf_vars["threshold"].set("Not set")
        self._conf_inf_vars["completed"].set(
            str(inf.get("completed_at_display") or "—")
        )
        val = inf.get("validation") if isinstance(inf.get("validation"), dict) else {}
        if val:
            self._conf_inf_vars["validation"].set(
                "Passed" if val.get("passed") else "Failed"
            )
        else:
            self._conf_inf_vars["validation"].set("—")
        detail_bits: list[str] = []
        if inf.get("stale_reason"):
            detail_bits.append(str(inf["stale_reason"]))
        if inf.get("operating_threshold_error"):
            detail_bits.append(str(inf["operating_threshold_error"]))
        if inf.get("error"):
            detail_bits.append(str(inf["error"]))
        self._conf_inf_vars["detail"].set(" · ".join(detail_bits))

        action = str(inf.get("action_label") or "Run Inference")
        can_run = bool(inf.get("can_run") and inf.get("model_key"))
        if hasattr(self, "_conf_inf_run_btn"):
            self._conf_inf_run_btn.configure(
                text=action,
                state="normal" if can_run else "disabled",
            )
        if hasattr(self, "_conf_inf_clear_btn"):
            scored = bool(inf.get("has_predictions")) or str(inf.get("status") or "") in (
                "completed",
                "out_of_date",
                "failed",
            )
            self._conf_inf_clear_btn.configure(
                state="normal" if scored and inf.get("model_key") else "disabled"
            )

    def _format_confidence_dataset_report(self, result: dict[str, Any]) -> str:
        """Row counts + outcome for Build Confidence Dataset dialogs / Context."""
        rep = result.get("report") if isinstance(result.get("report"), dict) else {}

        def _n(key: str, fallback: Any = None) -> str:
            val = rep.get(key, fallback)
            if val is None or val == "":
                return "—"
            try:
                return f"{int(val):,}"
            except (TypeError, ValueError):
                return str(val)

        rows_block = (
            f"Training Dataset Rows : {_n('dataset_rows')}\n"
            f"Seen Prediction Rows  : {_n('seen_prediction_rows')}\n"
            f"Unseen Prediction Rows: {_n('unseen_prediction_rows', 0)} (ignored)\n"
            f"Matched               : {_n('matched')}\n"
            f"Missing               : {_n('missing')}\n"
            f"Dropped unmatched     : {_n('dropped_unmatched', 0)}\n"
            f"Duplicates            : {_n('duplicates')}"
        )
        if result.get("ok"):
            labels = ", ".join(result.get("labels") or []) or "—"
            kept = rep.get("dataset_rows_after_trim")
            if kept is None:
                kept = result.get("row_count")
            kept_s = "—"
            try:
                if kept is not None:
                    kept_s = f"{int(kept):,}"
            except (TypeError, ValueError):
                kept_s = str(kept)
            warn = rep.get("null_label_warning") or result.get("warning")
            warn_block = f"\n\n⚠ {warn}" if warn else ""
            return (
                "Confidence Dataset ready\n\n"
                f"{rows_block}\n"
                f"Rows kept             : {kept_s}\n"
                f"Columns Added         : {int(rep.get('columns_added') or len(result.get('labels') or []) or 0)}\n"
                f"Saved As              : confidence_dataset\n"
                f"Labels                : {labels}"
                f"{warn_block}"
            )
        err = str(result.get("error") or "Failed").strip()
        return f"{err}\n\n{rows_block}"

    def _apply_confidence_create_row_counts(self, result: dict[str, Any]) -> None:
        """Update Context top panel with dataset row counts from create attempt."""
        if not hasattr(self, "_conf_ctx_vars"):
            return
        rep = result.get("report") if isinstance(result.get("report"), dict) else {}
        ds_rows = rep.get("dataset_rows")
        if ds_rows is None and result.get("row_count") is not None:
            ds_rows = result.get("row_count")
        if ds_rows is not None:
            try:
                self._conf_ctx_vars["dataset_rows"].set(f"{int(ds_rows):,}")
            except (TypeError, ValueError):
                self._conf_ctx_vars["dataset_rows"].set(str(ds_rows))
        if result.get("ok"):
            self._conf_ctx_vars["dataset_status"].set("Ready")
        else:
            missing = rep.get("missing")
            matched = rep.get("matched")
            status = "Create failed"
            try:
                if missing is not None and int(missing) > 0:
                    status = f"Aborted · Missing {int(missing):,}"
                elif matched is not None:
                    status = f"Aborted · Matched {int(matched):,}"
            except (TypeError, ValueError):
                pass
            self._conf_ctx_vars["dataset_status"].set(status)
        train_name = rep.get("training_dataset")
        if train_name:
            self._conf_ctx_vars["training_dataset"].set(str(train_name))

    def _set_confidence_progress(
        self,
        percent: float | None,
        message: str = "",
        *,
        dataset_rows: Any = None,
        active: bool = True,
    ) -> None:
        """Update Confidence tab progress bar + window status bar."""
        if hasattr(self, "_conf_progress"):
            if percent is None:
                self._conf_progress.configure(value=0.0)
                self._conf_progress_pct.set("")
            else:
                pct = max(0.0, min(100.0, float(percent)))
                self._conf_progress.configure(value=pct)
                self._conf_progress_pct.set(f"{pct:.0f}%")
        if message and hasattr(self, "_conf_action_status"):
            self._conf_action_status.set(message)
        if dataset_rows is not None and hasattr(self, "_conf_ctx_vars"):
            try:
                self._conf_ctx_vars["dataset_rows"].set(f"{int(dataset_rows):,}")
            except (TypeError, ValueError):
                self._conf_ctx_vars["dataset_rows"].set(str(dataset_rows))
        if hasattr(self, "_set_status_bar"):
            self._set_status_bar(
                status="Confidence Dataset" if active else "Ready",
                percent=percent if active else None,
                detail=message,
                active=active,
            )

    def _poll_confidence_create_progress(self) -> None:
        q = getattr(self, "_conf_progress_queue", None)
        if q is None:
            return
        try:
            while True:
                msg = q.get_nowait()
                if msg.get("_done"):
                    self._conf_progress_queue = None
                    result = msg.get("result") or {"ok": False, "error": "Unknown error"}
                    self._on_confidence_dataset_done(result)
                    return
                self._set_confidence_progress(
                    msg.get("percent"),
                    str(msg.get("message") or ""),
                    dataset_rows=msg.get("dataset_rows"),
                    active=True,
                )
        except queue.Empty:
            pass
        self.after(120, self._poll_confidence_create_progress)

    def _on_create_confidence_dataset(self) -> None:
        from chain_replay_ml.model_lab.confidence import create_confidence_dataset

        lab_path = getattr(self._lab, "db_path", None) if self._lab else None
        if not lab_path:
            messagebox.showwarning("Confidence Model", "Open a Research Lab first.", parent=self)
            return
        data_dir = chart_data_dir(self.chart_dir)
        if hasattr(self, "_conf_ctx_vars"):
            pred_n = self._conf_ctx_vars["prediction_rows"].get()
            self._conf_ctx_vars["dataset_status"].set("Creating…")
            try:
                from chain_replay_ml.model_lab.confidence_dataset import (
                    resolve_training_dataset,
                )

                resolved = resolve_training_dataset(lab_path, data_dir=data_dir)
                train_rows = int((resolved.get("meta") or {}).get("row_count") or 0)
                train_name = str(resolved.get("dataset_name") or "")
                if train_name:
                    self._conf_ctx_vars["training_dataset"].set(train_name)
                if train_rows > 0:
                    self._conf_ctx_vars["dataset_rows"].set(f"{train_rows:,}")
            except Exception:
                train_rows = 0
                pred_n = self._conf_ctx_vars["prediction_rows"].get()
        else:
            train_rows = 0
            pred_n = "—"

        self._conf_create_ds_btn.configure(state="disabled")
        self._set_confidence_progress(
            0,
            (
                f"Starting… Training rows: {train_rows:,}  ·  Prediction rows: {pred_n}"
                if train_rows
                else f"Starting Confidence Dataset…  Prediction rows: {pred_n}"
            ),
            dataset_rows=train_rows or None,
            active=True,
        )

        prog_q: queue.Queue[dict[str, Any]] = queue.Queue()
        self._conf_progress_queue = prog_q

        def on_progress(payload: dict[str, Any]) -> None:
            prog_q.put(dict(payload))

        def worker() -> None:
            try:
                result = create_confidence_dataset(
                    lab_path, data_dir=data_dir, on_progress=on_progress
                )
            except Exception as exc:
                result = {"ok": False, "error": str(exc)}
            prog_q.put({"_done": True, "result": result})

        threading.Thread(target=worker, daemon=True, name="conf-dataset").start()
        self.after(80, self._poll_confidence_create_progress)

    def _on_confidence_dataset_done(self, result: dict[str, Any]) -> None:
        self._conf_create_ds_btn.configure(state="normal")
        self._conf_progress_queue = None
        ok = bool(result.get("ok"))
        self._apply_confidence_create_row_counts(result)
        msg = self._format_confidence_dataset_report(result)
        if not ok:
            if hasattr(self, "_conf_progress"):
                self._conf_progress.configure(value=0.0)
                self._conf_progress_pct.set("")
            self._set_status_bar(
                status="Ready",
                percent=None,
                detail="Confidence Dataset aborted",
                active=False,
            )
            self._conf_action_status.set(
                f"Create failed · Training rows: "
                f"{(result.get('report') or {}).get('dataset_rows', '—')}"
            )
            messagebox.showerror("Build Confidence Dataset", msg, parent=self)
            self._refresh_confidence_tab()
            self._apply_confidence_create_row_counts(result)
            return

        row_n = int(result.get("row_count") or 0)
        if hasattr(self, "_conf_progress"):
            self._conf_progress.configure(value=100.0)
            self._conf_progress_pct.set("100%")
        self._set_status_bar(
            status="Ready",
            percent=100.0,
            detail=f"Confidence Dataset ready · {row_n:,} rows",
            active=False,
        )
        self._refresh_confidence_tab()
        self._apply_confidence_create_row_counts(result)
        self._conf_action_status.set(f"Confidence Dataset ready · {row_n:,} rows")
        messagebox.showinfo("Build Confidence Dataset", msg, parent=self)

    def _on_conf_models_scroll(self, *args: Any) -> None:
        self._conf_models_tree.yview(*args)
        self.after_idle(self._place_conf_train_buttons)

    def _on_conf_models_yscroll_set(self, first: str, last: str) -> None:
        if hasattr(self, "_conf_models_yscroll"):
            self._conf_models_yscroll.set(first, last)
        self.after_idle(self._place_conf_train_buttons)

    def _clear_conf_train_buttons(self) -> None:
        for btn in getattr(self, "_conf_train_buttons", []):
            try:
                btn.destroy()
            except tk.TclError:
                pass
        self._conf_train_buttons = []

    def _place_conf_train_buttons(self) -> None:
        """Overlay real Train buttons on the Train column (Treeview cannot host widgets)."""
        if not hasattr(self, "_conf_models_tree"):
            return
        tree = self._conf_models_tree
        self._clear_conf_train_buttons()
        if getattr(self, "_conf_train_busy", False):
            return
        meta = getattr(self, "_conf_train_row_meta", {}) or {}
        for iid in tree.get_children():
            try:
                bbox = tree.bbox(iid, "train")
            except tk.TclError:
                continue
            if not bbox:
                continue
            x, y, w, h = bbox
            kind = str(meta.get(iid) or "blocked")
            if kind == "train":
                text, state = "Train", "normal"
            elif kind == "retrain":
                text, state = "Retrain", "normal"
            elif kind == "need_ds":
                text, state = "Need DS", "normal"
            else:
                text, state = "Train", "disabled"

            def _cmd(k: str = str(iid), kind_s: str = kind) -> None:
                if kind_s in ("train", "retrain"):
                    self._train_confidence_model_key(k)
                elif kind_s == "need_ds":
                    messagebox.showinfo(
                        "Train Confidence Model",
                        "This target is not in the Confidence Dataset yet.\n\n"
                        "1. Confidence → Labels → Build Confidence Labels\n"
                        "2. Models → Build Confidence Dataset\n"
                        "3. Click Train on this row",
                        parent=self,
                    )

            btn = ttk.Button(tree, text=text, command=_cmd)
            btn.configure(state=state)
            btn.place(
                in_=tree,
                x=x + 2,
                y=y + 1,
                width=max(int(w) - 4, 72),
                height=max(int(h) - 2, 20),
            )
            self._conf_train_buttons.append(btn)

    def _on_conf_models_tree_double_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        tree = self._conf_models_tree
        col = tree.identify_column(event.x)
        if str(tree.column(col, "id") or "") == "train":
            return
        self._on_evaluate_confidence_model()

    def _on_train_confidence_model(self) -> None:
        """Train the selected table row (no popup). Prefer the Train column button."""
        key = self._selected_confidence_model_key()
        if not key:
            messagebox.showinfo(
                "Train Confidence Model",
                "Use the Train button on a row in the Confidence Models table.",
                parent=self,
            )
            return
        self._train_confidence_model_key(key)

    def _train_confidence_model_key(self, model_key: str) -> None:
        from chain_replay_ml.model_lab.confidence_train import train_confidence_model

        lab_path = getattr(self._lab, "db_path", None) if self._lab else None
        if not lab_path:
            messagebox.showwarning(
                "Confidence Model", "Open a Research Lab first.", parent=self
            )
            return
        key = str(model_key or "").strip()
        if not key:
            return
        if self._conf_train_busy:
            return
        if key not in getattr(self, "_conf_trainable_keys", set()):
            messagebox.showinfo(
                "Train Confidence Model",
                "This target cannot be trained yet.\n"
                "Rebuild the Confidence Dataset so the label column is present.",
                parent=self,
            )
            return

        label = key
        try:
            vals = self._conf_models_tree.item(key, "values") or ()
            if vals:
                label = str(vals[0] or key)
        except tk.TclError:
            pass

        self._conf_train_busy = True
        self._clear_conf_train_buttons()
        self._conf_action_status.set(f"Training {label}…")
        self._set_status_bar(
            status=f"Training Confidence Model · {label}",
            percent=0.0,
            detail="",
            active=True,
        )

        def worker() -> None:
            try:
                result = train_confidence_model(lab_path, key)
            except Exception as exc:
                result = {"ok": False, "error": str(exc)}
            self.after(0, lambda: done(result))

        def done(result: dict[str, Any]) -> None:
            self._conf_train_busy = False
            if not result.get("ok"):
                self._conf_action_status.set("Train failed")
                self._set_status_bar(
                    status="Ready",
                    percent=None,
                    detail=str(result.get("error") or "Train failed"),
                    active=False,
                )
                self._refresh_confidence_tab()
                messagebox.showerror(
                    "Train Confidence Model",
                    str(result.get("error") or "Failed"),
                    parent=self,
                )
                return
            metrics = result.get("metrics") or {}
            self._refresh_confidence_tab()
            self._set_status_bar(
                status="Ready",
                percent=100.0,
                detail=f"{result.get('label') or label} ready",
                active=False,
            )
            self._conf_action_status.set(
                f"{result.get('label') or label} ready · F1 {metrics.get('f1_pct')}%"
            )
            messagebox.showinfo(
                "Train Confidence Model",
                f"{result.get('label') or label} ready\n\n"
                f"F1: {metrics.get('f1_pct')}%\n"
                f"Precision: {metrics.get('precision_pct')}%\n"
                f"Recall: {metrics.get('recall_pct')}%\n"
                f"ROC AUC: {metrics.get('roc_auc')}",
                parent=self,
            )

        threading.Thread(target=worker, daemon=True, name="conf-train").start()

    def _on_evaluate_confidence_model(self) -> None:
        from chain_replay_ml.model_lab.confidence_train import evaluate_confidence_model

        lab_path = getattr(self._lab, "db_path", None) if self._lab else None
        if not lab_path:
            return
        key = self._selected_confidence_model_key()
        if not key:
            messagebox.showinfo(
                "Evaluate", "Select a Confidence Model in the table.", parent=self
            )
            return
        result = evaluate_confidence_model(lab_path, key)
        if not result.get("ok"):
            messagebox.showerror(
                "Model Evaluation",
                str(result.get("error") or "Failed"),
                parent=self,
            )
            return
        self._open_confidence_evaluation_dialog(result)
        # Keep tab summary in sync with the dialog
        self._fill_confidence_eval_panel(result)

    def _fill_confidence_eval_panel(self, result: dict[str, Any]) -> None:
        metrics = result.get("metrics") or {}
        eval_block = result.get("evaluation") or {}
        core = eval_block.get("core_metrics") or {}
        confusion = (
            (eval_block.get("confusion") if isinstance(eval_block.get("confusion"), dict) else None)
            or metrics.get("confusion")
            or result.get("confusion")
            or {}
        )
        lines = [
            f"{result.get('label')}  ·  {result.get('status')}",
            "",
            f"F1           {core.get('f1_pct', metrics.get('f1_pct'))}%",
            f"Precision    {core.get('precision_pct', metrics.get('precision_pct'))}%",
            f"Recall       {core.get('recall_pct', metrics.get('recall_pct'))}%",
            f"ROC AUC      {core.get('roc_auc', metrics.get('roc_auc'))}",
            f"PR AUC       {core.get('pr_auc', metrics.get('pr_auc'))}",
            f"Brier        {core.get('brier_score', metrics.get('brier_score'))}",
            f"Accuracy     {core.get('accuracy_pct', metrics.get('accuracy_pct'))}%",
            "",
            "Confusion Matrix",
            f"  TN={confusion.get('tn')}  FP={confusion.get('fp')}",
            f"  FN={confusion.get('fn')}  TP={confusion.get('tp')}",
            "",
            f"Created: {result.get('created_at') or '—'}",
            "",
            "(Double-click a model for the full Model Evaluation dialog)",
        ]
        if hasattr(self, "_conf_eval_text"):
            self._conf_eval_text.configure(state="normal")
            self._conf_eval_text.delete("1.0", "end")
            self._conf_eval_text.insert("1.0", "\n".join(lines))
            self._conf_eval_text.configure(state="disabled")

        if hasattr(self, "_conf_cal_tree"):
            for iid in self._conf_cal_tree.get_children():
                self._conf_cal_tree.delete(iid)
            for row in result.get("calibration") or []:
                hit = row.get("actual_hit_pct")
                band = str(row.get("band") or "—")
                if band == "90–100%":
                    band = ">90%"
                self._conf_cal_tree.insert(
                    "",
                    "end",
                    values=(
                        band,
                        row.get("rows") or 0,
                        f"{hit:.1f}" if hit is not None else "—",
                    ),
                )

    def _open_confidence_evaluation_dialog(self, result: dict[str, Any]) -> None:
        """Full Model Evaluation dialog (confusion, metrics, distribution, calibration)."""
        eval_block = result.get("evaluation") if isinstance(result.get("evaluation"), dict) else {}
        core = eval_block.get("core_metrics") or {}
        dist = eval_block.get("class_distribution") or {}
        thresh = eval_block.get("threshold") or {}
        confusion = eval_block.get("confusion") or result.get("confusion") or {}
        metrics = result.get("metrics") or {}

        def _fmt(v: Any, *, pct: bool = False, digits: int = 2) -> str:
            if v is None or v == "":
                return "—"
            try:
                x = float(v)
            except (TypeError, ValueError):
                return str(v)
            if pct:
                return f"{x:.{digits}f}%"
            if abs(x) >= 100 or abs(x) < 0.001 and x != 0:
                return f"{x:.{digits}f}"
            return f"{x:.{digits}f}"

        def _fmt_int(v: Any) -> str:
            if v is None or v == "":
                return "—"
            try:
                return f"{int(v):,}"
            except (TypeError, ValueError):
                return str(v)

        dlg = tk.Toplevel(self)
        label = str(result.get("label") or result.get("model_key") or "Model")
        dlg.title(f"Model Evaluation — {label}")
        dlg.transient(self)
        # Match main Research Lab height so footer actions stay on-screen
        try:
            self.update_idletasks()
            parent_h = max(int(self.winfo_height()), 700)
            parent_w = max(int(self.winfo_width()), 900)
        except tk.TclError:
            parent_h, parent_w = 900, 1000
        dlg_w = min(max(parent_w - 40, 900), 1200)
        dlg.geometry(f"{dlg_w}x{parent_h}")
        dlg.minsize(860, 640)

        # Footer pinned first so Save / Close are never clipped
        btns = ttk.Frame(dlg, padding=(12, 8, 12, 12))
        btns.pack(side="bottom", fill="x")

        header = ttk.Frame(dlg, padding=(12, 12, 12, 4))
        header.pack(side="top", fill="x")
        ttk.Label(header, text=f"Model Evaluation — {label}", font=SECTION_FONT).pack(
            anchor="w"
        )
        status = str(result.get("status") or "").replace("_", " ").title()
        created = str(result.get("created_at") or "")[:19].replace("T", " ")
        ttk.Label(
            header,
            text=f"Status: {status or '—'}  ·  Created: {created or '—'}",
            foreground=COL_MUTED,
        ).pack(anchor="w", pady=(2, 0))

        scroll = ScrollableFrame(dlg)
        scroll.pack(fill="both", expand=True)
        body = ttk.Frame(scroll.inner, padding=(8, 4, 8, 8))
        body.pack(fill="both", expand=True)

        # —— Confusion Matrix ——
        cm_box = ttk.LabelFrame(body, text="Confusion Matrix", padding=8)
        cm_box.pack(fill="x", pady=(0, 8))
        cm_cols = ("row", "ap", "an")
        cm_tree = ttk.Treeview(cm_box, columns=cm_cols, show="headings", height=2)
        cm_tree.heading("row", text="")
        cm_tree.heading("ap", text="Actual Positive")
        cm_tree.heading("an", text="Actual Negative")
        cm_tree.column("row", width=140, anchor="w")
        cm_tree.column("ap", width=140, anchor="center")
        cm_tree.column("an", width=140, anchor="center")
        cm_tree.pack(fill="x")
        cm_tree.insert(
            "",
            "end",
            values=(
                "Predicted Positive",
                _fmt_int(confusion.get("tp")),
                _fmt_int(confusion.get("fp")),
            ),
        )
        cm_tree.insert(
            "",
            "end",
            values=(
                "Predicted Negative",
                _fmt_int(confusion.get("fn")),
                _fmt_int(confusion.get("tn")),
            ),
        )

        # —— Core Metrics + Class Distribution (same row) ——
        metrics_row = ttk.Frame(body)
        metrics_row.pack(fill="x", pady=(0, 8))
        metrics_row.columnconfigure(0, weight=3, minsize=360)
        metrics_row.columnconfigure(1, weight=2, minsize=240)

        core_box = ttk.LabelFrame(metrics_row, text="Core Metrics", padding=8)
        core_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        core_rows = (
            ("True Positives (TP)", _fmt_int(core.get("tp", confusion.get("tp")))),
            ("False Positives (FP)", _fmt_int(core.get("fp", confusion.get("fp")))),
            ("True Negatives (TN)", _fmt_int(core.get("tn", confusion.get("tn")))),
            ("False Negatives (FN)", _fmt_int(core.get("fn", confusion.get("fn")))),
            ("Precision", _fmt(core.get("precision_pct", metrics.get("precision_pct")), pct=True)),
            ("Recall", _fmt(core.get("recall_pct", metrics.get("recall_pct")), pct=True)),
            ("F1 Score", _fmt(core.get("f1_pct", metrics.get("f1_pct")), pct=True)),
            ("Accuracy", _fmt(core.get("accuracy_pct", metrics.get("accuracy_pct")), pct=True)),
            ("ROC AUC", _fmt(core.get("roc_auc", metrics.get("roc_auc")), digits=4)),
            ("PR AUC", _fmt(core.get("pr_auc", metrics.get("pr_auc")), digits=4)),
            ("Brier Score", _fmt(core.get("brier_score", metrics.get("brier_score")), digits=4)),
        )
        for i, (name, val) in enumerate(core_rows):
            r, c = divmod(i, 2)
            ttk.Label(core_box, text=f"{name}:", foreground=COL_MUTED).grid(
                row=r, column=c * 2, sticky="w", padx=(0, 6), pady=2
            )
            ttk.Label(core_box, text=val).grid(
                row=r, column=c * 2 + 1, sticky="w", padx=(0, 18), pady=2
            )

        dist_box = ttk.LabelFrame(metrics_row, text="Class Distribution", padding=8)
        dist_box.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        dist_rows = (
            ("Total Rows", _fmt_int(dist.get("total_rows"))),
            ("Actual Positives", _fmt_int(dist.get("actual_positives"))),
            ("Actual Negatives", _fmt_int(dist.get("actual_negatives"))),
            ("Predicted Positives", _fmt_int(dist.get("predicted_positives"))),
            ("Predicted Negatives", _fmt_int(dist.get("predicted_negatives"))),
            ("Positive Rate (%)", _fmt(dist.get("positive_rate_pct"), pct=True)),
        )
        for i, (name, val) in enumerate(dist_rows):
            ttk.Label(dist_box, text=f"{name}:", foreground=COL_MUTED).grid(
                row=i, column=0, sticky="w", padx=(0, 6), pady=2
            )
            ttk.Label(dist_box, text=val).grid(
                row=i, column=1, sticky="w", padx=(0, 8), pady=2
            )

        # —— Threshold ——
        thr_box = ttk.LabelFrame(body, text="Threshold Information", padding=8)
        thr_box.pack(fill="x", pady=(0, 8))
        thr_val = thresh.get("classification_threshold", metrics.get("threshold", 0.5))
        op_thr = result.get("operating_threshold")
        if op_thr is None:
            op_thr = metrics.get("operating_threshold")
        thr_saved_var = tk.StringVar(
            value=_fmt(op_thr, digits=2) if op_thr is not None else "Not set"
        )
        thr_selected_var = tk.StringVar(value="— (select a row below)")
        thr_info_rows = (
            ("Eval Classification Threshold", _fmt(thr_val, digits=2)),
            ("Selected Threshold (preview)", thr_selected_var),
            ("Saved Operating Threshold", thr_saved_var),
            (
                "Mean Predicted Probability (Positive)",
                _fmt(thresh.get("mean_prob_actual_positive"), digits=4),
            ),
            (
                "Mean Predicted Probability (Negative)",
                _fmt(thresh.get("mean_prob_actual_negative"), digits=4),
            ),
        )
        for i, (name, val) in enumerate(thr_info_rows):
            ttk.Label(thr_box, text=f"{name}:", foreground=COL_MUTED).grid(
                row=i, column=0, sticky="w", padx=(0, 6), pady=2
            )
            if isinstance(val, tk.StringVar):
                ttk.Label(thr_box, textvariable=val).grid(row=i, column=1, sticky="w", pady=2)
            else:
                ttk.Label(thr_box, text=val).grid(row=i, column=1, sticky="w", pady=2)
        note = thresh.get("note")
        note_row = len(thr_info_rows)
        if note:
            ttk.Label(
                thr_box,
                text=str(note),
                foreground=COL_MUTED,
                wraplength=640,
                justify="left",
            ).grid(row=note_row, column=0, columnspan=2, sticky="w", pady=(6, 0))
            note_row += 1
        ttk.Label(
            thr_box,
            text=(
                "Select one row in Threshold Analysis to preview filter metrics. "
                "Save Operating Threshold stores only that one value on the model "
                "(used later by Confidence Inference — never a hard-coded 0.50)."
            ),
            foreground=COL_MUTED,
            wraplength=640,
            justify="left",
        ).grid(row=note_row, column=0, columnspan=2, sticky="w", pady=(6, 0))

        # —— Trading Filter Summary (updates when a Threshold Analysis row is selected) ——
        filter_title_var = tk.StringVar(
            value=f"Trading Filter Summary  (threshold {_fmt(thr_val, digits=2)})"
        )
        filter_box = ttk.LabelFrame(body, text="", padding=8)
        filter_box.pack(fill="x", pady=(0, 8))
        # LabelFrame title via separate label so we can update it
        ttk.Label(
            filter_box, textvariable=filter_title_var, font=("Segoe UI", 9, "bold")
        ).grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 4))
        filt = eval_block.get("trading_filter_summary") or {}
        filter_vars = {
            "good_kept": tk.StringVar(
                value=_fmt(filt.get("good_trades_kept_pct"), pct=True)
            ),
            "good_filt": tk.StringVar(
                value=_fmt(filt.get("good_trades_filtered_pct"), pct=True)
            ),
            "bad_filt": tk.StringVar(
                value=_fmt(filt.get("bad_trades_filtered_pct"), pct=True)
            ),
            "bad_pass": tk.StringVar(
                value=_fmt(filt.get("bad_trades_passed_pct"), pct=True)
            ),
        }
        filter_disp = (
            ("Good Trades Kept", filter_vars["good_kept"]),
            ("Good Trades Filtered", filter_vars["good_filt"]),
            ("Bad Trades Filtered", filter_vars["bad_filt"]),
            ("Bad Trades Passed", filter_vars["bad_pass"]),
        )
        for i, (name, var) in enumerate(filter_disp):
            r, c = divmod(i, 2)
            ttk.Label(filter_box, text=f"{name}:", foreground=COL_MUTED).grid(
                row=r + 1, column=c * 2, sticky="w", padx=(0, 6), pady=2
            )
            ttk.Label(
                filter_box, textvariable=var, font=("Segoe UI", 10, "bold")
            ).grid(row=r + 1, column=c * 2 + 1, sticky="w", padx=(0, 18), pady=2)
        ttk.Label(
            filter_box,
            text=(
                "Good Trades Kept = Recall (TP / actual hits).  "
                "Bad Trades Filtered = Specificity (TN / actual misses).  "
                "A useful filter raises Bad Trades Filtered without crushing Good Trades Kept."
            ),
            foreground=COL_MUTED,
            wraplength=680,
            justify="left",
        ).grid(row=3, column=0, columnspan=4, sticky="w", pady=(6, 0))

        # —— Threshold Analysis + Calibration (same row) ——
        ta_cal_row = ttk.Frame(body)
        ta_cal_row.pack(fill="both", expand=True, pady=(0, 4))
        ta_cal_row.columnconfigure(0, weight=3, minsize=420)
        ta_cal_row.columnconfigure(1, weight=2, minsize=260)

        ta_box = ttk.LabelFrame(ta_cal_row, text="Threshold Analysis", padding=8)
        ta_box.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        ta_cols = (
            "thr",
            "precision",
            "recall",
            "good_kept",
            "bad_filt",
            "pred_rate",
        )
        ta_tree = ttk.Treeview(ta_box, columns=ta_cols, show="headings", height=6)
        for cid, head, w in (
            ("thr", "Threshold", 70),
            ("precision", "Precision", 75),
            ("recall", "Recall", 70),
            ("good_kept", "Good Kept", 90),
            ("bad_filt", "Bad Filtered", 95),
            ("pred_rate", "Pred +%", 75),
        ):
            ta_tree.heading(cid, text=head)
            ta_tree.column(cid, width=w, anchor="center")
        ta_tree.pack(fill="both", expand=True)
        ta_rows = (
            eval_block.get("threshold_analysis")
            or result.get("threshold_analysis")
            or metrics.get("threshold_analysis")
            or []
        )
        # Index by threshold for live preview
        ta_by_thr: dict[float, dict[str, Any]] = {}
        for row in ta_rows or []:
            try:
                ta_by_thr[round(float(row.get("threshold")), 2)] = dict(row)
            except (TypeError, ValueError):
                pass
        default_thr_f = None
        try:
            default_thr_f = round(float(thr_val), 2)
        except (TypeError, ValueError):
            default_thr_f = 0.5
        if ta_rows:
            for row in ta_rows:
                thr = row.get("threshold")
                mark = ""
                try:
                    if default_thr_f is not None and abs(float(thr) - default_thr_f) < 1e-9:
                        mark = " ●"
                except (TypeError, ValueError):
                    pass
                good_kept = row.get("good_trades_kept_pct")
                if good_kept is None:
                    good_kept = row.get("recall_pct")
                bad_filt = row.get("bad_trades_filtered_pct")
                if bad_filt is None:
                    bad_filt = row.get("specificity_pct")
                ta_tree.insert(
                    "",
                    "end",
                    values=(
                        f"{float(thr):.2f}{mark}" if thr is not None else "—",
                        _fmt(row.get("precision_pct"), pct=True),
                        _fmt(row.get("recall_pct"), pct=True),
                        _fmt(good_kept, pct=True),
                        _fmt(bad_filt, pct=True),
                        _fmt(row.get("predicted_positive_rate_pct"), pct=True),
                    ),
                )
            ttk.Label(
                ta_box,
                text=(
                    "● = eval headline  ·  Click a row to preview  ·  "
                    "Save Operating Threshold stores that one value only"
                ),
                foreground=COL_MUTED,
                wraplength=480,
            ).pack(anchor="w", pady=(4, 0))
        else:
            legacy_msg = result.get("legacy_message") or (
                "This Confidence Model is Legacy: Threshold Analysis was not saved "
                "when it was trained. Retrain to generate the sweep "
                "(0.50 / 0.60 / 0.70 / 0.80 / 0.90), then select and Save Operating Threshold."
            )
            ttk.Label(
                ta_box,
                text=str(legacy_msg),
                foreground=COL_MUTED,
                wraplength=420,
            ).pack(anchor="w")

        cal_box = ttk.LabelFrame(ta_cal_row, text="Calibration", padding=8)
        cal_box.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        cal_cols = ("band", "rows", "hit_pct")
        cal_tree = ttk.Treeview(cal_box, columns=cal_cols, show="headings", height=6)
        cal_tree.heading("band", text="Probability Band")
        cal_tree.heading("rows", text="Rows")
        cal_tree.heading("hit_pct", text="Actual Hit %")
        cal_tree.column("band", width=110, anchor="w")
        cal_tree.column("rows", width=60, anchor="center")
        cal_tree.column("hit_pct", width=90, anchor="center")
        cal_tree.pack(fill="both", expand=True)
        for row in result.get("calibration") or []:
            hit = row.get("actual_hit_pct")
            band = str(row.get("band") or "—")
            if band == "90–100%":
                band = ">90%"
            cal_tree.insert(
                "",
                "end",
                values=(
                    band,
                    _fmt_int(row.get("rows")),
                    f"{float(hit):.1f}%" if hit is not None else "—",
                ),
            )

        model_key = str(result.get("model_key") or "")

        def _parse_selected_threshold() -> float | None:
            sel = ta_tree.selection()
            if not sel:
                return None
            vals = ta_tree.item(sel[0], "values")
            if not vals:
                return None
            raw = str(vals[0]).replace("●", "").strip()
            try:
                return float(raw)
            except (TypeError, ValueError):
                return None

        def _apply_threshold_preview(thr_f: float | None) -> None:
            if thr_f is None:
                thr_selected_var.set("— (select a row below)")
                return
            thr_selected_var.set(f"{thr_f:.2f}")
            filter_title_var.set(f"Trading Filter Summary  (threshold {thr_f:.2f})")
            row = ta_by_thr.get(round(thr_f, 2)) or {}
            good_kept = row.get("good_trades_kept_pct")
            if good_kept is None:
                good_kept = row.get("recall_pct")
            bad_filt = row.get("bad_trades_filtered_pct")
            if bad_filt is None:
                bad_filt = row.get("specificity_pct")
            # Derived counterparts when available
            good_filt = row.get("good_trades_filtered_pct")
            if good_filt is None and good_kept is not None:
                try:
                    good_filt = 100.0 - float(good_kept)
                except (TypeError, ValueError):
                    good_filt = None
            bad_pass = row.get("bad_trades_passed_pct")
            if bad_pass is None and bad_filt is not None:
                try:
                    bad_pass = 100.0 - float(bad_filt)
                except (TypeError, ValueError):
                    bad_pass = None
            filter_vars["good_kept"].set(_fmt(good_kept, pct=True))
            filter_vars["good_filt"].set(_fmt(good_filt, pct=True))
            filter_vars["bad_filt"].set(_fmt(bad_filt, pct=True))
            filter_vars["bad_pass"].set(_fmt(bad_pass, pct=True))

        def _on_ta_select(_event: object = None) -> None:
            _apply_threshold_preview(_parse_selected_threshold())

        ta_tree.bind("<<TreeviewSelect>>", _on_ta_select)

        def _save_operating_threshold() -> None:
            from chain_replay_ml.model_lab.confidence import set_operating_threshold

            lab_path = getattr(self._lab, "db_path", None) if self._lab else None
            if not lab_path or not model_key:
                messagebox.showerror(
                    "Save Operating Threshold",
                    "No Research Lab / model key available.",
                    parent=dlg,
                )
                return
            if result.get("is_legacy") or not ta_rows:
                messagebox.showinfo(
                    "Save Operating Threshold",
                    result.get("legacy_message")
                    or "Retrain this Confidence Model to generate Threshold Analysis first.",
                    parent=dlg,
                )
                return
            thr_sel = _parse_selected_threshold()
            if thr_sel is None:
                messagebox.showinfo(
                    "Save Operating Threshold",
                    "Select a row in Threshold Analysis first.",
                    parent=dlg,
                )
                return
            try:
                saved = set_operating_threshold(lab_path, model_key, thr_sel)
            except Exception as exc:
                messagebox.showerror("Save Operating Threshold", str(exc), parent=dlg)
                return
            thr_saved_var.set(f"{float(saved['operating_threshold']):.2f}")
            stale = " Inference marked Out of Date." if saved.get("inference_stale") else ""
            messagebox.showinfo(
                "Save Operating Threshold",
                f"Operating Threshold saved: {float(saved['operating_threshold']):.2f}."
                f"{stale}\n\n"
                "This single value is what Confidence Inference will use "
                "to decide Hit=1 vs Hit=0 on every Prediction row.",
                parent=dlg,
            )
            self._refresh_confidence_tab()

        save_thr_btn = ttk.Button(
            btns,
            text="Save Operating Threshold",
            command=_save_operating_threshold,
        )
        save_thr_btn.pack(side="left")
        if result.get("is_legacy") or not ta_rows:
            save_thr_btn.state(["disabled"])
        ttk.Button(btns, text="Close", command=dlg.destroy).pack(side="right")

    def _on_link_active_confidence_model(self) -> None:
        from chain_replay_ml.model_lab.confidence_manifest import set_active_model

        lab_path = getattr(self._lab, "db_path", None) if self._lab else None
        if not lab_path:
            return
        key = self._selected_confidence_model_key()
        if not key:
            messagebox.showinfo("Link Active Model", "Select a trained model.", parent=self)
            return
        try:
            set_active_model(lab_path, key)
        except Exception as exc:
            messagebox.showerror("Link Active Model", str(exc), parent=self)
            return
        self._refresh_confidence_tab()
        self._conf_action_status.set(f"Active → {key}")

    def _set_confidence_inference_progress(
        self,
        percent: float | None,
        message: str = "",
        *,
        status_label: str = "Confidence Inference",
        active: bool = True,
        processed: int | None = None,
        total: int | None = None,
        positive: int | None = None,
        negative: int | None = None,
    ) -> None:
        """Update Inference panel + Actions strip + window status bar."""
        if hasattr(self, "_conf_inf_vars"):
            if active:
                self._conf_inf_vars["status"].set("Running")
            if processed is not None and total is not None:
                self._conf_inf_vars["rows"].set(f"{int(processed):,} / {int(total):,}")
            if positive is not None and negative is not None:
                self._conf_inf_vars["pos_neg"].set(
                    f"{int(positive):,} / {int(negative):,}"
                )
            if message:
                self._conf_inf_vars["detail"].set(message)
        if hasattr(self, "_conf_inf_progress"):
            if percent is None:
                self._conf_inf_progress.configure(value=0.0)
                self._conf_inf_progress_var.set("")
            else:
                pct = max(0.0, min(100.0, float(percent)))
                self._conf_inf_progress.configure(value=pct)
                self._conf_inf_progress_var.set(
                    f"{pct:.0f}%" if not message else message[:40]
                )
        # Also drive the Actions progress strip (visible near Run Inference)
        if hasattr(self, "_conf_progress"):
            if percent is None:
                self._conf_progress.configure(value=0.0)
                self._conf_progress_pct.set("")
            else:
                pct = max(0.0, min(100.0, float(percent)))
                self._conf_progress.configure(value=pct)
                self._conf_progress_pct.set(f"{pct:.0f}%")
        if message and hasattr(self, "_conf_action_status"):
            self._conf_action_status.set(message)
        if hasattr(self, "_set_status_bar"):
            self._set_status_bar(
                status=status_label if active else "Ready",
                percent=percent if active else None,
                detail=message,
                active=active,
            )

    def _on_run_confidence_inference(self) -> None:
        from chain_replay_ml.model_lab.confidence import run_confidence_inference
        from chain_replay_ml.model_lab.confidence_manifest import TARGET_BY_KEY

        lab_path = getattr(self._lab, "db_path", None) if self._lab else None
        if not lab_path:
            messagebox.showwarning(
                "Confidence Inference", "Open a Research Lab first.", parent=self
            )
            return
        model_key = self._selected_confidence_model_key()
        if not model_key:
            messagebox.showinfo(
                "Confidence Inference",
                "Select a Confidence Model in the table first.",
                parent=self,
            )
            return
        label = TARGET_BY_KEY.get(model_key, {}).get("label") or model_key
        if not messagebox.askyesno(
            "Confidence Inference",
            f"Score the full Prediction Dataset with {label} "
            "and its saved Operating Threshold?\n\n"
            "Only final 0/1 decisions are written (no probabilities).",
            parent=self,
        ):
            return
        self._conf_inf_run_btn.configure(state="disabled")
        if hasattr(self, "_conf_inf_clear_btn"):
            self._conf_inf_clear_btn.configure(state="disabled")
        if hasattr(self, "_conf_inf_title_var"):
            self._conf_inf_title_var.set(f"Inference · {label}")
        start_msg = f"Starting {label} inference…"
        self._set_confidence_inference_progress(
            0.0,
            start_msg,
            status_label="Confidence Inference",
            active=True,
            processed=0,
            total=None,
        )
        self._conf_inf_vars["status"].set("Running")
        self._conf_inf_vars["validation"].set("—")
        self._conf_inf_vars["completed"].set("—")
        prog_q: queue.Queue[dict[str, Any]] = queue.Queue()
        self._conf_inf_queue = prog_q
        self._conf_inf_running_key = model_key

        def on_progress(payload: dict[str, Any]) -> None:
            prog_q.put(dict(payload))

        def worker() -> None:
            try:
                result = run_confidence_inference(
                    lab_path,
                    model_key=model_key,
                    data_dir=chart_data_dir(self.chart_dir),
                    on_progress=on_progress,
                )
            except Exception as exc:
                result = {"ok": False, "error": str(exc)}
            prog_q.put({"_done": True, "result": result})

        threading.Thread(target=worker, daemon=True).start()
        self.after(80, self._poll_confidence_inference_progress)

    def _poll_confidence_inference_progress(self) -> None:
        q = self._conf_inf_queue
        if q is None:
            return
        try:
            while True:
                payload = q.get_nowait()
                if payload.get("_done"):
                    self._conf_inf_queue = None
                    self._on_confidence_inference_done(payload.get("result") or {})
                    return
                processed = int(payload.get("processed") or 0)
                total = max(int(payload.get("total") or 0), 1)
                pct = min(100.0, 100.0 * processed / total)
                msg = str(payload.get("message") or "")
                eta = payload.get("eta_sec")
                if eta is not None:
                    try:
                        msg = f"{msg} · ETA {float(eta):.0f}s"
                    except (TypeError, ValueError):
                        pass
                self._set_confidence_inference_progress(
                    pct,
                    msg,
                    status_label="Confidence Inference",
                    active=True,
                    processed=processed,
                    total=int(payload.get("total") or total),
                    positive=payload.get("positive"),
                    negative=payload.get("negative"),
                )
        except queue.Empty:
            pass
        self.after(100, self._poll_confidence_inference_progress)

    def _on_confidence_inference_done(self, result: dict[str, Any]) -> None:
        ok = bool(result.get("ok"))
        thr = result.get("threshold")
        thr_s = f"{float(thr):.2f}" if thr is not None else "—"
        rows = int((result.get("validation") or {}).get("rows_updated") or 0)
        key = result.get("model_key") or getattr(self, "_conf_inf_running_key", "") or ""
        done_msg = (
            f"{'Complete' if ok else 'Failed'} · {key} · thr={thr_s} · {rows:,} rows"
            if ok
            else str(result.get("error") or "Inference failed.")
        )
        self._set_confidence_inference_progress(
            100.0 if ok else 0.0,
            done_msg,
            status_label="Confidence Inference" if ok else "Error",
            active=False,
        )
        if ok:
            messagebox.showinfo(
                "Confidence Inference",
                f"Inference complete ({key}).\nThreshold: {thr_s}\nRows: {rows:,}",
                parent=self,
            )
        else:
            messagebox.showerror(
                "Confidence Inference",
                str(result.get("error") or "Inference failed."),
                parent=self,
            )
        self._refresh_confidence_tab()
        self._set_status_bar(
            status="Ready",
            detail=done_msg if ok else f"Inference failed: {result.get('error')}",
            active=False,
        )

    def _on_clear_confidence_inference(self) -> None:
        from chain_replay_ml.model_lab.confidence import clear_confidence_inference
        from chain_replay_ml.model_lab.confidence_manifest import TARGET_BY_KEY

        lab_path = getattr(self._lab, "db_path", None) if self._lab else None
        if not lab_path:
            return
        model_key = self._selected_confidence_model_key()
        if not model_key:
            messagebox.showinfo(
                "Clear Inference", "Select a Confidence Model first.", parent=self
            )
            return
        label = TARGET_BY_KEY.get(model_key, {}).get("label") or model_key
        if not messagebox.askyesno(
            "Clear Inference",
            f"Clear {label} confidence columns from the Prediction Dataset?",
            parent=self,
        ):
            return
        try:
            clear_confidence_inference(lab_path, model_key)
        except Exception as exc:
            messagebox.showerror("Clear Inference", str(exc), parent=self)
            return
        self._refresh_confidence_tab()
        self._conf_action_status.set(f"{label} inference cleared")

    def _on_validate_confidence_mapping(self) -> None:
        """Open Confidence Dataset Mapping Validation panel."""
        from chain_replay_ml.model_lab.confidence_mapping_validation import (
            RR_MAPPING_TARGETS,
            validate_confidence_dataset_mapping,
        )

        lab_path = getattr(self._lab, "db_path", None) if self._lab else None
        if not lab_path:
            messagebox.showwarning(
                "Validate Mapping", "Open a Research Lab first.", parent=self
            )
            return

        dlg = tk.Toplevel(self)
        dlg.title("Confidence Dataset Mapping Validation")
        dlg.transient(self)
        dlg.geometry("980x560")
        dlg.minsize(800, 460)

        header = ttk.Frame(dlg, padding=(12, 12, 12, 4))
        header.pack(fill="x")
        ttk.Label(
            header,
            text="Confidence Dataset Mapping Validation",
            font=SECTION_FONT,
        ).pack(anchor="w")
        ttk.Label(
            header,
            text=(
                "One random positive sample per RR target · outcome metadata only · "
                "no feature values"
            ),
            foreground=COL_MUTED,
        ).pack(anchor="w", pady=(2, 0))

        body = ttk.Frame(dlg, padding=(12, 4, 12, 8))
        body.pack(fill="both", expand=True)

        rr_keys = tuple(t["key"] for t in RR_MAPPING_TARGETS)
        cols = ("field", *rr_keys)
        tree = ttk.Treeview(body, columns=cols, show="headings", height=12)
        tree.heading("field", text="Field")
        tree.column("field", width=140, anchor="w")
        for spec in RR_MAPPING_TARGETS:
            tree.heading(spec["key"], text=spec["label"])
            tree.column(spec["key"], width=120, anchor="center")
        tree.pack(fill="both", expand=True)

        detail = scrolledtext.ScrolledText(
            body, height=8, wrap="word", font=("Consolas", 9)
        )
        detail.pack(fill="both", expand=True, pady=(8, 0))
        detail.configure(state="disabled")

        status_var = tk.StringVar(value="")
        ttk.Label(body, textvariable=status_var, foreground=COL_MUTED).pack(
            anchor="w", pady=(6, 0)
        )

        def _fmt_yes_no(v: Any) -> str:
            if v is None or v == "":
                return "—"
            try:
                return "Yes" if int(v) == 1 else "No"
            except (TypeError, ValueError):
                return str(v)

        def _fmt_pct(v: Any) -> str:
            if v is None or v == "":
                return "—"
            try:
                return f"{float(v):.2f}%"
            except (TypeError, ValueError):
                return str(v)

        def _fmt_ratio(v: Any) -> str:
            if v is None or v == "":
                return "—"
            try:
                return f"{float(v):.2f}"
            except (TypeError, ValueError):
                return str(v)

        def _fmt_label(target_label: str, val: Any) -> str:
            if val is None or val == "":
                return "—"
            try:
                return f"{target_label} = {int(val)}"
            except (TypeError, ValueError):
                return f"{target_label} = {val}"

        def _cell(sample: dict[str, Any], kind: str) -> str:
            if not sample.get("available"):
                if kind == "mapping":
                    return "—"
                if kind == "trading_day":
                    return "No positive sample"
                return ""
            fields = sample.get("fields") or {}
            if kind == "trading_day":
                return str(fields.get("trading_day") or "—")
            if kind == "timestamp":
                return _fmt_unix_time(fields.get("timestamp"))
            if kind == "token":
                tok = fields.get("token")
                return str(tok) if tok is not None and tok != "" else "—"
            if kind == "target_hit":
                return _fmt_yes_no(fields.get("target_hit"))
            if kind == "maximum_profit":
                return _fmt_pct(fields.get("maximum_profit"))
            if kind == "maximum_drawdown":
                return _fmt_pct(fields.get("maximum_drawdown"))
            if kind == "profit_dd_ratio":
                return _fmt_ratio(fields.get("profit_dd_ratio"))
            if kind == "dataset_label":
                return _fmt_label(sample.get("label") or "", fields.get("dataset_label"))
            if kind == "prediction_label":
                return _fmt_label(
                    sample.get("label") or "", fields.get("prediction_label")
                )
            if kind == "prediction_row_id":
                return str(fields.get("prediction_row_id") or "—")
            if kind == "dataset_row_id":
                return str(fields.get("dataset_row_id") or "—")
            if kind == "row_ids_match":
                if fields.get("prediction_row_id") is None and fields.get("dataset_row_id") is None:
                    return "—"
                return "✓" if fields.get("row_ids_match") else "✗"
            if kind == "mapping":
                return "✓" if sample.get("mapping_ok") else "✗"
            return "—"

        def _fill_detail(report: dict[str, Any]) -> None:
            lines: list[str] = []
            join_keys = report.get("join_keys") or []
            if join_keys:
                lines.append(f"Join keys: {', '.join(str(k) for k in join_keys)}")
                lines.append("")
            for sample in report.get("samples") or []:
                label = sample.get("label") or sample.get("key")
                lines.append(f"{label} Sample")
                lines.append("-" * 40)
                if not sample.get("available"):
                    lines.append(str(sample.get("message") or "No positive sample."))
                    lines.append("")
                    continue
                f = sample.get("fields") or {}
                lines.append(
                    f"Prediction Row ID  : {f.get('prediction_row_id') or '—'}"
                )
                lines.append(
                    f"Dataset Row ID     : {f.get('dataset_row_id') or '—'}"
                )
                if f.get("prediction_id"):
                    lines.append(
                        f"Prediction ID      : {f.get('prediction_id')}"
                    )
                if f.get("row_ids_match"):
                    lines.append("Row IDs            : ✓ match")
                elif f.get("prediction_row_id") or f.get("dataset_row_id"):
                    lines.append("Row IDs            : ✗ differ")
                lines.append("")
                lines.append(f"Trading Day        : {f.get('trading_day') or '—'}")
                lines.append(f"Timestamp          : {_fmt_unix_time(f.get('timestamp'))}")
                tok = f.get("token")
                lines.append(
                    f"Token              : {tok if tok is not None and tok != '' else '—'}"
                )
                lines.append(f"Path Touch         : {_fmt_yes_no(f.get('target_hit'))}")
                lines.append(
                    f"Maximum Profit     : {_fmt_pct(f.get('maximum_profit'))}"
                )
                lines.append(
                    f"Maximum Drawdown   : {_fmt_pct(f.get('maximum_drawdown'))}"
                )
                lines.append(
                    f"Profit/DD Ratio    : {_fmt_ratio(f.get('profit_dd_ratio'))}"
                )
                lines.append("")
                lines.append(
                    f"Prediction Label   : {_fmt_label(label, f.get('prediction_label'))}"
                )
                lines.append(
                    f"Dataset Label      : {_fmt_label(label, f.get('dataset_label'))}"
                )
                lines.append("")
                if sample.get("mapping_ok"):
                    lines.append("✓ Mapping Verified")
                else:
                    lines.append(f"✗ Mapping Failed — {sample.get('message') or ''}")
                lines.append("")
            detail.configure(state="normal")
            detail.delete("1.0", "end")
            detail.insert("1.0", "\n".join(lines).rstrip() + "\n")
            detail.configure(state="disabled")

        def refresh_panel() -> None:
            for iid in tree.get_children():
                tree.delete(iid)
            try:
                report = validate_confidence_dataset_mapping(lab_path)
            except Exception as exc:
                status_var.set(str(exc))
                messagebox.showerror("Validate Mapping", str(exc), parent=dlg)
                return
            if not report.get("ok"):
                status_var.set(str(report.get("error") or "Validation unavailable"))
                messagebox.showwarning(
                    "Validate Mapping",
                    str(report.get("error") or "Confidence Dataset not found."),
                    parent=dlg,
                )
                return

            by_key = {s["key"]: s for s in (report.get("samples") or [])}
            samples = [by_key.get(k) or {} for k in rr_keys]
            rows = (
                ("Prediction Row ID", "prediction_row_id"),
                ("Dataset Row ID", "dataset_row_id"),
                ("Row IDs Match", "row_ids_match"),
                ("Trading Day", "trading_day"),
                ("Timestamp", "timestamp"),
                ("Token", "token"),
                ("Path Touch", "target_hit"),
                ("Maximum Profit", "maximum_profit"),
                ("Maximum Drawdown", "maximum_drawdown"),
                ("Profit / DD Ratio", "profit_dd_ratio"),
                ("Dataset Label", "dataset_label"),
                ("Prediction Label", "prediction_label"),
                ("Mapping", "mapping"),
            )
            for title, kind in rows:
                tree.insert(
                    "",
                    "end",
                    values=(title, *(_cell(s, kind) for s in samples)),
                )
            _fill_detail(report)
            if report.get("all_mapping_ok"):
                status_var.set("All available samples: Mapping Verified")
            else:
                status_var.set("Review mapping results below")

        btns = ttk.Frame(dlg, padding=12)
        btns.pack(fill="x")
        ttk.Button(btns, text="Resample", command=refresh_panel).pack(side="left")
        ttk.Button(btns, text="Close", command=dlg.destroy).pack(side="right")

        refresh_panel()

    def _on_delete_confidence_model(self) -> None:
        from chain_replay_ml.model_lab.confidence_manifest import delete_confidence_model

        lab_path = getattr(self._lab, "db_path", None) if self._lab else None
        if not lab_path:
            return
        key = self._selected_confidence_model_key()
        if not key:
            messagebox.showinfo("Delete", "Select a model to delete.", parent=self)
            return
        if not messagebox.askyesno(
            "Delete Confidence Model",
            f"Delete lab-local model '{key}'?\nThis does not affect the regression model.",
            parent=self,
        ):
            return
        delete_confidence_model(lab_path, key)
        self._refresh_confidence_tab()
        self._conf_action_status.set(f"Deleted {key}")

    def _on_create_confidence_model(self) -> None:
        """Legacy toolbar alias → Build Confidence Dataset."""
        self._on_create_confidence_dataset()

    def _on_link_confidence_model(self) -> None:
        """Legacy alias → Link Active Model."""
        self._on_link_active_confidence_model()

    def _on_mi_select(self) -> None:
        sel = self._mi_tree.selection()
        if not sel:
            return
        vals = self._mi_tree.item(sel[0], "values")
        if not vals:
            return
        name = str(vals[0])
        row = self._mi_rows_by_feature.get(name) or {}
        text = str(row.get("reason") or "")
        if not text:
            self._mi_evidence_var.set("—")
            return
        self._mi_evidence_var.set(text)

    def _set_mi_busy(self, busy: bool) -> None:
        self._mi_busy = busy
        state = "disabled" if busy else "normal"
        for btn in (getattr(self, "_mi_start_btn", None), getattr(self, "_mi_refresh_btn", None)):
            if btn is not None:
                try:
                    btn.configure(state=state)
                except tk.TclError:
                    pass

    def _refresh_model_improvement(self) -> None:
        """Compute Model Improvement off the UI thread (explicit Start / Refresh only)."""
        if self._lab is None:
            return
        if self._mi_busy:
            return
        db_path = self._lab.db_path
        self._set_mi_busy(True)
        self._mi_summary_var.set("Analyzing features… (UI stays responsive)")
        self._set_status_bar(
            status="Model Improvement…",
            detail="Computing research suggestions in background…",
            active=True,
        )

        def work() -> dict[str, Any]:
            from chain_replay_ml.model_lab.model_improvement import compute_model_improvement

            return compute_model_improvement(db_path)

        def apply(result: dict[str, Any]) -> None:
            self._set_mi_busy(False)
            for item in self._mi_tree.get_children():
                self._mi_tree.delete(item)
            self._mi_rows_by_feature = {}
            if result.get("error") or not result.get("available"):
                self._mi_summary_var.set(str(result.get("error") or "Unavailable"))
                self._mi_answers_var.set("")
                self._mi_evidence_var.set("")
                self._set_status_bar(status="Ready", detail="Model Improvement unavailable", active=False)
                return

            s = result.get("summary") or {}
            self._mi_summary_var.set(
                f"{s.get('total', 0)} features · "
                f"Strong Promote {s.get('strong_promote', 0)} · "
                f"Promote {s.get('promote', 0)} · "
                f"Review {s.get('review', 0)} · "
                f"Watch {s.get('watch', 0)} · "
                f"Candidate Remove {s.get('candidate_remove', 0)}"
            )
            ans = result.get("answers") or {}

            def _fmt_list(label: str, names: list[Any]) -> str:
                if not names:
                    return f"{label}: —"
                shown = ", ".join(str(n) for n in names[:8])
                more = f" (+{len(names) - 8})" if len(names) > 8 else ""
                return f"{label}: {shown}{more}"

            self._mi_answers_var.set(
                "\n".join(
                    [
                        "Next-experiment suggestions",
                        _fmt_list("Strong Promote", list(ans.get("strong_promote") or [])),
                        _fmt_list("Promote", list(ans.get("promote") or [])),
                        _fmt_list("Review", list(ans.get("review") or [])),
                        _fmt_list("Candidate Remove", list(ans.get("candidate_remove") or [])),
                    ]
                )
            )
            for row in result.get("features") or []:
                name = str(row.get("feature") or "")
                self._mi_rows_by_feature[name] = row
                self._mi_tree.insert(
                    "",
                    "end",
                    values=(
                        name,
                        f"{float(row.get('research_score') or 0):.0f}",
                        row.get("model_rank") if row.get("model_rank") is not None else "—",
                        row.get("research_rank") if row.get("research_rank") is not None else "—",
                        row.get("evidence") or "—",
                        row.get("recommendation"),
                    ),
                )
            self._mi_evidence_var.set("Select a feature to see structured evidence.")
            self._set_status_bar(
                status="Ready",
                detail=f"Model Improvement · {s.get('total', 0)} features",
                active=False,
            )

        def on_err(exc: Exception) -> None:
            self._set_mi_busy(False)
            self._mi_summary_var.set(f"Analysis failed: {exc}")
            self._set_status_bar(status="Ready", detail=f"Model Improvement failed: {exc}", active=False)

        def runner() -> None:
            err: Exception | None = None
            result: dict[str, Any] | None = None
            try:
                result = work()
            except Exception as exc:  # noqa: BLE001
                err = exc

            def done() -> None:
                if err is not None:
                    on_err(err)
                    return
                apply(result or {})

            try:
                self.after(0, done)
            except tk.TclError:
                pass

        self._mi_thread = threading.Thread(target=runner, daemon=True, name="lab-model-improvement")
        self._mi_thread.start()
    def _build_prediction_tab(self) -> None:
        ttk.Label(
            self._prediction_tab,
            text="Prediction Dataset",
            font=SECTION_FONT,
        ).pack(anchor="w", pady=(4, 6))

        self._prediction_sub_nb = ttk.Notebook(self._prediction_tab)
        self._prediction_sub_nb.pack(fill="both", expand=True)
        self._prediction_build_pane = ttk.Frame(self._prediction_sub_nb, padding=0)
        self._prediction_metadata_pane = ttk.Frame(self._prediction_sub_nb, padding=8)
        self._prediction_sub_nb.add(self._prediction_build_pane, text="Build & Explore")
        self._prediction_sub_nb.add(self._prediction_metadata_pane, text="Metadata")
        self._prediction_sub_nb.bind(
            "<<NotebookTabChanged>>", self._on_prediction_subtab_changed
        )
        self._pred_meta_loaded = False
        self._pred_meta_gen = 0
        self._pred_meta: dict[str, Any] = {}

        # —— Trading Days (single-day select) ——
        day_frm = ttk.LabelFrame(self._prediction_build_pane, text="Trading Days", padding=8)
        day_frm.pack(fill="x", pady=(0, 8))
        top = ttk.Frame(day_frm)
        top.pack(fill="x", pady=(0, 4))
        ttk.Label(
            top,
            text="Select one trading day (Seen or Unseen) for Build / Compute · "
            "Dataset Type is per model · Build All = Seen only",
            foreground=COL_MUTED,
        ).pack(side="left")
        ttk.Button(top, text="Refresh Days", command=self._sync_and_refresh_days).pack(
            side="right",
        )
        ttk.Button(top, text="Fill Path Ts", command=self._on_backfill_time_to_target).pack(
            side="right", padx=(0, 6),
        )
        self._btn_export = ttk.Button(top, text="Export…", command=self._on_export)
        self._btn_export.pack(side="right", padx=(0, 6))

        tree_wrap = ttk.Frame(day_frm)
        tree_wrap.pack(fill="x")
        self._day_tree = ttk.Treeview(
            tree_wrap,
            columns=_DAY_TABLE_COLS,
            show="headings",
            height=7,
            selectmode="browse",
        )
        headings = {
            "trading_day": "Trading Day",
            "dataset_type": "Dataset Type",
            "dataset_rows": "Dataset rows",
            "pred_rows": "Prediction rows",
            "status": "Status",
            "time_taken": "Time taken",
            "note": "Note",
            "dashboard": "Open Dashboard",
        }
        widths = {
            "trading_day": 110,
            "dataset_type": 90,
            "dataset_rows": 110,
            "pred_rows": 120,
            "status": 110,
            "time_taken": 90,
            "note": 180,
            "dashboard": 110,
        }
        for col in _DAY_TABLE_COLS:
            self._day_tree.heading(col, text=headings[col])
            anchor = "center" if col == "dashboard" else "w"
            self._day_tree.column(
                col,
                width=widths[col],
                stretch=(col == "note"),
                anchor=anchor,
            )
        yscroll = ttk.Scrollbar(tree_wrap, orient="vertical", command=self._day_tree.yview)
        self._day_tree.configure(yscrollcommand=yscroll.set)
        self._day_tree.pack(side="left", fill="x", expand=True)
        yscroll.pack(side="right", fill="y")
        self._day_tree.bind("<<TreeviewSelect>>", lambda _e: self._on_outcome_day_select())
        self._day_tree.bind("<Button-1>", self._on_day_tree_click, add="+")
        self._day_rows: dict[str, dict[str, Any]] = {}
        self._build_summary_var = tk.StringVar(value="")
        self._live_progress_var = tk.StringVar(value="")
        self._progress_pct_var = tk.StringVar(value="")
        self._progress_bar = ttk.Progressbar(day_frm, length=1, mode="determinate", maximum=100)
        # Keep vars for older helpers; bar is not shown in the new layout.
        self._build_mgr = day_frm
        self._btn_start = ttk.Button(day_frm)  # placeholder for legacy refs
        self._btn_test = self._btn_start
        self._btn_pause = self._btn_start
        self._btn_resume = self._btn_start
        self._btn_cancel = self._btn_start
        self._btn_generate = self._btn_start
        self._workers_var = self._build_workers
        self._overwrite_var = self._overwrite
        self._progress_frame = day_frm
        self._progress_text = self._live_progress_var
        self._worker_text = tk.StringVar(value="")
        self._pred_summary = self._build_summary_var
        self._pred_actions = top

        # —— Outcome compute (same controls as benchmark_outcome_gui) ——
        out = ttk.LabelFrame(
            self._prediction_build_pane, text="Outcome Compute", padding=8
        )
        out.pack(fill="x", pady=(0, 8))

        lim = ttk.Frame(out)
        lim.pack(fill="x")
        ttk.Label(lim, text="Row count").pack(side="left")
        ttk.Entry(lim, textvariable=self._row_limit, width=12).pack(side="left", padx=6)
        ttk.Button(lim, text="Use all dataset", command=self._outcome_use_all_dataset).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(lim, text="Use all pred", command=self._outcome_use_all_pred).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(lim, text="Clear (= all)", command=lambda: self._row_limit.set("")).pack(
            side="left"
        )
        self._row_hint = tk.StringVar(
            value="Select a day. Blank row count = all available rows."
        )
        ttk.Label(out, textvariable=self._row_hint, foreground=COL_MUTED).pack(
            anchor="w", pady=(4, 6)
        )

        bopts = ttk.Frame(out)
        bopts.pack(fill="x")
        ttk.Label(bopts, text="Build workers").pack(side="left")
        ttk.Entry(bopts, textvariable=self._build_workers, width=5).pack(side="left", padx=6)
        ttk.Checkbutton(
            bopts, text="Enrich path outcomes during build", variable=self._enrich_path
        ).pack(side="left", padx=12)
        ttk.Checkbutton(
            bopts, text="Overwrite existing day", variable=self._overwrite
        ).pack(side="left")

        # Triple Barrier Optional Scorer
        tb_opts = ttk.Frame(out)
        tb_opts.pack(fill="x", pady=(4, 0))
        self._tb_enable_var = tk.BooleanVar(value=bool(self._ui_state.get("model_lab.tb_enable", False)))
        self._tb_model_var = tk.StringVar(value=str(self._ui_state.get("model_lab.tb_model") or ""))
        ttk.Checkbutton(
            tb_opts, text="Enable Triple Barrier", variable=self._tb_enable_var, command=self._on_tb_enable_toggle
        ).pack(side="left")
        ttk.Label(tb_opts, text="Model:").pack(side="left", padx=(12, 4))
        self._tb_model_combo = ttk.Combobox(tb_opts, textvariable=self._tb_model_var, state="disabled", width=35)
        self._tb_model_combo.pack(side="left")
        self._tb_model_combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: self._ui_state.set("model_lab.tb_model", self._tb_model_var.get()),
            add="+",
        )
        self._populate_tb_model_combo()
        if self._tb_enable_var.get():
            self._on_tb_enable_toggle()

        prow = ttk.Frame(out)
        prow.pack(fill="x", pady=(6, 0))
        ttk.Label(prow, text="Pool size").pack(side="left")
        ttk.Entry(prow, textvariable=self._pool_size, width=6).pack(side="left", padx=6)
        ttk.Label(prow, text="Parallel chunks:").pack(side="left", padx=(12, 4))
        for w in _OUTCOME_CHUNK_CHOICES:
            ttk.Checkbutton(prow, text=str(w), variable=self._worker_vars[w]).pack(
                side="left", padx=2
            )

        acts = ttk.Frame(out)
        acts.pack(fill="x", pady=(8, 0))
        self._btn_outcome_build = ttk.Button(
            acts, text="1) Build predictions", command=self._outcome_start_build
        )
        self._btn_outcome_build.pack(side="left", padx=(0, 6))
        self._btn_outcome_build_all = ttk.Button(
            acts, text="Build All", command=self._outcome_start_build_all
        )
        self._btn_outcome_build_all.pack(side="left", padx=(0, 6))
        self._btn_outcome_compute = ttk.Button(
            acts, text="2) Compute outcomes", command=self._outcome_start_compute
        )
        self._btn_outcome_compute.pack(side="left", padx=(0, 6))
        self._btn_outcome_both = ttk.Button(
            acts, text="Build + Compute", command=self._outcome_start_both
        )
        self._btn_outcome_both.pack(side="left", padx=(0, 6))
        ttk.Button(acts, text="Copy log", command=self._outcome_copy_log).pack(
            side="left", padx=(0, 6)
        )
        ttk.Button(acts, text="Clear log", command=self._outcome_clear_log).pack(
            side="left", padx=(0, 6)
        )
        self._btn_outcome_clear_job = ttk.Button(
            acts, text="Clear stuck job", command=self._outcome_clear_stuck_job
        )
        self._btn_outcome_clear_job.pack(side="left")
        ttk.Label(acts, textvariable=self._outcome_status, foreground=COL_MUTED).pack(
            side="left", padx=12
        )

        log_frm = ttk.LabelFrame(
            out, text="Timeline log (build + compute)", padding=6
        )
        log_frm.pack(fill="both", expand=True, pady=(8, 0))
        self._outcome_log = tk.Text(
            log_frm, wrap="none", height=8, font=("Consolas", 9), undo=False
        )
        ylog = ttk.Scrollbar(log_frm, orient="vertical", command=self._outcome_log.yview)
        xlog = ttk.Scrollbar(log_frm, orient="horizontal", command=self._outcome_log.xview)
        self._outcome_log.configure(yscrollcommand=ylog.set, xscrollcommand=xlog.set)
        self._outcome_log.grid(row=0, column=0, sticky="nsew")
        ylog.grid(row=0, column=1, sticky="ns")
        xlog.grid(row=1, column=0, sticky="ew")
        log_frm.rowconfigure(0, weight=1)
        log_frm.columnconfigure(0, weight=1)

        # Explorer
        explorer = ttk.LabelFrame(self._prediction_build_pane, text="Explorer", padding=8)
        explorer.pack(fill="both", expand=True, pady=(4, 0))

        self._applied_filter_frame = ttk.LabelFrame(explorer, text="Applied Filter", padding=8)
        ttk.Label(
            self._applied_filter_frame,
            textvariable=self._explorer_filter_desc,
            font=("Consolas", 10),
            justify="left",
        ).pack(anchor="w")
        ttk.Label(
            self._applied_filter_frame,
            textvariable=self._explorer_filter_count,
            foreground=COL_MUTED,
        ).pack(anchor="w", pady=(2, 4))
        ttk.Button(
            self._applied_filter_frame,
            text="Clear Filter",
            command=self._clear_explorer_feature_filter,
        ).pack(anchor="w")
        # Shown only when a Feature Research filter is active
        self._applied_filter_frame.pack_forget()

        self._explorer_toolbar = ttk.Frame(explorer)
        self._explorer_toolbar.pack(fill="x", pady=(0, 6))
        filt = self._explorer_toolbar
        ttk.Label(filt, text="Search").pack(side="left")
        self._search_var = tk.StringVar()
        ttk.Entry(filt, textvariable=self._search_var, width=28).pack(side="left", padx=6)
        ttk.Button(filt, text="Apply", command=self._reload_explorer).pack(side="left")
        ttk.Button(filt, text="Prev", command=self._explorer_prev).pack(side="left", padx=(12, 0))
        ttk.Button(filt, text="Next", command=self._explorer_next).pack(side="left", padx=4)
        self._explorer_page_lbl = tk.StringVar(value="")
        ttk.Label(filt, textvariable=self._explorer_page_lbl, foreground=COL_MUTED).pack(
            side="left", padx=8,
        )
        ttk.Button(
            filt,
            text="Download CSV",
            command=self._on_download_explorer_csv,
        ).pack(side="right")

        tree_wrap2 = ttk.Frame(explorer)
        tree_wrap2.pack(fill="both", expand=True)
        self._explorer = ttk.Treeview(
            tree_wrap2,
            columns=_EXPLORER_COLS,
            show="headings",
            height=8,
        )
        # Fixed widths + no stretch so late columns stay reachable via horizontal scroll
        _explorer_widths = {
            "trading_day": 88,
            "timestamp": 96,
            "exit_at": 96,
            "current_ltp": 68,
            "predicted_future_ltp": 72,
            "actual_future_ltp": 68,
            "expected_move": 68,
            "actual_move": 68,
            "predicted_trend": 78,
            "actual_trend": 78,
            "direction_correct": 48,
            "maximum_profit": 72,
            "maximum_drawdown": 64,
            "dd_before_target": 88,
            "max_profit_at": 96,
            "max_drawdown_at": 96,
            "time_to_target": 72,
            "target_reached": 48,
            "target_reached_at": 96,
            "time_to_max_profit": 64,
            "time_to_max_drawdown": 64,
            "time_to_dd_before_target": 100,
            "prediction_error": 68,
            "absolute_error": 64,
            "premium_error_pct": 72,
        }
        for col in _EXPLORER_COLS:
            self._explorer.heading(
                col,
                text=_EXPLORER_HEADINGS.get(col, col),
                command=lambda c=col: self._sort_explorer(c),
            )
            w = int(_explorer_widths.get(col, 80))
            self._explorer.column(col, width=w, minwidth=w, stretch=False, anchor="center")
        yscroll2 = ttk.Scrollbar(tree_wrap2, orient="vertical", command=self._explorer.yview)
        xscroll2 = ttk.Scrollbar(tree_wrap2, orient="horizontal", command=self._explorer.xview)
        self._explorer.configure(yscrollcommand=yscroll2.set, xscrollcommand=xscroll2.set)
        tree_wrap2.rowconfigure(0, weight=1)
        tree_wrap2.columnconfigure(0, weight=1)
        self._explorer.grid(row=0, column=0, sticky="nsew")
        yscroll2.grid(row=0, column=1, sticky="ns")
        xscroll2.grid(row=1, column=0, sticky="ew")

        self._build_prediction_metadata_pane()

    def _build_prediction_metadata_pane(self) -> None:
        """Dataset Summary + Column Coverage + Stage Coverage (Phase 1.5/1.6)."""
        pane = self._prediction_metadata_pane

        top = ttk.Frame(pane)
        top.pack(fill="x", pady=(0, 8))
        ttk.Label(
            top,
            text="Auto-refreshes after Build / Compute Outcomes finish a day.",
            foreground=COL_MUTED,
        ).pack(side="left")
        ttk.Button(top, text="Refresh", command=self._on_refresh_prediction_metadata).pack(
            side="right"
        )
        self._pred_meta_updated_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self._pred_meta_updated_var, foreground=COL_MUTED).pack(
            side="right", padx=(0, 10)
        )

        # —— Dataset Summary ——
        summary_box = ttk.LabelFrame(pane, text="Dataset Summary", padding=8)
        summary_box.pack(fill="x", pady=(0, 8))

        stat_row = ttk.Frame(summary_box)
        stat_row.pack(fill="x")
        self._pred_meta_stat_vars: dict[str, tk.StringVar] = {}
        for key, label in (
            ("row_count", "Rows"),
            ("column_count", "Columns"),
            ("trading_days", "Trading Days"),
            ("completed_days", "Completed"),
            ("pending_days", "Pending"),
        ):
            card = tk.Frame(
                stat_row,
                bg=_KPI_CARD_BG,
                highlightbackground="#D0D7DE",
                highlightthickness=1,
                padx=12,
                pady=8,
            )
            card.pack(side="left", fill="x", expand=True, padx=(0, 8))
            tk.Label(
                card, text=label, bg=_KPI_CARD_BG, fg=_KPI_LABEL_FG, font=("Segoe UI", 9),
                anchor="w",
            ).pack(fill="x")
            var = tk.StringVar(value="—")
            self._pred_meta_stat_vars[key] = var
            tk.Label(
                card, textvariable=var, bg=_KPI_CARD_BG, fg=_KPI_VALUE_FG,
                font=("Segoe UI", 14, "bold"), anchor="w",
            ).pack(fill="x")

        # Badges are rebuilt on every render straight from metadata["stages"]
        # (registry-driven order/labels/status text) — nothing hardcoded here.
        self._pred_meta_badge_row = ttk.Frame(summary_box)
        self._pred_meta_badge_row.pack(fill="x", pady=(8, 0))
        ttk.Label(self._pred_meta_badge_row, text="Stages:", foreground=COL_MUTED).pack(
            side="left", padx=(0, 8)
        )

        # —— Stage Coverage ——
        stage_box = ttk.LabelFrame(pane, text="Stage Coverage", padding=8)
        stage_box.pack(fill="x", pady=(0, 8))

        stage_top = ttk.Frame(stage_box)
        stage_top.pack(fill="x", pady=(0, 4))
        ttk.Label(
            stage_top,
            text="Click a row to filter Column Coverage to that stage.",
            foreground=COL_MUTED,
        ).pack(side="left")
        self._pred_meta_filter_var = tk.StringVar(value="Showing: All columns")
        ttk.Label(stage_top, textvariable=self._pred_meta_filter_var, foreground=COL_MUTED).pack(
            side="left", padx=(12, 0)
        )
        ttk.Button(
            stage_top, text="Show All Columns", command=self._clear_pred_meta_stage_filter
        ).pack(side="right")

        stage_cols = ("stage", "status", "expected", "ready", "coverage", "notes")
        self._pred_meta_stage_tree = ttk.Treeview(
            stage_box, columns=stage_cols, show="headings", height=6, selectmode="browse"
        )
        for col, (title, width, anchor) in {
            "stage": ("Stage", 150, "w"),
            "status": ("Status", 110, "w"),
            "expected": ("Expected", 70, "center"),
            "ready": ("Ready", 60, "center"),
            "coverage": ("Coverage", 90, "e"),
            "notes": ("Notes", 400, "w"),
        }.items():
            self._pred_meta_stage_tree.heading(col, text=title)
            self._pred_meta_stage_tree.column(
                col,
                width=width,
                minwidth=width if col != "notes" else 200,
                anchor=anchor,
                stretch=(col == "notes"),
            )
        for bucket, color in _PRED_META_COVERAGE_COLORS.items():
            self._pred_meta_stage_tree.tag_configure(f"cov_{bucket}", foreground=color)
        self._pred_meta_stage_tree.pack(fill="x")
        self._pred_meta_stage_tree.bind(
            "<<TreeviewSelect>>", self._on_pred_meta_stage_selected
        )

        # —— Column Coverage (ALL prediction_dataset columns) ——
        cols_box = ttk.LabelFrame(pane, text="Column Coverage", padding=8)
        cols_box.pack(fill="both", expand=True)

        filter_row = ttk.Frame(cols_box)
        filter_row.pack(fill="x", pady=(0, 6))
        ttk.Label(filter_row, text="Search:").pack(side="left")
        self._pred_meta_search_var = tk.StringVar(value="")
        ttk.Entry(filter_row, textvariable=self._pred_meta_search_var, width=28).pack(
            side="left", padx=(4, 12)
        )
        self._pred_meta_search_var.trace_add("write", self._on_pred_meta_search_changed)
        ttk.Button(filter_row, text="Clear Search", command=self._clear_pred_meta_search).pack(
            side="left"
        )

        cols_wrap = ttk.Frame(cols_box)
        cols_wrap.pack(fill="both", expand=True)
        col_cols = ("column", "stage", "populated", "null", "coverage")
        self._pred_meta_col_tree = ttk.Treeview(
            cols_wrap, columns=col_cols, show="headings", height=14, selectmode="browse"
        )
        for col, (title, width, anchor) in {
            "column": ("Column", 220, "w"),
            "stage": ("Stage", 150, "w"),
            "populated": ("Populated", 100, "e"),
            "null": ("NULL", 100, "e"),
            "coverage": ("Coverage %", 110, "e"),
        }.items():
            self._pred_meta_col_tree.heading(
                col, text=title, command=lambda c=col: self._sort_pred_meta_columns(c)
            )
            self._pred_meta_col_tree.column(
                col, width=width, minwidth=width, anchor=anchor, stretch=(col == "column")
            )
        for bucket, color in _PRED_META_COVERAGE_COLORS.items():
            self._pred_meta_col_tree.tag_configure(f"cov_{bucket}", foreground=color)
        yscroll3 = ttk.Scrollbar(
            cols_wrap, orient="vertical", command=self._pred_meta_col_tree.yview
        )
        self._pred_meta_col_tree.configure(yscrollcommand=yscroll3.set)
        cols_wrap.rowconfigure(0, weight=1)
        cols_wrap.columnconfigure(0, weight=1)
        self._pred_meta_col_tree.grid(row=0, column=0, sticky="nsew")
        yscroll3.grid(row=0, column=1, sticky="ns")
        self._pred_meta_empty_var = tk.StringVar(value="")
        ttk.Label(cols_box, textvariable=self._pred_meta_empty_var, foreground=COL_MUTED).pack(
            anchor="w", pady=(4, 0)
        )

        self._pred_meta_stage_filter: str | None = None
        self._pred_meta_columns_all: list[dict[str, Any]] = []
        self._pred_meta_col_sort_state: tuple[str | None, bool] = (None, False)

    def _on_prediction_subtab_changed(self, _event: object | None = None) -> None:
        if self._lab is None:
            return
        try:
            current = self._prediction_sub_nb.select()
            tab = self._prediction_sub_nb.nametowidget(current)
        except (tk.TclError, KeyError):
            return
        if tab is self._prediction_metadata_pane and not self._pred_meta_loaded:
            self._pred_meta_loaded = True
            self._refresh_prediction_metadata()

    def _on_refresh_prediction_metadata(self) -> None:
        self._refresh_prediction_metadata()

    def _refresh_prediction_metadata(self) -> None:
        """Recompute Prediction Dataset Metadata on a worker thread (SQL aggregates only)."""
        if self._lab is None:
            return
        db_path = self._lab.db_path
        self._pred_meta_gen += 1
        gen = self._pred_meta_gen
        self._pred_meta_updated_var.set("Refreshing…")

        def work() -> dict[str, Any]:
            from chain_replay_ml.model_lab.prediction_dataset_metadata import (
                refresh_prediction_dataset_metadata,
            )

            return refresh_prediction_dataset_metadata(db_path)

        def done() -> None:
            if gen != self._pred_meta_gen:
                return
            try:
                metadata = fut["result"]
            except KeyError:
                self._pred_meta_updated_var.set(f"Error: {fut.get('error')}")
                return
            self._pred_meta = metadata
            self._render_prediction_metadata(metadata)

        fut: dict[str, Any] = {}

        def runner() -> None:
            try:
                fut["result"] = work()
            except Exception as exc:  # noqa: BLE001
                fut["error"] = str(exc)
            try:
                self.after(0, done)
            except tk.TclError:
                pass

        threading.Thread(target=runner, daemon=True, name="lab-pred-metadata").start()

    def _render_prediction_metadata(self, metadata: dict[str, Any]) -> None:
        row_count = int(metadata.get("row_count") or 0)
        self._pred_meta_stat_vars["row_count"].set(_fmt_int(row_count))
        self._pred_meta_stat_vars["column_count"].set(_fmt_int(metadata.get("column_count")))
        self._pred_meta_stat_vars["trading_days"].set(_fmt_int(metadata.get("trading_days")))

        completed_days = int(metadata.get("completed_days") or 0)
        pending_days = int(metadata.get("pending_days") or 0)
        total_days = int(metadata.get("total_catalog_days") or (completed_days + pending_days))
        self._pred_meta_stat_vars["completed_days"].set(
            f"{_fmt_int(completed_days)} / {_fmt_int(total_days)} Days"
        )
        self._pred_meta_stat_vars["pending_days"].set(f"{_fmt_int(pending_days)} Days")

        stages = list(metadata.get("stages") or [])

        # Registry-driven badges — order/labels/status text come straight from
        # the payload, never a hardcoded stage list in this Tk file.
        for child in list(self._pred_meta_badge_row.winfo_children())[1:]:
            child.destroy()
        for stage in stages:
            label = str(stage.get("name") or "")
            status_label = str(stage.get("status_label") or "")
            bucket = str(stage.get("coverage_bucket") or "empty")
            emoji = _PRED_META_COVERAGE_EMOJI.get(bucket, "")
            ttk.Label(
                self._pred_meta_badge_row,
                text=f"{emoji} {label}: {status_label}",
                padding=(6, 2),
            ).pack(side="left", padx=(0, 4))

        valid_labels = {str(s.get("name") or "") for s in stages}
        if self._pred_meta_stage_filter and self._pred_meta_stage_filter not in valid_labels:
            self._pred_meta_stage_filter = None

        self._pred_meta_stage_tree.delete(*self._pred_meta_stage_tree.get_children())
        for stage in stages:
            label = str(stage.get("name") or "")
            status_label = str(stage.get("status_label") or "")
            expected = int(stage.get("expected") or 0)
            ready = int(stage.get("ready") or 0)
            cov = float(stage.get("coverage_pct") or 0.0)
            bucket = str(stage.get("coverage_bucket") or "empty")
            emoji = _PRED_META_COVERAGE_EMOJI.get(bucket, "")
            notes = str(stage.get("notes") or "")
            iid = self._pred_meta_stage_tree.insert(
                "",
                "end",
                values=(label, status_label, expected, ready, f"{emoji} {cov:.1f}%", notes),
                tags=(f"cov_{bucket}",),
            )
            if label == self._pred_meta_stage_filter:
                self._pred_meta_stage_tree.selection_set(iid)

        self._pred_meta_columns_all = list(metadata.get("columns") or [])
        self._apply_pred_meta_column_filter()

        generated_at = str(metadata.get("generated_at") or "").strip()
        self._pred_meta_updated_var.set(
            f"Updated {generated_at}" if generated_at else ""
        )

    def _on_pred_meta_stage_selected(self, _event: object | None = None) -> None:
        """Requirement #2 — clicking a Stage Coverage row filters Column Coverage."""
        selection = self._pred_meta_stage_tree.selection()
        if not selection:
            return
        values = self._pred_meta_stage_tree.item(selection[0], "values")
        if not values:
            return
        self._pred_meta_stage_filter = str(values[0])
        self._apply_pred_meta_column_filter()

    def _clear_pred_meta_stage_filter(self) -> None:
        self._pred_meta_stage_filter = None
        try:
            self._pred_meta_stage_tree.selection_remove(*self._pred_meta_stage_tree.selection())
        except tk.TclError:
            pass
        self._apply_pred_meta_column_filter()

    def _on_pred_meta_search_changed(self, *_args: object) -> None:
        self._apply_pred_meta_column_filter()

    def _clear_pred_meta_search(self) -> None:
        self._pred_meta_search_var.set("")

    def _sort_pred_meta_columns(self, col: str) -> None:
        cur_key, cur_reverse = self._pred_meta_col_sort_state
        reverse = (not cur_reverse) if cur_key == col else False
        self._pred_meta_col_sort_state = (col, reverse)
        self._apply_pred_meta_column_filter()

    def _apply_pred_meta_column_filter(self) -> None:
        """Stage-click filter AND substring search AND heading-click sort."""
        rows = list(self._pred_meta_columns_all)
        stage_filter = self._pred_meta_stage_filter
        if stage_filter:
            rows = [r for r in rows if str(r.get("stage") or "") == stage_filter]

        search = self._pred_meta_search_var.get().strip().lower()
        if search:
            rows = [r for r in rows if search in str(r.get("name") or "").lower()]

        sort_key, reverse = self._pred_meta_col_sort_state
        if sort_key:
            def _sort_value(r: dict[str, Any]) -> Any:
                if sort_key == "column":
                    return str(r.get("name") or "").lower()
                if sort_key == "stage":
                    return str(r.get("stage") or "").lower()
                if sort_key == "populated":
                    return int(r.get("populated") or 0)
                if sort_key == "null":
                    return int(r.get("null") or 0)
                if sort_key == "coverage":
                    return float(r.get("coverage_pct") or 0.0)
                return 0

            rows = sorted(rows, key=_sort_value, reverse=reverse)

        self._pred_meta_col_tree.delete(*self._pred_meta_col_tree.get_children())
        for col in rows:
            bucket = str(col.get("coverage_bucket") or "empty")
            emoji = _PRED_META_COVERAGE_EMOJI.get(bucket, "")
            self._pred_meta_col_tree.insert(
                "",
                "end",
                values=(
                    col.get("name"),
                    col.get("stage"),
                    _fmt_int(col.get("populated")),
                    _fmt_int(col.get("null")),
                    f"{emoji} {float(col.get('coverage_pct') or 0.0):.2f}%",
                ),
                tags=(f"cov_{bucket}",),
            )

        total = len(self._pred_meta_columns_all)
        shown = len(rows)
        row_count = int((self._pred_meta or {}).get("row_count") or 0)
        if row_count <= 0:
            self._pred_meta_empty_var.set(
                "No prediction rows yet — build the Prediction Dataset first."
            )
        elif shown == total:
            self._pred_meta_empty_var.set(f"{total} columns · {_fmt_int(row_count)} rows")
        else:
            filters = []
            if stage_filter:
                filters.append(f"stage = {stage_filter}")
            if search:
                filters.append(f"search = '{search}'")
            self._pred_meta_empty_var.set(
                f"{shown} of {total} columns shown ({'; '.join(filters)}) · "
                f"{_fmt_int(row_count)} rows"
            )

        self._pred_meta_filter_var.set(
            f"Showing: {stage_filter} columns" if stage_filter else "Showing: All columns"
        )

    def _reset_tabs(self, *, has_lab: bool) -> None:
        # Preserve selected tab text across remount when possible
        selected_text = ""
        try:
            cur = self._notebook.select()
            if cur:
                selected_text = str(self._notebook.tab(cur, "text") or "")
        except tk.TclError:
            selected_text = ""

        for tab_id in list(self._notebook.tabs()):
            self._notebook.forget(tab_id)
        if has_lab:
            self._notebook.add(self._overview_tab, text="Overview")
            self._notebook.add(self._prediction_tab, text="Prediction Dataset")
            self._notebook.add(self._research_tab, text="Research Dashboard")
            self._notebook.add(self._strategy_sim_tab, text="Strategy Simulator")
            self._notebook.add(self._strike_dashboard_tab, text="Strike Dashboard")
            self._notebook.add(self._feature_research_tab, text="Feature Research")
            self._notebook.add(self._research_programs_tab, text="Research Programs")
            self._notebook.add(self._model_improvement_tab, text="Model Improvement")
            self._notebook.add(self._confidence_tab, text="Confidence")
            self._notebook.add(self._rr_validation_tab, text="RR Validation")
            tab_by_text = {
                "Overview": self._overview_tab,
                "Prediction Dataset": self._prediction_tab,
                "Research Dashboard": self._research_tab,
                "Strategy Simulator": self._strategy_sim_tab,
                "Strike Dashboard": self._strike_dashboard_tab,
                "Feature Research": self._feature_research_tab,
                "Research Programs": self._research_programs_tab,
                "Model Improvement": self._model_improvement_tab,
                "Confidence Model": self._confidence_tab,
                "Confidence": self._confidence_tab,
                "RR Validation": self._rr_validation_tab,
            }
            restore = tab_by_text.get(selected_text)
            if restore is None and not selected_text and not self._initial_tab:
                # Fresh window (no remount, no explicit deep-link) — fall back to
                # whichever tab the user had open last session.
                saved_text = str(self._ui_state.get("model_lab.tab") or "")
                restore = tab_by_text.get(saved_text)
            self._notebook.select(restore or self._overview_tab)
        else:
            self._notebook.add(self._empty_tab, text="Research Lab")
            self._notebook.select(self._empty_tab)
        # Honor a pending initial tab (e.g. open from Models → Research)
        if has_lab and self._initial_tab:
            self._apply_initial_tab()

    def _empty_state_detail(self) -> str:
        from chain_replay_ml.model_lab import list_research_lab_summaries, resolve_model_research_dir
        from chain_replay_ml.model_lab.paths import lab_db_stem

        root = resolve_model_research_dir()
        stem = lab_db_stem(self.model_name)
        lines = [
            f"Model: {self.model_name}",
            f"Looking for: {stem}_v*.db",
            f"Folder: {root}",
            "",
            "This model has no research workspace yet.",
            "Create one (Start Research) to store prediction datasets.",
        ]
        others = [
            s
            for s in list_research_lab_summaries(research_dir=root)
            if str(s.get("parent_model_name") or "") != self.model_name
        ]
        if others:
            lines.append("")
            lines.append("Other labs already on disk (select that model to open):")
            for s in others[:8]:
                rows = int(s.get("prediction_row_count") or 0)
                lines.append(
                    f"  • {s.get('parent_model_name')}  v{s.get('version')}  "
                    f"({rows:,} prediction rows)"
                )
        return "\n".join(lines)

    def refresh(self) -> None:
        from chain_replay_ml.model_lab import find_latest_lab

        try:
            self._lab = find_latest_lab(self.model_name)
        except Exception as exc:
            self._lab = None
            self._reset_tabs(has_lab=False)
            self._status.set("Status: Error")
            self._empty_status.set("Status\nError")
            self._empty_detail.set(f"Failed to open Research Lab:\n{exc}")
            self.title(f"Research Lab — {self.model_name}")
            if not self._build_running():
                self._set_status_bar(status="Error", detail=str(exc))
            return

        if self._lab is None:
            self._reset_tabs(has_lab=False)
            self._status.set("Status: Not Created")
            self._empty_status.set("Status\nNot Created")
            self._empty_detail.set(self._empty_state_detail())
            self.title(f"Research Lab — {self.model_name}")
            if not self._build_running():
                self._set_status_bar(status="Ready", detail="No research lab yet")
            return

        self._reset_tabs(has_lab=True)
        self._status.set(
            f"{self._lab.status} · Phase {self._lab.phase} · v{self._lab.version}"
        )
        self.title(f"Research Lab — {self.model_name} · v{self._lab.version}")
        self._render_overview(self._lab)
        self._try_reattach_prediction_job()
        self._refresh_prediction_tab()
        self._refresh_research_dashboard()
        # Feature Research is a separate heavy workload — compute only when that tab opens.
        self._refresh_research_programs_list()
        # Model Improvement is heavy — only via Start Analysis on that tab.
        self._refresh_confidence_tab()

    def _try_reattach_prediction_job(self) -> None:
        """Reclaim orphan prediction jobs after GUI reopen; unlock Build if dead."""
        if self._lab is None:
            return
        try:
            mgr = self._get_pred_manager()
            # Finalize jobs whose workers died / went heartbeat-stale while the
            # GUI was closed — otherwise status stays "running" forever.
            reclaimed = mgr.reclaim_stale_active_jobs()
            for snap in reclaimed:
                if snap.get("reclaimed"):
                    jid = str((snap.get("job") or {}).get("job_id") or snap.get("job_id") or "")
                    self._outcome_log_line(
                        f"Cleared stuck job {jid or '?'} "
                        f"(status={snap.get('status')}, no live workers). "
                        "Build buttons re-enabled — start a new run."
                    )
            jid = mgr.active_job_id()
            if not jid:
                self._pred_job_id = None
                return
            snap = mgr.progress(jid)
            status = str(snap.get("status") or "")
            alive = int(snap.get("workers_alive") or 0)
            if status in ("completed", "failed", "cancelled") or alive <= 0:
                self._pred_job_id = None
                if status in ("failed", "cancelled") or alive <= 0:
                    self._outcome_status.set(
                        f"Previous job {jid} ended ({status or 'reclaimed'})"
                    )
                    if not any(s.get("reclaimed") for s in reclaimed):
                        self._outcome_log_line(
                            f"Cleared stuck job {jid} (status={status}, alive={alive}). "
                            "Build buttons re-enabled — start a new run."
                        )
                return
            # Truly live workers from a prior session — keep monitoring unlocked
            # via Clear stuck job; do not leave Build disabled with no cancel path.
            self._pred_job_id = jid
            self._outcome_status.set(
                f"Job {jid} still running ({status}, {alive} worker(s)) — "
                "use Clear stuck job to unlock"
            )
            self._outcome_log_line(
                f"Detected live job {jid} after relaunch ({alive} worker(s)). "
                "Waiting for it to finish, or click Clear stuck job to abandon it."
            )
        except Exception:
            pass

    def _outcome_clear_stuck_job(self) -> None:
        """Abandon a stale/live prediction job left over after relaunch."""
        if self._lab is None:
            return
        try:
            mgr = self._get_pred_manager()
            jid = self._pred_job_id or mgr.active_job_id()
            if not jid:
                # Still try reclaim in case DB says running with dead workers.
                cleared = mgr.reclaim_stale_active_jobs()
                if cleared:
                    self._outcome_status.set("Cleared stale prediction job(s)")
                    self._outcome_log_line("Reclaimed stale prediction job(s) from lab DB.")
                else:
                    self._outcome_status.set("No active prediction job")
                self._pred_job_id = None
                self._set_build_buttons(running=False)
                return
            if not messagebox.askyesno(
                "Clear stuck job",
                f"Abandon prediction job {jid}?\n\n"
                "Worker processes will be terminated and Build buttons unlocked.\n"
                "Already-written prediction rows are kept.",
                parent=self,
            ):
                return
            snap = mgr.abandon_job(jid, reason="Cleared stuck job from Research Lab UI")
            self._pred_job_id = None
            self._outcome_stop_log.set()
            self._set_build_buttons(running=False)
            self._outcome_status.set(f"Cleared job {jid}")
            self._outcome_log_line(
                f"Abandoned job {jid} (status={snap.get('status')}). "
                "Build buttons re-enabled."
            )
            self._refresh_prediction_tab(full_metadata=False)
        except Exception as exc:
            messagebox.showerror("Clear stuck job", str(exc), parent=self)

    def _refresh_prediction_tab(
        self,
        *,
        defer_explorer: bool = False,
        full_metadata: bool = False,
    ) -> None:
        """Paint Prediction tab from Master + prediction_metadata.json.

        ``full_metadata=True`` (Refresh Days) verifies/rebuilds the sidecar from DB.
        Default open path never scans prediction_dataset.
        """
        if self._lab is None:
            return
        lab = self._lab
        db_path = lab.db_path
        data_dir = self._data_dir()
        running = self._build_running()
        self._set_build_buttons(running=running)
        self._pred_refresh_gen += 1
        gen = self._pred_refresh_gen
        self._live_progress_var.set(
            "Refreshing trading-day metadata…" if full_metadata else "Loading trading days…"
        )
        self._set_status_bar(
            status="Loading…",
            detail=(
                "Refresh Days — verifying DB and rebuilding metadata…"
                if full_metadata
                else "Master days + prediction_metadata.json…"
            ),
            active=True,
        )

        def work() -> None:
            err: Exception | None = None
            summary: dict[str, Any] = {}
            st: dict[str, Any] = {}
            registry_rows: dict[str, int] = {}
            parent_name: str | None = None
            try:
                from chain_replay_ml.model_lab.prediction_builder import (
                    prediction_build_summary,
                    prediction_dataset_status,
                    prediction_days_ui_skeleton,
                    sync_prediction_build_catalog,
                )

                if full_metadata:
                    result = sync_prediction_build_catalog(data_dir, db_path)
                    summary = (
                        result.get("summary")
                        if result.get("ok")
                        else None
                    ) or prediction_build_summary(db_path, data_dir=data_dir)
                    if result.get("ok"):
                        registry_rows = {
                            str(k): int(v)
                            for k, v in (result.get("rows_by_day") or {}).items()
                        }
                        parent_name = str(result.get("dataset") or "") or None
                    st = prediction_dataset_status(db_path, light=True)
                else:
                    summary = prediction_days_ui_skeleton(data_dir, db_path)
                    parent_name = str(summary.get("parent_dataset") or "") or None
                    st = prediction_dataset_status(db_path, light=True)
                    registry_rows = {
                        str(k): int(v)
                        for k, v in (summary.get("rows_by_day") or {}).items()
                    }
            except Exception as exc:  # noqa: BLE001
                err = exc

            def done() -> None:
                if gen != self._pred_refresh_gen or self._lab is None:
                    return
                if err is not None:
                    self._live_progress_var.set(f"Prediction tab error: {err}")
                    self._set_status_bar(status="Error", detail=str(err), active=False)
                    self._set_build_buttons(running=self._build_running())
                    return
                self._registry_rows_by_day = registry_rows
                self._parent_dataset = parent_name
                self._day_inv = [
                    (
                        day,
                        int(registry_rows.get(day) or d.get("rows_expected") or 0),
                        0,
                        int(d.get("row_count") or 0),
                    )
                    for d in (summary.get("days") or [])
                    for day in [str(d.get("trading_day") or "")]
                    if day
                ]
                self._apply_prediction_tab_snapshot(
                    summary,
                    st,
                    running=self._build_running(),
                    defer_explorer=defer_explorer,
                )
                self._set_build_buttons(running=self._build_running())

            try:
                self.after(0, done)
            except tk.TclError:
                pass

        threading.Thread(target=work, daemon=True, name="lab-pred-refresh").start()

    def _apply_prediction_tab_snapshot(
        self,
        summary: dict[str, Any],
        st: dict[str, Any],
        *,
        running: bool,
        defer_explorer: bool = False,
    ) -> None:
        from chain_replay_ml.model_lab.prediction_schema import PRED_STATUS_READY

        self._render_build_summary(summary)
        self._render_day_table(summary.get("days") or [])
        if not summary.get("days"):
            self._live_progress_var.set(
                "No Master trading days found — check Master DB link, or click Refresh Days."
            )
        else:
            self._live_progress_var.set("")

        status = str(st.get("status") or "not_generated")
        n = int(st.get("row_count") or 0)
        self._explorer_total_cache = n if n > 0 else None
        if status == PRED_STATUS_READY and n > 0:
            self._btn_export.configure(state="normal")
            if defer_explorer:
                self._clear_explorer()
                self._explorer_page_lbl.set(f"Loading first page… / {_fmt_int(n)}")
                self.after(30, self._reload_explorer_async)
            else:
                self._reload_explorer_async()
            if not running:
                self._set_status_bar(
                    status="Ready",
                    percent=100.0 if int(summary.get("remaining") or 0) == 0 else None,
                    detail=(
                        f"Prediction Dataset · {_fmt_int(n)} rows · "
                        f"{_fmt_int(summary.get('completed'))}/{_fmt_int(summary.get('total_days'))} days"
                    ),
                    active=False,
                )
        else:
            self._btn_export.configure(state="disabled" if n <= 0 else "normal")
            if n <= 0:
                self._clear_explorer()
                self._explorer_total_cache = None
            elif defer_explorer:
                self._clear_explorer()
                self._explorer_page_lbl.set(f"Loading first page… / {_fmt_int(n)}")
                self.after(30, self._reload_explorer_async)
            else:
                self._reload_explorer_async()
            if not running:
                self._set_status_bar(
                    status="Ready",
                    detail=(
                        f"Completed Days {_fmt_int(summary.get('completed'))}/"
                        f"{_fmt_int(summary.get('total_days'))}"
                        if summary.get("total_days")
                        else "Prediction Dataset not generated"
                    ),
                    active=False,
                )

    def _ensure_day_metadata(self, trading_day: str) -> dict[str, Any] | None:
        """Return Trading Days row from UI metadata (no prediction.db scan)."""
        day = str(trading_day or "").strip()
        if not day or self._lab is None:
            return None
        row = self._day_rows.get(day) or {}
        if _day_row_meta_ready(row):
            return row
        # Prefer sidecar over any DB enrich — Build/Compute must not COUNT(*).
        from chain_replay_ml.model_lab.prediction_metadata import (
            merge_master_and_metadata,
            read_prediction_metadata,
        )

        meta = read_prediction_metadata(self._lab.db_path)
        master_rows = {
            d: int(self._registry_rows_by_day.get(d) or 0)
            for d in set(self._registry_rows_by_day) | {day}
        }
        if day not in master_rows:
            master_rows[day] = int(row.get("rows_expected") or row.get("dataset_rows") or 0)
        merged_list = merge_master_and_metadata(master_rows, meta)
        enriched = next(
            (d for d in merged_list if str(d.get("trading_day")) == day),
            None,
        )
        if enriched is None:
            enriched = {
                "trading_day": day,
                "status": "waiting",
                "row_count": 0,
                "rows_expected": master_rows.get(day) or 0,
                "ui_meta_ready": True,
                "note": "pred missing — build needed",
            }
        enriched = dict(enriched)
        enriched["ui_meta_ready"] = True
        ds_n = int(
            self._registry_rows_by_day.get(day)
            or enriched.get("rows_expected")
            or 0
        )
        pred_n = int(enriched.get("row_count") or 0)
        enriched["dataset_rows"] = ds_n
        enriched["pred_rows"] = pred_n
        self._day_rows[day] = enriched
        if self._day_tree.exists(day):
            from chain_replay_ml.model_lab.prediction_schema import normalize_dataset_type

            st = str(enriched.get("status") or "waiting")
            # Coverage-accurate status: Complete only when every dataset row
            # has a prediction row — never trust a stale/raw "completed" that
            # predates a partial-row build. Never override in-flight/failed/
            # cancelled states.
            if pred_n > 0 and st in ("completed", "partial", "waiting", "skipped", ""):
                st = "completed" if (not ds_n or pred_n >= ds_n) else "partial"
            status_lbl = _STATUS_LABEL.get(st, st)
            dataset_type = normalize_dataset_type(enriched.get("dataset_type"))
            note = str(enriched.get("note") or "")
            if not note:
                if ds_n and not pred_n:
                    note = "pred missing — build needed"
                elif pred_n and ds_n and pred_n < ds_n:
                    note = f"partial vs dataset ({pred_n:,}/{ds_n:,})"
                elif st == "completed" or (pred_n and ds_n and pred_n >= ds_n):
                    note = "complete"
            build_sec = enriched.get("build_time_sec")
            if build_sec is not None:
                try:
                    time_taken = _fmt_sec(build_sec)
                except Exception:
                    time_taken = _fmt_day_duration(
                        enriched.get("started_at"), enriched.get("finished_at")
                    )
            else:
                time_taken = _fmt_day_duration(
                    enriched.get("started_at"), enriched.get("finished_at")
                )
            dash_lbl = "Open" if pred_n > 0 else "—"
            self._day_tree.item(
                day,
                values=(
                    day,
                    dataset_type,
                    _fmt_int(ds_n),
                    _fmt_int(pred_n),
                    status_lbl,
                    time_taken,
                    note,
                    dash_lbl,
                ),
            )
        self._on_outcome_day_select()
        return enriched

    def _sync_and_refresh_days(self, silent: bool = False) -> None:
        """Full refresh: verify DB catalog and rebuild prediction_metadata.json."""
        _ = silent
        self._refresh_prediction_tab(defer_explorer=True, full_metadata=True)

    def _on_backfill_time_to_target(self) -> None:
        """Migrate existing rows: compute Time To Target without rebuilding predictions."""
        if self._lab is None:
            return
        if self._build_running():
            messagebox.showinfo("Time To Target", "Wait for the current build to finish.", parent=self)
            return
        ok = messagebox.askyesno(
            "Fill Path Timestamps",
            "Fill absolute path timestamps for existing rows:\n\n"
            "• Target Reached Ts (NULL if never)\n"
            "• Max Profit Ts (MFE)\n"
            "• Max DD Ts (MAE)\n"
            "• Exit Ts (entry + horizon)\n"
            "• Time To Target (seconds)\n"
            "• DD Before Target + T→DD Before Target\n\n"
            "Does not change Current / Predicted / Actual / Max Profit / Max DD values.\n\n"
            "Continue? (may take several minutes)",
            parent=self,
        )
        if not ok:
            return

        data_dir = self._data_dir()
        db_path = self._lab.db_path
        self._live_progress_var.set("Backfilling Time To Target…")
        self._set_status_bar(status="Migrating…", detail="Time To Target backfill", active=True)

        def _worker() -> None:
            from chain_replay_ml.model_lab.prediction_migrate import backfill_time_to_target

            def on_prog(p: dict[str, Any]) -> None:
                self._progress = {**p, "phase": "migrate"}

            result = backfill_time_to_target(data_dir, db_path, on_progress=on_prog)
            self._progress = {**(self._progress or {}), "result": result, "phase": "migrate_done"}

        self._build_thread = threading.Thread(target=_worker, name="ttt-backfill", daemon=True)
        self._build_thread.start()
        self.after(400, self._poll_backfill_progress)

    def _poll_backfill_progress(self) -> None:
        p = self._progress or {}
        if self._build_running():
            day = str(p.get("current_day") or "")
            done = p.get("days_done")
            total = p.get("days_total")
            upd = p.get("updated")
            self._live_progress_var.set(
                f"Backfilling Time To Target\n"
                f"Day: {day or '—'}\n"
                f"Days: {_fmt_int(done)} / {_fmt_int(total)}\n"
                f"Rows updated: {_fmt_int(upd)}"
            )
            self.after(400, self._poll_backfill_progress)
            return
        self._build_thread = None
        result = p.get("result") if isinstance(p.get("result"), dict) else {}
        self._live_progress_var.set("")
        self._set_status_bar(
            status="Ready",
            detail=f"Time To Target · updated {_fmt_int(result.get('updated'))}",
            active=False,
        )
        self._reload_explorer()
        if result.get("ok"):
            messagebox.showinfo(
                "Fill T→Target",
                f"Updated: {_fmt_int(result.get('updated'))}\n"
                f"Missed target: {_fmt_int(result.get('missed'))}",
                parent=self,
            )
        elif result:
            messagebox.showerror(
                "Fill T→Target",
                str(result.get("error") or "Backfill failed"),
                parent=self,
            )

    def _filter_seen_days(self, days: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Build All scope — Seen days only (Unseen requires explicit user select)."""
        from chain_replay_ml.model_lab.prediction_schema import (
            DATASET_TYPE_SEEN,
            normalize_dataset_type,
        )

        return [
            d for d in days
            if isinstance(d, dict)
            and normalize_dataset_type(d.get("dataset_type")) == DATASET_TYPE_SEEN
        ]

    def _render_build_summary(self, summary: dict[str, Any]) -> None:
        from chain_replay_ml.model_lab.prediction_schema import (
            DATASET_TYPE_UNSEEN,
            normalize_dataset_type,
        )

        days = [d for d in (summary.get("days") or []) if isinstance(d, dict)]
        seen_n = 0
        unseen_n = 0
        completed = 0
        remaining = 0
        for d in days:
            dtype = normalize_dataset_type(d.get("dataset_type"))
            if dtype == DATASET_TYPE_UNSEEN:
                unseen_n += 1
            else:
                seen_n += 1
            st = str(d.get("status") or "")
            if st == "completed":
                completed += 1
            elif st != "skipped":
                remaining += 1

        self._build_summary_var.set(
            f"Seen Days: {_fmt_int(seen_n)} · "
            f"Unseen Days: {_fmt_int(unseen_n)} · "
            f"Completed: {_fmt_int(completed)} · "
            f"Remaining: {_fmt_int(remaining)}"
        )

    def _render_day_table(
        self,
        days: list[dict[str, Any]],
        *,
        live: dict[str, Any] | None = None,
    ) -> None:
        _ = live  # live progress unused in single-day bench-style table
        inv = {d: (ds, m, p) for d, ds, m, p in self._day_inv}
        prev = ""
        try:
            sel = self._day_tree.selection()
            if sel:
                prev = str(sel[0])
        except tk.TclError:
            prev = ""

        from chain_replay_ml.model_lab.prediction_schema import normalize_dataset_type

        for iid in self._day_tree.get_children():
            self._day_tree.delete(iid)
        self._day_rows = {}
        for d in days:
            if not isinstance(d, dict):
                continue
            day = str(d.get("trading_day") or "")
            if not day:
                continue
            st = str(d.get("status") or "waiting")
            pred_n = int(d.get("row_count") or 0)
            ds_n, _m, inv_pred = inv.get(day, (0, 0, pred_n))
            # Prefer registry sources[].rows; then Refresh Days rows_expected; then inventory.
            if not ds_n:
                ds_n = int(self._registry_rows_by_day.get(day) or 0)
            if not ds_n:
                ds_n = int(d.get("rows_expected") or 0)
            if pred_n <= 0 and inv_pred:
                pred_n = int(inv_pred)
            # Coverage-accurate status: a day only counts as Complete when
            # every dataset row has a corresponding prediction row — never
            # trust a raw "completed" store status on its own (a day can
            # finish building with fewer valid rows than the dataset).
            # Never override in-flight/failed/cancelled states.
            if pred_n > 0 and st in ("completed", "partial", "waiting", "skipped", ""):
                st = "completed" if (not ds_n or pred_n >= ds_n) else "partial"
            status_lbl = _STATUS_LABEL.get(st, st)
            note = ""
            if isinstance(d.get("note"), str) and d.get("note").strip():
                note = str(d.get("note")).strip()
            elif ds_n and not pred_n:
                note = "pred missing — build needed"
            elif pred_n and ds_n and pred_n < ds_n:
                note = f"partial vs dataset ({pred_n:,}/{ds_n:,})"
            elif st == "completed" or (pred_n and ds_n and pred_n >= ds_n):
                note = "complete"
            started = d.get("started_at")
            finished = d.get("finished_at")
            if (not started or not finished) and day in self._day_local_timing:
                started, finished = self._day_local_timing[day]
            build_sec = d.get("build_time_sec")
            if build_sec is not None:
                try:
                    time_taken = _fmt_sec(build_sec)
                except Exception:
                    time_taken = _fmt_day_duration(started, finished)
            else:
                time_taken = _fmt_day_duration(started, finished)
            # Per-model stored metadata (training days → Seen; else Unseen).
            # Legacy missing → Seen via normalize.
            ready = _day_row_meta_ready(d)
            if ready:
                dataset_type = normalize_dataset_type(d.get("dataset_type"))
            else:
                dataset_type = None
            dash_lbl = "Open" if (ready and pred_n > 0) else ("—" if ready else "…")
            self._day_tree.insert(
                "",
                "end",
                iid=day,
                values=(
                    day,
                    _fmt_day_meta_cell(dataset_type, ready=ready),
                    _fmt_day_meta_cell(_fmt_int(ds_n) if ds_n else None, ready=ready),
                    _fmt_day_meta_cell(_fmt_int(pred_n) if pred_n else None, ready=ready),
                    _fmt_day_meta_cell(status_lbl, ready=ready),
                    _fmt_day_meta_cell(time_taken if time_taken != "—" else None, ready=ready),
                    _fmt_day_meta_cell(note or None, ready=ready),
                    dash_lbl,
                ),
            )
            row = dict(d)
            row["ui_meta_ready"] = ready
            row["dataset_type"] = dataset_type if ready else None
            row["dataset_rows"] = ds_n if ready else 0
            row["pred_rows"] = pred_n if ready else 0
            self._day_rows[day] = row

        if prev and self._day_tree.exists(prev):
            self._day_tree.selection_set(prev)
            self._day_tree.see(prev)
            self._on_outcome_day_select()
        elif self._day_tree.get_children():
            last = self._day_tree.get_children()[-1]
            self._day_tree.selection_set(last)
            self._day_tree.see(last)
            self._on_outcome_day_select()

    def _on_day_tree_click(self, event: tk.Event) -> None:  # type: ignore[type-arg]
        region = self._day_tree.identify("region", event.x, event.y)
        if region != "cell":
            return
        col = self._day_tree.identify_column(event.x)
        iid = self._day_tree.identify_row(event.y)
        if not iid or col != f"#{len(_DAY_TABLE_COLS)}":
            return
        row = self._day_rows.get(str(iid)) or {}
        if int(row.get("pred_rows") or 0) <= 0:
            return
        self._open_day_research_dashboard(str(iid))

    def _selected_days_from_tree(self) -> list[str]:
        sel = self._day_tree.selection()
        if not sel:
            return []
        return [str(sel[0])]

    def _selected_day_inv_row(self) -> tuple[str, int, int, int] | None:
        days = self._selected_days_from_tree()
        if not days:
            return None
        day = days[0]
        row = self._day_rows.get(day) or {}
        ds_n = int(
            row.get("dataset_rows")
            or self._registry_rows_by_day.get(day)
            or 0
        )
        if not ds_n:
            for d, ds, _m, _p in self._day_inv:
                if d == day:
                    ds_n = int(ds or 0)
                    break
        return (
            day,
            ds_n,
            0,
            int(row.get("pred_rows") or row.get("row_count") or 0),
        )

    def _on_outcome_day_select(self) -> None:
        row = self._selected_day_inv_row()
        if not row:
            return
        day, dataset_n, _master_n, pred_n = row
        parent = self._parent_dataset or "parent registry"
        self._row_hint.set(
            f"{day}: dataset={dataset_n:,}  prediction={pred_n:,}. "
            f"Build max = dataset rows ({parent}); blank = all dataset."
        )

    def _persist_day_selection(self) -> None:
        if self._lab is None:
            return
        from chain_replay_ml.model_lab.store import ModelLabStore

        selected = self._selected_days_from_tree()
        with ModelLabStore(self._lab.db_path) as store:
            lab = store.read_info()
            if lab is None:
                return
            store.set_days_selected(lab.lab_uuid, selected)

    def _update_selected_count_label(self) -> None:
        from chain_replay_ml.model_lab.prediction_builder import prediction_build_summary

        if self._lab is None:
            return
        summary = dict(
            prediction_build_summary(self._lab.db_path, data_dir=self._data_dir(), light=True)
        )
        summary["selected"] = len(self._selected_days_from_tree())
        self._render_build_summary(summary)

    def _select_all_days(self) -> None:
        return

    def _select_no_days(self) -> None:
        return

    def _select_incomplete_days(self) -> None:
        return

    def _outcome_tools_dir(self) -> str:
        return os.path.join(self.chart_dir, "tools")

    def _outcome_parse_int(self, raw: str, *, allow_empty: bool = True) -> int | None:
        text = str(raw or "").strip().replace(",", "").replace("_", "")
        if not text:
            if allow_empty:
                return None
            raise ValueError("Value required")
        n = int(text)
        if n <= 0:
            raise ValueError("Must be a positive integer")
        return n

    def _outcome_selected_chunks(self) -> list[int]:
        return [w for w in _OUTCOME_CHUNK_CHOICES if self._worker_vars[w].get()]

    def _outcome_log_line(self, msg: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {msg}"

        def _do() -> None:
            try:
                self._outcome_log.insert(tk.END, line + "\n")
                self._outcome_log.see(tk.END)
                self.update_idletasks()
            except tk.TclError:
                pass

        if threading.current_thread() is threading.main_thread():
            _do()
        else:
            try:
                self.after(0, _do)
            except tk.TclError:
                pass

    def _outcome_clear_log(self) -> None:
        try:
            self._outcome_log.delete("1.0", tk.END)
        except tk.TclError:
            pass

    def _outcome_copy_log(self) -> None:
        try:
            sel = self._outcome_log.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            sel = ""
        payload = sel if sel.strip() else self._outcome_log.get("1.0", tk.END)
        payload = payload.strip("\n") + "\n" if payload.strip() else ""
        if not payload.strip():
            messagebox.showinfo("Copy", "Log is empty.", parent=self)
            return
        self.clipboard_clear()
        self.clipboard_append(payload)
        self.update_idletasks()
        self._outcome_status.set("Log copied")

    def _outcome_use_all_dataset(self) -> None:
        row = self._selected_day_inv_row()
        if not row:
            messagebox.showinfo("Day", "Select a trading day first.", parent=self)
            return
        day = row[0]
        if self._ensure_day_metadata(day) is None:
            return
        row = self._selected_day_inv_row()
        assert row is not None
        day, dataset_n, _m, _p = row
        if dataset_n <= 0:
            messagebox.showinfo(
                "Dataset",
                f"{day} has 0 parent-dataset rows.",
                parent=self,
            )
            return
        self._row_limit.set(str(dataset_n))
        self._outcome_status.set(f"Row count set to dataset total {dataset_n:,}")

    def _outcome_use_all_pred(self) -> None:
        row = self._selected_day_inv_row()
        if not row:
            messagebox.showinfo("Day", "Select a trading day first.", parent=self)
            return
        day = row[0]
        if self._ensure_day_metadata(day) is None:
            return
        row = self._selected_day_inv_row()
        assert row is not None
        day, _ds, _m, pred_n = row
        if pred_n <= 0:
            messagebox.showinfo("Prediction", f"No prediction rows for {day} yet.", parent=self)
            return
        self._row_limit.set(str(pred_n))
        self._outcome_status.set(f"Row count set to prediction total {pred_n:,}")

    def _outcome_validate(self) -> tuple[str, str, str] | None:
        if self._lab is None:
            messagebox.showerror("Lab", "No Research Lab open.", parent=self)
            return None
        day_row = self._selected_day_inv_row()
        data_dir = self._data_dir()
        if not day_row:
            messagebox.showerror("Day", "Select a trading day from the list.", parent=self)
            return None
        if not data_dir or not os.path.isdir(data_dir):
            messagebox.showerror("Data dir", f"Invalid data directory:\n{data_dir}", parent=self)
            return None
        return self._lab.db_path, day_row[0], data_dir

    def _outcome_start_build(self) -> None:
        if self._outcome_busy:
            return
        common = self._outcome_validate()
        if not common:
            return
        lab, day, data_dir = common
        if self._ensure_day_metadata(day) is None:
            return
        try:
            row_limit = self._outcome_parse_int(self._row_limit.get())
            build_workers = self._outcome_parse_int(
                self._build_workers.get(), allow_empty=False
            ) or 1
        except ValueError as exc:
            messagebox.showerror("Input", str(exc), parent=self)
            return
        build_workers = max(1, min(4, int(build_workers)))
        day_row = self._selected_day_inv_row()
        assert day_row is not None
        _d, dataset_n, master_n, _p = day_row
        if dataset_n <= 0:
            messagebox.showerror(
                "No dataset rows",
                f"Day {day} has 0 rows in the parent registry sources.\n"
                "Refresh Days or re-export the parent dataset from Master.",
                parent=self,
            )
            return
        if row_limit is not None and row_limit > dataset_n:
            if not messagebox.askokcancel(
                "Row count",
                f"Requested {row_limit:,} > dataset {dataset_n:,}.\n"
                f"Build will process at most {dataset_n:,} rows. Continue?",
                parent=self,
            ):
                return

        self._set_build_buttons(running=True)
        self._outcome_status.set("Building predictions…")
        self._outcome_stop_log.clear()
        self._outcome_log_line("=" * 60)
        self._outcome_log_line(f"BUILD start  day={day}")
        self._outcome_log_line(
            f"dataset_rows={dataset_n:,}  master_rows={master_n:,}  "
            f"row_limit={row_limit or 'ALL'}  workers={build_workers}"
        )
        self._outcome_log_line("Starting worker thread…")

        def work() -> None:
            try:
                self._outcome_log_line("Worker thread running — preparing build…")
                self._outcome_run_prediction_build(
                    lab=lab,
                    data_dir=data_dir,
                    day=day,
                    row_limit=row_limit,
                    worker_count=build_workers,
                    enrich_path=bool(self._enrich_path.get()),
                    overwrite=bool(self._overwrite.get()),
                )
                self.after(
                    0,
                    lambda: self._outcome_action_done("Build finished", refresh=True),
                )
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda e=exc: self._outcome_action_failed(e))

        threading.Thread(target=work, daemon=True, name="lab-outcome-build").start()

    def _outcome_resolve_stale_running_days(self, lab_db: str) -> list[str]:
        """Mark stuck 'running' days Complete (if they have rows) before Build All."""
        from chain_replay_ml.model_lab.prediction_schema import DAY_COMPLETED, DAY_RUNNING, DAY_WAITING
        from chain_replay_ml.model_lab.store import ModelLabStore

        fixed: list[str] = []
        with ModelLabStore(lab_db) as store:
            lab = store.read_info()
            if lab is None:
                return fixed
            counts = store.prediction_row_counts_by_day() or {}
            for d in store.list_build_days(lab.lab_uuid):
                day = str(d.get("trading_day") or "")
                if not day or str(d.get("status") or "") != DAY_RUNNING:
                    continue
                n = int(counts.get(day) or d.get("row_count") or 0)
                if n > 0:
                    store.set_build_day_status(
                        lab.lab_uuid,
                        day,
                        status=DAY_COMPLETED,
                        row_count=n,
                        progress_pct=100.0,
                        finished=True,
                    )
                    fixed.append(f"{day}=completed ({n:,} rows)")
                else:
                    store.set_build_day_status(
                        lab.lab_uuid,
                        day,
                        status=DAY_WAITING,
                        row_count=0,
                        progress_pct=0.0,
                    )
                    fixed.append(f"{day}=waiting (was stuck running, 0 rows)")
        return fixed

    def _outcome_incomplete_days(self, lab_db: str) -> list[str]:
        """Trading days that still need a full prediction build."""
        from chain_replay_ml.model_lab.prediction_schema import (
            DATASET_TYPE_SEEN,
            DAY_COMPLETED,
            DAY_SKIPPED,
            normalize_dataset_type,
        )
        from chain_replay_ml.model_lab.store import ModelLabStore

        with ModelLabStore(lab_db) as store:
            lab = store.read_info()
            if lab is None:
                return []
            counts = store.prediction_row_counts_by_day() or {}
            out: list[str] = []
            for d in store.list_build_days(lab.lab_uuid):
                day = str(d.get("trading_day") or "")
                if not day:
                    continue
                if normalize_dataset_type(d.get("dataset_type")) != DATASET_TYPE_SEEN:
                    continue
                st = str(d.get("status") or "")
                n = int(counts.get(day) or d.get("row_count") or 0)
                if st in (DAY_COMPLETED, DAY_SKIPPED) and n > 0:
                    continue
                out.append(day)
            return out

    def _outcome_ui_refresh_days_now(self) -> None:
        """Refresh Trading Days table on the UI thread (safe from worker via after)."""
        if self._lab is None:
            return
        try:
            from chain_replay_ml.model_lab.prediction_builder import prediction_build_summary

            summary = prediction_build_summary(
                self._lab.db_path, data_dir=self._data_dir(), light=True
            )
            self._render_build_summary(summary)
            self._render_day_table(summary.get("days") or [])
        except Exception as exc:  # noqa: BLE001
            self._outcome_log_line(f"[warn] day table refresh: {exc}")

    def _outcome_start_build_all(self) -> None:
        """
        Build every incomplete trading day, one after another.

        1) Resolve stuck 'running' days → Complete (if rows exist)
        2) Build next incomplete day
        3) After that day completes, refresh UI then start the next
        """
        if self._outcome_busy:
            return
        if self._lab is None:
            messagebox.showerror("Lab", "No Research Lab open.", parent=self)
            return
        lab = self._lab.db_path
        data_dir = self._data_dir()
        if not data_dir or not os.path.isdir(data_dir):
            messagebox.showerror("Data dir", f"Invalid data directory:\n{data_dir}", parent=self)
            return
        try:
            build_workers = self._outcome_parse_int(
                self._build_workers.get(), allow_empty=False
            ) or 1
        except ValueError as exc:
            messagebox.showerror("Input", str(exc), parent=self)
            return
        build_workers = max(1, min(4, int(build_workers)))
        enrich_path = bool(self._enrich_path.get())
        # Build All always scores full days (not capped) and does not overwrite completed days.
        overwrite = False

        # Preview queue after stale-running fix (read-only check first in worker)
        self._set_build_buttons(running=True)
        self._outcome_stop_log.clear()
        self._outcome_status.set("Build All…")
        self._outcome_log_line("=" * 60)
        self._outcome_log_line("BUILD ALL — resolve stuck running days, then build incomplete days one-by-one")

        def work() -> None:
            try:
                from chain_replay_ml.model_lab.prediction_builder import (
                    sync_prediction_build_catalog,
                )

                # Need Seen/Unseen + rows_expected before incomplete-day selection.
                sync_prediction_build_catalog(data_dir, lab)
                self.after(0, self._outcome_ui_refresh_days_now)
                fixed = self._outcome_resolve_stale_running_days(lab)
                if fixed:
                    for line in fixed:
                        self._outcome_log_line(f"[fix] {line}")
                else:
                    self._outcome_log_line("[fix] no stuck running days")

                done_evt = threading.Event()

                def refresh_ui() -> None:
                    self._outcome_ui_refresh_days_now()
                    done_evt.set()

                self.after(0, refresh_ui)
                done_evt.wait(timeout=30)

                queue = self._outcome_incomplete_days(lab)
                if not queue:
                    self.after(
                        0,
                        lambda: self._outcome_action_done(
                            "Build All — nothing to do (all days complete)",
                            refresh=True,
                        ),
                    )
                    return

                self._outcome_log_line(
                    f"Queue ({len(queue)} day(s)): {', '.join(queue)}"
                )
                for i, day in enumerate(queue, start=1):
                    if self._outcome_stop_log.is_set():
                        self._outcome_log_line("Build All stopped by user")
                        break
                    self.after(
                        0,
                        lambda d=day, i=i, n=len(queue): self._outcome_status.set(
                            f"Build All {i}/{n}: {d}"
                        ),
                    )
                    self._outcome_log_line("-" * 40)
                    self._outcome_log_line(f"BUILD ALL [{i}/{len(queue)}] day={day}")
                    # Highlight current day in the table
                    def select_day(d: str = day) -> None:
                        try:
                            if self._day_tree.exists(d):
                                self._day_tree.selection_set(d)
                                self._day_tree.see(d)
                                self._on_outcome_day_select()
                        except tk.TclError:
                            pass

                    self.after(0, select_day)
                    self._outcome_run_prediction_build(
                        lab=lab,
                        data_dir=data_dir,
                        day=day,
                        row_limit=None,
                        worker_count=build_workers,
                        enrich_path=enrich_path,
                        overwrite=overwrite,
                    )
                    self._outcome_log_line(f"Day {day} → Complete — starting next…")
                    done_evt.clear()
                    self.after(0, refresh_ui)
                    done_evt.wait(timeout=30)

                self.after(
                    0,
                    lambda: self._outcome_action_done(
                        f"Build All finished · {len(queue)} day queue processed",
                        refresh=True,
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda e=exc: self._outcome_action_failed(e))

        threading.Thread(target=work, daemon=True, name="lab-outcome-build-all").start()

    def _outcome_start_compute(self) -> None:
        if self._outcome_busy:
            return
        common = self._outcome_validate()
        if not common:
            return
        lab, day, data_dir = common
        if self._ensure_day_metadata(day) is None:
            return
        chunks = self._outcome_selected_chunks()
        if not chunks:
            messagebox.showerror(
                "Chunks", "Select at least one parallel chunk count.", parent=self
            )
            return
        try:
            row_limit = self._outcome_parse_int(self._row_limit.get())
        except ValueError as exc:
            messagebox.showerror("Row count", str(exc), parent=self)
            return
        day_row = self._selected_day_inv_row()
        assert day_row is not None
        _d, _ds, _m, pred_n = day_row
        if pred_n <= 0:
            messagebox.showerror(
                "No predictions",
                f"Day {day} has 0 prediction rows. Run Build predictions first.",
                parent=self,
            )
            return
        if row_limit is not None and row_limit > pred_n:
            if not messagebox.askokcancel(
                "Row count",
                f"Requested {row_limit:,} > prediction DB {pred_n:,}. "
                f"Compute will use all {pred_n:,} stored rows.\nContinue?",
                parent=self,
            ):
                return
            row_limit = None
        pool_size: int | None = None
        raw_ps = str(self._pool_size.get() or "").strip()
        if raw_ps:
            try:
                pool_size = int(raw_ps.replace(",", ""))
            except ValueError:
                messagebox.showerror("Pool size", "Pool size must be an integer.", parent=self)
                return
        self._set_build_buttons(running=True)
        self._outcome_status.set("Computing outcomes…")
        self._outcome_log_line("=" * 60)
        self._outcome_log_line(f"COMPUTE start  day={day}  chunks={chunks}  mode=pool")

        def work() -> None:
            try:
                tools_dir = self._outcome_tools_dir()
                if tools_dir not in sys.path:
                    sys.path.insert(0, tools_dir)
                import benchmark_outcome_parallel as bench

                def on_progress(msg: str) -> None:
                    self._outcome_log_line(f"[compute] {msg}")
                    self.after(0, lambda m=msg: self._outcome_status.set(m))

                # Each pool worker unpickles the full day — keep Research Lab pool small.
                safe_pool = max(1, min(int(pool_size or 2), 2))
                if pool_size is not None and int(pool_size) > safe_pool:
                    self._outcome_log_line(
                        f"[compute] Pool size capped {pool_size} → {safe_pool} "
                        f"(each worker copies full rows+timelines into RAM)"
                    )
                _runs, report = bench.run_benchmark_suite(
                    prediction_db=lab,
                    day=day,
                    data_dir=data_dir,
                    worker_counts=chunks,
                    row_limit=row_limit,
                    mode="pool",
                    pool_size=safe_pool,
                    on_progress=on_progress,
                )
                self._outcome_log_line(report.rstrip())
                self.after(
                    0,
                    lambda: self._outcome_action_done("Compute finished", refresh=True),
                )
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda e=exc: self._outcome_action_failed(e))

        threading.Thread(target=work, daemon=True, name="lab-outcome-compute").start()

    def _outcome_start_both(self) -> None:
        if self._outcome_busy:
            return
        common = self._outcome_validate()
        if not common:
            return
        lab, day, data_dir = common
        if self._ensure_day_metadata(day) is None:
            return
        chunks = self._outcome_selected_chunks()
        if not chunks:
            messagebox.showerror(
                "Chunks", "Select at least one parallel chunk count.", parent=self
            )
            return
        try:
            row_limit = self._outcome_parse_int(self._row_limit.get())
            build_workers = self._outcome_parse_int(
                self._build_workers.get(), allow_empty=False
            ) or 1
        except ValueError as exc:
            messagebox.showerror("Input", str(exc), parent=self)
            return
        build_workers = max(1, min(4, int(build_workers)))
        pool_size: int | None = None
        raw_ps = str(self._pool_size.get() or "").strip()
        if raw_ps:
            try:
                pool_size = int(raw_ps.replace(",", ""))
            except ValueError:
                messagebox.showerror("Pool size", "Pool size must be an integer.", parent=self)
                return

        self._set_build_buttons(running=True)
        self._outcome_status.set("Build + Compute…")
        self._outcome_stop_log.clear()
        self._outcome_log_line("=" * 60)
        self._outcome_log_line(f"BUILD+COMPUTE  day={day}  row_limit={row_limit or 'ALL'}")
        self._outcome_log_line("Starting worker thread…")

        def work() -> None:
            try:
                self._outcome_log_line("Worker thread running — preparing build…")
                self._outcome_run_prediction_build(
                    lab=lab,
                    data_dir=data_dir,
                    day=day,
                    row_limit=row_limit,
                    worker_count=build_workers,
                    enrich_path=bool(self._enrich_path.get()),
                    overwrite=bool(self._overwrite.get()),
                )
                self._outcome_log_line("Build complete — starting outcome compute…")
                enrich_path = bool(self._enrich_path.get())
                if enrich_path:
                    self._outcome_log_line(
                        "[compute] Skipped separate outcome pool — path outcomes "
                        "were already enriched during build "
                        "(avoids MemoryError from multi-process row/timeline copies)."
                    )
                    self.after(
                        0,
                        lambda: self._outcome_action_done(
                            "Build + Compute finished (outcomes during build)",
                            refresh=True,
                        ),
                    )
                    return
                tools_dir = self._outcome_tools_dir()
                if tools_dir not in sys.path:
                    sys.path.insert(0, tools_dir)
                import benchmark_outcome_parallel as bench

                def on_progress(msg: str) -> None:
                    self._outcome_log_line(f"[compute] {msg}")
                    self.after(0, lambda m=msg: self._outcome_status.set(m))

                # Cap pool for Research Lab — each worker unpickles full day rows+timelines.
                safe_pool = max(1, min(int(pool_size or 2), 2))
                _runs, report = bench.run_benchmark_suite(
                    prediction_db=lab,
                    day=day,
                    data_dir=data_dir,
                    worker_counts=chunks,
                    row_limit=row_limit,
                    mode="pool",
                    pool_size=safe_pool,
                    on_progress=on_progress,
                )
                self._outcome_log_line(report.rstrip())
                self.after(
                    0,
                    lambda: self._outcome_action_done(
                        "Build + Compute finished", refresh=True
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                self.after(0, lambda e=exc: self._outcome_action_failed(e))

        threading.Thread(target=work, daemon=True, name="lab-outcome-both").start()

    def _on_tb_enable_toggle(self) -> None:
        if getattr(self, "_tb_enable_var", None) and self._tb_enable_var.get():
            self._populate_tb_model_combo()
            if hasattr(self, "_tb_model_combo"):
                self._tb_model_combo.config(state="readonly")
        else:
            if hasattr(self, "_tb_model_combo"):
                self._tb_model_combo.config(state="disabled")
        try:
            self._ui_state.set("model_lab.tb_enable", bool(self._tb_enable_var.get()))
        except Exception:
            pass

    def _populate_tb_model_combo(self) -> None:
        try:
            from .selection_lists import get_sorted_models, refresh_combobox

            rows = get_sorted_models(self._data_dir(), lightweight=False)
            tb_models = tb_model_names_from_registry_rows(rows)
            if hasattr(self, "_tb_model_combo"):
                refresh_combobox(self._tb_model_combo, tb_models, var=self._tb_model_var)
        except Exception:
            pass

    def _outcome_run_prediction_build(
        self,
        *,
        lab: str,
        data_dir: str,
        day: str,
        row_limit: int | None,
        worker_count: int,
        enrich_path: bool,
        overwrite: bool,
    ) -> None:
        from chain_replay_ml.model_lab.prediction_manager import PredictionManager

        self._outcome_stop_log.clear()
        self._outcome_log_line(f"Preparing prediction job for {day}…")
        mgr = PredictionManager(lab)
        # Full-day builds mark Complete in Trading Days; capped runs stay partial.
        mark_complete = row_limit is None
        t_start = datetime.now().isoformat(timespec="seconds")
        tb_model = None
        if getattr(self, "_tb_enable_var", None) and self._tb_enable_var.get():
            tb_model = str(self._tb_model_var.get() or "").strip() or None

        result = mgr.create_and_start(
            data_dir,
            selected_days=[day],
            row_limit=row_limit,
            mark_day_complete=mark_complete,
            worker_count=worker_count,
            enrich_path_outcomes=enrich_path,
            overwrite=overwrite,
            resume=not overwrite,
            tb_model_name=tb_model,
            on_stage=self._outcome_log_line,
        )
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or str(result))

        job_id = str(result.get("job_id") or "")
        self._pred_job_id = job_id
        days_to_run = result.get("days_to_run") or [day]
        self._outcome_log_line(
            f"Job started: {job_id}  workers={result.get('worker_count')}  "
            f"days={', '.join(str(d) for d in days_to_run)}"
        )
        log_dir = os.path.join(os.path.dirname(lab), "logs")
        self._outcome_log_line(f"Worker logs folder: {log_dir}")
        self._outcome_stop_log.clear()
        log_offsets: dict[str, int] = {}

        while not self._outcome_stop_log.is_set():
            snap = mgr.progress(job_id, timeout_sec=1.0)
            status = str(snap.get("status") or "")
            detail = (
                f"status={status}  "
                f"days={snap.get('days_completed')}/{snap.get('days_total')}  "
                f"percent={float(snap.get('percent') or 0):.1f}%  "
                f"alive={snap.get('workers_alive')}  "
                f"day={snap.get('current_day') or '-'}"
            )
            self._outcome_log_line(f"[job] {detail}")
            if os.path.isdir(log_dir):
                for name in sorted(os.listdir(log_dir)):
                    if not name.startswith(f"prediction_worker_{job_id}_"):
                        continue
                    if not name.endswith(".log"):
                        continue
                    path = os.path.join(log_dir, name)
                    try:
                        with open(path, "r", encoding="utf-8", errors="replace") as fh:
                            fh.seek(log_offsets.get(path, 0))
                            chunk = fh.read()
                            log_offsets[path] = fh.tell()
                        if chunk.strip():
                            for line in chunk.splitlines():
                                self._outcome_log_line(f"[{name}] {line}")
                    except OSError:
                        pass
            if status in ("completed", "failed", "cancelled"):
                if status != "completed":
                    raise RuntimeError(f"Prediction job ended: {status} — {snap}")
                break
            # Workers crashed but job row still "running" — progress() now finalizes
            # as failed; if still stuck, treat as failure so UI unlocks.
            if int(snap.get("workers_alive") or 0) == 0:
                snap = mgr.progress(job_id, timeout_sec=1.0)
                status = str(snap.get("status") or "")
                if status == "completed":
                    break
                raise RuntimeError(
                    f"Workers exited (alive=0) with status={status}. "
                    "Job finalized — click Build again to retry."
                )
            time.sleep(1.5)
        t_end = datetime.now().isoformat(timespec="seconds")
        self._day_local_timing[day] = (t_start, t_end)
        self._outcome_log_line(
            f"Job {job_id} completed · day {day} · {_fmt_day_duration(t_start, t_end)}"
        )

    def _outcome_action_done(self, msg: str, *, refresh: bool = False) -> None:
        self._set_build_buttons(running=False)
        self._outcome_status.set(msg)
        self._outcome_log_line(msg)
        self._set_status_bar(status="Ready", detail=msg, active=False)
        if refresh:
            self._refresh_prediction_tab(defer_explorer=True)
            # Build/Compute finished a day — keep Metadata tab current even if
            # the user has not opened it yet.
            self._pred_meta_loaded = True
            self._refresh_prediction_metadata()

    def _outcome_action_failed(self, exc: BaseException) -> None:
        self._set_build_buttons(running=False)
        self._outcome_status.set("Failed")
        self._outcome_log_line(f"ERROR: {exc}")
        self._set_status_bar(status="Error", detail=str(exc), active=False)
        messagebox.showerror("Failed", str(exc), parent=self)

    def _set_build_buttons(self, *, running: bool) -> None:
        """Legacy name — drives Outcome Compute action buttons."""
        self._outcome_busy = bool(running)
        state = "disabled" if running else "normal"
        for btn in (
            getattr(self, "_btn_outcome_build", None),
            getattr(self, "_btn_outcome_build_all", None),
            getattr(self, "_btn_outcome_compute", None),
            getattr(self, "_btn_outcome_both", None),
        ):
            if btn is not None:
                try:
                    btn.configure(state=state)
                except tk.TclError:
                    pass

    def _show_progress_ui(self, visible: bool) -> None:
        # Build summary is always visible now
        return

    def _on_pause_build(self) -> None:
        return

    def _on_cancel_build(self) -> None:
        self._outcome_stop_log.set()

    def _on_resume_build(self) -> None:
        return

    def _get_pred_manager(self):
        from chain_replay_ml.model_lab.prediction_manager import PredictionManager

        if self._lab is None:
            raise RuntimeError("No lab open")
        if self._pred_manager is None or self._pred_manager.lab_db_path != os.path.abspath(
            self._lab.db_path
        ):
            self._pred_manager = PredictionManager(self._lab.db_path)
        return self._pred_manager

    def _on_start_build(self, *, resume: bool = True) -> None:
        _ = resume
        self._outcome_start_build()

    def _on_test_build(self) -> None:
        """Process first 2,000 rows of the first selected trading day (same worker architecture)."""
        if self._lab is None:
            return
        selected = self._selected_days_from_tree()
        if not selected:
            # Prefer currently highlighted day if any selection column was cleared
            messagebox.showwarning(
                "Test (2k)",
                "Select at least one trading day.\n\n"
                "Test uses the first selected day and inserts 2,000 rows.",
                parent=self,
            )
            return
        day = selected[0]
        ok = messagebox.askyesno(
            "Test (2k)",
            f"Run a test sample on {day}?\n\n"
            "• First 2,000 valid rows only\n"
            "• Same external-worker architecture as Start\n"
            "• Inserts into Prediction Dataset\n"
            "• Does not mark the day Completed (Start still runs full day)\n"
            "• Worker logs → lab logs/ folder\n"
            "• Per-prediction outcome timings (CSV + log sample)\n\n"
            "Continue?",
            parent=self,
        )
        if not ok:
            return
        self._on_generate_prediction(
            resume=True,
            overwrite=False,
            selected_days=[day],
            row_limit=2000,
            job_title=f"Test 2k · {day}",
        )

    def _on_generate_prediction(
        self,
        *,
        resume: bool = True,
        overwrite: bool | None = None,
        selected_days: list[str] | None = None,
        row_limit: int | None = None,
        job_title: str | None = None,
    ) -> None:
        if self._lab is None:
            return
        if self._build_running():
            messagebox.showinfo("Prediction Dataset", "Build already in progress.", parent=self)
            return

        if overwrite is None:
            overwrite = bool(self._overwrite_var.get())
        selected = selected_days if selected_days is not None else self._selected_days_from_tree()
        if not selected and not overwrite:
            messagebox.showwarning(
                "Prediction Dataset",
                "Select at least one trading day (or use Overwrite All).",
                parent=self,
            )
            return

        if overwrite and row_limit is None:
            ok = messagebox.askyesno(
                "Prediction Dataset",
                "Overwrite All will delete existing prediction rows and rebuild.\n\nContinue?",
                parent=self,
            )
            if not ok:
                return

        self._persist_day_selection()
        self._progress = {}
        self._progress_bar.configure(value=0.0)
        self._progress_pct_var.set("0%")
        title = job_title or "Prediction Dataset"
        self._live_progress_var.set(f"Starting {title}…")
        self._set_build_buttons(running=True)
        self._set_status_bar(status="Building…", percent=0.0, detail=f"Starting {title}", active=True)

        data_dir = self._data_dir()
        try:
            workers = int(self._workers_var.get() or "3")
        except (TypeError, ValueError):
            workers = 3
        workers = max(1, min(4, workers))

        from .build_progress_manager import get_build_progress_manager

        mgr_ui = get_build_progress_manager()
        mgr_ui.begin_job(
            "model_lab_prediction",
            title=title,
            cancel_fn=lambda: self._on_cancel_build(),
        )
        mgr_ui.publish({
            "status": "running",
            "job_kind": "model_lab_prediction",
            "job_title": title,
            "message": "Launching worker processes…",
            "percent": 0.0,
            "rows": 0,
            "total": 0,
            "worker_count": workers,
        })

        # Heavy prepare (catalog/model load) must NOT run on the Tk UI thread.
        overwrite_f = bool(overwrite)
        resume_f = bool(resume and not overwrite)
        selected_f = list(selected) if selected else None
        row_limit_f = row_limit
        mark_complete_f = row_limit is None
        workers_f = workers
        data_dir_f = data_dir
        tb_model_f = None
        if getattr(self, "_tb_enable_var", None) and self._tb_enable_var.get():
            tb_model_f = str(self._tb_model_var.get() or "").strip() or None
        try:
            pred_mgr = self._get_pred_manager()
        except Exception as exc:
            self._set_build_buttons(running=False)
            messagebox.showerror("Prediction Dataset", str(exc), parent=self)
            return

        def _start_job() -> None:
            try:
                result = pred_mgr.create_and_start(
                    data_dir_f,
                    overwrite=overwrite_f,
                    resume=resume_f,
                    selected_days=selected_f,
                    row_limit=row_limit_f,
                    mark_day_complete=mark_complete_f,
                    worker_count=workers_f,
                    tb_model_name=tb_model_f,
                )
            except Exception as exc:
                err = str(exc)
                self.after(0, lambda e=err: self._on_pred_job_start_failed(e))
                return
            self.after(0, lambda r=result: self._on_pred_job_started(r, workers_f))

        threading.Thread(target=_start_job, name="pred-job-start", daemon=True).start()

    def _on_pred_job_start_failed(self, err: str) -> None:
        self._set_build_buttons(running=False)
        messagebox.showerror("Prediction Dataset", err, parent=self)

    def _on_pred_job_started(self, result: dict[str, Any], workers: int) -> None:
        if not result.get("ok"):
            self._set_build_buttons(running=False)
            messagebox.showerror(
                "Prediction Dataset",
                str(result.get("error") or "Failed to start job"),
                parent=self,
            )
            return

        skipped_days = result.get("skipped_days") or []
        if skipped_days:
            names = ", ".join(str(d) for d, _why in skipped_days)
            messagebox.showwarning(
                "Prediction Dataset — days skipped",
                f"{len(skipped_days)} day(s) had no usable rows for the "
                f"target and were skipped (marked Skipped in Trading Days), "
                f"not queued to the worker:\n\n{names}\n\n"
                f"{skipped_days[0][1]}",
                parent=self,
            )

        self._pred_job_id = str(result["job_id"])
        self._pred_poll_tick = 0
        spawned = result.get("spawned") or []
        log_paths = [
            str(s.get("log_path") or "")
            for s in spawned
            if isinstance(s, dict) and s.get("ok") and s.get("log_path")
        ]
        logs_dir = ""
        if log_paths:
            logs_dir = os.path.dirname(log_paths[0])
        elif self._lab is not None:
            logs_dir = os.path.join(os.path.dirname(os.path.abspath(self._lab.db_path)), "logs")
        self._progress = {
            "job_id": self._pred_job_id,
            "worker_count": result.get("worker_count") or workers,
            "phase": "running",
            "message": f"Workers launched ({result.get('worker_count')})",
            "logs_dir": logs_dir,
        }
        if logs_dir:
            self._live_progress_var.set(
                f"Job: {self._pred_job_id}  ·  workers launched\n"
                f"Logs folder:\n{logs_dir}"
            )
        self.after(400, self._poll_build_progress)

    def _poll_build_progress(self) -> None:
        if self._lab is None or not self._pred_job_id:
            # Fall through for non-pred threads (e.g. path backfill) using old dict
            p = self._progress or {}
            if self._build_thread is not None and self._build_thread.is_alive():
                self.after(400, self._poll_build_progress)
            return

        import sqlite3

        from .build_progress_manager import get_build_progress_manager

        mgr = self._get_pred_manager()
        try:
            # Short timeout — never block Tk for the old 60s busy_timeout.
            snap = mgr.progress(self._pred_job_id, timeout_sec=1.0)
        except sqlite3.OperationalError:
            self.after(600, self._poll_build_progress)
            return
        except Exception:
            self.after(600, self._poll_build_progress)
            return

        job = snap.get("job") or {}
        status = str(snap.get("status") or job.get("status") or "")
        workers = snap.get("workers") or []
        days_done = int(snap.get("days_completed") or 0)
        days_total = int(snap.get("days_total") or 0)
        pct = float(snap.get("percent") or 0.0)
        day = str(snap.get("current_day") or "—")
        alive = int(snap.get("workers_alive") or 0)

        # Pick busiest worker for day row progress
        day_pct = None
        rows_loaded = None
        rows_day_total = None
        for w in workers:
            if w.get("status") == "running":
                day = str(w.get("assigned_day") or day)
                rows_loaded = w.get("current_row")
                rows_day_total = w.get("total_rows")
                day_pct = w.get("percent")
                break

        day_row_txt = ""
        if rows_loaded is not None and rows_day_total:
            day_row_txt = f"Rows {_fmt_int(rows_loaded)} / {_fmt_int(rows_day_total)}"
        day_pct_txt = f"{float(day_pct):.0f}%" if day_pct is not None else "—"

        msgs = [str(w.get("last_message") or "") for w in workers if w.get("last_message")]
        detail = msgs[0] if msgs else status
        logs_dir = str((self._progress or {}).get("logs_dir") or "")
        if not logs_dir and workers:
            for w in workers:
                lp = str(w.get("log_path") or "")
                if lp:
                    logs_dir = os.path.dirname(lp)
                    break
        live_lines = [
            f"Job: {self._pred_job_id}  ·  status={status}  ·  workers alive={alive}",
            f"Current Day: {day}",
            day_row_txt,
            f"Day Progress: {day_pct_txt}",
            f"Overall: {_fmt_int(days_done)} / {_fmt_int(days_total)} Days Complete",
            detail,
        ]
        if logs_dir:
            live_lines.append(f"Logs: {logs_dir}")
        self._live_progress_var.set("\n".join(x for x in live_lines if x))

        self._progress_bar.configure(value=max(0.0, min(100.0, pct)))
        self._progress_pct_var.set(f"{pct:.1f}%")
        self._set_status_bar(
            status="Paused" if status == "paused" else "Building…",
            percent=pct,
            detail=f"Day {day} · {_fmt_int(days_done)}/{_fmt_int(days_total)} days · {alive} workers",
            active=status in ("running", "pending"),
        )
        get_build_progress_manager().publish({
            "status": "running" if status == "running" else status,
            "job_kind": "model_lab_prediction",
            "job_title": "Prediction Dataset",
            "message": detail or status,
            "trading_day": day,
            "rows": days_done,
            "total": days_total,
            "current": days_done,
            "percent": pct,
            "worker_count": len(workers),
        })

        # Heavy build_summary rebuilds the full day table — throttle (not every 400ms).
        self._pred_poll_tick = int(getattr(self, "_pred_poll_tick", 0) or 0) + 1
        if self._pred_poll_tick == 1 or self._pred_poll_tick % 5 == 0:
            try:
                from chain_replay_ml.model_lab.prediction_builder import (
                    prediction_build_summary,
                )

                summary = prediction_build_summary(
                    self._lab.db_path, data_dir=self._data_dir(), light=True
                )
                summary = dict(summary)
                summary["completed"] = days_done
                self._render_build_summary(summary)
                self._render_day_table(
                    summary.get("days") or [],
                    live={
                        "current_day": day,
                        "day_progress_pct": day_pct,
                        "rows_loaded": rows_loaded,
                        "rows_day_total": rows_day_total,
                    },
                )
            except sqlite3.OperationalError:
                pass
            except Exception:
                pass

        # Keep polling while job active
        if status in ("running", "pending") or (status == "paused" and alive > 0):
            self.after(400, self._poll_build_progress)
            return

        if status == "paused" and alive == 0:
            self._set_build_buttons(running=False)
            self._live_progress_var.set("Paused — click Resume to continue")
            get_build_progress_manager().publish({
                "_done": True,
                "status": "cancelled",
                "job_kind": "model_lab_prediction",
                "job_title": "Prediction Dataset",
                "message": f"Paused · {_fmt_int(days_done)}/{_fmt_int(days_total)} days",
                "percent": pct,
            })
            self.refresh()
            return

        # Terminal
        self._set_build_buttons(running=False)
        ok = status == "completed"
        msg = {
            "completed": f"Ready · {_fmt_int(days_done)} days done",
            "failed": f"Failed · {job.get('error_message') or 'see worker logs'}",
            "cancelled": f"Cancelled · {_fmt_int(days_done)} days kept",
        }.get(status, f"Finished ({status})")
        get_build_progress_manager().publish({
            "_done": True,
            "status": "completed" if ok else "failed",
            "job_kind": "model_lab_prediction",
            "job_title": "Prediction Dataset",
            "message": msg,
            "percent": 100.0 if ok else pct,
            "rows": days_done,
            "total": days_total,
        })
        self._set_status_bar(
            status="Complete" if ok else status.title(),
            percent=100.0 if ok else pct,
            detail=msg,
            active=False,
        )
        self.refresh()
        if ok:
            messagebox.showinfo("Prediction Dataset", msg, parent=self)
        elif status in ("failed", "cancelled"):
            messagebox.showwarning("Prediction Dataset", msg, parent=self)
        self._pred_job_id = None if ok else self._pred_job_id

    def _build_running(self) -> bool:
        if self._outcome_busy:
            return True
        if self._lab is not None and self._pred_job_id:
            try:
                if self._get_pred_manager().is_busy():
                    return True
            except Exception:
                pass
        t = self._build_thread
        return bool(t is not None and t.is_alive())


    def _render_overview(self, lab: Any) -> None:
        ov = lab.to_overview_dict()
        ranking = ov.get("feature_ranking") or {}
        feats = ov.get("selected_features") or []
        artifacts = ov.get("artifact_pointers") or {}
        checksum = ov.get("model_checksum") or "—"
        if checksum != "—" and len(checksum) > 16:
            checksum_disp = f"{checksum[:12]}…{checksum[-8:]}"
        else:
            checksum_disp = checksum

        pred_status = str(ov.get("prediction_dataset_status") or "not_generated")
        pred_rows = ov.get("prediction_row_count")

        lines: list[str] = [
            f"Lab Name:              {ov.get('lab_name') or '—'}",
            f"Lab UUID:              {ov.get('lab_uuid') or '—'}",
            f"Lab ID:                {ov.get('lab_id') or '—'}",
            f"Version:               v{ov.get('version')}",
            f"Status:                {ov.get('status') or '—'}",
            f"Phase:                 {ov.get('phase')}",
            f"Schema Version:        {ov.get('lab_schema_version')}",
            f"Purpose:               {ov.get('purpose') or '—'}",
            f"Description:           {ov.get('description') or '—'}",
            f"Created:               {ov.get('created_at') or '—'}",
            f"Created By:            {ov.get('created_by') or '—'}",
            "",
            f"Parent Model:          {ov.get('parent_model_name') or '—'}",
            f"Parent Model ID:       {ov.get('parent_model_id') or '—'}",
            f"Model Checksum:        {checksum_disp}",
            f"Dataset:               {ov.get('dataset') or '—'}",
            f"Target:                {ov.get('target') or '—'}",
            f"Algorithm:             {ov.get('algorithm') or '—'}",
            f"Training Rows:         {fmt_val(ov.get('training_rows'))}",
            f"Original Features:     {fmt_val(ov.get('original_feature_count'))}",
            f"Selected Features:     {fmt_val(ov.get('selected_feature_count'))}",
            "",
            "— Prediction Dataset —",
            f"  Status:              {pred_status}",
            f"  Rows:                {fmt_val(pred_rows)}",
            "",
            "— Training Metrics —",
            _short_metric_blob(ov.get("training_metrics")),
            "",
            "— Holdout Metrics —",
            _short_metric_blob(ov.get("holdout_metrics")),
            "",
            "— Walk Forward Metrics —",
            _short_metric_blob(ov.get("walk_forward_metrics")),
            "",
            "— Selected Features —",
        ]
        if feats:
            for i, name in enumerate(feats, start=1):
                lines.append(f"  {i:3d}. {name}")
        else:
            lines.append("  (none)")

        lines.append("")
        lines.append("— Feature Ranking —")
        if ranking.get("available"):
            lines.append(f"  Source: {ranking.get('source') or '—'}")
            for row in (ranking.get("rows") or [])[:40]:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    f"  rank={fmt_val(row.get('final_rank'))}  "
                    f"{row.get('feature')}  "
                    f"folds={fmt_val(row.get('selected_in_folds'))}  "
                    f"gain={fmt_val(row.get('gain_importance_pct'))}"
                )
            extra = len(ranking.get("rows") or []) - 40
            if extra > 0:
                lines.append(f"  … +{extra} more")
            if ranking.get("note"):
                lines.append(f"  Note: {ranking['note']}")
        else:
            lines.append(f"  {ranking.get('message') or 'Feature ranking unavailable.'}")

        lines.append("")
        lines.append("— Artifact Pointers —")
        for key, item in artifacts.items():
            if key in ("model.pkl", "feature_importance.json", "walk_forward_results.json",
                       "holdout_metrics.json", "selected_features.csv"):
                continue
            if not isinstance(item, dict):
                continue
            status = item.get("status") or ("available" if item.get("available") else "unavailable")
            lines.append(f"  [{status}] {key}: {item.get('path') or '—'}")

        lines.append("")
        lines.append(f"Lab DB: {ov.get('db_path') or '—'}")

        self._overview_body.configure(state="normal")
        self._overview_body.delete("1.0", "end")
        self._overview_body.insert("1.0", "\n".join(lines))
        self._overview_body.configure(state="disabled")

    def _resolve_detail_doc(self) -> dict[str, Any]:
        if self._detail_doc and str(self._detail_doc.get("model_name") or "") == self.model_name:
            return self._detail_doc
        from chain_replay_ml.training.registry import load_model_detail

        doc = load_model_detail(self._data_dir(), self.model_name)
        self._detail_doc = doc
        return doc

    def _create_lab(self) -> None:
        from chain_replay_ml.model_lab import create_model_lab, default_lab_display_name, next_lab_version

        try:
            version = next_lab_version(self.model_name)
            defaults = prompt_create_lab(
                self,
                model_name=self.model_name,
                default_lab_name=default_lab_display_name(self.model_name, version),
            )
            if not defaults:
                return
            doc = self._resolve_detail_doc()
            info = create_model_lab(
                self._data_dir(),
                doc,
                lab_name=defaults["lab_name"],
                description=defaults.get("description") or None,
                purpose=defaults.get("purpose") or None,
            )
        except Exception as exc:
            messagebox.showerror("Start Research", str(exc), parent=self)
            return
        messagebox.showinfo(
            "Research Lab",
            f"Created {info.lab_name}\nUUID: {info.lab_uuid}\n\n{info.db_path}",
            parent=self,
        )
        if callable(self._on_created):
            try:
                self._on_created()
            except Exception:
                pass
        self.refresh()

    def _clear_explorer(self) -> None:
        for item in self._explorer.get_children():
            self._explorer.delete(item)
        self._explorer_page_lbl.set("")

    def _reload_explorer(self) -> None:
        """Public entry — always load off the UI thread."""
        self._reload_explorer_async()

    def _reload_explorer_async(self) -> None:
        if self._lab is None:
            return
        lab_db = self._lab.db_path
        data_dir = self._data_dir()
        search = self._search_var.get()
        where_sql = self._explorer_where_sql
        where_args = list(self._explorer_where_args)
        order_by = self._explorer_order
        limit = self._explorer_page
        offset = self._explorer_offset
        query_cols = [c for c in _EXPLORER_COLS if c not in _EXPLORER_VIRTUAL]
        cached_total = self._explorer_total_cache
        skip_count = (
            cached_total is not None
            and not str(search or "").strip()
            and not str(where_sql or "").strip()
        )

        self._explorer_load_gen += 1
        gen = self._explorer_load_gen
        self._clear_explorer()
        self._explorer_page_lbl.set("Loading…")

        def work() -> None:
            err: Exception | None = None
            cols: list[str] = []
            rows: list[tuple[Any, ...]] = []
            total = 0
            try:
                from chain_replay_ml.model_lab.store import ModelLabStore

                with ModelLabStore(lab_db) as store:
                    cols, rows = store.query_predictions(
                        columns=query_cols,
                        search=search,
                        where_sql=where_sql,
                        where_args=where_args,
                        order_by=order_by,
                        limit=limit,
                        offset=offset,
                        data_dir=data_dir,
                    )
                    if skip_count:
                        total = int(cached_total or 0)
                    else:
                        total = store.count_predictions(
                            search=search,
                            where_sql=where_sql,
                            where_args=where_args,
                            data_dir=data_dir,
                        )
            except Exception as exc:  # noqa: BLE001
                err = exc

            def done() -> None:
                if gen != self._explorer_load_gen:
                    return
                if err is not None:
                    self._explorer_page_lbl.set(str(err))
                    return
                if not skip_count and not str(search or "").strip() and not str(where_sql or "").strip():
                    self._explorer_total_cache = int(total)
                self._paint_explorer_rows(cols, rows, total)

            try:
                self.after(0, done)
            except tk.TclError:
                pass

        threading.Thread(target=work, daemon=True, name="lab-explorer").start()

    def _paint_explorer_rows(
        self,
        cols: list[str],
        rows: list[tuple[Any, ...]],
        total: int,
    ) -> None:
        self._clear_explorer()
        col_idx = {c: i for i, c in enumerate(cols)}

        def _cell(row: tuple[Any, ...], name: str) -> Any:
            i = col_idx.get(name)
            if i is None or i >= len(row):
                return None
            return row[i]

        for row in rows:
            current = _cell(row, "current_ltp")
            predicted = _cell(row, "predicted_future_ltp")
            actual = _cell(row, "actual_future_ltp")
            values: list[str] = []
            for col in _EXPLORER_COLS:
                val = _cell(row, col)
                if col in ("predicted_trend", "actual_trend"):
                    if not val:
                        move = None
                        try:
                            if col == "predicted_trend" and current is not None and predicted is not None:
                                move = float(predicted) - float(current)
                            elif col == "actual_trend" and current is not None and actual is not None:
                                move = float(actual) - float(current)
                        except (TypeError, ValueError):
                            move = None
                        if move is None:
                            values.append("—")
                        elif move > 0:
                            values.append("▲ UP")
                        elif move < 0:
                            values.append("▼ DOWN")
                        else:
                            values.append("— FLAT")
                    else:
                        s = str(val).upper()
                        if s == "UP":
                            values.append("▲ UP")
                        elif s == "DOWN":
                            values.append("▼ DOWN")
                        elif s == "FLAT":
                            values.append("— FLAT")
                        else:
                            values.append(str(val))
                    continue
                if col in ("direction_correct", "target_reached"):
                    values.append(_fmt_check(val))
                elif col in (
                    "time_to_target",
                    "time_to_max_profit",
                    "time_to_max_drawdown",
                    "time_to_dd_before_target",
                ):
                    values.append(_fmt_time_metric(val))
                elif col in (
                    "timestamp",
                    "target_reached_at",
                    "max_profit_at",
                    "max_drawdown_at",
                    "exit_at",
                ):
                    values.append(_fmt_unix_time(val))
                elif isinstance(val, float):
                    values.append(f"{val:.4g}")
                else:
                    values.append("" if val is None else str(val))
            self._explorer.insert("", "end", values=values)

        end = min(self._explorer_offset + len(rows), total)
        start = self._explorer_offset + 1 if rows else 0
        self._explorer_page_lbl.set(f"{start}–{end} / {_fmt_int(total)}")
        if self._explorer_where_sql:
            self._explorer_filter_count.set(
                f"{_fmt_int(total)} matching prediction"
                f"{'' if total == 1 else 's'}"
            )
            self._show_applied_filter_banner()
        else:
            self._hide_applied_filter_banner()

    def _explorer_prev(self) -> None:
        self._explorer_offset = max(0, self._explorer_offset - self._explorer_page)
        self._reload_explorer()

    def _explorer_next(self) -> None:
        self._explorer_offset += self._explorer_page
        self._reload_explorer()

    def _sort_explorer(self, col: str) -> None:
        if col in _EXPLORER_VIRTUAL:
            return
        cur = self._explorer_order
        if cur.startswith(col) and cur.endswith("ASC"):
            self._explorer_order = f"{col} DESC"
        else:
            self._explorer_order = f"{col} ASC"
        self._explorer_offset = 0
        self._reload_explorer()

    def _on_download_explorer_csv(self) -> None:
        """Export first 1,000 prediction rows (explorer order) to CSV."""
        if self._lab is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Download CSV (first 1,000 rows)",
            defaultextension=".csv",
            initialfile="prediction_sample_1000.csv",
            filetypes=[("CSV", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return

        query_cols = [c for c in _EXPLORER_COLS if c not in _EXPLORER_VIRTUAL]
        try:
            from chain_replay_ml.model_lab.store import ModelLabStore

            with ModelLabStore(self._lab.db_path) as store:
                cols, rows = store.query_predictions(
                    columns=query_cols,
                    search=self._search_var.get(),
                    order_by=self._explorer_order,
                    limit=1000,
                    offset=0,
                )
        except Exception as exc:
            messagebox.showerror("Download CSV", str(exc), parent=self)
            return

        if not cols:
            messagebox.showwarning("Download CSV", "No columns / no data to export.", parent=self)
            return

        import csv

        headers = [_EXPLORER_HEADINGS.get(c, c) for c in cols]
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.writer(fh)
                writer.writerow(headers)
                for row in rows:
                    out: list[Any] = []
                    for i, col in enumerate(cols):
                        val = row[i] if i < len(row) else None
                        if col in (
                            "timestamp",
                            "target_reached_at",
                            "max_profit_at",
                            "max_drawdown_at",
                            "exit_at",
                        ):
                            out.append(_fmt_unix_time(val) if val is not None else "")
                        elif val is None:
                            out.append("")
                        else:
                            out.append(val)
                    writer.writerow(out)
        except OSError as exc:
            messagebox.showerror("Download CSV", str(exc), parent=self)
            return

        messagebox.showinfo(
            "Download CSV",
            f"Saved {len(rows):,} rows →\n{path}",
            parent=self,
        )

    def _on_export(self) -> None:
        if self._lab is None:
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Export Prediction Dataset",
            defaultextension=".parquet",
            filetypes=[
                ("Parquet", "*.parquet"),
                ("CSV", "*.csv"),
                ("SQLite", "*.sqlite"),
                ("All", "*.*"),
            ],
        )
        if not path:
            return
        from chain_replay_ml.model_lab.prediction_export import export_prediction_dataset

        result = export_prediction_dataset(self._lab.db_path, path)
        if result.get("ok"):
            messagebox.showinfo("Export", f"Saved:\n{result.get('path')}", parent=self)
        else:
            messagebox.showerror("Export", str(result.get("error") or "Export failed"), parent=self)


_open_windows: dict[str, ModelLabWindow] = {}


def ensure_default_research_lab(
    chart_dir: str,
    model_name: str,
    *,
    detail_doc: dict[str, Any] | None = None,
) -> bool:
    """
    Create a Research Lab with the default name if none exists.

    No name dialog / confirmation. Returns True if a lab was created.
    Always loads full model detail (list-row docs are not enough for snapshots).
    """
    from chain_replay_ml.model_lab import (
        create_model_lab,
        default_lab_display_name,
        find_latest_lab,
        next_lab_version,
    )
    from chain_replay_ml.training.registry import load_model_detail

    name = str(model_name or "").strip()
    if not name:
        raise ValueError("model_name is required")
    if find_latest_lab(name) is not None:
        return False

    data_dir = chart_data_dir(chart_dir)
    # Prefer full package detail — registry list rows lack artifact snapshots.
    doc = load_model_detail(data_dir, name)
    if not doc or not str(doc.get("model_name") or "").strip():
        if detail_doc and str(detail_doc.get("model_name") or "").strip() == name:
            doc = detail_doc
        else:
            raise ValueError(f"Could not load model detail for {name}")
    version = next_lab_version(name)
    create_model_lab(
        data_dir,
        doc,
        lab_name=default_lab_display_name(name, version),
        purpose="General Research",
    )
    return True


def open_model_lab_window(
    master: tk.Misc,
    *,
    chart_dir: str,
    model_name: str,
    detail_doc: dict[str, Any] | None = None,
    ensure_lab: bool = False,
    initial_tab: str | None = None,
) -> ModelLabWindow:
    """
    Open Research Lab for a model.

    ensure_lab: if True and no lab exists, auto-create with default name
    (skips create dialog + confirmation popup). Runs off the UI thread.
    initial_tab: e.g. ``\"prediction\"`` to land on Prediction Dataset.
    """
    existing = _open_windows.get(model_name)
    if existing is not None:
        try:
            if existing.winfo_exists():
                existing.update_idletasks()
                place_toplevel_beside_main(existing, master)
                existing.lift()
                existing.focus_force()
                if ensure_lab:
                    existing._begin_async_open(ensure_lab=True)  # noqa: SLF001
                else:
                    existing.refresh_async()
                if (initial_tab or "").strip().lower() in (
                    "prediction",
                    "prediction_dataset",
                    "pred",
                ):
                    existing.select_prediction_tab()
                return existing
        except tk.TclError:
            _open_windows.pop(model_name, None)

    win = ModelLabWindow(
        master,
        chart_dir=chart_dir,
        model_name=model_name,
        detail_doc=detail_doc,
        initial_tab=initial_tab,
        defer_refresh=True,
    )

    def _on_close() -> None:
        _open_windows.pop(model_name, None)
        win.destroy()

    win.protocol("WM_DELETE_WINDOW", _on_close)
    _open_windows[model_name] = win
    win._begin_async_open(ensure_lab=ensure_lab)  # noqa: SLF001
    return win
