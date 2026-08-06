"""Build progress UI — mirrors Create Dataset pipeline table + validation checks."""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import scrolledtext, ttk
from typing import Any

from . import feature_policy_format as pol_fmt
from .build_profiler_format import format_build_profiler_report
from .gil_monitor_format import format_gil_report

BUILD_INTERNAL_STAGE_NAMES = [
    "Load Database",
    "Validate Sources",
    "Sampling Grid",
    "Strike Selection",
    "Prediction Targets",
    "Feature Generation",
    "Dataset Validation",
    "Write Parquet",
]

VALIDATION_SUMMARY_IDS = frozenset({
    "required_columns",
    "duplicates",
    "target_columns",
    "missing_values",
    "feature_count",
})


def _fmt_duration(sec: float | None) -> str:
    if sec is None:
        return "—"
    s = float(sec)
    if s < 60:
        return f"{s:.2f} s"
    if s < 3600:
        return f"{s / 60:.1f} min"
    return f"{s / 3600:.1f} h"


def _fmt_compact(n: int | float | None) -> str:
    if n is None:
        return "—"
    v = float(n)
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v / 1_000:.1f}K"
    return str(int(v))


def _status_text(status: str | None) -> str:
    st = str(status or "waiting").lower()
    if st == "done":
        return "✓ Done"
    if st == "running":
        return "▶ Running"
    if st == "failed":
        return "✗ Failed"
    if st == "skipped":
        return "Skipped"
    return "Waiting"


def _sub_status_text(status: str | None) -> str:
    st = str(status or "waiting").lower()
    if st == "done":
        return "✓"
    if st == "running":
        return "▶"
    if st == "failed":
        return "✗"
    if st == "skipped":
        return "Skipped"
    return "—"


def _stage_id_num(sub_id: Any) -> int | None:
    """Return 1–8 for pipeline stage substages; None for feature-group ids (strings)."""
    try:
        n = int(sub_id)
    except (TypeError, ValueError):
        return None
    return n if 1 <= n <= 8 else None


def _is_feature_group_substage(sub: dict[str, Any]) -> bool:
    if sub.get("parent_stage") == 6:
        return True
    return _stage_id_num(sub.get("id")) is None and sub.get("id") not in (None, "", "build")


def _visible_pipeline_subs(
    pipeline_subs: list[dict[str, Any]],
    payload: dict[str, Any],
) -> list[dict[str, Any]]:
    """Limit substages shown for debug modes — avoid a wall of 'Skipped' rows."""
    if payload.get("debug_load"):
        return [s for s in pipeline_subs if _stage_id_num(s.get("id")) == 1]
    if payload.get("debug_features"):
        return [s for s in pipeline_subs if _stage_id_num(s.get("id")) in (1, 6)]
    return pipeline_subs


def effective_substage_status(sub: dict[str, Any], payload: dict[str, Any]) -> str:
    st = str(sub.get("status") or "waiting").lower()
    if st in ("running", "done", "skipped", "failed"):
        return st
    sub_id = _stage_id_num(sub.get("id"))
    if sub_id is None:
        return st
    active = int(payload.get("stage") or 0)
    if sub_id == active:
        return "running"
    if active and sub_id < active:
        return "done"
    return st


def live_pipeline_payload(payload: dict[str, Any], received_at: float) -> dict[str, Any]:
    """Bump elapsed times between server progress events (mirrors web livePipelinePayload)."""
    if not payload or payload.get("status") != "running":
        return payload
    delta = time.time() - received_at
    out = dict(payload)
    pl = dict(out.get("pipeline") or {})
    active_build_stage = int(out.get("stage") or 0)

    def bump(obj: dict[str, Any] | None) -> None:
        if obj and obj.get("status") == "running":
            base = float(obj["elapsed_sec"]) if obj.get("elapsed_sec") is not None else 0.0
            next_sec = round(base + delta, 2)
            obj["elapsed_sec"] = next_sec
            obj["elapsed_label"] = _fmt_duration(next_sec)

    stages = [dict(st) for st in (pl.get("stages") or [])]
    substages = [dict(sub) for sub in (pl.get("substages") or [])]
    for st in stages:
        bump(st)
    for sub in substages:
        if _is_feature_group_substage(sub):
            bump(sub)
            continue
        if sub.get("parent_stage") not in (None, "build"):
            continue
        sub_id = _stage_id_num(sub.get("id"))
        if sub_id is not None and sub_id == active_build_stage and (sub.get("status") or "waiting") == "waiting":
            sub["status"] = "running"
        bump(sub)
    pl = dict(pl)
    pl["stages"] = stages
    pl["substages"] = substages
    if pl.get("total_elapsed_sec") is not None:
        next_total = round(float(pl["total_elapsed_sec"]) + delta, 2)
        pl["total_elapsed_sec"] = next_total
        pl["total_elapsed_label"] = _fmt_duration(next_total)
    out["pipeline"] = pl
    return out


def _substage_progress_text(sub: dict[str, Any], payload: dict[str, Any], sub_status: str) -> str:
    if sub_status not in ("running", "done"):
        return "—"
    sub_id = _stage_id_num(sub.get("id"))
    if sub_id == 6:
        g_done = payload.get("feature_groups_done")
        g_tot = payload.get("feature_groups_total")
        if g_tot:
            return f"{int(g_done or 0)} / {int(g_tot)} groups"
    if sub.get("progress_total"):
        cur = _fmt_compact(sub.get("progress_current"))
        tot = _fmt_compact(sub.get("progress_total"))
        unit = sub.get("progress_unit") or "rows"
        return f"{cur} / {tot} {unit}"
    stage = int(payload.get("stage") or 0)
    if sub_id is None:
        active_id = str(payload.get("feature_group_id") or payload.get("substage") or "")
        fg_id = str(sub.get("id") or "")
        fg_label = str(sub.get("label") or sub.get("name") or "")
        is_active = (
            sub_status == "running"
            and active_id
            and (fg_id == active_id or fg_label == active_id)
        )
        if is_active:
            cur = payload.get("sub_current")
            tot = payload.get("sub_total")
            if cur is not None and tot:
                return f"{_fmt_compact(cur)} / {_fmt_compact(tot)} rows"
        return "—"
    if sub_status == "running" and sub_id == stage:
        g_tot = payload.get("feature_groups_total")
        if sub_id == 6 and g_tot:
            return f"{int(payload.get('feature_groups_done') or 0)} / {int(g_tot)} groups"
        tot = payload.get("total")
        if tot:
            cur = _fmt_compact(payload.get("current"))
            tot_fmt = _fmt_compact(tot)
            unit = "days" if stage == 1 else "groups" if stage == 6 else "rows"
            return f"{cur} / {tot_fmt} {unit}"
    return "—"


def seed_pipeline_waiting() -> dict[str, Any]:
    return {
        "stages": [{
            "id": "build",
            "name": "Create Dataset",
            "status": "waiting",
            "elapsed_sec": None,
            "elapsed_label": None,
        }],
        "substages": [
            {
                "id": i + 1,
                "label": name,
                "name": name,
                "status": "waiting",
                "parent_stage": "build",
            }
            for i, name in enumerate(BUILD_INTERNAL_STAGE_NAMES)
        ],
        "total_elapsed_sec": 0,
        "total_elapsed_label": "0.00 s",
    }


class MasterBuildProgressPanel(ttk.Frame):
    """Pipeline progress panel aligned with ml_create_dataset.html build progress."""

    def __init__(self, master: tk.Misc, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self._payload: dict[str, Any] | None = None
        self._last_payload: dict[str, Any] | None = None
        self._received_at: float = 0.0
        self._build_ui()

    def _build_ui(self) -> None:
        self._notebook = ttk.Notebook(self)
        self._notebook.pack(fill="both", expand=True, padx=4, pady=4)

        progress_tab = ttk.Frame(self._notebook)
        self._notebook.add(progress_tab, text="Progress")

        main_row = ttk.Frame(progress_tab)
        main_row.pack(fill="both", expand=True, padx=0, pady=0)

        progress_col = ttk.LabelFrame(main_row, text="Build Progress", padding=4)
        progress_col.pack(side="left", fill="both", expand=True, padx=(0, 4))

        cols = ("stage", "status", "time", "progress")
        self.tree = ttk.Treeview(progress_col, columns=cols, show="headings", height=10)
        self.tree.heading("stage", text="Stage")
        self.tree.heading("status", text="Status")
        self.tree.heading("time", text="Time")
        self.tree.heading("progress", text="Progress")
        self.tree.column("stage", width=200, stretch=True)
        self.tree.column("status", width=80)
        self.tree.column("time", width=72)
        self.tree.column("progress", width=100)
        self.tree.pack(fill="both", expand=True)

        stats = ttk.Frame(progress_col)
        stats.pack(fill="x", pady=(4, 0))
        self.speed_var = tk.StringVar(value="—")
        self.eta_var = tk.StringVar(value="—")
        self.elapsed_var = tk.StringVar(value="—")
        ttk.Label(stats, text="Speed:").grid(row=0, column=0, sticky="w")
        ttk.Label(stats, textvariable=self.speed_var).grid(row=0, column=1, sticky="w", padx=(4, 12))
        ttk.Label(stats, text="ETA:").grid(row=0, column=2, sticky="w")
        ttk.Label(stats, textvariable=self.eta_var).grid(row=0, column=3, sticky="w", padx=(4, 12))
        ttk.Label(stats, text="Total:").grid(row=0, column=4, sticky="w")
        ttk.Label(stats, textvariable=self.elapsed_var).grid(row=0, column=5, sticky="w", padx=4)

        val_frame = ttk.LabelFrame(main_row, text="Validation Checks", padding=4)
        val_frame.pack(side="left", fill="both", expand=False, padx=(4, 0))
        self.val_list = tk.Listbox(val_frame, height=10, width=34, font=("Consolas", 9))
        val_scroll = ttk.Scrollbar(val_frame, orient="vertical", command=self.val_list.yview)
        self.val_list.configure(yscrollcommand=val_scroll.set)
        self.val_list.pack(side="left", fill="both", expand=True)
        val_scroll.pack(side="right", fill="y")

        self.message_var = tk.StringVar(value="")
        ttk.Label(progress_tab, textvariable=self.message_var, wraplength=680).pack(
            anchor="w", padx=6, pady=(4, 0),
        )
        self.meta_var = tk.StringVar(value="")
        ttk.Label(progress_tab, textvariable=self.meta_var, foreground="#888").pack(anchor="w", padx=6)

        health_frame = ttk.LabelFrame(progress_tab, text="Feature Health Report")
        self.health_text = scrolledtext.ScrolledText(
            health_frame, height=6, font=("Consolas", 9), wrap="word", state="disabled",
        )
        self.health_text.pack(fill="both", expand=True, padx=4, pady=4)
        health_frame.pack(fill="both", expand=True, padx=4, pady=4)

        self.completion_var = tk.StringVar(value="")
        ttk.Label(progress_tab, textvariable=self.completion_var, foreground="#81c784").pack(
            anchor="w", padx=6, pady=4,
        )
        self.error_var = tk.StringVar(value="")
        ttk.Label(progress_tab, textvariable=self.error_var, foreground="#ef5350", wraplength=680).pack(
            anchor="w", padx=6, pady=2,
        )

        profiler_tab = ttk.Frame(self._notebook)
        self._notebook.add(profiler_tab, text="Profiler")
        self.profiler_text = scrolledtext.ScrolledText(
            profiler_tab,
            height=24,
            font=("Consolas", 9),
            wrap="none",
            state="disabled",
        )
        self.profiler_text.pack(fill="both", expand=True, padx=4, pady=4)
        self._profiler_ready = False

    def reset(self) -> None:
        pl = seed_pipeline_waiting()
        pl["stages"][0]["status"] = "running"
        self._payload = {"status": "running", "pipeline": pl}
        self._profiler_ready = False
        self.profiler_text.configure(state="normal")
        self.profiler_text.delete("1.0", "end")
        self.profiler_text.insert(
            "1.0",
            "Profiler report will appear here when the build completes.",
        )
        self.profiler_text.configure(state="disabled")
        if hasattr(self, "_notebook"):
            self._notebook.select(0)
        self.render(self._payload)

    def render_live_tick(self) -> None:
        if not self._last_payload or self._last_payload.get("status") != "running":
            return
        live = live_pipeline_payload(self._last_payload, self._received_at)
        self._payload = live
        pl = live.get("pipeline") or {}
        self.elapsed_var.set(pl.get("total_elapsed_label") or _fmt_duration(pl.get("total_elapsed_sec")))
        self.eta_var.set(pl.get("eta_label") or "—")
        pl_rows = pl.get("rows_per_sec")
        stage = int(live.get("stage") or 0)
        unit = "steps/sec" if stage == 8 else "rows/sec"
        self.speed_var.set(f"{int(pl_rows):,} {unit}" if pl_rows else "—")
        # Bump elapsed labels on visible tree rows without a full rebuild.
        stage_items = list(pl.get("stages") or [])
        substages = list(pl.get("substages") or [])
        feature_groups = [s for s in substages if _is_feature_group_substage(s)]
        pipeline_subs = [
            s for s in substages
            if not _is_feature_group_substage(s) and s.get("parent_stage") in (None, "build")
        ]
        tree_items = self.tree.get_children()
        if not tree_items:
            self.render(live, from_tick=True)
            return
        idx = 0
        for st in stage_items:
            if idx >= len(tree_items):
                break
            time_lbl = st.get("elapsed_label") or _fmt_duration(st.get("elapsed_sec"))
            vals = list(self.tree.item(tree_items[idx], "values"))
            if len(vals) >= 3:
                vals[2] = time_lbl
                self.tree.item(tree_items[idx], values=vals)
            idx += 1
            if st.get("id") == "build":
                for sub in _visible_pipeline_subs(pipeline_subs, live):
                    if idx >= len(tree_items):
                        break
                    sub_time = sub.get("elapsed_label") or _fmt_duration(sub.get("elapsed_sec"))
                    vals = list(self.tree.item(tree_items[idx], "values"))
                    if len(vals) >= 3:
                        vals[2] = sub_time
                        self.tree.item(tree_items[idx], values=vals)
                    idx += 1
                    if _stage_id_num(sub.get("id")) == 6:
                        for fg in feature_groups:
                            if idx >= len(tree_items):
                                break
                            fg_time = fg.get("elapsed_label") or _fmt_duration(fg.get("elapsed_sec"))
                            vals = list(self.tree.item(tree_items[idx], "values"))
                            if len(vals) >= 3:
                                vals[2] = fg_time
                                self.tree.item(tree_items[idx], values=vals)
                            idx += 1

    def render(self, payload: dict[str, Any], *, from_tick: bool = False) -> None:
        if not from_tick:
            self._last_payload = payload
            self._received_at = time.time()
        self._payload = payload
        pl = payload.get("pipeline") or {}
        completed = payload.get("status") == "completed"
        failed = payload.get("status") in ("failed", "cancelled")
        debug_session = bool(payload.get("debug_load") or payload.get("debug_features"))
        show_subtree = (not completed) or debug_session

        for item in self.tree.get_children():
            self.tree.delete(item)

        substages = list(pl.get("substages") or [])
        feature_groups = [s for s in substages if _is_feature_group_substage(s)]
        pipeline_subs = [
            s for s in substages
            if not _is_feature_group_substage(s) and s.get("parent_stage") in (None, "build")
        ]
        for st in pl.get("stages") or []:
            sid = st.get("id")
            label = st.get("name") or sid
            if show_subtree and not completed:
                label = f"1. {label}" if sid == "build" else label
            time_lbl = st.get("elapsed_label") or _fmt_duration(st.get("elapsed_sec"))
            prog = "—"
            if st.get("progress_total"):
                cur = _fmt_compact(st.get("progress_current"))
                tot = _fmt_compact(st.get("progress_total"))
                unit = st.get("progress_unit") or "rows"
                prog = f"{cur} / {tot} {unit}"
            st_status = st.get("status") or "waiting"
            if sid == "build" and payload.get("status") == "running" and st_status == "waiting":
                st_status = "running"
            if completed and debug_session and sid == "build":
                st_status = "done"
            status = (
                _sub_status_text(st_status)
                if (debug_session and completed)
                else ("" if completed else _status_text(st_status))
            )
            self.tree.insert("", "end", values=(label, status, time_lbl, prog))
            if sid == "build" and show_subtree:
                for sub in _visible_pipeline_subs(pipeline_subs, payload):
                    sub_st = effective_substage_status(sub, payload)
                    sub_time = sub.get("elapsed_label") or _fmt_duration(sub.get("elapsed_sec"))
                    sub_prog = _substage_progress_text(sub, payload, sub_st)
                    self.tree.insert(
                        "",
                        "end",
                        values=(
                            f"    {sub.get('label') or sub.get('name')}",
                            _sub_status_text(sub_st),
                            sub_time,
                            sub_prog,
                        ),
                    )
                    if _stage_id_num(sub.get("id")) == 6 and feature_groups:
                        for fg in feature_groups:
                            fg_st = effective_substage_status(fg, payload)
                            fg_time = fg.get("elapsed_label") or _fmt_duration(fg.get("elapsed_sec"))
                            fg_prog = _substage_progress_text(fg, payload, fg_st)
                            self.tree.insert(
                                "",
                                "end",
                                values=(
                                    f"        {fg.get('label') or fg.get('name')}",
                                    _sub_status_text(fg_st),
                                    fg_time,
                                    fg_prog,
                                ),
                            )

        if completed and not debug_session:
            msg = payload.get("message") or "Build successful — Master Dataset updated"
            self.tree.insert("", "end", values=(f"✓ {msg}", "", pl.get("total_elapsed_label", ""), ""))

        pl_rows = pl.get("rows_per_sec")
        stage = int(payload.get("stage") or 0)
        unit = "steps/sec" if stage == 8 else "rows/sec"
        self.speed_var.set(f"{int(pl_rows):,} {unit}" if pl_rows else "—")
        self.eta_var.set(pl.get("eta_label") or "—")
        self.elapsed_var.set(pl.get("total_elapsed_label") or _fmt_duration(pl.get("total_elapsed_sec")))

        message = payload.get("message") or payload.get("current_check") or ""
        if payload.get("status") == "running":
            self.message_var.set(message)
        elif completed and debug_session:
            self.message_var.set(message)
        else:
            self.message_var.set("")
        meta_parts: list[str] = []
        if payload.get("source_day_index") and payload.get("source_day_total"):
            meta_parts.append(
                f"Trading day {payload['source_day_index']} / {payload['source_day_total']}",
            )
        if payload.get("ticks_in_memory") and not payload.get("rows"):
            meta_parts.append(f"Ticks in memory: {int(payload['ticks_in_memory']):,}")
            spot = payload.get("spot_ticks")
            chain = payload.get("chain_ticks")
            if spot is not None and chain is not None:
                meta_parts.append(f"({int(spot):,} spot + {int(chain):,} chain)")
        elif payload.get("rows"):
            meta_parts.append(f"Rows: {int(payload['rows']):,}")
        g_tot = payload.get("feature_groups_total")
        if g_tot is not None and int(payload.get("stage") or 0) == 6:
            g_done = int(payload.get("feature_groups_done") or 0)
            g_rem = int(payload.get("feature_groups_remaining") or max(0, int(g_tot) - g_done))
            current = payload.get("feature_group_current")
            if payload.get("status") == "running" and current:
                active_n = min(int(g_tot), g_done + 1)
                meta_parts.append(f"Groups: {g_done} done · {g_rem} remaining · group {active_n}/{int(g_tot)}")
                meta_parts.append(f"building {current}")
            else:
                meta_parts.append(f"Groups: {g_done} done · {g_rem} remaining")
            if payload.get("sub_current") is not None and payload.get("sub_total"):
                meta_parts.append(
                    f"{int(payload['sub_current']):,} / {int(payload['sub_total']):,} rows in group",
                )
        elif payload.get("sub_current") and payload.get("sub_total"):
            meta_parts.append(f"Progress: {payload['sub_current']:,} / {payload['sub_total']:,}")
        self.meta_var.set(" · ".join(meta_parts))

        self.val_list.delete(0, "end")
        checks = payload.get("validation_checks") or []
        display = checks
        if completed:
            display = [c for c in checks if c.get("id") in VALIDATION_SUMMARY_IDS and c.get("status") == "pass"]
        for c in display:
            icon = "✓" if c.get("status") == "pass" else ("✗" if c.get("status") == "fail" else "…")
            label = c.get("label") or c.get("id")
            self.val_list.insert("end", f"{icon}  {label}")

        report = (payload.get("feature_policy_report") or (payload.get("dataset_stats") or {}).get("feature_policy_report"))
        if report:
            text = pol_fmt.format_feature_health_report(report)
        else:
            text = ""
        self.health_text.configure(state="normal")
        self.health_text.delete("1.0", "end")
        if text:
            self.health_text.insert("1.0", text)
        else:
            hint = (
                "Feature health report will appear after build completes."
                if not payload.get("debug_load") and not payload.get("debug_features")
                else (
                    "Debug load complete — ticks in memory. Click Debug Features to build features."
                    if payload.get("debug_load")
                    else "Debug build — rows in memory only (no master DB write)."
                )
            )
            self.health_text.insert("1.0", hint)
        self.health_text.configure(state="disabled")

        self.error_var.set(payload.get("error") or "" if failed else "")
        if completed and payload.get("master_dataset_only"):
            stats = payload.get("dataset_stats") or {}
            path = payload.get("master_db_path") or stats.get("master_db_path") or "master DB"
            rows = int(stats.get("rows") or stats.get("rows_added") or 0)
            days = int(stats.get("trading_days") or 0)
            self.completion_var.set(
                f"Master DB updated · {path} · {rows:,} rows · {days} day(s)",
            )
        elif completed and payload.get("debug_features"):
            rows = int(payload.get("rows") or 0)
            feat_n = int(payload.get("feature_count") or 0)
            self.completion_var.set(
                f"Debug features done · {rows:,} rows in memory · {feat_n} features",
            )
        elif completed and payload.get("debug_load"):
            ticks = int(payload.get("ticks_in_memory") or 0)
            spot = int(payload.get("spot_ticks") or 0)
            chain = int(payload.get("chain_ticks") or 0)
            self.completion_var.set(
                f"Debug load done · {ticks:,} ticks in memory ({spot:,} spot + {chain:,} chain)",
            )
        elif completed:
            self.completion_var.set("Build completed.")
        else:
            self.completion_var.set("")
        self._render_profiler_tab(payload)

    def _render_profiler_tab(self, payload: dict[str, Any]) -> None:
        terminal = payload.get("status") in ("completed", "failed", "cancelled")
        report = payload.get("build_profiler_report")
        if not report:
            stats = payload.get("dataset_stats") or {}
            report = stats.get("build_profiler_report")
        gil_report = payload.get("gil_report")
        if not terminal or (not report and not gil_report):
            return
        parts: list[str] = []
        if gil_report:
            parts.append(format_gil_report(gil_report))
        if report:
            if parts:
                parts.append("\n" + "=" * 40 + "\n")
            parts.append(format_build_profiler_report(report))
        text = "\n".join(parts)
        self.profiler_text.configure(state="normal")
        self.profiler_text.delete("1.0", "end")
        self.profiler_text.insert("1.0", text)
        self.profiler_text.configure(state="disabled")
        if not self._profiler_ready:
            self._profiler_ready = True
            self._notebook.select(1)
