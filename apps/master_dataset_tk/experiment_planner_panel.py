"""Experiment Planner — Phase A: Proposal → Template → Job."""

from __future__ import annotations

import tkinter as tk
from datetime import datetime, timezone
from tkinter import messagebox, simpledialog, ttk
from typing import Any, Callable

from .build_service import chart_data_dir
from .experiment_job_runner import ExperimentJobRunner
from .lazy_panel import LazyLoadMixin
from .model_registry_widgets import COL_MUTED, COL_OK, COL_WARN, ScrollableFrame, fmt_num


def _stars(n: int) -> str:
    n = max(0, min(5, int(n or 0)))
    return "★" * n + "☆" * (5 - n)


def _format_experiment_score(score: dict[str, Any]) -> str:
    if not score:
        return "No score yet."
    return (
        f"Experiment Score  {score.get('overall', '—')} / 100  {_stars(score.get('stars') or 0)}\n"
        f"Novelty {score.get('novelty')} · Evidence {score.get('evidence_strength')} · "
        f"Expected Gain {score.get('expected_gain')}\n"
        f"Estimated Time {score.get('estimated_minutes')} min · GPU Cost {score.get('gpu_cost')}\n"
        f"Recommendation: {score.get('recommendation') or 'Review'}\n"
        f"Tags: {', '.join(score.get('tags') or []) or '—'}"
    )


def _parse_ts(value: str | None) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _format_elapsed(started_at: str | None, *, ended_at: str | None = None) -> str:
    start = _parse_ts(started_at)
    if not start:
        return "—"
    end = _parse_ts(ended_at) or datetime.now(timezone.utc)
    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)
    seconds = max(0, int((end - start).total_seconds()))
    minutes, secs = divmod(seconds, 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _format_activity_line(entry: dict[str, Any]) -> str:
    ts = _parse_ts(str(entry.get("ts") or ""))
    stamp = ts.strftime("%H:%M:%S") if ts else "--:--:--"
    step = str(entry.get("step") or "").replace("_", " ")
    message = str(entry.get("message") or "")
    level = str(entry.get("level") or "info")
    prefix = "!" if level == "error" else "·"
    if step:
        return f"[{stamp}] {prefix} {step}: {message}"
    return f"[{stamp}] {prefix} {message}"


def _impact_color(label: str) -> str:
    low = str(label or "").lower()
    if low in ("improved", "retrained"):
        return COL_OK
    if low in ("declined",):
        return COL_WARN
    return COL_MUTED


DECISION_OPTIONS = (
    ("promote_strategy", "Promote Strategy"),
    ("promote_model", "Promote Model"),
    ("archive_as_evidence", "Archive as Evidence"),
    ("repeat_modified_hypothesis", "Repeat with modified hypothesis"),
)


class ExperimentPlannerPanel(ttk.Frame, LazyLoadMixin):
    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        on_open_research_report: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._on_open_research_report = on_open_research_report
        self._pending_report: dict[str, Any] | None = None
        self._active_proposal: dict[str, Any] | None = None
        self._planner_vars: dict[str, tk.BooleanVar] = {}
        self._selected_template_id: str | None = None
        self._selected_job_id: str | None = None
        self._status_var = tk.StringVar(value="")
        self._goal_var = tk.StringVar(value="")
        self._job_runner = ExperimentJobRunner()
        self._live_job_id: str | None = None
        self._job_done_notified = False
        self._live_job_view_id: str | None = None
        self._live_job_widgets: dict[str, Any] | None = None
        self._live_log_count = 0
        self._decision_vars: dict[str, tk.BooleanVar] = {}
        self._static_job_view_id: str | None = None
        self._poll_handled_job_id: str | None = None
        self._build_ui()
        self._lazy_init()

    def _data_dir(self) -> str:
        return chart_data_dir(self.chart_dir)

    def poll_job_progress(self) -> None:
        """Poll only while a job thread is actively running."""
        if not self._job_runner.running:
            return

        job_id = self._live_job_id or self._job_runner.job_id
        if not job_id:
            return

        from chain_replay_ml.fold_research import get_experiment_job

        doc = get_experiment_job(self._data_dir(), job_id)
        if not doc:
            return

        progress = doc.get("progress") or {}
        msg = str(progress.get("message") or doc.get("current_step") or "running")
        self._status_var.set(f"Job #{doc.get('job_number')} running — {msg[:72]}")

        if (
            str(self._notebook.select()) == str(self._tab_job)
            and self._selected_job_id == job_id
        ):
            self._render_job_detail(doc, live=True)

    def _notify_job_finished(self, doc: dict[str, Any]) -> None:
        status = str(doc.get("status") or "")
        num = doc.get("job_number") or "—"
        if status == "failed":
            messagebox.showerror(
                "Experiment Planner",
                f"Job #{num} failed.\n\n{doc.get('error') or 'Unknown error'}",
            )
            return
        verdict = ((doc.get("results") or {}).get("verdict") or {}).get("verdict") or "Complete"
        messagebox.showinfo("Experiment Planner", f"Job #{num} complete.\nVerdict: {verdict}")

    def on_show(self) -> None:
        self.refresh(lazy=True)
        if not self._selected_template_id and not self._selected_job_id:
            self._on_tab_changed()

    def prefill_from_report(self, report: dict[str, Any]) -> None:
        self._pending_report = report
        self._active_proposal = None
        self._notebook.select(self._tab_proposal)
        self._render_proposal_from_report(report)

    def prefill_from_proposal(self, proposal_id: str) -> None:
        from chain_replay_ml.fold_research import get_experiment_proposal

        proposal = get_experiment_proposal(self._data_dir(), proposal_id)
        if not proposal:
            return
        self._active_proposal = proposal
        self._pending_report = None
        self._notebook.select(self._tab_proposal)
        self._render_proposal(proposal)

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Refresh", command=lambda: self.refresh(lazy=True)).pack(side="left")
        ttk.Label(toolbar, textvariable=self._status_var, foreground=COL_MUTED).pack(side="right")

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(body, width=240)
        body.add(left, weight=0)

        tmpl_fr = ttk.LabelFrame(left, text="Templates (by score)", padding=4)
        tmpl_fr.pack(fill="both", expand=True)
        self._template_list = tk.Listbox(tmpl_fr, height=12, exportselection=False, font=("Segoe UI", 9))
        tmpl_scroll = ttk.Scrollbar(tmpl_fr, orient="vertical", command=self._template_list.yview)
        self._template_list.configure(yscrollcommand=tmpl_scroll.set)
        self._template_list.pack(side="left", fill="both", expand=True)
        tmpl_scroll.pack(side="right", fill="y")
        self._template_list.bind("<<ListboxSelect>>", self._on_template_select)
        self._template_index: list[dict[str, Any]] = []

        job_fr = ttk.LabelFrame(left, text="Jobs", padding=4)
        job_fr.pack(fill="both", expand=True, pady=(6, 0))
        self._job_list = tk.Listbox(job_fr, height=8, exportselection=False, font=("Segoe UI", 9))
        job_scroll = ttk.Scrollbar(job_fr, orient="vertical", command=self._job_list.yview)
        self._job_list.configure(yscrollcommand=job_scroll.set)
        self._job_list.pack(side="left", fill="both", expand=True)
        job_scroll.pack(side="right", fill="y")
        self._job_list.bind("<<ListboxSelect>>", self._on_job_select)
        self._job_index: list[dict[str, Any]] = []

        right = ttk.Notebook(body)
        body.add(right, weight=1)
        self._notebook = right

        self._tab_proposal = ttk.Frame(right, padding=6)
        self._tab_template = ttk.Frame(right, padding=6)
        self._tab_job = ttk.Frame(right, padding=6)
        self._tab_legacy = ttk.Frame(right, padding=6)
        right.add(self._tab_proposal, text="Active Proposal")
        right.add(self._tab_template, text="Template")
        right.add(self._tab_job, text="Job Progress")
        right.add(self._tab_legacy, text="Legacy")

        for tab, attr in (
            (self._tab_proposal, "_proposal_host"),
            (self._tab_template, "_template_host"),
            (self._tab_job, "_job_host"),
            (self._tab_legacy, "_legacy_host"),
        ):
            scroll = ScrollableFrame(tab)
            scroll.pack(fill="both", expand=True)
            setattr(self, attr, scroll.inner)
            if attr == "_job_host":
                self._job_scroll = scroll

        self._legacy_list = tk.Listbox(self._tab_legacy, height=6, exportselection=False)
        self._legacy_list.pack(fill="x", pady=(0, 8))
        self._legacy_index: list[dict[str, Any]] = []

        self._notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _on_tab_changed(self, _event: tk.Event | None = None) -> None:
        tab_id = self._notebook.select()
        if tab_id == str(self._tab_template):
            if self._selected_template_id:
                self._load_template_detail(self._selected_template_id)
            else:
                self._render_empty_template()
        elif tab_id == str(self._tab_job):
            if self._selected_job_id:
                self._load_job_detail(self._selected_job_id)
            else:
                self._render_empty_job()
        elif tab_id == str(self._tab_proposal) and not self._active_proposal and not self._pending_report:
            self._render_empty_proposal()

    def _render_empty_template(self) -> None:
        host = self._template_host
        for w in host.winfo_children():
            w.destroy()
        ttk.Label(
            host,
            text="Select a template from the list, or select a job to view its template.",
            foreground=COL_MUTED,
            wraplength=640,
        ).pack(anchor="w", pady=8)

    def _render_empty_job(self) -> None:
        self._clear_live_job_view()
        self._static_job_view_id = None
        host = self._job_host
        self._clear_job_host()
        ttk.Label(
            host,
            text="Select a job from the list to view pipeline progress and results.",
            foreground=COL_MUTED,
            wraplength=640,
        ).pack(anchor="w", pady=8)

    def _render_empty_proposal(self) -> None:
        host = self._proposal_host
        for w in host.winfo_children():
            w.destroy()
        ttk.Label(
            host,
            text="No active proposal. Generate a Research Report and create a proposal from tab 6.",
            foreground=COL_MUTED,
            wraplength=640,
        ).pack(anchor="w", pady=8)

    def _select_listbox_by_id(
        self,
        listbox: tk.Listbox,
        index: list[dict[str, Any]],
        *,
        id_key: str,
        item_id: str,
    ) -> None:
        for i, row in enumerate(index):
            if str(row.get(id_key) or "") == item_id:
                listbox.selection_clear(0, "end")
                listbox.selection_set(i)
                listbox.see(i)
                break

    def refresh(self, *, lazy: bool = False) -> None:
        if lazy:
            self.lazy_load(
                load=self._fetch_planner_lists,
                apply=self._apply_planner_lists,
                message="Loading experiment planner…",
                status_var=self._status_var,
            )
            return
        self._apply_planner_lists(self._fetch_planner_lists())

    def _fetch_planner_lists(self) -> dict[str, Any]:
        from chain_replay_ml.fold_research import (
            get_experiment_job,
            list_experiment_jobs,
            list_experiment_templates,
            list_experiments,
        )

        templates = list_experiment_templates(self._data_dir(), limit=50)
        templates.sort(key=lambda t: int((t.get("score") or {}).get("overall") or 0), reverse=True)
        selected_template_id = self._selected_template_id
        if not selected_template_id and not self._selected_job_id and templates:
            selected_template_id = str(templates[0].get("template_id") or "")
        filter_tid = selected_template_id
        jobs = list_experiment_jobs(self._data_dir(), template_id=filter_tid, limit=50)
        legacy = list_experiments(self._data_dir(), limit=30)
        job_for_template: dict[str, Any] | None = None
        if self._selected_job_id and not selected_template_id:
            job_for_template = get_experiment_job(self._data_dir(), self._selected_job_id)
        return {
            "templates": templates,
            "jobs": jobs,
            "legacy": legacy,
            "selected_template_id": selected_template_id,
            "job_for_template": job_for_template,
        }

    def _apply_planner_lists(self, bundle: dict[str, Any]) -> None:
        templates = list(bundle.get("templates") or [])
        jobs = list(bundle.get("jobs") or [])
        legacy = list(bundle.get("legacy") or [])

        if bundle.get("selected_template_id") and not self._selected_template_id:
            self._selected_template_id = str(bundle["selected_template_id"])

        self._template_list.delete(0, "end")
        self._template_index.clear()
        for row in templates:
            num = row.get("template_number")
            score = (row.get("score") or {}).get("overall") or "—"
            stars = _stars((row.get("score") or {}).get("stars") or 0)
            tags = ", ".join((row.get("tags") or [])[:3])
            label = f"#{num}  {score}  {stars}  {tags[:18]}"
            self._template_list.insert("end", label)
            self._template_index.append(row)

        self._job_list.delete(0, "end")
        self._job_index.clear()
        for row in jobs:
            num = row.get("job_number")
            status = row.get("status") or "pending"
            step = row.get("current_step") or "—"
            label = f"Job #{num}  {status}  {step}"
            self._job_list.insert("end", label)
            self._job_index.append(row)

        self._legacy_list.delete(0, "end")
        self._legacy_index.clear()
        for row in legacy:
            num = row.get("experiment_number")
            status = row.get("status") or "pending"
            self._legacy_list.insert("end", f"Legacy #{num}  {status}")
            self._legacy_index.append(row)

        n_prop = len(self._active_proposal and [self._active_proposal] or [])
        self._status_var.set(f"{len(templates)} templates · {len(jobs)} jobs · {n_prop} active proposal")

        if self._selected_template_id:
            self._select_listbox_by_id(
                self._template_list,
                self._template_index,
                id_key="template_id",
                item_id=self._selected_template_id,
            )
        if self._selected_job_id:
            self._select_listbox_by_id(
                self._job_list,
                self._job_index,
                id_key="job_id",
                item_id=self._selected_job_id,
            )
            if not self._selected_template_id:
                job = bundle.get("job_for_template")
                if job:
                    tid = str(job.get("template_id") or "")
                    if tid:
                        self._selected_template_id = tid
                        self._select_listbox_by_id(
                            self._template_list,
                            self._template_index,
                            id_key="template_id",
                            item_id=tid,
                        )

        if self._pending_report and not self._active_proposal:
            self._render_proposal_from_report(self._pending_report)
        elif self._active_proposal:
            self._render_proposal(self._active_proposal)
        elif str(self._notebook.select()) == str(self._tab_proposal):
            self._render_empty_proposal()

        if self._selected_template_id:
            self._load_template_detail(self._selected_template_id)
        elif str(self._notebook.select()) == str(self._tab_template):
            self._render_empty_template()

        if self._selected_job_id:
            self._load_job_detail(self._selected_job_id)
        elif str(self._notebook.select()) == str(self._tab_job):
            self._render_empty_job()

    def _on_template_select(self, _event: tk.Event) -> None:
        sel = self._template_list.curselection()
        if not sel:
            return
        row = self._template_index[sel[0]]
        tid = str(row.get("template_id") or "")
        self._selected_template_id = tid
        self._load_template_detail(tid)
        self._notebook.select(self._tab_template)
        self.refresh()

    def _on_job_select(self, _event: tk.Event) -> None:
        sel = self._job_list.curselection()
        if not sel:
            return
        row = self._job_index[sel[0]]
        jid = str(row.get("job_id") or "")
        tid = str(row.get("template_id") or "")
        if jid != self._static_job_view_id:
            self._static_job_view_id = None
        self._selected_job_id = jid
        if tid:
            self._selected_template_id = tid
            self._load_template_detail(tid)
        self._load_job_detail(jid)
        self._notebook.select(self._tab_job)

    def _render_proposal_from_report(self, report: dict[str, Any]) -> None:
        from chain_replay_ml.fold_research import get_experiment_planner_view

        view = get_experiment_planner_view(self._data_dir(), report)
        if not view.get("ok"):
            return
        self._goal_var.set(str(view.get("suggested_goal") or ""))
        pseudo = {
            "proposal_number": "—",
            "status": "draft",
            "available_recommendations": view.get("items") or [],
            "selected_recommendations": [
                i for i in (view.get("items") or []) if i.get("accepted_default")
            ],
            "score": {},
            "goal": view.get("suggested_goal"),
            "strategy_label": view.get("strategy_label"),
            "model_id": view.get("model_id"),
            "research_report_id": report.get("report_id"),
            "baseline": {
                "profit_factor": view.get("baseline_pf"),
                "grade": view.get("baseline_grade"),
            },
            "_from_report": True,
        }
        self._render_proposal(pseudo, report=report)

    def _render_proposal(self, proposal: dict[str, Any], *, report: dict[str, Any] | None = None) -> None:
        host = self._proposal_host
        for w in host.winfo_children():
            w.destroy()
        self._planner_vars.clear()

        ttk.Label(
            host,
            text="Select recommendations, score, then freeze into an immutable Template.",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        prov = ttk.LabelFrame(host, text="Source", padding=8)
        prov.pack(fill="x", pady=(0, 8))
        num = proposal.get("proposal_number")
        status = proposal.get("status") or "draft"
        ttk.Label(prov, text=f"Proposal #{num}  ·  {status}").pack(anchor="w")
        ttk.Label(prov, text=f"Report  {str(proposal.get('research_report_id') or '—')[:16]}…").pack(anchor="w")
        ttk.Label(prov, text=f"Model  {proposal.get('model_id') or '—'}").pack(anchor="w")
        ttk.Label(prov, text=f"Strategy  {proposal.get('strategy_label') or '—'}").pack(anchor="w")
        follow_up = (proposal.get("score") or {}).get("follow_up") or {}
        if follow_up.get("suggestion_title"):
            parent_num = follow_up.get("parent_template_number") or "—"
            ttk.Label(
                prov,
                text=(
                    f"Follow-up from Template #{parent_num}  ·  "
                    f"{follow_up.get('suggestion_title')}  ·  "
                    f"Gain: {follow_up.get('expected_information_gain') or '—'}"
                ),
                foreground=COL_MUTED,
                wraplength=640,
            ).pack(anchor="w", pady=(4, 0))

        goal_fr = ttk.Frame(host)
        goal_fr.pack(fill="x", pady=(0, 8))
        ttk.Label(goal_fr, text="Goal", width=8).pack(side="left")
        self._goal_var.set(str(proposal.get("goal") or self._goal_var.get() or ""))
        ttk.Entry(goal_fr, textvariable=self._goal_var, width=64).pack(side="left", fill="x", expand=True)

        rec_fr = ttk.LabelFrame(host, text="Available Recommendations", padding=8)
        rec_fr.pack(fill="x", pady=(0, 8))
        available = proposal.get("available_recommendations") or []
        selected_keys = {
            str(i.get("key") or i.get("text"))
            for i in (proposal.get("selected_recommendations") or [])
        }
        for item in available:
            row = ttk.Frame(rec_fr)
            row.pack(fill="x", pady=2)
            key = str(item.get("key") or item.get("text"))
            default = key in selected_keys if selected_keys else bool(item.get("accepted_default", True))
            var = tk.BooleanVar(value=default)
            self._planner_vars[key] = var
            ttk.Checkbutton(row, variable=var).pack(side="left")
            ttk.Label(row, text=str(item.get("text") or ""), width=42, anchor="w").pack(side="left", padx=(4, 8))
            ttk.Label(row, text=str(item.get("target_label") or ""), foreground=COL_MUTED, width=28, anchor="w").pack(side="left")

        score_fr = ttk.LabelFrame(host, text="Experiment Score", padding=8)
        score_fr.pack(fill="x", pady=(0, 8))
        self._score_label = ttk.Label(score_fr, text=_format_experiment_score(proposal.get("score") or {}), wraplength=640, justify="left")
        self._score_label.pack(anchor="w")

        btn_row = ttk.Frame(host)
        btn_row.pack(fill="x", pady=(8, 0))
        if status == "draft":
            if proposal.get("_from_report"):
                ttk.Button(btn_row, text="Create Proposal", command=lambda: self._create_proposal(report)).pack(side="left")
            else:
                ttk.Button(btn_row, text="Update Selection", command=self._update_proposal).pack(side="left")
                ttk.Button(btn_row, text="Create Template", command=self._create_template).pack(side="left", padx=(8, 0))
            ttk.Button(btn_row, text="Score Proposal", command=lambda: self._score_proposal(report)).pack(side="left", padx=(8, 0))
        elif status == "converted":
            ttk.Label(
                host,
                text=f"Converted to Template #{proposal.get('template_id', '')[:8]}…",
                foreground=COL_OK,
            ).pack(anchor="w", pady=(8, 0))

        ttk.Button(btn_row, text="Select All", command=lambda: self._set_all_checks(True)).pack(side="left", padx=(8, 0))
        ttk.Button(btn_row, text="Clear All", command=lambda: self._set_all_checks(False)).pack(side="left", padx=(4, 0))

    def _set_all_checks(self, value: bool) -> None:
        for var in self._planner_vars.values():
            var.set(value)

    def _selected_keys(self) -> list[str]:
        keys: list[str] = []
        for key, var in self._planner_vars.items():
            if var.get():
                keys.append(key)
        return keys

    def _score_proposal(self, report: dict[str, Any] | None = None) -> None:
        from chain_replay_ml.fold_research import compute_experiment_score

        doc = report or self._pending_report
        if not doc and self._active_proposal:
            baseline = self._active_proposal.get("baseline") or {}
            doc = {
                "ok": True,
                "prediction_run_id": self._active_proposal.get("prediction_run_id"),
                "strategy_run_id": self._active_proposal.get("strategy_run_id"),
                "executive_summary": {
                    "model_id": self._active_proposal.get("model_id"),
                    "strategy": self._active_proposal.get("strategy_label"),
                },
                "baseline_metrics": {
                    "profit_factor": baseline.get("profit_factor"),
                    "win_rate_pct": baseline.get("win_rate_pct"),
                },
            }
        if not doc:
            return
        keys = self._selected_keys()
        if not keys:
            self._score_label.configure(text="Select at least one recommendation.", foreground=COL_WARN)
            return
        proposal = self._active_proposal or {}
        available = proposal.get("available_recommendations") or []
        if not available:
            from chain_replay_ml.fold_research import get_experiment_planner_view
            view = get_experiment_planner_view(self._data_dir(), doc)
            available = view.get("items") or []
        accepted = [i for i in available if str(i.get("key") or i.get("text")) in set(keys)]
        score = compute_experiment_score(
            self._data_dir(),
            doc,
            accepted_items=accepted,
            goal=self._goal_var.get().strip() or None,
        )
        self._score_label.configure(text=_format_experiment_score(score), foreground=COL_OK)

    def _create_proposal_from_suggestion(
        self,
        template_id: str,
        job_id: str,
        suggestion: dict[str, Any],
    ) -> None:
        from chain_replay_ml.fold_research import create_experiment_proposal_from_suggestion

        title = str(suggestion.get("title") or "follow-up experiment")
        if not messagebox.askyesno(
            "Create Proposal",
            f"Create a draft proposal for:\n\n{title}\n\n"
            f"Goal: {suggestion.get('goal') or '—'}",
        ):
            return
        out = create_experiment_proposal_from_suggestion(
            self._data_dir(),
            template_id,
            suggestion,
            source_job_id=job_id or None,
        )
        if not out.get("ok"):
            messagebox.showerror("Experiment Planner", out.get("error") or "Failed")
            return
        proposal = out.get("proposal") or {}
        self._active_proposal = proposal
        self._pending_report = None
        self._goal_var.set(str(proposal.get("goal") or ""))
        messagebox.showinfo(
            "Experiment Planner",
            f"Created Proposal #{proposal.get('proposal_number')} from suggestion.\n"
            "Review selection, then Create Template when ready.",
        )
        self._render_proposal(proposal)
        self.refresh()
        self._notebook.select(self._tab_proposal)

    def _create_proposal(self, report: dict[str, Any] | None) -> None:
        from chain_replay_ml.fold_research import create_experiment_proposal_from_report, update_experiment_proposal_selection

        doc = report or self._pending_report
        if not doc:
            messagebox.showinfo("Experiment Planner", "Load a Research Report first.")
            return
        keys = self._selected_keys()
        if not keys:
            messagebox.showinfo("Experiment Planner", "Select at least one recommendation.")
            return
        out = create_experiment_proposal_from_report(
            self._data_dir(),
            doc,
            goal=self._goal_var.get().strip() or None,
        )
        if not out.get("ok"):
            messagebox.showerror("Experiment Planner", out.get("error") or "Failed")
            return
        proposal = out.get("proposal") or {}
        pid = str(proposal.get("proposal_id") or "")
        out2 = update_experiment_proposal_selection(
            self._data_dir(),
            pid,
            selected_keys=keys,
            goal=self._goal_var.get().strip() or None,
        )
        if not out2.get("ok"):
            messagebox.showerror("Experiment Planner", out2.get("error") or "Failed")
            return
        self._active_proposal = out2.get("proposal")
        self._pending_report = None
        messagebox.showinfo("Experiment Planner", f"Created Proposal #{self._active_proposal.get('proposal_number')}")
        self._render_proposal(self._active_proposal)
        self.refresh()

    def _update_proposal(self) -> None:
        from chain_replay_ml.fold_research import update_experiment_proposal_selection

        proposal = self._active_proposal
        if not proposal:
            return
        keys = self._selected_keys()
        if not keys:
            messagebox.showinfo("Experiment Planner", "Select at least one recommendation.")
            return
        out = update_experiment_proposal_selection(
            self._data_dir(),
            str(proposal.get("proposal_id") or ""),
            selected_keys=keys,
            goal=self._goal_var.get().strip() or None,
        )
        if not out.get("ok"):
            messagebox.showerror("Experiment Planner", out.get("error") or "Failed")
            return
        self._active_proposal = out.get("proposal")
        self._render_proposal(self._active_proposal)

    def _create_template(self) -> None:
        from chain_replay_ml.fold_research import create_experiment_template_from_proposal

        proposal = self._active_proposal
        if not proposal:
            return
        self._update_proposal()
        proposal = self._active_proposal
        if not proposal:
            return
        dup = (proposal.get("score") or {}).get("duplicate_check") or {}
        if dup.get("should_warn"):
            if not messagebox.askyesno(
                "Likely Duplicate",
                f"{dup.get('recommendation') or 'Similar experiment exists.'}\n\nCreate template anyway?",
            ):
                return
        out = create_experiment_template_from_proposal(
            self._data_dir(),
            str(proposal.get("proposal_id") or ""),
        )
        if not out.get("ok"):
            messagebox.showerror("Experiment Planner", out.get("error") or "Failed")
            return
        template = out.get("template") or {}
        self._active_proposal = None
        tid = str(template.get("template_id") or "")
        self._selected_template_id = tid
        messagebox.showinfo(
            "Experiment Planner",
            f"Template #{template.get('template_number')} frozen.\nProposal converted — template is immutable.",
        )
        self.refresh()
        self._load_template_detail(tid)
        self._notebook.select(self._tab_template)

    def _load_template_detail(self, template_id: str) -> None:
        from chain_replay_ml.fold_research import get_experiment_template

        doc = get_experiment_template(self._data_dir(), template_id)
        if not doc:
            self._render_empty_template()
            return
        self._render_template_detail(doc)

    def _render_template_detail(self, doc: dict[str, Any]) -> None:
        host = self._template_host
        for w in host.winfo_children():
            w.destroy()

        num = doc.get("template_number")
        score = doc.get("score") or {}
        ttk.Label(host, text=f"Template #{num}", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(host, text=f"Goal: {doc.get('goal') or '—'}", font=("Segoe UI", 10)).pack(anchor="w", pady=(4, 4))
        ttk.Label(host, text=_format_experiment_score(score), justify="left", wraplength=640).pack(anchor="w", pady=(0, 8))

        stats = doc.get("job_stats") or {}
        ttk.Label(
            host,
            text=f"Jobs: {stats.get('total', 0)} total · {stats.get('completed', 0)} complete · {stats.get('running', 0)} running",
            foreground=COL_MUTED,
        ).pack(anchor="w")

        ch_fr = ttk.LabelFrame(host, text="Frozen Changes (immutable)", padding=8)
        ch_fr.pack(fill="x", pady=(8, 8))
        for ch in doc.get("accepted_changes") or []:
            target = str(ch.get("target") or "strategy_registry")
            if target == "strategy_registry":
                tag, color = "Phase B", COL_OK
            elif target in ("feature_registry", "master_dataset", "model_builder", "hyperparameter_optimization"):
                tag, color = "Phase C", COL_WARN
            elif target == "dataset_migration":
                tag, color = "Phase D", COL_WARN
            else:
                tag, color = "Phase B", COL_OK
            ttk.Label(
                ch_fr,
                text=f"✓ {ch.get('text')}  →  {ch.get('target_label')}  [{tag}]",
                foreground=color,
            ).pack(anchor="w", pady=1)

        routing = doc.get("routing") or {}
        phase_c = sum(
            len(routing.get(k) or [])
            for k in ("feature_changes", "model_changes", "optimization_changes")
        )
        phase_d = len(routing.get("dataset_changes") or [])
        if phase_c:
            ttk.Label(
                host,
                text=f"Run Job will train/retrain model ({phase_c} Phase C change(s)). This may take several minutes.",
                foreground=COL_WARN,
                wraplength=640,
            ).pack(anchor="w", pady=(0, 4))
        if phase_d:
            ttk.Label(
                host,
                text=f"{phase_d} dataset migration item(s) deferred to Phase D.",
                foreground=COL_WARN,
                wraplength=640,
            ).pack(anchor="w", pady=(0, 4))

        btn_row = ttk.Frame(host)
        btn_row.pack(fill="x", pady=(8, 0))
        run_state = "disabled" if self._job_runner.running else "normal"
        ttk.Button(
            btn_row,
            text="Run Job" if not self._job_runner.running else "Job Running…",
            command=lambda: self._run_template(str(doc.get("template_id") or ""), doc),
            state=run_state,
        ).pack(side="left")

    def _run_template(self, template_id: str, template: dict[str, Any] | None = None) -> None:
        from chain_replay_ml.fold_research import get_experiment_template

        if not template_id:
            return
        if self._job_runner.running:
            messagebox.showwarning("Experiment Planner", "An experiment job is already running.")
            return
        doc = template or get_experiment_template(self._data_dir(), template_id) or {}
        routing = doc.get("routing") or {}
        phase_c = sum(
            len(routing.get(k) or [])
            for k in ("feature_changes", "model_changes", "optimization_changes")
        )
        phase_d = len(routing.get("dataset_changes") or [])
        if phase_c:
            if not messagebox.askyesno(
                "Phase C Training",
                f"This template includes {phase_c} model/feature/HPO change(s).\n\n"
                "Run Job will clone the baseline model config and start training "
                "(walk-forward + new prediction run). This may take several minutes.\n\n"
                "Continue?",
            ):
                return
        elif phase_d:
            if not messagebox.askyesno(
                "Partial Run",
                f"This template includes {phase_d} dataset migration item(s) deferred to Phase D.\n\n"
                "Continue with runnable steps only?",
            ):
                return

        self._job_done_notified = False
        self._static_job_view_id = None
        self._poll_handled_job_id = None
        out = self._job_runner.start(
            self._data_dir(),
            template_id,
            on_done=self._on_job_done,
        )
        if not out.get("ok"):
            messagebox.showerror("Experiment Planner", out.get("error") or "Failed to start job")
            return
        job = out.get("job") or {}
        self._selected_job_id = str(job.get("job_id") or "")
        self._selected_template_id = template_id
        self._live_job_id = self._selected_job_id
        self._status_var.set(f"Job #{job.get('job_number')} running…")
        self.refresh()
        self._load_job_detail(self._selected_job_id)
        self._notebook.select(self._tab_job)

    def _on_job_done(self, result: dict[str, Any]) -> None:
        def _finish() -> None:
            job = result.get("job") or {}
            job_id = str(job.get("job_id") or self._live_job_id or "")
            self._job_runner.reset()
            self._live_job_id = None
            self._poll_handled_job_id = job_id or None
            self._static_job_view_id = None
            self._status_var.set("")
            if job_id:
                self._selected_job_id = job_id
                self._clear_live_job_view()
                self._load_job_detail(job_id)
            self.refresh()
            if not result.get("ok") and not self._job_done_notified:
                self._job_done_notified = True
                messagebox.showerror("Experiment Planner", result.get("error") or "Job failed")
            elif result.get("ok") and job and not self._job_done_notified:
                self._job_done_notified = True
                self._notify_job_finished(job)

        self.after(0, _finish)

    def _render_comparison_section(self, host: tk.Misc, doc: dict[str, Any]) -> None:
        comparison = doc.get("comparison") or {}
        if not comparison:
            return

        cmp_fr = ttk.LabelFrame(host, text="Comparison", padding=8)
        cmp_fr.pack(fill="x", pady=(8, 0))

        pf_b = comparison.get("baseline_pf")
        pf_a = comparison.get("after_pf")
        wr_b = comparison.get("baseline_win_rate_pct")
        wr_a = comparison.get("after_win_rate_pct")
        ttk.Label(
            cmp_fr,
            text=(
                f"PF  {fmt_num(pf_b, digits=2) if pf_b is not None else '—'}"
                f" → {fmt_num(pf_a, digits=2) if pf_a is not None else '—'}"
                f"  (Δ {comparison.get('pf_delta') if comparison.get('pf_delta') is not None else '—'})"
            ),
        ).pack(anchor="w")
        ttk.Label(
            cmp_fr,
            text=(
                f"Win Rate  {wr_b if wr_b is None else f'{wr_b:.1f}%'}"
                f" → {wr_a if wr_a is None else f'{wr_a:.1f}%'}"
            ),
        ).pack(anchor="w")
        ttk.Label(
            cmp_fr,
            text=(
                f"Trades  {comparison.get('baseline_trade_count') if comparison.get('baseline_trade_count') is not None else '—'}"
                f" → {comparison.get('after_trade_count') if comparison.get('after_trade_count') is not None else '—'}"
                f"  ({comparison.get('trade_count_delta_pct') if comparison.get('trade_count_delta_pct') is not None else '—'}%)"
            ),
        ).pack(anchor="w")
        ttk.Label(
            cmp_fr,
            text=f"Grade  {comparison.get('baseline_grade') or '—'} → {comparison.get('after_grade') or '—'}",
        ).pack(anchor="w", pady=(4, 0))

        impact = comparison.get("trading_impact") or {}
        if impact:
            impact_fr = ttk.LabelFrame(cmp_fr, text="Trading Impact", padding=8)
            impact_fr.pack(fill="x", pady=(8, 0))
            rows = (
                ("PF", impact.get("pf")),
                ("Win Rate", impact.get("win_rate")),
                ("Trades", impact.get("trades")),
                ("Prediction", impact.get("prediction")),
            )
            for name, label in rows:
                row = ttk.Frame(impact_fr)
                row.pack(fill="x", pady=1)
                ttk.Label(row, text=f"{name:<12}", width=14, anchor="w").pack(side="left")
                ttk.Label(
                    row,
                    text=str(label or "—"),
                    foreground=_impact_color(str(label or "")),
                    font=("Segoe UI", 9, "bold"),
                ).pack(side="left")
            if impact.get("conclusion"):
                ttk.Label(
                    impact_fr,
                    text=f"Conclusion  {impact.get('conclusion')}",
                    wraplength=620,
                    foreground=COL_MUTED,
                ).pack(anchor="w", pady=(6, 0))

    def _sync_job_scroll(self, *, to_bottom: bool = False) -> None:
        scroll = getattr(self, "_job_scroll", None)
        if not scroll:
            return
        canvas = scroll._canvas
        canvas.update_idletasks()
        canvas.configure(scrollregion=canvas.bbox("all"))
        if to_bottom:
            canvas.yview_moveto(1.0)
        else:
            canvas.yview_moveto(0.0)

    def _render_decision_section(self, host: tk.Misc, doc: dict[str, Any]) -> ttk.LabelFrame | None:
        job_id = str(doc.get("job_id") or "")
        if not job_id:
            return None

        decision = (doc.get("results") or {}).get("decision") or {}
        dec_fr = ttk.LabelFrame(host, text="Decision", padding=8)
        dec_fr.pack(fill="x", pady=(8, 0))
        ttk.Label(
            dec_fr,
            text="Most experiments won't become champions — they're still valuable evidence.",
            foreground=COL_MUTED,
            wraplength=620,
        ).pack(anchor="w", pady=(0, 6))

        self._decision_vars.clear()
        for key, label in DECISION_OPTIONS:
            var = tk.BooleanVar(value=bool(decision.get(key)))
            self._decision_vars[key] = var
            ttk.Checkbutton(dec_fr, text=label, variable=var).pack(anchor="w", pady=1)

        btn_row = ttk.Frame(dec_fr)
        btn_row.pack(fill="x", pady=(6, 0))
        ttk.Button(
            btn_row,
            text="Save Decision",
            command=lambda: self._save_job_decision(job_id),
        ).pack(side="left")
        return dec_fr

    def _render_follow_up_action(self, host: tk.Misc, doc: dict[str, Any]) -> None:
        job_id = str(doc.get("job_id") or "")
        if not job_id:
            return
        follow_up = (doc.get("results") or {}).get("follow_up")
        action_fr = ttk.Frame(host)
        action_fr.pack(fill="x", pady=(10, 0))
        ttk.Button(
            action_fr,
            text="Create Follow-up Experiment",
            command=lambda: self._create_follow_up_experiment(job_id),
        ).pack(side="left")
        if follow_up and follow_up.get("title"):
            ttk.Label(
                action_fr,
                text=f"  → {follow_up.get('title')}",
                foreground=COL_MUTED,
                wraplength=480,
            ).pack(side="left", padx=(8, 0))

    def _save_job_decision(self, job_id: str) -> None:
        from chain_replay_ml.fold_research import update_experiment_job_decision

        payload = {key: var.get() for key, var in self._decision_vars.items()}
        out = update_experiment_job_decision(self._data_dir(), job_id, payload)
        if not out.get("ok"):
            messagebox.showerror("Experiment Planner", out.get("error") or "Failed to save decision")
            return
        messagebox.showinfo("Experiment Planner", "Decision saved.")

    def _create_follow_up_experiment(self, job_id: str) -> None:
        from chain_replay_ml.fold_research import create_follow_up_template_from_job, get_experiment_job

        job = get_experiment_job(self._data_dir(), job_id) or {}
        follow_up = (job.get("results") or {}).get("follow_up") or {}
        title = str(follow_up.get("title") or "Follow-up experiment")
        goal = str(follow_up.get("goal") or "")
        changes = follow_up.get("selection") or []
        change_lines = "\n".join(f"  • {c.get('text')}" for c in changes[:5]) or "  • (auto-selected)"
        if not messagebox.askyesno(
            "Create Follow-up Experiment",
            f"Create a frozen follow-up template:\n\n"
            f"{title}\n\n"
            f"Goal: {goal or '—'}\n\n"
            f"Changes:\n{change_lines}\n\n"
            f"Review on the Template tab, then click Run Job.",
        ):
            return
        out = create_follow_up_template_from_job(self._data_dir(), job_id)
        if not out.get("ok"):
            messagebox.showerror("Experiment Planner", out.get("error") or "Failed")
            return
        template = out.get("template") or {}
        tid = str(template.get("template_id") or "")
        self._selected_template_id = tid
        messagebox.showinfo(
            "Experiment Planner",
            f"Follow-up Template #{template.get('template_number')} ready.\n"
            "Review changes, then click Run Job.",
        )
        self.refresh()
        self._load_template_detail(tid)
        self._notebook.select(self._tab_template)

    def _clear_job_host(self) -> None:
        for w in self._job_host.winfo_children():
            w.destroy()

    def _clear_live_job_view(self) -> None:
        self._live_job_view_id = None
        self._live_job_widgets = None
        self._live_log_count = 0

    def _pipeline_step_style(
        self,
        step: str,
        *,
        steps: list[str],
        current: str,
        status: str,
    ) -> tuple[str, str]:
        label = step.replace("_", " ").title()
        if status == "complete":
            return f"✓ {label}", COL_OK
        if step == current:
            return f"▶ {label}", COL_WARN
        if current in steps and steps.index(step) < steps.index(current):
            return f"✓ {label}", COL_OK
        return f"○ {label}", COL_MUTED

    def _build_live_job_view(self, doc: dict[str, Any]) -> None:
        host = self._job_host
        self._clear_job_host()
        self._clear_live_job_view()

        num = doc.get("job_number")
        progress = doc.get("progress") or {}
        steps = progress.get("steps") or []

        status_var = tk.StringVar(value="running — live updates")
        elapsed_var = tk.StringVar()
        progress_pct_var = tk.StringVar(value="Progress  0%")
        message_var = tk.StringVar()
        explanation_var = tk.StringVar()

        ttk.Label(host, text=f"Job #{num}", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(
            host,
            textvariable=status_var,
            foreground=COL_WARN,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(4, 2))
        ttk.Label(host, textvariable=elapsed_var, foreground=COL_MUTED).pack(anchor="w", pady=(0, 4))

        bar_fr = ttk.Frame(host)
        bar_fr.pack(fill="x", pady=(0, 8))
        ttk.Label(bar_fr, textvariable=progress_pct_var, foreground=COL_MUTED).pack(anchor="w")
        progress_bar = ttk.Progressbar(bar_fr, maximum=100, value=0)
        progress_bar.pack(fill="x", pady=(2, 0))

        msg_fr = ttk.LabelFrame(host, text="Current Step", padding=8)
        msg_fr.pack(fill="x", pady=(0, 8))
        ttk.Label(
            msg_fr,
            textvariable=message_var,
            foreground=COL_WARN,
            wraplength=640,
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w")
        ttk.Label(msg_fr, textvariable=explanation_var, foreground=COL_MUTED, wraplength=640).pack(anchor="w", pady=(4, 0))

        log_fr = ttk.LabelFrame(host, text="Live Activity", padding=8)
        log_fr.pack(fill="x", pady=(0, 8))
        log_text = tk.Text(log_fr, height=10, wrap="word", font=("Consolas", 9))
        log_scroll = ttk.Scrollbar(log_fr, orient="vertical", command=log_text.yview)
        log_text.configure(yscrollcommand=log_scroll.set, state="disabled")
        log_text.pack(side="left", fill="both", expand=True)
        log_scroll.pack(side="right", fill="y")
        log_text.tag_configure("error", foreground="#b00020")
        log_text.tag_configure("info", foreground="#333333")

        pipe_fr = ttk.LabelFrame(host, text="Pipeline", padding=8)
        pipe_fr.pack(fill="x", pady=(0, 8))
        step_labels: dict[str, ttk.Label] = {}
        for step in steps:
            text, fg = self._pipeline_step_style(
                step,
                steps=steps,
                current=str(doc.get("current_step") or ""),
                status=str(doc.get("status") or "running"),
            )
            lbl = ttk.Label(pipe_fr, text=text, foreground=fg)
            lbl.pack(anchor="w", pady=1)
            step_labels[step] = lbl

        self._live_job_view_id = str(doc.get("job_id") or "")
        self._live_job_widgets = {
            "status_var": status_var,
            "elapsed_var": elapsed_var,
            "progress_pct_var": progress_pct_var,
            "message_var": message_var,
            "explanation_var": explanation_var,
            "progress_bar": progress_bar,
            "log_text": log_text,
            "step_labels": step_labels,
            "steps": steps,
        }
        self._live_log_count = 0
        self._update_live_job_view(doc)

    def _append_live_activity(self, entries: list[dict[str, Any]]) -> None:
        widgets = self._live_job_widgets
        if not widgets:
            return
        log_text: tk.Text = widgets["log_text"]
        if len(entries) <= self._live_log_count:
            return
        log_text.configure(state="normal")
        for entry in entries[self._live_log_count:]:
            line = _format_activity_line(entry) + "\n"
            tag = "error" if entry.get("level") == "error" else "info"
            log_text.insert("end", line, tag)
        self._live_log_count = len(entries)
        log_text.configure(state="disabled")
        log_text.see("end")

    def _update_live_job_view(self, doc: dict[str, Any]) -> None:
        widgets = self._live_job_widgets
        if not widgets:
            return

        progress = doc.get("progress") or {}
        steps = widgets.get("steps") or progress.get("steps") or []
        current = str(doc.get("current_step") or "")
        step_index = int(progress.get("step_index") or 0)
        total_steps = int(progress.get("total_steps") or len(steps) or 1)
        pct = min(100, max(0, int((step_index / max(total_steps - 1, 1)) * 100)))

        widgets["status_var"].set("running — live updates")
        widgets["elapsed_var"].set(
            f"Elapsed: {_format_elapsed(str(doc.get('started_at') or ''))}"
        )
        widgets["progress_pct_var"].set(f"Progress  {pct}%")
        widgets["progress_bar"]["value"] = pct
        widgets["message_var"].set(str(progress.get("message") or current.replace("_", " ")))
        widgets["explanation_var"].set(str(progress.get("step_explanation") or ""))

        for step, lbl in (widgets.get("step_labels") or {}).items():
            text, fg = self._pipeline_step_style(
                step,
                steps=steps,
                current=current,
                status=str(doc.get("status") or "running"),
            )
            lbl.configure(text=text, foreground=fg)

        self._append_live_activity(progress.get("activity_log") or [])

    def _load_job_detail(self, job_id: str, *, force: bool = False) -> None:
        from chain_replay_ml.fold_research import get_experiment_job
        from chain_replay_ml.fold_research.experiment_pipeline import reprocess_experiment_job_closure

        doc = get_experiment_job(self._data_dir(), job_id)
        if not doc:
            self._static_job_view_id = None
            self._render_empty_job()
            return
        if doc.get("status") == "complete" and (
            not (doc.get("results") or {}).get("verdict")
            or not (doc.get("comparison") or {}).get("trading_impact")
            or not (doc.get("results") or {}).get("decision")
        ):
            try:
                out = reprocess_experiment_job_closure(self._data_dir(), job_id)
                if out.get("ok"):
                    doc = out.get("job") or doc
                    force = True
            except Exception:
                pass

        if not force and job_id == self._static_job_view_id and not self._job_runner.running:
            return

        is_live = (
            str(doc.get("job_id") or "") == self._live_job_id
            and self._job_runner.running
        )
        self._render_job_detail(doc, live=is_live, force=force)

    def _render_job_detail(self, doc: dict[str, Any], *, live: bool = False, force: bool = False) -> None:
        job_id = str(doc.get("job_id") or "")
        status = doc.get("status") or "pending"
        is_running = status == "running" or (live and self._job_runner.running)

        if is_running and live and job_id and job_id == self._live_job_id:
            if self._live_job_view_id != job_id or not self._live_job_widgets:
                self._build_live_job_view(doc)
            else:
                self._update_live_job_view(doc)
            return

        if not force and not is_running and job_id and job_id == self._static_job_view_id:
            return

        self._clear_live_job_view()
        host = self._job_host
        self._clear_job_host()

        num = doc.get("job_number")
        status = doc.get("status") or "pending"
        is_running = status == "running" or self._job_runner.running
        ttk.Label(host, text=f"Job #{num}", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        color = COL_OK if status == "complete" else (COL_WARN if is_running else COL_MUTED)
        if status == "failed":
            color = COL_WARN
        status_text = "running — live updates" if is_running else status
        ttk.Label(host, text=f"Status: {status_text}", foreground=color, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 2))
        elapsed = _format_elapsed(
            str(doc.get("started_at") or ""),
            ended_at=str(doc.get("completed_at") or "") if not is_running else None,
        )
        ttk.Label(host, text=f"Elapsed: {elapsed}", foreground=COL_MUTED).pack(anchor="w", pady=(0, 4))
        if doc.get("error"):
            ttk.Label(host, text=f"Error: {doc.get('error')}", foreground=COL_WARN, wraplength=640).pack(anchor="w", pady=(0, 8))

        tid = str(doc.get("template_id") or "")
        if tid:
            link_fr = ttk.Frame(host)
            link_fr.pack(fill="x", pady=(0, 8))
            ttk.Button(
                link_fr,
                text="View Template",
                command=lambda: self._open_template_tab(tid),
            ).pack(side="left")

        progress = doc.get("progress") or {}
        steps = progress.get("steps") or []
        current = doc.get("current_step") or ""
        step_index = int(progress.get("step_index") or 0)
        total_steps = int(progress.get("total_steps") or len(steps) or 1)
        if is_running and total_steps:
            pct = min(100, max(0, int((step_index / max(total_steps - 1, 1)) * 100)))
            bar_fr = ttk.Frame(host)
            bar_fr.pack(fill="x", pady=(0, 8))
            ttk.Label(bar_fr, text=f"Progress  {pct}%", foreground=COL_MUTED).pack(anchor="w")
            pb = ttk.Progressbar(bar_fr, maximum=100, value=pct)
            pb.pack(fill="x", pady=(2, 0))

        if progress.get("message"):
            msg_fr = ttk.LabelFrame(host, text="Current Step", padding=8)
            msg_fr.pack(fill="x", pady=(0, 8))
            ttk.Label(
                msg_fr,
                text=str(progress.get("message")),
                foreground=COL_WARN if is_running else COL_MUTED,
                wraplength=640,
                font=("Segoe UI", 10, "bold" if is_running else "normal"),
            ).pack(anchor="w")
            explanation = progress.get("step_explanation") or ""
            if explanation:
                ttk.Label(msg_fr, text=explanation, foreground=COL_MUTED, wraplength=640).pack(anchor="w", pady=(4, 0))

        activity = progress.get("activity_log") or []
        if activity or is_running:
            log_fr = ttk.LabelFrame(host, text="Live Activity", padding=8)
            log_fr.pack(fill="x", pady=(0, 8))
            log_text = tk.Text(log_fr, height=min(10, max(4, len(activity))), wrap="word", font=("Consolas", 9))
            log_scroll = ttk.Scrollbar(log_fr, orient="vertical", command=log_text.yview)
            log_text.configure(yscrollcommand=log_scroll.set, state="normal")
            for entry in activity:
                line = _format_activity_line(entry) + "\n"
                tag = "error" if entry.get("level") == "error" else "info"
                log_text.insert("end", line, tag)
            log_text.tag_configure("error", foreground="#b00020")
            log_text.tag_configure("info", foreground="#333333")
            log_text.configure(state="disabled")
            log_text.pack(side="left", fill="both", expand=True)
            log_scroll.pack(side="right", fill="y")
            if is_running:
                log_text.see("end")

        pipe_fr = ttk.LabelFrame(host, text="Pipeline", padding=8)
        pipe_fr.pack(fill="x", pady=(0, 8))
        for step in steps:
            text, fg = self._pipeline_step_style(
                step,
                steps=steps,
                current=current,
                status=status,
            )
            ttk.Label(pipe_fr, text=text, foreground=fg).pack(anchor="w", pady=1)

        if is_running:
            return

        results = doc.get("results") or {}
        verdict = results.get("verdict") or {}
        if verdict:
            v_fr = ttk.LabelFrame(host, text="Experiment Verdict", padding=8)
            v_fr.pack(fill="x", pady=(8, 0))
            v_color = COL_OK if verdict.get("verdict") == "Improvement" else (
                COL_WARN if verdict.get("verdict") == "Regression" else COL_MUTED
            )
            ttk.Label(
                v_fr,
                text=f"{verdict.get('verdict') or '—'}  ·  Confidence {verdict.get('confidence') or '—'}",
                foreground=v_color,
                font=("Segoe UI", 11, "bold"),
            ).pack(anchor="w")
            for reason in verdict.get("reasons") or []:
                ttk.Label(v_fr, text=f"• {reason}", wraplength=640).pack(anchor="w")
            if verdict.get("recommendation"):
                ttk.Label(v_fr, text=f"Recommendation: {verdict.get('recommendation')}", wraplength=640).pack(anchor="w", pady=(4, 0))

        info_gain = results.get("information_gain") or {}
        if info_gain:
            ttk.Label(
                host,
                text=f"Information Gain: {info_gain.get('label') or '—'} ({info_gain.get('score') or '—'}/100) — {info_gain.get('note') or ''}",
                foreground=COL_MUTED,
                wraplength=640,
            ).pack(anchor="w", pady=(6, 0))

        root_cause = results.get("root_cause") or {}
        if root_cause.get("most_likely"):
            rc_fr = ttk.LabelFrame(host, text="Root Cause", padding=8)
            rc_fr.pack(fill="x", pady=(8, 0))
            ttk.Label(rc_fr, text=str(root_cause.get("most_likely")), wraplength=640).pack(anchor="w")
            for factor in (root_cause.get("factors") or [])[1:3]:
                ttk.Label(rc_fr, text=f"• {factor}", foreground=COL_MUTED, wraplength=640).pack(anchor="w")

        self._render_comparison_section(host, doc)

        closure = results.get("closure") or {}
        if closure.get("checklist"):
            cl_fr = ttk.LabelFrame(host, text="Auto-closure", padding=8)
            cl_fr.pack(fill="x", pady=(8, 0))
            checks = closure.get("checklist") or {}
            ttk.Label(
                cl_fr,
                text=(
                    f"Knowledge Updated {'✓' if checks.get('knowledge_extracted') else '—'}  ·  "
                    f"Research Report {'✓' if checks.get('research_report') else '—'}  ·  "
                    f"Next Experiments {closure.get('next_experiments_count') or 0} generated"
                ),
            ).pack(anchor="w")

        next_exps = results.get("next_experiments") or []
        if next_exps:
            nx_fr = ttk.LabelFrame(host, text="Suggested Next Experiments", padding=8)
            nx_fr.pack(fill="x", pady=(8, 0))
            job_id = str(doc.get("job_id") or "")
            template_id = str(doc.get("template_id") or "")
            for item in next_exps[:5]:
                row = ttk.Frame(nx_fr)
                row.pack(fill="x", pady=2)
                stars = "★" * int(item.get("stars") or 0) + "☆" * (5 - int(item.get("stars") or 0))
                text_fr = ttk.Frame(row)
                text_fr.pack(side="left", fill="x", expand=True)
                ttk.Label(
                    text_fr,
                    text=f"{stars}  {item.get('title') or '—'}  ·  Gain: {item.get('expected_information_gain') or '—'}",
                    wraplength=520,
                ).pack(anchor="w")
                if item.get("reason"):
                    ttk.Label(text_fr, text=f"   {item.get('reason')}", foreground=COL_MUTED, wraplength=500).pack(anchor="w")
                if template_id:
                    ttk.Button(
                        row,
                        text="Create Proposal",
                        command=lambda s=item, tid=template_id, jid=job_id: self._create_proposal_from_suggestion(tid, jid, s),
                    ).pack(side="right", padx=(8, 0))

        knowledge = (doc.get("results") or {}).get("knowledge") or {}
        if knowledge.get("findings_updated"):
            ttk.Label(
                host,
                text=f"Knowledge Base updated — {knowledge.get('findings_updated')} finding(s).",
                foreground=COL_OK,
            ).pack(anchor="w", pady=(8, 0))

        self._render_decision_section(host, doc)
        self._render_follow_up_action(host, doc)

        outputs = doc.get("outputs") or {}
        if outputs:
            out_fr = ttk.LabelFrame(host, text="Outputs (technical)", padding=8)
            out_fr.pack(fill="x", pady=(8, 0))
            ttk.Label(out_fr, text=f"Phase  {outputs.get('phase') or '—'}").pack(anchor="w")
            ttk.Label(out_fr, text=f"Model  {outputs.get('model_name') or '—'}").pack(anchor="w")
            ttk.Label(out_fr, text=f"Training mode  {outputs.get('training_mode') or '—'}").pack(anchor="w")
            ttk.Label(out_fr, text=f"Prediction run  {str(outputs.get('prediction_run_id') or '—')[:16]}…").pack(anchor="w")
            ttk.Label(out_fr, text=f"Strategy version  {outputs.get('strategy_version_label') or outputs.get('strategy_version_id') or '—'}").pack(anchor="w")
            ttk.Label(out_fr, text=f"Strategy run  {str(outputs.get('strategy_run_id') or '—')[:16]}…").pack(anchor="w")
            ttk.Label(out_fr, text=f"Research report  {str(outputs.get('research_report_id') or '—')[:16]}…").pack(anchor="w")
            for note in outputs.get("notes") or []:
                ttk.Label(out_fr, text=f"Note: {note}", foreground=COL_MUTED, wraplength=640).pack(anchor="w")
            if outputs.get("partial_run"):
                ttk.Label(
                    out_fr,
                    text="Partial Phase B run — see deferred notes above.",
                    foreground=COL_WARN,
                ).pack(anchor="w", pady=(4, 0))

        ttk.Frame(host, height=8).pack()

        job_id = str(doc.get("job_id") or "")
        if job_id and not is_running:
            self._static_job_view_id = job_id
            self.after_idle(lambda: self._sync_job_scroll(to_bottom=False))

    def _open_template_tab(self, template_id: str) -> None:
        self._selected_template_id = template_id
        self._load_template_detail(template_id)
        self._notebook.select(self._tab_template)
        self.refresh()
