"""Research Programs — Phase D1 UI."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from typing import Any

from .build_service import chart_data_dir
from .lazy_panel import LazyLoadMixin
from .model_registry_widgets import COL_MUTED, COL_OK, COL_WARN, ScrollableFrame
from .research_campaign_coordinator import get_research_campaign_coordinator


class ResearchProgramPanel(ttk.Frame, LazyLoadMixin):
    def __init__(self, master: tk.Misc, *, chart_dir: str) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._selected_program_id: str | None = None
        self._selected_campaign_id: str | None = None
        self._status_var = tk.StringVar(value="")
        self._auto_run_var = tk.BooleanVar(value=False)
        self._coordinator = get_research_campaign_coordinator()
        self._coordinator.subscribe(self._on_coordinator_update)
        self._portfolio_mode = False
        self._build_ui()
        self._lazy_init()

    def _data_dir(self) -> str:
        return chart_data_dir(self.chart_dir)

    def on_show(self) -> None:
        self.refresh(lazy=True)

    def on_coordinator_tick(self) -> None:
        """Refresh local status while global coordinator runs jobs."""
        if self._coordinator.running:
            cid = self._coordinator.active_campaign_id or self._selected_campaign_id or ""
            if cid:
                self._status_var.set(f"Running experiment on campaign {cid[:8]}…")
        elif self._selected_campaign_id:
            self._load_campaign_detail(self._selected_campaign_id)

    def _on_coordinator_update(self) -> None:
        def _refresh() -> None:
            if self._selected_campaign_id:
                self._load_campaign_detail(self._selected_campaign_id)
            self.refresh(lazy=True)

        try:
            self.after(0, _refresh)
        except tk.TclError:
            pass

    def poll_scheduler(self) -> None:
        """Legacy hook — global tick runs from app; watch selected campaign."""
        if self._selected_campaign_id:
            self._coordinator.watch_campaign(self._selected_campaign_id)

    def _build_ui(self) -> None:
        toolbar = ttk.Frame(self, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Refresh", command=lambda: self.refresh(lazy=True)).pack(side="left")
        ttk.Button(toolbar, text="New Program", command=self._new_program).pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="New Campaign", command=self._new_campaign).pack(side="left", padx=(4, 0))
        ttk.Button(toolbar, text="Desk Portfolio", command=self._show_desk_portfolio).pack(side="left", padx=(8, 0))
        ttk.Button(toolbar, text="Campaign Outcome", command=self._show_campaign_outcome).pack(side="left", padx=(4, 0))
        ttk.Label(toolbar, textvariable=self._status_var, foreground=COL_MUTED).pack(side="right")

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=4)

        left = ttk.Frame(body, width=260)
        body.add(left, weight=0)

        prog_fr = ttk.LabelFrame(left, text="Research Programs", padding=4)
        prog_fr.pack(fill="both", expand=True)
        self._program_list = tk.Listbox(prog_fr, height=10, exportselection=False, font=("Segoe UI", 9))
        prog_scroll = ttk.Scrollbar(prog_fr, orient="vertical", command=self._program_list.yview)
        self._program_list.configure(yscrollcommand=prog_scroll.set)
        self._program_list.pack(side="left", fill="both", expand=True)
        prog_scroll.pack(side="right", fill="y")
        self._program_list.bind("<<ListboxSelect>>", self._on_program_select)
        self._program_index: list[dict[str, Any]] = []

        camp_fr = ttk.LabelFrame(left, text="Campaigns", padding=4)
        camp_fr.pack(fill="both", expand=True, pady=(6, 0))
        self._campaign_list = tk.Listbox(camp_fr, height=10, exportselection=False, font=("Segoe UI", 9))
        camp_scroll = ttk.Scrollbar(camp_fr, orient="vertical", command=self._campaign_list.yview)
        self._campaign_list.configure(yscrollcommand=camp_scroll.set)
        self._campaign_list.pack(side="left", fill="both", expand=True)
        camp_scroll.pack(side="right", fill="y")
        self._campaign_list.bind("<<ListboxSelect>>", self._on_campaign_select)
        self._campaign_index: list[dict[str, Any]] = []

        right = ttk.Frame(body)
        body.add(right, weight=1)
        scroll = ScrollableFrame(right)
        scroll.pack(fill="both", expand=True)
        self._detail_host = scroll.inner

    def refresh(self, *, lazy: bool = False) -> None:
        if lazy:
            self.lazy_load(
                load=self._fetch_refresh_data,
                apply=self._apply_refresh_data,
                message="Loading research programs…",
                status_var=self._status_var,
            )
            return
        self._apply_refresh_data(self._fetch_refresh_data())

    def _fetch_refresh_data(self) -> dict[str, Any]:
        from chain_replay_ml.fold_research import list_research_campaigns, list_research_programs

        programs = list_research_programs(self._data_dir(), limit=50)
        campaigns: list[dict[str, Any]] = []
        pid = self._selected_program_id
        if not pid and programs:
            pid = str(programs[0].get("program_id") or "")
        if pid:
            campaigns = list_research_campaigns(self._data_dir(), program_id=pid, limit=50)
        return {"programs": programs, "campaigns": campaigns, "program_id": pid}

    def _apply_refresh_data(self, bundle: dict[str, Any]) -> None:
        programs = list(bundle.get("programs") or [])
        campaigns = list(bundle.get("campaigns") or [])

        self._program_list.delete(0, "end")
        self._program_index.clear()
        for row in programs:
            num = row.get("program_number")
            imp = row.get("importance") or "—"
            stats = row.get("campaign_stats") or {}
            label = f"#{num}  {row.get('name', '')[:28]}  [{imp}]  {stats.get('total', 0)}c"
            self._program_list.insert("end", label)
            self._program_index.append(row)

        if self._selected_program_id:
            for i, row in enumerate(self._program_index):
                if str(row.get("program_id")) == self._selected_program_id:
                    self._program_list.selection_set(i)
                    break
        elif programs:
            self._selected_program_id = str(programs[0].get("program_id") or "")
            self._program_list.selection_set(0)

        self._campaign_list.delete(0, "end")
        self._campaign_index.clear()
        pid = self._selected_program_id
        if not pid:
            self._render_empty()
            self._status_var.set(f"{len(programs)} programs")
            return
        for row in campaigns:
            num = row.get("campaign_number")
            status = row.get("status") or "—"
            label = f"#{num}  {row.get('name', '')[:24]}  {status}"
            self._campaign_list.insert("end", label)
            self._campaign_index.append(row)
        if self._selected_campaign_id:
            for i, row in enumerate(self._campaign_index):
                if str(row.get("campaign_id")) == self._selected_campaign_id:
                    self._campaign_list.selection_set(i)
                    break

        if self._portfolio_mode:
            self._show_desk_portfolio()
        elif self._selected_campaign_id:
            self._load_campaign_detail(self._selected_campaign_id)
        elif self._selected_program_id:
            self._load_program_portfolio(self._selected_program_id)
        self._status_var.set(f"{len(programs)} programs")

    def _reload_campaigns(self) -> None:
        self.refresh(lazy=False)

    def _show_desk_portfolio(self) -> None:
        self._portfolio_mode = True
        self._selected_campaign_id = None
        self._campaign_list.selection_clear(0, "end")
        host = self._detail_host
        for w in host.winfo_children():
            w.destroy()

        from chain_replay_ml.fold_research import get_research_portfolio

        port = get_research_portfolio(self._data_dir())
        if not port.get("ok"):
            ttk.Label(host, text=port.get("error") or "Portfolio unavailable", foreground=COL_WARN).pack(anchor="w")
            return

        stats = port.get("global_stats") or {}
        ttk.Label(host, text="Desk Portfolio", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(
            host,
            text=(
                f"{len(port.get('programs') or [])} programs  ·  "
                f"{stats.get('total_campaigns', 0)} active campaigns  ·  "
                f"{stats.get('total_experiments', 0)} experiments  ·  "
                f"{stats.get('global_running_jobs', 0)} jobs running"
            ),
            foreground=COL_MUTED,
        ).pack(anchor="w", pady=(4, 8))

        prio_fr = ttk.LabelFrame(host, text="Priority Queue (running campaigns)", padding=6)
        prio_fr.pack(fill="x", pady=(0, 8))
        queue = port.get("priority_queue") or []
        if not queue:
            ttk.Label(prio_fr, text="No running campaigns.", foreground=COL_MUTED).pack(anchor="w")
        for card in queue[:6]:
            self._render_campaign_card(prio_fr, card, compact=True)

        prog_fr = ttk.LabelFrame(host, text="Programs", padding=6)
        prog_fr.pack(fill="both", expand=True)
        for prog in port.get("programs") or []:
            pst = prog.get("stats") or {}
            gen = prog.get("best_generalization") or {}
            gen_txt = f"  Gen {gen.get('overall')}" if gen.get("overall") is not None else ""
            ttk.Label(
                prog_fr,
                text=(
                    f"#{prog.get('program_number')}  {prog.get('name', '')[:36]}  "
                    f"[{prog.get('importance')}]  {pst.get('total_campaigns', 0)} campaigns  "
                    f"{pst.get('running_jobs', 0)} running{gen_txt}"
                ),
                font=("Consolas", 9),
            ).pack(anchor="w", pady=1)
        self._status_var.set("Desk portfolio")

    def _show_campaign_outcome(self) -> None:
        if not self._selected_campaign_id:
            messagebox.showinfo("Campaign Outcome", "Select a campaign first.", parent=self)
            return
        from chain_replay_ml.fold_research import get_campaign_outcome

        out = get_campaign_outcome(self._data_dir(), self._selected_campaign_id)
        if not out.get("ok"):
            messagebox.showerror("Campaign Outcome", out.get("error") or "Unavailable", parent=self)
            return

        win = tk.Toplevel(self.winfo_toplevel())
        win.title("Campaign Outcome")
        win.transient(self.winfo_toplevel())
        win.geometry("760x620")

        body = ScrollableFrame(win)
        body.pack(fill="both", expand=True)
        host = body.inner
        pad = ttk.Frame(host, padding=14)
        pad.pack(fill="both", expand=True)

        summary = out.get("executive_summary") or {}
        ttk.Label(
            pad,
            text=f"Campaign #{out.get('campaign_number')}: {out.get('campaign_name')}",
            font=("Segoe UI", 13, "bold"),
        ).pack(anchor="w")
        ttk.Label(pad, text=f"Status: {out.get('status')}", foreground=COL_MUTED).pack(anchor="w", pady=(2, 6))
        ttk.Label(pad, text=f"Question: {out.get('research_question')}", wraplength=700).pack(anchor="w", pady=(0, 10))

        concl_fr = ttk.LabelFrame(pad, text="Conclusion", padding=8)
        concl_fr.pack(fill="x", pady=(0, 8))
        ttk.Label(concl_fr, text=summary.get("conclusion") or "—", wraplength=680).pack(anchor="w")
        ttk.Label(
            concl_fr,
            text=summary.get("assessment") or "",
            foreground=COL_WARN,
            wraplength=680,
        ).pack(anchor="w", pady=(6, 0))
        ttk.Label(
            concl_fr,
            text=f"Recommendation: {summary.get('recommendation') or '—'}",
            foreground=COL_OK,
            wraplength=680,
        ).pack(anchor="w", pady=(6, 0))

        met_fr = ttk.LabelFrame(pad, text="Metrics", padding=8)
        met_fr.pack(fill="x", pady=(0, 8))
        bp = summary.get("baseline_pf")
        bbp = summary.get("best_pf")
        delta = summary.get("pf_delta")
        ttk.Label(
            met_fr,
            text=(
                f"Baseline PF: {bp if bp is not None else '—'}  ·  "
                f"Best PF: {bbp if bbp is not None else '—'}  ·  "
                f"Delta: {delta if delta is not None else '—'}  ·  "
                f"Best job: #{summary.get('best_job_number') or '—'}"
            ),
        ).pack(anchor="w")
        gen = summary.get("generalization") or {}
        if gen:
            ttk.Label(
                met_fr,
                text=f"Generalization: {gen.get('overall', '—')} {gen.get('label') or ''}",
            ).pack(anchor="w", pady=(4, 0))
        exp_done = summary.get("experiments_completed") or 0
        exp_lim = summary.get("experiments_budget")
        budget_txt = f"Experiments: {exp_done}"
        if exp_lim is not None:
            budget_txt += f" / {exp_lim}"
        if summary.get("budget_exhausted"):
            budget_txt += "  ·  BUDGET EXHAUSTED"
        ttk.Label(met_fr, text=budget_txt, foreground=COL_WARN if summary.get("budget_exhausted") else COL_MUTED).pack(
            anchor="w", pady=(4, 0),
        )
        verdicts = out.get("verdict_distribution") or {}
        if verdicts:
            vtxt = "  ·  ".join(f"{k}: {v}" for k, v in verdicts.items())
            ttk.Label(met_fr, text=f"Verdicts: {vtxt}", foreground=COL_MUTED).pack(anchor="w", pady=(4, 0))

        ch_fr = ttk.LabelFrame(pad, text="Changes Tested", padding=8)
        ch_fr.pack(fill="x", pady=(0, 8))
        changes = out.get("changes_tested") or []
        if not changes:
            ttk.Label(ch_fr, text="No completed experiments yet.", foreground=COL_MUTED).pack(anchor="w")
        for row in changes:
            ttk.Label(ch_fr, text=f"[{row.get('count')}x] {row.get('change')}", font=("Consolas", 9)).pack(anchor="w")

        kn_fr = ttk.LabelFrame(pad, text="Knowledge Gained", padding=8)
        kn_fr.pack(fill="x", pady=(0, 8))
        knowledge = out.get("knowledge_gained") or []
        if knowledge:
            for row in knowledge:
                ttk.Label(
                    kn_fr,
                    text=f"[{row.get('status')}] {row.get('finding')} ({row.get('confidence')})",
                    foreground=COL_OK,
                    wraplength=680,
                ).pack(anchor="w")
        else:
            findings = out.get("findings") or []
            if findings:
                for row in findings:
                    ttk.Label(
                        kn_fr,
                        text=f"[{row.get('status')}] {row.get('finding')} (n={row.get('evidence_count')})",
                        wraplength=680,
                    ).pack(anchor="w")
            else:
                ttk.Label(kn_fr, text="No knowledge promoted yet.", foreground=COL_MUTED).pack(anchor="w")

        gaps = out.get("knowledge_gaps") or []
        if gaps:
            gap_fr = ttk.LabelFrame(pad, text="Knowledge Gaps", padding=8)
            gap_fr.pack(fill="x", pady=(0, 8))
            for gap in gaps[:6]:
                note = gap.get("note") or gap.get("finding") or gap.get("topic") or "—"
                ttk.Label(gap_fr, text=f"  · {note}", foreground=COL_WARN, wraplength=680).pack(anchor="w")

        tl = out.get("timeline") or []
        if tl:
            tl_fr = ttk.LabelFrame(pad, text="Recent Experiments", padding=8)
            tl_fr.pack(fill="x", pady=(0, 8))
            for entry in reversed(tl[-8:]):
                txt = (
                    f"#{entry.get('job_number')}  {str(entry.get('change_text') or '')[:36]}  "
                    f"{entry.get('verdict') or '—'}  PF {entry.get('after_pf') or '—'}"
                )
                ttk.Label(tl_fr, text=txt, font=("Consolas", 9), foreground=COL_MUTED).pack(anchor="w")

        ttk.Button(pad, text="Close", command=win.destroy).pack(anchor="e", pady=(8, 0))

    def _load_program_portfolio(self, program_id: str) -> None:
        from chain_replay_ml.fold_research import get_program_portfolio

        port = get_program_portfolio(self._data_dir(), program_id)
        if not port.get("ok"):
            self._load_program_detail(program_id)
            return

        program = port.get("program") or {}
        stats = port.get("stats") or {}
        host = self._detail_host
        for w in host.winfo_children():
            w.destroy()

        ttk.Label(host, text=f"Program #{program.get('program_number')}", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(host, text=program.get("name") or "", font=("Segoe UI", 11)).pack(anchor="w", pady=(4, 2))
        ttk.Label(
            host,
            text=(
                f"Portfolio  ·  {stats.get('total_campaigns', 0)} campaigns  ·  "
                f"{stats.get('running_jobs', 0)} running  ·  "
                f"{stats.get('validated', 0)} validated  ·  "
                f"{stats.get('total_experiments', 0)} experiments"
            ),
            foreground=COL_MUTED,
        ).pack(anchor="w", pady=(2, 8))

        best_gen = port.get("best_generalization")
        if best_gen:
            ttk.Label(
                host,
                text=f"Best generalization: {best_gen.get('overall')} {best_gen.get('label', '')} — {best_gen.get('campaign_name', '')}",
                foreground=COL_OK,
            ).pack(anchor="w", pady=(0, 8))

        self._render_program_champion(host, program_id)

        cards_fr = ttk.LabelFrame(host, text="Campaign Portfolio", padding=6)
        cards_fr.pack(fill="both", expand=True)
        cards = port.get("campaigns") or []
        if not cards:
            ttk.Label(cards_fr, text="No campaigns yet — create one focused research question.", foreground=COL_MUTED).pack(anchor="w")
        for card in cards:
            self._render_campaign_card(cards_fr, card)

        self._render_objective_block(host, program.get("objective") or {}, "Program Objective")

    def _render_program_champion(self, host: tk.Misc, program_id: str) -> None:
        from chain_replay_ml.fold_research import get_program_champion_view

        view = get_program_champion_view(self._data_dir(), program_id)
        if not view.get("ok"):
            return

        fr = ttk.LabelFrame(host, text="Program Champion", padding=8)
        fr.pack(fill="x", pady=(0, 8))

        approved = view.get("approved")
        if approved:
            gen = approved.get("generalization") or {}
            ttk.Label(
                fr,
                text=(
                    f"✓ Approved  ·  {approved.get('campaign_name', '')}  ·  "
                    f"Gen {gen.get('overall', '—')}  ·  PF {approved.get('best_profit_factor', '—')}"
                ),
                foreground=COL_OK,
                font=("Segoe UI", 10, "bold"),
            ).pack(anchor="w")
            refs = approved.get("refs") or {}
            if refs.get("strategy_version_id"):
                ttk.Label(
                    fr,
                    text=f"Strategy version: {str(refs.get('strategy_version_id', ''))[:16]}…",
                    foreground=COL_MUTED,
                    font=("Consolas", 9),
                ).pack(anchor="w")
            return

        candidate = view.get("candidate")
        if not candidate:
            ttk.Label(fr, text="No champion candidate yet — validate a campaign with Gen ≥ 70.", foreground=COL_MUTED).pack(anchor="w")
            ttk.Button(
                fr,
                text="Refresh Candidate",
                command=lambda: self._refresh_champion_candidate(program_id),
            ).pack(anchor="w", pady=(4, 0))
            return

        gen = candidate.get("generalization") or {}
        ttk.Label(
            fr,
            text=(
                f"Candidate: {candidate.get('campaign_name', '')}  ·  "
                f"Score {candidate.get('composite_score', '—')}  ·  "
                f"Gen {gen.get('overall', '—')} {gen.get('label', '')}  ·  "
                f"Evidence {candidate.get('evidence_label', '—')}"
            ),
            foreground=COL_WARN,
            wraplength=620,
        ).pack(anchor="w")
        ttk.Label(
            fr,
            text=f"Recommendation: {candidate.get('recommendation', '—')}",
            foreground=COL_MUTED,
        ).pack(anchor="w", pady=(2, 4))
        btn_row = ttk.Frame(fr)
        btn_row.pack(fill="x")
        ttk.Button(
            btn_row,
            text="Approve Champion",
            command=lambda: self._approve_champion(program_id),
        ).pack(side="left")
        ttk.Button(
            btn_row,
            text="Dismiss",
            command=lambda: self._dismiss_champion(program_id),
        ).pack(side="left", padx=(8, 0))
        ttk.Button(
            btn_row,
            text="Refresh",
            command=lambda: self._refresh_champion_candidate(program_id),
        ).pack(side="left", padx=(8, 0))

    def _render_campaign_card(self, host: tk.Misc, card: dict[str, Any], *, compact: bool = False) -> None:
        status = card.get("status") or "—"
        color = COL_OK if status == "validated" else (COL_WARN if status == "running" else COL_MUTED)
        gen = card.get("generalization") or {}
        gen_txt = f"  Gen:{gen.get('overall')}" if gen.get("overall") is not None else ""
        run_txt = "  ▶ JOB" if card.get("job_running") else ""
        budget_txt = f"  {card.get('budget_pct')}%" if card.get("budget_pct") is not None else ""
        line = (
            f"#{card.get('campaign_number')}  {card.get('name', '')[:28]}  "
            f"{status}  {card.get('exploration_stage', '')}{gen_txt}{budget_txt}{run_txt}"
        )
        row = ttk.Frame(host)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=line, foreground=color, font=("Consolas", 9)).pack(side="left")
        if not compact:
            cid = str(card.get("campaign_id") or "")

            def _open(campaign_id: str = cid) -> None:
                self._selected_campaign_id = campaign_id
                for i, c in enumerate(self._campaign_index):
                    if str(c.get("campaign_id")) == campaign_id:
                        self._campaign_list.selection_clear(0, "end")
                        self._campaign_list.selection_set(i)
                        break
                self._load_campaign_detail(campaign_id)

            ttk.Button(row, text="Open", command=_open).pack(side="right")

    def _on_program_select(self, _event: tk.Event | None = None) -> None:
        sel = self._program_list.curselection()
        if not sel:
            return
        row = self._program_index[sel[0]]
        self._selected_program_id = str(row.get("program_id") or "")
        self._selected_campaign_id = None
        self._portfolio_mode = False
        self._campaign_list.selection_clear(0, "end")
        self._reload_campaigns()
        self._load_program_portfolio(self._selected_program_id)

    def _on_campaign_select(self, _event: tk.Event | None = None) -> None:
        sel = self._campaign_list.curselection()
        if not sel:
            return
        row = self._campaign_index[sel[0]]
        self._selected_campaign_id = str(row.get("campaign_id") or "")
        self._portfolio_mode = False
        self._coordinator.watch_campaign(
            self._selected_campaign_id,
            campaign_name=str(row.get("name") or ""),
        )
        self._load_campaign_detail(self._selected_campaign_id)

    def _render_empty(self) -> None:
        host = self._detail_host
        for w in host.winfo_children():
            w.destroy()
        ttk.Label(
            host,
            text="Create a Research Program to organize long-term strategy research.",
            foreground=COL_MUTED,
            wraplength=640,
        ).pack(anchor="w", pady=8)

    def _load_program_detail(self, program_id: str) -> None:
        from chain_replay_ml.fold_research import get_research_program

        doc = get_research_program(self._data_dir(), program_id)
        if not doc:
            self._render_empty()
            return
        host = self._detail_host
        for w in host.winfo_children():
            w.destroy()
        ttk.Label(host, text=f"Program #{doc.get('program_number')}", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(host, text=doc.get("name") or "", font=("Segoe UI", 11)).pack(anchor="w", pady=(4, 8))
        ttk.Label(host, text=f"Status: {doc.get('status')}  ·  Importance: {doc.get('importance')}", foreground=COL_MUTED).pack(anchor="w")
        if doc.get("description"):
            ttk.Label(host, text=doc.get("description"), wraplength=640, foreground=COL_MUTED).pack(anchor="w", pady=(4, 8))
        self._render_objective_block(host, doc.get("objective") or {}, "Program Objective")
        self._render_budget_block(host, doc.get("budget") or {}, "Program Budget Defaults")

    def _load_campaign_detail(self, campaign_id: str) -> None:
        from chain_replay_ml.fold_research import get_research_campaign

        doc = get_research_campaign(self._data_dir(), campaign_id)
        if not doc:
            return
        host = self._detail_host
        for w in host.winfo_children():
            w.destroy()

        ttk.Label(host, text=f"Campaign #{doc.get('campaign_number')}", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        ttk.Label(host, text=doc.get("name") or "", font=("Segoe UI", 11)).pack(anchor="w", pady=(4, 2))
        status = doc.get("status") or "created"
        color = COL_OK if status == "validated" else (COL_WARN if status == "running" else COL_MUTED)
        ttk.Label(host, text=f"Status: {status}", foreground=color, font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(2, 4))
        ttk.Label(host, text=f"Program: {doc.get('program_name') or '—'}", foreground=COL_MUTED).pack(anchor="w")
        ttk.Label(host, text=f"Question: {doc.get('research_question') or '—'}", wraplength=640).pack(anchor="w", pady=(6, 2))
        if doc.get("hypothesis"):
            ttk.Label(host, text=f"Hypothesis: {doc.get('hypothesis')}", wraplength=640, foreground=COL_MUTED).pack(anchor="w", pady=(0, 4))
        stopping = doc.get("resolved_stopping") or doc.get("stopping") or {}
        if stopping:
            ttk.Label(
                host,
                text=f"Stopping: min {stopping.get('min_jobs', 10)} · max {stopping.get('max_jobs', 50)} · auto-stop {'ON' if stopping.get('auto_stop', True) else 'OFF'}",
                foreground=COL_MUTED,
            ).pack(anchor="w", pady=(0, 8))

        cycle_fr = ttk.LabelFrame(host, text="Research Cycle", padding=8)
        cycle_fr.pack(fill="x", pady=(0, 8))
        ttk.Label(
            cycle_fr,
            text="Hypothesis → Experiment → Evidence → Decision",
            foreground=COL_MUTED,
        ).pack(anchor="w")
        self._cycle_step_var = tk.StringVar(value="")
        ttk.Label(cycle_fr, textvariable=self._cycle_step_var, foreground=COL_OK).pack(anchor="w", pady=(4, 0))

        self._scheduler_host = ttk.Frame(host)
        self._scheduler_host.pack(fill="both", expand=True, pady=(0, 8))

        self._render_objective_block(host, doc.get("resolved_objective") or {}, "Resolved Objective")
        self._render_budget_block(host, doc.get("resolved_budget") or {}, "Resolved Budget")

        btn_row = ttk.Frame(host)
        btn_row.pack(fill="x", pady=(12, 0))
        if status in ("created", "waiting"):
            ttk.Button(btn_row, text="Start Campaign", command=lambda: self._start_campaign(campaign_id)).pack(side="left")
        if status in ("running", "paused", "waiting"):
            ttk.Button(btn_row, text="Attach Baseline Report", command=lambda: self._attach_baseline(campaign_id)).pack(side="left")
            ttk.Button(btn_row, text="Seed Proposals", command=lambda: self._seed_proposals(campaign_id)).pack(side="left", padx=(8, 0))
            ttk.Button(btn_row, text="Seed KB Proposals", command=lambda: self._seed_kb_proposals(campaign_id)).pack(side="left", padx=(4, 0))
            ttk.Button(btn_row, text="Run Next Experiment", command=lambda: self._run_next()).pack(side="left", padx=(8, 0))
            ttk.Button(
                btn_row,
                text="Evaluate Generalization",
                command=lambda: self._evaluate_generalization(campaign_id),
            ).pack(side="left", padx=(8, 0))
            ttk.Checkbutton(
                btn_row,
                text="Auto-run",
                variable=self._auto_run_var,
                command=lambda: self._toggle_auto_run(campaign_id),
            ).pack(side="left", padx=(12, 0))
        if status == "running" and (doc.get("memory") or {}).get("validation_ready"):
            ttk.Button(
                btn_row,
                text="Mark Validated",
                command=lambda: self._mark_validated(campaign_id),
            ).pack(side="left", padx=(8, 0))
        if status in ("created", "running", "validated"):
            ttk.Button(
                btn_row,
                text="Retire Campaign",
                command=lambda: self._retire_campaign(campaign_id),
            ).pack(side="left", padx=(8, 0))
        if status == "validated":
            ttk.Button(
                btn_row,
                text="View Campaign Report",
                command=lambda: self._view_campaign_report(campaign_id),
            ).pack(side="left", padx=(8, 0))

        if doc.get("retired_reason"):
            ttk.Label(host, text=f"Retired: {doc.get('retired_reason')}", foreground=COL_WARN, wraplength=640).pack(anchor="w", pady=(8, 0))

        self._render_scheduler(campaign_id)
        self._render_knowledge_panel(campaign_id, program_id=str(doc.get("program_id") or ""))
        if status == "validated":
            self._render_campaign_report_summary(campaign_id)

    def _render_campaign_report_summary(self, campaign_id: str) -> None:
        from chain_replay_ml.fold_research import get_campaign_report

        out = get_campaign_report(self._data_dir(), campaign_id)
        if not out.get("ok"):
            return
        report = out.get("report") or {}
        summary = report.get("executive_summary") or {}
        host = self._detail_host
        fr = ttk.LabelFrame(host, text="Campaign Report", padding=8)
        fr.pack(fill="x", pady=(8, 0))
        ttk.Label(fr, text=summary.get("conclusion") or "—", wraplength=640).pack(anchor="w")
        ttk.Label(
            fr,
            text=f"Recommendation: {summary.get('recommendation') or '—'}",
            foreground=COL_MUTED,
        ).pack(anchor="w", pady=(4, 0))

    def _render_scheduler(self, campaign_id: str) -> None:
        from chain_replay_ml.fold_research import get_campaign_dashboard

        host = self._scheduler_host
        for w in host.winfo_children():
            w.destroy()

        dash = get_campaign_dashboard(self._data_dir(), campaign_id)
        if not dash.get("ok"):
            ttk.Label(host, text=dash.get("error") or "Dashboard unavailable", foreground=COL_WARN).pack(anchor="w")
            return

        view = dash.get("scheduler") or {}
        memory = view.get("memory") or {}
        cycle = view.get("cycle") or {}
        step = cycle.get("current_step") or memory.get("last_cycle_step") or "hypothesis"
        stage = cycle.get("exploration_stage") or memory.get("exploration_stage") or "explore"
        self._cycle_step_var.set(f"Step: {step.title()}  ·  Stage: {stage.title()}")
        self._auto_run_var.set(bool(memory.get("auto_run")))

        funnel = dash.get("funnel") or {}
        ttk.Label(
            host,
            text=(
                f"Queued: {funnel.get('proposals_queued', 0)}  ·  "
                f"Completed: {funnel.get('experiments_completed', 0)}  ·  "
                f"Tested: {funnel.get('hypotheses_tested', 0)}"
            ),
            foreground=COL_MUTED,
        ).pack(anchor="w", pady=(4, 0))

        manifest = view.get("manifest") or {}
        if manifest:
            man_fr = ttk.LabelFrame(host, text="Campaign Manifest", padding=6)
            man_fr.pack(fill="x", pady=(8, 0))
            winner = manifest.get("winner") or {}
            win_txt = winner.get("change_text") or "—"
            conf = manifest.get("confidence_pct")
            ttk.Label(
                man_fr,
                text=(
                    f"Jobs: {manifest.get('completed_jobs', 0)} completed  ·  "
                    f"Evidence: {manifest.get('evidence_count', 0)}  ·  "
                    f"Knowledge: {manifest.get('knowledge_created', 0)}"
                ),
                font=("Segoe UI", 10, "bold"),
            ).pack(anchor="w")
            ttk.Label(
                man_fr,
                text=f"Winner: {win_txt}  ·  Confidence: {conf or '—'}%",
                foreground=COL_OK if conf and int(conf) >= 85 else COL_MUTED,
            ).pack(anchor="w", pady=(2, 0))
            if manifest.get("resume_required"):
                ttk.Label(
                    man_fr,
                    text=f"Resume job #{manifest.get('current_job_number') or '?'} — checkpoint {manifest.get('last_checkpoint_at') or '—'}",
                    foreground=COL_WARN,
                ).pack(anchor="w", pady=(2, 0))
            stop = (view.get("budget") or {}).get("stop_decision") or {}
            if stop.get("label"):
                ttk.Label(man_fr, text=f"Stop policy: {stop.get('label')}", foreground=COL_MUTED).pack(anchor="w", pady=(2, 0))

        trend = dash.get("trend_summary")
        if trend:
            ttk.Label(
                host,
                text=(
                    f"PF trend: {trend.get('first_pf')} → {trend.get('latest_pf')}  "
                    f"(best {trend.get('best_pf')})"
                ),
                foreground=COL_OK if (trend.get("delta") or 0) >= 0 else COL_WARN,
            ).pack(anchor="w", pady=(2, 0))

        gen = memory.get("best_generalization") or cycle.get("best_generalization")
        if gen:
            gscore = int(gen.get("overall") or 0)
            glabel = gen.get("label") or "—"
            gcolor = COL_OK if gscore >= 70 else (COL_WARN if gscore >= 50 else COL_MUTED)
            ttk.Label(
                host,
                text=f"Generalization: {gscore}  {glabel}",
                foreground=gcolor,
                font=("Segoe UI", 10, "bold"),
            ).pack(anchor="w", pady=(4, 0))
        if cycle.get("validation_ready"):
            ttk.Label(host, text="Validation ready — generalization passed", foreground=COL_OK).pack(anchor="w")

        deps = dash.get("dependencies") or []
        if deps:
            dep_fr = ttk.LabelFrame(host, text="Dependencies", padding=4)
            dep_fr.pack(fill="x", pady=(8, 0))
            for dep in deps:
                mark = "✓" if dep.get("satisfied") else "✗"
                ttk.Label(
                    dep_fr,
                    text=f"{mark} #{dep.get('campaign_number')} {dep.get('name', '')[:24]}  {dep.get('status')}",
                    foreground=COL_OK if dep.get("satisfied") else COL_WARN,
                    font=("Consolas", 9),
                ).pack(anchor="w")

        timeline = dash.get("timeline") or []
        if timeline:
            tl_fr = ttk.LabelFrame(host, text="Experiment Timeline", padding=6)
            tl_fr.pack(fill="x", pady=(8, 0))
            for entry in reversed(timeline[-8:]):
                txt = (
                    f"#{entry.get('job_number')}  {str(entry.get('change_text') or '')[:28]}  "
                    f"{entry.get('verdict') or '—'}  PF {entry.get('after_pf') or '—'}  "
                    f"Δ{entry.get('pf_delta') or '—'}"
                )
                ttk.Label(tl_fr, text=txt, foreground=COL_MUTED, font=("Consolas", 9)).pack(anchor="w")

        log = cycle.get("hypothesis_log") or memory.get("hypothesis_log") or []
        if log:
            log_fr = ttk.LabelFrame(host, text="Hypothesis Log", padding=6)
            log_fr.pack(fill="x", pady=(8, 0))
            for entry in reversed(log[-5:]):
                txt = (
                    f"#{entry.get('job_number') or '?'}  {entry.get('change_text', '')[:32]}  "
                    f"{entry.get('verdict') or '—'}  PFΔ {entry.get('pf_delta') or '—'}"
                )
                ttk.Label(log_fr, text=txt, foreground=COL_MUTED, font=("Consolas", 9)).pack(anchor="w")

        burn = dash.get("budget_burn") or {}
        btxt = f"Experiments: {burn.get('experiments_used', 0)}"
        if burn.get("experiments_limit") is not None:
            btxt += f" / {burn.get('experiments_limit')}"
        if burn.get("experiments_pct") is not None:
            btxt += f"  ({burn.get('experiments_pct')}%)"
        if burn.get("exhausted"):
            btxt += "  ·  BUDGET EXHAUSTED"
        ttk.Label(host, text=btxt, foreground=COL_WARN if burn.get("exhausted") else COL_MUTED).pack(anchor="w", pady=(4, 0))

        running = view.get("running_job")
        if self._job_runner.running:
            jid = self._job_runner.job_id or (running or {}).get("job_number") or "…"
            ttk.Label(host, text=f"Job running: #{jid}", foreground=COL_OK).pack(anchor="w", pady=(4, 0))
        elif running:
            jid = running.get("job_number") or "…"
            row = ttk.Frame(host)
            row.pack(fill="x", pady=(4, 0))
            ttk.Label(row, text=f"Job stalled: #{jid} (queued, not executing)", foreground=COL_WARN).pack(side="left")
            ttk.Button(
                row,
                text="Resume Job",
                command=lambda: self._resume_or_clear_stale_job(str(running.get("job_id") or ""), campaign_id),
            ).pack(side="left", padx=(8, 0))

        queue_fr = ttk.LabelFrame(host, text="Proposal Queue (Objective Score)", padding=6)
        queue_fr.pack(fill="both", expand=True, pady=(8, 0))
        proposals = view.get("proposal_queue") or []
        if not proposals:
            ttk.Label(queue_fr, text="No draft proposals — attach baseline and seed.", foreground=COL_MUTED).pack(anchor="w")
            return
        for i, prop in enumerate(proposals[:8]):
            obj = prop.get("objective_score") or {}
            score = int(obj.get("overall") or 0)
            goal = str(prop.get("goal") or "")[:56]
            sel = prop.get("selected_recommendations") or []
            change = str((sel[0] or {}).get("text") or "")[:40] if sel else "—"
            prefix = "▶ " if i == 0 else "  "
            ttk.Label(
                queue_fr,
                text=f"{prefix}{score:3d}  {change}  —  {goal}",
                foreground=COL_OK if i == 0 else COL_MUTED,
                font=("Consolas", 9),
            ).pack(anchor="w")

    def _render_knowledge_panel(self, campaign_id: str, *, program_id: str = "") -> None:
        from chain_replay_ml.fold_research import get_knowledge_gaps, get_knowledge_pipeline_view

        host = self._detail_host
        view = get_knowledge_pipeline_view(
            self._data_dir(),
            program_id=program_id or None,
            campaign_id=campaign_id,
            limit=20,
        )
        if not view.get("ok"):
            return

        totals = view.get("totals") or {}
        fr = ttk.LabelFrame(host, text="Knowledge Pipeline (Evidence → Finding → Knowledge)", padding=8)
        fr.pack(fill="x", pady=(8, 0))
        ttk.Label(
            fr,
            text=(
                f"Evidence: {totals.get('evidence_linked', 0)}  ·  "
                f"Findings: {totals.get('finding', 0)}  ·  "
                f"Knowledge: {totals.get('knowledge', 0)}"
            ),
            foreground=COL_MUTED,
        ).pack(anchor="w")

        gaps = get_knowledge_gaps(self._data_dir(), campaign_id)
        gap_list = (gaps.get("knowledge_gaps") or [])[:4] if gaps.get("ok") else []
        if gap_list:
            gap_fr = ttk.Frame(fr)
            gap_fr.pack(fill="x", pady=(6, 0))
            ttk.Label(gap_fr, text="Knowledge gaps:", foreground=COL_WARN).pack(anchor="w")
            for gap in gap_list:
                note = gap.get("note") or gap.get("finding") or gap.get("topic") or "—"
                ttk.Label(gap_fr, text=f"  · {note}", foreground=COL_MUTED, wraplength=600).pack(anchor="w")

        items = view.get("findings") or []
        if not items:
            ttk.Label(fr, text="No findings yet — run experiments to collect evidence.", foreground=COL_MUTED).pack(anchor="w", pady=(6, 0))
            return

        list_fr = ttk.Frame(fr)
        list_fr.pack(fill="x", pady=(6, 0))
        for item in items[:6]:
            stage = item.get("lifecycle_stage") or "evidence_linked"
            status = item.get("status") or "candidate"
            conf = item.get("confidence") or "—"
            ev = item.get("evidence_count") or 0
            text = str(item.get("finding") or "")[:52]
            color = COL_OK if status == "knowledge" else (COL_WARN if stage == "finding" else COL_MUTED)
            row = ttk.Frame(list_fr)
            row.pack(fill="x", pady=1)
            ttk.Label(
                row,
                text=f"[{stage}] {text}  ({conf}, n={ev})",
                foreground=color,
                font=("Consolas", 9),
            ).pack(side="left", anchor="w")
            if status in ("confirmed", "supported") and stage != "knowledge":
                fid = str(item.get("finding_id") or "")
                ttk.Button(
                    row,
                    text="→ Knowledge",
                    width=12,
                    command=lambda f=fid: self._promote_finding(f),
                ).pack(side="right")

    def _promote_finding(self, finding_id: str) -> None:
        from chain_replay_ml.fold_research import promote_finding_to_knowledge

        out = promote_finding_to_knowledge(self._data_dir(), finding_id)
        if not out.get("ok"):
            messagebox.showerror("Knowledge", out.get("error") or "Promotion failed")
            return
        if self._selected_campaign_id:
            self._load_campaign_detail(self._selected_campaign_id)
        elif self._portfolio_mode:
            self._show_desk_portfolio()

    def _seed_kb_proposals(self, campaign_id: str) -> None:
        from chain_replay_ml.fold_research import seed_kb_campaign_proposals

        out = seed_kb_campaign_proposals(self._data_dir(), campaign_id)
        if not out.get("ok"):
            messagebox.showerror("KB Proposals", out.get("error") or "Failed")
            return
        count = int(out.get("count") or 0)
        messagebox.showinfo("KB Proposals", f"Created {count} KB-driven proposal(s).")
        self._load_campaign_detail(campaign_id)

    def _render_objective_block(self, host: tk.Misc, objective: dict[str, Any], title: str) -> None:
        fr = ttk.LabelFrame(host, text=title, padding=8)
        fr.pack(fill="x", pady=(8, 0))
        primary = objective.get("primary_goal") or {}
        ttk.Label(
            fr,
            text=f"Primary: {primary.get('direction', '—')} {primary.get('metric', '—')}",
        ).pack(anchor="w")
        for c in objective.get("constraints") or []:
            ttk.Label(
                fr,
                text=f"Constraint: {c.get('metric')} {c.get('op')} {c.get('value')}",
                foreground=COL_MUTED,
            ).pack(anchor="w")
        ttk.Label(fr, text=f"Importance: {objective.get('importance') or '—'}", foreground=COL_MUTED).pack(anchor="w", pady=(4, 0))

    def _render_budget_block(self, host: tk.Misc, budget: dict[str, Any], title: str) -> None:
        fr = ttk.LabelFrame(host, text=title, padding=8)
        fr.pack(fill="x", pady=(8, 0))
        for key in (
            "max_experiments", "max_gpu_hours", "max_cpu_hours",
            "max_wall_clock_hours", "max_storage_gb", "max_cost",
        ):
            val = budget.get(key)
            if val is not None:
                ttk.Label(fr, text=f"{key}: {val}", foreground=COL_MUTED).pack(anchor="w")

    def _new_program(self) -> None:
        from chain_replay_ml.fold_research import create_research_program

        name = simpledialog.askstring("Research Program", "Program name:", parent=self)
        if not name:
            return
        out = create_research_program(self._data_dir(), name=name.strip())
        if not out.get("ok"):
            messagebox.showerror("Research Program", out.get("error") or "Failed")
            return
        program = out.get("program") or {}
        self._selected_program_id = str(program.get("program_id") or "")
        self.refresh()

    def _new_campaign(self) -> None:
        from chain_replay_ml.fold_research import create_research_campaign

        if not self._selected_program_id:
            messagebox.showinfo("Research Program", "Select a program first.")
            return

        picked = self._pick_campaign_template()
        if not picked:
            return

        out = create_research_campaign(
            self._data_dir(),
            self._selected_program_id,
            name=picked["name"].strip(),
            research_question=picked["research_question"].strip(),
            hypothesis=picked.get("hypothesis"),
            success_criteria=picked.get("success_criteria"),
            failure_criteria=picked.get("failure_criteria"),
            stopping=picked.get("stopping"),
        )
        if not out.get("ok"):
            messagebox.showerror("Research Campaign", out.get("error") or "Failed")
            return
        campaign = out.get("campaign") or {}
        self._selected_campaign_id = str(campaign.get("campaign_id") or "")
        self.refresh()

    def _pick_campaign_template(self) -> dict[str, Any] | None:
        from chain_replay_ml.fold_research.campaign_templates import list_campaign_templates

        templates = list_campaign_templates()
        win = tk.Toplevel(self.winfo_toplevel())
        win.title("New Research Campaign")
        win.transient(self.winfo_toplevel())
        win.grab_set()
        win.geometry("720x380")

        body = ttk.Frame(win, padding=12)
        body.pack(fill="both", expand=True)

        ttk.Label(body, text="Campaign template:", font=("Segoe UI", 10, "bold")).pack(anchor="w")

        labels = [f"{t.get('name')} — {t.get('research_question')}" for t in templates]
        combo_var = tk.StringVar(value=labels[0] if labels else "")
        combo = ttk.Combobox(body, textvariable=combo_var, values=labels, state="readonly", width=96)
        combo.pack(fill="x", pady=(6, 10))

        desc_var = tk.StringVar(value=str(templates[0].get("description") or "") if templates else "")
        ttk.Label(body, textvariable=desc_var, foreground=COL_MUTED, wraplength=680).pack(anchor="w", pady=(0, 10))

        name_var = tk.StringVar(value=str(templates[0].get("name") or "") if templates else "")
        q_var = tk.StringVar(value=str(templates[0].get("research_question") or "") if templates else "")

        name_row = ttk.Frame(body)
        name_row.pack(fill="x", pady=(0, 6))
        ttk.Label(name_row, text="Campaign name:", width=16).pack(side="left")
        name_entry = ttk.Entry(name_row, textvariable=name_var, width=72)
        name_entry.pack(side="left", fill="x", expand=True)

        q_row = ttk.Frame(body)
        q_row.pack(fill="x", pady=(0, 6))
        ttk.Label(q_row, text="Research question:", width=16).pack(side="left", anchor="n", pady=(4, 0))
        q_entry = tk.Text(q_row, height=3, width=72, font=("Segoe UI", 10), wrap="word")
        q_entry.pack(side="left", fill="x", expand=True)
        if templates:
            q_entry.insert("1.0", str(templates[0].get("research_question") or ""))

        def _apply_template(_event: object = None) -> None:
            try:
                idx = labels.index(combo_var.get())
            except ValueError:
                return
            tmpl = templates[idx]
            desc_var.set(str(tmpl.get("description") or ""))
            is_custom = str(tmpl.get("id") or "") == "custom"
            name_var.set("" if is_custom else str(tmpl.get("name") or ""))
            q_entry.delete("1.0", "end")
            q_entry.insert("1.0", str(tmpl.get("research_question") or ""))
            name_entry.configure(state="normal")
            q_entry.configure(state="normal")

        combo.bind("<<ComboboxSelected>>", _apply_template)

        choice: dict[str, dict[str, Any] | None] = {"value": None}

        def _accept() -> None:
            name = name_var.get().strip()
            question = q_entry.get("1.0", "end").strip()
            if not name or not question:
                messagebox.showinfo("Research Campaign", "Enter campaign name and research question.", parent=win)
                return
            try:
                idx = labels.index(combo_var.get())
            except ValueError:
                idx = 0
            tmpl = templates[idx] if templates else {}
            choice["value"] = {
                "name": name,
                "research_question": question,
                "hypothesis": tmpl.get("hypothesis"),
                "success_criteria": tmpl.get("success_criteria"),
                "failure_criteria": tmpl.get("failure_criteria"),
                "stopping": tmpl.get("stopping"),
            }
            win.destroy()

        btn_row = ttk.Frame(body)
        btn_row.pack(fill="x", pady=(12, 0))
        ttk.Button(btn_row, text="Create Campaign", command=_accept).pack(side="right")
        ttk.Button(btn_row, text="Cancel", command=win.destroy).pack(side="right", padx=(0, 8))

        win.wait_window()
        return choice["value"]

    def _start_campaign(self, campaign_id: str) -> None:
        from chain_replay_ml.fold_research import start_research_campaign

        if not messagebox.askyesno(
            "Start Campaign",
            "Start campaign scheduler?\n\n"
            "Attach a baseline research report, then seed proposals or enable auto-run.",
        ):
            return
        out = start_research_campaign(self._data_dir(), campaign_id)
        if not out.get("ok"):
            messagebox.showerror("Research Campaign", out.get("error") or "Failed")
            return
        self.refresh()
        self._load_campaign_detail(campaign_id)

    def _attach_baseline(self, campaign_id: str) -> None:
        from chain_replay_ml.fold_research import attach_campaign_baseline, list_saved_research_reports

        reports = list_saved_research_reports(self._data_dir(), limit=200)
        if not reports:
            messagebox.showinfo("Baseline", "No saved research reports found. Run a simulation first.")
            return

        selected_id = self._pick_research_report(reports)
        if not selected_id:
            return
        out = attach_campaign_baseline(self._data_dir(), campaign_id, research_report_id=selected_id)
        if not out.get("ok"):
            messagebox.showerror("Baseline", out.get("error") or "Failed")
            return
        self._load_campaign_detail(campaign_id)

    def _pick_research_report(self, reports: list[dict[str, Any]]) -> str | None:
        win = tk.Toplevel(self.winfo_toplevel())
        win.title("Attach Baseline")
        win.transient(self.winfo_toplevel())
        win.grab_set()
        win.geometry("920x220")

        body = ttk.Frame(win, padding=12)
        body.pack(fill="both", expand=True)

        ttk.Label(
            body,
            text=f"Select baseline research report ({len(reports)} saved):",
        ).pack(anchor="w", pady=(0, 8))

        options: list[str] = []
        report_ids: list[str] = []
        for row in reports:
            rid = str(row.get("report_id") or "")
            if not rid:
                continue
            created = str(row.get("created_at") or "")[:10]
            options.append(
                f"{rid}  |  grade={row.get('grade') or '—'}  |  trades={row.get('trade_count') or '—'}  |  {created}"
            )
            report_ids.append(rid)

        if not options:
            messagebox.showinfo("Attach Baseline", "No valid reports found.", parent=win)
            win.destroy()
            return None

        combo_var = tk.StringVar(value=options[0])
        combo = ttk.Combobox(
            body,
            textvariable=combo_var,
            values=options,
            state="readonly",
            width=110,
            font=("Consolas", 9),
        )
        combo.pack(fill="x", pady=(0, 8))

        id_var = tk.StringVar(value=report_ids[0])
        id_row = ttk.Frame(body)
        id_row.pack(fill="x", pady=(0, 8))
        ttk.Label(id_row, text="Report ID:").pack(side="left")
        ttk.Entry(id_row, textvariable=id_var, state="readonly", width=96, font=("Consolas", 9)).pack(
            side="left", padx=(8, 0), fill="x", expand=True,
        )

        def _on_select(_event: object = None) -> None:
            try:
                idx = options.index(combo_var.get())
            except ValueError:
                return
            id_var.set(report_ids[idx])

        combo.bind("<<ComboboxSelected>>", _on_select)

        choice: dict[str, str | None] = {"report_id": None}

        def _accept() -> None:
            try:
                idx = options.index(combo_var.get())
            except ValueError:
                messagebox.showinfo("Attach Baseline", "Select a report from the dropdown.", parent=win)
                return
            choice["report_id"] = report_ids[idx]
            win.destroy()

        btn_row = ttk.Frame(body)
        btn_row.pack(fill="x", pady=(8, 0))
        ttk.Button(btn_row, text="OK", command=_accept).pack(side="right")
        ttk.Button(btn_row, text="Cancel", command=win.destroy).pack(side="right", padx=(0, 8))

        win.wait_window()
        return choice["report_id"]

    def _seed_proposals(self, campaign_id: str) -> None:
        from chain_replay_ml.fold_research import rank_campaign_proposals, seed_campaign_proposals

        out = seed_campaign_proposals(self._data_dir(), campaign_id)
        if not out.get("ok"):
            messagebox.showerror("Proposals", out.get("error") or "Failed")
            return
        rank_campaign_proposals(self._data_dir(), campaign_id)
        self._status_var.set(f"Seeded {out.get('count', 0)} proposals")
        self._load_campaign_detail(campaign_id)

    def _toggle_auto_run(self, campaign_id: str) -> None:
        from chain_replay_ml.fold_research import set_campaign_auto_run

        set_campaign_auto_run(self._data_dir(), campaign_id, enabled=self._auto_run_var.get())

    def _clear_job_pointers(self, campaign_id: str) -> None:
        from chain_replay_ml.fold_research.research_program import get_research_campaign, update_research_campaign

        camp = get_research_campaign(self._data_dir(), campaign_id)
        if not camp:
            return
        memory = dict(camp.get("memory") or {})
        changed = False
        for key in ("pending_job_id", "active_job_id"):
            if key in memory:
                memory.pop(key, None)
                changed = True
        if changed:
            update_research_campaign(self._data_dir(), campaign_id, memory=memory)

    def _resume_or_clear_stale_job(self, job_id: str, campaign_id: str) -> None:
        if not job_id or self._coordinator.running:
            return
        from chain_replay_ml.fold_research.experiment_pipeline_store import ExperimentPipelineStore

        with ExperimentPipelineStore(self._data_dir()) as store:
            job = store._load_job(job_id)
        if not job:
            self._clear_job_pointers(campaign_id)
            self._load_campaign_detail(campaign_id)
            return
        if job.get("status") != "running":
            self._clear_job_pointers(campaign_id)
            self._load_campaign_detail(campaign_id)
            return
        self._coordinator.execute_job(self._data_dir(), job_id, campaign_id)

    def _on_campaign_job_done(self, campaign_id: str, result: dict) -> None:
        """Handled by ResearchCampaignCoordinator — kept for compatibility."""
        _ = (campaign_id, result)

    def _run_next(self, *, auto: bool = False) -> None:
        if self._coordinator.running:
            return
        cid = self._selected_campaign_id
        if not cid:
            return
        out = self._coordinator.kick_campaign(self._data_dir(), cid)
        if not out.get("ok") and not auto:
            messagebox.showerror("Run Next", out.get("error") or "Failed")
        elif out.get("ok"):
            self._load_campaign_detail(cid)

    def _execute_job(self, job_id: str, campaign_id: str) -> None:
        if self._coordinator.running:
            return
        out = self._coordinator.execute_job(self._data_dir(), job_id, campaign_id)
        if out.get("ok"):
            self._load_campaign_detail(campaign_id)

    def _evaluate_generalization(self, campaign_id: str) -> None:
        from chain_replay_ml.fold_research import evaluate_campaign_generalization

        out = evaluate_campaign_generalization(self._data_dir(), campaign_id)
        if not out.get("ok"):
            messagebox.showerror("Generalization", out.get("error") or "Failed")
            return
        gen = out.get("generalization") or {}
        messagebox.showinfo(
            "Generalization Score",
            f"Score: {gen.get('overall')}  {gen.get('label')}\n"
            f"Promote recommended: {'Yes' if gen.get('promote_recommended') else 'No'}",
        )
        self._load_campaign_detail(campaign_id)

    def _mark_validated(self, campaign_id: str) -> None:
        from chain_replay_ml.fold_research import mark_campaign_validated

        if not messagebox.askyesno(
            "Mark Validated",
            "Confirm campaign passed generalization and is ready to leave active research?",
        ):
            return
        out = mark_campaign_validated(self._data_dir(), campaign_id)
        if not out.get("ok"):
            messagebox.showerror("Validate Campaign", out.get("error") or "Failed")
            return
        report = out.get("report") or {}
        summary = report.get("executive_summary") or {}
        champ = out.get("champion_candidate")
        msg = summary.get("conclusion") or "Campaign validated."
        if champ:
            msg += f"\n\nChampion candidate updated: {champ.get('campaign_name', '')}"
        messagebox.showinfo("Campaign Validated", msg)
        self.refresh()
        self._load_campaign_detail(campaign_id)

    def _view_campaign_report(self, campaign_id: str) -> None:
        from chain_replay_ml.fold_research import get_campaign_report

        out = get_campaign_report(self._data_dir(), campaign_id)
        if not out.get("ok"):
            messagebox.showerror("Campaign Report", out.get("error") or "No report")
            return
        report = out.get("report") or {}
        summary = report.get("executive_summary") or {}
        verdicts = report.get("verdict_distribution") or {}
        vtxt = ", ".join(f"{k}: {v}" for k, v in verdicts.items()) or "—"
        messagebox.showinfo(
            "Campaign Report",
            f"{summary.get('research_question', '')}\n\n"
            f"{summary.get('conclusion', '')}\n\n"
            f"Experiments: {summary.get('experiments_run', 0)}  ·  "
            f"Best PF: {summary.get('best_pf', '—')}  ·  "
            f"Gen: {(summary.get('generalization') or {}).get('overall', '—')}\n\n"
            f"Verdicts: {vtxt}\n\n"
            f"{summary.get('recommendation', '')}",
        )

    def _approve_champion(self, program_id: str) -> None:
        from chain_replay_ml.fold_research import approve_program_champion, get_program_champion_view

        view = get_program_champion_view(self._data_dir(), program_id)
        cand = view.get("candidate") or {}
        gen = (cand.get("generalization") or {}).get("overall", "—")
        if not messagebox.askyesno(
            "Approve Program Champion",
            f"Approve «{cand.get('campaign_name', '')}» as Program Champion?\n\n"
            f"Generalization: {gen}\n"
            f"This does NOT auto-deploy to production — it records research approval only.",
        ):
            return
        out = approve_program_champion(self._data_dir(), program_id)
        if not out.get("ok"):
            messagebox.showerror("Program Champion", out.get("error") or "Failed")
            return
        messagebox.showinfo("Program Champion", "Program champion approved.")
        self._load_program_portfolio(program_id)

    def _dismiss_champion(self, program_id: str) -> None:
        from chain_replay_ml.fold_research import dismiss_program_champion_candidate

        dismiss_program_champion_candidate(self._data_dir(), program_id)
        self._load_program_portfolio(program_id)

    def _refresh_champion_candidate(self, program_id: str) -> None:
        from chain_replay_ml.fold_research import refresh_program_champion_candidate

        refresh_program_champion_candidate(self._data_dir(), program_id)
        self._load_program_portfolio(program_id)

    def _retire_campaign(self, campaign_id: str) -> None:
        from chain_replay_ml.fold_research import retire_research_campaign

        reason = simpledialog.askstring("Retire Campaign", "Reason:", parent=self)
        if not reason:
            return
        out = retire_research_campaign(self._data_dir(), campaign_id, reason=reason.strip())
        if not out.get("ok"):
            messagebox.showerror("Research Campaign", out.get("error") or "Failed")
            return
        self.refresh()
        self._load_campaign_detail(campaign_id)
