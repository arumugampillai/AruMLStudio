"""Experiment Planner — Feature Studio tab (display Recommendation Engine artifacts).

Distinct from Strategy Lab ``experiment_planner_panel.py`` (fold research).
Advisory only: never auto-creates training jobs or edits feature selections.
"""

from __future__ import annotations

import csv
import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .build_service import chart_data_dir
from .lazy_panel import LazyLoadMixin
from .model_registry_widgets import (
    COL_MUTED,
    SECTION_FONT,
    fmt_num,
    fmt_val,
)


_TABLE_COLS = (
    ("experiment", "Experiment", 260),
    ("estimated_effort", "Effort", 70),
    ("status", "Status", 100),
    ("priority", "Priority", 70),
    ("category", "Category", 120),
    ("family", "Family", 80),
    ("evidence_score", "Evidence Score", 100),
    ("hypothesis", "Hypothesis", 200),
    ("affected_features", "Features", 90),
)

_STATUS_CHOICES = (
    "Not Started",
    "In Progress",
    "Completed",
    "Rejected",
    "Superseded",
)

_BENEFIT_LABELS = {
    "high": "High",
    "medium": "Medium",
    "low": "Low",
    "slightly_improved": "Slightly improved",
    "unknown": "Unknown",
    "none": "None",
}

_FEATURE_DETAIL_LABELS = (
    ("risk_score", "Risk"),
    ("rank_gain", "Importance Rank"),
    ("ks_statistic", "KS"),
    ("drift", "Drift"),
    ("drift_pct", "Drift %"),
    ("wasserstein_normalized", "Wasserstein"),
    ("null_drift_pp", "Null drift pp"),
    ("null_pct", "Null %"),
    ("gain", "Gain"),
    ("risk", "Risk label"),
)

_SECTION_MODEL = "Model Experiments"
_SECTION_FEATURE = "Feature Experiments"


class ExperimentPlannerStudioPanel(ttk.Frame, LazyLoadMixin):
    """Advisory suggestions from Recommendation Engine artifacts."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str,
        model_var: tk.StringVar | None = None,
        filter_var: tk.StringVar | None = None,
        top_n_var: tk.StringVar | None = None,
        top_n_only: tk.BooleanVar | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._model_names: list[str] = []
        self._rows: list[dict[str, Any]] = []
        self._display_rows: list[dict[str, Any]] = []
        self._summary: dict[str, Any] = {}
        self._meta: dict[str, Any] = {}
        self._experiment_state: dict[str, Any] = {"experiments": {}}
        self._package_dir: str | None = None
        self._sort_col = "priority"
        self._sort_desc = False
        self._iid_to_row: dict[str, dict[str, Any]] = {}

        self._status_var = tk.StringVar(
            value="Select a model and Load or Compute. Advisory only — does not run experiments."
        )
        self._model_var = model_var if model_var is not None else tk.StringVar()
        self._filter_var = filter_var if filter_var is not None else tk.StringVar()
        self._top_n_var = top_n_var if top_n_var is not None else tk.StringVar(value="20")
        self._top_n_only = (
            top_n_only if top_n_only is not None else tk.BooleanVar(value=False)
        )

        self._summary_vars = {
            "total": tk.StringVar(value="—"),
            "high": tk.StringVar(value="—"),
            "medium": tk.StringVar(value="—"),
            "low": tk.StringVar(value="—"),
            "risk_feat": tk.StringVar(value="—"),
            "top_sug": tk.StringVar(value="—"),
            "compute": tk.StringVar(value="—"),
            "inputs": tk.StringVar(value="—"),
        }

        self._build_ui()
        self._filter_var.trace_add("write", lambda *_: self._apply_filter_sort())
        self._top_n_only.trace_add("write", lambda *_: self._apply_filter_sort())
        self._top_n_var.trace_add("write", lambda *_: self._apply_filter_sort())
        self._lazy_init()

    def _data_dir(self) -> str:
        return chart_data_dir(self.chart_dir)

    def on_show(self) -> None:
        pass

    def open_with_model(self, model_name: str) -> None:
        self._model_var.set(str(model_name or "").strip())

    def apply_model_names(self, names: list[str]) -> None:
        self._model_names = list(names)
        self._status_var.set(
            f"{len(names)} model(s) available. Advisory suggestions only."
            if names
            else "No trained models on disk."
        )

    def refresh(self, *, lazy: bool = False) -> None:
        del lazy
        self.apply_model_names(self._model_names)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        hdr = ttk.Frame(self, padding=(8, 8, 8, 4))
        hdr.grid(row=0, column=0, sticky="ew")
        ttk.Label(hdr, text="Experiment Planner", font=SECTION_FONT).pack(side="left")
        ttk.Label(
            hdr,
            text="Advisory only · Recommendation Engine · reads studio artifacts · no retrain",
            foreground=COL_MUTED,
        ).pack(side="left", padx=(12, 0))

        summary = ttk.LabelFrame(self, text="Planner Summary", padding=8)
        summary.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))
        for i in range(4):
            summary.columnconfigure(i, weight=1)
        fields = (
            ("Total Experiments", "total"),
            ("High Priority", "high"),
            ("Medium Priority", "medium"),
            ("Low Priority", "low"),
            ("Highest Risk Feature", "risk_feat"),
            ("Highest Evidence Score", "top_sug"),
            ("Input Artifacts", "inputs"),
            ("Compute time", "compute"),
        )
        for i, (label, key) in enumerate(fields):
            cell = ttk.Frame(summary)
            cell.grid(
                row=i // 4,
                column=i % 4,
                sticky="ew",
                padx=4,
                pady=(0 if i < 4 else 6, 0),
            )
            ttk.Label(cell, text=label, foreground=COL_MUTED).pack(anchor="w")
            ttk.Label(cell, textvariable=self._summary_vars[key]).pack(anchor="w")

        note = ttk.Label(
            self,
            text="Suggestions are deterministic heuristics from existing Importance / Distribution / Drift / Diagnostics artifacts. Evidence Score is rule strength (0–100), not an ML probability. Status/notes live in experiment_state.json (survives Compute); planner.json is regenerable.",
            foreground=COL_MUTED,
            padding=(8, 0, 8, 4),
        )
        note.grid(row=2, column=0, sticky="ew")

        table_wrap = ttk.LabelFrame(
            self,
            text="Research experiments — Model vs Feature (double-click or View for evidence)",
            padding=4,
        )
        table_wrap.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 4))
        table_wrap.columnconfigure(0, weight=1)
        table_wrap.rowconfigure(0, weight=1)

        cols = tuple(c[0] for c in _TABLE_COLS)
        self._tree = ttk.Treeview(
            table_wrap,
            columns=cols,
            show="tree headings",
            selectmode="browse",
        )
        self._tree.heading("#0", text="")
        self._tree.column("#0", width=24, stretch=False, anchor="w")
        for key, title, width in _TABLE_COLS:
            self._tree.heading(
                key, text=title, command=lambda k=key: self._on_sort_header(k)
            )
            anchor = (
                "w"
                if key
                in (
                    "experiment",
                    "estimated_effort",
                    "status",
                    "priority",
                    "category",
                    "family",
                    "hypothesis",
                    "affected_features",
                )
                else "e"
            )
            self._tree.column(
                key,
                width=width,
                anchor=anchor,
                stretch=(key in ("experiment", "hypothesis")),
            )
        vsb = ttk.Scrollbar(table_wrap, orient="vertical", command=self._tree.yview)
        hsb = ttk.Scrollbar(table_wrap, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        self._tree.tag_configure("High", background="#ffe8e8")
        self._tree.tag_configure("Medium", background="#fff6e0")
        self._tree.tag_configure(
            "section", background="#e8eef5", font=("Segoe UI", 9, "bold")
        )
        self._tree.bind("<Double-1>", self._on_row_double_click)

        btn_row = ttk.Frame(table_wrap)
        btn_row.grid(row=2, column=0, sticky="w", pady=(4, 0))
        ttk.Button(btn_row, text="View", command=self._on_row_double_click).pack(
            side="left"
        )
        ttk.Label(btn_row, text="Actions:", foreground=COL_MUTED).pack(
            side="left", padx=(12, 4)
        )
        ttk.Button(
            btn_row, text="Mark In Progress", command=self._on_mark_in_progress
        ).pack(side="left", padx=(0, 4))
        ttk.Button(
            btn_row, text="Mark Complete", command=self._on_mark_complete
        ).pack(side="left", padx=(0, 4))
        ttk.Button(btn_row, text="Reject", command=self._on_reject).pack(
            side="left", padx=(0, 4)
        )
        ttk.Button(btn_row, text="Add Notes", command=self._on_add_notes).pack(
            side="left"
        )

        ttk.Label(
            self,
            textvariable=self._status_var,
            foreground=COL_MUTED,
            padding=(8, 2, 8, 8),
        ).grid(row=4, column=0, sticky="ew")

    def _selected_model(self) -> str:
        return str(self._model_var.get() or "").strip()

    def apply_artifacts(
        self, loaded: dict[str, Any] | None, model_name: str
    ) -> None:
        """Populate from controller-owned load (viewer only)."""
        if not loaded:
            self.mark_unavailable("Unavailable — no Experiment Planner artifacts.")
            return
        from chain_replay_ml.training.paths import model_package_dir

        self._package_dir = model_package_dir(self._data_dir(), model_name)
        self._experiment_state = dict(
            loaded.get("experiment_state")
            or {"experiments": {}}
        )
        self._apply_result(
            loaded.get("summary") or {},
            loaded.get("suggestions") or [],
            loaded.get("meta") or {},
            model_name,
        )

    def mark_unavailable(self, message: str) -> None:
        self._rows = []
        self._display_rows = []
        self._summary = {}
        self._meta = {}
        self._experiment_state = {"experiments": {}}
        self._package_dir = None
        self._iid_to_row.clear()
        for key in self._summary_vars:
            self._summary_vars[key].set("—")
        self._apply_filter_sort()
        self._status_var.set(message)

    def _on_load(self, *, quiet: bool = True) -> None:
        """Legacy self-load; Feature Studio controller owns Load Artifacts."""
        name = self._selected_model()
        if not name:
            if not quiet:
                messagebox.showwarning(
                    "Experiment Planner", "Select a model first.", parent=self
                )
            return
        from chain_replay_ml.recommendation_engine.writer import load_studio_artifacts
        from chain_replay_ml.training.paths import model_package_dir

        pkg = model_package_dir(self._data_dir(), name)
        loaded = load_studio_artifacts(pkg)
        if not loaded:
            self.mark_unavailable("No artifacts yet — click Compute.")
            if not quiet:
                messagebox.showinfo(
                    "Experiment Planner",
                    "No Experiment Planner artifacts found.\n"
                    "Click Compute after running Importance / Distribution / Drift.",
                    parent=self,
                )
            return
        self.apply_artifacts(loaded, name)

    def _apply_result(
        self,
        summary: dict[str, Any],
        suggestions: list[dict[str, Any]],
        meta: dict[str, Any],
        model_name: str,
    ) -> None:
        del model_name
        self._summary = dict(summary or {})
        self._rows = [r for r in suggestions if isinstance(r, dict)]
        self._meta = dict(meta or {})

        self._summary_vars["total"].set(fmt_val(self._summary.get("total_suggestions")))
        self._summary_vars["high"].set(fmt_val(self._summary.get("high_priority")))
        self._summary_vars["medium"].set(fmt_val(self._summary.get("medium_priority")))
        self._summary_vars["low"].set(fmt_val(self._summary.get("low_priority")))
        risk = self._summary.get("highest_risk_feature")
        rs = self._summary.get("highest_risk_score")
        if risk:
            self._summary_vars["risk_feat"].set(
                f"{risk}" + (f" ({float(rs):.1f})" if rs is not None else "")
            )
        else:
            self._summary_vars["risk_feat"].set("—")
        top = (
            self._summary.get("highest_evidence_suggestion")
            or self._summary.get("highest_confidence_suggestion")
            or {}
        )
        if isinstance(top, dict) and top.get("title"):
            score = top.get("evidence_score")
            if score is None and top.get("confidence") is not None:
                try:
                    score = int(round(float(top["confidence"]) * 100))
                except (TypeError, ValueError):
                    score = None
            score_txt = str(int(score)) if score is not None else "—"
            eid = str(top.get("experiment_id") or "").strip()
            title = str(top.get("title") or "")
            label = f"{eid} {title}".strip() if eid else title
            self._summary_vars["top_sug"].set(f"{label} ({score_txt})")
        else:
            self._summary_vars["top_sug"].set("—")
        inputs = self._meta.get("input_artifacts") or {}
        on = [k for k, v in inputs.items() if v]
        self._summary_vars["inputs"].set(", ".join(on) if on else "none")
        wall = self._meta.get("compute_time", self._meta.get("wall_time_sec"))
        self._summary_vars["compute"].set(
            f"{float(wall):.3f}s" if wall is not None else "—"
        )

        self._apply_filter_sort()
        self._status_var.set(
            f"Loaded {len(self._rows)} experiment(s) · advisory only"
        )

    def _on_sort_header(self, col: str) -> None:
        if self._sort_col == col:
            self._sort_desc = not self._sort_desc
        else:
            self._sort_col = col
            self._sort_desc = col == "evidence_score"
        self._apply_filter_sort()

    def _top_n(self) -> int:
        try:
            return max(1, int(self._top_n_var.get() or 20))
        except ValueError:
            return 20

    @staticmethod
    def _feature_names(feats: Any) -> list[str]:
        out: list[str] = []
        if not isinstance(feats, list):
            return out
        for item in feats:
            if isinstance(item, dict):
                name = str(item.get("feature") or "").strip()
                if name:
                    out.append(name)
            else:
                name = str(item).strip()
                if name:
                    out.append(name)
        return out

    @staticmethod
    def _hypothesis_text(row: dict[str, Any]) -> str:
        hyp = str(row.get("hypothesis") or "").strip()
        if hyp:
            return hyp
        benefit = row.get("expected_benefit")
        if isinstance(benefit, dict):
            return str(benefit.get("summary") or "").strip() or "—"
        return str(benefit or "") or "—"

    @staticmethod
    def _family_text(row: dict[str, Any]) -> str:
        fam = row.get("family")
        if fam is not None and str(fam).strip():
            return str(fam).strip()
        ev = row.get("evidence")
        if isinstance(ev, dict) and ev.get("family"):
            return str(ev["family"]).strip()
        return "—"

    @staticmethod
    def _experiment_label(row: dict[str, Any]) -> str:
        eid = str(row.get("experiment_id") or "").strip()
        title = str(row.get("title") or "").strip()
        if eid and title:
            return f"{eid}  {title}"
        return eid or title or "—"

    @staticmethod
    def _effort_text(row: dict[str, Any]) -> str:
        return str(row.get("estimated_effort") or "—")

    @staticmethod
    def _status_text(row: dict[str, Any]) -> str:
        return str(row.get("status") or "Not Started")

    @staticmethod
    def _scope_of(row: dict[str, Any]) -> str:
        scope = str(row.get("experiment_scope") or "").strip().lower()
        if scope in ("model", "feature"):
            return scope
        cat = str(row.get("category") or "").strip()
        if cat in (
            "Retraining",
            "Model Refresh",
            "Threshold Review",
            "Feature Addition",
        ):
            return "model"
        return "feature"

    @staticmethod
    def _benefit_summary(benefit: Any) -> str:
        if isinstance(benefit, dict):
            summary = str(benefit.get("summary") or "").strip()
            if summary:
                return summary
            return "—"
        return str(benefit or "") or "—"

    @staticmethod
    def _reason_table_text(row: dict[str, Any]) -> str:
        bullets = row.get("reason_bullets")
        if isinstance(bullets, list) and bullets:
            return " · ".join(str(b) for b in bullets if str(b).strip())
        return str(row.get("reason") or "")

    def _apply_filter_sort(self) -> None:
        needle = str(self._filter_var.get() or "").strip().lower()
        rows = list(self._rows)
        if needle:
            rows = [
                r
                for r in rows
                if needle in str(r.get("title") or "").lower()
                or needle in str(r.get("experiment_id") or "").lower()
                or needle in str(r.get("category") or "").lower()
                or needle in str(r.get("family") or "").lower()
                or needle in str(r.get("hypothesis") or "").lower()
                or needle in str(r.get("expected_experiment") or "").lower()
                or needle in str(r.get("estimated_effort") or "").lower()
                or needle in str(r.get("status") or "").lower()
                or needle in str(r.get("reason") or "").lower()
                or needle
                in " ".join(str(b) for b in (r.get("reason_bullets") or [])).lower()
                or needle in str(r.get("priority") or "").lower()
                or needle
                in " ".join(self._feature_names(r.get("affected_features"))).lower()
            ]

        pri_rank = {"High": 0, "Medium": 1, "Low": 2}
        effort_rank = {"Easy": 0, "Medium": 1, "High": 2}
        status_rank = {s: i for i, s in enumerate(_STATUS_CHOICES)}
        key = self._sort_col

        def sort_key(r: dict[str, Any]) -> tuple:
            if key == "experiment":
                return (0, self._experiment_label(r).lower())
            if key == "priority":
                return (0, pri_rank.get(str(r.get("priority") or ""), 9))
            if key == "estimated_effort":
                return (0, effort_rank.get(str(r.get("estimated_effort") or ""), 9))
            if key == "status":
                return (0, status_rank.get(str(r.get("status") or ""), 9))
            if key == "hypothesis":
                return (0, self._hypothesis_text(r).lower())
            if key == "expected_benefit":
                return (0, self._benefit_summary(r.get("expected_benefit")).lower())
            if key == "reason":
                return (0, self._reason_table_text(r).lower())
            if key == "affected_features":
                return (0, len(self._feature_names(r.get("affected_features"))))
            if key in ("category", "title", "family"):
                return (0, str(r.get(key) or "").lower())
            if key == "evidence_score":
                try:
                    return (0, float(r.get("evidence_score")))
                except (TypeError, ValueError):
                    try:
                        return (0, float(r.get("confidence") or 0) * 100.0)
                    except (TypeError, ValueError):
                        return (1, 0.0)
            val = r.get(key)
            try:
                return (0, float(val))
            except (TypeError, ValueError):
                return (1, 0.0)

        rows.sort(key=sort_key, reverse=self._sort_desc)

        if self._top_n_only.get():
            rows = rows[: self._top_n()]

        self._display_rows = rows
        self._render_table()

    def _fmt_features(self, feats: Any) -> str:
        names = self._feature_names(feats)
        if not names:
            return "—"
        n = len(names)
        return f"{n} Feature" if n == 1 else f"{n} Features"

    def _fmt_evidence_score(self, row: dict[str, Any]) -> str:
        score = row.get("evidence_score")
        if score is None and row.get("confidence") is not None:
            try:
                score = int(round(float(row["confidence"]) * 100))
            except (TypeError, ValueError):
                return "—"
        if score is None:
            return "—"
        try:
            return str(int(round(float(score))))
        except (TypeError, ValueError):
            return "—"

    def _row_values(self, row: dict[str, Any]) -> tuple:
        return (
            self._experiment_label(row),
            self._effort_text(row),
            self._status_text(row),
            str(row.get("priority") or ""),
            str(row.get("category") or ""),
            self._family_text(row),
            self._fmt_evidence_score(row),
            self._hypothesis_text(row),
            self._fmt_features(row.get("affected_features")),
        )

    def _render_table(self) -> None:
        tree = self._tree
        tree.delete(*tree.get_children())
        self._iid_to_row.clear()

        model_rows = [r for r in self._display_rows if self._scope_of(r) == "model"]
        feature_rows = [r for r in self._display_rows if self._scope_of(r) != "model"]

        sections = (
            (_SECTION_MODEL, model_rows),
            (_SECTION_FEATURE, feature_rows),
        )
        for section_title, section_rows in sections:
            if not section_rows:
                continue
            parent = tree.insert(
                "",
                "end",
                text="",
                values=(f"{section_title} ({len(section_rows)})", "", "", "", "", "", "", "", ""),
                tags=("section",),
                open=True,
            )
            for row in section_rows:
                pri = str(row.get("priority") or "")
                tags = [pri] if pri in ("High", "Medium") else []
                iid = tree.insert(
                    parent,
                    "end",
                    text="",
                    values=self._row_values(row),
                    tags=tuple(tags),
                )
                self._iid_to_row[iid] = row

    def _selected_row(self) -> dict[str, Any] | None:
        sel = self._tree.selection()
        if not sel:
            return None
        return self._iid_to_row.get(sel[0])

    def _status_key(self, row: dict[str, Any]) -> str:
        eid = str(row.get("experiment_id") or "").strip()
        if eid:
            return eid
        return str(row.get("id") or "").strip()

    def _prompt_note(
        self,
        *,
        title: str,
        prompt: str,
        required: bool = False,
    ) -> str | None:
        """Modal note entry. Returns None if cancelled; '' if optional empty OK."""
        win = tk.Toplevel(self)
        win.title(title)
        win.transient(self.winfo_toplevel())
        win.grab_set()
        win.geometry("420x200")
        result: dict[str, str | None] = {"value": None}

        ttk.Label(win, text=prompt, wraplength=400).pack(
            anchor="w", padx=10, pady=(10, 4)
        )
        txt = tk.Text(win, wrap="word", height=6, font=("Segoe UI", 10))
        txt.pack(fill="both", expand=True, padx=10, pady=4)
        txt.focus_set()

        btn = ttk.Frame(win)
        btn.pack(fill="x", padx=10, pady=(4, 10))

        def _ok() -> None:
            text = txt.get("1.0", "end").strip()
            if required and not text:
                messagebox.showwarning(title, "A note/reason is required.", parent=win)
                return
            result["value"] = text
            win.destroy()

        def _cancel() -> None:
            result["value"] = None
            win.destroy()

        ttk.Button(btn, text="OK", command=_ok).pack(side="right")
        ttk.Button(btn, text="Cancel", command=_cancel).pack(side="right", padx=(0, 6))
        win.bind("<Escape>", lambda *_: _cancel())
        self.wait_window(win)
        return result["value"]

    def _apply_action(
        self,
        *,
        action: str,
        note: str | None = None,
        row: dict[str, Any] | None = None,
        require_selection: bool = True,
    ) -> None:
        target = row if row is not None else self._selected_row()
        if target is None:
            if require_selection:
                messagebox.showinfo(
                    "Experiment Planner",
                    "Select an experiment row first.",
                    parent=self,
                )
            return
        if not self._package_dir:
            messagebox.showwarning(
                "Experiment Planner",
                "No model package loaded.",
                parent=self,
            )
            return
        key = self._status_key(target)
        if not key:
            return
        from chain_replay_ml.recommendation_engine.writer import (
            apply_experiment_state,
            record_experiment_action,
        )

        try:
            state = record_experiment_action(
                self._package_dir,
                experiment_id=key,
                action=action,
                note=note,
                internal_id=str(target.get("id") or "") or None,
            )
        except ValueError as exc:
            messagebox.showwarning("Experiment Planner", str(exc), parent=self)
            return
        except OSError as exc:
            messagebox.showerror("Experiment Planner", str(exc), parent=self)
            return

        self._experiment_state = state
        # Re-merge onto all rows
        self._rows = apply_experiment_state(self._rows, state)
        self._apply_filter_sort()
        status = "Not Started"
        entry = (state.get("experiments") or {}).get(key) or {}
        if isinstance(entry, dict):
            status = str(entry.get("status") or status)
        self._status_var.set(f"{action.replace('_', ' ').title()} → {key} ({status})")

    def _on_mark_in_progress(self) -> None:
        self._apply_action(action="mark_in_progress")

    def _on_mark_complete(self) -> None:
        note = self._prompt_note(
            title="Mark Complete",
            prompt="Optional completion note (recommended):",
            required=False,
        )
        if note is None:
            return
        self._apply_action(action="mark_complete", note=note)

    def _on_reject(self) -> None:
        note = self._prompt_note(
            title="Reject Experiment",
            prompt="Rejection reason (required):",
            required=True,
        )
        if note is None:
            return
        self._apply_action(action="reject", note=note)

    def _on_add_notes(self) -> None:
        note = self._prompt_note(
            title="Add Notes",
            prompt="Notes for this experiment:",
            required=True,
        )
        if note is None:
            return
        self._apply_action(action="add_notes", note=note)
    def _format_feature_block(self, item: Any, index: int) -> list[str]:
        if isinstance(item, dict):
            name = str(item.get("feature") or "").strip() or "(unnamed)"
            lines = [f"{index}. {name}"]
            for key, label in _FEATURE_DETAIL_LABELS:
                if key not in item or item.get(key) is None:
                    continue
                val = item[key]
                if isinstance(val, float):
                    if key in ("ks_statistic", "drift", "wasserstein_normalized"):
                        lines.append(
                            f"   {label} {val:.2f}"
                            if val < 10
                            else f"   {label} {val:.1f}"
                        )
                    else:
                        lines.append(f"   {label} {fmt_num(val, 2)}")
                else:
                    lines.append(f"   {label} {val}")
            return lines
        return [f"{index}. {item}"]

    def _family_why_lines(self, row: dict[str, Any]) -> list[str]:
        evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
        agg = evidence.get("aggregate") if isinstance(evidence.get("aggregate"), dict) else {}
        fam = self._family_text(row)
        if fam == "—" and not agg:
            return ["  (model-level experiment — no feature family)"]
        count = evidence.get("feature_count")
        if count is None:
            count = len(self._feature_names(row.get("affected_features")))
        highest = agg.get("highest_risk_feature")
        highest_score = agg.get("highest_risk_score")
        highest_txt = "—"
        if highest:
            highest_txt = str(highest)
            if highest_score is not None:
                try:
                    highest_txt = f"{highest} ({float(highest_score):.1f})"
                except (TypeError, ValueError):
                    pass
        return [
            f"  Family: {fam}",
            f"  Feature count: {count}",
            f"  Average Drift: {agg.get('avg_drift', '—')}",
            f"  Average KS: {agg.get('avg_ks', '—')}",
            f"  Average Importance Rank: {agg.get('avg_rank_gain', '—')}",
            f"  Highest Risk feature: {highest_txt}",
        ]

    def _on_row_double_click(self, _event: object | None = None) -> None:
        row = self._selected_row()
        if row is None:
            messagebox.showinfo(
                "Experiment detail", "Select an experiment row first.", parent=self
            )
            return

        benefit = row.get("expected_benefit")
        hyp = self._hypothesis_text(row)
        benefit_lines: list[str] = [hyp]
        if isinstance(benefit, dict):
            legacy_bits = []
            for key, label in (
                ("model_stability", "Model stability"),
                ("prediction_accuracy", "Prediction accuracy"),
                ("training_speed", "Training speed"),
            ):
                raw = str(benefit.get(key) or "").strip().lower()
                if raw and raw not in ("unknown", "none", ""):
                    legacy_bits.append(
                        f"  {label}: {_BENEFIT_LABELS.get(raw, raw)}"
                    )
            if legacy_bits:
                benefit_lines.extend(legacy_bits)

        bullets = row.get("reason_bullets")
        if isinstance(bullets, list) and bullets:
            reason_lines = [f"• {b}" for b in bullets]
        else:
            reason_lines = [str(row.get("reason") or "(none)")]

        feats = row.get("affected_features") or []
        feat_lines: list[str] = []
        if feats:
            for i, item in enumerate(feats, start=1):
                feat_lines.extend(self._format_feature_block(item, i))
                feat_lines.append("")
            if feat_lines and feat_lines[-1] == "":
                feat_lines.pop()
        else:
            feat_lines = ["(none)"]

        evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
        rule_name = evidence.get("rule_name") or evidence.get("rule") or "—"
        rule_id = evidence.get("rule_id") or str(row.get("id") or "").split("__", 1)[0]
        thresholds = (
            evidence.get("thresholds")
            if isinstance(evidence.get("thresholds"), dict)
            else {}
        )
        top = (
            evidence.get("top_contributors")
            if isinstance(evidence.get("top_contributors"), list)
            else []
        )
        agg = (
            evidence.get("aggregate")
            if isinstance(evidence.get("aggregate"), dict)
            else {}
        )

        th_lines = (
            [f"  {k}: {v}" for k, v in thresholds.items()]
            if thresholds
            else ["  (none recorded)"]
        )
        top_lines: list[str] = []
        if top:
            for i, item in enumerate(top, start=1):
                top_lines.extend(self._format_feature_block(item, i))
        else:
            top_lines = ["(none)"]

        agg_lines = (
            [f"  {k}: {v}" for k, v in agg.items() if v is not None]
            if agg
            else ["  (n/a)"]
        )

        steps = row.get("suggested_next_steps")
        if isinstance(steps, list) and steps:
            step_lines = [f"  □ {s}" for s in steps]
        else:
            step_lines = ["  (none)"]

        expected = str(row.get("expected_experiment") or "").strip() or "(none)"

        try:
            evidence_txt = (
                json.dumps(evidence, indent=2, default=str) if evidence else "(none)"
            )
        except (TypeError, ValueError):
            evidence_txt = str(evidence)

        eid = str(row.get("experiment_id") or "").strip() or "—"
        created_from = str(row.get("created_from") or "").strip() or "—"
        generated_at = str(row.get("generated_at") or "").strip() or "—"
        planner_ver = str(row.get("planner_version") or "").strip() or "—"

        findings = row.get("findings") if isinstance(row.get("findings"), list) else []
        recommendations = (
            row.get("recommendations")
            if isinstance(row.get("recommendations"), list)
            else []
        )
        if not recommendations and findings:
            recommendations = [
                str(f.get("recommendation") or "")
                for f in findings
                if isinstance(f, dict) and f.get("recommendation")
            ]

        finding_lines: list[str] = []
        if recommendations:
            finding_lines.append("Recommendations:")
            for rec in recommendations:
                finding_lines.append(f"  • {rec}")
        if findings and len(findings) > 1:
            finding_lines.append("")
            finding_lines.append(f"Findings ({len(findings)}):")
            for i, f in enumerate(findings, start=1):
                if not isinstance(f, dict):
                    continue
                finding_lines.append(
                    f"  {i}. {f.get('recommendation') or f.get('title') or f.get('rule')}"
                )
                if f.get("rule_id"):
                    finding_lines.append(f"     rule: {f.get('rule_id')}")
                if f.get("evidence_score") is not None:
                    finding_lines.append(f"     evidence: {f.get('evidence_score')}")
        if not finding_lines:
            finding_lines = ["  (single finding — see reason below)"]

        state_notes = row.get("state_notes") if isinstance(row.get("state_notes"), list) else []
        notes_lines: list[str] = []
        if state_notes:
            for n in state_notes:
                if not isinstance(n, dict):
                    continue
                at = n.get("at") or ""
                action = n.get("action") or ""
                text = n.get("text") or ""
                st = n.get("status") or ""
                prefix = f"[{at}] " if at else ""
                meta = " / ".join(p for p in (action, st) if p)
                notes_lines.append(f"  {prefix}{meta}: {text}" if meta else f"  {prefix}{text}")
        else:
            notes_lines = ["  (none)"]

        lines = [
            f"Experiment ID: {eid}",
            f"Title: {row.get('title')}",
            f"Internal id: {row.get('id')}",
            f"Category: {row.get('category')}",
            f"Scope: {self._scope_of(row)}",
            f"Family: {self._family_text(row)}",
            f"Priority: {row.get('priority')}  Evidence Score: {self._fmt_evidence_score(row)}",
            f"Estimated Effort: {self._effort_text(row)}",
            f"Status: {self._status_text(row)}",
            "",
            "Created From:",
            f"  Model: {created_from}",
            f"  Generated at: {generated_at}",
            f"  Planner version: {planner_ver}",
            "",
            "Reason:",
            *reason_lines,
            "",
            *finding_lines,
            "",
            "Hypothesis:",
            *benefit_lines,
            "",
            "Evidence aggregates:",
            *agg_lines,
            "",
            "Expected Experiment:",
            f"  {expected}",
            "",
            "Suggested Next Steps (advisory — not automated):",
            *step_lines,
            "",
            "Why this family?",
            *self._family_why_lines(row),
            "",
            "Rule:",
            f"  id: {rule_id}",
            f"  name: {rule_name}",
            "",
            "Thresholds:",
            *th_lines,
            "",
            "Top contributors:",
            *top_lines,
            "",
            f"Features ({len(self._feature_names(feats))}):",
            *feat_lines,
            "",
            "Status / notes history:",
            *notes_lines,
            "",
            "Evidence (full):",
            evidence_txt,
        ]
        self._show_detail_window(
            f"Experiment detail — {eid}", "\n".join(lines), row=row
        )

    def _show_detail_window(
        self, title: str, body: str, *, row: dict[str, Any] | None = None
    ) -> None:
        win = tk.Toplevel(self)
        win.title(title)
        win.transient(self.winfo_toplevel())
        win.geometry("720x620")
        win.columnconfigure(0, weight=1)
        win.rowconfigure(1, weight=1)

        if row is not None:
            bar = ttk.Frame(win, padding=(8, 8, 8, 0))
            bar.grid(row=0, column=0, columnspan=2, sticky="ew")
            ttk.Label(bar, text=f"Status: {self._status_text(row)}").pack(side="left")

            def _refresh_and_close_note(action: str, *, required: bool) -> None:
                prompt = (
                    "Rejection reason (required):"
                    if action == "reject"
                    else (
                        "Optional completion note:"
                        if action == "mark_complete"
                        else "Notes:"
                    )
                )
                if action == "mark_in_progress":
                    self._apply_action(action=action, row=row)
                    win.destroy()
                    return
                note = self._prompt_note(
                    title=action.replace("_", " ").title(),
                    prompt=prompt,
                    required=required,
                )
                if note is None:
                    return
                self._apply_action(action=action, note=note, row=row)
                win.destroy()

            ttk.Button(
                bar,
                text="Mark In Progress",
                command=lambda: _refresh_and_close_note("mark_in_progress", required=False),
            ).pack(side="left", padx=(12, 4))
            ttk.Button(
                bar,
                text="Mark Complete",
                command=lambda: _refresh_and_close_note("mark_complete", required=False),
            ).pack(side="left", padx=(0, 4))
            ttk.Button(
                bar,
                text="Reject",
                command=lambda: _refresh_and_close_note("reject", required=True),
            ).pack(side="left", padx=(0, 4))
            ttk.Button(
                bar,
                text="Add Notes",
                command=lambda: _refresh_and_close_note("add_notes", required=True),
            ).pack(side="left")
            ttk.Label(
                bar,
                text="Advisory only — does not run experiments",
                foreground=COL_MUTED,
            ).pack(side="left", padx=(12, 0))

        txt = tk.Text(win, wrap="word", font=("Consolas", 10))
        vsb = ttk.Scrollbar(win, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
        txt.grid(row=1, column=0, sticky="nsew", padx=(8, 0), pady=8)
        vsb.grid(row=1, column=1, sticky="ns", padx=(0, 8), pady=8)
        txt.insert("1.0", body)
        txt.configure(state="disabled")
        ttk.Button(win, text="Close", command=win.destroy).grid(
            row=2, column=0, columnspan=2, pady=(0, 8)
        )

    def _on_export(self) -> None:
        if not self._display_rows:
            messagebox.showinfo("Export", "Nothing to export.", parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self,
            title="Export Experiment Planner CSV",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
            initialfile=f"{self._selected_model() or 'planner'}_suggestions.csv",
        )
        if not path:
            return
        fields = [
            "experiment_id",
            "id",
            "priority",
            "category",
            "experiment_scope",
            "family",
            "title",
            "estimated_effort",
            "status",
            "evidence_score",
            "hypothesis",
            "expected_experiment",
            "suggested_next_steps",
            "recommendations",
            "affected_features",
            "reason",
            "reason_bullets",
            "created_from",
            "generated_at",
            "planner_version",
        ]
        try:
            with open(path, "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                for row in self._display_rows:
                    out = {k: row.get(k) for k in fields}
                    out["hypothesis"] = self._hypothesis_text(row)
                    out["family"] = self._family_text(row)
                    out["experiment_scope"] = self._scope_of(row)
                    feats = row.get("affected_features")
                    if isinstance(feats, list):
                        out["affected_features"] = ";".join(
                            self._feature_names(feats)
                        )
                    bullets = row.get("reason_bullets")
                    if isinstance(bullets, list):
                        out["reason_bullets"] = " | ".join(str(b) for b in bullets)
                    steps = row.get("suggested_next_steps")
                    if isinstance(steps, list):
                        out["suggested_next_steps"] = " | ".join(str(s) for s in steps)
                    recs = row.get("recommendations")
                    if isinstance(recs, list):
                        out["recommendations"] = " | ".join(str(r) for r in recs)
                    writer.writerow(out)
        except OSError as exc:
            messagebox.showerror("Export", str(exc), parent=self)
            return
        self._status_var.set(
            f"Exported {len(self._display_rows)} rows → {os.path.basename(path)}"
        )
