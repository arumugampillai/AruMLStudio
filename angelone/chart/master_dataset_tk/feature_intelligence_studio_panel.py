"""Standalone ML Research Studio page for Feature Intelligence (Sprint 9).

Model Builder → Feature Intelligence. Embeds `feature_intelligence.ui`
panels; no Feature Studio Load/Compute chrome. Read-only against FIC DB
via Public / Semantic Query API only (no direct SQLite).
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from feature_intelligence.api import (
    get_platform_summary,
    inspect_feature,
    search_features,
)
from feature_intelligence.core.config import load_config
from feature_intelligence.query.diagnostics import (
    MATCH_ALL_SQL,
    explorer_empty_hint,
    registry_and_frr_counts,
)
from feature_intelligence.ui.feature_inspector_panel import (
    FeatureInspectorPanel,
    FeatureIntelligenceSearchBar,
)
from feature_intelligence.ui.inspect_format import (
    HIT_GRID_COLUMNS,
    HIT_GRID_HEADINGS,
    SearchPlan,
    build_search_plan,
    filter_hits_by_plan,
    hit_grid_values,
    merge_hit_lists,
)


class FeatureIntelligenceStudioPanel(ttk.Frame):
    """Standalone Feature Intelligence explorer + inspector page."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        chart_dir: str = "",
        db_path: str | Path | None = None,
    ) -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._db_path = Path(db_path) if db_path else load_config().database.path
        abs_db = self._db_path.resolve()
        feat_c, frr_c = registry_and_frr_counts(abs_db)
        boot = (
            f"[FIC DEBUG] Feature Explorer opened db={abs_db} "
            f"feature_registry={feat_c} feature_research_record={frr_c}"
        )
        print(boot)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.rowconfigure(2, weight=2)

        self._hit_rows: list[dict[str, Any]] = []
        self._sort_col: str | None = None
        self._sort_reverse: bool = False
        self._inspect_gen: int = 0
        self._last_search_error: str | None = None

        explorer = ttk.LabelFrame(self, text="Feature Explorer", padding=6)
        explorer.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        explorer.columnconfigure(0, weight=1)

        self._bar = FeatureIntelligenceSearchBar(
            explorer,
            on_search=self._on_search,
            on_sync_registry=self._on_sync_registry,
        )
        self._bar.grid(row=0, column=0, sticky="ew")

        results = ttk.LabelFrame(self, text="Results", padding=4)
        results.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        results.columnconfigure(0, weight=1)
        results.rowconfigure(0, weight=1)

        self._tree = ttk.Treeview(
            results,
            columns=HIT_GRID_COLUMNS,
            show="headings",
            selectmode="browse",
            height=8,
        )
        for col in HIT_GRID_COLUMNS:
            self._tree.heading(
                col,
                text=HIT_GRID_HEADINGS[col],
                command=lambda c=col: self._sort_by(c),
            )
            width = 140 if col in ("canonical_name", "feature_uuid") else 110
            self._tree.column(col, width=width, minwidth=60, stretch=True)
        yscroll = ttk.Scrollbar(results, orient="vertical", command=self._tree.yview)
        xscroll = ttk.Scrollbar(results, orient="horizontal", command=self._tree.xview)
        self._tree.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        self._tree.bind("<<TreeviewSelect>>", self._on_hit_select)
        self._tree.bind("<Double-1>", self._on_hit_select)

        self._hits_status = tk.StringVar(value="Results: 0")
        ttk.Label(results, textvariable=self._hits_status, wraplength=720).grid(
            row=2, column=0, sticky="ew", padx=4, pady=(2, 0)
        )

        self.inspector = FeatureInspectorPanel(self)
        self.inspector.grid(row=2, column=0, sticky="nsew", padx=4, pady=4)

        self.after(50, self._refresh_platform_summary)

    def set_db_path(self, path: str | Path) -> None:
        self._db_path = Path(path)
        abs_db = self._db_path.resolve()
        feat_c, frr_c = registry_and_frr_counts(abs_db)
        print(
            f"[FIC DEBUG] Feature Explorer set_db_path db={abs_db} "
            f"feature_registry={feat_c} feature_research_record={frr_c}"
        )
        self._refresh_platform_summary()

    def _refresh_platform_summary(self) -> None:
        env = get_platform_summary(db_path=self._db_path)
        if env.ok and isinstance(env.data, dict):
            self.inspector.set_platform_summary(env.data)
        else:
            abs_db = self._db_path.resolve()
            feat_c, frr_c = registry_and_frr_counts(abs_db)
            hint = explorer_empty_hint(feature_count=feat_c, frr_count=frr_c)
            self.inspector.set_platform_summary(
                {
                    "counts": {
                        "primitives": None,
                        "features": feat_c,
                        "operators": None,
                        "transformations": None,
                        "ontology_records": None,
                        "research_records": frr_c,
                    },
                    "versions": {},
                    "read_only": True,
                    "empty_hint": hint,
                    "db_path": str(abs_db),
                }
            )

    def _run_engine_query(
        self, *, query: str | None = None, match_all: bool = False
    ) -> list[dict[str, Any]]:
        env = search_features(query=query, match_all=match_all, db_path=self._db_path)
        if not env.ok:
            self._last_search_error = (env.error or {}).get(
                "message", "search failed"
            )
            return []
        data = env.data or {}
        debug = data.get("debug") if isinstance(data, dict) else None
        if isinstance(debug, dict) and debug.get("empty_hint"):
            self._last_search_error = str(debug["empty_hint"])
        return list(data.get("items") or [])

    def _execute_plan(self, plan: SearchPlan) -> list[dict[str, Any]]:
        self._last_search_error = None
        if plan.mode == "structured":
            env = search_features(
                query=plan.structured_query, db_path=self._db_path
            )
            if not env.ok:
                msg = (env.error or {}).get("message", "search failed")
                self._last_search_error = msg
                messagebox.showerror("Semantic Query", msg, parent=self)
                return []
            data = env.data or {}
            debug = data.get("debug") if isinstance(data, dict) else None
            if isinstance(debug, dict) and debug.get("empty_hint"):
                self._last_search_error = str(debug["empty_hint"])
            return list(data.get("items") or [])

        # Scope strategies are OR-united (each branch filtered independently).
        buckets: list[list[dict[str, Any]]] = []
        client_filters = (
            plan.name_substring
            or plan.feat_substring
            or plan.research_substring
        )
        if plan.match_all and client_filters:
            all_items = self._run_engine_query(match_all=True)
            if plan.name_substring:
                buckets.append(
                    filter_hits_by_plan(
                        all_items,
                        SearchPlan(mode="scoped", name_substring=plan.name_substring),
                    )
                )
            if plan.feat_substring:
                buckets.append(
                    filter_hits_by_plan(
                        all_items,
                        SearchPlan(mode="scoped", feat_substring=plan.feat_substring),
                    )
                )
            if plan.research_substring:
                buckets.append(
                    filter_hits_by_plan(
                        all_items,
                        SearchPlan(
                            mode="scoped",
                            research_substring=plan.research_substring,
                        ),
                    )
                )
        elif plan.match_all:
            buckets.append(self._run_engine_query(match_all=True))

        for q in plan.engine_queries:
            buckets.append(self._run_engine_query(query=q))

        return merge_hit_lists(*buckets) if buckets else []

    def _populate_tree(self, items: list[dict[str, Any]]) -> None:
        self._tree.delete(*self._tree.get_children())
        self._hit_rows = list(items)
        for i, item in enumerate(items):
            self._tree.insert("", "end", iid=str(i), values=hit_grid_values(item))

    def _sort_by(self, col: str) -> None:
        if self._sort_col == col:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_col = col
            self._sort_reverse = False

        def _key(row: dict[str, Any]) -> str:
            val = row.get(col)
            return "" if val is None else str(val).lower()

        ordered = sorted(self._hit_rows, key=_key, reverse=self._sort_reverse)
        self._populate_tree(ordered)

    def _status_for_results(self, items: list[dict[str, Any]], plan: SearchPlan) -> str:
        abs_db = self._db_path.resolve()
        feat_c, frr_c = registry_and_frr_counts(abs_db)
        note = f" — {plan.note}" if plan.note else ""
        base = f"Results: {len(items)}{note}"
        if items:
            return base
        if self._last_search_error:
            return f"{base} — {self._last_search_error}"
        hint = explorer_empty_hint(feature_count=feat_c, frr_count=frr_c)
        if hint:
            return f"{base} — {hint}"
        return (
            f"{base} (db={abs_db}; features={feat_c}; FRR={frr_c}; "
            f"List All SQL: {MATCH_ALL_SQL})"
        )

    def _on_search(self) -> None:
        abs_db = self._db_path.resolve()
        feat_c, frr_c = registry_and_frr_counts(abs_db)
        raw = self._bar.query_var.get()
        print(
            f"[FIC DEBUG] Feature Explorer search raw={raw!r} "
            f"db={abs_db} feature_registry={feat_c} "
            f"feature_research_record={frr_c}"
        )
        plan = build_search_plan(raw, self._bar.selected_scopes())
        items = self._execute_plan(plan)
        self._sort_col = None
        self._sort_reverse = False
        self._populate_tree(items)
        self._hits_status.set(self._status_for_results(items, plan))
        # Keep dashboard visible until a row is selected
        if not items:
            self.inspector.clear()
            self._refresh_platform_summary()

    def _resolve_data_dir(self) -> Path | None:
        if self.chart_dir:
            candidate = Path(self.chart_dir) / "data"
            if candidate.is_dir():
                return candidate
        # Fallback: package chart/data next to feature_intelligence
        pkg_chart = Path(__file__).resolve().parent.parent
        candidate = pkg_chart / "data"
        return candidate if candidate.is_dir() else None

    def _on_sync_registry(self) -> None:
        """Admin action: pull legacy Feature Registry into FIC feature_registry."""
        data_dir = self._resolve_data_dir()
        if data_dir is None:
            messagebox.showerror(
                "Sync from Feature Registry",
                "Could not locate chart data directory "
                "(set chart_dir or ensure angelone/chart/data exists).",
                parent=self,
            )
            return
        if not messagebox.askyesno(
            "Sync from Feature Registry",
            "Import features from the legacy Feature Registry into FIC?\n\n"
            f"Source: {data_dir}\n"
            f"Target DB: {self._db_path.resolve()}\n\n"
            "Already-registered features are skipped. "
            "Research FRR shells are created for new imports.",
            parent=self,
        ):
            return

        db = self._db_path
        src = data_dir

        def _worker() -> None:
            err: str | None = None
            summary_dict: dict[str, Any] | None = None
            try:
                from feature_intelligence.registry.feature_service import (
                    FeatureRegistryService,
                )

                svc = FeatureRegistryService(db)
                summary = svc.synchronize_from_feature_registry(
                    src,
                    mode="lenient",
                    research_sync=True,
                )
                summary_dict = summary.to_dict()
            except Exception as exc:  # noqa: BLE001
                err = str(exc)

            def _apply() -> None:
                if err:
                    messagebox.showerror(
                        "Sync from Feature Registry",
                        f"Sync failed:\n{err}",
                        parent=self,
                    )
                    return
                assert summary_dict is not None
                failed_n = int(summary_dict.get("failed") or 0)
                fail_preview = ""
                failures = list(summary_dict.get("failures") or [])
                if failures:
                    lines = []
                    for f in failures[:12]:
                        if isinstance(f, dict):
                            lines.append(
                                f"  • {f.get('name') or '?'}: {f.get('reason')}"
                            )
                        else:
                            lines.append(f"  • {f}")
                    if len(failures) > 12:
                        lines.append(f"  … and {len(failures) - 12} more")
                    fail_preview = "\n\nFailures:\n" + "\n".join(lines)
                messagebox.showinfo(
                    "Sync from Feature Registry",
                    (
                        f"Total source: {summary_dict.get('total_source')}\n"
                        f"Already registered: {summary_dict.get('already_registered')}\n"
                        f"Newly imported: {summary_dict.get('newly_imported')}\n"
                        f"Failed: {failed_n}\n"
                        f"Research created: {summary_dict.get('research_created')}\n"
                        f"Duration: {summary_dict.get('duration_ms')} ms"
                        f"{fail_preview}"
                    ),
                    parent=self,
                )
                self._refresh_platform_summary()
                self._on_search()

            try:
                self.after(0, _apply)
            except tk.TclError:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def _selected_row(self) -> dict[str, Any] | None:
        sel = self._tree.selection()
        if not sel:
            return None
        try:
            idx = int(sel[0])
        except (TypeError, ValueError):
            return None
        if idx < 0 or idx >= len(self._hit_rows):
            return None
        return self._hit_rows[idx]

    def _on_hit_select(self, _event: object = None) -> None:
        row = self._selected_row()
        if not row:
            return
        self._inspect(research_uuid=str(row.get("research_uuid") or ""))

    def _inspect(
        self,
        *,
        feature_uuid: str | None = None,
        research_uuid: str | None = None,
        canonical_name: str | None = None,
    ) -> None:
        """Load inspector asynchronously so search stays responsive."""
        self._inspect_gen += 1
        gen = self._inspect_gen
        self.inspector.show_loading()
        db = self._db_path

        def _worker() -> None:
            env = inspect_feature(
                feature_uuid=feature_uuid,
                research_uuid=research_uuid,
                canonical_name=canonical_name,
                db_path=db,
            )

            def _apply() -> None:
                if gen != self._inspect_gen:
                    return
                if not env.ok:
                    messagebox.showerror(
                        "Feature Inspector",
                        (env.error or {}).get("message", "inspect failed"),
                        parent=self,
                    )
                    self.inspector.clear()
                    self._refresh_platform_summary()
                    return
                self.inspector.load_inspect(
                    env.data if isinstance(env.data, dict) else None
                )

            try:
                self.after(0, _apply)
            except tk.TclError:
                pass

        threading.Thread(target=_worker, daemon=True).start()
