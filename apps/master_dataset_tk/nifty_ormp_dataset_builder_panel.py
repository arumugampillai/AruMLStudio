"""NIFTY ORMP — Dataset Builder (slice build → labeled Dataset Registry entry)."""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any

from .ormp_service import (
    OrmpBuildInfo,
    ensure_ormp_importable,
    export_ormp_training_dataset,
    list_ormp_builds,
    suggest_ormp_dataset_name,
)


class NiftyOrmpDatasetBuilderPanel(ttk.Frame):
    """Historical Data → ORMP Overview → Dataset Builder tab."""

    def __init__(self, master: tk.Misc, *, chart_dir: str = "") -> None:
        super().__init__(master)
        self.chart_dir = chart_dir
        self._busy = False
        self._name_user_edited = False
        self._setting_name = False
        self._builds: list[OrmpBuildInfo] = []
        self._builds_by_label: dict[str, OrmpBuildInfo] = {}
        self._group_vars: dict[str, tk.BooleanVar] = {}
        self._feature_vars: dict[str, tk.BooleanVar] = {}
        self._group_expanded: dict[str, bool] = {}
        self._group_feature_frames: dict[str, ttk.Frame] = {}
        self._group_toggle_btns: dict[str, ttk.Button] = {}

        self._build_choice_var = tk.StringVar(value="")
        self._from_var = tk.StringVar(value="")
        self._to_var = tk.StringVar(value="")
        self._horizon_var = tk.StringVar(value="10")
        self._label_var = tk.StringVar(value="percent")
        self._name_var = tk.StringVar(value="")
        self._status_var = tk.StringVar(value="Select an ORMP build to export a training dataset.")
        self._summary_var = tk.StringVar(value="")
        self._feat_stats_var = tk.StringVar(value="")

        self._catalog: list[dict[str, Any]] = []
        self._build_ui()
        self._name_var.trace_add("write", self._on_name_typed)
        self._horizon_var.trace_add("write", lambda *_: self._refresh_auto_name())
        self._label_var.trace_add("write", lambda *_: self._refresh_auto_name())

    def set_chart_dir(self, chart_dir: str) -> None:
        self.chart_dir = chart_dir

    def on_show(self) -> None:
        self.refresh()

    def _build_ui(self) -> None:
        wrap = ttk.Frame(self, padding=16)
        wrap.pack(fill="both", expand=True)

        ttk.Label(
            wrap,
            text=(
                "Consume an existing ORMP build — select features, one label, and date range. "
                "Writes a Dataset Registry Parquet (no ORMP rebuild)."
            ),
            foreground="#888",
            wraplength=900,
        ).pack(anchor="w", pady=(0, 10))

        top = ttk.LabelFrame(wrap, text="Source build", padding=10)
        top.pack(fill="x", pady=(0, 8))
        r0 = ttk.Frame(top)
        r0.pack(fill="x")
        ttk.Label(r0, text="ORMP build", width=14).pack(side="left")
        self._build_combo = ttk.Combobox(
            r0, textvariable=self._build_choice_var, state="readonly", width=64
        )
        self._build_combo.pack(side="left", fill="x", expand=True)
        self._build_combo.bind("<<ComboboxSelected>>", lambda _e: self._on_build_selected())
        ttk.Button(r0, text="Refresh", command=self.refresh).pack(side="left", padx=(8, 0))

        r1 = ttk.Frame(top)
        r1.pack(fill="x", pady=(8, 0))
        ttk.Label(r1, text="From date", width=14).pack(side="left")
        ttk.Entry(r1, textvariable=self._from_var, width=14).pack(side="left")
        ttk.Label(r1, text="  To date", width=10).pack(side="left")
        ttk.Entry(r1, textvariable=self._to_var, width=14).pack(side="left")
        ttk.Label(
            r1, text="  (defaults = build coverage)", foreground="#888"
        ).pack(side="left", padx=(8, 0))

        mid = ttk.Frame(wrap)
        mid.pack(fill="both", expand=True, pady=(0, 8))

        left = ttk.LabelFrame(mid, text="Features", padding=8)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        stats = ttk.Frame(left)
        stats.pack(fill="x", pady=(0, 4))
        ttk.Button(stats, text="All groups", command=self._select_all_groups).pack(side="left")
        ttk.Button(stats, text="None", command=self._select_no_groups).pack(side="left", padx=(4, 0))
        ttk.Label(stats, textvariable=self._feat_stats_var, foreground="#888").pack(
            side="right"
        )

        canvas = tk.Canvas(left, highlightthickness=0, height=280)
        scroll = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        self._feat_inner = ttk.Frame(canvas)
        self._feat_inner.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.create_window((0, 0), window=self._feat_inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        right = ttk.LabelFrame(mid, text="Label & output", padding=10)
        right.pack(side="right", fill="y", padx=(6, 0))

        ttk.Label(right, text="Horizon").pack(anchor="w")
        hz = ttk.Frame(right)
        hz.pack(anchor="w", pady=(2, 10))
        for h in ("5", "10", "15"):
            ttk.Radiobutton(hz, text=f"{h}m", variable=self._horizon_var, value=h).pack(
                side="left", padx=(0, 8)
            )

        ttk.Label(right, text="Label type (one per dataset)").pack(anchor="w")
        for val, text in (
            ("points", "Points  (future − spot)"),
            ("percent", "Percent  ((future − spot) / spot)"),
            ("direction", "Direction  (sign)"),
        ):
            ttk.Radiobutton(
                right, text=text, variable=self._label_var, value=val
            ).pack(anchor="w", pady=1)
        ttk.Radiobutton(
            right, text="Custom label (Coming Soon)", state="disabled"
        ).pack(anchor="w", pady=(4, 10))

        ttk.Label(right, text="Dataset name").pack(anchor="w")
        ttk.Entry(right, textvariable=self._name_var, width=42).pack(anchor="w", pady=(2, 4))
        ttk.Button(right, text="Reset auto name", command=self._reset_auto_name).pack(
            anchor="w"
        )

        self._export_btn = ttk.Button(
            right, text="Export to Dataset Registry", command=self._start_export
        )
        self._export_btn.pack(anchor="w", pady=(16, 0), fill="x")

        ttk.Label(wrap, textvariable=self._status_var, foreground="#888").pack(anchor="w")
        ttk.Label(wrap, textvariable=self._summary_var, foreground="#58a6ff").pack(
            anchor="w", pady=(4, 0)
        )

        self._load_catalog()

    def _load_catalog(self, *, build_path: str | None = None) -> None:
        try:
            ensure_ormp_importable(self.chart_dir)
            if build_path and os.path.isfile(build_path):
                from ormp.training_export import feature_group_catalog_for_build

                self._catalog = feature_group_catalog_for_build(build_path)
            else:
                from ormp.feature_groups import feature_group_catalog

                self._catalog = feature_group_catalog()
        except Exception as exc:  # noqa: BLE001
            self._status_var.set(f"Feature catalog unavailable: {exc}")
            self._catalog = []
        self._render_feature_groups()

    def _render_feature_groups(self) -> None:
        for child in self._feat_inner.winfo_children():
            child.destroy()
        self._group_vars.clear()
        self._feature_vars.clear()
        self._group_feature_frames.clear()
        self._group_toggle_btns.clear()

        for group in self._catalog:
            gid = str(group["id"])
            label = str(group["label"])
            feats = list(group.get("features") or [])
            self._group_expanded.setdefault(gid, False)
            gvar = tk.BooleanVar(value=True)
            self._group_vars[gid] = gvar

            hdr = ttk.Frame(self._feat_inner)
            hdr.pack(fill="x", pady=(4, 0))
            btn = ttk.Button(
                hdr,
                text="▸",
                width=3,
                command=lambda g=gid: self._toggle_group(g),
            )
            btn.pack(side="left")
            self._group_toggle_btns[gid] = btn
            ttk.Checkbutton(
                hdr,
                text=f"{label}  ({len(feats)})",
                variable=gvar,
                command=lambda g=gid: self._on_group_toggled(g),
            ).pack(side="left", padx=(4, 0))

            feat_fr = ttk.Frame(self._feat_inner, padding=(28, 0, 0, 4))
            self._group_feature_frames[gid] = feat_fr
            for fname in feats:
                fvar = tk.BooleanVar(value=True)
                self._feature_vars[fname] = fvar
                ttk.Checkbutton(
                    feat_fr,
                    text=fname,
                    variable=fvar,
                    command=self._sync_group_from_features,
                ).pack(anchor="w")
            if self._group_expanded.get(gid):
                feat_fr.pack(fill="x")
                btn.configure(text="▾")

        self._update_feat_stats()

    def _toggle_group(self, gid: str) -> None:
        expanded = not self._group_expanded.get(gid, False)
        self._group_expanded[gid] = expanded
        fr = self._group_feature_frames.get(gid)
        btn = self._group_toggle_btns.get(gid)
        if fr is None:
            return
        if expanded:
            fr.pack(fill="x")
            if btn:
                btn.configure(text="▾")
        else:
            fr.pack_forget()
            if btn:
                btn.configure(text="▸")

    def _on_group_toggled(self, gid: str) -> None:
        enabled = bool(self._group_vars[gid].get())
        for group in self._catalog:
            if group["id"] != gid:
                continue
            for fname in group.get("features") or []:
                if fname in self._feature_vars:
                    self._feature_vars[fname].set(enabled)
        self._update_feat_stats()

    def _sync_group_from_features(self) -> None:
        for group in self._catalog:
            gid = str(group["id"])
            feats = list(group.get("features") or [])
            if not feats or gid not in self._group_vars:
                continue
            any_on = any(self._feature_vars[f].get() for f in feats if f in self._feature_vars)
            self._group_vars[gid].set(any_on)
        self._update_feat_stats()

    def _select_all_groups(self) -> None:
        for v in self._group_vars.values():
            v.set(True)
        for v in self._feature_vars.values():
            v.set(True)
        self._update_feat_stats()

    def _select_no_groups(self) -> None:
        for v in self._group_vars.values():
            v.set(False)
        for v in self._feature_vars.values():
            v.set(False)
        self._update_feat_stats()

    def _selected_features(self) -> list[str]:
        return [f for f, v in self._feature_vars.items() if v.get()]

    def _update_feat_stats(self) -> None:
        n = len(self._selected_features())
        g = sum(1 for v in self._group_vars.values() if v.get())
        self._feat_stats_var.set(f"{n} features · {g} groups")

    def refresh(self) -> None:
        try:
            builds = list_ormp_builds(self.chart_dir)
        except Exception as exc:  # noqa: BLE001
            self._status_var.set(f"Failed to list builds: {exc}")
            return
        self._builds = builds
        labels: list[str] = []
        self._builds_by_label = {}
        for b in builds:
            label = f"{b.display_name}  ·  {b.from_date or '?'} → {b.to_date or '?'}"
            labels.append(label)
            self._builds_by_label[label] = b
        self._build_combo["values"] = labels
        if labels:
            cur = self._build_choice_var.get()
            if cur not in self._builds_by_label:
                self._build_choice_var.set(labels[0])
            self._on_build_selected()
        else:
            self._build_choice_var.set("")
            self._load_catalog()
            self._status_var.set("No ORMP builds — create one on the Builds tab first.")

    def _selected_build(self) -> OrmpBuildInfo | None:
        return self._builds_by_label.get(self._build_choice_var.get())

    def _on_build_selected(self) -> None:
        b = self._selected_build()
        if not b:
            return
        self._load_catalog(build_path=b.path)
        if b.from_date:
            self._from_var.set(b.from_date)
        if b.to_date:
            self._to_var.set(b.to_date)
        self._name_user_edited = False
        self._refresh_auto_name()
        has_mc = any(g.get("id") == "ormp_market_context" for g in self._catalog)
        if not has_mc:
            self._status_var.set(
                f"Build has {sum(len(g.get('features') or []) for g in self._catalog)} features — "
                "Market Context missing; rebuild ORMP on Builds tab to enable EMA ratios."
            )
        else:
            self._status_var.set(
                f"Build ready — {sum(len(g.get('features') or []) for g in self._catalog)} "
                f"features across {len(self._catalog)} groups."
            )

    def _on_name_typed(self, *_args: object) -> None:
        if self._setting_name or not hasattr(self, "_export_btn"):
            return
        b = self._selected_build()
        if not b:
            return
        auto = suggest_ormp_dataset_name(
            b,
            horizon_min=int(self._horizon_var.get() or 10),
            label_type=self._label_var.get() or "percent",
            chart_dir=self.chart_dir,
        )
        if self._name_var.get().strip() != auto:
            self._name_user_edited = True

    def _refresh_auto_name(self) -> None:
        if self._name_user_edited:
            return
        b = self._selected_build()
        if not b:
            return
        try:
            hz = int(self._horizon_var.get() or 10)
        except ValueError:
            hz = 10
        name = suggest_ormp_dataset_name(
            b,
            horizon_min=hz,
            label_type=self._label_var.get() or "percent",
            chart_dir=self.chart_dir,
        )
        self._setting_name = True
        try:
            self._name_var.set(name)
        finally:
            self._setting_name = False

    def _reset_auto_name(self) -> None:
        self._name_user_edited = False
        self._refresh_auto_name()

    def _start_export(self) -> None:
        if self._busy:
            return
        b = self._selected_build()
        if not b:
            messagebox.showerror("ORMP Dataset", "Select an ORMP build.")
            return
        feats = self._selected_features()
        if not feats:
            messagebox.showerror("ORMP Dataset", "Select at least one feature.")
            return
        name = self._name_var.get().strip()
        if not name:
            messagebox.showerror("ORMP Dataset", "Dataset name is required.")
            return
        try:
            hz = int(self._horizon_var.get())
        except ValueError:
            messagebox.showerror("ORMP Dataset", "Invalid horizon.")
            return
        label_type = self._label_var.get()
        from_date = self._from_var.get().strip() or None
        to_date = self._to_var.get().strip() or None

        self._busy = True
        self._export_btn.state(["disabled"])
        self._status_var.set("Exporting…")
        self._summary_var.set("")

        def worker() -> None:
            err: str | None = None
            result: dict[str, Any] | None = None

            def on_progress(msg: str) -> None:
                self.after(0, lambda m=msg: self._status_var.set(m))

            try:
                result = export_ormp_training_dataset(
                    self.chart_dir,
                    build=b,
                    dataset_name=name,
                    feature_columns=feats,
                    label_type=label_type,
                    horizon_min=hz,
                    from_date=from_date,
                    to_date=to_date,
                    on_progress=on_progress,
                )
            except Exception as exc:  # noqa: BLE001
                err = str(exc)

            def done() -> None:
                self._busy = False
                self._export_btn.state(["!disabled"])
                if err:
                    self._status_var.set(f"Export failed: {err}")
                    messagebox.showerror("ORMP Dataset", err)
                    return
                assert result is not None
                if not result.get("ok"):
                    msg = str(result.get("error") or "Export failed")
                    self._status_var.set(msg)
                    messagebox.showerror("ORMP Dataset", msg)
                    return
                exported = int(result.get("rows_exported") or 0)
                dropped = int(result.get("rows_dropped_no_future") or 0)
                summary = (
                    f"Rows exported: {exported:,}\n"
                    f"Rows dropped (insufficient future horizon): {dropped:,}"
                )
                self._summary_var.set(summary.replace("\n", "  ·  "))
                self._status_var.set(
                    f"Registered: {result.get('dataset_name')} → {result.get('parquet_path')}"
                )
                messagebox.showinfo(
                    "ORMP Dataset",
                    f"Dataset registered.\n\n"
                    f"{summary}\n\n"
                    f"Name: {result.get('dataset_name')}\n"
                    f"Label: {result.get('label_column')}\n"
                    f"Features: {len(result.get('feature_columns') or [])}",
                )

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()
