"""Normalization / Regime / Math transform tabs for Master Data panel."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any


class NormRegimeMathTransformMixin:
    """UI + selection helpers for post-interaction transform families."""

    def _build_normalization_transform_section(self, parent: ttk.Frame) -> None:
        from chain_replay_ml.dataset_builder.transformations.normalization_ui import (
            DEFAULT_NORMALIZATION_METHODS,
            DEFAULT_NORMALIZATION_WINDOWS,
            METHOD_DISPLAY_LABELS,
            NORMALIZATION_METHODS,
        )

        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, anchor="nw")
        self._normalization_frame = frame

        ttk.Checkbutton(
            frame,
            text="Normalization",
            variable=self._normalization_enabled_var,
            command=self._on_normalization_settings_changed,
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text="Scale features (z-score, robust, min-max, percentile).",
            foreground="#666",
        ).pack(anchor="w", pady=(0, 4))

        body = ttk.Frame(frame)
        body.pack(fill="x", expand=False, pady=(4, 0))
        self._normalization_body = body

        feat_box = ttk.LabelFrame(body, text="Input Features", padding=4)
        feat_box.pack(side="left", fill="y", expand=False, padx=(0, 8))
        self._normalization_feat_host = self._make_scroll_host(feat_box, height=160, width=230)
        self._refresh_flat_feature_checkboxes(
            feat_host_attr="_normalization_feat_host",
            feature_vars_attr="_normalization_feature_vars",
            pending_attr="_pending_normalization_features",
            on_change=self._on_normalization_settings_changed,
        )

        win_box = ttk.LabelFrame(body, text="Windows (rows)", padding=4)
        win_box.pack(side="left", fill="y", padx=(0, 8))
        self._normalization_window_vars = {}
        for win in DEFAULT_NORMALIZATION_WINDOWS:
            var = tk.BooleanVar(value=True)
            self._normalization_window_vars[int(win)] = var
            ttk.Checkbutton(
                win_box,
                text=str(int(win)),
                variable=var,
                command=self._on_normalization_settings_changed,
            ).pack(anchor="w")

        meth_box = ttk.LabelFrame(body, text="Methods", padding=4)
        meth_box.pack(side="left", fill="y", padx=(0, 8))
        self._normalization_method_vars = {}
        for meth in NORMALIZATION_METHODS:
            label = METHOD_DISPLAY_LABELS.get(meth, meth)
            default = meth in DEFAULT_NORMALIZATION_METHODS
            var = tk.BooleanVar(value=default)
            self._normalization_method_vars[str(meth)] = var
            ttk.Checkbutton(
                meth_box,
                text=label,
                variable=var,
                command=self._on_normalization_settings_changed,
            ).pack(anchor="w")

        preview_box = ttk.LabelFrame(body, text="Preview", padding=4)
        preview_box.pack(side="left", fill="y", expand=False)
        ttk.Label(
            preview_box,
            textvariable=self._normalization_preview_var,
            foreground="#555",
            justify="left",
            wraplength=195,
            width=28,
        ).pack(anchor="nw")

        self._sync_normalization_body_state()
        self._update_lag_preview()

    def _build_regime_transform_section(self, parent: ttk.Frame) -> None:
        from chain_replay_ml.dataset_builder.transformations.regime_ui import (
            DEFAULT_REGIME_METHODS,
            DEFAULT_REGIME_WINDOWS,
            METHOD_DISPLAY_LABELS,
            REGIME_METHODS,
        )

        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, anchor="nw")
        self._regime_frame = frame

        ttk.Checkbutton(
            frame,
            text="Regime / Bucket",
            variable=self._regime_enabled_var,
            command=self._on_regime_settings_changed,
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text="Discretise continuous values (threshold / quantile buckets).",
            foreground="#666",
        ).pack(anchor="w", pady=(0, 4))

        body = ttk.Frame(frame)
        body.pack(fill="x", expand=False, pady=(4, 0))
        self._regime_body = body

        feat_box = ttk.LabelFrame(body, text="Input Features", padding=4)
        feat_box.pack(side="left", fill="y", expand=False, padx=(0, 8))
        self._regime_feat_host = self._make_scroll_host(feat_box, height=160, width=230)
        self._refresh_flat_feature_checkboxes(
            feat_host_attr="_regime_feat_host",
            feature_vars_attr="_regime_feature_vars",
            pending_attr="_pending_regime_features",
            on_change=self._on_regime_settings_changed,
        )

        win_box = ttk.LabelFrame(body, text="Windows (rows)", padding=4)
        win_box.pack(side="left", fill="y", padx=(0, 8))
        self._regime_window_vars = {}
        for win in DEFAULT_REGIME_WINDOWS:
            var = tk.BooleanVar(value=True)
            self._regime_window_vars[int(win)] = var
            ttk.Checkbutton(
                win_box,
                text=str(int(win)),
                variable=var,
                command=self._on_regime_settings_changed,
            ).pack(anchor="w")

        meth_box = ttk.LabelFrame(body, text="Methods", padding=4)
        meth_box.pack(side="left", fill="y", padx=(0, 8))
        self._regime_method_vars = {}
        for meth in REGIME_METHODS:
            label = METHOD_DISPLAY_LABELS.get(meth, meth)
            default = meth in DEFAULT_REGIME_METHODS
            var = tk.BooleanVar(value=default)
            self._regime_method_vars[str(meth)] = var
            ttk.Checkbutton(
                meth_box,
                text=label,
                variable=var,
                command=self._on_regime_settings_changed,
            ).pack(anchor="w")

        param_box = ttk.LabelFrame(body, text="Params", padding=4)
        param_box.pack(side="left", fill="y", padx=(0, 8))
        ttk.Label(param_box, text="Threshold").pack(anchor="w")
        ttk.Entry(param_box, textvariable=self._regime_threshold_var, width=8).pack(
            anchor="w", pady=(0, 4)
        )
        ttk.Label(param_box, text="Ternary low / high").pack(anchor="w")
        row = ttk.Frame(param_box)
        row.pack(anchor="w", pady=(0, 4))
        ttk.Entry(row, textvariable=self._regime_low_var, width=6).pack(side="left")
        ttk.Entry(row, textvariable=self._regime_high_var, width=6).pack(side="left", padx=(4, 0))
        ttk.Label(param_box, text="Quantile bins").pack(anchor="w")
        ttk.Entry(param_box, textvariable=self._regime_n_bins_var, width=8).pack(anchor="w")

        preview_box = ttk.LabelFrame(body, text="Preview", padding=4)
        preview_box.pack(side="left", fill="y", expand=False)
        ttk.Label(
            preview_box,
            textvariable=self._regime_preview_var,
            foreground="#555",
            justify="left",
            wraplength=195,
            width=28,
        ).pack(anchor="nw")

        self._sync_regime_body_state()
        self._update_lag_preview()

    def _build_math_transform_section(self, parent: ttk.Frame) -> None:
        from chain_replay_ml.dataset_builder.transformations.math_transform import MATH_OPS
        from chain_replay_ml.dataset_builder.transformations.math_ui import (
            DEFAULT_MATH_OPS,
            OP_DISPLAY_LABELS,
        )

        frame = ttk.Frame(parent)
        frame.pack(fill="both", expand=True, anchor="nw")
        self._math_frame = frame

        ttk.Checkbutton(
            frame,
            text="Math (Unary)",
            variable=self._math_enabled_var,
            command=self._on_math_settings_changed,
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text="Pointwise transforms — one input, one output, no windows.",
            foreground="#666",
        ).pack(anchor="w", pady=(0, 4))

        body = ttk.Frame(frame)
        body.pack(fill="x", expand=False, pady=(4, 0))
        self._math_body = body

        feat_box = ttk.LabelFrame(body, text="Input Features", padding=4)
        feat_box.pack(side="left", fill="y", expand=False, padx=(0, 8))
        self._math_feat_host = self._make_scroll_host(feat_box, height=160, width=230)
        self._refresh_flat_feature_checkboxes(
            feat_host_attr="_math_feat_host",
            feature_vars_attr="_math_feature_vars",
            pending_attr="_pending_math_features",
            on_change=self._on_math_settings_changed,
        )

        op_box = ttk.LabelFrame(body, text="Operations", padding=4)
        op_box.pack(side="left", fill="y", padx=(0, 8))
        self._math_op_vars = {}
        for op in MATH_OPS:
            label = OP_DISPLAY_LABELS.get(op, op)
            default = op in DEFAULT_MATH_OPS
            var = tk.BooleanVar(value=default)
            self._math_op_vars[str(op)] = var
            ttk.Checkbutton(
                op_box,
                text=label,
                variable=var,
                command=self._on_math_settings_changed,
            ).pack(anchor="w")

        clip_box = ttk.LabelFrame(body, text="Clip bounds", padding=4)
        clip_box.pack(side="left", fill="y", padx=(0, 8))
        ttk.Label(clip_box, text="clip_min").pack(anchor="w")
        ttk.Entry(clip_box, textvariable=self._math_clip_min_var, width=8).pack(
            anchor="w", pady=(0, 4)
        )
        ttk.Label(clip_box, text="clip_max (blank = none)").pack(anchor="w")
        ttk.Entry(clip_box, textvariable=self._math_clip_max_var, width=8).pack(anchor="w")

        preview_box = ttk.LabelFrame(body, text="Preview", padding=4)
        preview_box.pack(side="left", fill="y", expand=False)
        ttk.Label(
            preview_box,
            textvariable=self._math_preview_var,
            foreground="#555",
            justify="left",
            wraplength=195,
            width=28,
        ).pack(anchor="nw")

        self._sync_math_body_state()
        self._update_lag_preview()

    def _selected_normalization_features(self) -> list[str]:
        return self._selected_from_feature_vars("_normalization_feature_vars")

    def _selected_normalization_windows(self) -> list[int]:
        return sorted(
            win for win, var in self._normalization_window_vars.items() if var.get()
        )

    def _selected_normalization_methods(self) -> list[str]:
        from chain_replay_ml.dataset_builder.transformations.normalization import (
            NORMALIZATION_METHODS,
        )

        return [
            m
            for m in NORMALIZATION_METHODS
            if self._normalization_method_vars.get(m)
            and self._normalization_method_vars[m].get()
        ]

    def _selected_regime_features(self) -> list[str]:
        return self._selected_from_feature_vars("_regime_feature_vars")

    def _selected_regime_windows(self) -> list[int]:
        return sorted(win for win, var in self._regime_window_vars.items() if var.get())

    def _selected_regime_methods(self) -> list[str]:
        from chain_replay_ml.dataset_builder.transformations.regime import REGIME_METHODS

        return [
            m
            for m in REGIME_METHODS
            if self._regime_method_vars.get(m) and self._regime_method_vars[m].get()
        ]

    def _regime_params(self) -> dict[str, Any]:
        try:
            threshold = float(str(self._regime_threshold_var.get() or "0").strip())
        except ValueError:
            threshold = 0.0
        try:
            low = float(str(self._regime_low_var.get() or "-1").strip())
        except ValueError:
            low = -1.0
        try:
            high = float(str(self._regime_high_var.get() or "1").strip())
        except ValueError:
            high = 1.0
        try:
            n_bins = max(2, int(str(self._regime_n_bins_var.get() or "5").strip()))
        except ValueError:
            n_bins = 5
        return {
            "threshold": threshold,
            "low": low,
            "high": high,
            "n_bins": n_bins,
        }

    def _selected_math_features(self) -> list[str]:
        return self._selected_from_feature_vars("_math_feature_vars")

    def _selected_math_operations(self) -> list[str]:
        from chain_replay_ml.dataset_builder.transformations.math_transform import MATH_OPS

        return [
            op
            for op in MATH_OPS
            if self._math_op_vars.get(op) and self._math_op_vars[op].get()
        ]

    def _math_clip_bounds(self) -> tuple[float, float | None]:
        try:
            clip_min = float(str(self._math_clip_min_var.get() or "0").strip())
        except ValueError:
            clip_min = 0.0
        raw_max = str(self._math_clip_max_var.get() or "").strip()
        if not raw_max:
            return clip_min, None
        try:
            return clip_min, float(raw_max)
        except ValueError:
            return clip_min, None

    def _sync_normalization_body_state(self) -> None:
        self._sync_enable_body("_normalization_body", self._normalization_enabled_var)

    def _on_normalization_settings_changed(self) -> None:
        self._sync_normalization_body_state()
        self._update_lag_preview()
        self._save_prefs()

    def _sync_regime_body_state(self) -> None:
        self._sync_enable_body("_regime_body", self._regime_enabled_var)

    def _on_regime_settings_changed(self) -> None:
        self._sync_regime_body_state()
        self._update_lag_preview()
        self._save_prefs()

    def _sync_math_body_state(self) -> None:
        self._sync_enable_body("_math_body", self._math_enabled_var)

    def _on_math_settings_changed(self) -> None:
        self._sync_math_body_state()
        self._update_lag_preview()
        self._save_prefs()

    def _merge_post_interaction_transforms(self, base: dict[str, Any]) -> dict[str, Any]:
        from chain_replay_ml.dataset_builder.transformations.math_ui import (
            merge_math_into_config,
        )
        from chain_replay_ml.dataset_builder.transformations.normalization_ui import (
            merge_normalization_into_config,
        )
        from chain_replay_ml.dataset_builder.transformations.regime_ui import (
            merge_regime_into_config,
        )

        clip_min, clip_max = self._math_clip_bounds()
        rp = self._regime_params()
        with_math = merge_math_into_config(
            base,
            enabled=bool(self._math_enabled_var.get()),
            features=self._selected_math_features(),
            operations=self._selected_math_operations(),
            clip_min=clip_min,
            clip_max=clip_max,
            partition_by=["trading_day", "token"],
            sample_interval_sec=self._interval_sec(),
        )
        with_norm = merge_normalization_into_config(
            with_math,
            enabled=bool(self._normalization_enabled_var.get()),
            features=self._selected_normalization_features(),
            methods=self._selected_normalization_methods(),
            windows=self._selected_normalization_windows(),
            partition_by=["trading_day", "token"],
            sample_interval_sec=self._interval_sec(),
        )
        return merge_regime_into_config(
            with_norm,
            enabled=bool(self._regime_enabled_var.get()),
            features=self._selected_regime_features(),
            methods=self._selected_regime_methods(),
            windows=self._selected_regime_windows(),
            n_bins=int(rp["n_bins"]),
            threshold=float(rp["threshold"]),
            low=float(rp["low"]),
            high=float(rp["high"]),
            partition_by=["trading_day", "token"],
            sample_interval_sec=self._interval_sec(),
        )
