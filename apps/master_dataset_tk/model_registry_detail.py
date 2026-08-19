"""Render Model Registry detail tabs from load_model_detail() document."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

from .model_registry_widgets import (
    ACCENT,
    COL_HOLDOUT,
    COL_MUTED,
    COL_OK,
    COL_PRODUCTION,
    COL_TRAINING,
    COL_WARN,
    ScrollableFrame,
    clear_children,
    data_table,
    drift_score_bars,
    fmt_num,
    fmt_pct,
    fmt_rows,
    fmt_rupee,
    fmt_signed_pct,
    fmt_signed_rupee,
    fmt_val,
    importance_list,
    inline_spec_rows,
    json_block,
    kv_block,
    metrics_stage,
    outlier_impact_card,
    BODY_FONT,
    section_desc,
    section_title,
    spec_grid,
)


def _meta(doc: dict[str, Any]) -> dict[str, Any]:
    art = doc.get("metadata") or {}
    return art.get("data") if isinstance(art.get("data"), dict) else {}


def _cfg(doc: dict[str, Any]) -> dict[str, Any]:
    return doc.get("config") if isinstance(doc.get("config"), dict) else {}


def _summary(doc: dict[str, Any]) -> dict[str, Any]:
    return doc.get("training_summary") if isinstance(doc.get("training_summary"), dict) else {}


def _metrics(doc: dict[str, Any]) -> dict[str, Any]:
    return doc.get("metrics") if isinstance(doc.get("metrics"), dict) else {}


def _prod(doc: dict[str, Any]) -> dict[str, Any]:
    return doc.get("production_metrics") if isinstance(doc.get("production_metrics"), dict) else {}


def _prediction_type(doc: dict[str, Any]) -> str:
    prod = _prod(doc)
    cfg = _cfg(doc)
    meta = _meta(doc)
    summary = _summary(doc)
    raw = (
        prod.get("prediction_type")
        or cfg.get("prediction_type")
        or meta.get("prediction_type")
        or summary.get("prediction_type")
        or doc.get("prediction_type")
        or "regression"
    )
    return str(raw or "regression").strip().lower()


def _is_classification_model(doc: dict[str, Any]) -> bool:
    return _prediction_type(doc) in ("binary", "classification", "multiclass")


def _is_triple_barrier_model(doc: dict[str, Any]) -> bool:
    try:
        from chain_replay_ml.training.registry import resolve_model_registry_family

        row = doc.get("table_row") if isinstance(doc.get("table_row"), dict) else {}
        cfg = _cfg(doc)
        meta = _meta(doc)
        probe = {
            "label_strategy": (
                row.get("label_strategy")
                or cfg.get("label_strategy")
                or meta.get("label_strategy")
                or doc.get("label_strategy")
            ),
            "target": row.get("target") or cfg.get("target") or meta.get("target") or doc.get("target"),
        }
        return resolve_model_registry_family(probe) == "triple_barrier"
    except Exception:
        return False


def _resolve_label_run_id(doc: dict[str, Any]) -> str:
    row = doc.get("table_row") if isinstance(doc.get("table_row"), dict) else {}
    cfg = _cfg(doc)
    meta = _meta(doc)
    for src in (
        row.get("label_run_id"),
        cfg.get("label_run_id"),
        meta.get("label_run_id"),
        doc.get("label_run_id"),
        (cfg.get("training_config") or {}).get("label_run_id")
        if isinstance(cfg.get("training_config"), dict)
        else None,
    ):
        rid = str(src or "").strip()
        if rid:
            return rid
    return ""


def _fmt_auc(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "—"


def _metric_strength_tag(*, pct: Any = None, auc: Any = None) -> tuple[str, str]:
    """Return (emoji, color) for quick strength / weakness reading."""
    if auc is not None:
        try:
            v = float(auc)
        except (TypeError, ValueError):
            return ("", COL_MUTED)
        if v >= 0.75:
            return ("🟢", COL_OK)
        if v >= 0.65:
            return ("🟠", "#E65100")
        return ("🔴", COL_WARN)
    if pct is not None:
        try:
            v = float(pct)
        except (TypeError, ValueError):
            return ("", COL_MUTED)
        if v >= 55.0:
            return ("🟢", COL_OK)
        if v >= 40.0:
            return ("🟠", "#E65100")
        return ("🔴", COL_WARN)
    return ("", COL_MUTED)


def _fmt_count_with_pct(count: Any, total: int) -> str:
    try:
        n = int(count)
    except (TypeError, ValueError):
        return "—"
    if total <= 0:
        return f"{n:,}"
    return f"{n:,} ({100.0 * n / total:.1f}%)"


def _confusion_from_prod(prod: dict[str, Any]) -> dict[str, int] | None:
    conf = prod.get("confusion")
    if not isinstance(conf, dict):
        return None
    out: dict[str, int] = {}
    for key in ("tn", "fp", "fn", "tp"):
        try:
            out[key] = int(conf.get(key) or 0)
        except (TypeError, ValueError):
            out[key] = 0
    if sum(out.values()) <= 0:
        return None
    return out


def _threshold_analysis_candidates(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect first available Threshold Analysis rows from model detail blobs."""
    from chain_replay_ml.training.evaluator import normalize_threshold_analysis_rows

    prod = _prod(doc)
    metrics = _metrics(doc)
    wf = _wf(doc)
    wf_summary = wf.get("summary") if isinstance(wf.get("summary"), dict) else {}
    wf_data = wf_summary.get("data") if isinstance(wf_summary.get("data"), dict) else {}
    sources = [
        prod.get("threshold_analysis"),
        metrics.get("threshold_analysis"),
        (metrics.get("validation") or {}).get("threshold_analysis")
        if isinstance(metrics.get("validation"), dict)
        else None,
        (metrics.get("production_walk_forward") or {}).get("threshold_analysis")
        if isinstance(metrics.get("production_walk_forward"), dict)
        else None,
        wf_summary.get("threshold_analysis"),
        wf_data.get("threshold_analysis"),
    ]
    for src in sources:
        if isinstance(src, list) and src:
            return normalize_threshold_analysis_rows(src)
    return []


def resolve_threshold_analysis_rows(
    doc: dict[str, Any],
    *,
    chart_dir: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """
    Return (rows, source_label) for the Threshold Analysis tab.

    Prefers persisted metrics; falls back to Prediction Run OOS probs for older models.
    """
    from chain_replay_ml.training.evaluator import (
        attach_trades_per_day,
        normalize_threshold_analysis_rows,
        threshold_analysis_from_prediction_rows,
    )

    n_days = _threshold_analysis_day_count(doc, chart_dir=chart_dir)
    rows = _threshold_analysis_candidates(doc)
    if rows:
        return (
            attach_trades_per_day(normalize_threshold_analysis_rows(rows), n_days),
            "Walk-forward validation (persisted)",
        )

    if not chart_dir:
        return [], "unavailable"
    model_name = str(doc.get("model_name") or "").strip()
    if not model_name:
        return [], "unavailable"
    try:
        from chain_replay_ml.prediction_runs.registry import list_runs
        from chain_replay_ml.prediction_runs.store import PredictionRunStore

        from .build_service import chart_data_dir

        data_dir = chart_data_dir(chart_dir)
        runs = list_runs(data_dir, model_name, limit=5)
        if not runs:
            return [], "unavailable"
        run_id = str(runs[0].get("run_id") or "")
        if not run_id:
            return [], "unavailable"
        with PredictionRunStore(data_dir) as store:
            pred_rows = store.list_all_rows(run_id)
        rebuilt = threshold_analysis_from_prediction_rows(pred_rows)
        if rebuilt:
            # Prefer day count from prediction rows; fall back to package days.
            rebuilt_days = None
            for row in rebuilt:
                if row.get("n_days") is not None:
                    try:
                        rebuilt_days = int(row["n_days"])
                        break
                    except (TypeError, ValueError):
                        pass
            return (
                attach_trades_per_day(
                    normalize_threshold_analysis_rows(rebuilt),
                    rebuilt_days if rebuilt_days is not None else n_days,
                ),
                f"Prediction Run {run_id[:8]}…",
            )
    except Exception:
        return [], "unavailable"
    return [], "unavailable"


def _threshold_analysis_day_count(
    doc: dict[str, Any],
    *,
    chart_dir: str | None = None,
) -> int | None:
    """Best-effort trading-day count for Trades/Day = BUY Signals / days."""
    snap = doc.get("dataset_build_snapshot") if isinstance(doc.get("dataset_build_snapshot"), dict) else {}
    summary = _summary(doc)
    meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    cfg = doc.get("config") if isinstance(doc.get("config"), dict) else {}
    candidates = (
        doc.get("trading_days"),
        snap.get("trading_days"),
        summary.get("trading_days"),
        meta.get("trading_days"),
        meta.get("day_count"),
        cfg.get("trading_days"),
    )
    for raw in candidates:
        if isinstance(raw, (list, tuple)):
            days = [str(d).strip() for d in raw if str(d).strip()]
            if days:
                return len(days)
        try:
            n = int(raw)
            if n > 0:
                return n
        except (TypeError, ValueError):
            pass

    if not chart_dir:
        return None
    model_name = str(doc.get("model_name") or "").strip()
    dataset = str(
        doc.get("dataset")
        or cfg.get("dataset")
        or meta.get("dataset")
        or summary.get("dataset")
        or ""
    ).strip()
    if not model_name and not dataset:
        return None
    try:
        from chain_replay_ml.training.lifecycle_store import _extract_trading_days

        from .build_service import chart_data_dir

        data_dir = chart_data_dir(chart_dir)
        return _extract_trading_days(
            meta,
            cfg,
            data_dir=data_dir,
            dataset_name=dataset or None,
        )
    except Exception:
        return None


def _fmt_int_count(value: Any) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "—"


def _fmt_trades_per_day(value: Any) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v)):,}"
    return f"{v:,.1f}"


def _render_confusion_matrix(parent: tk.Misc, conf: dict[str, int]) -> None:
    """Hit/Miss confusion grid under Production Metrics (counts + % of total)."""
    section_title(parent, "Confusion Matrix")
    section_desc(parent, "Predicted vs actual Hit / Miss (production champion)")

    total = int(conf.get("tp", 0) + conf.get("fp", 0) + conf.get("fn", 0) + conf.get("tn", 0))
    grid = ttk.Frame(parent, padding=(4, 6))
    grid.pack(anchor="w", fill="x")

    # Column headers
    ttk.Label(grid, text="", width=12).grid(row=0, column=0, padx=4, pady=2)
    ttk.Label(grid, text="Actual", font=("Segoe UI", 9, "bold"), foreground=COL_MUTED).grid(
        row=0, column=1, columnspan=2, pady=(0, 2),
    )
    ttk.Label(grid, text="", width=12).grid(row=1, column=0)
    ttk.Label(grid, text="Hit", font=("Segoe UI", 9, "bold"), width=14, anchor="center").grid(
        row=1, column=1, padx=6, pady=2,
    )
    ttk.Label(grid, text="Miss", font=("Segoe UI", 9, "bold"), width=14, anchor="center").grid(
        row=1, column=2, padx=6, pady=2,
    )

    # Row headers + cells with percentages
    ttk.Label(grid, text="Pred Hit", font=("Segoe UI", 9, "bold"), width=12, anchor="e").grid(
        row=2, column=0, padx=(0, 8), pady=4, sticky="e",
    )
    ttk.Label(
        grid,
        text=_fmt_count_with_pct(conf.get("tp"), total),
        font=("Consolas", 10, "bold"),
        width=14,
        anchor="center",
    ).grid(row=2, column=1, padx=6, pady=4)
    ttk.Label(
        grid,
        text=_fmt_count_with_pct(conf.get("fp"), total),
        font=("Consolas", 10, "bold"),
        width=14,
        anchor="center",
    ).grid(row=2, column=2, padx=6, pady=4)

    ttk.Label(grid, text="Pred Miss", font=("Segoe UI", 9, "bold"), width=12, anchor="e").grid(
        row=3, column=0, padx=(0, 8), pady=4, sticky="e",
    )
    ttk.Label(
        grid,
        text=_fmt_count_with_pct(conf.get("fn"), total),
        font=("Consolas", 10, "bold"),
        width=14,
        anchor="center",
    ).grid(row=3, column=1, padx=6, pady=4)
    ttk.Label(
        grid,
        text=_fmt_count_with_pct(conf.get("tn"), total),
        font=("Consolas", 10, "bold"),
        width=14,
        anchor="center",
    ).grid(row=3, column=2, padx=6, pady=4)

    ttk.Label(
        grid,
        text=(
            f"Total rows: {total:,}  ·  "
            f"TP={conf.get('tp', 0):,}  FP={conf.get('fp', 0):,}  "
            f"FN={conf.get('fn', 0):,}  TN={conf.get('tn', 0):,}"
        ),
        foreground=COL_MUTED,
        font=("Segoe UI", 8),
    ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(8, 0))


def _render_classification_metric_row(
    parent: tk.Misc,
    label: str,
    value_text: str,
    *,
    emoji: str = "",
    color: str = COL_MUTED,
    label_width: int = 22,
    emphasize: bool = False,
) -> None:
    row = ttk.Frame(parent)
    row.pack(fill="x", pady=1)
    ttk.Label(
        row,
        text=label,
        width=label_width,
        anchor="w",
        font=("Segoe UI", 9, "bold") if emphasize else ("Segoe UI", 9),
        foreground=COL_MUTED if not emphasize else ACCENT,
    ).pack(side="left")
    ttk.Label(
        row,
        text=value_text,
        anchor="w",
        font=("Consolas", 10, "bold") if emphasize else ("Consolas", 10),
        foreground=color if value_text != "—" else COL_MUTED,
    ).pack(side="left", padx=(4, 0))
    if emoji:
        ttk.Label(row, text=emoji, font=("Segoe UI", 9)).pack(side="left", padx=(6, 0))


def _render_classification_production_metrics(parent: tk.Misc, prod: dict[str, Any]) -> None:
    """Trader-facing classification overview: rates + metrics left, composite right."""
    from chain_replay_ml.training.objective_scoring import classification_composite_breakdown

    # --- Class balance / signal rate (highest priority context) ---
    pos = prod.get("positive_rate_pct")
    pred_pos = prod.get("predicted_positive_rate_pct")
    if pos is None or pred_pos is None:
        conf = _confusion_from_prod(prod)
        if conf is not None:
            total = sum(conf.values())
            if total > 0:
                if pos is None:
                    pos = round(100.0 * (conf["tp"] + conf["fn"]) / total, 2)
                if pred_pos is None:
                    pred_pos = round(100.0 * (conf["tp"] + conf["fp"]) / total, 2)

    # Same-row layout: production metrics | composite
    split = ttk.Frame(parent)
    split.pack(fill="x", anchor="nw")
    split.columnconfigure(0, weight=3)
    split.columnconfigure(1, weight=2)
    left = ttk.Frame(split)
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
    right = ttk.Frame(split, padding=(8, 0, 0, 0))
    right.grid(row=0, column=1, sticky="nw")

    context = ttk.Frame(left, padding=(0, 2, 0, 6))
    context.pack(fill="x")
    _render_classification_metric_row(
        context,
        "Positive Class Rate",
        fmt_pct(pos) if pos is not None else "—",
        color=ACCENT,
        emphasize=True,
        label_width=20,
    )
    if pos is not None:
        try:
            neg = max(0.0, 100.0 - float(pos))
            ttk.Label(
                context,
                text=f"Negative samples: {neg:.1f}%",
                foreground=COL_MUTED,
                font=("Segoe UI", 8),
            ).pack(anchor="w", padx=(4, 0), pady=(0, 4))
        except (TypeError, ValueError):
            pass
    _render_classification_metric_row(
        context,
        "Predicted BUY Rate",
        fmt_pct(pred_pos) if pred_pos is not None else "—",
        color=ACCENT,
        emphasize=True,
        label_width=20,
    )
    thr = prod.get("threshold")
    if thr is None:
        thr = 0.5
    _render_classification_metric_row(
        context,
        "Decision Threshold",
        f"{float(thr):.2f}",
        color=ACCENT,
        emphasize=True,
        label_width=20,
    )

    # --- Core metrics in trader priority order ---
    section_desc(left, "When the model says BUY, Precision is the first question.")
    metrics_fr = ttk.Frame(left, padding=(0, 2, 0, 4))
    metrics_fr.pack(fill="x")

    ordered = (
        ("Precision", prod.get("precision_pct"), "pct"),
        ("Recall", prod.get("recall_pct"), "pct"),
        ("F1 Score", prod.get("f1_pct"), "pct"),
        ("ROC-AUC", prod.get("roc_auc"), "auc"),
        ("Accuracy", prod.get("accuracy_pct"), "pct"),
        ("Specificity", prod.get("specificity_pct"), "pct"),
    )
    for label, raw, kind in ordered:
        if kind == "auc":
            emoji, color = _metric_strength_tag(auc=raw)
            value_text = _fmt_auc(raw)
        else:
            emoji, color = _metric_strength_tag(pct=raw)
            value_text = fmt_pct(raw) if raw is not None else "—"
        _render_classification_metric_row(
            metrics_fr, label, value_text, emoji=emoji, color=color, label_width=14
        )

    # Keep secondary scores available but de-emphasized
    if prod.get("pr_auc") is not None or prod.get("brier_score") is not None:
        ttk.Label(
            metrics_fr,
            text=(
                f"PR-AUC {_fmt_auc(prod.get('pr_auc'))}  ·  "
                f"Brier {_fmt_auc(prod.get('brier_score'))}"
            ),
            foreground=COL_MUTED,
            font=("Segoe UI", 8),
        ).pack(anchor="w", pady=(4, 0))

    # --- Composite (right column, same row) ---
    section_title(right, "Composite", color=ACCENT)
    ttk.Label(
        right,
        text=fmt_num(prod.get("composite_score")),
        font=("Consolas", 16, "bold"),
        foreground=ACCENT,
    ).pack(anchor="w", pady=(0, 6))
    breakdown = classification_composite_breakdown(prod)
    if breakdown:
        ttk.Label(
            right,
            text="Built from:",
            foreground=COL_MUTED,
            font=("Segoe UI", 8),
        ).pack(anchor="w")
        for item in breakdown:
            ttk.Label(
                right,
                text=f"{item['weight_pct']:.0f}% {item['label']}",
                foreground=COL_MUTED,
                font=("Segoe UI", 8),
            ).pack(anchor="w", padx=(8, 0))

    # Confusion matrix directly under Composite
    conf = _confusion_from_prod(prod)
    if conf is not None:
        _render_confusion_matrix(right, conf)
    else:
        section_title(right, "Confusion Matrix")
        section_desc(
            right,
            "Not available yet — retrain this classification model to persist Hit/Miss counts.",
        )


def _production_metrics_table_rows(prod: dict[str, Any], *, classification: bool) -> list[tuple[str, str, str]]:
    src = prod.get("source_label") or prod.get("stage_label") or "Production champion"
    if classification:
        # Fallback tabular shape (Model Metrics tab / older callers).
        return [
            ("Positive Class Rate", fmt_pct(prod.get("positive_rate_pct")) if prod.get("positive_rate_pct") is not None else "—", src),
            ("Predicted BUY Rate", fmt_pct(prod.get("predicted_positive_rate_pct")) if prod.get("predicted_positive_rate_pct") is not None else "—", src),
            ("Decision Threshold", f"{float(prod.get('threshold') if prod.get('threshold') is not None else 0.5):.2f}", src),
            ("Precision", fmt_pct(prod.get("precision_pct")) if prod.get("precision_pct") is not None else "—", src),
            ("Recall", fmt_pct(prod.get("recall_pct")) if prod.get("recall_pct") is not None else "—", src),
            ("F1 Score", fmt_pct(prod.get("f1_pct")) if prod.get("f1_pct") is not None else "—", src),
            ("ROC-AUC", _fmt_auc(prod.get("roc_auc")), src),
            ("Accuracy", fmt_pct(prod.get("accuracy_pct")) if prod.get("accuracy_pct") is not None else "—", src),
            (
                "Specificity",
                fmt_pct(prod.get("specificity_pct")) if prod.get("specificity_pct") is not None else "—",
                src,
            ),
            ("Composite Score", fmt_num(prod.get("composite_score")), src),
        ]
    return [
        ("MAE (₹)", fmt_rupee(prod.get("mae")), src),
        ("RMSE (₹)", fmt_rupee(prod.get("rmse")), src),
        (
            "Direction Accuracy",
            fmt_pct(prod.get("directional_accuracy_pct")) if prod.get("directional_accuracy_pct") is not None else "—",
            src,
        ),
        ("Composite Score", fmt_num(prod.get("composite_score")), src),
    ]


def _wf(doc: dict[str, Any]) -> dict[str, Any]:
    return doc.get("walk_forward") if isinstance(doc.get("walk_forward"), dict) else {}


def _wf_display(doc: dict[str, Any]) -> dict[str, Any]:
    wf = _wf(doc)
    disp = wf.get("display") if isinstance(wf.get("display"), dict) else {}
    return disp


def _dataset_build_snapshot(doc: dict[str, Any]) -> dict[str, Any]:
    from chain_replay_ml.training.dataset_build_snapshot import resolve_dataset_build_snapshot

    return resolve_dataset_build_snapshot(doc)


def _dataset_build_summary_rows(doc: dict[str, Any]) -> list[tuple[str, Any]]:
    snap = _dataset_build_snapshot(doc)
    if not snap:
        return []

    rows: list[tuple[str, Any]] = []
    if snap.get("export_source"):
        rows.append(("Dataset source", snap.get("export_source")))
    if snap.get("market"):
        rows.append(("Market", snap.get("market")))
    if snap.get("trading_days") is not None:
        rows.append(("Trading days", fmt_rows(snap.get("trading_days"))))
    sm = snap.get("selection_method")
    if isinstance(sm, dict):
        summary = str(sm.get("summary") or "").strip()
        if summary and summary != "—":
            day_count = snap.get("trading_days")
            selection = (
                f"{fmt_rows(day_count)} days"
                if day_count is not None
                else summary
            )
            rows.append(("Dataset selection", selection))
    seen: set[str] = set()
    for label, key in (
        ("Strike selection", "strike_selection_label"),
        ("Sampling", "sampling_label"),
    ):
        val = snap.get(key)
        if val and str(val) != "—":
            rows.append((label, val))
            seen.add(label)
    for item in snap.get("filter_summary") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip()
        # Overview shows trading-day count only; skip the full dates list.
        if not label or label in seen or label == "Trading dates":
            continue
        rows.append((label, item.get("value") or "—"))
        seen.add(label)
    if snap.get("created_at"):
        rows.append(("Dataset created", snap.get("created_at")))
    if snap.get("snapshotted_at"):
        rows.append(("Snapshot at train", snap.get("snapshotted_at")))
    return rows


def _composite_score_block(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (int, float)):
        return {"score": raw}
    return {}


def _composite_score_artifact(block: dict[str, Any]) -> str:
    parts = [str(block.get("source_file") or "").strip(), str(block.get("source_path") or "").strip()]
    joined = " ".join(part for part in parts if part)
    return joined or "—"


def composite_score_audit_view(comp: dict[str, Any]) -> dict[str, Any] | None:
    """Build display payload for the Composite Score Audit section."""
    if not isinstance(comp, dict) or not comp:
        return None

    best = _composite_score_block(comp.get("best_validation_composite"))
    prod = _composite_score_block(comp.get("production_composite"))
    best_score = best.get("score")
    prod_score = prod.get("score")
    if best_score is None and prod_score is None:
        return None

    prod_source = prod.get("source") or "Retrained production model evaluated on walk-forward aggregate"
    champion = prod.get("champion")
    if champion:
        prod_source = f"{prod_source} (champion: {champion})"

    table_rows = [
        (
            "Best Validation Composite",
            fmt_num(best_score) if best_score is not None else "—",
            best.get("source") or "Optuna validation during HPO",
            _composite_score_artifact(best),
        ),
        (
            "Production Composite",
            fmt_num(prod_score) if prod_score is not None else "—",
            prod_source,
            _composite_score_artifact(prod),
        ),
    ]

    footnote: str | None = None
    footnote_warn = False
    diff_abs = comp.get("difference_abs")
    diff_pct = comp.get("difference_pct")
    values_differ = comp.get("values_differ")
    if best_score is not None and prod_score is not None and diff_abs is not None:
        sign = "+" if float(diff_abs) >= 0 else ""
        pct_part = ""
        if diff_pct is not None:
            pct_sign = "+" if float(diff_pct) >= 0 else ""
            pct_part = f" ({pct_sign}{float(diff_pct):.1f}%)"
        if values_differ:
            footnote = (
                f"Difference: {sign}{fmt_num(diff_abs)}{pct_part}. "
                "This is expected: Best Validation Composite is used for hyperparameter selection during Optuna "
                "(fast trial evaluation). Production Composite measures the deployed champion model retrained "
                "with full trees and re-evaluated across all walk-forward folds."
            )
            footnote_warn = True
        else:
            footnote = (
                f"Difference: {sign}{fmt_num(diff_abs)}{pct_part}. "
                "Values match — production re-evaluation aligned with the best Optuna trial."
            )

    purpose_rows: list[tuple[str, str]] = []
    if best.get("purpose"):
        purpose_rows.append(("Best validation purpose", str(best["purpose"])))
    if prod.get("purpose"):
        purpose_rows.append(("Production purpose", str(prod["purpose"])))

    return {
        "table_rows": table_rows,
        "footnote": footnote,
        "footnote_warn": footnote_warn,
        "purpose_rows": purpose_rows,
    }


def _render_composite_score_audit(parent: tk.Misc, comp: dict[str, Any]) -> None:
    view = composite_score_audit_view(comp)
    if not view:
        return

    section_title(parent, "Composite Score Audit")
    section_desc(
        parent,
        "Compares the Optuna trial score used to pick hyperparameters against the final production "
        "walk-forward re-evaluation score.",
    )
    data_table(
        parent,
        [
            ("metric", "Metric", 180),
            ("value", "Value", 90),
            ("source", "Source", 220),
            ("artifact", "Artifact", 200),
        ],
        list(view["table_rows"]),
        height=2,
    )
    if view.get("purpose_rows"):
        kv_block(parent, "What each score means", view["purpose_rows"])
    footnote = view.get("footnote")
    if footnote:
        tk.Label(
            parent,
            text=footnote,
            font=BODY_FONT,
            fg=COL_WARN if view.get("footnote_warn") else COL_MUTED,
            wraplength=900,
            justify="left",
            anchor="w",
        ).pack(anchor="w", pady=(6, 0))


def _fold_placement_display(doc: dict[str, Any]) -> str | None:
    disp = _wf_display(doc)
    label = disp.get("fold_placement_label")
    if label:
        return str(label)
    placement = disp.get("fold_placement")
    if placement:
        key = str(placement).strip().lower()
        return "Distributed" if key == "distributed" else "Anchored"
    summary = _summary(doc)
    meta = summary.get("walk_forward_meta") if isinstance(summary.get("walk_forward_meta"), dict) else {}
    if meta.get("fold_placement_label"):
        return str(meta["fold_placement_label"])
    if meta.get("fold_placement"):
        key = str(meta["fold_placement"]).strip().lower()
        return "Distributed" if key == "distributed" else "Anchored"
    if summary.get("fold_placement_label"):
        return str(summary["fold_placement_label"])
    if summary.get("fold_placement"):
        key = str(summary["fold_placement"]).strip().lower()
        return "Distributed" if key == "distributed" else "Anchored"
    cfg = _cfg(doc)
    split = cfg.get("split") if isinstance(cfg.get("split"), dict) else {}
    wf_cfg = split.get("walk_forward") if isinstance(split.get("walk_forward"), dict) else {}
    if wf_cfg.get("fold_placement"):
        key = str(wf_cfg["fold_placement"]).strip().lower()
        return "Distributed" if key == "distributed" else "Anchored"
    return None


def _wf_config_values(doc: dict[str, Any]) -> dict[str, Any]:
    if not doc.get("is_walk_forward"):
        return {}
    disp = _wf_display(doc)
    summary = _summary(doc)
    meta = summary.get("walk_forward_meta") if isinstance(summary.get("walk_forward_meta"), dict) else {}
    return {
        "window_mode": disp.get("window_mode") or meta.get("window_mode") or summary.get("walk_forward_window_mode"),
        "n_folds": disp.get("n_folds") or meta.get("n_folds") or summary.get("walk_forward_n_folds"),
        "train_window_size": disp.get("train_window_size") or meta.get("train_window_size") or summary.get("walk_forward_train_window"),
        "validation_window_size": disp.get("validation_window_size") or meta.get("validation_window_size") or summary.get("walk_forward_validation_window"),
        "fold_placement": _fold_placement_display(doc) or disp.get("fold_placement"),
    }


def _wf_summary_rows(doc: dict[str, Any]) -> list[tuple[str, Any]]:
    cfg = _wf_config_values(doc)
    if not cfg:
        return []
    return [
        ("WF folds", cfg.get("n_folds")),
        ("Fold placement", cfg.get("fold_placement")),
    ]


def _wf_window_config_rows(doc: dict[str, Any]) -> list[tuple[str, Any]]:
    cfg = _wf_config_values(doc)
    if not cfg:
        return []
    return [
        ("WF window mode", cfg.get("window_mode")),
        ("WF train window", cfg.get("train_window_size")),
        ("WF validation window", cfg.get("validation_window_size")),
    ]


def _strat_label(doc: dict[str, Any]) -> str:
    summary = _summary(doc)
    if summary.get("validation_strategy_label"):
        return str(summary["validation_strategy_label"])
    wf_meta = summary.get("walk_forward_meta")
    if isinstance(wf_meta, dict) and wf_meta.get("validation_strategy_label"):
        return str(wf_meta["validation_strategy_label"])
    disp = _wf_display(doc)
    if disp.get("validation_strategy_label"):
        return str(disp["validation_strategy_label"])
    row = doc.get("table_row") or {}
    strat = doc.get("validation_strategy") or {}
    if row.get("validation_strategy"):
        return str(row["validation_strategy"])
    if isinstance(strat, dict):
        return str(strat.get("label") or "—")
    return str(strat or "—")


def _resolve_premium_bands(doc: dict[str, Any]) -> list[dict[str, Any]]:
    prod = _prod(doc)
    metrics = _metrics(doc)
    val = metrics.get("validation") if isinstance(metrics.get("validation"), dict) else {}
    test = metrics.get("test") if isinstance(metrics.get("test"), dict) else {}
    candidates = [
        prod.get("premium_band_performance"),
        (metrics.get("production_walk_forward") or {}).get("premium_band_performance"),
        (metrics.get("walk_forward") or {}).get("premium_band_performance"),
        val.get("premium_band_performance"),
        test.get("premium_band_performance"),
        metrics.get("test", {}).get("premium_band_performance") if isinstance(metrics.get("test"), dict) else None,
    ]
    for rows in candidates:
        if isinstance(rows, list) and rows:
            return rows
    return []


def model_feature_preset(doc: dict[str, Any]) -> tuple[list[str], str | None]:
    """Feature list + dataset for Create Model handoff."""
    feats = doc.get("selected_features")
    if isinstance(feats, list) and feats:
        names = [str(f).strip() for f in feats if str(f).strip()]
        if names:
            meta = _meta(doc)
            cfg = _cfg(doc)
            row = doc.get("table_row") or {}
            ds = meta.get("dataset") or cfg.get("dataset") or row.get("dataset")
            return names, str(ds).strip() if ds else None
    wf = _wf(doc)
    sel_art = wf.get("selected_features") or {}
    rows = sel_art.get("rows") if isinstance(sel_art.get("rows"), list) else []
    names: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        feat = str(row.get("feature") or row.get("Feature") or "").strip()
        if not feat:
            continue
        selected = str(row.get("selected") or row.get("Selected") or "").strip().lower()
        if selected in ("yes", "true", "1", "y") or not selected:
            names.append(feat)
    if names:
        meta = _meta(doc)
        cfg = _cfg(doc)
        row = doc.get("table_row") or {}
        ds = meta.get("dataset") or cfg.get("dataset") or row.get("dataset")
        return names, str(ds).strip() if ds else None
    return [], None


def _render_builder_features_button(
    parent: tk.Misc,
    doc: dict[str, Any],
    on_builder_features: Callable[[str, list[str], str | None], None] | None,
) -> None:
    if not on_builder_features:
        return
    feats, dataset = model_feature_preset(doc)
    model_name = str(doc.get("model_name") or "").strip()
    if not feats or not model_name:
        return
    row = ttk.Frame(parent)
    row.pack(fill="x", pady=(0, 8))
    ttk.Button(
        row,
        text=f"Open Create Model ({len(feats)} features)",
        command=lambda: on_builder_features(model_name, feats, dataset),
    ).pack(side="left")


def _render_overview_feature_selection(parent: tk.Misc, doc: dict[str, Any]) -> None:
    """Feature Selection identity — Analysis strategy story for production models."""
    from chain_replay_ml.dataset_builder.analysis_feature_selection import (
        extract_feature_selection_lineage,
        feature_selection_overview_rows,
    )

    section_title(parent, "Feature Selection")
    lineage = extract_feature_selection_lineage(doc) or extract_feature_selection_lineage(
        _cfg(doc)
    )
    rows = feature_selection_overview_rows(lineage)
    if not rows:
        feats, _dataset = model_feature_preset(doc)
        n = len(feats) if feats else (
            (_meta(doc).get("feature_count") or (_summary(doc).get("features")))
        )
        section_desc(
            parent,
            "This model was not created from an Analysis Feature Selection handoff.\n"
            "Feature Selection identity is recorded when you use ► Create Model Builder "
            "from Research Lab.",
        )
        inline_spec_rows(
            parent,
            [
                ("Source", "Model Builder"),
                ("Selected Features", n if n is not None else "—"),
            ],
            label_width=22,
        )
        if feats:
            btn_row = ttk.Frame(parent)
            btn_row.pack(fill="x", pady=(8, 0))
            ttk.Button(
                btn_row,
                text="View Features",
                command=lambda: _show_feature_list_dialog(
                    parent, feats, title="Selected Features"
                ),
            ).pack(side="left")
        return

    section_desc(
        parent,
        "How this production model's Final Feature Dataset was built in Analysis.",
    )
    inline_spec_rows(parent, rows, label_width=22)
    feats = [
        str(f).strip()
        for f in (lineage.get("features") or [])
        if str(f).strip()
    ]
    if not feats:
        feats, _ = model_feature_preset(doc)
    btn_row = ttk.Frame(parent)
    btn_row.pack(fill="x", pady=(10, 0))
    ttk.Label(btn_row, text="Feature Set", width=22, anchor="w").pack(side="left")
    ttk.Button(
        btn_row,
        text="View Features",
        command=lambda: _show_feature_list_dialog(
            parent,
            feats,
            title=(
                f"Feature Set · {lineage.get('n_selected_features') or len(feats)} features"
            ),
            summary=str(lineage.get("strategy_label") or ""),
        ),
    ).pack(side="left")


def _show_feature_list_dialog(
    parent: tk.Misc,
    features: list[str],
    *,
    title: str = "Features",
    summary: str = "",
) -> None:
    win = tk.Toplevel(parent)
    win.title(title)
    win.geometry("480x520")
    txt = tk.Text(win, wrap="word", font=("Consolas", 9))
    txt.pack(fill="both", expand=True, padx=8, pady=8)
    header = (summary.strip() + "\n\n") if summary.strip() else ""
    body = "\n".join(f"{i}. {f}" for i, f in enumerate(features, start=1))
    txt.insert("end", header + body if body else "(no features)")
    txt.configure(state="disabled")


def render_overview(
    scroll: ScrollableFrame,
    doc: dict[str, Any],
    *,
    on_builder_features: Callable[[str, list[str], str | None], None] | None = None,
    chart_dir: str | None = None,
) -> None:
    clear_children(scroll.inner)
    parent = scroll.inner
    meta = _meta(doc)
    cfg = _cfg(doc)
    summary = _summary(doc)
    row = doc.get("table_row") or {}
    prod = _prod(doc)
    metrics = _metrics(doc)
    prod_source = prod.get("source_path") or prod.get("source") or "walk_forward/champion_aggregate.json"

    split_row = ttk.Frame(parent)
    split_row.pack(fill="both", expand=True)
    split_row.columnconfigure(0, weight=1)
    split_row.columnconfigure(1, weight=1)
    split_row.rowconfigure(0, weight=1)

    summary_col = ttk.Frame(split_row)
    summary_col.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
    metrics_col = ttk.Frame(split_row)
    metrics_col.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

    nb = ttk.Notebook(summary_col)
    nb.pack(fill="both", expand=True)

    tab_summary = ttk.Frame(nb, padding=4)
    tab_feat_sel = ttk.Frame(nb, padding=4)
    tab_features = ttk.Frame(nb, padding=4)
    tab_fp = ttk.Frame(nb, padding=4)
    nb.add(tab_summary, text="Summary")
    if _is_triple_barrier_model(doc):
        tab_label_run = ttk.Frame(nb, padding=4)
        nb.add(tab_label_run, text="Label Run")
        _render_overview_label_run(tab_label_run, doc, chart_dir=chart_dir)
    nb.add(tab_feat_sel, text="Feature Selection")
    nb.add(tab_features, text="Top 20 Features")
    nb.add(tab_fp, text="Pipeline Fingerprint")

    # Phase 4C.4: Taxonomy & Context Champion Resolution
    tax_info = {}
    champ_for_ctx = "—"
    chall_for_ctx = "—"
    try:
        from chain_replay_ml.model_taxonomy import format_model_taxonomy_display
        from chain_replay_ml.training.lifecycle_store import get_champion_for_context
        data_source_dir = chart_dir or (doc.get("data_dir") if isinstance(doc.get("data_dir"), str) else "")
        tax_info = format_model_taxonomy_display(row or meta or cfg or doc)
        if data_source_dir and tax_info.get("context_key"):
            champ_doc = get_champion_for_context(data_source_dir, tax_info["context_key"])
            if champ_doc:
                champ_for_ctx = str(champ_doc.get("champion_model_name") or champ_doc.get("current_model_name") or "—")
                chall_for_ctx = str(champ_doc.get("challenger_model_name") or "—")
    except Exception:
        tax_info = {}

    inline_spec_rows(
        tab_summary,
        [
            ("Task Type", tax_info.get("task_label") or meta.get("task_type") or "Regression"),
            ("Market Regime", tax_info.get("regime_display") or "R000 — ALL_REGIMES"),
            ("Population Tier", tax_info.get("population_badge") or "EXPERIMENTAL"),
            ("Lifecycle Status", tax_info.get("lifecycle_label") or row.get("status") or "Active"),
            ("Context Key", tax_info.get("context_key") or "—"),
            ("Context Champion", champ_for_ctx),
            ("Context Challenger", chall_for_ctx),
            ("Validation strategy", _strat_label(doc)),
            ("Algorithm", meta.get("algorithm") or cfg.get("algorithm_label") or cfg.get("algorithm")),
            ("Dataset", meta.get("dataset") or cfg.get("dataset") or row.get("dataset")),
            ("Target", meta.get("target") or cfg.get("target") or row.get("target")),
            *(_dataset_build_summary_rows(doc)),
            *(_wf_summary_rows(doc)),
            ("Rows", fmt_rows(meta.get("row_count") or summary.get("rows") or row.get("rows"))),
            ("Selected features", meta.get("feature_count") or summary.get("features") or row.get("feature_count")),
            ("Trees trained", summary.get("trees_trained")),
            ("Early stopped", "Yes" if summary.get("early_stopped") else ("No" if summary.get("early_stopped") is False else "—")),
            ("Training time", f"{summary.get('training_time_sec')}s" if summary.get("training_time_sec") is not None else "—"),
            ("Total elapsed", f"{summary.get('total_elapsed_sec')}s" if summary.get("total_elapsed_sec") is not None else "—"),
            ("Trained at", meta.get("trained_at") or row.get("trained_at")),
            ("Status", row.get("status") or "ready"),
            ("Model version", meta.get("model_version") or cfg.get("model_version")),
            ("Description", meta.get("model_description") or cfg.get("model_description")),
            ("Strike selection", doc.get("strike_selection_label")),
            ("Sampling", doc.get("sampling_interval_label")),
        ],
        label_width=18,
    )

    _render_overview_feature_selection(tab_feat_sel, doc)

    section_desc(tab_features, "Gain importance from the deployed champion model (feature_importance.csv).")
    _render_builder_features_button(tab_features, doc, on_builder_features)
    importance_list(tab_features, doc.get("feature_importance") or [], limit=20)

    fp = doc.get("pipeline_fingerprint") if isinstance(doc.get("pipeline_fingerprint"), dict) else {}
    if fp:
        spec_grid(
            tab_fp,
            [
                ("Dataset version", fp.get("dataset_version")),
                ("Builder version", fp.get("builder_version")),
                ("Feature registry", fp.get("feature_registry_version")),
                ("Schema hash", fp.get("schema_registry_hash") or fp.get("implementation_hash")),
                ("Sampling", fp.get("sampling") or fp.get("sampling_interval_sec")),
            ],
        )
    else:
        ttk.Label(tab_fp, text="Not available", foreground=COL_MUTED).pack(anchor="w")

    section_title(metrics_col, "Production Metrics")
    section_desc(metrics_col, f"Source: {prod_source}")
    classification = _is_classification_model(doc)
    if classification:
        _render_classification_production_metrics(metrics_col, prod)
    else:
        style = ttk.Style(metrics_col)
        # +25% vs prior compact sizing (rowheight 12 → 15, rows 3/5 → 4/6)
        style.configure("ProdMetrics.Treeview", font=("Segoe UI", 7), rowheight=15)
        style.configure("ProdMetrics.Treeview.Heading", font=("Segoe UI", 7, "bold"))
        data_table(
            metrics_col,
            [
                ("metric", "Metric", 70),
                ("value", "Value", 50),
                ("source", "Source", 60),
            ],
            _production_metrics_table_rows(prod, classification=False),
            height=4,
            expand=False,
            style="ProdMetrics.Treeview",
        )

    comp = metrics.get("composite_scores") if isinstance(metrics.get("composite_scores"), dict) else {}
    if comp:
        _render_composite_score_audit(metrics_col, comp)


def _render_overview_label_run(
    parent: tk.Misc,
    doc: dict[str, Any],
    *,
    chart_dir: str | None = None,
) -> None:
    """Rich Label Run metadata for Triple Barrier models (Overview → Label Run)."""
    run_id = _resolve_label_run_id(doc)
    row = doc.get("table_row") if isinstance(doc.get("table_row"), dict) else {}
    status_hint = str(row.get("label_run_status") or "").strip().lower()

    section_title(parent, "Label Run")
    section_desc(
        parent,
        "Immutable labeling artifact joined to the Feature Dataset at train time. "
        "Strategy parameters below are from the run — not editable on the model.",
    )

    if not run_id:
        ttk.Label(
            parent,
            text="No Label Run linked to this model (label_run_id missing).",
            foreground=COL_WARN,
        ).pack(anchor="w", pady=8)
        return

    data_dir = None
    if chart_dir:
        try:
            from .build_service import chart_data_dir

            data_dir = chart_data_dir(chart_dir)
        except Exception:
            data_dir = None

    meta: dict[str, Any] = {}
    exists = False
    load_error: str | None = None
    if data_dir and run_id:
        try:
            from chain_replay_ml.label_runs import get_label_run, load_label_run_meta

            rec = get_label_run(data_dir, run_id)
            exists = bool(rec.exists)
            if exists:
                meta = load_label_run_meta(data_dir, run_id)
            else:
                load_error = "Label Run files are missing on disk (Deleted)."
        except Exception as exc:
            load_error = str(exc)

    status_label = "Available" if exists else ("Deleted" if status_hint == "deleted" or load_error else "Unknown")
    status_color = COL_OK if exists else COL_WARN

    header = ttk.Frame(parent)
    header.pack(fill="x", pady=(4, 8))
    ttk.Label(header, text=run_id, font=("Segoe UI", 10, "bold"), foreground=ACCENT).pack(
        side="left", anchor="w"
    )
    ttk.Label(header, text=f"  ·  {status_label}", foreground=status_color, font=("Segoe UI", 9, "bold")).pack(
        side="left", anchor="w"
    )

    if load_error and not meta:
        ttk.Label(parent, text=load_error, foreground=COL_WARN, wraplength=520).pack(anchor="w", pady=4)
        inline_spec_rows(
            parent,
            [
                ("Label Run ID", run_id),
                ("Model link status", status_label),
            ],
            label_width=18,
        )
        return

    params = meta.get("parameters") if isinstance(meta.get("parameters"), dict) else {}
    encoding = meta.get("label_encoding") if isinstance(meta.get("label_encoding"), dict) else {}
    join_keys = meta.get("join_keys") if isinstance(meta.get("join_keys"), list) else []
    days = params.get("days") if isinstance(params.get("days"), list) else []

    rows_n = int(meta.get("rows") or 0)
    valid_n = int(meta.get("valid_rows") or 0)
    invalid_n = int(meta.get("invalid_rows") or 0)
    valid_pct = (100.0 * valid_n / rows_n) if rows_n > 0 else None

    barrier = str(params.get("barrier_type") or "").strip() or "—"
    unit = "%" if barrier == "percentage" else ("pts" if barrier == "points" else "")
    tp = params.get("tp_value")
    sl = params.get("sl_value")
    tp_txt = f"{tp}{unit}" if tp is not None else "—"
    sl_txt = f"{sl}{unit}" if sl is not None else "—"

    section_title(parent, "Barrier parameters")
    inline_spec_rows(
        parent,
        [
            ("Barrier type", barrier),
            ("Take profit (TP)", tp_txt),
            ("Stop loss (SL)", sl_txt),
            ("Holding seconds", params.get("holding_seconds")),
            ("Truncate at close", "Yes" if params.get("truncate_at_close") else ("No" if "truncate_at_close" in params else "—")),
            ("Max path gap sec", params.get("max_path_gap_sec") if params.get("max_path_gap_sec") is not None else "—"),
            ("Path source", params.get("path_source") or "—"),
            ("Trading days", f"{len(days)} day(s)" if days else "—"),
        ],
        label_width=18,
    )
    if days:
        section_desc(parent, "Days: " + ", ".join(str(d) for d in days[:24]) + ("…" if len(days) > 24 else ""))

    section_title(parent, "Identity")
    inline_spec_rows(
        parent,
        [
            ("Run ID", meta.get("run_id") or run_id),
            ("Strategy", meta.get("strategy") or "—"),
            ("Strategy version", meta.get("strategy_version") or "—"),
            ("Engine version", meta.get("engine_version") or "—"),
            ("Created at", meta.get("created_at") or "—"),
            ("Artifact kind", meta.get("artifact_kind") or "label_run"),
            ("Schema version", meta.get("schema_version")),
        ],
        label_width=18,
    )

    section_title(parent, "Dataset linkage")
    inline_spec_rows(
        parent,
        [
            ("Feature dataset", meta.get("dataset_id") or "—"),
            ("Dataset hash", meta.get("dataset_hash") or "—"),
            ("Join keys", ", ".join(str(k) for k in join_keys) if join_keys else "—"),
            ("Primary target", meta.get("primary_target") or "label_id"),
            ("Display target", meta.get("display_target") or "—"),
        ],
        label_width=18,
    )

    section_title(parent, "Coverage")
    inline_spec_rows(
        parent,
        [
            ("Rows", fmt_rows(rows_n)),
            ("Valid rows", fmt_rows(valid_n)),
            ("Invalid rows", fmt_rows(invalid_n)),
            ("Valid %", f"{valid_pct:.2f}%" if valid_pct is not None else "—"),
        ],
        label_width=18,
    )

    if encoding:
        section_title(parent, "Label encoding")
        enc_rows = [
            (str(name), str(code))
            for name, code in sorted(
                encoding.items(),
                key=lambda kv: int(kv[1]) if str(kv[1]).lstrip("-").isdigit() else 0,
            )
        ]
        data_table(
            parent,
            [("name", "Label", 100), ("id", "label_id", 80)],
            enc_rows,
            height=min(6, max(3, len(enc_rows) + 1)),
            expand=False,
        )

    section_title(parent, "Artifacts")
    try:
        from chain_replay_ml.label_runs.paths import label_run_meta_path, label_run_parquet_path

        pq = label_run_parquet_path(data_dir, run_id) if data_dir else "—"
        mj = label_run_meta_path(data_dir, run_id) if data_dir else "—"
    except Exception:
        pq, mj = "—", "—"
    inline_spec_rows(
        parent,
        [
            ("Parquet", pq),
            ("Meta JSON", mj),
        ],
        label_width=18,
    )

    section_title(parent, "Raw metadata")
    json_block(parent, meta, height=12)


def render_model_metrics(scroll: ScrollableFrame, doc: dict[str, Any]) -> None:
    clear_children(scroll.inner)
    parent = scroll.inner
    prod = _prod(doc)
    prod_source = prod.get("source_path") or prod.get("source") or "walk_forward/champion_aggregate.json"
    classification = _is_classification_model(doc)

    section_title(parent, "Model Metrics")
    section_desc(parent, f"Source: {prod_source}")

    if classification:
        _render_classification_production_metrics(parent, prod)
        return

    top_row = ttk.Frame(parent)
    top_row.pack(fill="both", expand=True)
    top_row.columnconfigure(0, weight=1)
    top_row.columnconfigure(1, weight=3)
    top_row.rowconfigure(0, weight=1)

    left_col = ttk.Frame(top_row, padding=(0, 4))
    left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
    right_col = ttk.Frame(top_row, padding=(0, 4))
    right_col.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

    kv_block(
        left_col,
        "Regression",
        [
            ("MAE", fmt_rupee(prod.get("mae"))),
            ("RMSE", fmt_rupee(prod.get("rmse"))),
            ("Premium MAE", fmt_pct(prod.get("premium_mae_pct"))),
            ("Premium RMSE", fmt_pct(prod.get("premium_rmse_pct"))),
            ("Median Absolute Error", fmt_rupee(prod.get("medae") or prod.get("median_error"))),
            ("95th Percentile Error", fmt_rupee(prod.get("p95_error"))),
            ("Prediction Bias", fmt_signed_rupee(prod.get("prediction_bias"))),
            ("Prediction Bias %", fmt_signed_pct(prod.get("prediction_bias_pct"))),
        ],
    )

    bands = _resolve_premium_bands(doc)
    section_title(right_col, "Premium Band Performance")
    if not bands:
        section_desc(right_col, "Not available — retrain to compute premium-normalized band metrics.")
    else:
        rows = []
        for r in bands:
            if not isinstance(r, dict):
                continue
            label = r.get("band_label") or (f"₹{r['band']}" if r.get("band") else "—")
            rows.append((
                label,
                fmt_rows(r.get("samples")),
                fmt_num(r.get("mae"), 2),
                fmt_num(r.get("rmse"), 2),
                fmt_pct(r.get("premium_mae_pct"), 1),
                fmt_pct(r.get("premium_rmse_pct"), 1),
                fmt_pct(r.get("directional_accuracy_pct"), 1) if r.get("directional_accuracy_pct") is not None else "—",
            ))
        data_table(
            right_col,
            [
                ("band", "Band", 80),
                ("samples", "Samples", 70),
                ("mae", "MAE", 70),
                ("rmse", "RMSE", 70),
                ("mae_pct", "MAE%", 60),
                ("rmse_pct", "RMSE%", 70),
                ("dir", "Direction", 80),
            ],
            rows,
            height=min(8, len(rows) + 1),
        )

    kv_block(
        parent,
        "Classification",
        [
            ("Direction Accuracy", fmt_pct(prod.get("directional_accuracy_pct"))),
            ("Composite Score", fmt_num(prod.get("composite_score"))),
        ],
    )


def render_classification_confusion(
    scroll: ScrollableFrame,
    doc: dict[str, Any],
) -> None:
    """Standalone confusion-matrix page for a classification package member."""
    clear_children(scroll.inner)
    parent = scroll.inner
    if not _is_classification_model(doc):
        section_title(parent, "Confusion Matrix")
        section_desc(parent, "Available for classification models only.")
        return
    conf = _confusion_from_prod(_prod(doc))
    if conf is None:
        section_title(parent, "Confusion Matrix")
        section_desc(
            parent,
            "Not available yet — retrain this classifier to persist Hit/Miss counts.",
        )
        return
    _render_confusion_matrix(parent, conf)


def render_threshold_analysis(
    scroll: ScrollableFrame,
    doc: dict[str, Any],
    *,
    chart_dir: str | None = None,
) -> None:
    """Classification Probability Threshold sweep — Precision / Recall / F1 / BUY Signals."""
    clear_children(scroll.inner)
    parent = scroll.inner
    section_title(parent, "Threshold Analysis")
    if not _is_classification_model(doc):
        section_desc(parent, "Available for binary / classification models only.")
        return

    prod = _prod(doc)
    try:
        default_thr = round(float(prod.get("threshold") if prod.get("threshold") is not None else 0.5), 2)
    except (TypeError, ValueError):
        default_thr = 0.5

    rows, source = resolve_threshold_analysis_rows(doc, chart_dir=chart_dir)
    section_desc(
        parent,
        "How Precision / Recall / F1 and BUY signal volume change as you raise the "
        f"decision threshold. ● = current Decision Threshold ({default_thr:.2f}).  "
        f"Source: {source}.",
    )
    if not rows:
        section_desc(
            parent,
            "Not available yet — retrain this classification model (or run walk-forward "
            "with Prediction Runs) to persist the threshold sweep "
            "(0.30 → 0.90).",
        )
        return

    table_rows: list[tuple[Any, ...]] = []
    day_count: int | None = None
    for row in rows:
        thr = row.get("threshold")
        mark = ""
        try:
            if thr is not None and abs(float(thr) - default_thr) < 1e-9:
                mark = " ●"
        except (TypeError, ValueError):
            pass
        buy = row.get("buy_signals")
        if buy is None:
            buy = row.get("predicted_positives")
        hit = row.get("hit_rate_pct")
        if hit is None:
            hit = row.get("precision_pct")
        fp = row.get("false_positives")
        if fp is None:
            fp = row.get("fp")
        fn = row.get("false_negatives")
        if fn is None:
            fn = row.get("fn")
        tpd = row.get("trades_per_day")
        if day_count is None and row.get("n_days") is not None:
            try:
                day_count = int(row.get("n_days"))
            except (TypeError, ValueError):
                day_count = None
        table_rows.append(
            (
                f"{float(thr):.2f}{mark}" if thr is not None else "—",
                fmt_pct(row.get("precision_pct")),
                fmt_pct(row.get("recall_pct")),
                fmt_pct(row.get("f1_pct")),
                fmt_pct(row.get("accuracy_pct")),
                _fmt_int_count(buy),
                _fmt_trades_per_day(tpd),
                fmt_pct(hit),
                _fmt_int_count(fp),
                _fmt_int_count(fn),
            )
        )

    data_table(
        parent,
        [
            ("thr", "Threshold", 80),
            ("precision", "Precision", 80),
            ("recall", "Recall", 70),
            ("f1", "F1", 70),
            ("accuracy", "Accuracy", 80),
            ("buy", "BUY Signals", 90),
            ("tpd", "Trades/Day", 90),
            ("hit", "Hit Rate", 80),
            ("fp", "False Positives", 100),
            ("fn", "False Negatives", 110),
        ],
        table_rows,
        height=min(12, max(4, len(table_rows) + 1)),
    )
    day_note = f" over {day_count} trading day(s)" if day_count else ""
    ttk.Label(
        parent,
        text=(
            "Hit Rate = Precision (wins among BUY signals).  "
            f"Trades/Day = BUY Signals ÷ trading days{day_note}.  "
            "Higher threshold → fewer BUY Signals, usually higher Precision, lower Recall."
        ),
        foreground=COL_MUTED,
        font=("Segoe UI", 8),
        wraplength=720,
    ).pack(anchor="w", pady=(0, 8))


def render_validation_metrics(scroll: ScrollableFrame, doc: dict[str, Any]) -> None:
    clear_children(scroll.inner)
    parent = scroll.inner
    is_wf = bool(doc.get("is_walk_forward"))
    metrics = _metrics(doc)
    prod = _prod(doc)
    val = metrics.get("validation") if isinstance(metrics.get("validation"), dict) else {}
    test = metrics.get("test") if isinstance(metrics.get("test"), dict) else {}
    train_meta = doc.get("training_metadata") if isinstance(doc.get("training_metadata"), dict) else {}
    summary = _summary(doc)
    prod_source = prod.get("source_path") or prod.get("source") or "walk_forward/champion_aggregate.json"

    section_title(parent, "Evaluation stages")

    stages = ttk.Frame(parent)
    stages.pack(fill="x")

    val_metrics = [
        ("MAE (₹)", fmt_rupee(val.get("mae"))),
        ("RMSE (₹)", fmt_rupee(val.get("rmse"))),
        ("Premium MAE (%)", fmt_pct(val.get("premium_mae_pct"))),
        ("Premium RMSE (%)", fmt_pct(val.get("premium_rmse_pct"))),
        ("Validation R²", fmt_num(val.get("r2")) if val.get("r2") is not None else "—"),
        ("Validation MAPE", fmt_num(val.get("mape")) if val.get("mape") is not None else "—"),
        ("Val Directional Acc.", fmt_pct(val.get("directional_accuracy_pct")) if val.get("directional_accuracy_pct") is not None else "—"),
        ("Best Iteration", fmt_val(train_meta.get("best_iteration") or summary.get("best_iteration"))),
        ("Early Stopping Round", fmt_val(train_meta.get("early_stopping_rounds") or summary.get("early_stopping_rounds"))),
        ("Training RMSE", fmt_val(train_meta.get("train_rmse") or summary.get("train_rmse"))),
    ]

    metrics_stage(
        stages,
        "Training Validation",
        "Used during model training and early stopping."
        + (" For walk-forward, reflects the champion candidate." if is_wf else ""),
        val_metrics,
        color=COL_TRAINING,
    )

    if is_wf:
        if _is_classification_model(doc):
            prod_metrics = [
                ("Positive Class Rate", fmt_pct(prod.get("positive_rate_pct")) if prod.get("positive_rate_pct") is not None else "—"),
                ("Predicted BUY Rate", fmt_pct(prod.get("predicted_positive_rate_pct")) if prod.get("predicted_positive_rate_pct") is not None else "—"),
                ("Decision Threshold", f"{float(prod.get('threshold') if prod.get('threshold') is not None else 0.5):.2f}"),
                ("Precision", fmt_pct(prod.get("precision_pct")) if prod.get("precision_pct") is not None else "—"),
                ("Recall", fmt_pct(prod.get("recall_pct")) if prod.get("recall_pct") is not None else "—"),
                ("F1 Score", fmt_pct(prod.get("f1_pct")) if prod.get("f1_pct") is not None else "—"),
                ("ROC-AUC", _fmt_auc(prod.get("roc_auc"))),
                ("Accuracy", fmt_pct(prod.get("accuracy_pct")) if prod.get("accuracy_pct") is not None else "—"),
                ("Specificity", fmt_pct(prod.get("specificity_pct")) if prod.get("specificity_pct") is not None else "—"),
                ("Composite Score", fmt_num(prod.get("composite_score")) if prod.get("composite_score") is not None else "—"),
            ]
        else:
            prod_metrics = [
                ("MAE (₹)", fmt_rupee(prod.get("mae"))),
                ("RMSE (₹)", fmt_rupee(prod.get("rmse"))),
                ("Premium MAE (%)", fmt_pct(prod.get("premium_mae_pct"))),
                ("Premium RMSE (%)", fmt_pct(prod.get("premium_rmse_pct"))),
                ("Direction Accuracy", fmt_pct(prod.get("directional_accuracy_pct")) if prod.get("directional_accuracy_pct") is not None else "—"),
                ("Composite Score", fmt_num(prod.get("composite_score")) if prod.get("composite_score") is not None else "—"),
            ]
        metrics_stage(
            stages,
            "Production Performance",
            f"Deployed champion model after final retrain. Source: {prod_source}.",
            prod_metrics,
            color=COL_PRODUCTION,
            footnote="Production Composite uses the retrained champion WF aggregate.",
        )

    test_metrics = [
        ("MAE (₹)", fmt_rupee(test.get("mae"))),
        ("RMSE (₹)", fmt_rupee(test.get("rmse"))),
        ("Premium MAE (%)", fmt_pct(test.get("premium_mae_pct"))),
        ("Premium RMSE (%)", fmt_pct(test.get("premium_rmse_pct"))),
        ("Test R²", fmt_num(test.get("r2")) if test.get("r2") is not None else "—"),
        ("Test MAPE", fmt_num(test.get("mape")) if test.get("mape") is not None else "—"),
        ("Test Directional Acc.", fmt_pct(test.get("directional_accuracy_pct")) if test.get("directional_accuracy_pct") is not None else "—"),
    ]
    metrics_stage(
        stages,
        "Holdout Test",
        "Final holdout test set — estimates generalization on unseen data.",
        test_metrics,
        color=COL_HOLDOUT,
        footnote="Test MAPE is unreliable for low-premium OTM options. Prefer RMSE, MAE, and Direction Accuracy."
        if test.get("mape") is not None else None,
    )

    if metrics.get("walk_forward_summary"):
        section_title(parent, "Walk-forward summary (metrics.json)")
        json_block(parent, metrics.get("walk_forward_summary"), height=8)


def render_walk_forward(scroll: ScrollableFrame, doc: dict[str, Any]) -> None:
    clear_children(scroll.inner)
    parent = scroll.inner
    wf = _wf(doc)
    metrics = _metrics(doc)
    val = metrics.get("validation") if isinstance(metrics.get("validation"), dict) else {}
    disp = _wf_display(doc)
    elim = doc.get("feature_elimination") if isinstance(doc.get("feature_elimination"), dict) else {}
    wf_summary_art = wf.get("summary") or {}
    wf_summary = wf_summary_art.get("data") if isinstance(wf_summary_art.get("data"), dict) else {}
    fs_meta = wf_summary.get("feature_selection") if isinstance(wf_summary.get("feature_selection"), dict) else {}

    section_desc(
        parent,
        "Detailed walk-forward configuration, per-fold metrics, and artifact exports. "
        "Production numbers in the Registry come from the aggregated champion summary.",
    )

    row = ttk.Frame(parent)
    row.pack(fill="x")
    left = ttk.Frame(row)
    left.pack(side="left", fill="both", expand=True, padx=(0, 8))
    right = ttk.Frame(row)
    right.pack(side="left", fill="both", expand=True)

    section_title(left, "Walk Forward Summary", color=ACCENT)
    wf_cfg = _wf_config_values(doc)
    inline_spec_rows(
        left,
        [
            ("Validation strategy", _strat_label(doc)),
            ("Number of folds", wf_cfg.get("n_folds") or disp.get("n_folds")),
            *(_wf_window_config_rows(doc)),
            ("Fold placement", wf_cfg.get("fold_placement")),
            ("Mean Validation RMSE", val.get("rmse") or disp.get("mean_validation_rmse")),
            ("Std Validation RMSE", val.get("std_rmse") or disp.get("std_validation_rmse")),
            ("Mean Validation MAE", val.get("mae") or disp.get("mean_validation_mae")),
            ("Std Validation MAE", val.get("std_mae") or disp.get("std_validation_mae")),
            ("Mean Directional Accuracy", val.get("directional_accuracy_pct") or disp.get("mean_directional_accuracy_pct")),
            ("Optimization Metric", disp.get("optimization_metric")),
            ("Feature Elimination Method", elim.get("method_label") or elim.get("method") or fs_meta.get("method")),
            ("Features Before Elimination", elim.get("started_features") or fs_meta.get("started_features")),
            ("Features After Elimination", elim.get("finished_features") or fs_meta.get("finished_features")),
            ("HPO enabled", "Yes" if disp.get("hyperparameter_optimization_enabled") else ("No" if disp.get("hyperparameter_optimization_enabled") is False else "—")),
            ("Optuna trials", disp.get("hpo_n_trials")),
            ("Best composite score", disp.get("best_composite_score")),
        ],
        label_width=22,
    )

    section_title(right, "Fold Metrics", color=ACCENT)
    champ = wf.get("champion_aggregate") or {}
    champ_data = champ.get("data") if isinstance(champ.get("data"), dict) else {}
    folds = champ_data.get("fold_results") if isinstance(champ_data.get("fold_results"), list) else []
    if not folds and isinstance(wf_summary.get("fold_results"), list):
        folds = wf_summary["fold_results"]
    if champ_data.get("fold_results"):
        section_desc(right, "Production champion re-evaluation folds")
    elif folds:
        section_desc(right, "Initial walk-forward folds (summary.json)")

    from chain_replay_ml.training.fold_comparison import model_fold_metrics_table

    data_dir = doc.get("_data_dir") if isinstance(doc.get("_data_dir"), str) else None
    fold_table = model_fold_metrics_table(doc, data_dir=data_dir)
    if fold_table.get("hit_note"):
        section_desc(right, str(fold_table["hit_note"]))
    is_cls = bool(fold_table.get("is_classification")) or _is_classification_model(doc)
    fold_rows = []
    for r in fold_table.get("rows") or []:
        if not isinstance(r, dict):
            continue
        if is_cls:
            fold_rows.append((
                r.get("fold", "—"),
                fmt_pct(r.get("accuracy_pct")) if r.get("accuracy_pct") is not None else "—",
                fmt_pct(r.get("precision_pct")) if r.get("precision_pct") is not None else "—",
                fmt_pct(r.get("recall_pct")) if r.get("recall_pct") is not None else "—",
                fmt_pct(r.get("f1_pct")) if r.get("f1_pct") is not None else "—",
                _fmt_auc(r.get("roc_auc")),
                fmt_num(r.get("composite_score"), 4),
                fmt_val(r.get("feature_count") if r.get("feature_count") is not None else r.get("trees_trained")),
            ))
        else:
            fold_rows.append((
                r.get("fold", "—"),
                fmt_rows(r.get("validation_rows")),
                r.get("validation_days_label") or "—",
                fmt_pct(r.get("endpoint_hit_pct") if r.get("endpoint_hit_pct") is not None else r.get("target_hit_pct"))
                if (r.get("endpoint_hit_pct") if r.get("endpoint_hit_pct") is not None else r.get("target_hit_pct")) is not None
                else "—",
                fmt_num(r.get("rmse"), 4),
                fmt_num(r.get("mae"), 4),
                fmt_num(r.get("r2"), 4),
                fmt_num(r.get("mape"), 4),
                fmt_pct(r.get("directional_accuracy_pct")) if r.get("directional_accuracy_pct") is not None else "—",
                fmt_num(r.get("composite_score"), 4),
                fmt_val(r.get("trees_trained")),
            ))
    if fold_rows:
        if is_cls:
            cols = [
                ("fold", "Fold", 40),
                ("acc", "Accuracy", 70),
                ("prec", "Precision", 70),
                ("rec", "Recall", 60),
                ("f1", "F1", 55),
                ("auc", "AUC", 55),
                ("comp", "Composite", 70),
                ("feats", "Feats", 50),
            ]
        else:
            cols = [
                ("fold", "Fold", 40),
                ("vrows", "Val Rows", 65),
                ("vdays", "Validation Days", 120),
                ("hit", "Endpoint Hit %", 95),
                ("rmse", "RMSE", 60),
                ("mae", "MAE", 60),
                ("r2", "R²", 50),
                ("mape", "MAPE", 55),
                ("dir", "Direction", 70),
                ("comp", "Composite", 70),
                ("trees", "Trees", 50),
            ]
        data_table(
            right,
            cols,
            fold_rows,
            height=min(12, len(fold_rows) + 1),
        )
    else:
        ttk.Label(right, text="Not available", foreground=COL_MUTED).pack(anchor="w")


def render_selected_features(
    scroll: ScrollableFrame,
    doc: dict[str, Any],
    *,
    on_builder_features: Callable[[str, list[str], str | None], None] | None = None,
    chart_dir: str | None = None,
    tk_root: tk.Misc | None = None,
) -> None:
    clear_children(scroll.inner)
    parent = scroll.inner
    wf = _wf(doc)
    sel_art = wf.get("selected_features") or {}
    rows = sel_art.get("rows") if isinstance(sel_art.get("rows"), list) else []

    section_title(parent, "Walk-Forward Feature Selection Results")
    section_desc(
        parent,
        "Feature rankings from walk-forward RFE and fold aggregation (walk_forward/selected_features.csv).",
    )
    _render_builder_features_button(parent, doc, on_builder_features)
    _render_start_research_button(parent, doc, chart_dir=chart_dir, tk_root=tk_root)
    if not rows:
        ttk.Label(parent, text="Not available", foreground=COL_MUTED).pack(anchor="w")
        return

    table_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        feat = row.get("feature") or row.get("Feature") or ""
        if not feat:
            continue
        rank = row.get("final_rank") or row.get("Final Rank")
        gain = row.get("gain_importance_pct") or row.get("Gain Importance %")
        table_rows.append((
            feat,
            fmt_val(rank),
            fmt_num(gain, 2) if gain not in (None, "") else "—",
            row.get("selected_in_folds") or row.get("Selected in Folds") or "—",
            row.get("selected") or row.get("Selected") or "—",
        ))
    data_table(
        parent,
        [
            ("feature", "Feature", 220),
            ("rank", "Final Rank", 70),
            ("gain", "WF Selection Gain %", 120),
            ("folds", "Selected in Folds", 110),
            ("sel", "Selected", 70),
        ],
        table_rows,
        height=min(16, len(table_rows) + 1),
    )


def _render_start_research_button(
    parent: tk.Misc,
    doc: dict[str, Any],
    *,
    chart_dir: str | None,
    tk_root: tk.Misc | None,
) -> None:
    model_name = str(doc.get("model_name") or "").strip()
    if not model_name or not chart_dir or tk_root is None:
        return
    hint = "Opens Research Lab (Model Lab) for this frozen model"
    try:
        from chain_replay_ml.model_lab import find_latest_lab

        lab = find_latest_lab(model_name)
        if lab is not None:
            ov = getattr(lab, "_prediction_overview", None) or {}
            rows = int(ov.get("prediction_row_count") or 0)
            pred = str(ov.get("prediction_dataset_status") or "")
            if rows > 0:
                hint = f"Lab v{lab.version} ready · {rows:,} prediction rows ({pred or 'ready'})"
            else:
                hint = f"Lab v{lab.version} · {lab.status} · prediction dataset not generated yet"
        else:
            hint = "No Research Lab yet for this model — click to create"
    except Exception:
        pass
    row = ttk.Frame(parent)
    row.pack(fill="x", pady=(0, 8))
    ttk.Button(
        row,
        text="Start Research",
        command=lambda: _open_start_research(tk_root, chart_dir=chart_dir, doc=doc),
    ).pack(side="left")
    ttk.Label(
        row,
        text=hint,
        foreground=COL_MUTED,
    ).pack(side="left", padx=8)


def _open_start_research(tk_root: tk.Misc, *, chart_dir: str, doc: dict[str, Any]) -> None:
    from .model_lab_window import open_model_lab_window

    model_name = str(doc.get("model_name") or "").strip()
    if not model_name:
        messagebox.showwarning("Start Research", "Select a model first.")
        return
    try:
        open_model_lab_window(
            tk_root,
            chart_dir=chart_dir,
            model_name=model_name,
            detail_doc=doc,
            ensure_lab=True,
            initial_tab="prediction",
        )
    except Exception as exc:
        messagebox.showerror("Start Research", str(exc), parent=tk_root)


def _compute_hpo_status(doc: dict[str, Any]) -> dict[str, Any]:
    cfg = _cfg(doc)
    split = cfg.get("split") if isinstance(cfg.get("split"), dict) else {}
    wf_cfg = split.get("walk_forward") if isinstance(split.get("walk_forward"), dict) else {}
    hpo_cfg = split.get("hyperparameter_optimization") if isinstance(split.get("hyperparameter_optimization"), dict) else {}
    if not hpo_cfg and isinstance(wf_cfg.get("hyperparameter_optimization"), dict):
        hpo_cfg = wf_cfg["hyperparameter_optimization"]

    metrics = _metrics(doc)
    opt = metrics.get("optimization_result") if isinstance(metrics.get("optimization_result"), dict) else {}
    wf = _wf(doc)
    bp_art = wf.get("best_parameters") or {}
    bp = bp_art.get("data") if isinstance(bp_art.get("data"), dict) else (bp_art if isinstance(bp_art, dict) else {})

    trials = int(
        bp.get("n_trials_completed")
        or bp.get("n_trials_target")
        or (metrics.get("hyperparameter_optimization") or {}).get("n_trials")
        or 0
    )
    hpo_enabled = hpo_cfg.get("enabled") is True or opt.get("enabled") is True
    performed = bool(hpo_enabled and trials > 0)

    prod = _prod(doc)
    composite = prod.get("composite_score")
    if composite is None:
        comp_scores = metrics.get("composite_scores") if isinstance(metrics.get("composite_scores"), dict) else {}
        composite = comp_scores.get("production_composite")

    params = cfg.get("parameters") if isinstance(cfg.get("parameters"), dict) else {}
    if not params:
        params = bp.get("base_parameters") or bp.get("best_parameters") or bp.get("full_parameters") or {}

    trial_summary = bp.get("trial_summary") if isinstance(bp.get("trial_summary"), dict) else {}
    best_trial = trial_summary.get("best_trial")
    if best_trial is None:
        best_trial = bp.get("best_trial_number")

    return {
        "performed": performed,
        "status_label": "Completed" if performed else "Not Performed",
        "best_composite": composite,
        "best_trial": best_trial,
        "baseline_parameters": params if isinstance(params, dict) else {},
    }


def _param_preview(params: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in ("max_depth", "learning_rate", "subsample"):
        val = params.get(key)
        if val is None:
            continue
        if key == "learning_rate":
            try:
                parts.append(f"learning rate = {float(val):.4f}")
            except (TypeError, ValueError):
                parts.append(f"learning rate = {val}")
        else:
            parts.append(f"{key.replace('_', ' ')} = {val}")
    return " · ".join(parts) if parts else "—"


def render_retrain(
    scroll: ScrollableFrame,
    doc: dict[str, Any],
    *,
    on_lifecycle: Callable[[str, str], None] | None = None,
) -> None:
    from tkinter import messagebox

    clear_children(scroll.inner)
    parent = scroll.inner
    model_name = str(doc.get("model_name") or "").strip()
    hpo = _compute_hpo_status(doc)
    performed = bool(hpo.get("performed"))
    params = hpo.get("baseline_parameters") or {}
    param_preview = _param_preview(params)

    def _lifecycle(mode: str) -> None:
        if not model_name:
            messagebox.showinfo("Model Builder", "No model selected.")
            return
        if mode == "calibration_only":
            messagebox.showinfo("Calibration Only", "Calibration-only is not implemented yet.")
            return
        if on_lifecycle:
            on_lifecycle(model_name, mode)
        else:
            messagebox.showinfo(
                "Model Builder",
                "Open Create Model from the main navigation to run this action.",
            )

    def _view_trials() -> None:
        if not model_name:
            messagebox.showinfo("Model Builder", "No model selected.")
            return
        if on_lifecycle:
            on_lifecycle(model_name, "complete_optimization")
        else:
            messagebox.showinfo(
                "Model Builder",
                "Open Create Model from the main navigation to review optimization trials.",
            )

    section_title(parent, "Hyperparameter Optimization")

    status_text = f"{'✓' if performed else '✗'} {hpo.get('status_label') or 'Not Performed'}"
    status_color = COL_OK if performed else COL_WARN
    status_row = ttk.Frame(parent)
    status_row.pack(fill="x", pady=(0, 6))
    ttk.Label(status_row, text="Status", font=(BODY_FONT[0], BODY_FONT[1], "bold"), foreground=COL_MUTED, width=18).pack(side="left")
    ttk.Label(status_row, text=status_text, font=BODY_FONT, foreground=status_color).pack(side="left")

    if performed:
        metric_rows = [
            ("Best Composite", fmt_num(hpo.get("best_composite")) if hpo.get("best_composite") is not None else "—"),
            ("Best Trial", fmt_val(hpo.get("best_trial"))),
            ("Current Parameters", param_preview),
        ]
    else:
        metric_rows = [
            ("Current Model", "Baseline Parameters"),
            ("Parameters", param_preview),
        ]
    spec_grid(parent, metric_rows, columns=1)

    section_desc(
        parent,
        "Actions open Model Builder with dataset, target, features, validation, and parameters preloaded — nothing to re-select.",
    )

    actions = ttk.Frame(parent)
    actions.pack(fill="x", pady=(8, 0))

    ttk.Button(actions, text="Retrain", command=lambda: _lifecycle("retrain")).pack(side="left", padx=(0, 6), pady=2)
    primary_label = "Re-run Optimization" if performed else "Complete Optimization"
    ttk.Button(actions, text=primary_label, command=lambda: _lifecycle("complete_optimization")).pack(side="left", padx=6, pady=2)
    ttk.Button(actions, text="Feature Optimization", command=lambda: _lifecycle("feature_optimization")).pack(side="left", padx=6, pady=2)

    cal_btn = ttk.Button(
        actions,
        text="Calibration Only",
        command=lambda: _lifecycle("calibration_only"),
    )
    cal_btn.state(["disabled"])
    cal_btn.pack(side="left", padx=6, pady=2)

    if performed:
        ttk.Button(
            actions,
            text="View Trials",
            command=_view_trials,
        ).pack(side="left", padx=6, pady=2)


def _history_metrics(h: dict[str, Any]) -> dict[str, Any]:
    """Read evaluation metrics for a lifecycle history row (package-authoritative).

    Prefer ``production_metrics`` / top-level fields filled by
    ``_enrich_history_row_from_disk``. Never trust deprecated DB metric columns alone.
    """
    if not isinstance(h, dict):
        return {}
    prod = h.get("production_metrics") if isinstance(h.get("production_metrics"), dict) else {}
    if not prod:
        metrics = h.get("metrics") if isinstance(h.get("metrics"), dict) else {}
        nested = metrics.get("production") if isinstance(metrics.get("production"), dict) else {}
        prod = nested
    mae = prod.get("mae") if prod.get("mae") is not None else h.get("mae")
    rmse = prod.get("rmse") if prod.get("rmse") is not None else h.get("rmse")
    dir_pct = (
        prod.get("directional_accuracy_pct")
        if prod.get("directional_accuracy_pct") is not None
        else h.get("directional_accuracy_pct")
    )
    composite = (
        prod.get("composite_score") if prod.get("composite_score") is not None else h.get("composite_score")
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "directional_accuracy_pct": dir_pct,
        "composite_score": composite,
    }


def render_lifecycle(scroll: ScrollableFrame, doc: dict[str, Any]) -> None:
    clear_children(scroll.inner)
    parent = scroll.inner
    lc = doc.get("model_lifecycle") if isinstance(doc.get("model_lifecycle"), dict) else {}
    history = lc.get("history") if isinstance(lc.get("history"), list) else []
    row = doc.get("table_row") or {}
    prod_doc = _prod(doc)

    if not history:
        section_desc(parent, "No multi-version history yet. This model is the first version in its family.")
        section_title(parent, "Current Snapshot")
        spec_grid(
            parent,
            [
                ("Version", "v1"),
                ("Dataset", row.get("dataset")),
                ("Rows", fmt_rows(row.get("rows"))),
                ("Features", row.get("feature_count")),
                ("Target", row.get("target")),
                ("Algorithm", row.get("algorithm")),
                ("Validation", row.get("validation_strategy")),
                ("MAE", fmt_rupee(prod_doc.get("mae"))),
                ("RMSE", fmt_rupee(prod_doc.get("rmse"))),
                ("Direction %", fmt_pct(prod_doc.get("directional_accuracy_pct")) if prod_doc.get("directional_accuracy_pct") is not None else "—"),
                ("Trained at", row.get("trained_at")),
            ],
        )
        return

    imp = lc.get("improvement") if isinstance(lc.get("improvement"), dict) else {}
    since = imp.get("improvement_since_initial") if isinstance(imp.get("improvement_since_initial"), dict) else {}
    cur_m = imp.get("current_metrics") if isinstance(imp.get("current_metrics"), dict) else _history_metrics(history[-1])
    if cur_m.get("mae") is None and prod_doc.get("mae") is not None:
        cur_m = {
            "mae": prod_doc.get("mae"),
            "rmse": prod_doc.get("rmse"),
            "directional_accuracy_pct": prod_doc.get("directional_accuracy_pct"),
            "composite_score": prod_doc.get("composite_score"),
        }

    parent.columnconfigure(0, weight=1)
    parent.rowconfigure(0, weight=1)

    top_row = ttk.Frame(parent)
    top_row.pack(fill="both", expand=True)
    top_row.columnconfigure(0, weight=2)
    top_row.columnconfigure(1, weight=3)
    top_row.rowconfigure(0, weight=1)

    summary_col = ttk.Frame(top_row, padding=(0, 4))
    summary_col.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
    evo_col = ttk.Frame(top_row, padding=(0, 4))
    evo_col.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

    section_title(summary_col, "Improvement Summary")
    inline_spec_rows(
        summary_col,
        [
            ("Current version", imp.get("current_version") or history[-1].get("version_label")),
            ("Version count", imp.get("version_count") or len(history)),
            ("Initial version", imp.get("initial_version")),
            ("Current MAE", fmt_rupee(cur_m.get("mae"))),
            ("Current RMSE", fmt_rupee(cur_m.get("rmse"))),
            ("Current Direction", fmt_pct(cur_m.get("directional_accuracy_pct")) if cur_m.get("directional_accuracy_pct") is not None else "—"),
            ("MAE change", since.get("mae") or "—"),
            ("RMSE change", since.get("rmse") or "—"),
            ("Direction change", since.get("directional_accuracy_pct") or "—"),
        ],
        label_width=16,
    )

    section_title(evo_col, "Metrics Evolution")
    evo_rows = []
    for h in history:
        if not isinstance(h, dict):
            continue
        m = _history_metrics(h)
        evo_rows.append((
            h.get("version_label") or "—",
            fmt_rupee(m.get("mae")),
            fmt_rupee(m.get("rmse")),
            fmt_pct(m.get("directional_accuracy_pct")) if m.get("directional_accuracy_pct") is not None else "—",
            fmt_num(m.get("composite_score")) if m.get("composite_score") is not None else "—",
        ))
    data_table(
        evo_col,
        [
            ("ver", "Version", 50),
            ("mae", "MAE", 80),
            ("rmse", "RMSE", 80),
            ("dir", "Direction %", 90),
            ("comp", "Composite", 80),
        ],
        evo_rows,
        height=min(8, len(evo_rows) + 1),
    )

    section_title(parent, "Version History")
    hist_rows = []
    for h in history:
        if not isinstance(h, dict):
            continue
        m = _history_metrics(h)
        hist_rows.append((
            h.get("version_label") or "—",
            h.get("dataset") or "—",
            fmt_rows(h.get("row_count")),
            h.get("feature_count") or h.get("selected_feature_count") or "—",
            fmt_rupee(m.get("mae")),
            fmt_rupee(m.get("rmse")),
            fmt_pct(m.get("directional_accuracy_pct")) if m.get("directional_accuracy_pct") is not None else "—",
            (h.get("trained_at") or "")[:19],
        ))
    data_table(
        parent,
        [
            ("ver", "Version", 50),
            ("dataset", "Dataset", 120),
            ("rows", "Rows", 70),
            ("feat", "Features", 70),
            ("mae", "MAE", 70),
            ("rmse", "RMSE", 70),
            ("dir", "Dir %", 70),
            ("trained", "Trained", 130),
        ],
        hist_rows,
        height=min(10, len(hist_rows) + 1),
    )


def render_artifacts(scroll: ScrollableFrame, doc: dict[str, Any]) -> None:
    clear_children(scroll.inner)
    parent = scroll.inner
    inv = doc.get("artifact_inventory") if isinstance(doc.get("artifact_inventory"), dict) else {}
    pkg_files = doc.get("package_files") if isinstance(doc.get("package_files"), list) else []

    section_title(parent, "Saved artifacts")
    if inv:
        rows = []
        for path, art in sorted(inv.items()):
            ok = isinstance(art, dict) and art.get("available")
            rows.append((path, "Available" if ok else "Not available"))
        data_table(parent, [("path", "Artifact", 280), ("status", "Status", 100)], rows, height=min(12, len(rows) + 1))
    else:
        ttk.Label(parent, text="Not available", foreground=COL_MUTED).pack(anchor="w")

    section_title(parent, "Package files")
    if pkg_files:
        for f in pkg_files:
            ttk.Label(parent, text=str(f), font=("Consolas", 9)).pack(anchor="w")
    else:
        ttk.Label(parent, text="Not available", foreground=COL_MUTED).pack(anchor="w")

    section_title(parent, "Training log")
    log = doc.get("training_log") or ""
    if log:
        from tkinter import scrolledtext
        txt = scrolledtext.ScrolledText(parent, height=10, font=("Consolas", 9))
        txt.pack(fill="both", expand=True, pady=(4, 8))
        txt.insert("end", log[:50000])
        txt.configure(state="disabled")
    else:
        ttk.Label(parent, text="Not available", foreground=COL_MUTED).pack(anchor="w")


def render_model_research(
    parent: tk.Misc,
    *,
    chart_dir: str,
    model_name: str,
    on_run_program: Callable[[str, str], None] | None = None,
) -> None:
    """Model -> Research tab (Phase F5)."""
    from .build_service import chart_data_dir

    clear_children(parent)
    data_dir = chart_data_dir(chart_dir)

    if not model_name:
        ttk.Label(parent, text="Select a model to view research.", foreground=COL_MUTED).pack(anchor="w", pady=8)
        return

    try:
        from chain_replay_ml.fold_research.model_research import get_model_research_view

        view = get_model_research_view(data_dir, model_name)
    except Exception as exc:
        ttk.Label(parent, text=f"Research view unavailable: {exc}", foreground=COL_WARN).pack(anchor="w", pady=8)
        return

    if not view.get("ok"):
        ttk.Label(parent, text=view.get("error") or "Research unavailable", foreground=COL_WARN).pack(anchor="w")
        return

    section_title(parent, "Research Programs on this Model")
    section_desc(
        parent,
        "Reusable research programs execute on this model. Train -> Research -> Certify -> Deploy.",
    )

    toolbar = ttk.Frame(parent)
    toolbar.pack(fill="x", pady=(4, 8))
    if on_run_program:
        ttk.Button(
            toolbar,
            text="Run Research Program...",
            command=lambda: on_run_program(model_name, data_dir),
        ).pack(side="left")

    programs = view.get("programs") or []
    if not programs:
        ttk.Label(parent, text="No program runs yet.", foreground=COL_MUTED).pack(anchor="w")
    else:
        for pr in programs:
            fr = ttk.LabelFrame(
                parent,
                text=f"{pr.get('program_name') or 'Program'}  ({pr.get('status')})",
                padding=8,
            )
            fr.pack(fill="x", pady=(0, 8))
            ttk.Label(
                fr,
                text=f"Run #{pr.get('run_number')}  -  Type: {pr.get('program_type') or 'strategy'}",
                foreground=COL_MUTED,
            ).pack(anchor="w")
            for c in pr.get("campaigns") or []:
                man = c.get("manifest") or {}
                ttk.Label(
                    fr,
                    text=f"  {c.get('name')}: {c.get('status')}  -  jobs {man.get('completed_jobs', 0)}",
                    font=("Consolas", 9),
                ).pack(anchor="w")

    cert = view.get("certification") or {}
    grade = cert.get("research_grade") or "-"
    gen = cert.get("generalization_pct")
    ready = "YES" if cert.get("production_ready") else "NO"
    kv_block(parent, "Certification", [
        ("Research Grade", grade),
        ("Generalization", f"{gen}%" if gen is not None else "-"),
        ("Production Ready", ready),
        ("Champion Candidate", "YES" if cert.get("champion_candidate") else "NO"),
    ])

    knowledge = view.get("knowledge") or {}
    kv_block(parent, "Knowledge", [
        ("Findings", str(knowledge.get("count") or 0)),
        ("Promoted", str(knowledge.get("promoted") or 0)),
        ("Best Stop", f"{knowledge.get('best_stop')}%" if knowledge.get("best_stop") else "-"),
        ("Confidence", f"{knowledge.get('confidence_pct')}%" if knowledge.get("confidence_pct") else "-"),
        ("Premium", str(knowledge.get("premium") or "-")[:48]),
        ("Hold", str(knowledge.get("hold") or "-")[:48]),
    ])

    portfolio = view.get("portfolio_report") or {}
    if portfolio:
        kv_block(parent, "Research Portfolio Report", [
            ("Programs", str(portfolio.get("programs") or 0)),
            ("Campaigns", str(portfolio.get("campaigns") or 0)),
            ("Experiments", str(portfolio.get("experiments") or 0)),
            ("Knowledge", str(portfolio.get("knowledge") or 0)),
            ("Rejected Hypotheses", str(portfolio.get("rejected_hypotheses") or 0)),
            ("GPU Hours", str(portfolio.get("gpu_hours") or 0)),
            ("Best PF", fmt_num(portfolio.get("best_pf"))),
            ("Certification", str(portfolio.get("certification") or "-")),
        ])

    # Phase 4D: Persistent Research Memory & Multi-Model Robustness Scorecard
    try:
        from chain_replay_ml.research_memory.db import connect_analysis_db
        from chain_replay_ml.research_memory.ranking import rank_models_in_context

        conn = connect_analysis_db(data_dir)
        try:
            bm_row = conn.execute(
                "SELECT * FROM model_benchmarks WHERE model_name = ? ORDER BY created_at DESC LIMIT 1;",
                (model_name,),
            ).fetchone()
            if bm_row:
                sig_hash = bm_row["signature_hash"]
                ctx_key = bm_row["context_key"]
                section_title(parent, "Phase 4D Research Memory & Robustness Scorecard")
                dossiers = rank_models_in_context(data_dir, ctx_key)
                target_d = next((d for d in dossiers if d["signature_hash"] == sig_hash), None)
                if target_d:
                    score = target_d.get("robustness_score", 0.0)
                    p_rank = target_d.get("pareto_rank", 1)
                    kv_block(parent, "Robustness Summary", [
                        ("Context Key", ctx_key),
                        ("Robustness Score", f"{score:.2f} / 100.00"),
                        ("Pareto Rank", f"Tier {p_rank}"),
                        ("Recommendation", target_d.get("recommendation_status", "VALIDATED")),
                        ("Policy", target_d.get("ranking_policy_version", "ROB_POLICY_v1.0")),
                    ])
                    breakdown = target_d.get("score_breakdown", {})
                    p_rows = [
                        ("Base Performance", f"+{breakdown.get('base_performance_contribution', 0.0):.2f} pts"),
                        ("Fold Variance Penalty", f"{breakdown.get('fold_variance_penalty', 0.0):.2f} pts"),
                        ("Worst Fold Drawdown Penalty", f"{breakdown.get('worst_fold_penalty', 0.0):.2f} pts"),
                        ("Calibration (ECE) Penalty", f"{breakdown.get('calibration_penalty', 0.0):.2f} pts"),
                        ("Regime Degradation Penalty", f"{breakdown.get('regime_degradation_penalty', 0.0):.2f} pts"),
                        ("Experimental Risk Penalty", f"{breakdown.get('experimental_risk_penalty', 0.0):.2f} pts"),
                        ("Parsimony Penalty", f"{breakdown.get('parsimony_penalty', 0.0):.2f} pts"),
                    ]
                    data_table(
                        parent,
                        [("dim", "Evaluation Dimension", 240), ("impact", "Score Impact", 160)],
                        p_rows,
                    )
        finally:
            conn.close()
    except Exception:
        pass


def _holdout_feature_mean(val: Any) -> str:
    if val is None:
        return "—"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "—"
    if v != v:
        return "—"
    av = abs(v)
    if av >= 1000:
        return f"{v:.0f}"
    if av >= 100:
        return f"{v:.1f}"
    if av >= 1:
        return f"{v:.2f}"
    return f"{v:.4f}".rstrip("0").rstrip(".")


def _holdout_drift_pct(val: Any) -> str:
    if val is None:
        return "—"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "—"
    if v != v:
        return "—"
    prefix = "+" if v > 0 else ""
    return f"{prefix}{v:.0f}%"


def _holdout_shift_text(shift: Any) -> str:
    if shift is None:
        return "—"
    try:
        val = float(shift)
    except (TypeError, ValueError):
        return "—"
    if val != val:
        return "—"
    prefix = "+" if val > 0 else ""
    return f"{prefix}{fmt_num(val, 4)}"


def _holdout_pct_change_display(val: Any) -> str:
    if val is None:
        return "—"
    try:
        n = float(val)
    except (TypeError, ValueError):
        return "—"
    if n != n:
        return "—"
    prefix = "+" if n > 0 else ""
    return f"{prefix}{n:.0f}%"


def _holdout_direction_pts_display(val: Any) -> str:
    if val is None:
        return "—"
    try:
        n = float(val)
    except (TypeError, ValueError):
        return "—"
    if n != n:
        return "—"
    prefix = "+" if n > 0 else "−" if n < 0 else ""
    return f"{prefix}{abs(n):.1f} pts"


def _holdout_error_change_row(change: dict[str, Any]) -> tuple[str, ...]:
    return (
        change.get("label") or "Change",
        _holdout_pct_change_display(change.get("mae_pct_change")),
        _holdout_pct_change_display(change.get("rmse_pct_change")),
        _holdout_pct_change_display(change.get("premium_mae_pct_change")),
        _holdout_pct_change_display(change.get("premium_rmse_pct_change")),
        _holdout_direction_pts_display(change.get("direction_pts_change")),
    )


def _holdout_error_metric_row(label: str, metrics: dict[str, Any]) -> tuple[str, ...]:
    return (
        label,
        fmt_rupee(metrics.get("mae")),
        fmt_rupee(metrics.get("rmse")),
        fmt_pct(metrics.get("premium_mae_pct")) if metrics.get("premium_mae_pct") is not None else "—",
        fmt_pct(metrics.get("premium_rmse_pct")) if metrics.get("premium_rmse_pct") is not None else "—",
        fmt_pct(metrics.get("directional_accuracy_pct")) if metrics.get("directional_accuracy_pct") is not None else "—",
    )


def _render_model_summary(parent: tk.Misc, pa: dict[str, Any]) -> None:
    summary = pa.get("model_summary") if isinstance(pa.get("model_summary"), dict) else {}
    if not summary:
        return
    box = ttk.LabelFrame(parent, text="Model Summary", padding=10)
    box.pack(fill="x", pady=(0, 12))

    overall_row = ttk.Frame(box)
    overall_row.pack(fill="x", pady=(0, 8))
    ttk.Label(overall_row, text="Overall Quality", font=BODY_FONT, foreground=COL_MUTED, width=28).pack(side="left")
    stars = str(summary.get("overall_stars_display") or "")
    label = str(summary.get("overall_quality") or "—")
    ttk.Label(
        overall_row,
        text=f"{stars} {label}".strip(),
        font=("Segoe UI", 11, "bold"),
        foreground="#C68A00",
    ).pack(side="left")

    def _quality_row(key: str, value: str) -> None:
        row = ttk.Frame(box)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=key, font=BODY_FONT, foreground=COL_MUTED, width=28, anchor="w").pack(side="left")
        val = value or "—"
        color = COL_MUTED
        if val in ("Excellent", "Good"):
            color = COL_OK
        elif val in ("Poor", "Fair"):
            color = COL_WARN if val == "Poor" else COL_HOLDOUT
        elif val == "Moderate":
            color = COL_HOLDOUT
        ttk.Label(row, text=val, font=BODY_FONT, foreground=color, anchor="w").pack(side="left", fill="x", expand=True)

    _quality_row("Typical prediction quality:", str(summary.get("typical_prediction_quality") or "—"))
    _quality_row("Extreme outlier handling:", str(summary.get("extreme_outlier_handling") or "—"))

    weakness_row = ttk.Frame(box)
    weakness_row.pack(fill="x", pady=(2, 0))
    ttk.Label(
        weakness_row,
        text="Main weakness:",
        font=BODY_FONT,
        foreground=COL_MUTED,
        width=28,
        anchor="w",
    ).pack(side="left")
    ttk.Label(
        weakness_row,
        text=str(summary.get("main_weakness") or "—"),
        font=BODY_FONT,
        wraplength=520,
        justify="left",
        anchor="w",
    ).pack(side="left", fill="x", expand=True)


def _render_holdout_report(parent: tk.Misc, report: dict[str, Any]) -> None:
    pa = report.get("premium_analysis") if isinstance(report.get("premium_analysis"), dict) else {}
    if pa:
        _render_model_summary(parent, pa)

    similarity = report.get("similarity_pct")
    if similarity is not None:
        section_title(parent, "Similarity Score", color=COL_HOLDOUT)
        sim_frame = ttk.Frame(parent)
        sim_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(
            sim_frame,
            text=f"Training vs Holdout — {float(similarity):.0f}% similar",
            font=("Segoe UI", 11, "bold"),
            foreground=COL_OK if float(similarity) >= 80 else (COL_WARN if float(similarity) >= 60 else COL_HOLDOUT),
        ).pack(anchor="w")

    drift_scores = report.get("drift_scores") or {}
    pred = report.get("prediction_errors") or {}
    production_wf = pred.get("production_wf") if isinstance(pred.get("production_wf"), dict) else {}
    holdout_test = pred.get("holdout_test") if isinstance(pred.get("holdout_test"), dict) else {}
    if drift_scores or production_wf or holdout_test:
        section_title(parent, "Drift Score", color=ACCENT)
        drift_row = ttk.Frame(parent)
        drift_row.pack(fill="x", expand=True, pady=(0, 8))
        left = ttk.Frame(drift_row, padding=(0, 4))
        left.pack(side="left", fill="both", expand=True, padx=(0, 6))
        right = ttk.Frame(drift_row, padding=(0, 4))
        right.pack(side="left", fill="both", expand=True, padx=(6, 0))

        if drift_scores:
            drift_score_bars(left, [
                ("Target Drift", float(drift_scores.get("target") or 0)),
                ("Feature Drift", float(drift_scores.get("feature") or 0)),
                ("Premium Drift", float(drift_scores.get("premium") or 0)),
                ("Volatility Drift", float(drift_scores.get("volatility") or 0)),
            ]).pack(fill="both", expand=True)

        if production_wf or holdout_test:
            err_rows = []
            if production_wf:
                err_rows.append(_holdout_error_metric_row("Production WF", production_wf))
            if holdout_test:
                err_rows.append(_holdout_error_metric_row("Holdout Test", holdout_test))
            change = pred.get("change") if isinstance(pred.get("change"), dict) else {}
            if production_wf and holdout_test and change:
                err_rows.append(_holdout_error_change_row(change))
            data_table(
                right,
                [
                    ("region", "Region", 96),
                    ("mae", "MAE", 72),
                    ("rmse", "RMSE", 72),
                    ("pmae", "Premium MAE", 88),
                    ("prmse", "Premium RMSE", 96),
                    ("dir", "Direction Acc", 96),
                ],
                err_rows,
                height=len(err_rows) + 1,
            )

    diagnosis = report.get("diagnosis") or {}
    section_title(parent, "Degradation Diagnosis", color=COL_HOLDOUT)
    cause_frame = ttk.Frame(parent)
    cause_frame.pack(fill="x", pady=(0, 4))
    ttk.Label(cause_frame, text="Primary cause", font=("Segoe UI", 9), foreground=COL_MUTED).pack(anchor="w")
    ttk.Label(
        cause_frame,
        text=diagnosis.get("label") or "—",
        font=("Segoe UI", 11, "bold"),
    ).pack(anchor="w")
    conf = diagnosis.get("confidence_pct")
    if conf is not None:
        ttk.Label(cause_frame, text=f"Confidence: {conf:.0f}%", font=BODY_FONT, foreground=COL_MUTED).pack(anchor="w", pady=(2, 0))

    evidence = diagnosis.get("evidence") or diagnosis.get("signals") or []
    if evidence:
        ttk.Label(parent, text="Evidence", font=("Segoe UI", 9, "bold"), foreground=COL_MUTED).pack(anchor="w", pady=(6, 2))
        for item in evidence:
            ttk.Label(parent, text=f"✓ {item}", foreground=COL_MUTED, wraplength=720, justify="left").pack(anchor="w")

    likely = diagnosis.get("likely_reason")
    if likely:
        ttk.Label(parent, text="Likely reason", font=("Segoe UI", 9, "bold"), foreground=COL_MUTED).pack(anchor="w", pady=(8, 2))
        ttk.Label(parent, text=likely, wraplength=720, justify="left").pack(anchor="w")

    premium_root = (report.get("premium_analysis") or {}).get("root_cause") or {}
    if premium_root:
        _render_root_cause_checklist(parent, premium_root, title="Primary Cause")
    pa = report.get("premium_analysis") or {}
    if pa:
        _render_prediction_quality_summary(parent, pa)

    feat_rank = report.get("feature_drift_ranking") or []
    if feat_rank:
        section_title(parent, "Feature Drift Ranking", color=ACCENT)
        section_desc(parent, "How each feature's mean shifted from WF to holdout, weighted by model importance.")
        rank_rows = []
        for row in feat_rank:
            if not isinstance(row, dict):
                continue
            drift_val = row.get("drift")
            imp_val = row.get("importance")
            rank_rows.append((
                row.get("feature") or "—",
                _holdout_feature_mean(row.get("wf_mean")),
                _holdout_feature_mean(row.get("holdout_mean")),
                _holdout_drift_pct(row.get("drift_pct")),
                fmt_num(float(drift_val), 2) if drift_val is not None else "—",
                fmt_num(float(imp_val), 2) if imp_val is not None else "—",
                row.get("risk_label") or "—",
            ))
        data_table(
            parent,
            [
                ("feature", "Feature", 180),
                ("wf", "WF Mean", 72),
                ("ho", "Holdout Mean", 72),
                ("chg", "Drift", 58),
                ("drift", "Drift Score", 68),
                ("imp", "Importance", 72),
                ("risk", "Risk", 72),
            ],
            rank_rows,
            height=min(14, len(rank_rows) + 1),
        )

    overview = report.get("overview") or {}
    section_title(parent, "Region Overview", color=COL_HOLDOUT)
    kv_block(parent, "", [
        ("WF region rows", fmt_rows(overview.get("wf_rows"))),
        ("Holdout rows", fmt_rows(overview.get("holdout_rows"))),
        ("Holdout indices", f"{overview.get('holdout_start')} – {overview.get('holdout_stop')}"),
        ("WF trading days", f"{overview.get('wf_day_start') or '—'} → {overview.get('wf_day_end') or '—'}"),
        ("Holdout trading days", f"{overview.get('holdout_day_start') or '—'} → {overview.get('holdout_day_end') or '—'}"),
        ("Volatility column", overview.get("volatility_column") or "—"),
    ])

    section_title(parent, "Region Comparison", color=ACCENT)
    section_desc(parent, "Walk-forward region vs untouched holdout. Shift = holdout minus WF (or normalized mean shift for means).")
    comp_rows = []
    for row in report.get("region_comparison") or []:
        if not isinstance(row, dict):
            continue
        comp_rows.append((
            row.get("category") or "—",
            row.get("metric") or "—",
            fmt_num(row.get("wf"), 4) if row.get("wf") is not None else "—",
            fmt_num(row.get("holdout"), 4) if row.get("holdout") is not None else "—",
            _holdout_shift_text(row.get("shift")),
        ))
    if comp_rows:
        data_table(
            parent,
            [
                ("cat", "Dimension", 130),
                ("metric", "Metric", 120),
                ("wf", "WF Region", 90),
                ("ho", "Holdout", 90),
                ("shift", "Shift", 80),
            ],
            comp_rows,
            height=min(18, len(comp_rows) + 1),
        )

    section_title(parent, "Holdout by Premium Band", color=ACCENT)
    band_rows = []
    for b in report.get("holdout_by_premium_band") or []:
        if not isinstance(b, dict):
            continue
        band_rows.append((
            b.get("band_label") or b.get("band") or "—",
            fmt_rows(b.get("samples")),
            fmt_rupee(b.get("mae")),
            fmt_rupee(b.get("rmse")),
            fmt_pct(b.get("premium_mae_pct")) if b.get("premium_mae_pct") is not None else "—",
            fmt_pct(b.get("directional_accuracy_pct")) if b.get("directional_accuracy_pct") is not None else "—",
        ))
    if band_rows:
        data_table(
            parent,
            [
                ("band", "Band", 70),
                ("n", "Rows", 60),
                ("mae", "MAE", 80),
                ("rmse", "RMSE", 80),
                ("pmae", "Prem MAE %", 90),
                ("dir", "Direction", 80),
            ],
            band_rows,
            height=min(8, len(band_rows) + 1),
        )
    else:
        ttk.Label(parent, text="Not available", foreground=COL_MUTED).pack(anchor="w")

    section_title(parent, "Holdout by Trading Day", color=ACCENT)
    day_rows = []
    for d in report.get("holdout_by_trading_day") or []:
        if not isinstance(d, dict):
            continue
        day_rows.append((
            d.get("trading_day") or "—",
            fmt_rows(d.get("rows")),
            fmt_rupee(d.get("mae")),
            fmt_rupee(d.get("rmse")),
            fmt_pct(d.get("premium_mae_pct")) if d.get("premium_mae_pct") is not None else "—",
            fmt_pct(d.get("directional_accuracy_pct")) if d.get("directional_accuracy_pct") is not None else "—",
        ))
    if day_rows:
        data_table(
            parent,
            [
                ("day", "Trading day", 110),
                ("n", "Rows", 60),
                ("mae", "MAE", 80),
                ("rmse", "RMSE", 80),
                ("pmae", "Prem MAE %", 90),
                ("dir", "Direction", 80),
            ],
            day_rows,
            height=min(12, len(day_rows) + 1),
        )
    else:
        ttk.Label(parent, text="Trading day column not available in dataset.", foreground=COL_MUTED).pack(anchor="w")


def _render_root_cause_checklist(parent: tk.Misc, root: dict[str, Any], *, title: str = "Primary Cause") -> None:
    if not root:
        return
    section_title(parent, title, color=COL_HOLDOUT)
    primary = root.get("primary_cause")
    if primary:
        ttk.Label(
            parent,
            text=str(primary),
            font=("Segoe UI", 11, "bold"),
            wraplength=720,
            justify="left",
        ).pack(anchor="w", pady=(0, 6))
    if root.get("rmse_note"):
        section_desc(parent, str(root["rmse_note"]))
    for item in root.get("checklist") or []:
        if not isinstance(item, dict) or not item.get("text"):
            continue
        ttk.Label(
            parent,
            text=f"✓ {item['text']}",
            foreground=COL_OK if item.get("status") == "ok" else COL_MUTED,
            wraplength=720,
            justify="left",
        ).pack(anchor="w")
    for warning in root.get("warnings") or []:
        ttk.Label(
            parent,
            text=f"⚠ {warning}",
            foreground=COL_WARN,
            wraplength=720,
            justify="left",
        ).pack(anchor="w", pady=(4, 0))
    for bullet in root.get("detail_bullets") or []:
        ttk.Label(
            parent,
            text=f"• {bullet}",
            foreground=COL_MUTED,
            wraplength=720,
            justify="left",
        ).pack(anchor="w")
    if root.get("conclusion"):
        ttk.Label(
            parent,
            text=str(root["conclusion"]),
            wraplength=720,
            justify="left",
            foreground=COL_MUTED,
        ).pack(anchor="w", pady=(8, 0))


def _render_prediction_quality_summary(parent: tk.Misc, pa: dict[str, Any]) -> None:
    quality = pa.get("quality_summary") if isinstance(pa.get("quality_summary"), dict) else {}
    if not quality:
        return
    rel = quality.get("relative_error") if isinstance(quality.get("relative_error"), dict) else {}
    row = ttk.Frame(parent)
    row.pack(fill="x", pady=(0, 12))

    left = ttk.LabelFrame(row, text="Prediction Quality", padding=8)
    left.pack(side="left", fill="both", expand=True, padx=(0, 8))

    table_rows: list[tuple[str, str]] = []
    for key, label in (
        ("median", "Median Relative Error"),
        ("p90", "90th Percentile"),
        ("p95", "95th Percentile"),
        ("p99", "99th Percentile"),
    ):
        val = rel.get(key)
        table_rows.append((label, f"{float(val):.1f}%" if val is not None else "—"))
    prmse = quality.get("premium_rmse_pct")
    table_rows.append(("Premium RMSE", f"{float(prmse):.1f}%" if prmse is not None else "—"))
    excl = quality.get("premium_rmse_excl_top_pct")
    excl_pct = int(quality.get("exclude_top_pct") or 1)
    table_rows.append((
        f"Premium RMSE (excluding top {excl_pct}%)",
        f"{float(excl):.1f}%" if excl is not None else "—",
    ))
    data_table(
        left,
        [("metric", "Metric", 240), ("value", "Value", 90)],
        table_rows,
        height=len(table_rows) + 1,
    )

    impact = pa.get("outlier_impact") if isinstance(pa.get("outlier_impact"), dict) else {}
    if impact:
        card = outlier_impact_card(row, impact)
        card.pack(side="left", fill="y", padx=(0, 0))


def _render_premium_analysis(parent: tk.Misc, report: dict[str, Any], *, model_name: str = "") -> None:
    pa = report.get("premium_analysis") or {}
    if not pa:
        ttk.Label(parent, text="Premium analysis not available.", foreground=COL_MUTED).pack(anchor="w", pady=8)
        return

    toolbar = ttk.Frame(parent)
    toolbar.pack(fill="x", pady=(0, 8))
    ttk.Button(
        toolbar,
        text="Download CSV",
        command=lambda: _download_premium_analysis_csv(report, model_name),
    ).pack(side="left")

    body = ttk.Frame(parent)
    body.pack(fill="both", expand=True)

    _render_model_summary(body, pa)

    root = pa.get("root_cause") or {}
    _render_root_cause_checklist(body, root)
    _render_prediction_quality_summary(body, pa)

    section_title(body, "Premium RMSE Breakdown", color=ACCENT)
    section_desc(body, "Holdout relative-squared-error contribution by premium band (LTP baseline).")
    band_rows = []
    for b in pa.get("band_breakdown") or []:
        if not isinstance(b, dict):
            continue
        band_rows.append((
            b.get("band_label") or b.get("band") or "—",
            fmt_rows(b.get("rows")),
            f"{float(b['premium_rmse_pct']):.0f}%" if b.get("premium_rmse_pct") is not None else "—",
            f"{float(b['contribution_pct']):.0f}%" if b.get("contribution_pct") is not None else "—",
        ))
    if band_rows:
        data_table(
            body,
            [("band", "Premium Band", 80), ("n", "Rows", 70), ("rmse", "RMSE %", 70), ("contrib", "Contribution", 90)],
            band_rows,
            height=min(8, len(band_rows) + 1),
        )

    section_title(body, "Outlier Contribution", color=ACCENT)
    section_desc(
        body,
        "Share of total Σ((pred−actual)/actual)² across holdout rows — the Premium RMSE "
        "numerator before taking √mean. Not absolute error (MAE) and not a share of the final RMSE scalar.",
    )
    outlier_rows = []
    for o in pa.get("outlier_contribution") or []:
        if not isinstance(o, dict):
            continue
        outlier_rows.append((
            o.get("label") or "—",
            f"{float(o['contribution_pct']):.1f}%" if o.get("contribution_pct") is not None else "—",
        ))
    if outlier_rows:
        data_table(
            body,
            [("label", "Largest Errors", 120), ("contrib", "Contribution to Σ(rel²)", 180)],
            outlier_rows,
            height=len(outlier_rows) + 1,
        )

    section_title(body, "Error Distribution", color=ACCENT)
    dist = pa.get("error_distribution") or {}
    wf_d = dist.get("wf") if isinstance(dist.get("wf"), dict) else {}
    ho_d = dist.get("holdout") if isinstance(dist.get("holdout"), dict) else {}
    dist_rows = []
    for key, label in (("median", "Median Error"), ("p90", "90th Percentile"), ("p95", "95th Percentile"), ("p99", "99th Percentile"), ("max", "Max Error")):
        dist_rows.append((
            label,
            fmt_rupee(wf_d.get(key)),
            fmt_rupee(ho_d.get(key)),
        ))
    if dist_rows:
        data_table(
            body,
            [("metric", "Metric", 120), ("wf", "WF", 80), ("ho", "Holdout", 80)],
            dist_rows,
            height=len(dist_rows) + 1,
        )

    section_title(body, "Worst Trading Days", color=ACCENT)
    day_rows = []
    for d in pa.get("worst_trading_days") or []:
        if not isinstance(d, dict):
            continue
        day_rows.append((
            d.get("trading_day") or "—",
            fmt_rupee(d.get("mae")),
            fmt_pct(d.get("directional_accuracy_pct")) if d.get("directional_accuracy_pct") is not None else "—",
        ))
    if day_rows:
        data_table(
            body,
            [("day", "Day", 110), ("mae", "MAE", 80), ("dir", "Direction", 80)],
            day_rows,
            height=min(10, len(day_rows) + 1),
        )
    else:
        ttk.Label(body, text="Not available", foreground=COL_MUTED).pack(anchor="w")

    section_title(body, "Top Error Samples", color=ACCENT)
    sample_rows = []
    for s in pa.get("top_error_samples") or []:
        if not isinstance(s, dict):
            continue
        err = s.get("error")
        err_txt = f"+{err}" if isinstance(err, (int, float)) and float(err) > 0 else str(err)
        sample_rows.append((
            s.get("time") or "—",
            s.get("strike") or "—",
            fmt_num(s.get("actual"), 2),
            fmt_num(s.get("predicted"), 2),
            err_txt,
            s.get("reason") or "—",
        ))
    if sample_rows:
        data_table(
            body,
            [
                ("time", "Time", 100),
                ("strike", "Strike", 90),
                ("actual", "Actual", 60),
                ("pred", "Predicted", 70),
                ("err", "Error", 60),
                ("reason", "Reason", 120),
            ],
            sample_rows,
            height=min(12, len(sample_rows) + 1),
        )
    else:
        ttk.Label(body, text="Not available", foreground=COL_MUTED).pack(anchor="w")


def _render_top1_error_analysis(
    parent: tk.Misc,
    report: dict[str, Any],
    *,
    model_name: str = "",
    chart_dir: str = "",
) -> None:
    pa = report.get("premium_analysis") or {}
    analysis = pa.get("top1_analysis") if isinstance(pa.get("top1_analysis"), dict) else {}
    if not analysis.get("ok"):
        ttk.Label(
            parent,
            text=analysis.get("error") or "Top 1% analysis not available.",
            foreground=COL_MUTED,
            wraplength=720,
        ).pack(anchor="w", pady=8)
        return

    toolbar = ttk.Frame(parent)
    toolbar.pack(fill="x", pady=(0, 8))
    ttk.Button(
        toolbar,
        text="Download CSV",
        command=lambda: _download_top1_analysis_csv(report, model_name),
    ).pack(side="left", padx=(0, 8))
    ttk.Button(
        toolbar,
        text="Save as Knowledge",
        command=lambda: _save_top1_knowledge(report, model_name, chart_dir),
    ).pack(side="left")

    body = ttk.Frame(parent)
    body.pack(fill="both", expand=True)

    ex = analysis.get("executive_summary") or {}
    section_title(body, "Investigation Report", color=COL_HOLDOUT)
    ttk.Label(
        body,
        text=f"{ex.get('stars_display') or ''} {ex.get('title') or 'Top 1% Error Investigation'}".strip(),
        font=("Segoe UI", 12, "bold"),
        foreground="#C68A00",
    ).pack(anchor="w")
    ttk.Label(
        body,
        text=f"Rows analyzed: {fmt_rows(ex.get('rows_analyzed'))}",
        font=BODY_FONT,
    ).pack(anchor="w", pady=(4, 0))
    if ex.get("avg_premium_error_pct") is not None:
        ttk.Label(
            body,
            text=f"Average Premium Error: {float(ex['avg_premium_error_pct']):.0f}%",
            font=BODY_FONT,
        ).pack(anchor="w")
    ttk.Label(body, text="Main Pattern:", font=("Segoe UI", 9, "bold"), foreground=COL_MUTED).pack(anchor="w", pady=(8, 2))
    for p in ex.get("patterns") or []:
        ttk.Label(body, text=f"• {p}", foreground=COL_MUTED, wraplength=720).pack(anchor="w")

    section_title(body, "Top 1% vs Remaining 99%", color=ACCENT)
    cmp_rows = []
    for row in analysis.get("metric_comparison") or []:
        if not isinstance(row, dict):
            continue
        diff = row.get("difference_pct")
        diff_txt = f"+{diff:.0f}%" if diff is not None and float(diff) > 0 else (
            f"{diff:.0f}%" if diff is not None else "—"
        )
        cmp_rows.append((
            row.get("metric") or "—",
            fmt_num(row.get("top1_mean"), 4) if row.get("top1_mean") is not None else "—",
            fmt_num(row.get("rest_mean"), 4) if row.get("rest_mean") is not None else "—",
            diff_txt,
        ))
    if cmp_rows:
        data_table(
            body,
            [("metric", "Metric", 140), ("top", "Top 1%", 90), ("rest", "Remaining 99%", 110), ("diff", "Difference", 90)],
            cmp_rows,
            height=min(12, len(cmp_rows) + 1),
        )

    section_title(body, "Distribution Comparison", color=ACCENT)
    dist_rows = []
    for row in analysis.get("distribution_comparison") or []:
        if not isinstance(row, dict):
            continue
        top = row.get("top1") if isinstance(row.get("top1"), dict) else {}
        rest = row.get("rest") if isinstance(row.get("rest"), dict) else {}
        dist_rows.append((
            row.get("feature") or "—",
            "Top 1%",
            fmt_num(top.get("p25"), 3), fmt_num(top.get("median"), 3),
            fmt_num(top.get("p75"), 3), fmt_num(top.get("p95"), 3),
        ))
        dist_rows.append((
            row.get("feature") or "—",
            "Remaining 99%",
            fmt_num(rest.get("p25"), 3), fmt_num(rest.get("median"), 3),
            fmt_num(rest.get("p75"), 3), fmt_num(rest.get("p95"), 3),
        ))
    if dist_rows:
        data_table(
            body,
            [("feat", "Feature", 120), ("grp", "Group", 100), ("p25", "P25", 70), ("med", "Median", 70), ("p75", "P75", 70), ("p95", "P95", 70)],
            dist_rows,
            height=min(16, len(dist_rows) + 1),
        )

    section_title(body, "Time Analysis", color=ACCENT)
    hour_rows = [(r.get("hour") or "—", fmt_rows(r.get("count"))) for r in analysis.get("time_analysis") or [] if isinstance(r, dict)]
    if hour_rows:
        data_table(body, [("hour", "Hour", 80), ("n", "Count", 80)], hour_rows, height=len(hour_rows) + 1)
        late = ex.get("late_session_pct")
        if late is not None:
            ttk.Label(body, text=f"{late:.0f}% of catastrophic errors occur after 2:00 PM.", foreground=COL_WARN, wraplength=720).pack(anchor="w", pady=(4, 0))

    section_title(body, "Expiry Analysis", color=ACCENT)
    exp_rows = [
        (r.get("category") or "—", f"{float(r['percentage']):.0f}%" if r.get("percentage") is not None else "—")
        for r in analysis.get("expiry_analysis") or [] if isinstance(r, dict)
    ]
    if exp_rows:
        data_table(body, [("cat", "Category", 120), ("pct", "Percentage", 90)], exp_rows, height=len(exp_rows) + 1)

    section_title(body, "Premium Analysis", color=ACCENT)
    prem_rows = [
        (r.get("band") or "—", f"{float(r['percentage']):.0f}%" if r.get("percentage") is not None else "—")
        for r in analysis.get("premium_band_analysis") or [] if isinstance(r, dict)
    ]
    if prem_rows:
        data_table(body, [("band", "Band", 90), ("pct", "Percentage", 90)], prem_rows, height=len(prem_rows) + 1)

    section_title(body, "Strike Distance", color=ACCENT)
    strike_rows = [(r.get("distance") or "—", fmt_rows(r.get("count"))) for r in analysis.get("strike_distance") or [] if isinstance(r, dict)]
    if strike_rows:
        data_table(body, [("dist", "Distance", 80), ("n", "Count", 80)], strike_rows, height=len(strike_rows) + 1)

    section_title(body, "Greeks", color=ACCENT)
    greek_rows = []
    for g in analysis.get("greeks_ranking") or []:
        if not isinstance(g, dict):
            continue
        diff = g.get("difference_pct")
        greek_rows.append((
            g.get("feature") or "—",
            f"+{diff:.0f}%" if diff is not None and float(diff) > 0 else (f"{diff:.0f}%" if diff is not None else "—"),
            g.get("rank_stars") or "—",
        ))
    if greek_rows:
        data_table(body, [("feat", "Feature", 100), ("diff", "Difference", 90), ("rank", "Rank", 100)], greek_rows, height=len(greek_rows) + 1)

    section_title(body, "Feature Importance × Drift × Top 1%", color=ACCENT)
    feat_rows = []
    for r in analysis.get("feature_risk_matrix") or []:
        if not isinstance(r, dict):
            continue
        feat_rows.append((
            r.get("feature") or "—",
            _holdout_drift_pct(r.get("drift_pct")),
            fmt_num(r.get("importance"), 2) if r.get("importance") is not None else "—",
            f"+{float(r['top1_difference_pct']):.0f}%" if r.get("top1_difference_pct") is not None else "—",
            r.get("risk") or "—",
        ))
    if feat_rows:
        data_table(
            body,
            [("feat", "Feature", 140), ("drift", "Drift", 70), ("imp", "Importance", 80), ("diff", "Top1% Diff", 90), ("risk", "Risk", 50)],
            feat_rows,
            height=min(12, len(feat_rows) + 1),
        )

    drivers = analysis.get("driver_analysis") if isinstance(analysis.get("driver_analysis"), dict) else {}
    if drivers:
        section_title(body, "Primary Driver Analysis", color=ACCENT)
        primary = drivers.get("primary_driver") or (analysis.get("conclusion") or {}).get("primary_driver")
        method = drivers.get("importance_method") or "separation"
        if primary:
            ttk.Label(
                body,
                text=f"Primary driver: {primary} ({method})",
                font=("Segoe UI", 10, "bold"),
                foreground="#C68A00",
            ).pack(anchor="w", pady=(0, 6))
        driver_rows = []
        for row in drivers.get("driver_separation_ranking") or []:
            if not isinstance(row, dict):
                continue
            err_pct = row.get("error_contribution_pct")
            driver_rows.append((
                row.get("driver") or "—",
                fmt_num(row.get("separation_score"), 2) if row.get("separation_score") is not None else "—",
                f"{float(err_pct):.1f}%" if err_pct is not None else "—",
                fmt_num(row.get("top_median"), 4) if row.get("top_median") is not None else "—",
                fmt_num(row.get("rest_median"), 4) if row.get("rest_median") is not None else "—",
            ))
        if driver_rows:
            data_table(
                body,
                [
                    ("driver", "Driver", 110),
                    ("sep", "Separation", 80),
                    ("err", "Error %", 70),
                    ("top", "Top Median", 90),
                    ("rest", "Rest Median", 90),
                ],
                driver_rows,
                height=len(driver_rows) + 1,
            )

        imp_rows = []
        for row in drivers.get("feature_error_importance") or []:
            if not isinstance(row, dict):
                continue
            imp_rows.append((
                row.get("feature") or "—",
                f"{float(row.get('error_contribution_pct') or 0):.1f}%",
                row.get("method") or "—",
            ))
        if imp_rows:
            section_title(body, "Top 1% Error Feature Importance", color=ACCENT)
            data_table(
                body,
                [("feat", "Feature", 160), ("contrib", "Error Contribution", 120), ("method", "Method", 100)],
                imp_rows[:15],
                height=min(10, len(imp_rows[:15]) + 1),
            )

        rec = drivers.get("feature_recommendations") if isinstance(drivers.get("feature_recommendations"), dict) else {}
        if rec.get("notes"):
            ttk.Label(body, text=str(rec["notes"]), wraplength=720, foreground=COL_MUTED).pack(anchor="w", pady=(4, 0))

    conc = analysis.get("conclusion") or {}
    section_title(body, "AI Conclusion", color=COL_HOLDOUT)
    ttk.Label(body, text=conc.get("title") or "Top 1% Investigation", font=("Segoe UI", 11, "bold")).pack(anchor="w")
    if conc.get("confidence_pct") is not None:
        ttk.Label(body, text=f"Confidence: {float(conc['confidence_pct']):.0f}%", foreground=COL_MUTED).pack(anchor="w", pady=(2, 6))
    ttk.Label(body, text="Root Cause", font=("Segoe UI", 9, "bold"), foreground=COL_MUTED).pack(anchor="w", pady=(4, 2))
    for rc in conc.get("root_causes") or []:
        ttk.Label(body, text=f"✓ {rc}", foreground=COL_OK, wraplength=720).pack(anchor="w")
    ttk.Label(body, text="Recommendation", font=("Segoe UI", 9, "bold"), foreground=COL_MUTED).pack(anchor="w", pady=(8, 2))
    ttk.Label(body, text=str(conc.get("recommendation") or "—"), wraplength=720, justify="left").pack(anchor="w")
    if conc.get("finding_text"):
        ttk.Label(body, text=str(conc["finding_text"]), wraplength=720, foreground=COL_MUTED, justify="left").pack(anchor="w", pady=(8, 0))


def _render_holdout_days_analysis_tab(
    parent: tk.Misc,
    report: dict[str, Any],
    *,
    model_name: str = "",
) -> None:
    """Holdout Days Analysis tab — Analyze button reveals per-day deep dive."""
    toolbar = ttk.Frame(parent)
    toolbar.pack(fill="x", pady=(0, 8))
    body = ttk.Frame(parent)
    body.pack(fill="both", expand=True)

    status = tk.StringVar(value="")
    ttk.Label(body, textvariable=status, foreground=COL_MUTED, wraplength=720).pack(anchor="w")

    intro = ttk.Frame(body)
    intro.pack(fill="x", anchor="w", pady=(4, 0))
    section_title(intro, "Holdout Days Analysis", color=COL_HOLDOUT)
    section_desc(
        intro,
        "Per-trading-day metrics, premium bands, volatility, vs average training day, and regime-shift assessment.",
    )
    ttk.Label(
        intro,
        text="Click Analyze to generate the detailed report for holdout trading days.",
        foreground=COL_MUTED,
        wraplength=720,
    ).pack(anchor="w", pady=(6, 0))

    def _run() -> None:
        analysis = report.get("holdout_days_analysis")
        if not isinstance(analysis, dict) or not analysis.get("ok"):
            err = analysis.get("error") if isinstance(analysis, dict) else None
            messagebox.showwarning(
                "Holdout Days Analysis",
                err or "No holdout days analysis available. Run the main Holdout Analyze first.",
            )
            return
        clear_children(body)
        status_bar = ttk.Frame(body)
        status_bar.pack(fill="x", pady=(0, 4))
        ttk.Label(
            status_bar,
            text="Analysis complete.",
            foreground=COL_OK,
        ).pack(side="left")
        ttk.Button(
            status_bar,
            text="Download CSV",
            command=lambda: _download_holdout_days_analysis_csv(report, model_name),
        ).pack(side="right")
        _render_holdout_days_analysis_body(body, analysis)

    ttk.Button(toolbar, text="Analyze", command=_run).pack(side="left", padx=(0, 8))
    ttk.Button(
        toolbar,
        text="Download CSV",
        command=lambda: _download_holdout_days_analysis_csv(report, model_name),
    ).pack(side="left")


def _render_holdout_days_analysis_body(parent: tk.Misc, analysis: dict[str, Any]) -> None:
    content = ttk.Frame(parent)
    content.pack(fill="both", expand=True)

    ex = analysis.get("executive_summary") or {}
    section_title(content, "Executive Summary", color=COL_HOLDOUT)
    days = ex.get("trading_days") or []
    ttk.Label(
        content,
        text=f"Trading days: {', '.join(str(d) for d in days) if days else '—'}",
        font=("Segoe UI", 11, "bold"),
    ).pack(anchor="w")
    ttk.Label(
        content,
        text=(
            f"Holdout days: {ex.get('holdout_day_count') or 0}  ·  "
            f"Training days (avg basis): {ex.get('train_day_count') or 0}  ·  "
            f"Regime-shift flags: {ex.get('regime_shift_days') or 0}"
        ),
        font=BODY_FONT,
    ).pack(anchor="w", pady=(4, 0))
    tol = analysis.get("hit_rate_tolerance_pct")
    if tol is not None:
        ttk.Label(
            content,
            text=f"Endpoint Hit % = share of rows with |pred−actual|/|actual| ≤ {float(tol):.0f}%.",
            foreground=COL_MUTED,
            wraplength=720,
        ).pack(anchor="w", pady=(2, 0))
    for line in ex.get("lines") or []:
        ttk.Label(content, text=f"• {line}", foreground=COL_MUTED, wraplength=720).pack(anchor="w")

    train = analysis.get("training_day_average") or {}
    if train:
        section_title(content, "Average Training Day (WF region)", color=ACCENT)
        section_desc(content, "Baseline for comparing each holdout trading day.")
        train_rows = [
            ("Trading days", fmt_rows(train.get("trading_days"))),
            ("Rows / day", fmt_num(train.get("rows_mean"), 0) if train.get("rows_mean") is not None else "—"),
            ("MAE", fmt_rupee(train.get("mae"))),
            ("RMSE", fmt_rupee(train.get("rmse"))),
            ("Direction Acc", fmt_pct(train.get("directional_accuracy_pct")) if train.get("directional_accuracy_pct") is not None else "—"),
            ("Endpoint Hit %", fmt_pct(train.get("endpoint_hit_pct") if train.get("endpoint_hit_pct") is not None else train.get("hit_rate_pct")) if (train.get("endpoint_hit_pct") if train.get("endpoint_hit_pct") is not None else train.get("hit_rate_pct")) is not None else "—"),
            ("Premium MAE %", fmt_pct(train.get("premium_mae_pct"), 2) if train.get("premium_mae_pct") is not None else "—"),
            ("Spot range %", fmt_num(train.get("spot_range_pct"), 3) if train.get("spot_range_pct") is not None else "—"),
            ("Spot |return| %", fmt_num(train.get("spot_abs_return_pct"), 3) if train.get("spot_abs_return_pct") is not None else "—"),
            ("Premium std", fmt_rupee(train.get("premium_std"))),
            ("|IV z| mean", fmt_num(train.get("iv_abs_mean"), 3) if train.get("iv_abs_mean") is not None else "—"),
        ]
        data_table(
            content,
            [("metric", "Metric", 140), ("value", "Value", 120)],
            train_rows,
            height=min(14, len(train_rows) + 1),
        )

    for day in analysis.get("days") or []:
        if not isinstance(day, dict):
            continue
        name = str(day.get("trading_day") or "—")
        regime = day.get("regime") or {}
        section_title(content, f"Trading Day: {name}", color=COL_HOLDOUT)
        ttk.Label(
            content,
            text=f"{regime.get('label') or '—'}  ·  Rows: {fmt_rows(day.get('rows'))}",
            font=("Segoe UI", 10, "bold"),
            foreground=COL_WARN if regime.get("is_regime_shift") else COL_OK,
        ).pack(anchor="w")
        if day.get("gap_pct") is not None:
            ttk.Label(
                content,
                text=f"Opening gap vs prior close: {float(day['gap_pct']):+.3f}%",
                foreground=COL_MUTED,
            ).pack(anchor="w")

        m = day.get("metrics") or {}
        section_title(content, "Prediction Metrics", color=ACCENT)
        metric_rows = [
            ("MAE", fmt_rupee(m.get("mae"))),
            ("RMSE", fmt_rupee(m.get("rmse"))),
            ("Direction Accuracy", fmt_pct(m.get("directional_accuracy_pct")) if m.get("directional_accuracy_pct") is not None else "—"),
            ("Endpoint Hit %", fmt_pct(m.get("endpoint_hit_pct") if m.get("endpoint_hit_pct") is not None else m.get("hit_rate_pct")) if (m.get("endpoint_hit_pct") if m.get("endpoint_hit_pct") is not None else m.get("hit_rate_pct")) is not None else "—"),
            ("Premium MAE %", fmt_pct(m.get("premium_mae_pct"), 2) if m.get("premium_mae_pct") is not None else "—"),
            ("Premium RMSE %", fmt_pct(m.get("premium_rmse_pct"), 2) if m.get("premium_rmse_pct") is not None else "—"),
            ("Prediction bias", fmt_rupee(m.get("prediction_bias"))),
        ]
        data_table(
            content,
            [("metric", "Metric", 140), ("value", "Value", 120)],
            metric_rows,
            height=len(metric_rows) + 1,
        )

        section_title(content, "Premium-Band Performance", color=ACCENT)
        band_rows = []
        for b in day.get("premium_bands") or []:
            if not isinstance(b, dict):
                continue
            band_rows.append((
                b.get("band_label") or b.get("band") or "—",
                fmt_rows(b.get("samples")),
                fmt_rupee(b.get("mae")),
                fmt_rupee(b.get("rmse")),
                fmt_pct(b.get("premium_mae_pct"), 2) if b.get("premium_mae_pct") is not None else "—",
                fmt_pct(b.get("directional_accuracy_pct")) if b.get("directional_accuracy_pct") is not None else "—",
            ))
        if band_rows:
            data_table(
                content,
                [
                    ("band", "Band", 80),
                    ("n", "Rows", 70),
                    ("mae", "MAE", 70),
                    ("rmse", "RMSE", 70),
                    ("pmae", "Prem MAE %", 90),
                    ("dir", "Direction", 80),
                ],
                band_rows,
                height=min(10, len(band_rows) + 1),
            )

        section_title(content, "Volatility Statistics", color=ACCENT)
        v = day.get("volatility") or {}
        vol_rows = [
            ("Spot open → close", f"{fmt_num(v.get('spot_open'), 2)} → {fmt_num(v.get('spot_close'), 2)}"),
            ("Spot high / low", f"{fmt_num(v.get('spot_high'), 2)} / {fmt_num(v.get('spot_low'), 2)}"),
            ("Spot range", f"{fmt_rupee(v.get('spot_range'))} ({fmt_num(v.get('spot_range_pct'), 3)}%)" if v.get("spot_range") is not None else "—"),
            ("Spot return %", f"{float(v['spot_return_pct']):+.3f}%" if v.get("spot_return_pct") is not None else "—"),
            ("Spot std", fmt_num(v.get("spot_std"), 4) if v.get("spot_std") is not None else "—"),
            ("Premium mean / std", f"{fmt_rupee(v.get('premium_mean'))} / {fmt_rupee(v.get('premium_std'))}"),
            ("Premium CV %", fmt_num(v.get("premium_cv_pct"), 2) if v.get("premium_cv_pct") is not None else "—"),
            ("Premium range", fmt_rupee(v.get("premium_range"))),
            ("|IV z| mean / IV std", f"{fmt_num(v.get('iv_abs_mean'), 3)} / {fmt_num(v.get('iv_std'), 3)}"),
            ("Expiry day", "Yes" if v.get("is_expiry_day") else ("No" if v.get("is_expiry_day") is False else "—")),
        ]
        data_table(
            content,
            [("metric", "Metric", 160), ("value", "Value", 220)],
            vol_rows,
            height=len(vol_rows) + 1,
        )

        section_title(content, "Vs Average Training Day", color=ACCENT)
        vs_rows = []
        for row in (day.get("vs_training_day_avg") or {}).get("rows") or []:
            if not isinstance(row, dict):
                continue
            worse = row.get("worse")
            vs_rows.append((
                row.get("metric") or "—",
                fmt_num(row.get("holdout_day"), 4) if row.get("holdout_day") is not None else "—",
                fmt_num(row.get("train_day_avg"), 4) if row.get("train_day_avg") is not None else "—",
                f"{float(row['ratio']):.2f}×" if row.get("ratio") is not None else "—",
                fmt_num(row.get("delta"), 3) if row.get("delta") is not None else "—",
                "Worse" if worse else ("OK" if worse is False else "—"),
            ))
        if vs_rows:
            data_table(
                content,
                [
                    ("metric", "Metric", 130),
                    ("day", "Holdout Day", 90),
                    ("avg", "Train Avg", 90),
                    ("ratio", "Ratio", 60),
                    ("delta", "Delta", 70),
                    ("flag", "Flag", 60),
                ],
                vs_rows,
                height=min(12, len(vs_rows) + 1),
            )

        section_title(content, "Regime-Shift Assessment", color=ACCENT)
        flags = regime.get("flags") or []
        ttk.Label(
            content,
            text=f"Flags: {', '.join(flags) if flags else 'none'}",
            font=BODY_FONT,
        ).pack(anchor="w", pady=(2, 4))
        for reason in regime.get("reasons") or []:
            ttk.Label(content, text=f"• {reason}", foreground=COL_MUTED, wraplength=720).pack(anchor="w")


def _download_holdout_days_analysis_csv(report: dict[str, Any], model_name: str) -> None:
    from chain_replay_ml.training.holdout_days_analysis import build_holdout_days_analysis_csv

    analysis = report.get("holdout_days_analysis")
    if not isinstance(analysis, dict) or not analysis.get("ok"):
        messagebox.showwarning(
            "Holdout Days Analysis",
            "No holdout days analysis to export. Run Holdout Analyze, then Analyze in this tab.",
        )
        return
    safe_name = (model_name or "model").strip().replace(" ", "_")
    path = filedialog.asksaveasfilename(
        title="Save Holdout Days Analysis CSV",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialfile=f"{safe_name}_holdout_days_analysis.csv",
    )
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(build_holdout_days_analysis_csv(analysis))
    except OSError as exc:
        messagebox.showerror("Holdout Days Analysis", f"Could not save CSV:\n{exc}")
        return
    messagebox.showinfo("Holdout Days Analysis", f"Saved to\n{path}")


def _download_top1_analysis_csv(report: dict[str, Any], model_name: str) -> None:
    from chain_replay_ml.training.holdout_top1_analysis import build_top1_analysis_csv

    pa = report.get("premium_analysis") or {}
    analysis = pa.get("top1_analysis")
    if not isinstance(analysis, dict) or not analysis.get("ok"):
        messagebox.showwarning("Top 1% Analysis", "No investigation data to export. Run Analyze first.")
        return
    safe_name = (model_name or "model").strip().replace(" ", "_")
    path = filedialog.asksaveasfilename(
        title="Save Top 1% Investigation CSV",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialfile=f"{safe_name}_top1_investigation.csv",
    )
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(build_top1_analysis_csv(analysis))
    except OSError as exc:
        messagebox.showerror("Top 1% Analysis", f"Could not save CSV:\n{exc}")
        return
    messagebox.showinfo("Top 1% Analysis", f"Saved to\n{path}")


def _save_top1_knowledge(report: dict[str, Any], model_name: str, chart_dir: str) -> None:
    from chain_replay_ml.training.holdout_top1_analysis import save_top1_investigation_knowledge
    from .build_service import chart_data_dir

    pa = report.get("premium_analysis") or {}
    analysis = pa.get("top1_analysis")
    if not isinstance(analysis, dict) or not analysis.get("ok"):
        messagebox.showwarning("Knowledge", "Run holdout analysis first.")
        return
    data_dir = chart_data_dir(chart_dir)
    out = save_top1_investigation_knowledge(data_dir, analysis, model_name=model_name)
    if not out.get("ok"):
        messagebox.showerror("Knowledge", out.get("error") or "Failed to save")
        return
    finding = out.get("finding") or {}
    messagebox.showinfo(
        "Knowledge Saved",
        f"Finding saved to Knowledge Base.\n\n"
        f"Category: Model Weakness\n"
        f"Key: {out.get('finding_key') or '—'}\n"
        f"Status: {finding.get('status') or 'candidate'}",
    )


def _download_premium_analysis_csv(report: dict[str, Any], model_name: str) -> None:
    from chain_replay_ml.training.holdout_premium_analysis import build_premium_analysis_csv

    pa = report.get("premium_analysis")
    if not isinstance(pa, dict) or not pa:
        messagebox.showwarning("Premium Analysis", "No premium analysis data to export. Run Analyze first.")
        return
    safe_name = (model_name or "model").strip().replace(" ", "_")
    path = filedialog.asksaveasfilename(
        title="Save Premium Analysis CSV",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialfile=f"{safe_name}_premium_analysis.csv",
    )
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(build_premium_analysis_csv(pa))
    except OSError as exc:
        messagebox.showerror("Premium Analysis", f"Could not save CSV:\n{exc}")
        return
    messagebox.showinfo("Premium Analysis", f"Saved to\n{path}")


def render_holdout_performance(
    scroll: ScrollableFrame,
    doc: dict[str, Any],
    *,
    chart_dir: str,
    on_analyze: Callable[[Callable[[], dict[str, Any]], Callable[[dict[str, Any]], None]], None] | None = None,
) -> None:
    clear_children(scroll.inner)
    parent = scroll.inner

    if not doc.get("is_walk_forward"):
        ttk.Label(parent, text="Holdout analysis is only available for walk-forward models.", foreground=COL_MUTED).pack(anchor="w", pady=8)
        return

    content = ttk.Frame(parent)
    status_var = tk.StringVar(value="")
    toolbar = ttk.Frame(parent)
    toolbar.pack(fill="x", pady=(0, 4))
    ttk.Button(
        toolbar,
        text="Analyze",
        command=lambda: _run_holdout_analyze(content, status_var, toolbar, doc, chart_dir, on_analyze),
    ).pack(side="left")
    ttk.Label(toolbar, textvariable=status_var, foreground=COL_MUTED).pack(side="left", padx=12)
    content.pack(fill="both", expand=True)


def _download_holdout_performance_csv(report: dict[str, Any], model_name: str) -> None:
    from chain_replay_ml.training.holdout_performance import build_holdout_performance_csv

    if not report.get("ok"):
        messagebox.showwarning("Holdout Performance", "No holdout report to export. Run Analyze first.")
        return
    safe_name = (model_name or "model").strip().replace(" ", "_")
    path = filedialog.asksaveasfilename(
        title="Save Holdout Performance CSV",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        initialfile=f"{safe_name}_holdout_performance.csv",
    )
    if not path:
        return
    try:
        with open(path, "w", encoding="utf-8", newline="") as fh:
            fh.write(build_holdout_performance_csv(report))
    except OSError as exc:
        messagebox.showerror("Holdout Performance", f"Could not save CSV:\n{exc}")
        return
    messagebox.showinfo("Holdout Performance", f"Saved to\n{path}")


def _show_holdout_download_toolbar(toolbar: tk.Misc, report: dict[str, Any], model_name: str) -> None:
    for child in toolbar.winfo_children():
        child.destroy()
    ttk.Button(
        toolbar,
        text="Download CSV",
        command=lambda: _download_holdout_performance_csv(report, model_name),
    ).pack(side="right")


def _hide_holdout_toolbar(toolbar: tk.Misc) -> None:
    try:
        toolbar.destroy()
    except tk.TclError:
        pass


def _run_holdout_analyze(
    content: tk.Misc,
    status_var: tk.StringVar,
    toolbar: tk.Misc,
    doc: dict[str, Any],
    chart_dir: str,
    on_analyze: Callable[[Callable[[], dict[str, Any]], Callable[[dict[str, Any]], None]], None] | None,
) -> None:
    from .build_service import chart_data_dir

    clear_children(content)
    status_var.set("Analyzing holdout performance…")
    data_dir = chart_data_dir(chart_dir)
    model_name = str(doc.get("model_name") or "")

    def load() -> dict[str, Any]:
        from chain_replay_ml.training.holdout_performance import build_holdout_performance_report
        from chain_replay_ml.training.registry import load_model_detail

        detail = load_model_detail(data_dir, model_name) if model_name else doc
        return build_holdout_performance_report(data_dir, detail)

    def apply(report: dict[str, Any]) -> None:
        clear_children(content)
        status_var.set("")
        if not report.get("ok"):
            ttk.Label(content, text=report.get("error") or "Analysis failed", foreground=COL_WARN, wraplength=720).pack(anchor="w", pady=8)
            return
        _show_holdout_download_toolbar(toolbar, report, model_name)
        notebook = ttk.Notebook(content)
        notebook.pack(fill="both", expand=True)
        for tab_id, label in (
            ("overview", "Overview"),
            ("premium", "Premium Analysis"),
            ("top1", "Top 1% Error Analysis"),
            ("holdout_days", "Holdout Days Analysis"),
        ):
            frame = ttk.Frame(notebook)
            scroll = ScrollableFrame(frame)
            scroll.pack(fill="both", expand=True)
            notebook.add(frame, text=label)
            if tab_id == "overview":
                _render_holdout_report(scroll.inner, report)
            elif tab_id == "premium":
                _render_premium_analysis(scroll.inner, report, model_name=model_name)
            elif tab_id == "top1":
                _render_top1_error_analysis(scroll.inner, report, model_name=model_name, chart_dir=chart_dir)
            else:
                _render_holdout_days_analysis_tab(scroll.inner, report, model_name=model_name)

    if on_analyze is not None:
        on_analyze(load, apply)
        return

    import threading

    def worker() -> None:
        err: Exception | None = None
        result: dict[str, Any] | None = None
        try:
            result = load()
        except Exception as exc:
            err = exc

        def finish() -> None:
            status_var.set("")
            if err is not None:
                clear_children(content)
                ttk.Label(content, text=str(err), foreground=COL_WARN, wraplength=720).pack(anchor="w", pady=8)
                return
            if result is not None:
                apply(result)

        try:
            content.after(0, finish)
        except tk.TclError:
            pass

    threading.Thread(target=worker, daemon=True, name="holdout-analyze").start()
