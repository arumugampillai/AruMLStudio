"""Prediction Package probability filter for the Strategy Simulator.

Distinct from the Confidence "Classifier Filter" (which keeps rows by a hard
``*_pred == 1`` label for strategy success/failure metadata). This filter gates
entry rows by a Probability Ladder member's stored probability column:

    pred_prob_up_2pct_5m >= selected_threshold

Probabilities live in the Prediction Dataset; the operating threshold belongs to
the strategy run, so changing it never requires rebuilding predictions.
"""

from __future__ import annotations

import json
import os
from typing import Any

from chain_replay_ml.training.paths import model_artifact_paths
from chain_replay_ml.training.prediction_packages import (
    PROBABILITY_OUTPUT_COLUMNS,
    discover_prediction_package_members,
)

PROBABILITY_DISABLED = "Disabled"

# Fallback when a member has no persisted threshold sweep.
DEFAULT_THRESHOLD = 0.50


def _load_json(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def probability_filter_options(
    data_dir: str,
    *,
    dataset: str,
    anchor_target: str,
    anchor_model_name: str | None = None,
) -> list[dict[str, Any]]:
    """Available Probability Ladder members for the currently loaded package.

    Never hardcoded: only members that actually have a trained model in this
    package are returned, in canonical ladder order.
    """
    members = discover_prediction_package_members(
        data_dir,
        dataset=dataset,
        anchor_target=anchor_target,
        anchor_model_name=anchor_model_name,
    )
    options: list[dict[str, Any]] = []
    for member in members:
        if not member.get("available") or not member.get("model_name"):
            continue
        options.append(
            {
                "key": str(member.get("key") or ""),
                "label": f"{member.get('label')} Probability",
                "ladder_label": str(member.get("label") or ""),
                "column": str(member.get("output_column") or ""),
                "model_name": str(member.get("model_name") or ""),
                "target": str(member.get("target") or ""),
                "role": str(member.get("role") or ""),
            }
        )
    return options


def probability_filter_labels(options: list[dict[str, Any]]) -> list[str]:
    return [PROBABILITY_DISABLED, *[str(o["label"]) for o in options]]


def option_from_label(
    options: list[dict[str, Any]],
    label: str,
) -> dict[str, Any] | None:
    text = str(label or "").strip()
    if not text or text == PROBABILITY_DISABLED:
        return None
    for option in options:
        if str(option.get("label")) == text:
            return option
    return None


def member_threshold_analysis(
    data_dir: str,
    model_name: str,
) -> tuple[list[dict[str, Any]], str, float | None, float | None]:
    """Return (threshold rows, source label, decision threshold, ROC-AUC).

    Reads the member package's ``metrics.json`` directly — same precedence the
    Model Registry Threshold Analysis tab uses (production walk-forward first).
    """
    from chain_replay_ml.training.evaluator import (
        attach_trades_per_day,
        normalize_threshold_analysis_rows,
    )
    from chain_replay_ml.training.lifecycle_store import _extract_trading_days

    name = str(model_name or "").strip()
    if not name:
        return [], "unavailable", None, None

    paths = model_artifact_paths(data_dir, name)
    metrics = _load_json(paths["metrics_json"])
    if not metrics:
        return [], "unavailable", None, None

    candidates: list[tuple[str, dict[str, Any]]] = []
    for key, label in (
        ("production_walk_forward", "Production walk-forward"),
        ("walk_forward", "Walk-forward"),
        ("validation", "Training validation"),
        ("test", "Holdout test"),
    ):
        block = metrics.get(key)
        if isinstance(block, dict):
            candidates.append((label, block))
    candidates.append(("Persisted metrics", metrics))

    def _num(value: Any) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    decision_threshold: float | None = None
    roc_auc: float | None = None
    for _label, block in candidates:
        if decision_threshold is None:
            decision_threshold = _num(block.get("threshold"))
        if roc_auc is None:
            roc_auc = _num(block.get("roc_auc") or block.get("mean_roc_auc"))
    if decision_threshold is not None:
        decision_threshold = round(decision_threshold, 2)

    n_days: int | None = None
    try:
        meta = _load_json(os.path.join(paths["package_dir"], "metadata.json"))
        cfg = _load_json(paths["config_json"])
        n_days = _extract_trading_days(
            meta if isinstance(meta, dict) else {},
            cfg if isinstance(cfg, dict) else {},
            data_dir=data_dir,
            dataset_name=(cfg or {}).get("dataset") if isinstance(cfg, dict) else None,
        )
    except Exception:
        n_days = None

    for label, block in candidates:
        raw_rows = block.get("threshold_analysis")
        if isinstance(raw_rows, list) and raw_rows:
            return (
                attach_trades_per_day(normalize_threshold_analysis_rows(raw_rows), n_days),
                label,
                decision_threshold,
                roc_auc,
            )
    return [], "unavailable", decision_threshold, roc_auc


def threshold_composite_scores(
    rows: list[dict[str, Any]],
    *,
    roc_auc: float | None = None,
) -> dict[float, float]:
    """Score every threshold row with the trader-facing classification composite.

    Same weighting the Model Builder uses to pick champions (Precision 40%,
    F1 30%, ROC-AUC 20%, Recall 10%), evaluated per operating point. ROC-AUC is
    threshold-independent, so it only shifts every row by a constant.
    """
    from chain_replay_ml.training.objective_scoring import classification_composite_score

    scores: dict[float, float] = {}
    for row in rows or []:
        try:
            thr = round(float(row.get("threshold")), 2)
        except (TypeError, ValueError):
            continue
        if row.get("precision_pct") is None and row.get("f1_pct") is None:
            continue
        scores[thr] = round(
            float(
                classification_composite_score(
                    {
                        "precision_pct": row.get("precision_pct"),
                        "f1_pct": row.get("f1_pct"),
                        "recall_pct": row.get("recall_pct"),
                        "roc_auc": roc_auc,
                    }
                )
            ),
            6,
        )
    return scores


def recommended_threshold(
    rows: list[dict[str, Any]],
    *,
    decision_threshold: float | None = None,
    roc_auc: float | None = None,
) -> tuple[float, str]:
    """Pick the member's recommended operating threshold — Best Composite.

    Precision is weighted highest because a false positive here is a losing
    trade, so the recommendation usually sits above 0.50. Ties break toward the
    lower threshold (more signals). Falls back to the persisted decision
    threshold, then 0.50.
    """
    scores = threshold_composite_scores(rows, roc_auc=roc_auc)
    if scores:
        best_thr = max(scores, key=lambda thr: (scores[thr], -thr))
        return round(best_thr, 2), "Best Composite (Precision 40% · F1 30% · AUC 20% · Recall 10%)"
    if decision_threshold is not None:
        return round(float(decision_threshold), 2), "Model decision threshold"
    return DEFAULT_THRESHOLD, "Default (no threshold sweep persisted)"


def resolve_member_threshold_defaults(
    data_dir: str,
    model_name: str,
) -> dict[str, Any]:
    """Threshold rows + recommended operating point for one ladder member."""
    rows, source, decision, roc_auc = member_threshold_analysis(data_dir, model_name)
    thr, criterion = recommended_threshold(
        rows,
        decision_threshold=decision,
        roc_auc=roc_auc,
    )
    return {
        "model_name": str(model_name or ""),
        "rows": rows,
        "source": source,
        "decision_threshold": decision,
        "roc_auc": roc_auc,
        "composite_scores": threshold_composite_scores(rows, roc_auc=roc_auc),
        "recommended_threshold": thr,
        "recommended_criterion": criterion,
    }


def normalize_probability_threshold(value: Any) -> float:
    try:
        thr = float(value)
    except (TypeError, ValueError):
        return DEFAULT_THRESHOLD
    if thr != thr:  # NaN
        return DEFAULT_THRESHOLD
    return max(0.0, min(1.0, round(thr, 4)))


def apply_probability_filter(
    rows: list[dict[str, Any]],
    *,
    column: str | None,
    threshold: Any,
    label: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep rows whose probability column is >= threshold.

    Pipeline stage: Prediction Dataset → Probability Filter → Strategy Rules.
    NULL probabilities (member missing for that row) never pass.
    """
    col = str(column or "").strip()
    meta: dict[str, Any] = {
        "active": False,
        "label": PROBABILITY_DISABLED,
        "column": None,
        "threshold": None,
        "rows_before": len(rows),
        "rows_after": len(rows),
        "rows_removed": 0,
        "rows_null": 0,
    }
    if not col:
        return rows, meta

    thr = normalize_probability_threshold(threshold)
    meta.update(
        {
            "active": True,
            "label": str(label or col),
            "column": col,
            "threshold": thr,
        }
    )

    kept: list[dict[str, Any]] = []
    nulls = 0
    for row in rows:
        raw = row.get(col)
        if raw is None:
            nulls += 1
            continue
        try:
            val = float(raw)
        except (TypeError, ValueError):
            nulls += 1
            continue
        if val != val:
            nulls += 1
            continue
        if val >= thr:
            kept.append(row)

    meta["rows_null"] = nulls
    meta["rows_after"] = len(kept)
    meta["rows_removed"] = max(0, len(rows) - len(kept))

    if nulls == len(rows) and rows:
        raise ValueError(
            f"Probability column {col} is empty on this Prediction Dataset. "
            "Rebuild the Prediction Dataset so package member probabilities are stored, "
            "then retry."
        )
    if not kept:
        raise ValueError(
            f"Probability filter {meta['label']} >= {thr:.2f} removed all "
            f"{len(rows):,} prediction rows. Lower the threshold and retry."
        )
    return kept, meta


def probability_row_summary(
    lab_db_path: str,
    *,
    column: str | None,
    threshold: Any,
    trading_days: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Fast SQL preview of how the probability filter would thin the dataset."""
    from chain_replay_ml.model_lab.store import ModelLabStore

    from .lab_source import _day_filter_sql

    col = str(column or "").strip()
    where_sql, where_args = _day_filter_sql(
        trading_days=trading_days,
        date_from=date_from,
        date_to=date_to,
    )
    clause = f"WHERE {where_sql}" if where_sql else ""

    with ModelLabStore(lab_db_path) as store:
        store.ensure_prediction_schema()
        total = int(
            store.conn.execute(
                f"SELECT COUNT(*) FROM prediction_dataset {clause}",
                where_args,
            ).fetchone()[0]
            or 0
        )
        if not col:
            return {
                "ok": True,
                "active": False,
                "label": PROBABILITY_DISABLED,
                "column": None,
                "threshold": None,
                "prediction_rows": total,
                "rows_kept": total,
                "rows_removed": 0,
                "rows_kept_pct": 100.0 if total else 0.0,
                "rows_removed_pct": 0.0,
                "rows_null": 0,
            }

        if col not in PROBABILITY_OUTPUT_COLUMNS:
            return {"ok": False, "error": f"Unknown probability column: {col}"}
        if col not in set(store._prediction_table_columns()):
            return {
                "ok": False,
                "error": f"Column {col} missing — rebuild the Prediction Dataset.",
                "column": col,
                "prediction_rows": total,
            }

        thr = normalize_probability_threshold(threshold)
        where_keep = f'"{col}" >= ?'
        where_null = f'"{col}" IS NULL'
        if where_sql:
            where_keep = f"({where_sql}) AND {where_keep}"
            where_null = f"({where_sql}) AND {where_null}"
        kept = int(
            store.conn.execute(
                f"SELECT COUNT(*) FROM prediction_dataset WHERE {where_keep}",
                list(where_args) + [thr],
            ).fetchone()[0]
            or 0
        )
        nulls = int(
            store.conn.execute(
                f"SELECT COUNT(*) FROM prediction_dataset WHERE {where_null}",
                where_args,
            ).fetchone()[0]
            or 0
        )
        removed = max(0, total - kept)
        return {
            "ok": True,
            "active": True,
            "label": col,
            "column": col,
            "threshold": thr,
            "prediction_rows": total,
            "rows_kept": kept,
            "rows_removed": removed,
            "rows_null": nulls,
            "rows_kept_pct": round(100.0 * kept / total, 2) if total else 0.0,
            "rows_removed_pct": round(100.0 * removed / total, 2) if total else 0.0,
        }
