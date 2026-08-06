"""Triple Barrier (TB) side-scorer filter for the Strategy Simulator.

Distinct from the Prediction Package Probability Filter (which gates on a
regression-ladder probability column). This filter gates entry rows by the
persisted Triple Barrier side-scorer columns written by the Prediction
Dataset builder:

    tb_pred_class == selected_class_id  AND  tb_pred_probability >= threshold

Both columns are read-only here — this module never runs TB inference, never
writes to the Prediction Dataset, and never creates new tables. Class ids are
never hardcoded: they are resolved from the TB model's on-disk metadata
(``config.json`` -> ``label_run_id`` -> Label Registry ``label_encoding``),
falling back to the model config's own ``label_encoding`` field, and only as
a last resort to the canonical Triple Barrier strategy encoding (marked as
such via ``source`` so callers can tell a real mapping from a default one).
"""

from __future__ import annotations

import json
import os
from typing import Any

TB_DISABLED = "Disabled"
TB_PROB_COLUMN = "tb_pred_probability"
TB_CLASS_COLUMN = "tb_pred_class"
MISSING_TB_REASON = "Missing Triple Barrier prediction"

# Fallback when the caller has not chosen an operating threshold yet.
DEFAULT_TB_THRESHOLD = 0.60

# TB Probability Distribution buckets (Part C, optional): 0.50-0.60 ... 0.90-1.00.
_PROB_BUCKETS: tuple[tuple[float, float], ...] = tuple(
    (round(0.5 + i * 0.1, 2), round(0.6 + i * 0.1, 2)) for i in range(5)
)


def _bucket_label(lo: float, hi: float) -> str:
    return f"{lo:.2f}-{hi:.2f}"


def _bucket_for(value: float) -> str | None:
    if value < _PROB_BUCKETS[0][0]:
        return None
    for lo, hi in _PROB_BUCKETS:
        if (lo <= value < hi) or (hi >= 1.0 and lo <= value <= 1.0):
            return _bucket_label(lo, hi)
    return None


def empty_probability_distribution() -> dict[str, int]:
    return {_bucket_label(lo, hi): 0 for lo, hi in _PROB_BUCKETS}


def _load_json(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def resolve_tb_class_options(data_dir: str, tb_model_name: str | None) -> dict[str, Any]:
    """Resolve class_id -> label for a Triple Barrier model, from metadata only.

    Returns ``{"ok", "tb_model_name", "label_run_id", "classes": [{"class_id", "label"}, ...],
    "source"}``. ``source`` is one of ``label_run``, ``model_config`` or
    ``default_triple_barrier_encoding`` — the last is only used when no
    persisted metadata is found and is never silently treated as authoritative
    by callers (they should surface ``source`` in the UI).
    """
    name = str(tb_model_name or "").strip()
    if not name:
        return {
            "ok": False,
            "error": "No Triple Barrier model selected",
            "tb_model_name": None,
            "label_run_id": None,
            "classes": [],
            "source": None,
        }

    from chain_replay_ml.training.model_runtime import resolve_prediction_model_package
    from chain_replay_ml.training.paths import model_artifact_paths

    pkg = resolve_prediction_model_package(data_dir, name)
    label_run_id = str(pkg.get("label_run_id") or "").strip() or None

    encoding: dict[str, int] | None = None
    source: str | None = None

    if label_run_id:
        try:
            from chain_replay_ml.label_runs.registry import get_label_run

            rec = get_label_run(data_dir, label_run_id)
            if rec.exists and isinstance(rec.label_encoding, dict) and rec.label_encoding:
                encoding = {str(k): int(v) for k, v in rec.label_encoding.items()}
                source = "label_run"
        except Exception:
            encoding = None

    if not encoding:
        paths = model_artifact_paths(data_dir, name)
        cfg = _load_json(paths["config_json"])
        enc = cfg.get("label_encoding") if isinstance(cfg, dict) else None
        if isinstance(enc, dict) and enc:
            try:
                encoding = {str(k): int(v) for k, v in enc.items()}
                source = "model_config"
            except (TypeError, ValueError):
                encoding = None

    if not encoding:
        from chain_replay_ml.outcome_label_engine.triple_barrier import LABEL_ENCODING

        encoding = dict(LABEL_ENCODING)
        source = "default_triple_barrier_encoding"

    classes = [
        {"class_id": int(cid), "label": str(label)}
        for label, cid in sorted(encoding.items(), key=lambda kv: kv[1])
    ]
    return {
        "ok": True,
        "tb_model_name": name,
        "label_run_id": label_run_id,
        "classes": classes,
        "source": source,
    }


def discover_tb_model_name(lab_db_path: str) -> dict[str, Any]:
    """Read the persisted ``tb_model_name`` off the loaded lab's Prediction Dataset.

    Read-only SQL preview — never infers or guesses. If a lab was built with a
    TB scorer selected, every row shares the same ``tb_model_name``; this
    picks the most common non-null value observed (defensive against mixed
    builds) and reports how many distinct values were found.
    """
    from chain_replay_ml.model_lab.store import ModelLabStore

    if not lab_db_path or not os.path.isfile(lab_db_path):
        return {"tb_model_name": None, "distinct_count": 0, "note": "Prediction Dataset not found."}

    with ModelLabStore(lab_db_path) as store:
        store.ensure_prediction_schema()
        available = store._prediction_table_columns()
        if TB_CLASS_COLUMN not in available or "tb_model_name" not in available:
            return {
                "tb_model_name": None,
                "distinct_count": 0,
                "note": "Prediction Dataset predates Triple Barrier columns — rebuild to enable.",
            }
        rows = store.conn.execute(
            "SELECT tb_model_name, COUNT(*) AS c FROM prediction_dataset "
            "WHERE tb_model_name IS NOT NULL AND tb_model_name != '' "
            "GROUP BY tb_model_name ORDER BY c DESC"
        ).fetchall()

    if not rows:
        return {
            "tb_model_name": None,
            "distinct_count": 0,
            "note": "No Triple Barrier scorer was selected when this Prediction Dataset was built.",
        }
    top_name = str(rows[0][0])
    note = (
        f"{len(rows)} Triple Barrier model name(s) found on this dataset — using the most common ({top_name})."
        if len(rows) > 1
        else ""
    )
    return {"tb_model_name": top_name, "distinct_count": len(rows), "note": note}


def tb_filter_options(data_dir: str, lab_db_path: str) -> dict[str, Any]:
    """Combine lab discovery + metadata resolution for the Sim panel dropdown/radios."""
    disc = discover_tb_model_name(lab_db_path)
    tb_model_name = disc.get("tb_model_name")
    if not tb_model_name:
        return {
            "available": False,
            "tb_model_name": None,
            "label_run_id": None,
            "classes": [],
            "source": None,
            "note": disc.get("note") or "No Triple Barrier predictions in this Prediction Dataset.",
        }
    resolved = resolve_tb_class_options(data_dir, tb_model_name)
    return {
        "available": bool(resolved.get("ok")) and bool(resolved.get("classes")),
        "tb_model_name": tb_model_name,
        "label_run_id": resolved.get("label_run_id"),
        "classes": resolved.get("classes") or [],
        "source": resolved.get("source"),
        "note": disc.get("note") or "",
    }


def normalize_tb_threshold(value: Any) -> float:
    try:
        thr = float(value)
    except (TypeError, ValueError):
        return DEFAULT_TB_THRESHOLD
    if thr != thr:  # NaN
        return DEFAULT_TB_THRESHOLD
    return max(0.0, min(1.0, round(thr, 4)))


def _empty_tb_meta() -> dict[str, Any]:
    return {
        "active": False,
        "class_id": None,
        "label": TB_DISABLED,
        "threshold": None,
        "rows_before": 0,
        "rows_after": 0,
        "rows_removed": 0,
        "rows_null": 0,
        "skip_reason": MISSING_TB_REASON,
        "class_counts": {},
        "avg_tb_probability": None,
        "probability_distribution": {},
    }


def apply_tb_filter(
    rows: list[dict[str, Any]],
    *,
    class_id: int | None,
    threshold: Any,
    label: str | None = None,
    class_labels: dict[int, str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep rows whose Triple Barrier prediction matches ``class_id`` at/above ``threshold``.

    Pipeline stage: Probability Filter -> Triple Barrier Filter -> Confidence
    Filter -> Strategy Rules (AND of independent predicates — final trade set
    is order-independent). ``class_id=None`` means the filter is disabled and
    rows pass through unchanged (Acceptance: TB disabled -> identical results).

    Rows with NULL ``tb_pred_class`` / ``tb_pred_probability`` are always
    excluded when the filter is active, and counted under ``rows_null`` with
    ``skip_reason`` = "Missing Triple Barrier prediction" — they are never
    guessed into a class.
    """
    meta = _empty_tb_meta()
    meta["rows_before"] = len(rows)
    meta["rows_after"] = len(rows)
    if class_id is None:
        return rows, meta

    cid = int(class_id)
    thr = normalize_tb_threshold(threshold)
    labels_map = {int(k): str(v) for k, v in (class_labels or {}).items()}
    class_label = str(label or labels_map.get(cid) or f"class_{cid}")
    meta.update({"active": True, "class_id": cid, "label": class_label, "threshold": thr})

    kept: list[dict[str, Any]] = []
    nulls = 0
    class_counts: dict[str, int] = {}
    prob_sum = 0.0
    prob_n = 0
    buckets = empty_probability_distribution()

    for row in rows:
        raw_class = row.get(TB_CLASS_COLUMN)
        raw_prob = row.get(TB_PROB_COLUMN)
        if raw_class is None or raw_prob is None:
            nulls += 1
            continue
        try:
            row_class = int(float(raw_class))
            row_prob = float(raw_prob)
        except (TypeError, ValueError):
            nulls += 1
            continue
        if row_prob != row_prob:  # NaN
            nulls += 1
            continue

        row_label = labels_map.get(row_class, f"class_{row_class}")
        class_counts[row_label] = class_counts.get(row_label, 0) + 1

        if row_class != cid:
            continue
        bucket = _bucket_for(row_prob)
        if bucket is not None:
            buckets[bucket] += 1
        if row_prob >= thr:
            kept.append(row)
            prob_sum += row_prob
            prob_n += 1

    meta["rows_null"] = nulls
    meta["rows_after"] = len(kept)
    meta["rows_removed"] = max(0, len(rows) - len(kept))
    meta["class_counts"] = class_counts
    meta["avg_tb_probability"] = round(prob_sum / prob_n, 4) if prob_n else None
    meta["probability_distribution"] = buckets

    if nulls == len(rows) and rows:
        raise ValueError(
            "Triple Barrier prediction columns are empty on this Prediction Dataset. "
            "Rebuild the Prediction Dataset with a Triple Barrier scorer selected, then retry."
        )
    if not kept:
        raise ValueError(
            f"Triple Barrier filter {class_label} >= {thr:.2f} removed all "
            f"{len(rows):,} prediction rows. Lower the threshold or pick a different class."
        )
    return kept, meta


def tb_row_summary(
    lab_db_path: str,
    *,
    class_id: int | None,
    threshold: Any,
    trading_days: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> dict[str, Any]:
    """Fast SQL preview of how the TB filter would thin the dataset."""
    from chain_replay_ml.model_lab.store import ModelLabStore
    from chain_replay_ml.strategy_simulator.lab_source import _day_filter_sql

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
        if class_id is None:
            return {
                "ok": True,
                "active": False,
                "label": TB_DISABLED,
                "class_id": None,
                "threshold": None,
                "prediction_rows": total,
                "rows_kept": total,
                "rows_removed": 0,
                "rows_kept_pct": 100.0 if total else 0.0,
                "rows_removed_pct": 0.0,
                "rows_null": 0,
            }

        available = store._prediction_table_columns()
        if TB_CLASS_COLUMN not in available or TB_PROB_COLUMN not in available:
            return {
                "ok": False,
                "error": "Triple Barrier columns missing — rebuild the Prediction Dataset.",
                "prediction_rows": total,
            }

        thr = normalize_tb_threshold(threshold)
        where_keep = f'"{TB_CLASS_COLUMN}" = ? AND "{TB_PROB_COLUMN}" >= ?'
        where_null = f'("{TB_CLASS_COLUMN}" IS NULL OR "{TB_PROB_COLUMN}" IS NULL)'
        if where_sql:
            where_keep = f"({where_sql}) AND {where_keep}"
            where_null = f"({where_sql}) AND {where_null}"
        kept = int(
            store.conn.execute(
                f"SELECT COUNT(*) FROM prediction_dataset WHERE {where_keep}",
                list(where_args) + [int(class_id), thr],
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
            "label": str(class_id),
            "class_id": int(class_id),
            "threshold": thr,
            "prediction_rows": total,
            "rows_kept": kept,
            "rows_removed": removed,
            "rows_null": nulls,
            "rows_kept_pct": round(100.0 * kept / total, 2) if total else 0.0,
            "rows_removed_pct": round(100.0 * removed / total, 2) if total else 0.0,
        }
