"""SQLite store for model family registry (current champion) and full training history.

Evaluation metrics (MAE, RMSE, Direction, Composite, etc.) are **not** stored or
trusted from this database. Package ``metrics.json`` is the single authoritative
source, resolved via ``registry._resolve_authoritative_metrics``.

Legacy metric columns / JSON blobs may still exist in the schema for backward
compatibility but are deprecated: writers leave them null/empty and readers
ignore them in favor of on-disk package resolution.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import datetime, timezone
from typing import Any

from .paths import model_artifact_paths, model_package_dir, models_dir, safe_model_name

LIFECYCLE_LABELS = {
    "new_model": "New Model",
    "retrain": "Retrain",
    "complete_optimization": "Complete Optimization",
    "feature_optimization": "Feature Optimization",
    "calibration": "Calibration",
    "rollback": "Rollback",
}

# Deprecated: retained in SQLite schema for backward compatibility only.
# Do not write meaningful values; do not use for UI display.
DEPRECATED_LIFECYCLE_METRIC_KEYS: tuple[str, ...] = (
    "mae",
    "rmse",
    "directional_accuracy_pct",
    "composite_score",
    "premium_mae_pct",
    "premium_rmse_pct",
    "medae",
    "p95_error",
    "prediction_bias",
    "prediction_bias_pct",
    "premium_band_performance",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def lifecycle_db_path(data_dir: str) -> str:
    return os.path.join(models_dir(data_dir), ".lifecycle_registry.db")


def _connect(data_dir: str) -> sqlite3.Connection:
    path = lifecycle_db_path(data_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    ensure_lifecycle_tables(conn)
    return conn


def ensure_lifecycle_tables(conn: sqlite3.Connection) -> None:
    # Metric columns / current_metrics_json / metrics_json are DEPRECATED.
    # Kept for schema compatibility with existing DBs; new writes leave them null/empty.
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS model_registry (
            model_id TEXT PRIMARY KEY,
            display_name TEXT,
            current_model_name TEXT NOT NULL,
            current_version TEXT,
            current_version_number INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'ready',
            created_on TEXT NOT NULL,
            updated_on TEXT NOT NULL,
            current_metrics_json TEXT
        );
        CREATE TABLE IF NOT EXISTS model_history (
            history_id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_history_id INTEGER,
            model_id TEXT NOT NULL,
            model_name TEXT NOT NULL UNIQUE,
            version_label TEXT NOT NULL,
            version_number INTEGER NOT NULL,
            lifecycle TEXT NOT NULL,
            parent_model_name TEXT,
            trained_at TEXT NOT NULL,
            dataset TEXT,
            target TEXT,
            algorithm TEXT,
            validation_strategy TEXT,
            row_count INTEGER,
            trading_days INTEGER,
            feature_count INTEGER,
            mae REAL,
            rmse REAL,
            directional_accuracy_pct REAL,
            composite_score REAL,
            premium_mae_pct REAL,
            premium_rmse_pct REAL,
            medae REAL,
            p95_error REAL,
            prediction_bias REAL,
            prediction_bias_pct REAL,
            hpo_trials INTEGER,
            parameters_changed INTEGER,
            changes_json TEXT,
            metrics_json TEXT,
            FOREIGN KEY (parent_history_id) REFERENCES model_history(history_id)
        );
        CREATE INDEX IF NOT EXISTS idx_model_history_model_id
            ON model_history(model_id, version_number);
        CREATE INDEX IF NOT EXISTS idx_model_history_parent
            ON model_history(parent_model_name);
        """
    )
    migrate_lifecycle_schema_v2(conn)
    conn.commit()


def migrate_lifecycle_schema_v2(conn: sqlite3.Connection) -> None:
    """Idempotently ensure Phase 4C.2 taxonomy columns and indexes in lifecycle registry."""
    # 1. Premium columns in model_history (existing backward compatibility)
    history_cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(model_history)").fetchall()}
    for col in (
        "premium_mae_pct",
        "premium_rmse_pct",
        "medae",
        "p95_error",
        "prediction_bias",
        "prediction_bias_pct",
    ):
        if col not in history_cols:
            conn.execute(f"ALTER TABLE model_history ADD COLUMN {col} REAL")

    # 2. Phase 4C.2 Taxonomy columns in model_history
    taxonomy_history_additions = [
        ("task_type", "TEXT DEFAULT 'DIRECTION_CLASSIFIER'"),
        ("regime_id", "TEXT DEFAULT 'R000'"),
        ("regime_name", "TEXT DEFAULT 'ALL_REGIMES'"),
        ("population", "TEXT DEFAULT 'EXPERIMENTAL'"),
        ("status", "TEXT DEFAULT 'ACTIVE'"),
        ("context_key", "TEXT"),
        ("package_model_id", "TEXT"),
        ("metadata_json", "TEXT"),
    ]
    for col_name, col_def in taxonomy_history_additions:
        if col_name not in history_cols:
            conn.execute(f"ALTER TABLE model_history ADD COLUMN {col_name} {col_def}")

    # 3. Phase 4C.2 Taxonomy columns in model_registry
    registry_cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(model_registry)").fetchall()}
    taxonomy_registry_additions = [
        ("task_type", "TEXT DEFAULT 'DIRECTION_CLASSIFIER'"),
        ("regime_id", "TEXT DEFAULT 'R000'"),
        ("context_key", "TEXT"),
        ("champion_model_name", "TEXT"),
        ("challenger_model_name", "TEXT"),
        ("regime_scope", "TEXT DEFAULT 'ALL_REGIMES'"),
    ]
    for col_name, col_def in taxonomy_registry_additions:
        if col_name not in registry_cols:
            conn.execute(f"ALTER TABLE model_registry ADD COLUMN {col_name} {col_def}")

    # 4. Composite query indexes
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_model_history_context ON model_history(context_key, population, status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_model_history_regime ON model_history(regime_id, task_type)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_model_registry_context ON model_registry(context_key)"
    )


def _ensure_history_premium_columns(conn: sqlite3.Connection) -> None:
    migrate_lifecycle_schema_v2(conn)


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def _json_loads(raw: str | None) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _history_row_dict(row: sqlite3.Row) -> dict[str, Any]:
    """Map a history DB row to a dict, stripping deprecated stored metrics.

    Evaluation metrics are filled later from the package via
    ``_enrich_history_row_from_disk``.
    """
    doc = dict(row)
    doc["changes"] = _json_loads(doc.pop("changes_json", None)) or {}
    # Pop and discard deprecated metric blob — do not promote to display fields.
    _json_loads(doc.pop("metrics_json", None))
    doc["metrics"] = {}
    deprecated: dict[str, Any] = {}
    for key in DEPRECATED_LIFECYCLE_METRIC_KEYS:
        if key in doc and doc[key] is not None:
            deprecated[key] = doc[key]
        doc[key] = None
    if deprecated:
        doc["_deprecated_db_metrics"] = deprecated
    doc["lifecycle_label"] = LIFECYCLE_LABELS.get(str(doc.get("lifecycle") or ""), doc.get("lifecycle"))
    model_name = str(doc.get("model_name") or "")
    if model_name:
        # Package path is metadata only (not a metric).
        # data_dir is unknown here; filled during enrich.
        doc.setdefault("package_name", model_name)
    return doc


def _num_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, dict):
        for key in ("score", "value", "composite_score", "mean_composite_score"):
            if key in value:
                nested = _num_or_none(value[key])
                if nested is not None:
                    return nested
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    if num != num:  # NaN
        return None
    return num


def _resolve_production_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Deprecated shim — prefer ``_read_package_production_metrics`` / authoritative resolver.

    Kept for unit tests that pass an in-memory metrics dict without a package.
    """
    from .registry import _resolve_authoritative_metrics

    return _resolve_authoritative_metrics(
        strategy={"key": "time_series", "label": "Time Series Split"},
        metrics_doc=metrics if isinstance(metrics, dict) else {},
        summary_doc={},
        wf_summary_doc=None,
    )


def _trading_days_from_dataset_name(data_dir: str, dataset_name: str | None) -> int | None:
    if not dataset_name:
        return None
    try:
        from ..dataset_builder.auditor import list_datasets

        target = str(dataset_name).strip()
        for row in list_datasets(data_dir):
            if str(row.get("dataset_name") or "").strip() == target:
                day_count = row.get("day_count")
                if day_count is not None:
                    return int(day_count)
    except Exception:
        return None
    return None


def _extract_trading_days(
    metadata: dict[str, Any],
    config_doc: dict[str, Any] | None = None,
    *,
    data_dir: str | None = None,
    dataset_name: str | None = None,
) -> int | None:
    for src in (metadata, (config_doc or {}).get("dataset_metadata") or {}, config_doc or {}):
        if not isinstance(src, dict):
            continue
        for key in ("trading_days", "day_count", "n_trading_days"):
            if src.get(key) is not None:
                try:
                    return int(src[key])
                except (TypeError, ValueError):
                    pass
        days = src.get("days")
        if isinstance(days, list) and days:
            return len(days)
    if data_dir:
        ds_name = dataset_name or (config_doc or {}).get("dataset") or metadata.get("dataset")
        from_registry = _trading_days_from_dataset_name(data_dir, ds_name)
        if from_registry is not None:
            return from_registry
    return None


def _read_json_dict(path: str) -> dict[str, Any] | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _read_package_production_metrics(data_dir: str, model_name: str) -> dict[str, Any]:
    """Resolve production metrics from package metrics.json (authoritative).

    Never calls ``load_model_detail`` (avoids recursion). Uses
    ``registry._resolve_authoritative_metrics`` so Lifecycle matches Overview/list.
    """
    from .registry import (
        _detect_validation_strategy,
        _load_json_artifact,
        _resolve_authoritative_metrics,
    )

    safe = safe_model_name(model_name)
    paths = model_artifact_paths(data_dir, safe)
    metrics_art = _load_json_artifact(paths["metrics_json"])
    metrics = metrics_art.get("data") if isinstance(metrics_art.get("data"), dict) else {}
    if not metrics:
        return {}

    cfg_art = _load_json_artifact(paths["config_json"])
    cfg = cfg_art.get("data") if isinstance(cfg_art.get("data"), dict) else {}
    summary_art = _load_json_artifact(paths["training_summary_json"])
    summary = summary_art.get("data") if isinstance(summary_art.get("data"), dict) else {}

    wf_summary: dict[str, Any] | None = None
    wf_summary_path = os.path.join(paths["package_dir"], "walk_forward", "summary.json")
    wf_raw = _read_json_dict(wf_summary_path)
    if wf_raw:
        wf_summary = wf_raw

    strategy = _detect_validation_strategy(cfg) if cfg else {"key": "time_series", "label": "Time Series Split"}
    pred_type = str(
        (cfg.get("prediction_type") if isinstance(cfg, dict) else None)
        or metrics.get("prediction_type")
        or "regression"
    ).strip().lower()
    refs = dict((wf_summary or {}).get("reference_stats") or {}) or None

    return _resolve_authoritative_metrics(
        strategy=strategy,
        metrics_doc=metrics,
        summary_doc=summary,
        wf_summary_doc=wf_summary,
        prediction_type=pred_type,
        score_refs=refs,
    )


def _apply_package_metrics_to_history_row(row: dict[str, Any], prod: dict[str, Any]) -> dict[str, Any]:
    """Overwrite deprecated/empty metric fields with package-resolved production metrics."""
    out = dict(row)
    for key in DEPRECATED_LIFECYCLE_METRIC_KEYS:
        if key in prod:
            out[key] = prod.get(key)
    out["production_metrics"] = prod
    out["metrics"] = {"production": {k: prod.get(k) for k in DEPRECATED_LIFECYCLE_METRIC_KEYS if k in prod}}
    out["metrics_source"] = "package"
    if prod.get("stage_key"):
        out["metrics_stage_key"] = prod.get("stage_key")
        out["metrics_source_path"] = prod.get("source_path")
    return out


def _enrich_history_row_from_disk(data_dir: str, row: dict[str, Any]) -> dict[str, Any]:
    """Attach package-authoritative metrics (always) and trading_days if missing.

    Ignores any deprecated metric values previously stored in the lifecycle DB.
    """
    out = dict(row)
    model_name = str(out.get("model_name") or "")
    if model_name:
        out["package_path"] = model_package_dir(data_dir, model_name)
        out["package_name"] = safe_model_name(model_name)
    needs_days = out.get("trading_days") is None
    try:
        if model_name:
            prod = _read_package_production_metrics(data_dir, model_name)
            if prod:
                out = _apply_package_metrics_to_history_row(out, prod)
        if needs_days and model_name:
            days = _read_package_trading_days(
                data_dir,
                model_name,
                dataset_name=out.get("dataset"),
            )
            if days is not None:
                out["trading_days"] = days
    except Exception:
        return row
    return out


def _read_package_trading_days(
    data_dir: str,
    model_name: str,
    *,
    dataset_name: str | None = None,
) -> int | None:
    from .registry import _load_json_artifact

    safe = safe_model_name(model_name)
    paths = model_artifact_paths(data_dir, safe)
    meta_art = _load_json_artifact(os.path.join(paths["package_dir"], "metadata.json"))
    meta = meta_art.get("data") if isinstance(meta_art.get("data"), dict) else {}
    cfg_art = _load_json_artifact(paths["config_json"])
    cfg = cfg_art.get("data") if isinstance(cfg_art.get("data"), dict) else {}
    return _extract_trading_days(
        meta,
        cfg,
        data_dir=data_dir,
        dataset_name=dataset_name or cfg.get("dataset") or meta.get("dataset"),
    )


def _dataset_feature_count_map(data_dir: str) -> dict[str, int]:
    out: dict[str, int] = {}
    try:
        from ..dataset_builder.auditor import list_datasets

        for row in list_datasets(data_dir):
            name = str(row.get("dataset_name") or "").strip()
            fc = row.get("feature_count")
            if name and fc is not None:
                out[name] = int(fc)
    except Exception:
        return out
    return out


def _dataset_registry_feature_count(
    data_dir: str,
    dataset_name: str | None,
    *,
    cache: dict[str, int] | None = None,
) -> int | None:
    if not dataset_name:
        return None
    target = str(dataset_name).strip()
    if cache is not None:
        cached = cache.get(target)
        if cached is not None:
            return cached
        return None
    try:
        from ..dataset_builder.auditor import list_datasets

        for row in list_datasets(data_dir):
            if str(row.get("dataset_name") or "").strip() == target:
                fc = row.get("feature_count")
                if fc is not None:
                    return int(fc)
    except Exception:
        return None
    return None


def _load_package_config_dict(paths: dict[str, str]) -> dict[str, Any]:
    pkg = paths["package_dir"]
    for filename in ("training_config.json", "config.json"):
        raw = _read_json_dict(os.path.join(pkg, filename))
        if not raw:
            continue
        if raw.get("dataset") or raw.get("features"):
            return raw
        nested = raw.get("data")
        if isinstance(nested, dict) and (nested.get("dataset") or nested.get("features")):
            return nested
    return {}


def _version_package_context(data_dir: str, model_name: str) -> dict[str, Any]:
    from .registry import _interval_from_dataset_name, _resolve_sampling_interval_sec, _selected_feature_names

    safe = safe_model_name(model_name)
    paths = model_artifact_paths(data_dir, safe)
    cfg = _load_package_config_dict(paths)
    dataset = str(cfg.get("dataset") or "")
    selected = _selected_feature_names(data_dir, safe, paths)
    input_feats = cfg.get("features")
    if not isinstance(input_feats, list):
        input_feats = selected
    input_count = len(input_feats) if input_feats else (len(selected) if selected else None)
    sampling = _resolve_sampling_interval_sec(data_dir, config=cfg, dataset_name=dataset)
    if sampling is None and dataset:
        sampling = _interval_from_dataset_name(dataset)
    return {
        "sampling_interval_sec": sampling,
        "selected_features": selected,
        "selected_feature_count": len(selected) if selected else None,
        "input_feature_count": input_count,
    }


def _enrich_lifecycle_history_rows(data_dir: str, history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach dataset sampling + feature evolution fields for lifecycle UI."""
    if not history:
        return history
    from .registry import _interval_from_dataset_name

    dataset_fc_cache = _dataset_feature_count_map(data_dir)
    enriched: list[dict[str, Any]] = []
    prev_selected: set[str] | None = None
    prev_selected_count: int | None = None
    for row in history:
        out = dict(row)
        ctx: dict[str, Any] = {}
        try:
            ctx = _version_package_context(data_dir, str(out.get("model_name") or ""))
        except Exception:
            ctx = {}

        if ctx.get("sampling_interval_sec") is not None:
            out["sampling_interval_sec"] = ctx["sampling_interval_sec"]
        elif out.get("sampling_interval_sec") is None:
            ds_interval = _interval_from_dataset_name(str(out.get("dataset") or ""))
            if ds_interval is not None:
                out["sampling_interval_sec"] = ds_interval

        selected = ctx.get("selected_features") or []
        if selected:
            selected_count = len(selected)
            out["selected_feature_count"] = selected_count
        elif out.get("feature_count") is not None:
            selected_count = int(out["feature_count"])
            out["selected_feature_count"] = selected_count
        else:
            selected_count = 0
            out["selected_feature_count"] = 0

        registry_total = _dataset_registry_feature_count(
            data_dir,
            out.get("dataset"),
            cache=dataset_fc_cache,
        )
        input_count = ctx.get("input_feature_count")

        if prev_selected_count is None:
            total_fc = registry_total if registry_total is not None else (input_count if input_count is not None else selected_count)
            out["total_feature_count"] = int(total_fc)
            out["features_removed"] = 0
            out["features_added"] = 0
        else:
            pool = int(prev_selected_count)
            out["total_feature_count"] = pool
            current_set = set(selected) if selected else None
            if current_set is not None and prev_selected is not None:
                out["features_removed"] = len(prev_selected - current_set)
                out["features_added"] = len(current_set - prev_selected)
            else:
                out["features_removed"] = max(0, pool - selected_count)
                out["features_added"] = max(0, selected_count - pool)

        if selected:
            prev_selected = set(selected)
        prev_selected_count = selected_count

        enriched.append(out)
    return enriched


def _count_parameter_changes(parent_params: dict[str, Any] | None, child_params: dict[str, Any] | None) -> int:
    if not parent_params or not child_params:
        return 0
    keys = set(parent_params) | set(child_params)
    return sum(1 for k in keys if parent_params.get(k) != child_params.get(k))


def _build_changes(
    *,
    parent: dict[str, Any] | None,
    current: dict[str, Any],
) -> dict[str, Any]:
    if not parent:
        return {"summary": ["Initial model version"]}
    changes: list[str] = []
    detail: dict[str, Any] = {}

    def _delta_num(label: str, key: str, fmt: str = "{:+,}") -> None:
        p = parent.get(key)
        c = current.get(key)
        if p is None or c is None or p == c:
            return
        try:
            diff = int(c) - int(p)
            if diff > 0:
                changes.append(f"{label}: {fmt.format(diff)}")
            elif diff < 0:
                changes.append(f"{label}: {fmt.format(diff)}")
            detail[key] = {"from": p, "to": c, "delta": diff}
        except (TypeError, ValueError):
            if p != c:
                changes.append(f"{label}: {p} → {c}")
                detail[key] = {"from": p, "to": c}

    _delta_num("Rows", "row_count")
    _delta_num("Trading days", "trading_days")
    _delta_num("Features", "feature_count")

    if parent.get("dataset") != current.get("dataset"):
        changes.append(f"Dataset: {parent.get('dataset')} → {current.get('dataset')}")
        detail["dataset"] = {"from": parent.get("dataset"), "to": current.get("dataset")}
    if parent.get("target") != current.get("target"):
        changes.append(f"Target: {parent.get('target')} → {current.get('target')}")
        detail["target"] = {"from": parent.get("target"), "to": current.get("target")}
    if parent.get("validation_strategy") != current.get("validation_strategy"):
        changes.append("Validation strategy changed")
        detail["validation_strategy"] = {
            "from": parent.get("validation_strategy"),
            "to": current.get("validation_strategy"),
        }
    else:
        changes.append("Validation: Same")
    if parent.get("target") == current.get("target"):
        if not any(c.startswith("Target:") for c in changes):
            detail.setdefault("target", {"from": current.get("target"), "to": current.get("target"), "same": True})

    p_params = int(parent.get("parameters_changed") or 0)
    c_params = int(current.get("parameters_changed") or 0)
    if c_params:
        changes.append(f"Parameters: {c_params} changed")
        detail["parameters_changed"] = c_params

    lifecycle = str(current.get("lifecycle") or "")
    if lifecycle == "retrain" and parent.get("dataset") == current.get("dataset"):
        if parent.get("row_count") != current.get("row_count"):
            rd = detail.get("row_count", {})
            if rd.get("delta", 0) > 0:
                changes.insert(0, "Dataset grew (retrain on expanded data)")

    return {"summary": changes or ["No material changes recorded"], "detail": detail}


def get_history_by_model_name(data_dir: str, model_name: str) -> dict[str, Any] | None:
    name = safe_model_name(model_name)
    with _connect(data_dir) as conn:
        row = conn.execute(
            "SELECT * FROM model_history WHERE model_name = ?",
            (name,),
        ).fetchone()
    return _history_row_dict(row) if row else None


def get_model_id_for_name(data_dir: str, model_name: str) -> str | None:
    row = get_history_by_model_name(data_dir, model_name)
    if row:
        return str(row.get("model_id") or "")
    return safe_model_name(model_name)


def list_history_for_model(data_dir: str, *, model_id: str | None = None, model_name: str | None = None) -> list[dict[str, Any]]:
    if not model_id and model_name:
        row = get_history_by_model_name(data_dir, model_name)
        model_id = row.get("model_id") if row else safe_model_name(model_name)
    if not model_id:
        return []
    with _connect(data_dir) as conn:
        rows = conn.execute(
            "SELECT * FROM model_history WHERE model_id = ? ORDER BY version_number ASC, history_id ASC",
            (str(model_id),),
        ).fetchall()
    return [_history_row_dict(r) for r in rows]


def list_model_champions(data_dir: str) -> list[dict[str, Any]]:
    with _connect(data_dir) as conn:
        rows = conn.execute(
            "SELECT * FROM model_registry ORDER BY updated_on DESC",
        ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        doc = dict(row)
        # Deprecated: current_metrics_json is ignored for display.
        doc.pop("current_metrics_json", None)
        doc["current_metrics"] = {}
        doc["metrics_source"] = "package"
        out.append(doc)
    return out


def build_improvement_summary(history: list[dict[str, Any]]) -> dict[str, Any]:
    if not history:
        return {}
    initial = history[0]
    current = history[-1]
    counts = {
        "retrain": 0,
        "complete_optimization": 0,
        "feature_optimization": 0,
        "calibration": 0,
        "rollback": 0,
        "new_model": 0,
    }
    hpo_runs = 0
    for row in history:
        lc = str(row.get("lifecycle") or "")
        if lc in counts and lc != "new_model":
            counts[lc] += 1
        elif lc == "new_model" and row is not initial:
            pass
        if lc == "complete_optimization":
            hpo_runs += 1
        if int(row.get("hpo_trials") or 0) > 0 and lc != "complete_optimization":
            hpo_runs += 1

    def _pct_change(old: float | None, new: float | None, *, lower_better: bool = False) -> float | None:
        if old is None or new is None:
            return None
        try:
            o, n = float(old), float(new)
        except (TypeError, ValueError):
            return None
        if o == 0:
            return None
        pct = ((n - o) / abs(o)) * 100.0
        if lower_better:
            pct = -pct
        return round(pct, 1)

    def _fmt_delta(old: Any, new: Any, pct: float | None, *, suffix: str = "") -> str:
        if old is None or new is None:
            return "—"
        arrow = "▲" if pct is not None and pct > 0 else ("▼" if pct is not None and pct < 0 else "→")
        pct_txt = f" {arrow} {abs(pct):.1f}%" if pct is not None else ""
        return f"{old}{suffix} → {new}{suffix}{pct_txt}"

    row_pct = _pct_change(initial.get("row_count"), current.get("row_count"))
    if initial.get("row_count") and current.get("row_count"):
        try:
            mult = float(current["row_count"]) / float(initial["row_count"])
            row_mult = round(mult, 1)
        except (TypeError, ValueError, ZeroDivisionError):
            row_mult = None
    else:
        row_mult = None

    return {
        "model_id": current.get("model_id"),
        "current_version": current.get("version_label"),
        "current_model_name": current.get("model_name"),
        "initial_version": initial.get("version_label"),
        "version_count": len(history),
        "current_metrics": {
            "mae": current.get("mae"),
            "rmse": current.get("rmse"),
            "directional_accuracy_pct": current.get("directional_accuracy_pct"),
            "composite_score": current.get("composite_score"),
            "premium_mae_pct": current.get("premium_mae_pct"),
            "premium_rmse_pct": current.get("premium_rmse_pct"),
            "medae": current.get("medae"),
            "p95_error": current.get("p95_error"),
            "prediction_bias": current.get("prediction_bias"),
            "prediction_bias_pct": current.get("prediction_bias_pct"),
            "premium_band_performance": current.get("premium_band_performance")
            or ((current.get("metrics") or {}).get("production") or {}).get("premium_band_performance")
            or [],
        },
        "improvement_since_initial": {
            "mae": _fmt_delta(initial.get("mae"), current.get("mae"),
                               _pct_change(initial.get("mae"), current.get("mae"), lower_better=True)),
            "rmse": _fmt_delta(initial.get("rmse"), current.get("rmse"),
                               _pct_change(initial.get("rmse"), current.get("rmse"), lower_better=True)),
            "directional_accuracy_pct": _fmt_delta(
                initial.get("directional_accuracy_pct"), current.get("directional_accuracy_pct"),
                _pct_change(initial.get("directional_accuracy_pct"), current.get("directional_accuracy_pct")),
                suffix="%",
            ),
            "composite_score": _fmt_delta(
                initial.get("composite_score"), current.get("composite_score"),
                _pct_change(initial.get("composite_score"), current.get("composite_score")),
            ),
            "premium_mae_pct": _fmt_delta(
                initial.get("premium_mae_pct"), current.get("premium_mae_pct"),
                _pct_change(initial.get("premium_mae_pct"), current.get("premium_mae_pct"), lower_better=True),
                suffix="%",
            ),
            "premium_rmse_pct": _fmt_delta(
                initial.get("premium_rmse_pct"), current.get("premium_rmse_pct"),
                _pct_change(initial.get("premium_rmse_pct"), current.get("premium_rmse_pct"), lower_better=True),
                suffix="%",
            ),
            "medae": _fmt_delta(
                initial.get("medae"), current.get("medae"),
                _pct_change(initial.get("medae"), current.get("medae"), lower_better=True),
            ),
            "p95_error": _fmt_delta(
                initial.get("p95_error"), current.get("p95_error"),
                _pct_change(initial.get("p95_error"), current.get("p95_error"), lower_better=True),
            ),
            "prediction_bias": _fmt_delta(
                initial.get("prediction_bias"), current.get("prediction_bias"),
                _pct_change(initial.get("prediction_bias"), current.get("prediction_bias"), lower_better=True),
            ),
            "dataset_rows": _fmt_delta(initial.get("row_count"), current.get("row_count"), row_pct),
            "dataset_rows_multiplier": row_mult,
            "trading_days": _fmt_delta(initial.get("trading_days"), current.get("trading_days"),
                                       _pct_change(initial.get("trading_days"), current.get("trading_days"))),
            "features": _fmt_delta(initial.get("feature_count"), current.get("feature_count"),
                                   _pct_change(initial.get("feature_count"), current.get("feature_count"))),
        },
        "lifecycle_counts": {
            "retrains": counts["retrain"],
            "complete_optimizations": counts["complete_optimization"],
            "feature_optimizations": counts["feature_optimization"],
            "calibrations": counts["calibration"],
            "rollbacks": counts["rollback"],
            "hyperparameter_runs": hpo_runs,
        },
        "timeline": [
            {
                "trained_at": row.get("trained_at"),
                "lifecycle": row.get("lifecycle"),
                "lifecycle_label": row.get("lifecycle_label"),
                "version_label": row.get("version_label"),
                "model_name": row.get("model_name"),
                "row_count": row.get("row_count"),
                "trading_days": row.get("trading_days"),
                "feature_count": row.get("feature_count"),
                "mae": row.get("mae"),
                "rmse": row.get("rmse"),
                "directional_accuracy_pct": row.get("directional_accuracy_pct"),
                "composite_score": row.get("composite_score"),
                "premium_mae_pct": row.get("premium_mae_pct"),
                "premium_rmse_pct": row.get("premium_rmse_pct"),
                "medae": row.get("medae"),
                "p95_error": row.get("p95_error"),
                "prediction_bias": row.get("prediction_bias"),
                "prediction_bias_pct": row.get("prediction_bias_pct"),
            }
            for row in history
        ],
        "comparison_table": [
            {
                "metric": "Dataset Rows",
                "initial": initial.get("row_count"),
                "current": current.get("row_count"),
                "improvement_pct": row_pct,
            },
            {
                "metric": "Trading Days",
                "initial": initial.get("trading_days"),
                "current": current.get("trading_days"),
                "improvement_pct": _pct_change(initial.get("trading_days"), current.get("trading_days")),
            },
            {
                "metric": "Features",
                "initial": initial.get("feature_count"),
                "current": current.get("feature_count"),
                "improvement_pct": _pct_change(initial.get("feature_count"), current.get("feature_count")),
            },
            {
                "metric": "MAE",
                "initial": initial.get("mae"),
                "current": current.get("mae"),
                "improvement_pct": _pct_change(initial.get("mae"), current.get("mae"), lower_better=True),
            },
            {
                "metric": "RMSE",
                "initial": initial.get("rmse"),
                "current": current.get("rmse"),
                "improvement_pct": _pct_change(initial.get("rmse"), current.get("rmse"), lower_better=True),
            },
            {
                "metric": "Premium MAE (%)",
                "initial": initial.get("premium_mae_pct"),
                "current": current.get("premium_mae_pct"),
                "improvement_pct": _pct_change(
                    initial.get("premium_mae_pct"), current.get("premium_mae_pct"), lower_better=True,
                ),
            },
            {
                "metric": "Premium RMSE (%)",
                "initial": initial.get("premium_rmse_pct"),
                "current": current.get("premium_rmse_pct"),
                "improvement_pct": _pct_change(
                    initial.get("premium_rmse_pct"), current.get("premium_rmse_pct"), lower_better=True,
                ),
            },
            {
                "metric": "MedAE",
                "initial": initial.get("medae"),
                "current": current.get("medae"),
                "improvement_pct": _pct_change(initial.get("medae"), current.get("medae"), lower_better=True),
            },
            {
                "metric": "P95 Error",
                "initial": initial.get("p95_error"),
                "current": current.get("p95_error"),
                "improvement_pct": _pct_change(
                    initial.get("p95_error"), current.get("p95_error"), lower_better=True,
                ),
            },
            {
                "metric": "Prediction Bias",
                "initial": initial.get("prediction_bias"),
                "current": current.get("prediction_bias"),
                "improvement_pct": _pct_change(
                    initial.get("prediction_bias"), current.get("prediction_bias"), lower_better=True,
                ),
            },
            {
                "metric": "Direction",
                "initial": initial.get("directional_accuracy_pct"),
                "current": current.get("directional_accuracy_pct"),
                "improvement_pct": _pct_change(
                    initial.get("directional_accuracy_pct"), current.get("directional_accuracy_pct"),
                ),
            },
            {
                "metric": "Composite",
                "initial": initial.get("composite_score"),
                "current": current.get("composite_score"),
                "improvement_pct": _pct_change(initial.get("composite_score"), current.get("composite_score")),
            },
        ],
    }


def record_training_history(
    *,
    data_dir: str,
    model_name: str,
    trained_at: str,
    config: Any,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
    matrix_report: dict[str, Any],
    lineage: dict[str, Any] | None = None,
    validation_strategy: str | None = None,
    hpo_trials: int | None = None,
    parameters_changed: int | None = None,
) -> dict[str, Any]:
    """Insert a history row and upsert the champion registry entry."""
    from .config import TrainingConfig

    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be TrainingConfig")

    # ``metrics`` retained for API compatibility; not persisted (package is authoritative).
    _ = metrics

    name = safe_model_name(model_name)
    # Evaluation metrics are NOT persisted — package metrics.json is authoritative.
    from .artifacts import resolve_training_row_count

    row_count = resolve_training_row_count(metadata=metadata, matrix_report=matrix_report)
    trading_days = _extract_trading_days(
        metadata,
        config.to_dict(),
        data_dir=data_dir,
        dataset_name=config.dataset,
    )
    feature_count = len(config.features or [])
    parent_name = str((lineage or {}).get("parent_model_id") or "").strip() or None
    lifecycle = str((lineage or {}).get("lifecycle_mode") or "new_model").strip().lower()
    if not parent_name:
        lifecycle = "new_model"
    model_id = str((lineage or {}).get("ancestor_model_id") or name).strip()

    parent_hist = get_history_by_model_name(data_dir, parent_name) if parent_name else None
    version_number = int(parent_hist["version_number"]) + 1 if parent_hist else 1
    version_label = f"v{version_number}"
    parent_history_id = parent_hist["history_id"] if parent_hist else None

    parent_params = None
    child_params = dict(config.parameters or {})
    if parent_name and parameters_changed is None:
        try:
            from .registry import load_model_detail
            parent_detail = load_model_detail(data_dir, parent_name)
            parent_cfg = parent_detail.get("config") or {}
            parent_params = dict(parent_cfg.get("parameters") or {})
            parameters_changed = _count_parameter_changes(parent_params, child_params)
        except Exception:
            parameters_changed = 0

    snapshot = {
        "row_count": row_count,
        "trading_days": trading_days,
        "feature_count": feature_count,
        "dataset": config.dataset,
        "target": config.target,
        "validation_strategy": validation_strategy,
        "parameters_changed": parameters_changed or 0,
        "lifecycle": lifecycle,
    }
    changes = _build_changes(parent=parent_hist, current=snapshot)

    # Deprecated columns: always NULL / empty JSON (schema retained for compat).
    empty_metrics_json = "{}"

    # Phase 4C.2: Model Taxonomy Resolution
    from ..model_taxonomy import (
        BASELINE_REGIME_CATALOG,
        DEFAULT_REGIME_ID,
        DEFAULT_REGIME_NAME,
        ModelContextKey,
        ModelLifecycleStatus,
        ModelMetadata,
        ModelPopulationTier,
        RegimeScope,
        RegimeSpec,
        TaskSpec,
        TaskType,
        infer_task_type_from_target,
    )

    strat_id = str(getattr(config, "label_strategy", "") or getattr(config, "strategy_id", "") or "")
    pred_type = str(getattr(config, "prediction_type", "") or "")
    tt = infer_task_type_from_target(config.target, strategy_id=strat_id, prediction_type=pred_type)
    task_type_str = tt.value

    # Regime resolution priority:
    # 1. Explicit training configuration regime_id
    # 2. Dataset regime metadata
    # 3. R000 / ALL_REGIMES fallback
    lc_dict = config.lifecycle if isinstance(getattr(config, "lifecycle", None), dict) else {}
    reg_id = str(lc_dict.get("regime_id") or "").strip()
    if not reg_id:
        ds_meta = getattr(config, "dataset_metadata", None)
        if isinstance(ds_meta, dict):
            reg_id = str(ds_meta.get("regime_id") or "").strip()
    if not reg_id:
        reg_id = DEFAULT_REGIME_ID

    reg_name = str(lc_dict.get("regime_name") or "").strip()
    if not reg_name and reg_id in BASELINE_REGIME_CATALOG:
        reg_name = BASELINE_REGIME_CATALOG[reg_id]["name"]
    elif not reg_name:
        reg_name = DEFAULT_REGIME_NAME

    pop_str = str(lc_dict.get("population") or (lineage or {}).get("population") or ModelPopulationTier.EXPERIMENTAL.value).strip().upper()
    try:
        pop_tier = ModelPopulationTier.from_str(pop_str)
    except Exception:
        pop_tier = ModelPopulationTier.EXPERIMENTAL

    status_str = str(lc_dict.get("status") or "ACTIVE").strip().upper()
    try:
        lifecycle_status = ModelLifecycleStatus.from_str(status_str)
    except Exception:
        lifecycle_status = ModelLifecycleStatus.ACTIVE

    market = str(getattr(config, "market", "") or lc_dict.get("market") or "NIFTY").upper().strip()
    interval_sec = int(lc_dict.get("sampling_interval_sec") or 3)
    horizon = str(lc_dict.get("prediction_horizon") or "5m").strip()
    ctx_key = ModelContextKey(
        market=market,
        sampling_interval_sec=interval_sec,
        task_type=tt,
        prediction_horizon=horizon,
        regime_id=reg_id,
    )
    context_key_str = ctx_key.canonical_key_str()
    package_model_id = str((lineage or {}).get("package_model_id") or name).strip()

    meta_obj = ModelMetadata(
        model_id=package_model_id,
        model_name=name,
        version=version_number,
        model_family_id=model_id,
        task=TaskSpec(task_type=tt, target=config.target, prediction_horizon=horizon),
        regime=RegimeSpec(regime_id=reg_id, regime_name=reg_name),
        market_context={"market": market, "sampling_interval_sec": interval_sec},
        population=pop_tier,
        status=lifecycle_status,
        algorithm=config.algorithm,
        feature_count=feature_count,
        lineage=dict(lineage or {}),
        metrics_summary={},
        registered_at=trained_at,
    )
    metadata_json_str = meta_obj.to_json(indent=None)

    with _connect(data_dir) as conn:
        conn.execute(
            """
            INSERT INTO model_history (
                parent_history_id, model_id, model_name, version_label, version_number,
                lifecycle, parent_model_name, trained_at, dataset, target, algorithm,
                validation_strategy, row_count, trading_days, feature_count,
                mae, rmse, directional_accuracy_pct, composite_score,
                premium_mae_pct, premium_rmse_pct,
                medae, p95_error, prediction_bias, prediction_bias_pct,
                hpo_trials, parameters_changed, changes_json, metrics_json,
                task_type, regime_id, regime_name, population, status,
                context_key, package_model_id, metadata_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                parent_history_id,
                model_id,
                name,
                version_label,
                version_number,
                lifecycle,
                parent_name,
                trained_at,
                config.dataset,
                config.target,
                config.algorithm,
                validation_strategy,
                row_count,
                trading_days,
                feature_count,
                None,  # mae (deprecated)
                None,  # rmse (deprecated)
                None,  # directional_accuracy_pct (deprecated)
                None,  # composite_score (deprecated)
                None,  # premium_mae_pct (deprecated)
                None,  # premium_rmse_pct (deprecated)
                None,  # medae (deprecated)
                None,  # p95_error (deprecated)
                None,  # prediction_bias (deprecated)
                None,  # prediction_bias_pct (deprecated)
                hpo_trials,
                parameters_changed,
                json.dumps(changes),
                empty_metrics_json,
                task_type_str,
                reg_id,
                reg_name,
                pop_tier.value,
                lifecycle_status.value,
                context_key_str,
                package_model_id,
                metadata_json_str,
            ),
        )
        history_id = int(conn.execute("SELECT last_insert_rowid()").fetchone()[0])
        now = _utc_now()
        existing = conn.execute(
            "SELECT model_id FROM model_registry WHERE model_id = ?",
            (model_id,),
        ).fetchone()
        regime_scope_val = RegimeScope.ALL_REGIMES.value if reg_id == DEFAULT_REGIME_ID else RegimeScope.SPECIALIZED.value
        if existing:
            family_display = str((lineage or {}).get("ancestor_model_id") or model_id).strip()
            conn.execute(
                """
                UPDATE model_registry SET
                    current_model_name = ?,
                    current_version = ?,
                    current_version_number = ?,
                    updated_on = ?,
                    current_metrics_json = ?,
                    display_name = ?,
                    task_type = ?,
                    regime_id = ?,
                    context_key = ?,
                    champion_model_name = ?,
                    challenger_model_name = ?,
                    regime_scope = ?
                WHERE model_id = ?
                """,
                (
                    name,
                    version_label,
                    version_number,
                    now,
                    empty_metrics_json,
                    family_display,
                    task_type_str,
                    reg_id,
                    context_key_str,
                    name,
                    parent_name,
                    regime_scope_val,
                    model_id,
                ),
            )
        else:
            family_display = str((lineage or {}).get("ancestor_model_id") or model_id).strip()
            conn.execute(
                """
                INSERT INTO model_registry (
                    model_id, display_name, current_model_name, current_version,
                    current_version_number, status, created_on, updated_on, current_metrics_json,
                    task_type, regime_id, context_key, champion_model_name, challenger_model_name, regime_scope
                ) VALUES (?, ?, ?, ?, ?, 'ready', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    family_display,
                    name,
                    version_label,
                    version_number,
                    trained_at,
                    now,
                    empty_metrics_json,
                    task_type_str,
                    reg_id,
                    context_key_str,
                    name,
                    parent_name,
                    regime_scope_val,
                ),
            )
        conn.commit()

    history = list_history_for_model(data_dir, model_id=model_id)
    history = [_enrich_history_row_from_disk(data_dir, h) for h in history]
    return {
        "history_id": history_id,
        "model_id": model_id,
        "version_label": version_label,
        "version_number": version_number,
        "lifecycle": lifecycle,
        "changes": changes,
        "improvement": build_improvement_summary(history),
    }


_rebuild_lifecycle_active = False


def _list_model_packages_for_index(data_dir: str) -> list[dict[str, Any]]:
    """Scan model packages without registry lifecycle enrichment (avoids rebuild recursion)."""
    root = models_dir(data_dir)
    if not os.path.isdir(root):
        return []
    rows: list[dict[str, Any]] = []
    for entry in sorted(os.listdir(root)):
        if entry.startswith("."):
            continue
        pkg = os.path.join(root, entry)
        if not os.path.isdir(pkg):
            continue
        trained_at = ""
        meta_path = os.path.join(pkg, "metadata.json")
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, encoding="utf-8") as fh:
                    meta = json.load(fh)
                if isinstance(meta, dict):
                    trained_at = str(meta.get("trained_at") or "")
            except Exception:
                trained_at = ""
        rows.append({"model_name": entry, "trained_at": trained_at})
    return rows


def rebuild_lifecycle_index(data_dir: str) -> dict[str, Any]:
    """Backfill history/registry from on-disk model packages (idempotent per model_name)."""
    global _rebuild_lifecycle_active
    if _rebuild_lifecycle_active:
        return {"indexed": 0, "skipped": 0, "reentrant": True}

    from .registry import load_model_detail

    root = models_dir(data_dir)
    if not os.path.isdir(root):
        return {"indexed": 0, "skipped": 0}

    _rebuild_lifecycle_active = True
    indexed = 0
    skipped = 0
    try:
        models = _list_model_packages_for_index(data_dir)

        # Process in trained_at order so versions assign correctly
        def _sort_key(m: dict[str, Any]) -> str:
            return str(m.get("trained_at") or "")

        for row in sorted(models, key=_sort_key):
            name = str(row.get("model_name") or "")
            if not name or name.startswith("."):
                continue
            if get_history_by_model_name(data_dir, name):
                skipped += 1
                continue
            try:
                detail = load_model_detail(data_dir, name)
            except Exception:
                skipped += 1
                continue
            meta_art = detail.get("metadata") or {}
            meta = meta_art.get("data") if isinstance(meta_art.get("data"), dict) else {}
            lineage = meta.get("lineage") if isinstance(meta.get("lineage"), dict) else {}
            cfg = detail.get("config") or {}
            from .config import normalize_training_config

            try:
                tc = normalize_training_config(cfg)
            except Exception:
                skipped += 1
                continue
            metrics = detail.get("metrics") or {}
            matrix_report = meta.get("matrix_report") or {}
            if not matrix_report and meta.get("row_count"):
                matrix_report = {"x_shape": [meta.get("row_count"), len(tc.features or [])]}
            hpo_trials = 0
            wf = detail.get("walk_forward") or {}
            bp = wf.get("best_parameters") or {}
            bp_data = bp.get("data") if isinstance(bp.get("data"), dict) else {}
            hpo_trials = int(bp_data.get("n_trials_completed") or bp_data.get("n_trials_target") or 0)
            strat = detail.get("validation_strategy") or {}
            val_label = strat.get("label") if isinstance(strat, dict) else str(row.get("validation_strategy") or "")
            trained_at = str(meta.get("trained_at") or row.get("trained_at") or _utc_now())
            try:
                record_training_history(
                    data_dir=data_dir,
                    model_name=name,
                    trained_at=trained_at,
                    config=tc,
                    metrics=metrics,
                    metadata=meta,
                    matrix_report=matrix_report,
                    lineage=lineage or None,
                    validation_strategy=val_label,
                    hpo_trials=hpo_trials or None,
                )
            except Exception:
                skipped += 1
                continue
            indexed += 1

        return {"indexed": indexed, "skipped": skipped, "families": len(list_model_champions(data_dir))}
    finally:
        _rebuild_lifecycle_active = False


def get_model_lifecycle_view(data_dir: str, *, model_name: str | None = None, model_id: str | None = None) -> dict[str, Any]:
    """Full lifecycle payload for model detail UI."""
    t_all = time.perf_counter()
    stages: list[dict[str, Any]] = []

    def _mark(label: str, started: float) -> None:
        stages.append({"label": label, "ms": round((time.perf_counter() - started) * 1000, 1)})

    t0 = time.perf_counter()
    rebuild_lifecycle_index(data_dir)
    _mark("rebuild_lifecycle_index", t0)

    t0 = time.perf_counter()
    if model_name and not model_id:
        hist_row = get_history_by_model_name(data_dir, model_name)
        model_id = hist_row.get("model_id") if hist_row else safe_model_name(model_name)
    history = list_history_for_model(data_dir, model_id=model_id)
    if not history and model_name:
        history = list_history_for_model(data_dir, model_id=safe_model_name(model_name))
    _mark("list_history", t0)

    t0 = time.perf_counter()
    history = [_enrich_history_row_from_disk(data_dir, h) for h in history]
    _mark("enrich_history_metrics", t0)

    t0 = time.perf_counter()
    history = _enrich_lifecycle_history_rows(data_dir, history)
    _mark("enrich_feature_evolution", t0)

    t0 = time.perf_counter()
    champion = None
    with _connect(data_dir) as conn:
        if model_id:
            champion = _row_to_dict(conn.execute(
                "SELECT * FROM model_registry WHERE model_id = ?",
                (str(model_id),),
            ).fetchone())
    if champion:
        # Ignore deprecated current_metrics_json; use package-resolved last history row.
        champion.pop("current_metrics_json", None)
        if history:
            last = history[-1]
            champion["current_metrics"] = {
                key: last.get(key) for key in DEPRECATED_LIFECYCLE_METRIC_KEYS
            }
            champion["metrics_source"] = last.get("metrics_source") or "package"
            champion["package_path"] = last.get("package_path")
        else:
            champion["current_metrics"] = {}
            champion["metrics_source"] = "package"
    _mark("load_champion", t0)

    t0 = time.perf_counter()
    improvement = build_improvement_summary(history)
    _mark("improvement_summary", t0)

    return {
        "model_id": model_id,
        "history": history,
        "improvement": improvement,
        "champion": champion,
        "_timing": {
            "total_ms": round((time.perf_counter() - t_all) * 1000, 1),
            "stages": stages,
        },
    }


def delete_history_for_model(data_dir: str, model_name: str) -> None:
    """Remove history rows for a deleted package; re-point champion if needed."""
    name = safe_model_name(model_name)
    with _connect(data_dir) as conn:
        row = conn.execute("SELECT * FROM model_history WHERE model_name = ?", (name,)).fetchone()
        if not row:
            return
        model_id = row["model_id"]
        conn.execute("DELETE FROM model_history WHERE model_name = ?", (name,))
        remaining = conn.execute(
            "SELECT * FROM model_history WHERE model_id = ? ORDER BY version_number DESC LIMIT 1",
            (model_id,),
        ).fetchone()
        if remaining:
            conn.execute(
                """
                UPDATE model_registry SET
                    current_model_name = ?,
                    current_version = ?,
                    current_version_number = ?,
                    updated_on = ?,
                    current_metrics_json = ?
                WHERE model_id = ?
                """,
                (
                    remaining["model_name"],
                    remaining["version_label"],
                    remaining["version_number"],
                    _utc_now(),
                    "{}",  # deprecated — metrics resolved from package
                    model_id,
                ),
            )
        else:
            conn.execute("DELETE FROM model_registry WHERE model_id = ?", (model_id,))
        conn.commit()


def get_champion_for_context(data_dir: str, context_key: Any) -> dict[str, Any] | None:
    """Look up the active champion model package for a specific canonical context key."""
    key_str = context_key.canonical_key_str() if hasattr(context_key, "canonical_key_str") else str(context_key).strip()
    with _connect(data_dir) as conn:
        row = conn.execute(
            "SELECT * FROM model_registry WHERE context_key = ? ORDER BY updated_on DESC LIMIT 1",
            (key_str,),
        ).fetchone()
        if not row:
            row = conn.execute(
                "SELECT * FROM model_registry WHERE model_id = ? ORDER BY updated_on DESC LIMIT 1",
                (key_str,),
            ).fetchone()
    if not row:
        return None
    doc = dict(row)
    doc.pop("current_metrics_json", None)
    return doc


def set_champion_for_context(
    data_dir: str,
    context_key: Any,
    champion_model_name: str,
    challenger_model_name: str | None = None,
) -> None:
    """Set or update the champion and optional challenger for a canonical context key."""
    key_str = context_key.canonical_key_str() if hasattr(context_key, "canonical_key_str") else str(context_key).strip()
    champ_safe = safe_model_name(champion_model_name)
    chall_safe = safe_model_name(challenger_model_name) if challenger_model_name else None
    now = _utc_now()

    with _connect(data_dir) as conn:
        existing = conn.execute(
            "SELECT model_id FROM model_registry WHERE context_key = ? OR model_id = ?",
            (key_str, key_str),
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE model_registry SET
                    current_model_name = ?,
                    champion_model_name = ?,
                    challenger_model_name = ?,
                    updated_on = ?
                WHERE model_id = ?
                """,
                (champ_safe, champ_safe, chall_safe, now, existing[0]),
            )
        else:
            conn.execute(
                """
                INSERT INTO model_registry (
                    model_id, display_name, current_model_name, current_version,
                    current_version_number, status, created_on, updated_on, current_metrics_json,
                    context_key, champion_model_name, challenger_model_name
                ) VALUES (?, ?, ?, 'v1', 1, 'ready', ?, ?, '{}', ?, ?, ?)
                """,
                (key_str, key_str, champ_safe, now, now, key_str, champ_safe, chall_safe),
            )
        conn.commit()


def list_context_champions(data_dir: str) -> list[dict[str, Any]]:
    """List all registered champions keyed by context and family."""
    with _connect(data_dir) as conn:
        rows = conn.execute(
            "SELECT * FROM model_registry ORDER BY updated_on DESC",
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d.pop("current_metrics_json", None)
        out.append(d)
    return out
