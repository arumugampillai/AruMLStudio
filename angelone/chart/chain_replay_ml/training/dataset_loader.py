"""Load audited parquet + schema; select features and target → X, y."""

from __future__ import annotations

import logging
import os
from typing import Any

import pandas as pd

from chain_replay_ml.dataset_builder.expected_spec import expected_spec_path
from chain_replay_ml.dataset_builder.schema_registry import load_feature_registry
from chain_replay_ml.dataset_builder.writer import _safe_filename, datasets_dir

from .config import TrainingConfig
from .load_backend import measure_span, resolve_training_load_backend
from .memory_utils import cap_training_rows, compact_numeric_frame, release_memory

logger = logging.getLogger(__name__)

_IDENTITY_COLUMNS = (
    "trading_day", "timestamp", "token", "strike", "option_type",
    "spot", "ltp", "symbol", "market", "expiry", "master_row_id", "sample_id",
)


class DatasetLoaderError(Exception):
    pass


def dataset_parquet_exists(data_dir: str, dataset_name: str) -> bool:
    """True when ``data/datasets/<dataset>.parquet`` is on disk."""
    name = str(dataset_name or "").strip()
    if not name:
        return False
    safe = _safe_filename(name)
    return os.path.isfile(os.path.join(datasets_dir(data_dir), f"{safe}.parquet"))


def model_dataset_parquet_exists(data_dir: str, model_name: str) -> bool:
    """True when the model's configured training dataset parquet exists."""
    from .paths import model_package_dir, safe_model_name

    pkg = model_package_dir(data_dir, safe_model_name(model_name))
    for fname in ("config.json", "training_config.json", "metadata.json"):
        path = os.path.join(pkg, fname)
        if not os.path.isfile(path) or os.path.getsize(path) <= 0:
            continue
        try:
            import json

            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        ds = str(doc.get("dataset") or doc.get("dataset_name") or "").strip()
        if not ds and isinstance(doc.get("dataset_metadata"), dict):
            ds = str(doc["dataset_metadata"].get("dataset_name") or "").strip()
        if ds:
            return dataset_parquet_exists(data_dir, ds)
    return False


def _parquet_columns(parquet_path: str, wanted: list[str]) -> list[str]:
    """Resolve which columns exist in parquet without loading row data."""
    try:
        import pyarrow.parquet as pq

        schema = pq.read_schema(parquet_path)
        available = set(schema.names)
        return [c for c in wanted if c in available]
    except Exception:
        return wanted


def parquet_column_names(parquet_path: str) -> set[str] | None:
    """Return parquet column names from file metadata only (no row IO)."""
    try:
        import pyarrow.parquet as pq

        return set(pq.read_schema(parquet_path).names)
    except Exception:
        return None


def missing_parquet_columns(parquet_path: str, columns: list[str]) -> list[str]:
    """Columns absent from parquet schema; empty list when all exist or schema unreadable."""
    available = parquet_column_names(parquet_path)
    if available is None:
        return []
    return [c for c in columns if c not in available]


def _parquet_num_rows(parquet_path: str) -> int | None:
    try:
        import pyarrow.parquet as pq

        meta = pq.ParquetFile(parquet_path).metadata
        return int(meta.num_rows) if meta is not None else None
    except Exception:
        return None


def _load_sidecar_meta(
    data_dir: str,
    dataset_name: str,
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    safe = _safe_filename(dataset_name)
    out_dir = datasets_dir(data_dir)
    parquet_path = f"{out_dir}/{safe}.parquet"
    meta_path = f"{out_dir}/{safe}.json"
    expected_path = expected_spec_path(data_dir, safe)

    if not os.path.isfile(parquet_path):
        raise DatasetLoaderError(
            f"Dataset parquet missing for '{safe}': {parquet_path}. "
            "This model references a dataset that is no longer on disk under data/datasets/. "
            "Choose a model whose training dataset still exists, or restore/rebuild that parquet."
        )

    metadata: dict[str, Any] = {}
    expected: dict[str, Any] = {}
    if os.path.isfile(meta_path):
        import json

        with open(meta_path, encoding="utf-8") as fh:
            metadata = json.load(fh)
    if os.path.isfile(expected_path):
        import json

        with open(expected_path, encoding="utf-8") as fh:
            expected = json.load(fh)
    return parquet_path, metadata, expected


def load_dataset_frame(
    data_dir: str,
    dataset_name: str,
    *,
    columns: list[str] | None = None,
    max_rows_hint: int | None = None,
    premium_filter: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Pandas load path (reference / fallback).

    When safe, reads only trailing Parquet row groups that cover the training
    row cap (Opt 1) instead of the full file.
    """
    from .memory_utils import MAX_TRAINING_ROWS
    from .row_group_prune import (
        plan_tail_row_groups,
        read_parquet_row_groups,
        target_rows_for_prune,
    )

    parquet_path, metadata, expected = _load_sidecar_meta(data_dir, dataset_name)

    read_cols = None
    if columns:
        read_cols = _parquet_columns(parquet_path, columns)
        if not read_cols:
            raise DatasetLoaderError("No requested columns found in parquet")

    prune_info: dict[str, Any] | None = None
    target = target_rows_for_prune(
        premium_filter=bool(premium_filter),
        max_rows=int(max_rows_hint) if max_rows_hint is not None else MAX_TRAINING_ROWS,
    )
    plan = plan_tail_row_groups(parquet_path, max_rows=target, metadata=metadata)
    if plan is not None and plan.pruned:
        table = read_parquet_row_groups(parquet_path, plan.indices, columns=read_cols)
        df = table.to_pandas()
        prune_info = plan.as_dict()
    else:
        df = pd.read_parquet(parquet_path, columns=read_cols)
        if plan is not None:
            prune_info = plan.as_dict()

    if prune_info is not None:
        metadata = dict(metadata)
        metadata["row_group_prune"] = prune_info
    return df, metadata, expected


def _engine_filters_from_config(config: TrainingConfig) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if (
        config.premium_selection_enabled
        and config.premium_min is not None
        and config.premium_max is not None
    ):
        lo = float(config.premium_min)
        hi = float(config.premium_max)
        if lo > hi:
            lo, hi = hi, lo
        filters["premium_min"] = lo
        filters["premium_max"] = hi
    if config.trading_days:
        filters["trading_days"] = list(config.trading_days)
    else:
        if config.start_day:
            filters["start_day"] = str(config.start_day)
        if config.end_day:
            filters["end_day"] = str(config.end_day)
    return filters


def apply_config_day_filter(
    df: pd.DataFrame,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Apply optional trading-day list / start–end bounds (pandas path)."""
    if "trading_day" not in df.columns:
        return df, None
    if config.trading_days:
        wanted = {str(d) for d in config.trading_days}
        out = df[df["trading_day"].astype(str).isin(wanted)]
        return out.reset_index(drop=True), {
            "stage": "training_day_filter",
            "mode": "trading_days",
            "days": sorted(wanted),
            "rows_before": int(len(df)),
            "rows_after": int(len(out)),
        }
    start = str(config.start_day or "").strip()
    end = str(config.end_day or "").strip()
    if not start and not end:
        return df, None
    days = df["trading_day"].astype(str)
    mask = pd.Series(True, index=df.index)
    if start:
        mask &= days >= start
    if end:
        mask &= days <= end
    out = df.loc[mask]
    return out.reset_index(drop=True), {
        "stage": "training_day_filter",
        "mode": "range",
        "start_day": start or None,
        "end_day": end or None,
        "rows_before": int(len(df)),
        "rows_after": int(len(out)),
    }


def load_dataset_frame_via_engine(
    data_dir: str,
    dataset_name: str,
    *,
    columns: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    order_by: list[str] | tuple[str, ...] | None = None,
    max_rows_hint: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Dataset Engine load — column prune + predicate pushdown.

    Returns ``(df, metadata, expected, load_info)``.

    Training uses direct Arrow→pandas (no Polars bridge) and optional
    ``order_by`` so Python mergesort can be skipped. Tail row-group prune
    (Opt 1) is applied when chronology is trusted.
    """
    from chain_replay_ml.dataset_engine import query_dataset

    from .memory_utils import MAX_TRAINING_ROWS
    from .row_group_prune import plan_tail_row_groups, target_rows_for_prune

    parquet_path, metadata, expected = _load_sidecar_meta(data_dir, dataset_name)
    read_cols = None
    if columns:
        read_cols = _parquet_columns(parquet_path, columns)
        if not read_cols:
            raise DatasetLoaderError("No requested columns found in parquet")

    order_cols = list(order_by) if order_by else list(_ORDER_COLS)
    # Only order by columns that will be present after column prune.
    if read_cols is not None:
        order_cols = [c for c in order_cols if c in read_cols]

    filt = dict(filters or {})
    premium_filter = any(k in filt for k in ("premium_min", "premium_max", "ltp_min", "ltp_max"))
    target = target_rows_for_prune(
        premium_filter=premium_filter,
        max_rows=int(max_rows_hint) if max_rows_hint is not None else MAX_TRAINING_ROWS,
    )
    prune_plan = plan_tail_row_groups(parquet_path, max_rows=target, metadata=metadata)
    row_groups = list(prune_plan.indices) if prune_plan is not None and prune_plan.pruned else None

    try:
        result = query_dataset(
            parquet_path,
            columns=read_cols,
            filters=filters or None,
            order_by=order_cols or None,
            row_groups=row_groups,
        )
    except ImportError as exc:
        raise DatasetLoaderError(
            "Dataset Engine backend unavailable (install duckdb). "
            "Set ARUNEO_DATASET_ENGINE=off to use pandas."
        ) from exc
    except Exception as exc:
        raise DatasetLoaderError(f"Dataset Engine load failed: {exc}") from exc

    from chain_replay_ml.frame_backend import arrow_table_to_pandas

    # Opt-2: default direct Arrow→pandas (skip Polars). Restore with env.
    via_polars = _train_frame_bridge_via_polars()
    df, frame_bridge = arrow_table_to_pandas(result.table, via_polars=via_polars)
    stats = result.stats
    engine_ordered = bool(order_cols) and bool((stats.extra or {}).get("order_by"))
    load_info = {
        "backend": "dataset_engine",
        "frame_bridge": frame_bridge,
        "rows_returned": int(stats.rows_returned or len(df)),
        "columns_returned": int(len(df.columns)),
        "partitions_scanned": stats.partitions_scanned,
        "partitions_pruned": stats.partitions_pruned,
        "engine_execution_time_sec": stats.execution_time_sec,
        "engine_columns_read": list(stats.columns_read or ()),
        "engine_extra": dict(stats.extra or {}),
        "row_order_applied": "engine_order_by" if engine_ordered else "none",
        "skip_python_sort": bool(engine_ordered),
        "row_group_prune": prune_plan.as_dict() if prune_plan is not None else None,
    }
    if prune_plan is not None:
        metadata = dict(metadata)
        metadata["row_group_prune"] = prune_plan.as_dict()
    return df, metadata, expected, load_info


_ORDER_COLS = ("trading_day", "timestamp", "token")


def _train_frame_bridge_via_polars() -> bool:
    """Training default: direct Arrow→pandas. Set ARUNEO_TRAIN_FRAME_BRIDGE=polars to restore."""
    import os

    raw = str(os.getenv("ARUNEO_TRAIN_FRAME_BRIDGE", "arrow") or "arrow").strip().lower()
    return raw in ("polars", "arrow_polars", "via_polars")


def _metadata_claims_sorted(metadata: dict[str, Any] | None) -> bool:
    meta = metadata if isinstance(metadata, dict) else {}
    if meta.get("is_sorted") is True:
        return True
    row_order = meta.get("row_order")
    if isinstance(row_order, (list, tuple)) and [str(x) for x in row_order] == list(_ORDER_COLS):
        return True
    if str(row_order or "").replace(" ", "") in (
        "trading_day,timestamp,token",
        "trading_day|timestamp|token",
    ):
        return True
    return False


def _stabilize_row_order(
    df: pd.DataFrame,
    *,
    metadata: dict[str, Any] | None = None,
    already_ordered: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Deterministic row order so Engine vs Pandas training matrices match.

    Skip Python mergesort when:
    - caller already ordered (Engine ``ORDER BY``), or
    - dataset sidecar claims ``is_sorted`` / matching ``row_order``.
    """
    info: dict[str, Any] = {"python_sort": False, "skip_reason": None}
    keys = [c for c in _ORDER_COLS if c in df.columns]
    if not keys:
        return df.reset_index(drop=True), {**info, "skip_reason": "no_order_cols"}
    if already_ordered:
        info["skip_reason"] = "already_ordered"
        return df.reset_index(drop=True), info
    if _metadata_claims_sorted(metadata):
        info["skip_reason"] = "metadata_is_sorted"
        return df.reset_index(drop=True), info
    info["python_sort"] = True
    return df.sort_values(keys, kind="mergesort").reset_index(drop=True), info


def apply_config_premium_filter(
    df: pd.DataFrame,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    """Apply Create Model Premium Selection LTP band when enabled."""
    if not config.premium_selection_enabled:
        return df, None
    if config.premium_min is None or config.premium_max is None:
        return df, None
    from chain_replay_ml.dataset_builder.premium_ltp_filter import apply_premium_ltp_filter_frame

    try:
        result = apply_premium_ltp_filter_frame(
            df,
            premium_min=float(config.premium_min),
            premium_max=float(config.premium_max),
        )
    except ValueError as exc:
        raise DatasetLoaderError(str(exc)) from exc
    report = dict(result.get("report") or {})
    report["stage"] = "training_premium_selection"
    return result["frame"], report


def _premium_report_from_engine(
    *,
    config: TrainingConfig,
    rows_after: int,
    parquet_path: str,
) -> dict[str, Any] | None:
    if not config.premium_selection_enabled:
        return None
    if config.premium_min is None or config.premium_max is None:
        return None
    lo = float(config.premium_min)
    hi = float(config.premium_max)
    if lo > hi:
        lo, hi = hi, lo
    rows_before = _parquet_num_rows(parquet_path)
    return {
        "stage": "training_premium_selection",
        "ltp_column": "ltp",
        "premium_min": lo,
        "premium_max": hi,
        "rows_before": rows_before,
        "rows_after": int(rows_after),
        "rows_dropped": (
            None if rows_before is None else max(0, int(rows_before) - int(rows_after))
        ),
        "applied_via": "dataset_engine",
    }


def compare_training_load_backends(
    data_dir: str,
    config: TrainingConfig,
) -> dict[str, Any]:
    """Run pandas and Dataset Engine loads; compare metrics and training matrices.

    Does not train. Matrices are compared after the same stabilize + ``select_xy``
    path used by training (order-independent load → deterministic sort).
    """
    import numpy as np

    features = list(config.features)
    target = config.target
    wanted_cols = list(dict.fromkeys([*features, target, *_IDENTITY_COLUMNS, "ltp_to_spot_ratio"]))
    parquet_path, _, _ = _load_sidecar_meta(data_dir, config.dataset)
    filters = _engine_filters_from_config(config)
    schema = load_schema_registry()

    def _pandas_xy() -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
        df, meta, _ = load_dataset_frame(
            data_dir,
            config.dataset,
            columns=wanted_cols,
            premium_filter=bool(filters),
        )
        df, _ = apply_config_premium_filter(df, config)
        df, day_report = apply_config_day_filter(df, config)
        df, sort_info = _stabilize_row_order(df, metadata=meta)
        info = {
            "rows_returned": int(len(df)),
            "columns_returned": int(len(df.columns)),
            "partitions_scanned": None,
            "row_stabilize": sort_info,
            "row_group_prune": meta.get("row_group_prune") if isinstance(meta, dict) else None,
            "day_filter": day_report,
        }
        X, y, _ = select_xy(df, config, schema=schema)
        return X, y, info

    def _engine_xy() -> tuple[pd.DataFrame, pd.Series, dict[str, Any]]:
        df, meta, _, load_info = load_dataset_frame_via_engine(
            data_dir,
            config.dataset,
            columns=wanted_cols,
            filters=filters,
        )
        df, sort_info = _stabilize_row_order(
            df,
            metadata=meta,
            already_ordered=bool(load_info.get("skip_python_sort")),
        )
        info = {
            "rows_returned": int(len(df)),
            "columns_returned": int(len(df.columns)),
            "partitions_scanned": load_info.get("partitions_scanned"),
            "partitions_pruned": load_info.get("partitions_pruned"),
            "engine_execution_time_sec": load_info.get("engine_execution_time_sec"),
            "frame_bridge": load_info.get("frame_bridge"),
            "row_stabilize": sort_info,
        }
        X, y, _ = select_xy(df, config, schema=schema)
        return X, y, info

    (Xp, yp, p_info), pandas_m = measure_span(_pandas_xy)
    try:
        (Xe, ye, e_info), engine_m = measure_span(_engine_xy)
        engine_err = None
    except Exception as exc:  # noqa: BLE001
        Xe = ye = None
        e_info, engine_m, engine_err = {}, {}, str(exc)

    matrices_equal = None
    rows_match = None
    if engine_err is None and Xe is not None and ye is not None:
        rows_match = int(p_info["rows_returned"]) == int(e_info["rows_returned"])
        try:
            x_ok = np.allclose(
                Xp.to_numpy(dtype=float, copy=False),
                Xe.to_numpy(dtype=float, copy=False),
                equal_nan=True,
            )
            y_ok = np.allclose(
                yp.to_numpy(dtype=float, copy=False),
                ye.to_numpy(dtype=float, copy=False),
                equal_nan=True,
            )
            matrices_equal = bool(x_ok and y_ok and list(Xp.columns) == list(Xe.columns))
        except Exception as exc:  # noqa: BLE001
            matrices_equal = False
            engine_err = f"matrix compare failed: {exc}"

    return {
        "parquet_path": parquet_path,
        "filters": filters,
        "pandas": {
            **pandas_m,
            **p_info,
        },
        "dataset_engine": (
            None
            if engine_err and Xe is None
            else {
                **engine_m,
                **e_info,
            }
        ),
        "dataset_engine_error": engine_err if Xe is None else None,
        "rows_match": rows_match,
        "matrices_equal": matrices_equal,
    }


def load_schema_registry() -> dict[str, Any]:
    return load_feature_registry()


def select_xy(
    df: pd.DataFrame,
    config: TrainingConfig,
    *,
    schema: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Return feature matrix X, target y, and ordered feature column names."""
    features = list(config.features)
    target = config.target
    if not features:
        raise DatasetLoaderError("No features selected")
    if not target:
        raise DatasetLoaderError("No target selected")

    missing_feats = [c for c in features if c not in df.columns]
    if missing_feats:
        raise DatasetLoaderError(f"Features missing from dataset: {', '.join(missing_feats[:5])}")

    if target not in df.columns:
        raise DatasetLoaderError(f"Target column not found: {target}")

    registry = schema or load_schema_registry()
    _ = registry

    X = df[features].copy()
    y = df[target].copy()
    return X, y, features


def _log_dataset_load_metrics(dataset_name: str, load_metrics: dict[str, Any]) -> None:
    """One INFO line per training load — audit trail for observation period."""
    backend = load_metrics.get("backend")
    logger.info(
        "dataset_load dataset=%s backend=%s load_time_sec=%s peak_rss_mb=%s "
        "rows=%s cols=%s partitions_scanned=%s partitions_pruned=%s fallback=%s",
        dataset_name,
        backend,
        load_metrics.get("load_time_sec"),
        load_metrics.get("peak_rss_mb"),
        load_metrics.get("rows_returned"),
        load_metrics.get("columns_returned"),
        load_metrics.get("partitions_scanned"),
        load_metrics.get("partitions_pruned"),
        bool(load_metrics.get("engine_fallback")),
    )


def load_training_xy(
    data_dir: str,
    config: TrainingConfig,
) -> tuple[pd.DataFrame, pd.Series, list[str], dict[str, Any], dict[str, Any], pd.DataFrame]:
    features = list(config.features)
    target = config.target
    if not features:
        raise DatasetLoaderError("No features selected")
    if not target:
        raise DatasetLoaderError("No target selected")

    wanted_cols = list(dict.fromkeys([*features, target, *_IDENTITY_COLUMNS, "ltp_to_spot_ratio"]))
    label_run_id = str(getattr(config, "label_run_id", None) or "").strip()
    feature_wanted = list(wanted_cols)
    if label_run_id:
        feature_wanted = [c for c in wanted_cols if c != target]
    backend = resolve_training_load_backend()
    load_metrics: dict[str, Any] = {"backend": backend}
    prem_report: dict[str, Any] | None = None
    df: pd.DataFrame
    metadata: dict[str, Any]
    expected: dict[str, Any]

    if backend == "dataset_engine":
        filters = _engine_filters_from_config(config)

        def _load_engine():
            return load_dataset_frame_via_engine(
                data_dir,
                config.dataset,
                columns=feature_wanted,
                filters=filters,
            )

        try:
            (df, metadata, expected, engine_info), span = measure_span(_load_engine)
            load_metrics.update(span)
            load_metrics.update(engine_info)
            df, sort_info = _stabilize_row_order(
                df,
                metadata=metadata,
                already_ordered=bool(engine_info.get("skip_python_sort")),
            )
            load_metrics["row_stabilize"] = sort_info
            parquet_path, _, _ = _load_sidecar_meta(data_dir, config.dataset)
            prem_report = _premium_report_from_engine(
                config=config,
                rows_after=len(df),
                parquet_path=parquet_path,
            )
        except Exception as exc:
            # Phase 1: any Engine failure → Pandas (operational safety).
            logger.warning(
                "dataset_load engine fallback dataset=%s reason=%s",
                config.dataset,
                exc,
            )
            backend = "pandas"
            load_metrics["backend"] = "pandas"
            load_metrics["engine_fallback"] = True
            load_metrics["engine_fallback_reason"] = f"{exc.__class__.__name__}: {exc}"

    if backend == "pandas":
        def _load_pandas():
            frame, meta, exp = load_dataset_frame(
                data_dir,
                config.dataset,
                columns=feature_wanted,
                premium_filter=bool(
                    config.premium_selection_enabled
                    and config.premium_min is not None
                    and config.premium_max is not None
                ),
            )
            frame, report = apply_config_premium_filter(frame, config)
            frame, day_report = apply_config_day_filter(frame, config)
            frame, sort_info = _stabilize_row_order(frame, metadata=meta)
            return frame, meta, exp, report, sort_info, day_report

        (df, metadata, expected, prem_report, sort_info, day_report), span = measure_span(
            _load_pandas
        )
        load_metrics.update(span)
        load_metrics["rows_returned"] = int(len(df))
        load_metrics["columns_returned"] = int(len(df.columns))
        load_metrics["row_stabilize"] = sort_info
        load_metrics["row_group_prune"] = (
            metadata.get("row_group_prune") if isinstance(metadata, dict) else None
        )
        if day_report:
            load_metrics["day_filter"] = day_report
        load_metrics.setdefault("partitions_scanned", None)

    if prem_report:
        metadata = dict(metadata)
        metadata["premium_selection"] = prem_report
        if int(prem_report.get("rows_after") or 0) < 1:
            lo = prem_report.get("premium_min")
            hi = prem_report.get("premium_max")
            raise DatasetLoaderError(
                f"Premium Selection removed all rows (LTP {lo}–{hi}). "
                "Widen the range or disable the filter."
            )

    # Phase X: Feature Dataset ⟕ Label Run (once, before row cap / matrix build).
    if label_run_id:
        try:
            from chain_replay_ml.label_runs import join_feature_frame_with_label_run

            df, join_info = join_feature_frame_with_label_run(
                df, data_dir, label_run_id, drop_invalid=True
            )
            load_metrics["label_join"] = join_info
            load_metrics["rows_returned"] = int(len(df))
            metadata = dict(metadata)
            metadata["label_run_id"] = label_run_id
            metadata["label_join"] = join_info
            try:
                from chain_replay_ml.label_runs import load_label_run_meta

                lr_meta = load_label_run_meta(data_dir, label_run_id)
                if isinstance(lr_meta.get("label_encoding"), dict):
                    metadata["label_encoding"] = dict(lr_meta["label_encoding"])
            except Exception:
                pass
        except Exception as exc:
            raise DatasetLoaderError(
                f"Label Run join failed ({label_run_id}): {exc}"
            ) from exc

    metadata = dict(metadata)
    metadata["dataset_load"] = load_metrics
    _log_dataset_load_metrics(config.dataset, load_metrics)

    df, cap_meta = cap_training_rows(df)
    if cap_meta:
        metadata["training_memory"] = cap_meta
    identity_cols = [c for c in _IDENTITY_COLUMNS if c in df.columns]
    context_cols = list(
        dict.fromkeys(
            [
                *identity_cols,
                *[c for c in ("ltp_to_spot_ratio",) if c in df.columns and c not in identity_cols],
            ]
        )
    )
    context_df = df[context_cols].copy() if context_cols else pd.DataFrame()
    schema = load_schema_registry()
    X, y, features = select_xy(df, config, schema=schema)
    del df
    release_memory()

    X = compact_numeric_frame(X)
    y = pd.to_numeric(y, errors="coerce").astype("float32")
    # Binary Hit on OLE label_id (TP/SL/TIME) → {0,1} before any fold/trainer.
    try:
        from .label_prep import adapt_target_for_prediction_type

        y, adapt_meta = adapt_target_for_prediction_type(
            y,
            prediction_type=str(getattr(config, "prediction_type", None) or "regression"),
            target=str(getattr(config, "target", None) or ""),
            label_encoding=(
                metadata.get("label_encoding")
                if isinstance(metadata.get("label_encoding"), dict)
                else None
            ),
        )
        if adapt_meta.get("mode") not in (None, "passthrough"):
            metadata["label_adapt"] = adapt_meta
    except ValueError as exc:
        raise DatasetLoaderError(str(exc)) from exc
    release_memory()
    return X, y, features, metadata, expected, context_df
