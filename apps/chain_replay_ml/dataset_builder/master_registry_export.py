"""Export filtered master SQLite rows into the dataset registry (Parquet + JSON)."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Callable, Sequence
from zoneinfo import ZoneInfo

import pandas as pd

from .expected_spec import write_expected_spec
from .master_defaults import (
    default_master_feature_selection,
    default_master_prediction_targets,
    default_master_sampling,
    default_master_strike_selection,
)
from .master_naming import path_relative_to_data_dir, resolve_master_db_path
from .master_status import _read_table_info, _sample_filter_where
from .master_store import MasterStore
from .feature_plugins import resolve_implemented_features_for_selection
from .orchestrator import _load_feature_registry
from .pipeline_identity import (
    BUILDER_VERSION,
    METADATA_VERSION,
    build_pipeline_fingerprint,
    build_version_metadata_fields,
)
from .expected_spec import strike_selection_metadata
from .lookback_policy import build_dataset_configuration
from .spec_identity import compute_spec_hash_from_fingerprint
from .writer import _coerce_parquet_frame, _safe_filename, datasets_dir, ensure_parquet_engine

_CHUNK = 50_000
_IST = ZoneInfo("Asia/Kolkata")


class MasterRegistryExportError(Exception):
    def __init__(self, detail: str) -> None:
        self.detail = detail
        super().__init__(detail)


def master_registry_dataset_name(
    *,
    feature_count: int,
    interval_sec: int,
    created_at: datetime | None = None,
) -> str:
    """MS_{features}f_{interval}s_{HHMM} using local IST clock."""
    dt = created_at or datetime.now(_IST)
    hhmm = f"{dt.hour:02d}{dt.minute:02d}"
    return f"MS_{int(feature_count)}f_{int(interval_sec)}s_{hhmm}"


def _resolve_unique_dataset_name(data_dir: str, base_name: str) -> str:
    safe = _safe_filename(base_name)
    out_dir = datasets_dir(data_dir)
    candidate = safe
    n = 2
    while os.path.isfile(os.path.join(out_dir, f"{candidate}.parquet")) or os.path.isfile(
        os.path.join(out_dir, f"{candidate}.json")
    ):
        candidate = f"{safe}_{n}"
        n += 1
    return candidate


def _filter_summary_dict(
    *,
    all_days: bool,
    trading_day: str | None,
    selected_days: Sequence[str] | None = None,
    token: str | None,
    atm_band_filter: int | None,
    premium_min: float | None,
    premium_max: float | None,
    delta_min: float | None,
    delta_max: float | None,
    premium_enabled: bool = False,
    delta_enabled: bool = False,
    no_null_data: bool = False,
) -> dict[str, Any]:
    from .dataset_selection_engine import DatasetSelectionSpec

    days = [str(d).strip() for d in (selected_days or []) if str(d).strip()]
    spec = DatasetSelectionSpec(
        mode="post_filter",
        all_days=bool(all_days),
        single_day=trading_day,
        selected_days=days,
        token=token,
        atm_band=atm_band_filter,
        premium_min=premium_min,
        premium_max=premium_max,
        premium_enabled=bool(premium_enabled),
        delta_min=delta_min,
        delta_max=delta_max,
        delta_enabled=bool(delta_enabled),
    )
    return spec.to_filter_summary_dict() | ({"no_null_data": True} if no_null_data else {})


def build_master_selection_method(
    *,
    market: str,
    interval_sec: int,
    all_days: bool,
    trading_day: str | None,
    selected_days: Sequence[str] | None = None,
    token: str | None,
    atm_band_filter: int | None,
    premium_min: float | None,
    premium_max: float | None,
    delta_min: float | None,
    delta_max: float | None,
    premium_enabled: bool = False,
    delta_enabled: bool = False,
    no_null_data: bool = False,
    trading_day_filter: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from .dataset_selection_engine import format_day_scope_label
    from .trading_day_filter import trading_day_filter_label

    parts: list[str] = []
    days = [str(d).strip() for d in (selected_days or []) if str(d).strip()]
    parts.append(
        format_day_scope_label(
            all_days=all_days,
            selected_days=days,
            trading_day=trading_day,
        )
    )
    if token:
        parts.append(f"token {token}")
    if atm_band_filter is not None:
        parts.append(f"ATM ±{int(atm_band_filter)}")
    else:
        parts.append("ATM all")
    if premium_enabled and premium_min is not None and premium_max is not None:
        parts.append(f"LTP {float(premium_min):g}–{float(premium_max):g}")
    else:
        parts.append("Premium off")
    if delta_enabled and delta_min is not None and delta_max is not None:
        parts.append(f"|δ| {float(delta_min):g}–{float(delta_max):g}")
    else:
        parts.append("Delta off")
    if no_null_data:
        parts.append("No null data")
    if isinstance(trading_day_filter, dict) and trading_day_filter.get("mode"):
        mode = str(trading_day_filter.get("mode") or "all")
        if mode != "all":
            parts.append(trading_day_filter_label(mode))

    criteria = _filter_summary_dict(
        all_days=all_days,
        trading_day=trading_day,
        selected_days=days,
        token=token,
        atm_band_filter=atm_band_filter,
        premium_min=premium_min,
        premium_max=premium_max,
        delta_min=delta_min,
        delta_max=delta_max,
        premium_enabled=premium_enabled,
        delta_enabled=delta_enabled,
        no_null_data=no_null_data,
    )
    summary = " · ".join(parts)
    return {
        "method": "master_filter_export",
        "source": "master_db",
        "label": "Master DB filter",
        "summary": summary,
        "market": str(market or "NIFTY").upper(),
        "interval_sec": int(interval_sec),
        "criteria": criteria,
    }


def selection_method_for_registry(meta: dict[str, Any]) -> str | None:
    """Registry display string for selection method (master export or build)."""
    sm = meta.get("selection_method")
    if isinstance(sm, dict):
        text = str(sm.get("summary") or sm.get("label") or "").strip()
        return text or None
    if isinstance(sm, str) and sm.strip():
        return sm.strip()

    export_src = str(meta.get("export_source") or meta.get("source") or "").lower()
    mf = meta.get("master_filter")
    if export_src == "master_filter_export" or isinstance(mf, dict):
        crit = mf if isinstance(mf, dict) else {}
        return build_master_selection_method(
            market=str(meta.get("market") or "NIFTY"),
            interval_sec=int((meta.get("sampling") or {}).get("interval_sec") or 10),
            all_days=bool(crit.get("all_days")),
            trading_day=str(crit.get("trading_day") or "").strip() or None,
            token=str(crit.get("token") or "").strip() or None,
            atm_band_filter=crit.get("atm_band_filter"),
            premium_min=crit.get("premium_min"),
            premium_max=crit.get("premium_max"),
            delta_min=crit.get("delta_min"),
            delta_max=crit.get("delta_max"),
            premium_enabled=bool(crit.get("premium_enabled"))
            or (crit.get("premium_min") is not None and crit.get("premium_max") is not None),
            delta_enabled=bool(crit.get("delta_enabled"))
            or (crit.get("delta_min") is not None and crit.get("delta_max") is not None),
            no_null_data=bool(crit.get("no_null_data")),
        ).get("summary")

    strike = meta.get("strike_selection") if isinstance(meta.get("strike_selection"), dict) else {}
    if strike:
        from .expected_spec import format_strike_selection_label

        label = format_strike_selection_label(strike)
        if label:
            return label
    return None


def _export_filtered_parquet(
    conn: sqlite3.Connection,
    *,
    export_cols: list[str],
    where_sql: str,
    params: list[Any],
    parquet_path: str,
    all_days: bool,
    market: str = "NIFTY",
    on_progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Stream matching rows to Parquet; accumulate day/expiry stats (no post-COUNT)."""
    import time

    from .classification_labels import (
        attach_up_pct_classification_labels_5m,
        can_generate_up_pct_labels_5m,
    )

    ensure_parquet_engine()

    col_sql = ", ".join(f'"{c}"' for c in export_cols)
    order_sql = "trading_day, timestamp, token" if all_days else "timestamp, token"
    cur = conn.execute(
        f'SELECT {col_sql} FROM samples WHERE {where_sql} ORDER BY {order_sql}',
        params,
    )

    day_i = export_cols.index("trading_day") if "trading_day" in export_cols else -1
    market_i = export_cols.index("market") if "market" in export_cols else -1
    expiry_i = export_cols.index("expiry") if "expiry" in export_cols else -1
    mkt_default = str(market or "NIFTY").upper()
    generate_cls = can_generate_up_pct_labels_5m(export_cols)
    generated_cls_labels: list[str] = []

    day_counts: dict[str, int] = {}
    market_counts: dict[str, int] = {}
    expiry_counts: dict[str, int] = {}
    # (day, market, expiry) → n  for dominant-expiry selection
    day_mkt_exp: dict[tuple[str, str, str], int] = {}
    first_seen: dict[tuple[str, str, str], int] = {}
    seen_order = 0

    os.makedirs(os.path.dirname(parquet_path) or ".", exist_ok=True)
    tmp_path = parquet_path + ".tmp"
    writer = None
    fixed_schema = None
    written = 0
    t0 = time.perf_counter()
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        while True:
            raw = cur.fetchmany(_CHUNK)
            if not raw:
                break
            # Accumulate metadata from the same rows written to parquet.
            for row in raw:
                if day_i >= 0:
                    day = str(row[day_i] or "").strip()
                    if day:
                        day_counts[day] = day_counts.get(day, 0) + 1
                else:
                    day = ""
                mkt = (
                    str(row[market_i] or mkt_default).upper()
                    if market_i >= 0
                    else mkt_default
                )
                market_counts[mkt] = market_counts.get(mkt, 0) + 1
                exp = str(row[expiry_i] or "") if expiry_i >= 0 else ""
                if exp:
                    expiry_counts[exp] = expiry_counts.get(exp, 0) + 1
                if day:
                    key = (day, mkt, exp)
                    if key not in first_seen:
                        first_seen[key] = seen_order
                        seen_order += 1
                    day_mkt_exp[key] = day_mkt_exp.get(key, 0) + 1

            chunk = _coerce_parquet_frame(pd.DataFrame(raw, columns=export_cols))
            if generate_cls:
                added = attach_up_pct_classification_labels_5m(chunk)
                if added and not generated_cls_labels:
                    generated_cls_labels = list(added)
            try:
                from chain_replay_ml.frame_backend import frame_to_arrow_table_via_polars

                # Already coerced above; skip second coerce for stable chunk schemas.
                table = frame_to_arrow_table_via_polars(chunk, coerce=False)
            except Exception:
                table = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                fixed_schema = table.schema
                writer = pq.ParquetWriter(tmp_path, fixed_schema)
            elif table.schema != fixed_schema:
                table = table.cast(fixed_schema)
            writer.write_table(table)
            written += len(raw)
            if on_progress:
                elapsed = max(time.perf_counter() - t0, 1e-9)
                rate = written / elapsed
                on_progress(
                    f"Writing registry Parquet… {written:,} rows ({rate:,.0f}/s)",
                    written,
                    0,
                )
        if written == 0:
            raise MasterRegistryExportError(
                "No rows matched the selected filters.\nDataset was not created."
            )
        if writer is not None:
            writer.close()
            writer = None
        if os.path.isfile(parquet_path):
            os.remove(parquet_path)
        os.replace(tmp_path, parquet_path)
    except MasterRegistryExportError:
        if writer is not None:
            writer.close()
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise
    except ValueError as exc:
        if writer is not None:
            writer.close()
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise MasterRegistryExportError(
            f"Parquet export failed (schema mismatch): {exc}"
        ) from exc
    except Exception:
        if writer is not None:
            writer.close()
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
        raise

    elapsed_sec = max(time.perf_counter() - t0, 0.0)
    days_meta = _days_meta_from_export_counts(
        day_mkt_exp,
        first_seen=first_seen,
        market_default=mkt_default,
    )
    return {
        "row_count": written,
        "elapsed_sec": round(elapsed_sec, 3),
        "rows_per_sec": round(written / elapsed_sec, 1) if elapsed_sec > 0 else 0.0,
        "day_counts": day_counts,
        "market_counts": market_counts,
        "expiry_counts": expiry_counts,
        "days_meta": days_meta,
        "classification_labels": generated_cls_labels,
    }


def _days_meta_from_export_counts(
    day_mkt_exp: dict[tuple[str, str, str], int],
    *,
    first_seen: dict[tuple[str, str, str], int],
    market_default: str,
) -> list[dict[str, Any]]:
    """Dominant expiry per day — same rule as ORDER BY n DESC, first wins on ties."""
    best: dict[str, tuple[int, int, str, str]] = {}
    # value: (count, -first_seen, market, expiry) so max() prefers higher count, earlier seen
    for (day, mkt, exp), n in day_mkt_exp.items():
        order = int(first_seen.get((day, mkt, exp), 10**9))
        cand = (int(n), -order, mkt, exp)
        prev = best.get(day)
        if prev is None or cand > prev:
            best[day] = cand
    out: list[dict[str, Any]] = []
    mkt_default = str(market_default or "NIFTY").upper()
    for day in sorted(best.keys()):
        _n, _ord, mkt, exp = best[day]
        out.append({
            "trading_day": day,
            "market": str(mkt or mkt_default).upper(),
            "expiry": str(exp or ""),
            "source_id": f"{mkt_default}_{day}",
        })
    return out


def _days_meta_from_filtered(
    conn: sqlite3.Connection,
    *,
    where_sql: str,
    params: list[Any],
    market: str,
) -> list[dict[str, Any]]:
    """Legacy SQL path — kept for tests/debug; export uses in-loop accumulation."""
    rows = conn.execute(
        f"""
        SELECT trading_day, market, expiry, COUNT(*) AS n
        FROM samples
        WHERE {where_sql}
        GROUP BY trading_day, market, expiry
        ORDER BY trading_day, n DESC
        """,
        params,
    ).fetchall()
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    mkt = str(market or "NIFTY").upper()
    for r in rows:
        td = str(r[0] or "").strip()
        if not td or td in seen:
            continue
        seen.add(td)
        out.append({
            "trading_day": td,
            "market": str(r[1] or mkt).upper(),
            "expiry": str(r[2] or ""),
            "source_id": f"{mkt}_{td}",
        })
    return out


def _source_results_from_days(days_meta: list[dict[str, Any]], row_counts: dict[str, int]) -> list[dict[str, Any]]:
    return [
        {
            "trading_day": d["trading_day"],
            "market": d["market"],
            "expiry": d.get("expiry") or "",
            "status": "loaded",
            "rows": int(row_counts.get(d["trading_day"], 0)),
        }
        for d in days_meta
    ]


def create_master_registry_dataset(
    data_dir: str,
    *,
    market: str,
    interval_sec: int,
    trading_day: str | None = None,
    selected_days: Sequence[str] | None = None,
    token: str | None = None,
    master_db_path: str | None = None,
    atm_band_filter: int | None = None,
    premium_min: float | None = None,
    premium_max: float | None = None,
    delta_min: float | None = None,
    delta_max: float | None = None,
    all_days: bool = False,
    dataset_name: str | None = None,
    audit_validation_required: bool = False,
    premium_enabled: bool = False,
    delta_enabled: bool = False,
    no_null_data: bool = False,
    trading_day_filter: dict[str, Any] | None = None,
    transformation_config: dict[str, Any] | None = None,
    keep_pipeline_owned: bool = False,
    dataset_kind: str | None = None,
    pipeline_no_null_report: bool = False,
    registry_export_features: frozenset[str] | None = None,
    pipeline_provenance: dict[str, Any] | None = None,
    base_pipeline_export_features: frozenset[str] | None = None,
    feature_project_id: str | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> dict[str, Any]:
    """Export filtered master rows to dataset registry as Parquet + metadata JSON.

    ``keep_pipeline_owned=True`` retains pipeline-owned / interaction columns on the
    Master table (used by Analysis Dataset builds). Default registry exports still
    strip those columns so the Feature Registry stays canonical.

    ``pipeline_no_null_report=True`` streams a diagnostic Pipeline No-Null Report
    (Activity Log) after transforms — does not change export data by itself.
    """
    td = str(trading_day or "").strip() or None
    day_list = [str(d).strip() for d in (selected_days or []) if str(d).strip()]
    if not all_days and not td and not day_list:
        raise MasterRegistryExportError(
            "trading_day or selected_days is required unless all_days is true",
        )

    path = master_db_path or resolve_master_db_path(
        data_dir,
        market=market,
        sampling_interval_sec=interval_sec,
    )
    if not os.path.isfile(path):
        raise MasterRegistryExportError("Master database file does not exist")

    token_val = str(token or "").strip() or None
    atm_filter = int(atm_band_filter) if atm_band_filter is not None and int(atm_band_filter) >= 0 else None
    prem_lo = float(premium_min) if premium_enabled and premium_min is not None else None
    prem_hi = float(premium_max) if premium_enabled and premium_max is not None else None
    delta_lo = float(delta_min) if delta_enabled and delta_min is not None else None
    delta_hi = float(delta_max) if delta_enabled and delta_max is not None else None
    # Analysis / unseen Dataset: apply LTP premium after transforms + No-Null (not in SQL WHERE).
    # Master registry export keeps early SQL premium filtering unchanged.
    kind_norm = str(dataset_kind or "").strip().lower()
    analysis_like = kind_norm in ("analysis", "unseen")
    defer_premium_until_end = bool(
        analysis_like and prem_lo is not None and prem_hi is not None
    )
    sql_prem_lo = None if defer_premium_until_end else prem_lo
    sql_prem_hi = None if defer_premium_until_end else prem_hi
    premium_report: dict[str, Any] | None = None

    filter_meta: dict[str, Any] | None = None
    if isinstance(trading_day_filter, dict) and trading_day_filter:
        from .trading_day_filter import enrich_trading_day_filter_dates

        filter_meta = {
            "mode": str(trading_day_filter.get("mode") or "all"),
            "selected_days": int(trading_day_filter.get("selected_days") or 0),
            "exported_days": int(trading_day_filter.get("exported_days") or 0),
            "selected_dates": [
                str(d).strip() for d in (trading_day_filter.get("selected_dates") or []) if str(d).strip()
            ],
            "exported_dates": [
                str(d).strip() for d in (trading_day_filter.get("exported_dates") or []) if str(d).strip()
            ],
            "excluded_dates": [
                str(d).strip() for d in (trading_day_filter.get("excluded_dates") or []) if str(d).strip()
            ],
            "expiry_dates": [
                str(d).strip() for d in (trading_day_filter.get("expiry_dates") or []) if str(d).strip()
            ],
        }

    _progress_early: Callable[[str, int, int], None] | None = on_progress

    def _progress(msg: str, cur: int = 0, tot: int = 0, **detail: Any) -> None:
        if _progress_early is not None:
            try:
                _progress_early(str(msg), int(cur or 0), int(tot or 0), **detail)
            except TypeError:
                try:
                    _progress_early(str(msg), int(cur or 0), int(tot or 0))
                except Exception:
                    pass
            except Exception:
                pass

    _progress("Create Dataset: loading master metadata…", 0, 0)
    store = MasterStore(path)
    store.open()
    try:
        master_config = store.get_meta("master_config") or {}
        build_schema = store.get_meta("build_schema") or {}
        feature_columns = list(build_schema.get("feature_columns") or [])
        target_columns = list(build_schema.get("target_columns") or [])
        feature_count = int(
            build_schema.get("feature_count")
            or master_config.get("feature_count")
            or len(feature_columns)
            or 0
        )
        target_count = int(
            build_schema.get("target_count")
            or master_config.get("target_count")
            or len(target_columns)
            or 0
        )
        master_day_rows = store.read_master_days()
        from .master_feature_project import (
            ensure_master_feature_project_id,
            normalize_feature_project_id,
            validate_feature_project_id,
        )

        if feature_project_id:
            bound_feature_project_id = validate_feature_project_id(
                data_dir,
                normalize_feature_project_id(feature_project_id),
            )
        else:
            bound_feature_project_id = ensure_master_feature_project_id(store, data_dir)
    finally:
        store.close()

    if filter_meta is not None:
        from .trading_day_filter import enrich_trading_day_filter_dates

        filter_meta = enrich_trading_day_filter_dates(
            filter_meta,
            master_day_rows=master_day_rows,
        )
    if feature_count <= 0 and feature_columns:
        feature_count = len(feature_columns)

    created_dt = datetime.now(_IST)
    base_name = dataset_name or master_registry_dataset_name(
        feature_count=feature_count,
        interval_sec=interval_sec,
        created_at=created_dt,
    )
    safe_name = _resolve_unique_dataset_name(data_dir, base_name)

    _progress(f"Create Dataset: opening master DB for export ({safe_name})…", 0, 0)
    conn = sqlite3.connect(path)
    generated_cls_labels: list[str] = []
    from .transformations import (
        default_transformation_config,
        describe_pipeline,
        normalize_transformation_config,
    )

    transformation_config = normalize_transformation_config(
        transformation_config if transformation_config is not None else default_transformation_config()
    )
    pipe_plan = describe_pipeline(transformation_config)
    transformations_enabled = int(pipe_plan.enabled) > 0
    # When Feature Transformations are enabled, No-Null runs only AFTER they
    # finish (Analysis and Manual). Do not pre-filter Registry rows before
    # Pipeline Features — Lag/Diff/Interaction must see the full selected
    # partition; one No-Null pass then applies to Registry ∪ Pipeline columns.
    staged_analysis_no_null = False
    defer_no_null_until_after_transform = bool(
        no_null_data and transformations_enabled
    )
    try:
        col_names = [c["name"] for c in _read_table_info(conn, "samples")]
        if not col_names:
            raise MasterRegistryExportError("samples table has no columns")

        # Drop obsolete Master feature columns (pipeline-owned / retired) so the
        # exported registry dataset matches Feature Registry (_REGISTRY_FEATURES).
        # Analysis Dataset builds set keep_pipeline_owned=True to keep Pipeline Features.
        from .transformations.lag_ui import (
            canonical_registry_feature_names,
            filter_to_registry_features,
        )

        registry_names = canonical_registry_feature_names()
        if data_dir:
            from .feature_registry_store import disabled_registry_feature_names, load_store

            registry_names -= disabled_registry_feature_names(load_store(data_dir))
        if feature_columns and registry_names and not keep_pipeline_owned:
            kept_features = filter_to_registry_features(feature_columns)
            stale_features = set(feature_columns) - set(kept_features)
            # Also drop known migrated columns even if build_schema list is incomplete.
            try:
                from .feature_migration import is_pipeline_owned, is_retired
                from .feature_ownership import is_interaction_feature

                for c in col_names:
                    if c in registry_names:
                        continue
                    if is_pipeline_owned(c) or is_retired(c) or is_interaction_feature(c):
                        stale_features.add(c)
            except Exception:
                pass
            if stale_features:
                col_names = [c for c in col_names if c not in stale_features]
                feature_columns = list(kept_features)
                feature_count = len(feature_columns)
                _progress(
                    f"Feature Registry filter: kept {feature_count} canonical features, "
                    f"dropped {len(stale_features)} obsolete Master columns.",
                    0,
                    0,
                )
        elif keep_pipeline_owned:
            _progress(
                "Analysis Dataset mode: keeping Registry + Pipeline Feature columns.",
                0,
                0,
            )

        where_sql, params = _sample_filter_where(
            trading_day=td,
            selected_days=day_list or None,
            token=token_val,
            atm_band_filter=atm_filter,
            premium_min=sql_prem_lo,
            premium_max=sql_prem_hi,
            delta_min=delta_lo,
            delta_max=delta_hi,
            column_names=col_names,
        )

        export_cols = list(col_names)
        fetch_where = where_sql
        no_null_dropped_columns: list[str] = []
        no_null_report: dict[str, Any] | None = None
        registry_no_null_report: dict[str, Any] | None = None
        if no_null_data and not defer_no_null_until_after_transform:
            from .non_null_filter import apply_non_null_filter, format_non_null_report

            nn_input_cols = list(col_names)
            if staged_analysis_no_null:
                # Registry No-Null only — stale Pipeline columns on Master must not
                # shrink the row set before Pipeline Features are (re)built.
                try:
                    from .transformations.lag_ui import META_SKIP_COLUMNS

                    reg_set = set(registry_names or ())
                    nn_input_cols = [
                        c
                        for c in col_names
                        if c in META_SKIP_COLUMNS or c in reg_set
                    ]
                    if not nn_input_cols:
                        nn_input_cols = list(col_names)
                except Exception:
                    nn_input_cols = list(col_names)
                _progress(
                    "No-Null filter: Registry Features "
                    "(before Pipeline Features)…",
                    0,
                    0,
                )
            else:
                _progress(
                    "No-Null filter starting — column discovery can take several minutes "
                    "on large masters (UI will update when each step finishes)…",
                    0,
                    0,
                )
            nn = apply_non_null_filter(
                conn,
                nn_input_cols,
                where_sql,
                params,
                log=False,
                debug=False,
                on_stage=lambda msg: _progress(msg, 0, 0),
            )
            fetch_where = str(nn["where_sql"])
            dropped_nn = set(nn["dropped_columns"])
            if staged_analysis_no_null:
                # Keep non-registry columns for the export stream; only drop
                # Registry columns that Step 1 found 100% NULL.
                export_cols = [c for c in col_names if c not in dropped_nn]
                no_null_dropped_columns = [c for c in nn["dropped_columns"]]
                registry_no_null_report = dict(nn.get("report") or {})
                registry_no_null_report["stage"] = "registry_pre_transformation"
                no_null_report = dict(registry_no_null_report)
            else:
                export_cols = list(nn["kept_columns"])
                no_null_dropped_columns = list(nn["dropped_columns"])
                no_null_report = dict(nn.get("report") or {})
            if not export_cols:
                raise MasterRegistryExportError(
                    "No active columns remain after removing 100% NULL columns."
                )
            # remaining_null audit is debug-only; Step 2 WHERE already enforces NOT NULL.
            if no_null_report:
                _progress(format_non_null_report(no_null_report), 0, 0)
        out_dir = datasets_dir(data_dir)
        parquet_path = os.path.join(out_dir, f"{safe_name}.parquet")
        json_path = os.path.join(out_dir, f"{safe_name}.json")

        _progress("Loading Master Dataset...", 0, 0)

        _progress(
            f"Stage input: streaming {len(export_cols)} columns "
            f"({len(no_null_dropped_columns)} empty columns already dropped)…",
            0,
            0,
        )
        export_stats = _export_filtered_parquet(
            conn,
            export_cols=export_cols,
            where_sql=fetch_where,
            params=params,
            parquet_path=parquet_path,
            all_days=all_days,
            market=market,
            on_progress=on_progress,
        )
        row_count = int(export_stats["row_count"])
        row_counts_by_day = {
            str(k): int(v) for k, v in (export_stats.get("day_counts") or {}).items()
        }
        days_meta = list(export_stats.get("days_meta") or [])
        generated_cls_labels = list(export_stats.get("classification_labels") or [])
        if registry_no_null_report is not None:
            registry_no_null_report["rows_after"] = row_count
            registry_no_null_report["rows_by_day_after"] = dict(row_counts_by_day)
            if no_null_report is not None and no_null_report is registry_no_null_report:
                pass
            elif no_null_report is not None and str(no_null_report.get("stage")) == (
                "registry_pre_transformation"
            ):
                no_null_report["rows_after"] = row_count
            _progress(
                f"Registry No-Null complete: {row_count:,} rows "
                f"(Pipeline Features not applied yet).",
                0,
                0,
            )
    finally:
        conn.close()

    # Apply feature transformations on the full export (Lag needs complete partitions).
    transformed_columns: list[str] = []
    transformation_meta = dict(transformation_config)
    pipeline_no_null_report_text: str | None = None
    final_column_count = (
        len(export_cols) + len(generated_cls_labels) + len(transformed_columns)
    )
    from .transformations import (
        format_pipeline_log_lines,
        run_transformation_pipeline,
    )
    from .transformations.base import TransformContext
    from .transformations.lag import LagConfigError

    if transformations_enabled:
        _progress("Feature Transformations: applying to loaded master dataset…", 0, 0)

        def _xform_log(msg: str) -> None:
            _progress(str(msg), 0, 0)

        ctx = TransformContext(
            config=transformation_config,
            data_dir=data_dir,
            dataset_name=safe_name,
            sample_interval_sec=float(interval_sec),
            dataset_info={
                "market": str(market or "NIFTY").upper(),
                "sample_interval_sec": float(interval_sec),
            },
            metadata={"sampling": {"interval_sec": int(interval_sec)}},
            logger=_xform_log,
            progress_callback=on_progress,
        )

        # Analysis / unseen / large / multi-stage exports: transform one trading_day at a time
        # so peak RAM stays bounded (full-frame path OOMs around ~300k rows × 400 cols).
        use_partitioned = (
            analysis_like
            or int(row_count or 0) >= 80_000
            or int(pipe_plan.enabled or 0) >= 8
        )
        final_frame: Any = None
        need_write_final_frame = bool(no_null_data or defer_premium_until_end)
        need_final_frame = bool(
            need_write_final_frame or pipeline_no_null_report
        )
        try:
            if use_partitioned:
                from .transformations.pipeline import run_transformation_pipeline_on_parquet

                _progress(
                    "Feature Transformations: day-at-a-time "
                    "(Fast per day when RAM allows; Safe token/wave fallback)…",
                    0,
                    0,
                )
                pipe = run_transformation_pipeline_on_parquet(
                    parquet_path,
                    transformation_config,
                    context=ctx,
                    log_fn=_xform_log,
                    partition_col="trading_day",
                    on_partition_progress=lambda msg, cur, tot, **detail: _progress(
                        msg, cur, tot, **detail
                    ),
                )
                transformation_meta = dict(pipe.metadata_block)
                transformed_columns = list(pipe.created_columns)
                # Never load the full transformed parquet into RAM for No-Null /
                # Premium — that OOM/crashed Analysis builds (~7GB / 19 days).
                # Apply filters day-at-a-time and rewrite parquet in place.
                if need_write_final_frame:
                    from .non_null_filter import (
                        apply_non_null_filter_on_parquet,
                        format_non_null_report,
                    )
                    from .premium_ltp_filter import apply_premium_ltp_filter_frame

                    pipeline_step2 = (
                        list(transformed_columns) if staged_analysis_no_null else None
                    )
                    prem_lo_f = float(prem_lo) if prem_lo is not None else None
                    prem_hi_f = float(prem_hi) if prem_hi is not None else None
                    if no_null_data:
                        _progress(
                            "No-Null / Premium: day-at-a-time after Feature Transformations…",
                            0,
                            0,
                        )
                        filtered = apply_non_null_filter_on_parquet(
                            parquet_path,
                            step2_columns=pipeline_step2,
                            transformation_config=transformation_config,
                            on_stage=lambda msg: _progress(msg, 0, 0),
                            premium_min=prem_lo_f if defer_premium_until_end else None,
                            premium_max=prem_hi_f if defer_premium_until_end else None,
                        )
                        pipe_dropped = list(filtered.get("dropped_columns") or [])
                        no_null_dropped_columns = list(
                            dict.fromkeys(list(no_null_dropped_columns) + pipe_dropped)
                        )
                        no_null_report = dict(filtered.get("report") or {})
                        _progress(format_non_null_report(no_null_report), 0, 0)
                        if defer_premium_until_end:
                            premium_report = dict(filtered.get("premium_report") or {})
                            if premium_report:
                                _progress(
                                    f"Premium filter: kept {premium_report.get('rows_after', 0):,} / "
                                    f"{premium_report.get('rows_before', 0):,} rows "
                                    f"(LTP {prem_lo_f:g}–{prem_hi_f:g})",
                                    0,
                                    0,
                                )
                        row_count = int(filtered.get("row_count") or 0)
                        final_column_count = int(filtered.get("column_count") or 0)
                        kept_set = set(filtered.get("kept_columns") or [])
                        transformed_columns = [c for c in transformed_columns if c in kept_set]
                    elif defer_premium_until_end and prem_lo_f is not None and prem_hi_f is not None:
                        # Premium only — still day-at-a-time (no full-frame load).
                        import gc

                        import pyarrow as pa
                        import pyarrow.parquet as pq

                        _progress(
                            f"Premium filter: LTP {prem_lo_f:g}–{prem_hi_f:g} "
                            "(day-at-a-time after transforms)…",
                            0,
                            0,
                        )
                        days_table = pq.read_table(parquet_path, columns=["trading_day"])
                        days = sorted(
                            {
                                str(x)
                                for x in days_table.column(0).to_pylist()
                                if x is not None and str(x).strip()
                            }
                        )
                        del days_table
                        out_tmp = f"{parquet_path}.premium.tmp"
                        writer = None
                        before_n = 0
                        after_n = 0
                        for day in days:
                            df = pq.read_table(
                                parquet_path, filters=[("trading_day", "=", day)]
                            ).to_pandas()
                            before_n += len(df)
                            prem = apply_premium_ltp_filter_frame(
                                df, premium_min=prem_lo_f, premium_max=prem_hi_f
                            )
                            out = prem["frame"]
                            after_n += len(out)
                            if out.empty:
                                del df, out, prem
                                gc.collect()
                                continue
                            try:
                                from chain_replay_ml.frame_backend import (
                                    frame_to_arrow_table_via_polars,
                                )

                                table = frame_to_arrow_table_via_polars(out, coerce=True)
                            except Exception:
                                table = pa.Table.from_pandas(out, preserve_index=False)
                            if writer is None:
                                writer = pq.ParquetWriter(
                                    out_tmp, table.schema, compression="zstd"
                                )
                            writer.write_table(table)
                            del df, out, prem, table
                            gc.collect()
                        if writer is not None:
                            writer.close()
                        if not os.path.isfile(out_tmp):
                            raise MasterRegistryExportError(
                                "No rows remain after the LTP premium filter "
                                f"({prem_lo_f:g}–{prem_hi_f:g})."
                            )
                        os.replace(out_tmp, parquet_path)
                        premium_report = {
                            "stage": "post_transform_day_at_a_time",
                            "premium_min": prem_lo_f,
                            "premium_max": prem_hi_f,
                            "rows_before": int(before_n),
                            "rows_after": int(after_n),
                            "rows_dropped": max(int(before_n - after_n), 0),
                        }
                        row_count = int(after_n)
                        final_column_count = len(pq.read_schema(parquet_path).names)
                        _progress(
                            f"Premium filter: kept {after_n:,} / {before_n:,} rows "
                            f"(LTP {prem_lo_f:g}–{prem_hi_f:g})",
                            0,
                            0,
                        )
                    if row_count < 1 and (no_null_data or defer_premium_until_end):
                        raise MasterRegistryExportError(
                            "No rows remain after Feature Transformations and "
                            "the No-Null / premium filters."
                        )
                    # Filters already rewrote parquet — skip later full-frame write.
                    need_write_final_frame = False
                    final_frame = None
                else:
                    try:
                        import pyarrow.parquet as pq

                        final_column_count = len(pq.read_schema(parquet_path).names)
                    except Exception:
                        final_column_count = 0
            else:
                try:
                    df_export = pd.read_parquet(parquet_path)
                except Exception as exc:
                    raise MasterRegistryExportError(
                        f"Failed to reload export for transformations: {exc}"
                    ) from exc
                try:
                    pipe = run_transformation_pipeline(
                        df_export,
                        transformation_config,
                        context=ctx,
                        log_fn=_xform_log,
                    )
                except Exception as exc:
                    msg = str(exc).lower()
                    is_oom = isinstance(exc, MemoryError) or "unable to allocate" in msg
                    if not is_oom:
                        raise
                    _progress(
                        "Out of memory on full-frame transform — retrying day-by-day…",
                        0,
                        0,
                    )
                    try:
                        del df_export
                    except Exception:
                        pass
                    import gc

                    gc.collect()
                    from .transformations.pipeline import run_transformation_pipeline_on_parquet

                    pipe = run_transformation_pipeline_on_parquet(
                        parquet_path,
                        transformation_config,
                        context=ctx,
                        log_fn=_xform_log,
                        partition_col="trading_day",
                        on_partition_progress=lambda msg, cur, tot, **detail: _progress(
                        msg, cur, tot, **detail
                    ),
                    )
                    transformation_meta = dict(pipe.metadata_block)
                    transformed_columns = list(pipe.created_columns)
                    if need_final_frame:
                        final_frame = pd.read_parquet(parquet_path)
                    else:
                        try:
                            import pyarrow.parquet as pq

                            final_column_count = len(pq.read_schema(parquet_path).names)
                        except Exception:
                            final_frame = pd.read_parquet(parquet_path)
                            final_column_count = len(final_frame.columns)
                            del final_frame
                            final_frame = None
                else:
                    transformation_meta = dict(pipe.metadata_block)
                    transformed_columns = list(pipe.created_columns)
                    final_frame = pipe.frame
                    del df_export
        except MemoryError as exc:
            raise MasterRegistryExportError(
                "Out of memory while applying Feature Transformations. "
                "Close other apps and retry; analysis builds process one day at a time. "
                f"Detail: {exc}"
            ) from exc
        except Exception as exc:
            msg = str(exc).lower()
            if "unable to allocate" in msg:
                raise MasterRegistryExportError(
                    "Out of memory while applying Feature Transformations. "
                    "Close other apps and retry; analysis builds process one day at a time. "
                    f"Detail: {exc}"
                ) from exc
            if isinstance(exc, (LagConfigError, ValueError, MasterRegistryExportError)):
                raise MasterRegistryExportError(str(exc)) from exc
            raise MasterRegistryExportError(str(exc)) from exc

        if final_frame is not None and (no_null_data or pipeline_no_null_report):
            from .non_null_filter import (
                apply_non_null_filter_frame,
                format_non_null_report,
            )
            from .pipeline_no_null_report import (
                build_pipeline_no_null_report_text,
                emit_pipeline_no_null_report_lines,
            )

            pipeline_step2 = (
                list(transformed_columns) if staged_analysis_no_null else None
            )
            # Keep pre-filter view for NULL attribution (filter returns a new frame).
            pre_pipeline_nn_frame = final_frame
            pre_pipeline_cols = list(transformed_columns)

            if no_null_data:
                _progress(
                    "No-Null filter: running after Feature Transformations…",
                    0,
                    0,
                )
                nn_frame = apply_non_null_filter_frame(
                    final_frame,
                    step2_columns=pipeline_step2,
                    transformation_config=transformation_config,
                    on_stage=lambda msg: _progress(msg, 0, 0),
                )
                final_frame = nn_frame["frame"]
                pipe_dropped = list(nn_frame["dropped_columns"])
                no_null_dropped_columns = list(
                    dict.fromkeys(list(no_null_dropped_columns) + pipe_dropped)
                )
                pipe_report = dict(nn_frame["report"])
                no_null_report = pipe_report
                if not list(nn_frame["kept_columns"]):
                    raise MasterRegistryExportError(
                        "No active columns remain after transformations and "
                        "removing 100% NULL columns."
                    )
                if final_frame.empty:
                    raise MasterRegistryExportError(
                        "No rows remain after Feature Transformations and the "
                        "No-Null row filter."
                    )
                transformed_columns = [
                    c for c in transformed_columns if c in final_frame.columns
                ]
                generated_cls_labels = [
                    c for c in generated_cls_labels if c in final_frame.columns
                ]
                _progress(format_non_null_report(no_null_report), 0, 0)

            if pipeline_no_null_report and pre_pipeline_cols:
                _progress(
                    "Pipeline No-Null Report: attributing NULL pipeline features…",
                    0,
                    0,
                )
                try:
                    pipeline_no_null_report_text = build_pipeline_no_null_report_text(
                        pre_pipeline_nn_frame,
                        pipeline_columns=pre_pipeline_cols,
                        transformation_config=transformation_config,
                        rows_after_filter=(
                            int(len(final_frame)) if no_null_data else None
                        ),
                        filter_applied=bool(no_null_data),
                    )
                    emit_pipeline_no_null_report_lines(
                        pipeline_no_null_report_text,
                        progress_fn=_progress,
                    )
                except Exception as exc:
                    _progress(f"Pipeline No-Null Report failed: {exc}", 0, 0)

        if final_frame is not None and defer_premium_until_end:
            from .premium_ltp_filter import apply_premium_ltp_filter_frame

            _progress(
                f"Premium filter: LTP {prem_lo:g}–{prem_hi:g} (after No-Null)…",
                0,
                0,
            )
            try:
                prem = apply_premium_ltp_filter_frame(
                    final_frame,
                    premium_min=float(prem_lo),
                    premium_max=float(prem_hi),
                )
            except ValueError as exc:
                raise MasterRegistryExportError(str(exc)) from exc
            final_frame = prem["frame"]
            premium_report = dict(prem["report"])
            if final_frame.empty:
                raise MasterRegistryExportError(
                    "No rows remain after the LTP premium filter "
                    f"({prem_lo:g}–{prem_hi:g})."
                )
            _progress(
                f"Premium filter: kept {premium_report['rows_after']:,} / "
                f"{premium_report['rows_before']:,} rows "
                f"(LTP {prem_lo:g}–{prem_hi:g})",
                0,
                0,
            )

        if final_frame is not None and need_write_final_frame:
            _progress("Export Dataset: writing final Parquet…", 0, 0)
            try:
                ensure_parquet_engine()
                from .writer import _write_parquet

                _write_parquet(final_frame, parquet_path)
            except Exception as exc:
                raise MasterRegistryExportError(
                    f"Failed to write transformed parquet: {exc}"
                ) from exc
            final_column_count = len(final_frame.columns)
        elif final_frame is not None:
            # Report-only load: parquet already has transform outputs.
            final_column_count = len(final_frame.columns)
            del final_frame
            final_frame = None

        # Post-transform No-Null / premium can change rows and columns;
        # refresh all export statistics from the matrix actually written.
        if (no_null_data or defer_premium_until_end) and final_frame is not None:
            row_count = len(final_frame)
            row_counts_by_day = (
                {
                    str(k): int(v)
                    for k, v in final_frame.groupby("trading_day").size().items()
                }
                if "trading_day" in final_frame.columns
                else {}
            )
            market_counts = (
                {
                    str(k): int(v)
                    for k, v in final_frame.groupby("market").size().items()
                }
                if "market" in final_frame.columns
                else {str(market or "NIFTY").upper(): row_count}
            )
            expiry_counts = (
                {
                    str(k): int(v)
                    for k, v in final_frame.groupby("expiry").size().items()
                    if str(k)
                }
                if "expiry" in final_frame.columns
                else {}
            )
            day_mkt_exp: dict[tuple[str, str, str], int] = {}
            first_seen: dict[tuple[str, str, str], int] = {}
            if "trading_day" in final_frame.columns:
                for i, row in final_frame.iterrows():
                    day = str(row.get("trading_day") or "").strip()
                    if not day:
                        continue
                    mkt = str(row.get("market") or market or "NIFTY").upper()
                    exp = str(row.get("expiry") or "")
                    key = (day, mkt, exp)
                    first_seen.setdefault(key, int(i) if isinstance(i, int) else 0)
                    day_mkt_exp[key] = day_mkt_exp.get(key, 0) + 1
            days_meta = _days_meta_from_export_counts(
                day_mkt_exp,
                first_seen=first_seen,
                market_default=str(market or "NIFTY").upper(),
            )
            export_stats = {
                **export_stats,
                "row_count": row_count,
                "day_counts": row_counts_by_day,
                "market_counts": market_counts,
                "expiry_counts": expiry_counts,
                "days_meta": days_meta,
            }
        elif final_column_count <= 0:
            try:
                import pyarrow.parquet as pq

                final_column_count = len(pq.read_schema(parquet_path).names)
            except Exception:
                pass

        for line in format_pipeline_log_lines(pipe):
            _progress(line, 0, 0)
    else:
        for line in format_pipeline_log_lines(pipe_plan):
            _progress(line, 0, 0)

        # No transforms: Analysis premium still runs last (after optional SQL No-Null).
        if defer_premium_until_end:
            from .premium_ltp_filter import apply_premium_ltp_filter_frame

            _progress(
                f"Premium filter: LTP {prem_lo:g}–{prem_hi:g} (after No-Null)…",
                0,
                0,
            )
            try:
                frame = pd.read_parquet(parquet_path)
            except Exception as exc:
                raise MasterRegistryExportError(
                    f"Failed to reload export for premium filter: {exc}"
                ) from exc
            try:
                prem = apply_premium_ltp_filter_frame(
                    frame,
                    premium_min=float(prem_lo),
                    premium_max=float(prem_hi),
                )
            except ValueError as exc:
                raise MasterRegistryExportError(str(exc)) from exc
            frame = prem["frame"]
            premium_report = dict(prem["report"])
            if frame.empty:
                raise MasterRegistryExportError(
                    "No rows remain after the LTP premium filter "
                    f"({prem_lo:g}–{prem_hi:g})."
                )
            try:
                ensure_parquet_engine()
                from .writer import _write_parquet

                _write_parquet(frame, parquet_path)
            except Exception as exc:
                raise MasterRegistryExportError(
                    f"Failed to write premium-filtered parquet: {exc}"
                ) from exc
            row_count = len(frame)
            final_column_count = len(frame.columns)
            row_counts_by_day = (
                {
                    str(k): int(v)
                    for k, v in frame.groupby("trading_day").size().items()
                }
                if "trading_day" in frame.columns
                else {}
            )
            market_counts = (
                {
                    str(k): int(v)
                    for k, v in frame.groupby("market").size().items()
                }
                if "market" in frame.columns
                else {str(market or "NIFTY").upper(): row_count}
            )
            expiry_counts = (
                {
                    str(k): int(v)
                    for k, v in frame.groupby("expiry").size().items()
                    if str(k)
                }
                if "expiry" in frame.columns
                else {}
            )
            day_mkt_exp = {}
            first_seen = {}
            if "trading_day" in frame.columns:
                for i, row in frame.iterrows():
                    day = str(row.get("trading_day") or "").strip()
                    if not day:
                        continue
                    mkt = str(row.get("market") or market or "NIFTY").upper()
                    exp = str(row.get("expiry") or "")
                    key = (day, mkt, exp)
                    first_seen.setdefault(key, int(i) if isinstance(i, int) else 0)
                    day_mkt_exp[key] = day_mkt_exp.get(key, 0) + 1
            days_meta = _days_meta_from_export_counts(
                day_mkt_exp,
                first_seen=first_seen,
                market_default=str(market or "NIFTY").upper(),
            )
            export_stats = {
                **export_stats,
                "row_count": row_count,
                "day_counts": row_counts_by_day,
                "market_counts": market_counts,
                "expiry_counts": expiry_counts,
                "days_meta": days_meta,
            }
            _progress(
                f"Premium filter: kept {premium_report['rows_after']:,} / "
                f"{premium_report['rows_before']:,} rows "
                f"(LTP {prem_lo:g}–{prem_hi:g})",
                0,
                0,
            )
            del frame

    if (
        keep_pipeline_owned
        and analysis_like
        and registry_export_features is not None
    ):
        from .registry_export_prune import prune_registry_columns_in_parquet

        prune_registry_columns_in_parquet(
            parquet_path,
            selected_registry=frozenset(registry_export_features),
            data_dir=data_dir,
            on_progress=lambda msg: _progress(str(msg), 0, 0),
        )
        try:
            import pyarrow.parquet as pq

            final_column_count = len(pq.read_schema(parquet_path).names)
        except Exception:
            pass

    from .classification_labels import (
        classification_label_meta,
        merge_classification_targets,
    )

    cls_label_meta: dict[str, Any] | None = None
    if generated_cls_labels:
        target_columns = merge_classification_targets(
            target_columns, generated=generated_cls_labels
        )
        target_count = len(target_columns)
        cls_label_meta = classification_label_meta()

    registry = _load_feature_registry()
    sampling = default_master_sampling(interval_sec)
    strike_selection = dict(default_master_strike_selection())
    actual_atm_band = int(
        master_config.get("atm_band")
        or atm_filter
        or strike_selection.get("atmBand")
        or 10
    )
    strike_selection["atmBand"] = actual_atm_band
    prediction_targets = default_master_prediction_targets()
    feature_selection = default_master_feature_selection(registry)
    enabled_groups, implemented, pending, per_group = resolve_implemented_features_for_selection(
        feature_selection,
        registry,
    )

    # Feature list must match columns actually written (after No-Null / premium).
    # Otherwise Create Model advertises ghost columns (e.g. 100% NULL VWAPs).
    declared_features = list(
        dict.fromkeys(list(feature_columns or implemented) + list(transformed_columns))
    )
    actual_col_names: set[str] | None = None
    try:
        import pyarrow.parquet as pq

        actual_col_names = set(pq.read_schema(parquet_path).names)
    except Exception:
        actual_col_names = None
    if actual_col_names is not None:
        declared_features = [c for c in declared_features if c in actual_col_names]
        transformed_columns = [c for c in transformed_columns if c in actual_col_names]
        if final_column_count <= 0:
            final_column_count = len(actual_col_names)
    feature_columns = declared_features
    feature_count = len(declared_features)

    horizons_sec = [int(h) for h in (prediction_targets.get("horizonsSec") or [])]
    from .feature_plugins import horizon_label

    target_labels = [horizon_label(h) for h in horizons_sec]
    if generated_cls_labels:
        # Keep short horizon labels for regression; append classification column names.
        target_labels = list(target_labels) + list(generated_cls_labels)
    dataset_configuration = build_dataset_configuration(
        sampling=sampling,
        horizons_sec=horizons_sec,
    )
    lb_method = dataset_configuration["lookback_policy"]["method"]
    pipeline_fingerprint = build_pipeline_fingerprint(
        sampling_interval_sec=int(interval_sec),
        atm_band=int(master_config.get("atm_band") or strike_selection.get("atmBand") or 10),
        feature_count=feature_count or len(implemented),
        target_horizons_sec=horizons_sec,
        lookback_policy=lb_method,
        registry=registry,
    )
    spec_hash = compute_spec_hash_from_fingerprint(pipeline_fingerprint, dataset_configuration)

    sources_for_spec = [
        {
            "source_id": d["source_id"],
            "trading_day": d["trading_day"],
            "market": d["market"],
            "expiry": d.get("expiry") or "",
            "date": d["trading_day"],
        }
        for d in days_meta
    ]
    expected_path = write_expected_spec(
        data_dir=data_dir,
        dataset_name=safe_name,
        sources=sources_for_spec,
        sampling=sampling,
        strike_selection=strike_selection,
        prediction_targets=prediction_targets,
        feature_selection=feature_selection,
        registry=registry,
    )

    filter_criteria = _filter_summary_dict(
        all_days=all_days,
        trading_day=td,
        selected_days=day_list,
        token=token_val,
        atm_band_filter=atm_filter,
        premium_min=prem_lo,
        premium_max=prem_hi,
        delta_min=delta_lo,
        delta_max=delta_hi,
        premium_enabled=premium_enabled,
        delta_enabled=delta_enabled,
        no_null_data=no_null_data,
    )
    selection_method = build_master_selection_method(
        market=market,
        interval_sec=interval_sec,
        all_days=all_days,
        trading_day=td,
        selected_days=day_list,
        token=token_val,
        atm_band_filter=atm_filter,
        premium_min=prem_lo,
        premium_max=prem_hi,
        delta_min=delta_lo,
        delta_max=delta_hi,
        premium_enabled=premium_enabled,
        delta_enabled=delta_enabled,
        no_null_data=no_null_data,
        trading_day_filter=filter_meta,
    )

    source_results = _source_results_from_days(days_meta, row_counts_by_day)
    mkt = str(market or master_config.get("market") or "NIFTY").upper()

    metadata: dict[str, Any] = {
        "dataset_name": safe_name,
        "created_at": created_dt.astimezone(timezone.utc).isoformat(),
        "version": METADATA_VERSION,
        "builder_version": BUILDER_VERSION,
        "export_source": "master_filter_export",
        "dataset_kind": str(dataset_kind or ("analysis" if keep_pipeline_owned else "registry")),
        "feature_project_id": bound_feature_project_id,
        "keep_pipeline_owned": bool(keep_pipeline_owned),
        "storage_backend": "parquet",
        "master_db_path": path_relative_to_data_dir(path, data_dir),
        "market": mkt,
        "days": days_meta,
        "trading_days": len(days_meta),
        "sources": source_results,
        "expected_spec": path_relative_to_data_dir(expected_path, data_dir),
        # Experiment identity: wall-clock meaning of row-span transform names
        # (e.g. ltp_roll_std_20 at 3s ≠ same name at 6s). Prefer this key when
        # comparing / loading experiments; sampling.interval_sec stays as well.
        "sample_interval_sec": int(interval_sec),
        "sampling": {
            "interval_sec": int(interval_sec),
            "method": str(sampling.get("samplingMethod") or "fixed_interval"),
        },
        "strike_selection": strike_selection_metadata(strike_selection),
        "prediction_targets": target_labels,
        "prediction_target_columns": target_columns,
        "classification_labels_5m": cls_label_meta,
        "feature_profile": str(feature_selection.get("profile") or "default"),
        "feature_groups": enabled_groups,
        "enabled_features": list(feature_selection.get("enabledFeatures") or implemented),
        "feature_groups_implemented": list(per_group.keys()),
        "feature_columns": list(feature_columns),
        "feature_columns_pending": pending,
        "feature_count": int(feature_count),
        "transformed_feature_columns": list(transformed_columns),
        "target_count": target_count or len(target_columns),
        "row_count": row_count,
        "column_count": final_column_count if final_column_count > 0 else (
            len(actual_col_names) if actual_col_names is not None else final_column_count
        ),
        # Create Model load can skip Python mergesort when this is true
        # (rows written chronologically: trading_day → timestamp → token).
        "is_sorted": True,
        "row_order": ["trading_day", "timestamp", "token"],
        "export_stats": {
            "rows_exported": row_count,
            "rows_per_day": row_counts_by_day,
            "rows_per_market": dict(export_stats.get("market_counts") or {}),
            "rows_per_expiry": dict(export_stats.get("expiry_counts") or {}),
            "elapsed_sec": export_stats.get("elapsed_sec"),
            "rows_per_sec": export_stats.get("rows_per_sec"),
        },
        "dataset_configuration": dataset_configuration,
        "lookback_policy": lb_method,
        "master_filter": filter_criteria,
        "selection_method": selection_method,
        **dict(transformation_meta),
        "no_null_dropped_columns": no_null_dropped_columns if no_null_data else None,
        "no_null_report": no_null_report if no_null_data else None,
        "pipeline_no_null_report": pipeline_no_null_report_text,
        "premium_filter_deferred": bool(defer_premium_until_end),
        "premium_report": premium_report if defer_premium_until_end else None,
        "audit_validation_required": bool(audit_validation_required),
        **build_version_metadata_fields(pipeline_fingerprint),
        "dataset_spec_hash": spec_hash,
        "output_parquet": path_relative_to_data_dir(parquet_path, data_dir),
        "output_json": path_relative_to_data_dir(json_path, data_dir),
    }
    if filter_meta is not None:
        from .trading_day_filter import enrich_trading_day_filter_dates

        exported_from_days = [
            str(d.get("trading_day") or "").strip()
            for d in days_meta
            if str(d.get("trading_day") or "").strip()
        ]
        filter_meta = enrich_trading_day_filter_dates(
            filter_meta,
            exported_dates=exported_from_days,
            master_day_rows=master_day_rows,
        )
        metadata["trading_day_filter"] = filter_meta
    if registry_export_features is not None:
        metadata["registry_export_features"] = sorted(
            {str(n).strip() for n in registry_export_features if str(n).strip()}
        )
    if base_pipeline_export_features is not None:
        metadata["base_pipeline_export_features"] = sorted(
            {str(n).strip() for n in base_pipeline_export_features if str(n).strip()}
        )
    if isinstance(pipeline_provenance, dict) and pipeline_provenance:
        metadata["pipeline_provenance"] = dict(pipeline_provenance)
        metadata["pipeline_id"] = str(pipeline_provenance.get("pipeline_id") or "")
        metadata["pipeline_name"] = str(pipeline_provenance.get("pipeline_name") or "")
        metadata["pipeline_type"] = str(pipeline_provenance.get("pipeline_type") or "")
        metadata["pipeline_snapshot_id"] = str(pipeline_provenance.get("pipeline_snapshot_id") or "")
        metadata["pipeline_feature_count"] = len(pipeline_provenance.get("candidate_features") or [])

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    try:
        from .dataset_selection_engine import DatasetSelectionEngine, DatasetSelectionSpec
        from .selection_preview_calibration import record_selection_calibration

        export_spec = DatasetSelectionSpec.from_registry_criteria(filter_criteria)
        export_spec.market = str(market or "NIFTY").upper()
        export_spec.interval_sec = int(interval_sec)
        preview_result = DatasetSelectionEngine(export_spec, path).preview()
        record_selection_calibration(
            data_dir,
            build_kind="registry_export",
            spec=export_spec,
            preview=preview_result,
            actual_rows=row_count,
            actual_days=len(days_meta),
            master_db_path=path,
            metadata_version=preview_result.metadata_version,
        )
    except Exception:
        pass

    return {
        "ok": True,
        "dataset_name": safe_name,
        "requested_name": base_name,
        "row_count": row_count,
        "feature_count": feature_count,
        "target_count": target_count,
        "trading_days": [d["trading_day"] for d in days_meta],
        "filter": filter_criteria,
        "selection_method": selection_method,
        "audit_validation_required": bool(audit_validation_required),
        "parquet_path": parquet_path,
        "json_path": json_path,
        "expected_path": expected_path,
        "registry_url": "",
        "export_stats": metadata.get("export_stats"),
        "pipeline_no_null_report": pipeline_no_null_report_text,
    }
