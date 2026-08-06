"""Safe feature migration for master SQLite datasets — temp table, validate, merge."""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

from chain_replay_ml.export_atm_pipeline import STRIKE_STEP, normalize_index_name
from chain_replay_ml.features_atm_band import find_atm_strike

from .chain_maps import precompute_chain_maps
from .day_context import DayContext, SourceSpec, load_day_context
from .extended_features import OptionFeatureState
from .feature_enrichment import SCORING_INFRA_COLUMNS
from .expected_spec import DATASET_METADATA_COLUMNS
from .feature_grid_policy import resolve_feature_grid_step_sec
from .feature_plugins import GROUP_FEATURE_SOURCES
from .lookback_policy import lookback_policy, read_dataset_configuration
from .master_fingerprint import build_identity_from_build
from .master_store import MasterStore, _SAFE_COL, _sql_type
from .orchestrator import _load_feature_registry
from .pipeline_identity import feature_registry_version_label, schema_registry_hash
from .registry_features import build_registry_features_at_ts
from .rolling_controllers import SpotControllers
from .schema_registry import load_schema_registry

_CHART_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIGRATION_TEMP_TABLE = "master_feature_migration_temp"
MIGRATION_META_KEY = "feature_migration"

MIGRATION_STEP_DEFS: tuple[dict[str, str], ...] = (
    {"id": "load_registry", "label": "Load Registry Features"},
    {"id": "compare_schema", "label": "Compare Database Schema"},
    {"id": "backup", "label": "Backup Database"},
    {"id": "add_columns", "label": "Add Missing Columns"},
    {"id": "init_defaults", "label": "Initialize Default Values"},
    {"id": "populate", "label": "Populate Feature Values"},
)

_migration_lock = threading.Lock()
_migration_active = False


class FeatureMigrationError(Exception):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_sample_id(trading_day: str, timestamp: float, token: str) -> str:
    return f"{trading_day}|{float(timestamp):.6f}|{token}"


def _implemented_registry_features() -> dict[str, str]:
    out: dict[str, str] = {}
    for gid, mapping in GROUP_FEATURE_SOURCES.items():
        for feat, src in mapping.items():
            if src is not None:
                out[feat] = gid
    return out


def _groups_for_features(features: list[str], registry: dict[str, Any]) -> list[str]:
    impl = _implemented_registry_features()
    groups_meta = registry.get("groups") or {}
    group_order = list(registry.get("groupOrder") or groups_meta.keys())
    feat_to_gid: dict[str, str] = {}
    for gid in group_order:
        for feat in (groups_meta.get(gid) or {}).get("features") or []:
            feat_to_gid[feat] = gid
    gids: list[str] = []
    for feat in features:
        gid = feat_to_gid.get(feat) or impl.get(feat)
        if gid and gid not in gids:
            gids.append(gid)
    return gids


def _all_sample_columns(conn: sqlite3.Connection) -> set[str]:
    return {row[1] for row in conn.execute("PRAGMA table_info(samples)").fetchall()}


def _infra_sample_columns() -> set[str]:
    """Non-feature columns that must not appear as migration / missing targets."""
    return set(DATASET_METADATA_COLUMNS) | set(SCORING_INFRA_COLUMNS) | {
        "trading_day",
        "timestamp",
        "token",
        "symbol",
        "market",
        "expiry",
        "option_type",
        "strike",
        "ltp",
        "delta",
        "abs_delta",
    }


def _sample_feature_columns(conn: sqlite3.Connection) -> set[str]:
    infra = _infra_sample_columns()
    cols = _all_sample_columns(conn)
    return {c for c in cols if c not in infra and not c.startswith("future_ltp_")}


def _columns_already_in_master(conn: sqlite3.Connection) -> set[str]:
    """Registry feature names that already have a physical column in samples."""
    return _all_sample_columns(conn)


def _feature_has_any_value(conn: sqlite3.Connection, feature: str) -> bool:
    """True if at least one non-NULL value exists (cheap LIMIT 1 probe)."""
    if not _SAFE_COL.match(feature):
        return False
    row = conn.execute(
        f'SELECT 1 FROM samples WHERE "{feature}" IS NOT NULL LIMIT 1',
    ).fetchone()
    return bool(row)


def analyze_master_feature_migration(store: MasterStore) -> dict[str, Any]:
    """Compare master feature columns vs current registry.

    A feature is listed as missing when it is implemented in the registry and either:
      - has no physical column in ``samples``, or
      - is absent from ``build_schema.feature_columns`` (schema/meta lag).

    Driven by Feature Registry + ``build_schema`` meta so the UI does not require
    ad-hoc master SQL. Physical presence only classifies the gap.
    """
    registry = load_schema_registry()
    impl = _implemented_registry_features()
    build_schema = store.get_meta("build_schema") or {}
    stored_features = list(build_schema.get("feature_columns") or [])
    stored_set = set(stored_features)

    conn = store.conn
    present_in_db = _sample_feature_columns(conn)
    existing_columns = _columns_already_in_master(conn)
    infra = _infra_sample_columns()
    current_count = len(stored_features) if stored_features else len(present_in_db)

    registry_features = sorted(
        f for f in impl.keys()
        if f not in infra and not str(f).startswith("future_ltp_")
    )
    registry_count = len(registry_features)

    missing: list[dict[str, Any]] = []
    already_in_db: list[str] = []
    groups_meta = registry.get("groups") or {}
    for feat in registry_features:
        in_db = feat in existing_columns
        in_schema = feat in stored_set
        if in_db and in_schema:
            continue
        if not in_db:
            reason = "missing_column"
            detail = "Not in master samples schema (needs ADD COLUMN + backfill)"
        else:
            # Physical column exists but build_schema meta is behind.
            already_in_db.append(feat)
            reason = "not_in_schema"
            detail = "In DB columns but missing from build_schema.feature_columns"

        gid = impl[feat]
        missing.append({
            "name": feat,
            "group_id": gid,
            "group": (groups_meta.get(gid) or {}).get("label") or gid,
            "in_build_schema": in_schema,
            "in_db": in_db,
            "reason": reason,
            "detail": detail,
        })

    meta = store.read_master_meta_dict()
    progress = store.get_meta(MIGRATION_META_KEY) or {}
    temp_exists = bool(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (MIGRATION_TEMP_TABLE,),
        ).fetchone()
    )

    by_reason: dict[str, int] = {}
    for row in missing:
        key = str(row.get("reason") or "missing")
        by_reason[key] = by_reason.get(key, 0) + 1

    return {
        "current_feature_count": current_count,
        "registry_feature_count": registry_count,
        "missing_count": len(missing),
        "available_count": len(missing),
        "missing_features": missing,
        "missing_by_reason": by_reason,
        "already_in_db_not_in_schema": sorted(already_in_db),
        "stored_features": stored_features,
        "present_in_db_count": len(present_in_db),
        "feature_registry_version": meta.get("feature_registry_version")
        or feature_registry_version_label(registry=registry, feature_count=registry_count),
        "schema_hash": meta.get("schema_hash") or schema_registry_hash(),
        "metadata_version": int(meta.get("metadata_version") or 0),
        "migration_status": progress.get("status") if isinstance(progress, dict) else "idle",
        "temp_table_exists": temp_exists,
        "migration_progress": progress if isinstance(progress, dict) else None,
        "trading_days": [d["trading_day"] for d in store.read_master_days()],
    }


def _validate_feature_names(features: list[str]) -> list[str]:
    impl = _implemented_registry_features()
    out: list[str] = []
    for feat in features:
        name = str(feat).strip()
        if not name:
            continue
        if not _SAFE_COL.match(name):
            raise FeatureMigrationError(f"Unsafe feature name: {name}")
        if name not in impl:
            raise FeatureMigrationError(f"Feature not implemented in registry: {name}")
        if name not in out:
            out.append(name)
    if not out:
        raise FeatureMigrationError("No features selected for migration")
    return out


def _drop_temp_table(conn: sqlite3.Connection) -> None:
    conn.execute(f'DROP TABLE IF EXISTS "{MIGRATION_TEMP_TABLE}"')


def _create_temp_table(conn: sqlite3.Connection, features: list[str]) -> None:
    _drop_temp_table(conn)
    col_defs = ", ".join(f'"{c}" REAL' for c in features)
    conn.execute(
        f"""
        CREATE TABLE "{MIGRATION_TEMP_TABLE}" (
            sample_id TEXT PRIMARY KEY,
            trading_day TEXT NOT NULL,
            timestamp REAL NOT NULL,
            token TEXT NOT NULL,
            {col_defs}
        )
        """
    )
    conn.execute(
        f'CREATE INDEX IF NOT EXISTS idx_mig_temp_day ON "{MIGRATION_TEMP_TABLE}" (trading_day)'
    )


def _backup_master_db(db_path: str) -> dict[str, Any]:
    """Copy master DB to *.bak beside the live file (overwrite previous bak)."""
    src = os.path.abspath(db_path)
    if not os.path.isfile(src):
        raise FeatureMigrationError(f"Master DB not found for backup: {src}")
    bak = f"{src}.bak"
    t0 = time.perf_counter()
    shutil.copy2(src, bak)
    size = os.path.getsize(bak)
    return {
        "path": bak,
        "filename": os.path.basename(bak),
        "bytes": size,
        "size_gb": round(size / (1024 ** 3), 2),
        "elapsed_sec": round(time.perf_counter() - t0, 2),
    }


def _timeline_template() -> list[dict[str, Any]]:
    return [
        {
            "id": step["id"],
            "label": step["label"],
            "status": "pending",
            "detail": None,
            "started_at": None,
            "finished_at": None,
            "elapsed_sec": None,
        }
        for step in MIGRATION_STEP_DEFS
    ]


def _mark_timeline_step(
    timeline: list[dict[str, Any]],
    step_id: str,
    *,
    status: str,
    detail: str | None = None,
    started_at: str | None = None,
    finished_at: str | None = None,
    elapsed_sec: float | None = None,
) -> None:
    for step in timeline:
        if step.get("id") != step_id:
            continue
        step["status"] = status
        if detail is not None:
            step["detail"] = detail
        if started_at is not None:
            step["started_at"] = started_at
        if finished_at is not None:
            step["finished_at"] = finished_at
        if elapsed_sec is not None:
            step["elapsed_sec"] = elapsed_sec
        break


def _feature_queue(features: list[str], current: str | None, completed: list[str] | None = None) -> list[dict[str, Any]]:
    done = set(completed or [])
    cur = str(current or "").strip() or None
    out: list[dict[str, Any]] = []
    for name in features:
        if name in done:
            st = "done"
        elif cur and name == cur:
            st = "running"
        else:
            st = "pending"
        out.append({"name": name, "status": st})
    return out


def _persist_live_progress(store: MasterStore, patch: dict[str, Any]) -> dict[str, Any]:
    prog = dict(store.get_meta(MIGRATION_META_KEY) or {})
    prog.update(patch)
    store.set_meta(MIGRATION_META_KEY, prog)
    return prog


def _day_source_from_samples(conn: sqlite3.Connection, trading_day: str, default_market: str) -> SourceSpec | None:
    row = conn.execute(
        """
        SELECT market, expiry, COUNT(*) AS n
        FROM samples
        WHERE trading_day = ?
        GROUP BY market, expiry
        ORDER BY n DESC
        LIMIT 1
        """,
        (trading_day,),
    ).fetchone()
    market = str(row[0] if row and row[0] else default_market or "NIFTY").upper()
    expiry = str(row[1] if row and row[1] else "").strip()
    if not expiry:
        exp_row = conn.execute(
            "SELECT expiry FROM samples WHERE trading_day = ? AND expiry IS NOT NULL LIMIT 1",
            (trading_day,),
        ).fetchone()
        expiry = str(exp_row[0]).strip() if exp_row and exp_row[0] else ""
    if not expiry:
        return None
    return SourceSpec(
        source_id=f"{trading_day}|{market}|{expiry}",
        trading_day=trading_day,
        market=market,
        expiry=expiry,
    )


def _read_day_samples(conn: sqlite3.Connection, trading_day: str) -> list[dict[str, Any]]:
    cols = [row[1] for row in conn.execute("PRAGMA table_info(samples)").fetchall()]
    col_sql = ", ".join(f'"{c}"' for c in cols)
    rows = conn.execute(
        f'SELECT {col_sql} FROM samples WHERE trading_day = ? ORDER BY token, timestamp',
        (trading_day,),
    ).fetchall()
    return [dict(zip(cols, r)) for r in rows]


def _insert_temp_batch(
    conn: sqlite3.Connection,
    features: list[str],
    batch: list[dict[str, Any]],
) -> None:
    if not batch:
        return
    cols = ["sample_id", "trading_day", "timestamp", "token", *features]
    placeholders = ", ".join("?" for _ in cols)
    col_sql = ", ".join(f'"{c}"' for c in cols)
    sql = f'INSERT OR REPLACE INTO "{MIGRATION_TEMP_TABLE}" ({col_sql}) VALUES ({placeholders})'
    params = []
    for row in batch:
        params.append([
            row["sample_id"],
            row["trading_day"],
            float(row["timestamp"]),
            row["token"],
            *[row.get(f) for f in features],
        ])
    conn.executemany(sql, params)


def compute_migration_day(
    store: MasterStore,
    *,
    trading_day: str,
    features: list[str],
    enabled_groups: list[str],
    lookback_policy_doc: dict[str, Any] | None,
    grid_step: float,
    default_market: str,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Recompute selected features for one trading day into the temp table."""
    day = str(trading_day).strip()
    conn = store.conn
    sample_rows = _read_day_samples(conn, day)
    if not sample_rows:
        return {"trading_day": day, "rows": 0, "skipped": True}

    source = _day_source_from_samples(conn, day, default_market)
    if not source:
        raise FeatureMigrationError(f"Cannot resolve market/expiry for {day}")

    chart_dir = _CHART_DIR
    ctx = load_day_context(chart_dir, source, feature_grid_step_sec=grid_step)
    index_key = normalize_index_name(ctx.source.market)
    strike_step = STRIKE_STEP.get(index_key, 50)
    all_ts = sorted({float(r["timestamp"]) for r in sample_rows})
    chain_maps = precompute_chain_maps(
        index_tl=ctx.index_tl,
        strike_mapping=ctx.strike_mapping,
        timestamps=all_ts,
        strike_step=strike_step,
    )

    feat_active = frozenset(features)
    batch: list[dict[str, Any]] = []
    rows_done = 0
    opt_states: dict[str, OptionFeatureState] = {}
    spot_controllers = SpotControllers()

    for row in sample_rows:
        token = str(row["token"])
        strike_r = float(row.get("strike") or 0)
        opt_type = str(row.get("option_type") or "")
        ts = float(row["timestamp"])
        entry = ctx.strike_mapping.get((strike_r, opt_type))
        out_row: dict[str, Any] = {
            "sample_id": make_sample_id(day, ts, token),
            "trading_day": day,
            "timestamp": ts,
            "token": token,
        }
        if not entry:
            for f in features:
                out_row[f] = None
            batch.append(out_row)
            rows_done += 1
            continue

        _tok, _sym, opt_tl = entry
        spot = ctx.index_tl.ltp_rupees_at(ts)
        if spot is None or spot <= 0:
            for f in features:
                out_row[f] = None
            batch.append(out_row)
            rows_done += 1
            continue

        state_key = f"{day}:{token}"
        if state_key not in opt_states:
            opt_states[state_key] = OptionFeatureState()
        atm = find_atm_strike(spot, strike_step)
        picked = build_registry_features_at_ts(
            ts=ts,
            strike=strike_r,
            option_type=opt_type,
            opt_tl=opt_tl,
            index_tl=ctx.index_tl,
            strike_mapping=ctx.strike_mapping,
            chain_maps=chain_maps,
            opt_state=opt_states[state_key],
            strike_step=strike_step,
            expiry_ts=float(ctx.expiry_ts),
            open_ts=float(ctx.open_ts),
            close_ts=float(ctx.close_ts),
            enabled_groups=enabled_groups,
            trading_day=day,
            expiry_norm=str(ctx.expiry_norm),
            lookback_policy_doc=lookback_policy_doc,
            atm_strike=atm,
            active_features=feat_active,
            feature_grid_step_sec=grid_step,
            spot_controllers=spot_controllers,
        )
        for f in features:
            out_row[f] = picked.get(f)
        batch.append(out_row)
        rows_done += 1
        if len(batch) >= 500:
            _insert_temp_batch(conn, features, batch)
            batch.clear()
            conn.commit()
            if on_progress:
                on_progress({"trading_day": day, "rows_done": rows_done, "rows_total": len(sample_rows)})

    if batch:
        _insert_temp_batch(conn, features, batch)
    conn.commit()

    return {"trading_day": day, "rows": rows_done, "skipped": False}


def validate_migration(store: MasterStore, features: list[str]) -> dict[str, Any]:
    """Validate temp table against master samples before merge."""
    conn = store.conn
    if not conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (MIGRATION_TEMP_TABLE,),
    ).fetchone():
        raise FeatureMigrationError("Migration temp table does not exist")

    master_count = int(conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
    temp_count = int(conn.execute(f'SELECT COUNT(*) FROM "{MIGRATION_TEMP_TABLE}"').fetchone()[0])

    master_ids = {
        make_sample_id(str(r[0]), float(r[1]), str(r[2]))
        for r in conn.execute("SELECT trading_day, timestamp, token FROM samples").fetchall()
    }
    temp_ids = {
        str(r[0])
        for r in conn.execute(f'SELECT sample_id FROM "{MIGRATION_TEMP_TABLE}"').fetchall()
    }
    missing_ids = sorted(master_ids - temp_ids)
    extra_ids = sorted(temp_ids - master_ids)

    dup_rows = conn.execute(
        f"""
        SELECT sample_id, COUNT(*) AS n FROM "{MIGRATION_TEMP_TABLE}"
        GROUP BY sample_id HAVING n > 1
        """
    ).fetchall()
    duplicate_ids = [{"sample_id": str(r[0]), "count": int(r[1])} for r in dup_rows]

    feature_stats: dict[str, Any] = {}
    warnings: list[str] = []
    for feat in features:
        row = conn.execute(
            f"""
            SELECT
                COUNT(*) AS n,
                SUM(CASE WHEN "{feat}" IS NULL THEN 1 ELSE 0 END) AS nulls,
                MIN("{feat}") AS vmin,
                MAX("{feat}") AS vmax,
                AVG("{feat}") AS vmean
            FROM "{MIGRATION_TEMP_TABLE}"
            """,
        ).fetchone()
        n = int(row[0] or 0)
        nulls = int(row[1] or 0)
        null_pct = round(nulls / n * 100.0, 2) if n > 0 else 0.0
        std_row = conn.execute(
            f"""
            SELECT SQRT(AVG((t."{feat}" - agg.mean) * (t."{feat}" - agg.mean)))
            FROM "{MIGRATION_TEMP_TABLE}" t
            CROSS JOIN (SELECT AVG("{feat}") AS mean FROM "{MIGRATION_TEMP_TABLE}") agg
            WHERE t."{feat}" IS NOT NULL
            """
        ).fetchone()
        std_val = round(float(std_row[0]), 6) if std_row and std_row[0] is not None else None
        feature_stats[feat] = {
            "null_count": nulls,
            "null_pct": null_pct,
            "min": row[2],
            "max": row[3],
            "mean": round(float(row[4]), 6) if row[4] is not None else None,
            "std": std_val,
        }
        if null_pct > 0:
            warnings.append(f"{feat}: {null_pct}% NULL")

    passed = (
        master_count == temp_count
        and len(missing_ids) == 0
        and len(extra_ids) == 0
        and len(duplicate_ids) == 0
    )

    return {
        "passed": passed,
        "accuracy": "exact" if passed else "failed",
        "master_row_count": master_count,
        "temp_row_count": temp_count,
        "row_count_match": master_count == temp_count,
        "missing_ids_count": len(missing_ids),
        "missing_ids_sample": missing_ids[:10],
        "extra_ids_count": len(extra_ids),
        "extra_ids_sample": extra_ids[:10],
        "duplicate_ids_count": len(duplicate_ids),
        "duplicate_ids": duplicate_ids[:10],
        "feature_stats": feature_stats,
        "warnings": warnings,
        "validated_at": _utc_now(),
    }


def commit_migration(store: MasterStore, *, validation: dict[str, Any]) -> dict[str, Any]:
    """Merge validated temp columns into samples — single transaction."""
    if not validation.get("passed"):
        raise FeatureMigrationError("Validation did not pass — master samples not modified")

    progress = store.get_meta(MIGRATION_META_KEY) or {}
    features = list(progress.get("features") or [])
    if not features:
        raise FeatureMigrationError("No features recorded in migration progress")

    conn = store.conn
    conn.execute("BEGIN IMMEDIATE")
    try:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(samples)").fetchall()}
        for col in features:
            if col in existing:
                continue
            if not _SAFE_COL.match(col):
                raise FeatureMigrationError(f"Unsafe column: {col}")
            conn.execute(f'ALTER TABLE samples ADD COLUMN "{col}" {_sql_type(col)}')

        for col in features:
            conn.execute(
                f"""
                UPDATE samples SET "{col}" = (
                    SELECT t."{col}" FROM "{MIGRATION_TEMP_TABLE}" t
                    WHERE t.trading_day = samples.trading_day
                      AND t.timestamp = samples.timestamp
                      AND t.token = samples.token
                )
                """
            )

        conn.execute(f'DROP TABLE IF EXISTS "{MIGRATION_TEMP_TABLE}"')
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    build_schema = store.get_meta("build_schema") or {}
    stored = list(build_schema.get("feature_columns") or [])
    updated_features = list(dict.fromkeys([*stored, *features]))
    target_columns = list(build_schema.get("target_columns") or [])

    build_schema.update({
        "feature_columns": updated_features,
        "feature_count": len(updated_features),
    })
    store.set_meta("build_schema", build_schema)

    master_config = store.get_meta("master_config") or {}
    if isinstance(master_config, dict):
        master_config["feature_count"] = len(updated_features)
        store.set_meta("master_config", master_config)

    registry = _load_feature_registry()
    meta = store.read_master_meta_dict()
    market = str(meta.get("market") or master_config.get("market") or "NIFTY").upper()
    interval_sec = int(meta.get("sampling_interval_sec") or master_config.get("sampling_interval_sec") or 10)
    atm_band = int(master_config.get("atm_band") or 10)
    horizons_sec = []
    for col in target_columns:
        m = re.match(r"future_ltp_(\d+)([smh])", col)
        if m:
            val, unit = int(m.group(1)), m.group(2)
            sec = val * 60 if unit == "m" else val * 3600 if unit == "h" else val
            horizons_sec.append(sec)
    if not horizons_sec:
        horizons_sec = [5, 10, 30, 60, 300]

    lb_doc = store.get_meta("dataset_configuration") or {}
    lb_method = str((lb_doc.get("lookback_policy") or {}).get("method") or "calendar")

    identity = build_identity_from_build(
        market=market,
        sampling_interval_sec=interval_sec,
        atm_band=atm_band,
        feature_count=len(updated_features),
        target_horizons_sec=horizons_sec,
        lookback_policy=lb_method,
        registry=registry,
        target_columns=target_columns,
        created_from="feature_migration",
    )
    store.update_build_identity(identity)
    store.sync_schema_meta_fields()

    started_at = progress.get("started_at")
    committed_at = _utc_now()
    total_elapsed_sec = None
    try:
        if started_at:
            total_elapsed_sec = round(
                (
                    datetime.fromisoformat(str(committed_at))
                    - datetime.fromisoformat(str(started_at))
                ).total_seconds(),
                1,
            )
    except (TypeError, ValueError):
        total_elapsed_sec = None

    timeline = list(progress.get("timeline") or _timeline_template())
    for step_id in ("load_registry", "compare_schema", "backup", "add_columns", "init_defaults", "populate"):
        _mark_timeline_step(timeline, step_id, status="done")
    _mark_timeline_step(
        timeline,
        "populate",
        status="done",
        detail=f"Committed {len(features)} feature(s)",
        finished_at=committed_at,
        elapsed_sec=total_elapsed_sec,
    )

    finished = {
        "status": "completed",
        "job_id": progress.get("job_id"),
        "features": features,
        "started_at": started_at,
        "committed_at": committed_at,
        "total_elapsed_sec": total_elapsed_sec,
        "feature_count": len(updated_features),
        "validation": validation,
        "timeline": timeline,
        "backup": progress.get("backup"),
        "schema_compare": progress.get("schema_compare"),
        "populate": progress.get("populate"),
        "completed_days": progress.get("completed_days") or [],
        "pending_days": [],
        "current_step": "completed",
    }
    store.set_meta(MIGRATION_META_KEY, finished)

    return {
        "ok": True,
        "features_merged": features,
        "feature_count": len(updated_features),
        "metadata_version": store.read_master_meta().metadata_version,
        "feature_registry_version": identity.get("feature_registry_version"),
        "schema_hash": identity.get("schema_hash"),
        "started_at": started_at,
        "committed_at": committed_at,
        "total_elapsed_sec": total_elapsed_sec,
        "progress": finished,
    }


def rollback_migration(store: MasterStore) -> dict[str, Any]:
    """Drop temp table and reset migration state — master samples untouched."""
    conn = store.conn
    conn.execute(f'DROP TABLE IF EXISTS "{MIGRATION_TEMP_TABLE}"')
    conn.commit()
    store.set_meta(MIGRATION_META_KEY, {"status": "rolled_back", "rolled_back_at": _utc_now()})
    return {"ok": True, "status": "rolled_back"}


def _can_resume_migration(store: MasterStore, progress: dict[str, Any], features: list[str]) -> bool:
    """True when temp table + same feature set + remaining days exist."""
    if not isinstance(progress, dict):
        return False
    status = str(progress.get("status") or "")
    if status not in ("computing", "preparing"):
        return False
    saved = list(progress.get("features") or [])
    if sorted(saved) != sorted(features):
        return False
    pending = list(progress.get("pending_days") or [])
    if not pending:
        return False
    conn = store.conn
    return bool(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (MIGRATION_TEMP_TABLE,),
        ).fetchone()
    )


def start_feature_migration(
    store: MasterStore,
    features: list[str],
    *,
    resume: bool = False,
) -> dict[str, Any]:
    """Create temp table, backup DB, and initialize migration progress timeline."""
    features = _validate_feature_names(features)
    conn = store.conn
    progress = store.get_meta(MIGRATION_META_KEY) or {}

    # Resume must run before the "already in samples" guard — interrupted jobs can leave
    # columns in samples (e.g. after a partial commit) while temp + pending days remain.
    if resume and _can_resume_migration(store, progress, features):
        # Recalculate remaining rows estimate for the live drawer.
        pending = list(progress.get("pending_days") or [])
        completed = list(progress.get("completed_days") or [])
        populate = dict(progress.get("populate") or {})
        total_rows = int(populate.get("rows_total") or 0)
        if not total_rows:
            total_rows = int(conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] or 0)
            populate["rows_total"] = total_rows
        timeline = list(progress.get("timeline") or _timeline_template())
        for step_id in ("load_registry", "compare_schema", "backup", "add_columns", "init_defaults"):
            _mark_timeline_step(timeline, step_id, status="done")
        _mark_timeline_step(
            timeline,
            "populate",
            status="running",
            detail=(
                f"Resumed · {len(completed)} day(s) done · "
                f"{len(pending)} remaining"
            ),
        )
        progress = dict(progress)
        progress.update({
            "status": "computing",
            "resumed_at": _utc_now(),
            "current_step": "populate",
            "timeline": timeline,
            "populate": populate,
        })
        store.set_meta(MIGRATION_META_KEY, progress)
        return {
            "ok": True,
            "resumed": True,
            "job_id": progress.get("job_id"),
            "features": features,
            "pending_days": pending,
            "completed_days": completed,
            "progress": progress,
            "backup": progress.get("backup"),
        }

    if resume:
        raise FeatureMigrationError(
            "Cannot resume migration — no matching in-progress job "
            "(temp table missing, features changed, or no pending days). "
            "Use Rollback, then Start again."
        )

    existing_columns = _columns_already_in_master(conn)
    build_schema = store.get_meta("build_schema") or {}
    stored_set = set(build_schema.get("feature_columns") or [])
    # Skip features already fully registered; keep only real migration targets.
    fully_present = [
        f for f in features
        if f in existing_columns and f in stored_set and _feature_has_any_value(conn, f)
    ]
    features = [f for f in features if f not in set(fully_present)]
    if not features:
        raise FeatureMigrationError(
            "Nothing to migrate — selected features are already in master schema with data: "
            + ", ".join(fully_present[:8])
            + ("…" if len(fully_present) > 8 else "")
        )

    days = [d["trading_day"] for d in store.read_master_days()]
    if not days:
        raise FeatureMigrationError("No trading days in master dataset")

    registry = load_schema_registry()
    impl = _implemented_registry_features()
    analysis_preview = analyze_master_feature_migration(store)
    total_rows = int(conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] or 0)
    started_at = _utc_now()
    timeline = _timeline_template()
    job_id = uuid.uuid4().hex

    # Step 1 — Load Registry Features
    t0 = time.perf_counter()
    _mark_timeline_step(
        timeline,
        "load_registry",
        status="running",
        started_at=started_at,
        detail=f"{len(impl)} features",
    )
    store.set_meta(
        MIGRATION_META_KEY,
        {
            "status": "preparing",
            "job_id": job_id,
            "features": features,
            "started_at": started_at,
            "current_step": "load_registry",
            "timeline": timeline,
            "completed_days": [],
            "pending_days": list(days),
            "validation": None,
            "populate": {
                "current_feature": None,
                "trading_day": None,
                "rows_done": 0,
                "rows_total": total_rows,
                "day_rows_done": 0,
                "day_rows_total": 0,
                "features_completed": [],
                "feature_queue": _feature_queue(features, None),
                "elapsed_sec": 0,
                "eta_sec": None,
                "pct": 0,
            },
        },
    )
    _mark_timeline_step(
        timeline,
        "load_registry",
        status="done",
        detail=f"{len(impl)} features",
        finished_at=_utc_now(),
        elapsed_sec=round(time.perf_counter() - t0, 2),
    )

    # Step 2 — Compare Database Schema
    t1 = time.perf_counter()
    _mark_timeline_step(timeline, "compare_schema", status="running", started_at=_utc_now())
    store.set_meta(
        MIGRATION_META_KEY,
        {
            **(store.get_meta(MIGRATION_META_KEY) or {}),
            "current_step": "compare_schema",
            "timeline": timeline,
        },
    )
    compare_detail = (
        f"Current : {analysis_preview.get('current_feature_count')} · "
        f"Registry : {analysis_preview.get('registry_feature_count')} · "
        f"Missing : {len(features)}"
    )
    _mark_timeline_step(
        timeline,
        "compare_schema",
        status="done",
        detail=compare_detail,
        finished_at=_utc_now(),
        elapsed_sec=round(time.perf_counter() - t1, 2),
    )

    # Step 3 — Backup Database
    t2 = time.perf_counter()
    _mark_timeline_step(timeline, "backup", status="running", started_at=_utc_now())
    store.set_meta(
        MIGRATION_META_KEY,
        {
            **(store.get_meta(MIGRATION_META_KEY) or {}),
            "current_step": "backup",
            "timeline": timeline,
        },
    )
    backup = _backup_master_db(store.db_path)
    backup_detail = f"{backup['filename']} · {backup['size_gb']} GB"
    _mark_timeline_step(
        timeline,
        "backup",
        status="done",
        detail=backup_detail,
        finished_at=_utc_now(),
        elapsed_sec=round(time.perf_counter() - t2, 2),
    )

    # Step 4 — Add Missing Columns (temp staging table)
    t3 = time.perf_counter()
    _mark_timeline_step(timeline, "add_columns", status="running", started_at=_utc_now())
    store.set_meta(
        MIGRATION_META_KEY,
        {
            **(store.get_meta(MIGRATION_META_KEY) or {}),
            "current_step": "add_columns",
            "timeline": timeline,
            "backup": backup,
        },
    )
    _create_temp_table(conn, features)
    conn.commit()
    _mark_timeline_step(
        timeline,
        "add_columns",
        status="done",
        detail=f"{len(features)} / {len(features)} completed",
        finished_at=_utc_now(),
        elapsed_sec=round(time.perf_counter() - t3, 2),
    )

    # Step 5 — Initialize Default Values
    t4 = time.perf_counter()
    _mark_timeline_step(timeline, "init_defaults", status="running", started_at=_utc_now())
    store.set_meta(
        MIGRATION_META_KEY,
        {
            **(store.get_meta(MIGRATION_META_KEY) or {}),
            "current_step": "init_defaults",
            "timeline": timeline,
        },
    )
    _mark_timeline_step(
        timeline,
        "init_defaults",
        status="done",
        detail=f"{len(features)} columns",
        finished_at=_utc_now(),
        elapsed_sec=round(time.perf_counter() - t4, 2),
    )

    # Step 6 — ready to populate
    _mark_timeline_step(timeline, "populate", status="running", started_at=_utc_now())
    prog = {
        "status": "computing",
        "job_id": job_id,
        "features": features,
        "registry_feature_count": len(impl),
        "started_at": started_at,
        "completed_days": [],
        "pending_days": list(days),
        "validation": None,
        "current_step": "populate",
        "timeline": timeline,
        "backup": backup,
        "schema_compare": {
            "current": analysis_preview.get("current_feature_count"),
            "registry": analysis_preview.get("registry_feature_count"),
            "missing": len(features),
        },
        "populate": {
            "current_feature": features[0] if features else None,
            "trading_day": days[0] if days else None,
            "rows_done": 0,
            "rows_total": total_rows,
            "day_rows_done": 0,
            "day_rows_total": 0,
            "features_completed": [],
            "feature_queue": _feature_queue(features, features[0] if features else None),
            "elapsed_sec": 0,
            "eta_sec": None,
            "pct": 0,
        },
        "feature_registry_version": feature_registry_version_label(
            registry=registry, feature_count=len(impl),
        ),
    }
    store.set_meta(MIGRATION_META_KEY, prog)
    return {
        "ok": True,
        "job_id": job_id,
        "features": features,
        "pending_days": days,
        "progress": prog,
        "backup": backup,
    }


def run_feature_migration_compute(
    store: MasterStore,
    *,
    trading_day: str | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Compute one day or all pending days into temp table."""
    progress = store.get_meta(MIGRATION_META_KEY) or {}
    if progress.get("status") != "computing":
        raise FeatureMigrationError("No active migration — call start first")

    features = list(progress.get("features") or [])
    pending = list(progress.get("pending_days") or [])
    completed = list(progress.get("completed_days") or [])
    timeline = list(progress.get("timeline") or _timeline_template())
    populate = dict(progress.get("populate") or {})
    started_mono = time.perf_counter()
    started_at = progress.get("started_at")
    try:
        started_ts = datetime.fromisoformat(str(started_at)).timestamp() if started_at else time.time()
    except ValueError:
        started_ts = time.time()

    if trading_day:
        days_to_run = [str(trading_day).strip()] if str(trading_day).strip() in pending else []
        if not days_to_run and str(trading_day).strip() not in completed:
            raise FeatureMigrationError(f"Day {trading_day} is not pending")
    else:
        days_to_run = list(pending)

    if not days_to_run:
        return {"ok": True, "message": "No pending days", "progress": progress}

    registry = _load_feature_registry()
    enabled_groups = _groups_for_features(features, registry)
    lb_doc = lookback_policy(read_dataset_configuration({"dataset_configuration": store.get_meta("dataset_configuration")}, None))
    grid_step = resolve_feature_grid_step_sec(
        dataset_configuration=store.get_meta("dataset_configuration") or {},
    )
    meta = store.read_master_meta_dict()
    default_market = str(meta.get("market") or "NIFTY")
    total_rows = int(populate.get("rows_total") or 0)
    if not total_rows:
        total_rows = int(store.conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0] or 0)

    last_persist_at = [0.0]

    def _update_live(patch: dict[str, Any], *, force_persist: bool = False) -> None:
        nonlocal progress, populate, timeline
        populate.update(patch)
        elapsed = max(0.0, time.time() - started_ts)
        rows_done = int(populate.get("rows_done") or 0)
        rows_pct = round(100.0 * rows_done / total_rows, 1) if total_rows > 0 else 0.0
        days_done = len(completed)
        days_total = days_done + len(pending)
        day_done = int(populate.get("day_rows_done") or 0)
        day_total = int(populate.get("day_rows_total") or 0)
        day_pct = round(100.0 * day_done / day_total, 1) if day_total > 0 else 0.0
        day_frac = (float(day_done) / float(day_total)) if day_total > 0 else 0.0
        # Overall % by trading days (current day contributes its row fraction).
        if days_total > 0:
            overall_pct = round(100.0 * (days_done + day_frac) / days_total, 1)
        else:
            overall_pct = rows_pct
        eta = None
        if rows_done > 0 and total_rows > rows_done and elapsed > 0:
            rate = rows_done / elapsed
            eta = round((total_rows - rows_done) / rate, 1) if rate > 0 else None
        populate["elapsed_sec"] = round(elapsed, 1)
        populate["eta_sec"] = eta
        populate["rows_pct"] = rows_pct
        populate["day_pct"] = day_pct
        populate["days_done"] = days_done
        populate["days_total"] = days_total
        populate["pct"] = overall_pct
        populate["feature_queue"] = _feature_queue(
            features,
            populate.get("current_feature"),
            list(populate.get("features_completed") or []),
        )
        day_label = populate.get("trading_day") or "—"
        _mark_timeline_step(
            timeline,
            "populate",
            status="running",
            detail=(
                f"Days {days_done}/{days_total} ({overall_pct}%) · "
                f"{day_label} {day_pct}% · "
                f"{rows_done:,}/{total_rows:,} rows"
            ),
        )
        progress = {
            **progress,
            "status": "computing",
            "current_step": "populate",
            "timeline": timeline,
            "populate": populate,
            "pending_days": list(pending),
            "completed_days": list(completed),
        }
        # Persist sparingly — status-bar callbacks need frequent in-memory updates.
        now_m = time.monotonic()
        if force_persist or (now_m - last_persist_at[0]) >= 2.0:
            progress = _persist_live_progress(store, progress)
            last_persist_at[0] = now_m
        if on_progress:
            on_progress(dict(progress))

    results: list[dict[str, Any]] = []
    rows_done_before = int(populate.get("rows_done") or 0)
    for day_idx, day in enumerate(days_to_run):
        # Rotate "current feature" display across selected features while computing the day.
        feat_cursor = features[day_idx % len(features)] if features else None
        day_n = int(
            store.conn.execute(
                "SELECT COUNT(*) FROM samples WHERE trading_day = ?",
                (day,),
            ).fetchone()[0]
            or 0
        )
        populate["current_feature"] = feat_cursor
        populate["trading_day"] = day
        populate["day_rows_done"] = 0
        populate["day_rows_total"] = day_n
        _update_live({}, force_persist=True)

        def _day_progress(evt: dict[str, Any]) -> None:
            day_done = int(evt.get("rows_done") or 0)
            day_total = int(evt.get("rows_total") or 0)
            populate["day_rows_done"] = day_done
            populate["day_rows_total"] = day_total
            populate["rows_done"] = rows_done_before + day_done
            populate["trading_day"] = evt.get("trading_day") or day
            # Advance displayed feature through the queue based on day progress.
            if features and day_total > 0:
                idx = min(len(features) - 1, int(day_done / max(1, day_total / len(features))))
                populate["current_feature"] = features[idx]
                populate["features_completed"] = features[:idx]
            _update_live({})

        result = compute_migration_day(
            store,
            trading_day=day,
            features=features,
            enabled_groups=enabled_groups,
            lookback_policy_doc=lb_doc,
            grid_step=float(grid_step),
            default_market=default_market,
            on_progress=_day_progress,
        )
        results.append(result)
        rows_done_before += int(result.get("rows") or 0)
        populate["rows_done"] = rows_done_before
        if day in pending:
            pending.remove(day)
        if day not in completed:
            completed.append(day)
        if features:
            populate["features_completed"] = list(features)
            populate["current_feature"] = features[-1]
        populate["day_rows_done"] = int(populate.get("day_rows_total") or 0)
        _update_live({}, force_persist=True)

    progress["completed_days"] = completed
    progress["pending_days"] = pending
    if not pending:
        _mark_timeline_step(
            timeline,
            "populate",
            status="done",
            detail=f"{total_rows:,} rows · {len(features)} features",
            finished_at=_utc_now(),
            elapsed_sec=round(time.perf_counter() - started_mono, 1),
        )
        progress["status"] = "validated_pending"
        progress["computed_at"] = _utc_now()
        progress["current_step"] = "populate"
        progress["timeline"] = timeline
        populate["pct"] = 100
        populate["eta_sec"] = 0
        populate["feature_queue"] = _feature_queue(features, None, list(features))
        progress["populate"] = populate
    store.set_meta(MIGRATION_META_KEY, progress)

    return {
        "ok": True,
        "days_computed": results,
        "pending_days": pending,
        "completed_days": completed,
        "progress": progress,
    }


def migration_in_progress() -> bool:
    return _migration_active


def run_feature_migration_job(
    db_path: str,
    action: str,
    *,
    features: list[str] | None = None,
    trading_day: str | None = None,
    resume: bool = False,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Synchronous migration action runner for API layer."""
    global _migration_active
    with _migration_lock:
        if action in ("start", "compute", "validate", "commit") and _migration_active:
            raise FeatureMigrationError("A feature migration is already running")
        if action in ("start", "compute", "validate", "commit"):
            _migration_active = True

    store = MasterStore(db_path)
    store.open()
    try:
        if action == "analyze":
            return analyze_master_feature_migration(store)
        if action == "start":
            return start_feature_migration(store, list(features or []), resume=resume)
        if action == "compute":
            return run_feature_migration_compute(
                store,
                trading_day=trading_day,
                on_progress=on_progress,
            )
        if action == "validate":
            progress = store.get_meta(MIGRATION_META_KEY) or {}
            feats = list(progress.get("features") or [])
            report = validate_migration(store, feats)
            progress["validation"] = report
            progress["status"] = "validated" if report.get("passed") else "validation_failed"
            store.set_meta(MIGRATION_META_KEY, progress)
            return {"ok": True, "validation": report}
        if action == "commit":
            progress = store.get_meta(MIGRATION_META_KEY) or {}
            validation = progress.get("validation")
            if not validation:
                validation = validate_migration(store, list(progress.get("features") or []))
            if not validation.get("passed"):
                raise FeatureMigrationError("Commit blocked — validation failed")
            return commit_migration(store, validation=validation)
        if action == "rollback":
            return rollback_migration(store)
        if action == "status":
            progress = store.get_meta(MIGRATION_META_KEY) or {}
            # Keep status polls light while computing — skip full schema analyze.
            analysis = None
            if str(progress.get("status") or "") not in ("computing", "preparing"):
                analysis = analyze_master_feature_migration(store)
            return {
                "progress": progress,
                "analysis": analysis,
                "running": migration_in_progress(),
            }
        raise FeatureMigrationError(f"Unknown action: {action}")
    finally:
        store.close()
        if action in ("start", "compute", "validate", "commit"):
            with _migration_lock:
                _migration_active = False
