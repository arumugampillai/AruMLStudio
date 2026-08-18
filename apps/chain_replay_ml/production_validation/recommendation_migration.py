"""Idempotent migration from feature_recommendation_history.json to SQLite Evidence DB."""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

from .dataset_context import LEGACY_UNKNOWN_CONTEXT_ID, resolve_context_or_legacy
from .evidence_store import (
    append_validation_evidence,
    get_connection,
)
from .recommendation_policy import load_recommendation_policy

MIGRATION_META_KEY = "json_migration_completed"
LEGACY_JSON_FILENAME = "feature_recommendation_history.json"


def legacy_json_path(data_dir: str) -> str:
    return os.path.join(data_dir, LEGACY_JSON_FILENAME)


def is_migration_completed(conn: sqlite3.Connection) -> bool:
    cur = conn.execute(
        "SELECT meta_value FROM migration_meta WHERE meta_key = ?",
        (MIGRATION_META_KEY,),
    )
    row = cur.fetchone()
    return bool(row and str(row["meta_value"]).lower() == "true")


def mark_migration_completed(conn: sqlite3.Connection) -> None:
    from .evidence_store import _utc_now

    with conn:
        conn.execute(
            """
            INSERT INTO migration_meta (meta_key, meta_value, updated_at)
            VALUES (?, 'true', ?)
            ON CONFLICT(meta_key) DO UPDATE SET
                meta_value='true',
                updated_at=excluded.updated_at;
            """,
            (MIGRATION_META_KEY, _utc_now()),
        )


def migrate_legacy_recommendation_json(
    data_dir: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Perform idempotent migration of feature_recommendation_history.json to SQLite Evidence Store."""
    conn = get_connection(data_dir)
    try:
        if is_migration_completed(conn) and not force:
            return {
                "status": "already_completed",
                "migrated_entries": 0,
                "legacy_unknown_count": 0,
            }

        json_path = legacy_json_path(data_dir)
        if not os.path.isfile(json_path):
            mark_migration_completed(conn)
            return {
                "status": "no_legacy_json_found",
                "migrated_entries": 0,
                "legacy_unknown_count": 0,
            }

        try:
            with open(json_path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except Exception as exc:
            return {
                "status": "error_reading_json",
                "error": str(exc),
                "migrated_entries": 0,
            }

        if not isinstance(doc, dict):
            mark_migration_completed(conn)
            return {"status": "invalid_json_format", "migrated_entries": 0}

        entries = doc.get("entries") or []
        if not isinstance(entries, list):
            entries = []

        policy = load_recommendation_policy(data_dir)
        total_migrated = 0
        legacy_unknown_count = 0

        # Group entries by resolved context to batch insert efficiently
        entries_by_context: dict[str, tuple[Any, list[dict[str, Any]]]] = {}

        for entry in entries:
            if not isinstance(entry, dict):
                continue
            model_name = str(entry.get("model_name") or "")
            feat_name = str(entry.get("feature_name") or entry.get("feature_id") or "").strip()
            if not feat_name:
                continue

            rec = str(entry.get("recommendation") or "KEEP").strip().upper()
            if rec not in ("KEEP", "WATCH", "REMOVE"):
                rec = "KEEP"

            # Resolve context from model package if available
            context = resolve_context_or_legacy(data_dir, model_name)
            if context.context_id == LEGACY_UNKNOWN_CONTEXT_ID:
                legacy_unknown_count += 1

            f_source = str(entry.get("feature_source") or "").strip().lower()
            pid = str(entry.get("pipeline_id") or "").strip().upper() or None
            snap_id = str(entry.get("pipeline_snapshot_id") or "").strip() or None

            if not f_source or f_source not in ("registry", "base_pipeline", "experimental"):
                from chain_replay_ml.dataset_builder.feature_sources_catalog import registry_feature_names
                from chain_replay_ml.dataset_builder.pipeline_registry_store import (
                    ensure_default_existing_pipeline,
                    is_base_pipeline_record,
                    load_store as load_pipeline_store,
                )
                from chain_replay_ml.training.paths import model_package_dir, safe_model_name

                pkg = model_package_dir(data_dir, safe_model_name(model_name))
                cfg_path = os.path.join(pkg, "config.json")
                m_cfg: dict[str, Any] = {}
                if os.path.isfile(cfg_path):
                    try:
                        with open(cfg_path, encoding="utf-8") as cfh:
                            m_cfg = json.load(cfh)
                    except Exception:
                        m_cfg = {}

                if not pid:
                    pid = str(m_cfg.get("pipeline_id") or "").strip().upper() or None
                if not snap_id:
                    snap_id = str(m_cfg.get("pipeline_snapshot_id") or "").strip() or None

                ds_meta = m_cfg.get("dataset_metadata") or {}
                reg_names = set(ds_meta.get("registry_features") or []) | set(registry_feature_names(data_dir=data_dir))
                pstore = ensure_default_existing_pipeline(data_dir)
                base_candidates: set[str] = set(ds_meta.get("base_pipeline_features") or [])
                for prec in (pstore.get("pipelines") or {}).values():
                    if isinstance(prec, dict) and is_base_pipeline_record(prec):
                        base_candidates.update(str(x).strip() for x in (prec.get("candidate_features") or []))
                exp_set = set(ds_meta.get("other_pipeline_features") or [])

                if feat_name in base_candidates:
                    f_source = "base_pipeline"
                elif feat_name in exp_set or (pid and feat_name not in reg_names):
                    f_source = "experimental"
                else:
                    f_source = "registry"

            run_id = str(entry.get("production_validation_run_id") or "").strip()
            if run_id:
                from chain_replay_ml.training.paths import safe_model_name
                ev_id = f"ev_{run_id}_{safe_model_name(model_name)}_{feat_name}"
            else:
                ev_id = f"mig_{entry.get('id') or total_migrated}_{model_name}_{feat_name}"

            ev_row = {
                "evidence_id": ev_id,
                "feature_name": feat_name,
                "feature_source": f_source,
                "pipeline_id": pid if f_source == "experimental" else None,
                "pipeline_snapshot_id": snap_id if f_source == "experimental" else None,
                "recommendation": rec,
                "validation_run_id": run_id or "legacy_run",
                "model_name": model_name,
                "recommendation_detail": entry.get("recommendation_detail"),
                "run_timestamp": entry.get("generated_date"),
            }

            if context.context_id not in entries_by_context:
                entries_by_context[context.context_id] = (context, [])
            entries_by_context[context.context_id][1].append(ev_row)
            total_migrated += 1

        # Insert batch for each context
        for cid, (ctx, rows) in entries_by_context.items():
            append_validation_evidence(
                conn,
                context=ctx,
                evidence_rows=rows,
                policy=policy,
            )

        mark_migration_completed(conn)
        return {
            "status": "completed",
            "migrated_entries": total_migrated,
            "legacy_unknown_count": legacy_unknown_count,
            "contexts_count": len(entries_by_context),
        }
    finally:
        conn.close()
