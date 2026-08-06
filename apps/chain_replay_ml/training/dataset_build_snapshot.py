"""Resolve frozen dataset-build metadata stored on trained model packages."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from chain_replay_ml.dataset_builder.dataset_summary import build_dataset_build_snapshot
from chain_replay_ml.replay_config import load_dataset_metadata_json

from .paths import model_artifact_paths, models_dir, safe_model_name


def resolve_dataset_build_snapshot(doc: dict[str, Any]) -> dict[str, Any]:
    """Read dataset build snapshot from a load_model_detail() document."""
    snap = doc.get("dataset_build_snapshot")
    if isinstance(snap, dict) and snap:
        return dict(snap)

    cfg = doc.get("config") if isinstance(doc.get("config"), dict) else {}
    snap = cfg.get("dataset_build_snapshot")
    if isinstance(snap, dict) and snap:
        return dict(snap)

    meta_art = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    meta = meta_art.get("data") if isinstance(meta_art.get("data"), dict) else {}
    snap = meta.get("dataset_build_snapshot")
    if isinstance(snap, dict) and snap:
        return dict(snap)

    legacy = cfg.get("dataset_metadata") if isinstance(cfg.get("dataset_metadata"), dict) else {}
    if legacy.get("filter_summary") or legacy.get("master_filter") or legacy.get("selection_method"):
        return dict(legacy)
    return {}


def dataset_meta_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Rebuild dataset-metadata shape from a model snapshot (dataset JSON may be deleted)."""
    if not isinstance(snapshot, dict) or not snapshot:
        return {}

    lineage = snapshot.get("dataset_lineage") if isinstance(snapshot.get("dataset_lineage"), dict) else {}
    days: list[dict[str, Any]] = []
    labels = str(snapshot.get("trading_day_labels") or lineage.get("trading_day_labels") or "").strip()
    if labels and labels != "—":
        for part in labels.split(","):
            td = part.strip()
            if td:
                days.append({"trading_day": td})

    meta: dict[str, Any] = {
        "dataset_name": snapshot.get("dataset_name"),
        "market": snapshot.get("market"),
        "row_count": snapshot.get("row_count"),
        "feature_count": snapshot.get("feature_count"),
        "target_count": snapshot.get("target_count"),
        "column_count": snapshot.get("column_count"),
        "trading_days": snapshot.get("trading_days") or lineage.get("trading_days"),
        "created_at": snapshot.get("created_at"),
        "export_source": snapshot.get("export_source"),
        "builder_version": snapshot.get("builder_version") or snapshot.get("dataset_version"),
        "dataset_version": snapshot.get("dataset_version") or snapshot.get("builder_version"),
        "audit_validation_required": snapshot.get("audit_validation_required"),
        "no_null_dropped_columns": snapshot.get("no_null_dropped_columns"),
    }
    if days:
        meta["days"] = days
    if isinstance(snapshot.get("selection_method"), dict):
        meta["selection_method"] = dict(snapshot["selection_method"])
    if isinstance(snapshot.get("master_filter"), dict):
        meta["master_filter"] = dict(snapshot["master_filter"])
    if isinstance(snapshot.get("trading_day_filter"), dict):
        meta["trading_day_filter"] = dict(snapshot["trading_day_filter"])
    if isinstance(snapshot.get("sampling"), dict):
        meta["sampling"] = dict(snapshot["sampling"])
    if isinstance(snapshot.get("strike_selection"), dict):
        meta["strike_selection"] = dict(snapshot["strike_selection"])
    if isinstance(snapshot.get("pipeline_fingerprint"), dict):
        meta["pipeline_fingerprint"] = dict(snapshot["pipeline_fingerprint"])
    return meta


def _load_json_file(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json_file(path: str, doc: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")


def compact_dataset_metadata(snapshot: dict[str, Any], *, dataset_name: str) -> dict[str, Any]:
    """Compact dataset metadata block stored on config/metadata.json."""
    return {
        "row_count": snapshot.get("row_count"),
        "feature_count": snapshot.get("feature_count"),
        "target_count": snapshot.get("target_count"),
        "dataset_version": snapshot.get("dataset_version") or snapshot.get("builder_version"),
        "dataset_name": dataset_name,
        "trading_days": snapshot.get("trading_days"),
        "trading_day_labels": snapshot.get("trading_day_labels"),
        "market": snapshot.get("market"),
        "sampling_label": snapshot.get("sampling_label"),
        "strike_selection_label": snapshot.get("strike_selection_label"),
        "filter_summary": snapshot.get("filter_summary"),
        "selection_method": snapshot.get("selection_method"),
        "master_filter": snapshot.get("master_filter"),
        "trading_day_filter": snapshot.get("trading_day_filter"),
        "export_source": snapshot.get("export_source"),
        "created_at": snapshot.get("created_at"),
        "snapshotted_at": snapshot.get("snapshotted_at"),
    }


def snapshot_is_complete(snapshot: dict[str, Any]) -> bool:
    if not isinstance(snapshot, dict) or not snapshot:
        return False
    if snapshot.get("filter_summary"):
        return True
    if isinstance(snapshot.get("master_filter"), dict) and snapshot["master_filter"]:
        return True
    if isinstance(snapshot.get("selection_method"), dict) and snapshot["selection_method"].get("summary"):
        return True
    return False


_TRADING_DAY_FILTER_SUMMARY_LABELS = frozenset({
    "trading day filter",
    "excluded expiry dates",
    "expiry dates included",
    "excluded dates",
    "expiry dates (in selection)",
})


def _filter_summary_has_trading_day_filter(snapshot: dict[str, Any]) -> bool:
    for item in snapshot.get("filter_summary") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or "").strip().casefold()
        if label == "trading day filter" and str(item.get("value") or "").strip() not in ("", "—", "-"):
            return True
    return False


def enrich_snapshot_trading_day_filter(
    snapshot: dict[str, Any] | None,
    data_dir: str | None,
    *,
    dataset_name: str | None = None,
) -> dict[str, Any]:
    """
    Ensure snapshot carries trading-day filter rows for UI comparison.

    Older model packages snapshotted filters before trading_day_filter existed.
    Reload from the live dataset registry JSON when needed.
    """
    snap = dict(snapshot) if isinstance(snapshot, dict) else {}
    if _filter_summary_has_trading_day_filter(snap):
        return snap

    tdf = snap.get("trading_day_filter") if isinstance(snap.get("trading_day_filter"), dict) else None
    name = str(
        dataset_name
        or snap.get("dataset_name")
        or "",
    ).strip()

    if (not tdf or not tdf.get("mode")) and data_dir and name:
        meta = load_dataset_metadata_json(data_dir, name)
        if isinstance(meta.get("trading_day_filter"), dict) and meta["trading_day_filter"]:
            tdf = dict(meta["trading_day_filter"])
            if not snap.get("trading_day_labels") and meta.get("days"):
                from chain_replay_ml.dataset_builder.dataset_summary import _trading_day_labels

                snap["trading_day_labels"] = _trading_day_labels(meta)
            if snap.get("trading_days") is None and meta.get("trading_days") is not None:
                snap["trading_days"] = meta.get("trading_days")
            if not snap.get("master_db_path") and meta.get("master_db_path"):
                snap["master_db_path"] = meta.get("master_db_path")

    if not isinstance(tdf, dict) or not tdf:
        return snap

    from chain_replay_ml.dataset_builder.trading_day_filter import (
        enrich_trading_day_filter_dates,
        trading_day_filter_summary_rows,
    )

    labels = str(snap.get("trading_day_labels") or "").strip()
    exported = (
        [p.strip() for p in labels.split(",") if p.strip()]
        if labels and labels != "—"
        else None
    )
    master_day_rows: list[dict[str, Any]] = []
    if data_dir and (
        not tdf.get("excluded_dates")
        and not tdf.get("expiry_dates")
        and not tdf.get("selected_dates")
    ):
        # Build a minimal meta for master-day backfill.
        probe = {
            "trading_day_filter": tdf,
            "master_db_path": snap.get("master_db_path"),
            "_data_dir": data_dir,
            "days": [
                {"trading_day": d} for d in (exported or [])
            ],
        }
        if name and not probe.get("master_db_path"):
            meta = load_dataset_metadata_json(data_dir, name)
            if meta:
                probe["master_db_path"] = meta.get("master_db_path")
                if not exported and meta.get("days"):
                    from chain_replay_ml.dataset_builder.dataset_summary import _trading_day_labels

                    snap["trading_day_labels"] = _trading_day_labels(meta)
                    labels = str(snap.get("trading_day_labels") or "").strip()
                    exported = (
                        [p.strip() for p in labels.split(",") if p.strip()]
                        if labels and labels != "—"
                        else None
                    )
        from chain_replay_ml.dataset_builder.dataset_summary import _master_day_rows_for_filter_backfill

        master_day_rows = _master_day_rows_for_filter_backfill(probe)

    enriched = enrich_trading_day_filter_dates(
        tdf,
        exported_dates=exported,
        master_day_rows=master_day_rows or None,
    )
    snap["trading_day_filter"] = enriched

    filter_rows = [
        item for item in (snap.get("filter_summary") or [])
        if isinstance(item, dict)
        and str(item.get("label") or "").strip().casefold() not in _TRADING_DAY_FILTER_SUMMARY_LABELS
    ]
    filter_rows.extend(
        trading_day_filter_summary_rows(
            enriched,
            exported_dates=exported,
            master_day_rows=master_day_rows or None,
        )
    )
    snap["filter_summary"] = filter_rows
    return snap


def _build_fallback_dataset_meta(
    *,
    config_doc: dict[str, Any],
    metadata_doc: dict[str, Any],
    dataset_name: str,
    pipeline_fingerprint: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Best-effort dataset meta when registry JSON was deleted."""
    replay = config_doc.get("replay_config") if isinstance(config_doc.get("replay_config"), dict) else {}
    legacy = config_doc.get("dataset_metadata") if isinstance(config_doc.get("dataset_metadata"), dict) else {}
    fp = pipeline_fingerprint or metadata_doc.get("pipeline_fingerprint") or legacy.get("pipeline_fingerprint") or {}
    strike = replay.get("strike_selection") if isinstance(replay.get("strike_selection"), dict) else {}
    if str(strike.get("mode") or "").upper() == "ATM_BAND" and strike.get("band") is not None:
        strike = {"mode": "atm_band", "band": strike.get("band")}
    sampling = replay.get("sampling") if isinstance(replay.get("sampling"), dict) else {}
    if not sampling and isinstance(fp, dict) and fp.get("sampling_interval_sec") is not None:
        sampling = {"interval_sec": fp.get("sampling_interval_sec"), "method": "fixed_interval"}
    return {
        "dataset_name": dataset_name,
        "market": replay.get("market") or fp.get("market") or "NIFTY",
        "row_count": legacy.get("row_count") or metadata_doc.get("row_count"),
        "feature_count": legacy.get("feature_count") or metadata_doc.get("feature_count"),
        "dataset_version": legacy.get("dataset_version") or metadata_doc.get("dataset_version"),
        "builder_version": legacy.get("dataset_version") or metadata_doc.get("dataset_version"),
        "sampling": sampling,
        "strike_selection": strike,
        "pipeline_fingerprint": fp if isinstance(fp, dict) else {},
        "dataset_configuration": replay.get("dataset_configuration") if isinstance(replay.get("dataset_configuration"), dict) else {},
    }


def _find_peer_snapshot(
    data_dir: str,
    *,
    dataset_name: str,
    exclude_model: str,
) -> dict[str, Any] | None:
    base = models_dir(data_dir)
    if not os.path.isdir(base):
        return None
    for entry in sorted(os.listdir(base)):
        if entry.startswith(".") or entry == exclude_model:
            continue
        pkg = os.path.join(base, entry)
        if not os.path.isdir(pkg):
            continue
        paths = model_artifact_paths(data_dir, entry)
        snap = _load_json_file(paths["dataset_build_snapshot_json"])
        if not snapshot_is_complete(snap):
            cfg = _load_json_file(paths["config_json"])
            snap = resolve_dataset_build_snapshot({"config": cfg, "dataset_build_snapshot": snap})
        if not snapshot_is_complete(snap):
            continue
        snap_name = str(snap.get("dataset_name") or "").strip()
        cfg = _load_json_file(paths["config_json"])
        cfg_name = str(cfg.get("dataset") or "").strip()
        if snap_name == dataset_name or cfg_name == dataset_name:
            return dict(snap)
    return None


def backfill_model_dataset_snapshot(
    data_dir: str,
    model_name: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Backfill dataset build snapshot for one model from dataset registry JSON."""
    safe = safe_model_name(model_name)
    paths = model_artifact_paths(data_dir, safe)
    pkg = paths["package_dir"]
    if not os.path.isdir(pkg):
        return {"ok": False, "model_name": safe, "status": "missing", "error": "Model package not found"}

    config_doc = _load_json_file(paths["config_json"])
    metadata_doc = _load_json_file(os.path.join(pkg, "metadata.json"))
    existing = resolve_dataset_build_snapshot({
        "config": config_doc,
        "metadata": {"data": metadata_doc},
        "dataset_build_snapshot": _load_json_file(paths["dataset_build_snapshot_json"]),
    })
    if not force and snapshot_is_complete(existing):
        return {
            "ok": True,
            "model_name": safe,
            "status": "skipped",
            "reason": "snapshot already present",
            "dataset": config_doc.get("dataset") or metadata_doc.get("dataset"),
        }

    dataset_name = str(
        config_doc.get("dataset")
        or metadata_doc.get("dataset")
        or existing.get("dataset_name")
        or "",
    ).strip()
    if not dataset_name:
        return {"ok": False, "model_name": safe, "status": "error", "error": "No dataset name on model package"}

    dataset_meta = load_dataset_metadata_json(data_dir, dataset_name)
    snapshot_source = "dataset_json"
    if not dataset_meta:
        peer = _find_peer_snapshot(data_dir, dataset_name=dataset_name, exclude_model=safe)
        if peer:
            dataset_meta = dataset_meta_from_snapshot(peer)
            snapshot_source = "peer_model"
        else:
            fp = _load_json_file(paths["pipeline_fingerprint_json"])
            dataset_meta = _build_fallback_dataset_meta(
                config_doc=config_doc,
                metadata_doc=metadata_doc,
                dataset_name=dataset_name,
                pipeline_fingerprint=fp,
            )
            snapshot_source = "package_fallback"
            if not dataset_meta.get("row_count") and not dataset_meta.get("sampling"):
                return {
                    "ok": False,
                    "model_name": safe,
                    "status": "error",
                    "dataset": dataset_name,
                    "error": f"Dataset metadata not found: {dataset_name}",
                }

    trained_at = (
        config_doc.get("trained_at")
        or metadata_doc.get("trained_at")
        or existing.get("snapshotted_at")
        or datetime.now(timezone.utc).isoformat()
    )
    snapshot = build_dataset_build_snapshot(
        dataset_meta,
        dataset_name=dataset_name,
        snapshotted_at=trained_at,
    )
    snapshot["snapshot_source"] = snapshot_source
    compact = compact_dataset_metadata(snapshot, dataset_name=dataset_name)

    if dry_run:
        return {
            "ok": True,
            "model_name": safe,
            "status": "dry_run",
            "dataset": dataset_name,
            "trading_days": snapshot.get("trading_days"),
            "filter_summary": snapshot.get("filter_summary"),
        }

    config_doc["dataset_build_snapshot"] = snapshot
    config_doc["dataset_metadata"] = compact
    _write_json_file(paths["config_json"], config_doc)

    metadata_doc["dataset_build_snapshot"] = snapshot
    metadata_doc["dataset_metadata"] = compact
    if not metadata_doc.get("pipeline_fingerprint") and snapshot.get("pipeline_fingerprint"):
        metadata_doc["pipeline_fingerprint"] = snapshot["pipeline_fingerprint"]
    _write_json_file(os.path.join(pkg, "metadata.json"), metadata_doc)

    _write_json_file(paths["dataset_build_snapshot_json"], snapshot)

    fingerprint = _load_json_file(paths["pipeline_fingerprint_json"])
    if not fingerprint and snapshot.get("pipeline_fingerprint"):
        _write_json_file(paths["pipeline_fingerprint_json"], dict(snapshot["pipeline_fingerprint"]))

    return {
        "ok": True,
        "model_name": safe,
        "status": "updated",
        "dataset": dataset_name,
        "trading_days": snapshot.get("trading_days"),
        "trading_day_labels": snapshot.get("trading_day_labels"),
        "filter_summary": snapshot.get("filter_summary"),
        "snapshot_source": snapshot_source,
    }


def backfill_all_model_dataset_snapshots(
    data_dir: str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Backfill dataset build snapshots for every model package under data/models."""
    base = models_dir(data_dir)
    if not os.path.isdir(base):
        return {"ok": True, "updated": 0, "skipped": 0, "errors": 0, "results": []}

    results: list[dict[str, Any]] = []
    updated = skipped = errors = 0
    for entry in sorted(os.listdir(base)):
        if entry.startswith("."):
            continue
        pkg = os.path.join(base, entry)
        if not os.path.isdir(pkg):
            continue
        result = backfill_model_dataset_snapshot(
            data_dir,
            entry,
            dry_run=dry_run,
            force=force,
        )
        results.append(result)
        status = str(result.get("status") or "")
        if status in ("updated", "dry_run"):
            updated += 1
        elif status == "skipped":
            skipped += 1
        elif status in ("error", "missing"):
            errors += 1

    return {
        "ok": errors == 0,
        "updated": updated,
        "skipped": skipped,
        "errors": errors,
        "dry_run": dry_run,
        "results": results,
    }
