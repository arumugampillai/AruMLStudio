"""Append / incremental dataset build helpers."""

from __future__ import annotations

import json
import os
from typing import Any

from .writer import _safe_filename, datasets_dir

SessionKey = tuple[str, str, str]


def session_key(trading_day: str, market: str, expiry: str) -> SessionKey:
    return (
        str(trading_day or "").strip(),
        str(market or "NIFTY").upper().strip(),
        str(expiry or "").strip(),
    )


def format_day_label(trading_day: str) -> str:
    try:
        y, m, d = (int(p) for p in str(trading_day).split("-"))
        months = [
            "Jan", "Feb", "Mar", "Apr", "May", "Jun",
            "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
        ]
        return f"{d:02d}-{months[m - 1]}"
    except (TypeError, ValueError, IndexError):
        return str(trading_day)


def _session_entry(
    *,
    trading_day: str,
    market: str,
    expiry: str,
    source_id: str = "",
    date_label: str = "",
) -> dict[str, Any]:
    return {
        "trading_day": trading_day,
        "market": str(market or "").upper(),
        "expiry": expiry,
        "source_id": source_id or f"{trading_day}|{market}|{expiry}",
        "label": date_label or format_day_label(trading_day),
        "session_key": list(session_key(trading_day, market, expiry)),
    }


def sessions_from_metadata(meta: dict[str, Any]) -> set[SessionKey]:
    keys: set[SessionKey] = set()
    for day in meta.get("days") or []:
        keys.add(session_key(
            str(day.get("trading_day") or ""),
            str(day.get("market") or ""),
            str(day.get("expiry") or ""),
        ))
    return keys


def existing_day_entries(meta: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for day in meta.get("days") or []:
        out.append(_session_entry(
            trading_day=str(day.get("trading_day") or ""),
            market=str(day.get("market") or ""),
            expiry=str(day.get("expiry") or ""),
            source_id=str(day.get("source_id") or ""),
        ))
    return out


def load_dataset_metadata(data_dir: str, dataset_name: str) -> tuple[dict[str, Any], dict[str, str]]:
    safe_name = _safe_filename(dataset_name)
    out_dir = datasets_dir(data_dir)
    json_path = os.path.join(out_dir, f"{safe_name}.json")
    parquet_path = os.path.join(out_dir, f"{safe_name}.parquet")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Metadata not found for {safe_name}")
    with open(json_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    return meta, {
        "json": json_path,
        "parquet": parquet_path,
        "safe_name": safe_name,
    }


def plan_append_sessions(
    existing_meta: dict[str, Any],
    sources: list[dict[str, Any]],
    *,
    unavailable_source_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Compare existing dataset sessions with selected build sources."""
    existing = sessions_from_metadata(existing_meta)
    unavailable = unavailable_source_ids or set()
    already_present: list[dict[str, Any]] = []
    new_days: list[dict[str, Any]] = []
    duplicate_days: list[dict[str, Any]] = []
    skipped_days: list[dict[str, Any]] = []
    new_sources: list[dict[str, Any]] = []
    selected_entries: list[dict[str, Any]] = []
    seen_selected: set[SessionKey] = set()

    for raw in sources:
        trading_day = str(raw.get("trading_day") or "")
        market = str(raw.get("market") or "NIFTY").upper()
        expiry = str(raw.get("expiry") or "")
        source_id = str(raw.get("source_id") or f"{trading_day}|{market}|{expiry}")
        entry = _session_entry(
            trading_day=trading_day,
            market=market,
            expiry=expiry,
            source_id=source_id,
            date_label=str(raw.get("date") or raw.get("date_label") or ""),
        )
        selected_entries.append(entry)

        if source_id in unavailable:
            skipped_days.append(entry)
            continue

        key = session_key(trading_day, market, expiry)
        if key in seen_selected:
            duplicate_days.append(entry)
            continue
        seen_selected.add(key)

        if key in existing:
            already_present.append(entry)
        else:
            new_days.append(entry)
            new_sources.append(dict(raw))

    preserved = [
        d for d in existing_day_entries(existing_meta)
        if session_key(d["trading_day"], d["market"], d["expiry"]) not in seen_selected
    ]

    return {
        "dataset_name": existing_meta.get("dataset_name"),
        "existing_days": existing_day_entries(existing_meta),
        "selected_days": selected_entries,
        "already_present": already_present,
        "new_days": new_days,
        "duplicate_days": duplicate_days,
        "skipped_days": skipped_days,
        "preserved_days": preserved,
        "will_append": new_days,
        "new_sources": new_sources,
        "existing_row_count": int(existing_meta.get("row_count") or 0),
    }


def validate_append_compatible(
    existing_meta: dict[str, Any],
    *,
    sampling: dict[str, Any],
    strike_selection: dict[str, Any],
    prediction_targets: dict[str, Any],
    feature_selection: dict[str, Any],
    registry: dict[str, Any] | None = None,
) -> list[str]:
    """Ensure append build config matches the existing dataset pipeline spec."""
    from .expected_spec import _strike_mode, strike_selection_metadata
    from .feature_plugins import horizon_column_name, resolve_implemented_features_for_selection
    from .lookback_policy import build_dataset_configuration
    from .pipeline_identity import build_pipeline_fingerprint
    from .spec_identity import compute_spec_hash_from_fingerprint

    errors: list[str] = []
    horizons_sec = [int(h) for h in (prediction_targets.get("horizonsSec") or [])]
    target_columns = [horizon_column_name(h) for h in horizons_sec]
    step_sec = int(sampling.get("trainingIntervalSec") or 10)
    atm_band = int(strike_selection.get("atmBand") or 10)

    existing_targets = list(existing_meta.get("prediction_target_columns") or [])
    if existing_targets and existing_targets != target_columns:
        errors.append(
            "Prediction targets differ from existing dataset "
            f"({', '.join(existing_targets)} vs {', '.join(target_columns)})"
        )

    existing_feats = list(existing_meta.get("feature_columns") or [])
    _, implemented, _, _ = resolve_implemented_features_for_selection(
        feature_selection, registry or _load_registry(),
    )
    if existing_feats and existing_feats != implemented:
        errors.append(
            f"Feature columns differ from existing dataset ({len(existing_feats)} vs {len(implemented)})"
        )

    existing_sampling = existing_meta.get("sampling") or {}
    if int(existing_sampling.get("interval_sec") or 0) not in (0, step_sec):
        errors.append(
            f"Sampling interval differs ({existing_sampling.get('interval_sec')}s vs {step_sec}s)"
        )

    existing_strike = existing_meta.get("strike_selection") or {}
    existing_mode = str(existing_strike.get("mode") or "ATM_BAND").upper()
    new_mode = _strike_mode(strike_selection)
    if existing_mode != new_mode:
        errors.append(f"Strike selection mode differs ({existing_mode} vs {new_mode})")
    elif existing_mode == "ATM_BAND":
        if int(existing_strike.get("band") or 0) not in (0, atm_band):
            errors.append(
                f"ATM band differs ({existing_strike.get('band')} vs {atm_band})"
            )
    elif existing_mode == "DELTA_RANGE":
        new_meta = strike_selection_metadata(strike_selection)
        for key in ("delta_type", "delta_min", "delta_max"):
            if existing_strike.get(key) != new_meta.get(key):
                errors.append(f"Delta range {key} differs from existing dataset")
                break

    existing_hash = str(existing_meta.get("dataset_spec_hash") or "").strip()
    if existing_hash:
        dataset_configuration = build_dataset_configuration(
            sampling=sampling,
            horizons_sec=horizons_sec,
        )
        pipeline_fingerprint = build_pipeline_fingerprint(
            sampling_interval_sec=step_sec,
            atm_band=atm_band,
            feature_count=len(implemented),
            target_horizons_sec=horizons_sec,
            lookback_policy=dataset_configuration["lookback_policy"]["method"],
            registry=registry or _load_registry(),
        )
        new_hash = compute_spec_hash_from_fingerprint(pipeline_fingerprint, dataset_configuration)
        if new_hash != existing_hash:
            errors.append("Build configuration spec hash does not match existing dataset")

    return errors


def estimate_rows_for_sources(
    *,
    source_count: int,
    sampling: dict[str, Any],
    strike_selection: dict[str, Any],
    prediction_targets: dict[str, Any],
) -> int:
    if source_count <= 0:
        return 0
    step_sec = int(sampling.get("trainingIntervalSec") or 10)
    horizons_sec = [int(h) for h in (prediction_targets.get("horizonsSec") or [])]
    max_horizon = max(horizons_sec) if horizons_sec else 0
    usable_sec = 22500 - 60 - max_horizon
    if usable_sec <= 0 or step_sec <= 0:
        return 0
    sample_points = (usable_sec // step_sec) + 1
    mode = str(strike_selection.get("mode") or "atm_band").lower()
    atm_band = int(strike_selection.get("atmBand") or 10)
    strikes_per_ts = 18 if mode == "delta_range" else (2 * atm_band + 1) * 2
    return sample_points * strikes_per_ts * source_count


def merge_metadata_for_append(
    existing_meta: dict[str, Any],
    *,
    new_days_meta: list[dict[str, Any]],
    new_source_results: list[dict[str, Any]],
    new_warnings: list[str],
    new_target_trimmed: int,
    build_performance: dict[str, Any],
    append_job_id: str,
) -> dict[str, Any]:
    from datetime import datetime, timezone

    merged = dict(existing_meta)
    old_days = list(existing_meta.get("days") or [])
    old_sources = list(existing_meta.get("sources") or [])
    old_keys = {
        session_key(d.get("trading_day", ""), d.get("market", ""), d.get("expiry", ""))
        for d in old_days
    }
    for day in new_days_meta:
        key = session_key(day.get("trading_day", ""), day.get("market", ""), day.get("expiry", ""))
        if key not in old_keys:
            old_days.append(day)
            old_keys.add(key)
    merged["days"] = old_days
    merged["sources"] = old_sources + list(new_source_results)
    merged["warnings"] = list(existing_meta.get("warnings") or []) + list(new_warnings)
    merged["target_trimmed_rows"] = int(existing_meta.get("target_trimmed_rows") or 0) + int(new_target_trimmed)
    merged["updated_at"] = datetime.now(timezone.utc).isoformat()
    history = list(existing_meta.get("append_history") or [])
    history.append({
        "job_id": append_job_id,
        "at": merged["updated_at"],
        "sessions_added": len(new_days_meta),
        "rows_added": None,
        "build_performance": build_performance,
    })
    merged["append_history"] = history
    return merged


def validate_append_merge(
    existing_df,
    new_rows: list[dict[str, Any]],
) -> list[str]:
    """Check schema alignment and primary-key collisions before merging."""
    if not new_rows:
        return ["No new rows to append"]
    existing_cols = set(existing_df.columns)
    new_cols = set(new_rows[0].keys())
    if existing_cols != new_cols:
        missing = sorted(existing_cols - new_cols)
        extra = sorted(new_cols - existing_cols)
        parts = []
        if missing:
            parts.append(f"missing in new rows: {', '.join(missing[:5])}")
        if extra:
            parts.append(f"extra in new rows: {', '.join(extra[:5])}")
        return ["Column mismatch between existing dataset and new rows (" + "; ".join(parts) + ")"]

    key_cols = ["trading_day", "market", "expiry", "timestamp", "strike", "option_type"]
    missing_key_cols = [c for c in key_cols if c not in existing_cols]
    if missing_key_cols:
        return [f"Existing dataset missing key columns: {', '.join(missing_key_cols)}"]

    existing_keys: set[tuple[Any, ...]] = set()
    for chunk_start in range(0, len(existing_df), 25_000):
        chunk = existing_df.iloc[chunk_start:chunk_start + 25_000]
        for row in chunk[key_cols].itertuples(index=False, name=None):
            existing_keys.add(tuple(row))

    collisions = 0
    for row in new_rows:
        key = tuple(row.get(c) for c in key_cols)
        if key in existing_keys:
            collisions += 1
    if collisions:
        return [f"{collisions:,} new rows collide with existing dataset keys"]
    return []


def _load_registry() -> dict[str, Any]:
    from .schema_registry import load_feature_registry

    return load_feature_registry()
