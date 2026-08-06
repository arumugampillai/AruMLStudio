"""Build contract (*.expected.json) written before pipeline stage 1."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from .feature_plugins import horizon_column_name, horizon_label
from .schema_registry import metadata_column_names, resolve_feature_selection
from .writer import _safe_filename, datasets_dir

EXPECTED_SPEC_VERSION = 1

# Identity / session columns in the Parquet row (not ML features or prediction targets).
DATASET_METADATA_COLUMNS: tuple[str, ...] = tuple(metadata_column_names())


def _strike_mode(strike_selection: dict[str, Any]) -> str:
    mode = str(strike_selection.get("mode") or "atm_band").lower()
    if mode == "atm_band":
        return "ATM_BAND"
    if mode == "premium_band":
        return "PREMIUM_BAND"
    if mode == "delta_range":
        return "DELTA_RANGE"
    if mode == "custom":
        return "CUSTOM"
    return mode.upper()


def resolve_atm_band(meta: dict[str, Any] | None) -> int | str | None:
    """Canonical ATM band for a dataset — pipeline fingerprint wins over stale strike_selection defaults."""
    if not isinstance(meta, dict) or not meta:
        return None
    strike_sel = meta.get("strike_selection") if isinstance(meta.get("strike_selection"), dict) else {}
    mode_key = str((strike_sel or {}).get("mode") or "atm_band").lower()
    if mode_key not in ("", "atm_band"):
        return None

    fp = meta.get("pipeline_fingerprint")
    if isinstance(fp, dict) and fp.get("atm_band") is not None:
        band = fp.get("atm_band")
        if str(band).lower() == "all":
            return "all"
        try:
            return int(band)
        except (TypeError, ValueError):
            pass

    mf = meta.get("master_filter")
    if isinstance(mf, dict) and mf.get("atm_band_filter") is not None:
        band = mf.get("atm_band_filter")
        if str(band).lower() == "all":
            return "all"
        try:
            return int(band)
        except (TypeError, ValueError):
            pass

    sm = meta.get("selection_method")
    if isinstance(sm, dict):
        crit = sm.get("criteria") if isinstance(sm.get("criteria"), dict) else {}
        if crit.get("atm_band_filter") is not None:
            band = crit.get("atm_band_filter")
            if str(band).lower() == "all":
                return "all"
            try:
                return int(band)
            except (TypeError, ValueError):
                pass

    if isinstance(strike_sel, dict):
        band = strike_sel.get("band")
        if band is None:
            band = strike_sel.get("atmBand")
        if band is not None:
            if str(band).lower() == "all":
                return "all"
            try:
                return int(band)
            except (TypeError, ValueError):
                return band
    return None


def strike_selection_metadata(strike_selection: dict[str, Any]) -> dict[str, Any]:
    """Normalized strike-selection block for expected.json / dataset metadata."""
    mode_key = str(strike_selection.get("mode") or "atm_band").lower()
    out: dict[str, Any] = {"mode": _strike_mode(strike_selection)}
    if mode_key == "atm_band":
        band = strike_selection.get("atmBand", 10)
        out["band"] = band if str(band).lower() == "all" else int(band or 10)
    elif mode_key == "premium_band":
        out["premium_min"] = float(strike_selection.get("premiumMin") or 15)
        out["premium_max"] = float(strike_selection.get("premiumMax") or 30)
    elif mode_key == "custom":
        out["custom_offsets"] = list(strike_selection.get("customOffsets") or [])
    elif mode_key == "delta_range":
        out["delta_type"] = str(strike_selection.get("deltaType") or "absolute").lower()
        out["delta_min"] = float(strike_selection.get("deltaMin") or 0.15)
        out["delta_max"] = float(strike_selection.get("deltaMax") or 0.50)
    return out


def format_strike_selection_label(strike_selection: dict[str, Any] | None) -> str | None:
    """Human-readable strike selection for registry / model summaries."""
    if not isinstance(strike_selection, dict) or not strike_selection:
        return None
    mode_key = str(strike_selection.get("mode") or "atm_band").lower()
    if mode_key == "atm_band":
        band = strike_selection.get("band")
        if band is None:
            band = strike_selection.get("atmBand")
        if band is None:
            return "ATM Band"
        if str(band).lower() == "all":
            return "ATM All"
        try:
            return f"ATM ±{int(band)}"
        except (TypeError, ValueError):
            return f"ATM ±{band}"
    if mode_key == "premium_band":
        lo = strike_selection.get("premium_min") or strike_selection.get("premiumMin")
        hi = strike_selection.get("premium_max") or strike_selection.get("premiumMax")
        if lo is not None and hi is not None:
            return f"Premium {lo}–{hi}"
        return "Premium Band"
    if mode_key == "delta_range":
        lo = strike_selection.get("delta_min") or strike_selection.get("deltaMin")
        hi = strike_selection.get("delta_max") or strike_selection.get("deltaMax")
        dtype = str(strike_selection.get("delta_type") or strike_selection.get("deltaType") or "absolute")
        if lo is not None and hi is not None:
            return f"Delta {dtype} {lo}–{hi}"
        return "Delta Range"
    if mode_key == "custom":
        offsets = strike_selection.get("custom_offsets") or strike_selection.get("customOffsets") or []
        if offsets:
            return f"Custom ({len(offsets)} offsets)"
        return "Custom Strikes"
    return _strike_mode(strike_selection).replace("_", " ").title()


def strike_selection_display_label(meta: dict[str, Any] | None) -> str | None:
    """Strike selection only — ATM ±N (not day/token/premium master-filter summary)."""
    if not isinstance(meta, dict) or not meta:
        return None
    strike_sel = meta.get("strike_selection") if isinstance(meta.get("strike_selection"), dict) else {}
    mode_key = str((strike_sel or {}).get("mode") or "").lower()
    if mode_key and mode_key not in ("", "atm_band"):
        return format_strike_selection_label(strike_sel)

    band = resolve_atm_band(meta)
    if band is not None:
        return format_strike_selection_label({"mode": "atm_band", "band": band})

    sm = meta.get("selection_method")
    summary = ""
    if isinstance(sm, str):
        summary = sm.strip()
    elif isinstance(sm, dict):
        summary = str(sm.get("summary") or sm.get("label") or "").strip()
    if summary:
        import re

        match = re.search(r"ATM\s*±\s*(\d+)", summary, re.IGNORECASE)
        if match:
            return f"ATM ±{int(match.group(1))}"
        if re.search(r"ATM\s+all", summary, re.IGNORECASE):
            return "ATM All"
    return format_strike_selection_label(strike_sel) if strike_sel else None


def format_sampling_interval_label(sec: int | float | str | None) -> str | None:
    """Human-readable sampling interval (e.g. 3s, 1m)."""
    if sec is None or sec == "":
        return None
    try:
        n = int(sec)
    except (TypeError, ValueError):
        text = str(sec).strip().lower()
        return text or None
    if n >= 60 and n % 60 == 0:
        return f"{n // 60}m"
    return f"{n}s"


def sampling_interval_display_label(meta: dict[str, Any] | None) -> str | None:
    """Sampling interval from dataset metadata."""
    if not isinstance(meta, dict) or not meta:
        return None
    cfg = meta.get("dataset_configuration") if isinstance(meta.get("dataset_configuration"), dict) else {}
    sampling = meta.get("sampling") if isinstance(meta.get("sampling"), dict) else {}
    sm = meta.get("selection_method")
    fp = meta.get("pipeline_fingerprint")
    sec = None
    for key in ("sampling_interval_sec", "feature_grid_step_sec"):
        if cfg.get(key) is not None:
            sec = cfg.get(key)
            break
    if sec is None and sampling.get("interval_sec") is not None:
        sec = sampling.get("interval_sec")
    if sec is None and sampling.get("trainingIntervalSec") is not None:
        sec = sampling.get("trainingIntervalSec")
    if sec is None and isinstance(sm, dict) and sm.get("interval_sec") is not None:
        sec = sm.get("interval_sec")
    if sec is None and isinstance(fp, dict):
        if fp.get("sampling_interval_sec") is not None:
            sec = fp.get("sampling_interval_sec")
        elif fp.get("sampling"):
            return str(fp.get("sampling"))
    return format_sampling_interval_label(sec)


def build_expected_spec(
    *,
    dataset_name: str,
    sources: list[dict[str, Any]],
    sampling: dict[str, Any],
    strike_selection: dict[str, Any],
    prediction_targets: dict[str, Any],
    feature_selection: dict[str, Any],
    registry: dict[str, Any],
) -> dict[str, Any]:
    enabled_groups, feature_names = resolve_feature_selection(feature_selection, registry)
    horizons_sec = [int(h) for h in (prediction_targets.get("horizonsSec") or [])]
    target_columns = [horizon_column_name(h) for h in horizons_sec]
    metadata_columns = list(DATASET_METADATA_COLUMNS)
    n_features = len(feature_names)
    markets = {str(s.get("market") or "").upper() for s in sources if s.get("market")}
    market = markets.pop() if len(markets) == 1 else "MIXED"

    from .lookback_policy import build_dataset_configuration
    from .pipeline_identity import (
        BUILDER_VERSION,
        build_pipeline_fingerprint,
        git_commit,
    )
    from .spec_identity import compute_spec_hash_from_fingerprint

    dataset_configuration = build_dataset_configuration(
        sampling=sampling,
        horizons_sec=horizons_sec,
    )
    lookback_method = dataset_configuration["lookback_policy"]["method"]
    pipeline_fingerprint = build_pipeline_fingerprint(
        sampling_interval_sec=int(sampling.get("trainingIntervalSec") or 10),
        atm_band=int(strike_selection.get("atmBand") or 10),
        feature_count=n_features,
        target_horizons_sec=horizons_sec,
        lookback_policy=lookback_method,
        registry=registry,
        builder_version=BUILDER_VERSION,
    )

    selected_sources = [
        {
            "source_id": str(s.get("source_id") or ""),
            "trading_day": str(s.get("trading_day") or ""),
            "market": str(s.get("market") or "").upper(),
            "expiry": str(s.get("expiry") or ""),
            "date": s.get("date"),
        }
        for s in sources
    ]

    n_targets = len(target_columns)
    n_metadata = len(metadata_columns)
    all_column_names = list(dict.fromkeys(metadata_columns + feature_names + target_columns))

    return {
        "version": EXPECTED_SPEC_VERSION,
        "dataset_name": _safe_filename(dataset_name),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "market": market,
        "selected_sources": selected_sources,
        "sampling": {
            "interval_sec": int(sampling.get("trainingIntervalSec") or 10),
            "method": str(sampling.get("samplingMethod") or "fixed_interval"),
        },
        "strike_selection": strike_selection_metadata(strike_selection),
        "prediction_targets": [horizon_label(h) for h in horizons_sec],
        "prediction_target_columns": target_columns,
        "feature_profile": str(feature_selection.get("profile") or "default"),
        "feature_groups": enabled_groups,
        "enabled_features": feature_names,
        "dataset_configuration": dataset_configuration,
        "lookback_policy": lookback_method,
        "pipeline_fingerprint": pipeline_fingerprint,
        "dataset_version": BUILDER_VERSION,
        "builder_version": BUILDER_VERSION,
        "git_commit": git_commit(),
        "feature_registry_version": pipeline_fingerprint["feature_registry_version"],
        "feature_registry_hash": pipeline_fingerprint["feature_registry_hash"],
        "pipeline_stage_hashes": pipeline_fingerprint["pipeline_stage_hashes"],
        "dataset_spec_hash": compute_spec_hash_from_fingerprint(
            pipeline_fingerprint, dataset_configuration,
        ),
        "expected": {
            "feature_groups": len(enabled_groups),
            "expected_feature_columns": n_features,
            "expected_target_columns": n_targets,
            "expected_metadata_columns": n_metadata,
            "expected_total_columns": len(all_column_names),
            "feature_column_names": feature_names,
            "metadata_column_names": metadata_columns,
            "target_column_names": target_columns,
            "all_column_names": all_column_names,
        },
    }


def write_expected_spec(
    *,
    data_dir: str,
    dataset_name: str,
    sources: list[dict[str, Any]],
    sampling: dict[str, Any],
    strike_selection: dict[str, Any],
    prediction_targets: dict[str, Any],
    feature_selection: dict[str, Any],
    registry: dict[str, Any],
) -> str:
    """Write ``<dataset_name>.expected.json`` before any database access."""
    spec = build_expected_spec(
        dataset_name=dataset_name,
        sources=sources,
        sampling=sampling,
        strike_selection=strike_selection,
        prediction_targets=prediction_targets,
        feature_selection=feature_selection,
        registry=registry,
    )
    safe_name = spec["dataset_name"]
    out_dir = datasets_dir(data_dir)
    path = os.path.join(out_dir, f"{safe_name}.expected.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(spec, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def expected_spec_path(data_dir: str, dataset_name: str) -> str:
    safe_name = _safe_filename(dataset_name)
    return os.path.join(datasets_dir(data_dir), f"{safe_name}.expected.json")
