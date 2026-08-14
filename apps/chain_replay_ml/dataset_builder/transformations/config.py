"""Transformation configuration helpers."""

from __future__ import annotations

from typing import Any

TRANSFORMATION_PIPELINE_VERSION = 1


def default_transformation_config() -> dict[str, Any]:
    """Default — no transformations enabled."""
    return {
        "transformation_pipeline_version": TRANSFORMATION_PIPELINE_VERSION,
        "transformations": [],
    }


def normalize_transformation_config(raw: Any | None) -> dict[str, Any]:
    """Normalize user/metadata config into the versioned pipeline object.

    Accepts:
    - ``None`` / missing → default empty config
    - ``{"transformation_pipeline_version": N, "transformations": [...]}``
    - ``{"transformations": [...]}`` (version filled in)
    - a bare list (treated as the transformations list)
    """
    if raw is None:
        return default_transformation_config()
    if isinstance(raw, list):
        return {
            "transformation_pipeline_version": TRANSFORMATION_PIPELINE_VERSION,
            "transformations": [_normalize_entry(e) for e in raw if e is not None],
        }
    if not isinstance(raw, dict):
        return default_transformation_config()

    version = raw.get("transformation_pipeline_version", TRANSFORMATION_PIPELINE_VERSION)
    try:
        version_i = int(version)
    except (TypeError, ValueError):
        version_i = TRANSFORMATION_PIPELINE_VERSION

    entries = raw.get("transformations")
    if entries is None:
        if "id" in raw or "enabled" in raw:
            return {
                "transformation_pipeline_version": version_i,
                "transformations": [_normalize_entry(raw)],
            }
        return {
            "transformation_pipeline_version": version_i,
            "transformations": [],
        }
    if not isinstance(entries, list):
        return {
            "transformation_pipeline_version": version_i,
            "transformations": [],
        }
    return {
        "transformation_pipeline_version": version_i,
        "transformations": [_normalize_entry(e) for e in entries if e is not None],
    }


def _normalize_entry(entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return {"id": str(entry), "enabled": False}
    out: dict[str, Any] = {
        "id": str(entry.get("id") or "").strip(),
        "enabled": bool(entry.get("enabled", False)),
    }
    if entry.get("order") is not None:
        try:
            out["order"] = int(entry["order"])
        except (TypeError, ValueError):
            pass
    if entry.get("name") is not None:
        out["name"] = str(entry.get("name") or "")
    params = entry.get("params")
    if isinstance(params, dict):
        out["params"] = dict(params)
    depends = entry.get("depends_on")
    if isinstance(depends, (list, tuple)):
        out["depends_on"] = [str(d).strip() for d in depends if str(d).strip()]
    for key, value in entry.items():
        if key in out or key in ("id", "enabled", "order", "name", "params", "depends_on"):
            continue
        out[key] = value
    return out


def config_for_metadata(config: dict[str, Any] | None) -> dict[str, Any]:
    """Return the configuration object persisted into dataset metadata."""
    return normalize_transformation_config(config)


def merge_transformation_configs(
    base: dict[str, Any] | None,
    extra: dict[str, Any] | None,
) -> dict[str, Any]:
    """Append ``extra`` transformation stages after ``base`` (experimental after catalogue)."""
    primary = normalize_transformation_config(base)
    secondary = normalize_transformation_config(extra)
    merged = list(primary.get("transformations") or [])
    max_order = 0
    for stage in merged:
        try:
            max_order = max(max_order, int(stage.get("order") or 0))
        except (TypeError, ValueError):
            continue
    offset = max_order + 100
    for stage in secondary.get("transformations") or []:
        if not isinstance(stage, dict) or not stage.get("enabled"):
            continue
        copy = dict(stage)
        if copy.get("order") is not None:
            try:
                copy["order"] = int(copy["order"]) + offset
            except (TypeError, ValueError):
                copy["order"] = offset
        else:
            copy["order"] = offset
        merged.append(copy)
    return {
        "transformation_pipeline_version": int(
            primary.get("transformation_pipeline_version") or TRANSFORMATION_PIPELINE_VERSION
        ),
        "transformations": merged,
    }


def _seconds_compatible_with_interval(sec: float, interval: float) -> bool:
    if interval <= 0:
        return False
    rows = float(sec) / interval
    rows_i = int(round(rows))
    return abs(rows - rows_i) <= 1e-9 and rows_i >= 1


def _filter_timed_config_items(items: list[Any], interval: float) -> list[Any]:
    kept: list[Any] = []
    for item in items:
        if isinstance(item, dict) and item.get("seconds") is not None:
            try:
                sec = float(item["seconds"])
            except (TypeError, ValueError):
                kept.append(item)
                continue
            if sec > 0 and not _seconds_compatible_with_interval(sec, interval):
                continue
        kept.append(item)
    return kept


def _derived_output_compatible(output: dict[str, Any], interval: float) -> bool:
    terms = output.get("terms")
    if not isinstance(terms, list):
        return True
    for term in terms:
        if not isinstance(term, dict) or term.get("seconds") is None:
            continue
        try:
            sec = float(term["seconds"])
        except (TypeError, ValueError):
            continue
        if sec > 0 and not _seconds_compatible_with_interval(sec, interval):
            return False
    return True


def prune_transformation_config_for_interval(
    config: dict[str, Any] | None,
    sample_interval_sec: float,
) -> dict[str, Any]:
    """Drop horizons/windows whose seconds are not exact multiples of the sample interval."""
    cfg = normalize_transformation_config(config)
    try:
        interval = float(sample_interval_sec)
    except (TypeError, ValueError):
        return cfg
    if interval <= 0:
        return cfg

    kept_stages: list[dict[str, Any]] = []
    for raw in list(cfg.get("transformations") or []):
        if not isinstance(raw, dict):
            continue
        stage = dict(raw)
        if not stage.get("enabled"):
            kept_stages.append(stage)
            continue
        params = dict(stage.get("params") or {})
        params["sample_interval_sec"] = interval

        for key in ("horizons", "windows", "periods"):
            items = params.get(key)
            if isinstance(items, list) and items:
                filtered = _filter_timed_config_items(items, interval)
                if not filtered:
                    continue
                params[key] = filtered

        lag_seconds = params.get("lag_seconds")
        if isinstance(lag_seconds, list):
            filtered_lags: list[Any] = []
            for raw_sec in lag_seconds:
                try:
                    sec = float(raw_sec)
                except (TypeError, ValueError):
                    continue
                if _seconds_compatible_with_interval(sec, interval):
                    filtered_lags.append(raw_sec)
            if lag_seconds and not filtered_lags:
                continue
            params["lag_seconds"] = filtered_lags

        outputs = params.get("outputs")
        if isinstance(outputs, list) and outputs:
            kept_outputs: list[Any] = []
            for out in outputs:
                if isinstance(out, str) and str(out).strip():
                    # rolling_ohlc uses string output names, not derived-style dicts.
                    kept_outputs.append(str(out).strip())
                    continue
                if not isinstance(out, dict):
                    continue
                if not _derived_output_compatible(out, interval):
                    continue
                kept_outputs.append(out)
            if not kept_outputs:
                continue
            params["outputs"] = kept_outputs

        stage["params"] = params
        kept_stages.append(stage)

    return {
        "transformation_pipeline_version": int(
            cfg.get("transformation_pipeline_version") or TRANSFORMATION_PIPELINE_VERSION
        ),
        "transformations": kept_stages,
    }


__all__ = [
    "TRANSFORMATION_PIPELINE_VERSION",
    "config_for_metadata",
    "default_transformation_config",
    "merge_transformation_configs",
    "normalize_transformation_config",
    "prune_transformation_config_for_interval",
]
