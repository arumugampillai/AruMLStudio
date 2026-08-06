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
