"""Resolve active / explicit / newest registry model name."""

from __future__ import annotations

from .paths import safe_model_name


def resolve_default_model_name(
    data_dir: str,
    model_name: str | None = None,
) -> str | None:
    """Active model, explicit name, or newest trained package (registry)."""
    from .registry import get_active_model, get_trained_model, list_trained_models

    preferred = str(model_name or "").strip()
    if preferred:
        safe = safe_model_name(preferred)
        if get_trained_model(data_dir, safe):
            return safe
        return None
    active = get_active_model(data_dir)
    if active and get_trained_model(data_dir, active):
        return active
    rows = list_trained_models(data_dir, lightweight=True)
    if rows:
        name = str(rows[0].get("model_name") or "").strip()
        return safe_model_name(name) if name else None
    return None
