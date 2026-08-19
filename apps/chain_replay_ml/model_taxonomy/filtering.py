"""Model Research Lab Taxonomy & Population Filtering Service (Phase 4C.4).

Provides pure, deterministic faceted filtering, context champion resolution, and
row formatting for the Model Research Lab UI.
"""

from __future__ import annotations

from typing import Any

from .adapter import resolve_model_metadata_or_legacy
from .enums import (
    BASELINE_REGIME_CATALOG,
    DEFAULT_REGIME_ID,
    DEFAULT_REGIME_NAME,
    ModelLifecycleStatus,
    ModelPopulationTier,
    RegimeScope,
    TaskType,
)
from .regime_registry_store import get_regime_record, list_regimes
from .specs import ModelContextKey, ModelMetadata


def filter_model_records(
    records: list[dict[str, Any]],
    *,
    task_type: str | None = None,
    regime_id: str | None = None,
    population: str | None = None,
    lifecycle_status: str | None = None,
) -> list[dict[str, Any]]:
    """Filter model dictionaries across the four orthogonal taxonomy dimensions.
    
    Pure function; does not mutate input records.
    """
    t_filter = str(task_type or "").strip().upper()
    r_filter = str(regime_id or "").strip().upper()
    p_filter = str(population or "").strip().upper()
    l_filter = str(lifecycle_status or "").strip().upper()

    # Normalize 'ALL' / blank
    if t_filter in ("ALL", "ALL TASKS", ""):
        t_filter = None
    if r_filter in ("ALL", "ALL REGIMES", ""):
        r_filter = None
    elif "—" in r_filter:
        r_filter = r_filter.split("—")[0].strip()
    elif " " in r_filter:
        r_filter = r_filter.split()[0].strip()

    if p_filter in ("ALL", "ALL POPULATIONS", ""):
        p_filter = None
    if l_filter in ("ALL", "ALL STATUSES", ""):
        l_filter = None

    filtered: list[dict[str, Any]] = []
    for r in records:
        meta = resolve_model_metadata_or_legacy(r, fallback_model_name=str(r.get("model_name") or r.get("name") or ""))

        # 1. Task Type filter
        if t_filter is not None:
            if meta.task.task_type.value != t_filter:
                continue

        # 2. Market Regime filter
        if r_filter is not None:
            if meta.regime.regime_id != r_filter and meta.regime.regime_name != r_filter:
                continue

        # 3. Population Tier filter
        if p_filter is not None:
            if meta.population.value != p_filter:
                continue

        # 4. Lifecycle Status filter
        if l_filter is not None:
            if meta.status.value != l_filter:
                continue

        filtered.append(r)

    return filtered


def format_model_taxonomy_display(
    model_row: dict[str, Any],
    *,
    champions_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Format model row with human-readable taxonomy badges, context key, and champion status."""
    meta = resolve_model_metadata_or_legacy(
        model_row,
        fallback_model_name=str(model_row.get("model_name") or model_row.get("name") or ""),
    )

    ctx_key_str = meta.context_key.canonical_key_str()
    name = str(model_row.get("model_name") or model_row.get("name") or "")

    # Champion status
    is_champ = False
    is_chall = False
    champ_badge = meta.population.value

    if champions_map and ctx_key_str in champions_map:
        champ_rec = champions_map[ctx_key_str]
        champ_name = str(champ_rec.get("champion_model_name") or champ_rec.get("current_model_name") or "")
        chall_name = str(champ_rec.get("challenger_model_name") or "")
        if name and name == champ_name:
            is_champ = True
            champ_badge = "👑 CHAMPION"
        elif name and name == chall_name:
            is_chall = True
            champ_badge = "⚔️ CHALLENGER"

    return {
        "model_name": name,
        "task_type": meta.task.task_type.value,
        "task_label": meta.task.task_type.value.replace("_", " ").title(),
        "regime_id": meta.regime.regime_id,
        "regime_name": meta.regime.regime_name,
        "regime_display": f"{meta.regime.regime_id} — {meta.regime.regime_name}",
        "population": meta.population.value,
        "population_badge": champ_badge,
        "status": meta.status.value,
        "lifecycle_label": meta.status.value.title(),
        "context_key": ctx_key_str,
        "is_champion": is_champ,
        "is_challenger": is_chall,
    }


def get_context_champions_map(data_dir: str) -> dict[str, dict[str, Any]]:
    """Build a mapping of context_key -> champion registry dict for fast lookup."""
    try:
        from ..research_memory.champion_history import list_context_champions
        rows = list_context_champions(data_dir)
        return {str(r.get("context_key") or r.get("model_id")): r for r in rows if r.get("context_key") or r.get("model_id")}
    except Exception:
        return {}
