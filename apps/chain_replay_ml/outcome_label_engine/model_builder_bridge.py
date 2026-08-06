"""Strategy-agnostic helpers for Model Builder ↔ Outcome Label Engine.

Model Builder must not branch on strategy names. Use registry metadata,
``get_config_schema()``, and ``get_target_definitions()`` only.
"""

from __future__ import annotations

from typing import Any

from .base import OutcomeLabelStrategy
from .registry import (
    get_strategy,
    list_metadata,
    list_strategies,
)
from .types import (
    ProblemType,
    StrategyMetadata,
    defaults_from_config_schema,
    validate_config_against_schema,
)

# Map Create Model prediction_type → OLE problem types (capabilities filter).
_PRED_TO_PROBLEM: dict[str, frozenset[ProblemType]] = {
    "regression": frozenset({"regression"}),
    "binary": frozenset({"binary_classification"}),
    "classification": frozenset({"multiclass", "binary_classification"}),
}


def problem_types_for_prediction_type(prediction_type: str) -> frozenset[str]:
    pred = str(prediction_type or "").strip().lower()
    return frozenset(_PRED_TO_PROBLEM.get(pred, frozenset()))


def strategies_for_prediction_type(
    prediction_type: str | None = None,
) -> list[OutcomeLabelStrategy]:
    """Strategies whose capabilities intersect the selected prediction type."""
    strategies = list_strategies()
    if not prediction_type:
        return strategies
    wanted = problem_types_for_prediction_type(prediction_type)
    if not wanted:
        return strategies
    return [
        s
        for s in strategies
        if wanted.intersection(s.capabilities.supported_problem_types)
    ]


def metadata_for_prediction_type(
    prediction_type: str | None = None,
) -> list[StrategyMetadata]:
    return [s.metadata for s in strategies_for_prediction_type(prediction_type)]


def strategy_selector_rows(
    prediction_type: str | None = None,
) -> list[dict[str, str]]:
    """UI rows: strategy_id, display_name, description, category — no hardcoding."""
    rows: list[dict[str, str]] = []
    for meta in metadata_for_prediction_type(prediction_type):
        rows.append(
            {
                "strategy_id": meta.strategy_id,
                "display_name": meta.display_name,
                "description": meta.description,
                "category": meta.category,
                "version": meta.version,
            }
        )
    return rows


def config_schema_fields(strategy_id: str) -> list[tuple[str, dict[str, Any]]]:
    """Ordered (field_name, spec) from ``get_config_schema()`` for generic rendering."""
    strategy = get_strategy(strategy_id)
    schema = strategy.get_config_schema()
    return [(str(k), dict(v) if isinstance(v, dict) else {"default": v}) for k, v in schema.items()]


def default_params_for_strategy(strategy_id: str) -> dict[str, Any]:
    strategy = get_strategy(strategy_id)
    return defaults_from_config_schema(strategy.get_config_schema())


def merge_strategy_params(
    strategy_id: str,
    params: dict[str, Any] | None,
) -> dict[str, Any]:
    strategy = get_strategy(strategy_id)
    raw = dict(params or {})
    normalize = getattr(strategy, "normalize_config_params", None)
    if callable(normalize):
        raw = dict(normalize(raw))
    return validate_config_against_schema(raw, strategy.get_config_schema())


def preferred_target_column(
    strategy_id: str,
    available_columns: list[str],
) -> str | None:
    """Prefer ``primary_target`` from strategy definitions when present in the dataset."""
    strategy = get_strategy(strategy_id)
    defs = strategy.get_target_definitions()
    available = {str(c) for c in available_columns}
    if defs.primary_target in available:
        return defs.primary_target
    if defs.display_target and defs.display_target in available:
        return defs.display_target
    return None


def target_definitions_dict(strategy_id: str) -> dict[str, Any]:
    return get_strategy(strategy_id).get_target_definitions().to_dict()


def resolve_training_target(
    *,
    strategy_id: str,
    available_columns: list[str],
    current_target: str | None = None,
) -> str:
    """Pick the column Model Builder should train on (primary_target when possible)."""
    cols = [str(c) for c in available_columns if c]
    cur = str(current_target or "").strip()
    preferred = preferred_target_column(strategy_id, cols)
    if preferred:
        return preferred
    if cur and cur in cols:
        return cur
    return cols[0] if cols else cur


def default_strategy_id_for_prediction_type(prediction_type: str) -> str:
    """Prefer Fixed Horizon for regression; otherwise first matching registry entry."""
    pred = str(prediction_type or "").strip().lower()
    matching = strategies_for_prediction_type(pred)
    if not matching:
        metas = list_metadata()
        return metas[0].strategy_id if metas else "fixed_horizon"
    if pred == "regression":
        for s in matching:
            if s.metadata.strategy_id == "fixed_horizon":
                return "fixed_horizon"
    return matching[0].metadata.strategy_id
