"""Strategy registry and discovery for the Outcome Label Engine."""

from __future__ import annotations

from .base import OutcomeLabelStrategy
from .types import ProblemType, StrategyCapabilities, StrategyMetadata

_REGISTRY: dict[str, OutcomeLabelStrategy] = {}


class StrategyAlreadyRegisteredError(ValueError):
    pass


class StrategyNotFoundError(KeyError):
    pass


def clear_registry() -> None:
    """Remove all registered strategies (tests only)."""
    _REGISTRY.clear()


def register_strategy(
    strategy: OutcomeLabelStrategy,
    *,
    replace: bool = False,
) -> OutcomeLabelStrategy:
    """Register a strategy instance by ``metadata.strategy_id``."""
    sid = strategy.metadata.strategy_id
    if not sid:
        raise ValueError("strategy_id must be non-empty")
    if strategy.capabilities.strategy_id != sid:
        raise ValueError(
            f"capabilities.strategy_id={strategy.capabilities.strategy_id!r} "
            f"must match metadata.strategy_id={sid!r}"
        )
    if sid in _REGISTRY and not replace:
        raise StrategyAlreadyRegisteredError(
            f"strategy already registered: {sid!r}"
        )
    _REGISTRY[sid] = strategy
    return strategy


def unregister_strategy(strategy_id: str) -> None:
    _REGISTRY.pop(strategy_id, None)


def get_strategy(strategy_id: str) -> OutcomeLabelStrategy:
    try:
        return _REGISTRY[strategy_id]
    except KeyError as exc:
        raise StrategyNotFoundError(strategy_id) from exc


def list_strategy_ids() -> list[str]:
    return sorted(_REGISTRY.keys())


def list_strategies() -> list[OutcomeLabelStrategy]:
    return [_REGISTRY[k] for k in list_strategy_ids()]


def list_metadata() -> list[StrategyMetadata]:
    """Registry discovery for UI selectors (display_name / description / category)."""
    return [s.metadata for s in list_strategies()]


def filter_by_problem_type(
    problem_type: ProblemType | str,
) -> list[OutcomeLabelStrategy]:
    """Capability filtering: strategies that support the given Model Builder problem type."""
    want = str(problem_type)
    return [
        s
        for s in list_strategies()
        if want in s.capabilities.supported_problem_types
    ]


def filter_by_source(source_kind: str) -> list[OutcomeLabelStrategy]:
    return [
        s
        for s in list_strategies()
        if source_kind in s.capabilities.supported_sources
    ]


def discover_for_ui(
    *,
    problem_type: ProblemType | str | None = None,
    source_kind: str | None = None,
) -> list[StrategyMetadata]:
    """Filtered metadata list for Model Builder / OLE run UI."""
    strategies = list_strategies()
    if problem_type is not None:
        want = str(problem_type)
        strategies = [
            s for s in strategies if want in s.capabilities.supported_problem_types
        ]
    if source_kind is not None:
        strategies = [
            s for s in strategies if source_kind in s.capabilities.supported_sources
        ]
    return [s.metadata for s in strategies]


def get_capabilities(strategy_id: str) -> StrategyCapabilities:
    return get_strategy(strategy_id).capabilities
