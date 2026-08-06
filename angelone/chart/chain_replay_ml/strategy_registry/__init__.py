"""Strategy Registry — Phase 2 foundation (versioned immutable strategy profiles)."""

from __future__ import annotations

from .registry import (
    compare_strategy_versions,
    get_default_template,
    get_strategy_detail,
    get_strategy_version,
    list_strategies,
)
from .service import (
    archive_strategy,
    clone_strategy_version,
    create_strategy,
    create_strategy_version,
    set_champion_version,
)
from .store import StrategyRegistryStore

__all__ = [
    "StrategyRegistryStore",
    "archive_strategy",
    "clone_strategy_version",
    "compare_strategy_versions",
    "create_strategy",
    "create_strategy_version",
    "get_default_template",
    "get_strategy_detail",
    "get_strategy_version",
    "list_strategies",
    "set_champion_version",
]
