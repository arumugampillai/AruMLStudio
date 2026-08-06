"""Read API for strategy registry."""

from __future__ import annotations

from typing import Any

from .service import (
    compare_strategy_versions,
    get_default_template,
    get_strategy_detail,
    get_strategy_version,
    list_strategies,
)

__all__ = [
    "compare_strategy_versions",
    "get_default_template",
    "get_strategy_detail",
    "get_strategy_version",
    "list_strategies",
]
