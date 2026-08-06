"""Research Lab — Phase 4 model × strategy matrix and leaderboards."""

from __future__ import annotations

from .compare import compare_strategy_run_trades, compare_strategy_runs
from .leaderboard import LEADERBOARD_MODES, build_leaderboard, build_research_summary
from .matrix import build_model_strategy_grid, build_research_matrix
from .service import (
    create_research_session,
    get_leaderboard,
    get_matrix,
    get_research_session,
    get_summary,
    list_research_sessions,
    update_research_session,
)
from .sessions_store import ResearchSessionStore

__all__ = [
    "LEADERBOARD_MODES",
    "ResearchSessionStore",
    "build_leaderboard",
    "build_model_strategy_grid",
    "build_research_matrix",
    "build_research_summary",
    "compare_strategy_run_trades",
    "compare_strategy_runs",
    "create_research_session",
    "get_leaderboard",
    "get_matrix",
    "get_research_session",
    "get_summary",
    "list_research_sessions",
    "update_research_session",
]
