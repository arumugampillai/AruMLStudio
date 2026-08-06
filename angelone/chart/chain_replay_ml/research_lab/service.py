"""Research Lab service layer."""

from __future__ import annotations

from typing import Any

from .compare import compare_strategy_run_trades, compare_strategy_runs
from .leaderboard import build_leaderboard, build_research_summary
from .matrix import build_model_strategy_grid, build_research_matrix
from .sessions_store import ResearchSessionStore


def get_matrix(data_dir: str, *, filters: dict[str, Any] | None = None, limit: int = 200) -> dict[str, Any]:
    matrix = build_research_matrix(data_dir, filters=filters, limit=limit)
    grid = build_model_strategy_grid(matrix)
    matrix["grid"] = grid.get("grid")
    return matrix


def get_leaderboard(
    data_dir: str,
    *,
    mode: str = "best_composite",
    filters: dict[str, Any] | None = None,
    limit: int = 25,
) -> dict[str, Any]:
    return build_leaderboard(data_dir, mode=mode, filters=filters, limit=limit)


def get_summary(data_dir: str) -> dict[str, Any]:
    return build_research_summary(data_dir)


def create_research_session(data_dir: str, *, title: str, notes: str | None = None) -> dict[str, Any]:
    with ResearchSessionStore(data_dir) as store:
        session = store.create_session(title=title, notes=notes)
    return {"ok": True, "session": session}


def update_research_session(data_dir: str, session_id: str, doc: dict[str, Any]) -> dict[str, Any]:
    with ResearchSessionStore(data_dir) as store:
        session = store.update_session(session_id, doc)
    if not session:
        return {"ok": False, "error": "session not found"}
    return {"ok": True, "session": session}


def list_research_sessions(data_dir: str, *, limit: int = 50) -> dict[str, Any]:
    with ResearchSessionStore(data_dir) as store:
        sessions = store.list_sessions(limit=limit)
    return {"ok": True, "sessions": sessions}


def get_research_session(data_dir: str, session_id: str) -> dict[str, Any]:
    with ResearchSessionStore(data_dir) as store:
        session = store.get_session(session_id)
    if not session:
        return {"ok": False, "error": "session not found"}
    matrix = build_research_matrix(
        data_dir,
        filters={"prediction_run_id": None},
        limit=500,
    )
    run_ids = set(session.get("strategy_run_ids") or [])
    pred_ids = set(session.get("prediction_run_ids") or [])
    session_rows = [
        r for r in matrix.get("rows") or []
        if r.get("strategy_run_id") in run_ids or r.get("prediction_run_id") in pred_ids
    ]
    return {"ok": True, "session": session, "linked_rows": session_rows}
