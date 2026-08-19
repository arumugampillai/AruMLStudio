"""Champion & Challenger Transition History (Phase 4D.6).

Maintains an immutable, append-only historical audit trail of champion transitions
and promotion evidence in `<data_dir>/analysis.db`.

Invariants:
1. History ≠ Current State: This module stores immutable historical promotion evidence.
   Current operational champion state remains governed by `.lifecycle_registry.db`.
2. Context-Scoped History: Champion histories are strictly partitioned by `ModelContextKey`.
   Trend (R001) transitions never collide with Sideways (R002) transitions.
3. Append-Only Immutability: Previous transitions are never overwritten or deleted.
4. Human Governance Boundary: Records human approval evidence (`promoted_by`, `promotion_reason`).
   Does NOT automatically modify production trading configurations.
"""

from __future__ import annotations

from datetime import datetime, timezone
import sqlite3
from typing import Any

from .db import connect_analysis_db, init_analysis_db


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def record_champion_transition(
    data_dir: str,
    *,
    context_key: str,
    new_champion_name: str,
    new_robustness_score: float,
    previous_champion_name: str | None = None,
    previous_robustness_score: float | None = None,
    ranking_policy_version: str = "ROB_POLICY_v1.0",
    promoted_by: str = "HUMAN_RESEARCHER",
    promotion_reason: str = "Human researcher approved champion candidate proposal",
    transition_timestamp: str | None = None,
) -> int:
    """Append a new champion transition record into `champion_history`.
    
    Returns:
        The autoincrement `transition_id`.
    """
    init_analysis_db(data_dir)
    ts = transition_timestamp or _utc_now_iso()
    new_score = float(new_robustness_score)
    prev_score = float(previous_robustness_score) if previous_robustness_score is not None else None
    delta = round(new_score - prev_score, 4) if prev_score is not None else 0.0

    conn = connect_analysis_db(data_dir)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO champion_history (
                    context_key, previous_champion_name, new_champion_name,
                    previous_robustness_score, new_robustness_score, score_delta,
                    ranking_policy_version, promoted_by, promotion_reason,
                    transition_timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(context_key).strip(),
                    str(previous_champion_name).strip() if previous_champion_name else None,
                    str(new_champion_name).strip(),
                    prev_score,
                    new_score,
                    delta,
                    str(ranking_policy_version).strip(),
                    str(promoted_by).strip(),
                    str(promotion_reason).strip(),
                    ts,
                ),
            )
            return cursor.lastrowid or 0
    finally:
        conn.close()


def get_champion_history_for_context(
    data_dir: str,
    context_key: str,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Retrieve all historical champion transitions for a ModelContextKey sorted newest first."""
    init_analysis_db(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        rows = conn.execute(
            """
            SELECT * FROM champion_history
            WHERE context_key = ?
            ORDER BY transition_timestamp DESC, transition_id DESC
            LIMIT ?;
            """,
            (str(context_key).strip(), int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_latest_champion_transition(
    data_dir: str,
    context_key: str,
) -> dict[str, Any] | None:
    """Retrieve the most recent champion transition record for a ModelContextKey."""
    history = get_champion_history_for_context(data_dir, context_key, limit=1)
    return history[0] if history else None


def reconstruct_champion_at_timestamp(
    data_dir: str,
    context_key: str,
    timestamp_iso: str,
) -> dict[str, Any] | None:
    """Historical time-travel query: reconstruct which champion was in effect at a specific timestamp."""
    init_analysis_db(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        row = conn.execute(
            """
            SELECT * FROM champion_history
            WHERE context_key = ? AND transition_timestamp <= ?
            ORDER BY transition_timestamp DESC, transition_id DESC
            LIMIT 1;
            """,
            (str(context_key).strip(), str(timestamp_iso).strip()),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()
