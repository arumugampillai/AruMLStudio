"""Champion & Challenger Transition History (Phase 4D.6 & Authoritative AruMLStudio Governance).

Maintains an immutable, append-only historical audit trail of champion transitions
and promotion evidence in `<data_dir>/analysis.db`.

Authoritative Sources of Truth:
1. Model Artifacts: `<data_dir>/models/<safe_model_name>/`
2. Currently Active Model Pointer: `<data_dir>/models/.active_model.json`
3. Research Memory: `<data_dir>/analysis.db`
4. Research Champion History: `<data_dir>/analysis.db -> champion_history`
5. Feature Governance: `feature_registry_store.json` + `apps/feature_recommendation_evidence.db`

Invariants:
1. Context-Scoped Governance: Champion histories are strictly partitioned by `ModelContextKey`.
   Trend (R001) transitions never collide with Sideways (R002) transitions.
2. Append-Only Immutability: Previous transitions are never overwritten or deleted.
3. Human Governance Boundary: Records human approval evidence (`promoted_by`, `promotion_reason`).
   Does NOT automatically modify production trading configurations.
"""

from __future__ import annotations

from datetime import datetime, timezone
import sqlite3
from typing import Any

from .db import connect_analysis_db, init_analysis_db

# Memory cache for challenger model names associated with context transitions
_CHALLENGER_CACHE: dict[tuple[str, str], str] = {}


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


def get_champion_for_context(
    data_dir: str,
    context_key: Any,
) -> dict[str, Any] | None:
    """Retrieve active champion metadata for a canonical context key from analysis.db."""
    key_str = context_key.canonical_key_str() if hasattr(context_key, "canonical_key_str") else str(context_key).strip()
    latest = get_latest_champion_transition(data_dir, key_str)
    if latest:
        challenger = _CHALLENGER_CACHE.get((data_dir, key_str))
        return {
            "model_id": key_str,
            "context_key": key_str,
            "champion_model_name": latest["new_champion_name"],
            "current_model_name": latest["new_champion_name"],
            "challenger_model_name": challenger,
            "champion_robustness_score": latest["new_robustness_score"],
            "updated_on": latest["transition_timestamp"],
            "promoted_at": latest["transition_timestamp"],
            "promoted_by": latest["promoted_by"],
            "promotion_reason": latest["promotion_reason"],
            "ranking_policy_version": latest["ranking_policy_version"],
        }
    return None


def set_champion_for_context(
    data_dir: str,
    context_key: Any,
    champion_model_name: str,
    challenger_model_name: str | None = None,
    robustness_score: float = 75.0,
    promoted_by: str = "HUMAN_RESEARCHER",
    promotion_reason: str = "Human researcher approved champion candidate proposal",
) -> None:
    """Record champion transition for context in analysis.db champion_history."""
    key_str = context_key.canonical_key_str() if hasattr(context_key, "canonical_key_str") else str(context_key).strip()
    if challenger_model_name:
        _CHALLENGER_CACHE[(data_dir, key_str)] = challenger_model_name
    elif (data_dir, key_str) in _CHALLENGER_CACHE:
        _CHALLENGER_CACHE.pop((data_dir, key_str), None)

    prev = get_champion_for_context(data_dir, key_str)
    # If the latest champion is already this model, don't create a redundant duplicate entry
    if prev and prev.get("champion_model_name") == champion_model_name:
        return

    prev_name = prev["champion_model_name"] if prev else None
    prev_score = prev["champion_robustness_score"] if prev else None
    record_champion_transition(
        data_dir,
        context_key=key_str,
        new_champion_name=champion_model_name,
        new_robustness_score=robustness_score,
        previous_champion_name=prev_name,
        previous_robustness_score=prev_score,
        promoted_by=promoted_by,
        promotion_reason=promotion_reason,
    )


def list_context_champions(data_dir: str) -> list[dict[str, Any]]:
    """List all latest registered context champions across all context keys from analysis.db."""
    init_analysis_db(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        rows = conn.execute(
            """
            SELECT h1.* FROM champion_history h1
            JOIN (
                SELECT context_key, MAX(transition_timestamp) as max_ts, MAX(transition_id) as max_id
                FROM champion_history
                GROUP BY context_key
            ) h2 ON h1.context_key = h2.context_key 
                AND h1.transition_timestamp = h2.max_ts 
                AND h1.transition_id = h2.max_id
            ORDER BY h1.transition_timestamp DESC;
            """
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            key_str = d["context_key"]
            challenger = _CHALLENGER_CACHE.get((data_dir, key_str))
            out.append({
                "model_id": key_str,
                "context_key": key_str,
                "champion_model_name": d["new_champion_name"],
                "current_model_name": d["new_champion_name"],
                "challenger_model_name": challenger,
                "champion_robustness_score": d["new_robustness_score"],
                "updated_on": d["transition_timestamp"],
                "promoted_at": d["transition_timestamp"],
                "promoted_by": d["promoted_by"],
                "promotion_reason": d["promotion_reason"],
                "ranking_policy_version": d["ranking_policy_version"],
            })
        return out
    finally:
        conn.close()
