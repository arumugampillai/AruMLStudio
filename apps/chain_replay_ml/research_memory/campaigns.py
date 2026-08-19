"""Research Campaign Lifecycle, Budget Quota & Execution Linker (Phase 4D.6).

Manages the lifecycle of autonomous and semi-autonomous research campaigns, enforcing
experiment quotas, resource budgets, deterministic experiment linking, and transactional
crash-safe state transitions in `<data_dir>/analysis.db`.

Invariants:
1. Context-Scoped Boundaries: Every campaign targets a specific `ModelContextKey`.
2. Strict Lifecycle Transitions: Enforces valid state progression (CREATED -> RUNNING -> COMPLETED).
   Terminal states (COMPLETED, FAILED, CANCELLED) cannot be re-opened.
3. Transactional Quota Allocation: Atomically allocates experiment slots to prevent race
   conditions across concurrent workers.
4. Canonical Experiment Identity: Experiments are linked strictly via `signature_hash`
   from Phase 4D.2 (`experiment_signatures`).
5. Research Memory Principle: Append-only execution history. Crashes and failures are recorded
   without destroying historical experiment evidence.
"""

from __future__ import annotations

from datetime import datetime, timezone
import enum
import json
import sqlite3
from typing import Any

from .db import connect_analysis_db, init_analysis_db
from .signature import check_experiment_exists


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class CampaignStatus(str, enum.Enum):
    """Lifecycle states for research campaigns."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


# Valid state machine transitions
_VALID_TRANSITIONS: dict[str, set[str]] = {
    CampaignStatus.CREATED.value: {CampaignStatus.RUNNING.value, CampaignStatus.CANCELLED.value},
    CampaignStatus.RUNNING.value: {CampaignStatus.PAUSED.value, CampaignStatus.COMPLETED.value, CampaignStatus.FAILED.value, CampaignStatus.CANCELLED.value},
    CampaignStatus.PAUSED.value: {CampaignStatus.RUNNING.value, CampaignStatus.CANCELLED.value},
    CampaignStatus.COMPLETED.value: set(),   # Terminal
    CampaignStatus.FAILED.value: set(),      # Terminal
    CampaignStatus.CANCELLED.value: set(),   # Terminal
}


def create_campaign(
    data_dir: str,
    *,
    context_key: str,
    campaign_name: str | None = None,
    description: str | None = None,
    max_experiments_limit: int = 100,
    max_duration_seconds: float = 14400.0,
    memory_limit_mb: int = 8192,
    total_planned: int = 0,
    ranking_policy_version: str = "ROB_POLICY_v1.0",
    campaign_id: str | None = None,
) -> str:
    """Create a new research campaign record in `<data_dir>/analysis.db`.
    
    Returns:
        The campaign_id.
    """
    init_analysis_db(data_dir)
    c_id = campaign_id or f"CAMP_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')[:19]}_{context_key[-8:]}"
    c_name = str(campaign_name or f"Campaign {c_id}").strip()
    now_iso = _utc_now_iso()

    conn = connect_analysis_db(data_dir)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO research_campaigns (
                    campaign_id, campaign_name, context_key, description,
                    ranking_policy_version, status, max_experiments_limit,
                    max_duration_seconds, memory_limit_mb, total_planned,
                    completed_count, skipped_duplicate_count, failed_count,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 0, ?, ?);
                """,
                (
                    c_id,
                    c_name,
                    str(context_key).strip(),
                    str(description or "").strip() or None,
                    str(ranking_policy_version).strip(),
                    CampaignStatus.CREATED.value,
                    int(max_experiments_limit),
                    float(max_duration_seconds),
                    int(memory_limit_mb),
                    int(total_planned),
                    now_iso,
                    now_iso,
                ),
            )
        return c_id
    finally:
        conn.close()


def get_campaign(data_dir: str, campaign_id: str) -> dict[str, Any] | None:
    """Retrieve full campaign record by ID."""
    init_analysis_db(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        row = conn.execute(
            "SELECT * FROM research_campaigns WHERE campaign_id = ?;",
            (str(campaign_id).strip(),),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def list_campaigns_for_context(
    data_dir: str,
    context_key: str,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """List all campaigns targeting a specific ModelContextKey sorted by creation time descending."""
    init_analysis_db(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        rows = conn.execute(
            """
            SELECT * FROM research_campaigns
            WHERE context_key = ?
            ORDER BY created_at DESC
            LIMIT ?;
            """,
            (str(context_key).strip(), int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _transition_campaign_status(
    data_dir: str,
    campaign_id: str,
    target_status: CampaignStatus,
    *,
    termination_reason: str | None = None,
    error_message: str | None = None,
) -> bool:
    """Atomically validate and execute a lifecycle state transition."""
    init_analysis_db(data_dir)
    now_iso = _utc_now_iso()
    t_val = target_status.value

    conn = connect_analysis_db(data_dir)
    try:
        with conn:
            row = conn.execute(
                "SELECT status, start_time FROM research_campaigns WHERE campaign_id = ?;",
                (str(campaign_id).strip(),),
            ).fetchone()

            if not row:
                return False

            curr_status = row["status"]
            allowed = _VALID_TRANSITIONS.get(curr_status, set())
            if t_val not in allowed:
                raise ValueError(
                    f"Invalid campaign transition: Cannot move '{campaign_id}' from '{curr_status}' to '{t_val}'."
                )

            # Update fields based on transition
            if t_val == CampaignStatus.RUNNING.value and not row["start_time"]:
                conn.execute(
                    """
                    UPDATE research_campaigns
                    SET status = ?, start_time = ?, updated_at = ?
                    WHERE campaign_id = ?;
                    """,
                    (t_val, now_iso, now_iso, campaign_id),
                )
            elif t_val in (CampaignStatus.COMPLETED.value, CampaignStatus.FAILED.value, CampaignStatus.CANCELLED.value):
                reason = termination_reason or (f"FAILED: {error_message}" if error_message else t_val)
                conn.execute(
                    """
                    UPDATE research_campaigns
                    SET status = ?, end_time = ?, termination_reason = ?, updated_at = ?
                    WHERE campaign_id = ?;
                    """,
                    (t_val, now_iso, reason, now_iso, campaign_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE research_campaigns
                    SET status = ?, updated_at = ?
                    WHERE campaign_id = ?;
                    """,
                    (t_val, now_iso, campaign_id),
                )
        return True
    finally:
        conn.close()


def start_campaign(data_dir: str, campaign_id: str) -> bool:
    """Transition campaign from CREATED/PAUSED to RUNNING."""
    return _transition_campaign_status(data_dir, campaign_id, CampaignStatus.RUNNING)


def pause_campaign(data_dir: str, campaign_id: str) -> bool:
    """Transition campaign from RUNNING to PAUSED."""
    return _transition_campaign_status(data_dir, campaign_id, CampaignStatus.PAUSED)


def resume_campaign(data_dir: str, campaign_id: str) -> bool:
    """Transition campaign from PAUSED to RUNNING."""
    return _transition_campaign_status(data_dir, campaign_id, CampaignStatus.RUNNING)


def complete_campaign(
    data_dir: str,
    campaign_id: str,
    *,
    termination_reason: str = "QUOTA_REACHED",
) -> bool:
    """Transition campaign to COMPLETED terminal state."""
    return _transition_campaign_status(
        data_dir, campaign_id, CampaignStatus.COMPLETED, termination_reason=termination_reason
    )


def fail_campaign(
    data_dir: str,
    campaign_id: str,
    error_message: str,
) -> bool:
    """Transition campaign to FAILED terminal state."""
    return _transition_campaign_status(
        data_dir, campaign_id, CampaignStatus.FAILED, error_message=error_message
    )


def cancel_campaign(
    data_dir: str,
    campaign_id: str,
    *,
    reason: str = "CANCELLED_BY_USER",
) -> bool:
    """Transition campaign to CANCELLED terminal state."""
    return _transition_campaign_status(
        data_dir, campaign_id, CampaignStatus.CANCELLED, termination_reason=reason
    )


def allocate_experiment_slot(
    data_dir: str,
    campaign_id: str,
) -> tuple[bool, int, str | None]:
    """Atomically check quota and allocate the next unique trial index.
    
    Uses single-statement atomic UPDATE ... RETURNING to guarantee thread/process safety.
    
    Returns:
        (is_allocated, next_trial_index, error_or_rejection_reason)
    """
    init_analysis_db(data_dir)
    now_iso = _utc_now_iso()
    c_id = str(campaign_id).strip()

    conn = connect_analysis_db(data_dir)
    try:
        with conn:
            # Single-statement atomic conditional allocation
            cursor = conn.execute(
                """
                UPDATE research_campaigns
                SET total_planned = total_planned + 1, updated_at = ?
                WHERE campaign_id = ?
                  AND status = 'RUNNING'
                  AND (completed_count + failed_count) < max_experiments_limit
                RETURNING total_planned;
                """,
                (now_iso, c_id),
            )
            row = cursor.fetchone()
            if row:
                return (True, int(row["total_planned"]), None)

            # If no row returned, query why
            diag = conn.execute(
                "SELECT status, max_experiments_limit, completed_count, failed_count FROM research_campaigns WHERE campaign_id = ?;",
                (c_id,),
            ).fetchone()

            if not diag:
                return (False, 0, f"Campaign '{c_id}' not found.")
            if diag["status"] != CampaignStatus.RUNNING.value:
                return (False, 0, f"Campaign '{c_id}' is not RUNNING (current status: {diag['status']}).")
            
            consumed = int(diag["completed_count"]) + int(diag["failed_count"])
            limit = int(diag["max_experiments_limit"])
            return (False, consumed, f"Campaign quota exhausted ({consumed}/{limit}).")
    finally:
        conn.close()


def link_experiment_to_campaign(
    data_dir: str,
    *,
    campaign_id: str,
    trial_index: int,
    signature_hash: str,
    model_name: str | None = None,
    execution_status: str = "SUCCESS",
    elapsed_sec: float | None = None,
    memory_peak_mb: float | None = None,
    error_message: str | None = None,
) -> int:
    """Link an executed experiment signature to a campaign and atomically update counters.
    
    Returns:
        The autoincrement `campaign_exp_id`.
    """
    init_analysis_db(data_dir)
    now_iso = _utc_now_iso()
    norm_status = str(execution_status).upper().strip()
    c_id = str(campaign_id).strip()

    conn = connect_analysis_db(data_dir)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO campaign_experiments (
                    campaign_id, trial_index, signature_hash, model_name,
                    execution_status, elapsed_sec, memory_peak_mb, error_message,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    c_id,
                    int(trial_index),
                    str(signature_hash).strip(),
                    str(model_name).strip() if model_name else None,
                    norm_status,
                    float(elapsed_sec) if elapsed_sec is not None else None,
                    float(memory_peak_mb) if memory_peak_mb is not None else None,
                    str(error_message).strip() if error_message else None,
                    now_iso,
                    now_iso,
                ),
            )
            camp_exp_id = cursor.lastrowid or 0

            # Atomically increment appropriate campaign counters
            if norm_status == "SUCCESS":
                conn.execute(
                    """
                    UPDATE research_campaigns
                    SET completed_count = completed_count + 1, updated_at = ?
                    WHERE campaign_id = ?;
                    """,
                    (now_iso, c_id),
                )
            elif norm_status == "SKIPPED_DUPLICATE":
                conn.execute(
                    """
                    UPDATE research_campaigns
                    SET skipped_duplicate_count = skipped_duplicate_count + 1, updated_at = ?
                    WHERE campaign_id = ?;
                    """,
                    (now_iso, c_id),
                )
            elif norm_status == "FAILED":
                conn.execute(
                    """
                    UPDATE research_campaigns
                    SET failed_count = failed_count + 1, updated_at = ?
                    WHERE campaign_id = ?;
                    """,
                    (now_iso, c_id),
                )

            return camp_exp_id
    finally:
        conn.close()


def get_campaign_experiments(
    data_dir: str,
    campaign_id: str,
) -> list[dict[str, Any]]:
    """Retrieve all linked experiments for a campaign ordered by trial index."""
    init_analysis_db(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        rows = conn.execute(
            """
            SELECT * FROM campaign_experiments
            WHERE campaign_id = ?
            ORDER BY trial_index ASC;
            """,
            (str(campaign_id).strip(),),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
