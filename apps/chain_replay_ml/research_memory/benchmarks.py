"""Model Benchmark & Normalized Metrics Persistence (Phase 4D.3).

Manages the persistence and retrieval of evaluation events (`benchmark_runs`),
model-level performance scorecards (`model_benchmarks`), and normalized granular
fold/validation metrics (`benchmark_metrics`) in `<data_dir>/analysis.db`.

Invariants:
1. Canonical identity link: every model benchmark links to an `experiment_signatures` record.
2. Context-scoped isolation: benchmarks are strictly partitioned by `ModelContextKey`.
3. Normalized & extensible metrics: arbitrary fold-level, calibration, and latency metrics
   are stored without table schema changes.
4. Pure fact persistence: records empirical measurements without making automated promotion decisions.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3
from typing import Any

from .db import connect_analysis_db, init_analysis_db
from .signature import check_experiment_exists


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_benchmark_run(
    data_dir: str,
    *,
    context_key: str,
    benchmark_scope: str = "CONTEXT_LEADERBOARD",
    ranking_policy_version: str = "ROB_POLICY_v1.0",
    campaign_id: str | None = None,
    benchmark_run_id: str | None = None,
    evaluation_criteria: dict[str, Any] | None = None,
    incumbent_champion_name: str | None = None,
) -> str:
    """Create a new benchmark run evaluation record in analysis.db.
    
    Returns:
        The benchmark_run_id.
    """
    init_analysis_db(data_dir)
    run_id = benchmark_run_id or f"BM_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')[:19]}_{context_key[-8:]}"
    now_iso = _utc_now_iso()
    criteria_json = json.dumps(evaluation_criteria or {}, sort_keys=True)

    conn = connect_analysis_db(data_dir)
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO benchmark_runs (
                    benchmark_run_id, campaign_id, context_key, benchmark_scope,
                    ranking_policy_version, run_timestamp, models_evaluated_count,
                    top_model_name, incumbent_champion_name, evaluation_criteria_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 0, NULL, ?, ?, ?);
                """,
                (
                    run_id,
                    campaign_id,
                    str(context_key).strip(),
                    str(benchmark_scope).strip(),
                    str(ranking_policy_version).strip(),
                    now_iso,
                    incumbent_champion_name,
                    criteria_json,
                    now_iso,
                ),
            )
        return run_id
    finally:
        conn.close()


def record_model_benchmark(
    data_dir: str,
    *,
    benchmark_run_id: str,
    signature_hash: str,
    model_name: str,
    context_key: str,
    algorithm: str,
    dataset_name: str,
    feature_count: int,
    primary_metric_name: str,
    primary_metric_value: float,
    secondary_metric_value: float | None = None,
    wf_folds_count: int = 5,
    fold_metric_mean: float | None = None,
    fold_metric_std: float | None = None,
    fold_metric_min: float | None = None,
    fold_metric_max: float | None = None,
    worst_fold_drawdown: float | None = None,
    temporal_stability_score: float = 1.0,
    brier_score: float | None = None,
    log_loss: float | None = None,
    expected_calibration_error: float | None = None,
    training_time_sec: float | None = None,
    inference_latency_us: float | None = None,
    model_size_bytes: int | None = None,
    robustness_score: float | None = None,
    rank_in_context: int = 1,
    recommendation_status: str = "BENCHMARKED",
    ranking_policy_version: str = "ROB_POLICY_v1.0",
    granular_metrics: list[dict[str, Any]] | None = None,
) -> int:
    """Record a model's benchmark performance scorecard and associated normalized metrics.
    
    Returns:
        The autoincrement `benchmark_id`.
    """
    init_analysis_db(data_dir)
    now_iso = _utc_now_iso()

    # Default fold metrics if not explicitly passed
    f_mean = float(fold_metric_mean if fold_metric_mean is not None else primary_metric_value)
    f_std = float(fold_metric_std if fold_metric_std is not None else 0.0)
    f_min = float(fold_metric_min if fold_metric_min is not None else primary_metric_value)
    f_max = float(fold_metric_max if fold_metric_max is not None else primary_metric_value)
    rob_score = float(robustness_score if robustness_score is not None else primary_metric_value)

    conn = connect_analysis_db(data_dir)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO model_benchmarks (
                    benchmark_run_id, model_name, signature_hash, context_key,
                    algorithm, dataset_name, feature_count, ranking_policy_version,
                    primary_metric_name, primary_metric_value, secondary_metric_value,
                    wf_folds_count, fold_metric_mean, fold_metric_std, fold_metric_min,
                    fold_metric_max, worst_fold_drawdown, temporal_stability_score,
                    brier_score, log_loss, expected_calibration_error,
                    training_time_sec, inference_latency_us, model_size_bytes,
                    robustness_score, rank_in_context, recommendation_status,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(benchmark_run_id).strip(),
                    str(model_name).strip(),
                    str(signature_hash).strip(),
                    str(context_key).strip(),
                    str(algorithm).lower().strip(),
                    str(dataset_name).strip(),
                    int(feature_count),
                    str(ranking_policy_version).strip(),
                    str(primary_metric_name).strip(),
                    float(primary_metric_value),
                    float(secondary_metric_value) if secondary_metric_value is not None else None,
                    int(wf_folds_count),
                    f_mean,
                    f_std,
                    f_min,
                    f_max,
                    float(worst_fold_drawdown) if worst_fold_drawdown is not None else None,
                    float(temporal_stability_score),
                    float(brier_score) if brier_score is not None else None,
                    float(log_loss) if log_loss is not None else None,
                    float(expected_calibration_error) if expected_calibration_error is not None else None,
                    float(training_time_sec) if training_time_sec is not None else None,
                    float(inference_latency_us) if inference_latency_us is not None else None,
                    int(model_size_bytes) if model_size_bytes is not None else None,
                    rob_score,
                    int(rank_in_context),
                    str(recommendation_status).strip(),
                    now_iso,
                ),
            )
            benchmark_id = cursor.lastrowid

            # Update count in benchmark_runs
            conn.execute(
                """
                UPDATE benchmark_runs
                SET models_evaluated_count = models_evaluated_count + 1
                WHERE benchmark_run_id = ?;
                """,
                (benchmark_run_id,),
            )

            # Insert granular normalized metrics if provided
            if granular_metrics and benchmark_id:
                for m in granular_metrics:
                    m_name = str(m.get("metric_name") or "unknown").strip()
                    m_stage = str(m.get("metric_stage") or "TEST").upper().strip()
                    f_idx = m.get("fold_index")
                    m_val = float(m.get("metric_value", 0.0))
                    m_type = str(m.get("metric_type") or "SCALAR_FLOAT").upper().strip()
                    conn.execute(
                        """
                        INSERT INTO benchmark_metrics (
                            benchmark_id, metric_name, metric_stage, fold_index,
                            metric_value, metric_type, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?);
                        """,
                        (
                            benchmark_id,
                            m_name,
                            m_stage,
                            int(f_idx) if f_idx is not None else None,
                            m_val,
                            m_type,
                            now_iso,
                        ),
                    )

            return benchmark_id or 0
    finally:
        conn.close()


def record_benchmark_metrics(
    data_dir: str,
    *,
    benchmark_id: int,
    metrics: list[dict[str, Any]],
) -> int:
    """Batch insert normalized granular metric rows for an existing model benchmark."""
    init_analysis_db(data_dir)
    now_iso = _utc_now_iso()
    inserted = 0

    conn = connect_analysis_db(data_dir)
    try:
        with conn:
            for m in metrics:
                conn.execute(
                    """
                    INSERT INTO benchmark_metrics (
                        benchmark_id, metric_name, metric_stage, fold_index,
                        metric_value, metric_type, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        int(benchmark_id),
                        str(m.get("metric_name") or "unknown").strip(),
                        str(m.get("metric_stage") or "TEST").upper().strip(),
                        int(m["fold_index"]) if m.get("fold_index") is not None else None,
                        float(m.get("metric_value", 0.0)),
                        str(m.get("metric_type") or "SCALAR_FLOAT").upper().strip(),
                        now_iso,
                    ),
                )
                inserted += 1
        return inserted
    finally:
        conn.close()


# ==============================================================================
# Query Services
# ==============================================================================

def get_benchmark_run(data_dir: str, benchmark_run_id: str) -> dict[str, Any] | None:
    """Retrieve full benchmark run metadata by ID."""
    init_analysis_db(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        row = conn.execute(
            "SELECT * FROM benchmark_runs WHERE benchmark_run_id = ?;",
            (benchmark_run_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_model_benchmarks_for_context(
    data_dir: str,
    context_key: str,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Retrieve all model benchmark scorecards for a ModelContextKey sorted by robustness score descending."""
    init_analysis_db(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        rows = conn.execute(
            """
            SELECT * FROM model_benchmarks
            WHERE context_key = ?
            ORDER BY robustness_score DESC, primary_metric_value DESC
            LIMIT ?;
            """,
            (context_key, int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_model_benchmark_by_id(data_dir: str, benchmark_id: int) -> dict[str, Any] | None:
    """Retrieve a single model benchmark scorecard by its ID."""
    init_analysis_db(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        row = conn.execute(
            "SELECT * FROM model_benchmarks WHERE benchmark_id = ?;",
            (benchmark_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_benchmark_metrics(data_dir: str, benchmark_id: int) -> list[dict[str, Any]]:
    """Retrieve all granular normalized metrics for a given benchmark scorecard."""
    init_analysis_db(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        rows = conn.execute(
            """
            SELECT * FROM benchmark_metrics
            WHERE benchmark_id = ?
            ORDER BY metric_stage, fold_index, metric_name;
            """,
            (benchmark_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
