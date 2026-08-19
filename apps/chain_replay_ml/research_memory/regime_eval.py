"""Market Regime Performance & Cross-Regime Degradation Evaluation (Phase 4D.4).

Evaluates model performance across market regimes (R000..R007), calculates empirical
degradation percentages when models operate outside their native training regime,
and persists evaluation records into `<data_dir>/analysis.db`.

Invariants:
1. Strict State Distinction: Native regime evaluations, non-native stress tests, and
   missing evaluations are explicitly distinguished. Missing data is never recorded as zero.
2. Descriptive Degradation: Calculates empirical performance deltas without making
   automated promotion or rejection decisions.
3. Cryptographic Regime Lineage: Preserves `tested_regime_hash` from `regime_registry_store.json`.
4. Lineage Integrity: Every regime evaluation links to `experiment_signatures`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import sqlite3
from typing import Any

from chain_replay_ml.model_taxonomy.regime_registry_store import load_regime_registry
from .db import connect_analysis_db, init_analysis_db


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def calculate_regime_degradation(
    native_metric: float,
    tested_metric: float,
    *,
    higher_is_better: bool = True,
) -> float:
    """Calculate empirical percentage performance degradation.
    
    Degradation is positive when performance drops outside native regime, capped at 0.0 minimum.
    """
    try:
        nm = float(native_metric)
        tm = float(tested_metric)
    except (TypeError, ValueError):
        return 0.0

    if abs(nm) < 1e-12:
        return 0.0

    if higher_is_better:
        # e.g. ROC-AUC: 0.70 native -> 0.63 tested = (0.70 - 0.63) / 0.70 * 100 = 10.0% drop
        deg = ((nm - tm) / abs(nm)) * 100.0
    else:
        # e.g. RMSE: 0.02 native -> 0.03 tested = (0.03 - 0.02) / 0.02 * 100 = 50.0% error increase
        deg = ((tm - nm) / abs(nm)) * 100.0

    return round(max(0.0, deg), 4)


def record_regime_evaluation(
    data_dir: str,
    *,
    model_name: str,
    signature_hash: str,
    tested_regime_id: str,
    tested_regime_hash: str,
    is_native_regime: bool,
    sample_count: int,
    primary_metric: float,
    regime_degradation_pct: float | None = None,
) -> int:
    """Record a single regime evaluation row in `<data_dir>/analysis.db`.
    
    Returns:
        The autoincrement `eval_id`.
    """
    init_analysis_db(data_dir)
    now_iso = _utc_now_iso()
    deg_pct = float(regime_degradation_pct if regime_degradation_pct is not None else 0.0)

    conn = connect_analysis_db(data_dir)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO regime_evaluations (
                    model_name, signature_hash, tested_regime_id, tested_regime_hash,
                    is_native_regime, sample_count, primary_metric, regime_degradation_pct,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(model_name).strip(),
                    str(signature_hash).strip(),
                    str(tested_regime_id).upper().strip(),
                    str(tested_regime_hash).strip(),
                    1 if is_native_regime else 0,
                    int(sample_count),
                    float(primary_metric),
                    deg_pct,
                    now_iso,
                ),
            )
            return cursor.lastrowid or 0
    finally:
        conn.close()


def record_multi_regime_evaluations(
    data_dir: str,
    *,
    model_name: str,
    signature_hash: str,
    native_regime_id: str,
    native_metric: float,
    evaluations: list[dict[str, Any]],
    higher_is_better: bool = True,
) -> list[int]:
    """Atomically record multiple regime evaluation slices (native + non-native stress tests).
    
    Each item in `evaluations` should contain:
    - tested_regime_id (e.g. "R001", "R002")
    - tested_regime_hash (SHA-256 string)
    - sample_count (int > 0)
    - primary_metric (float)
    """
    init_analysis_db(data_dir)
    now_iso = _utc_now_iso()
    norm_native_id = str(native_regime_id).upper().strip()
    inserted_ids: list[int] = []

    conn = connect_analysis_db(data_dir)
    try:
        with conn:
            for ev in evaluations:
                rid = str(ev.get("tested_regime_id") or "").upper().strip()
                rhash = str(ev.get("tested_regime_hash") or "").strip()
                samples = int(ev.get("sample_count", 0))
                metric = float(ev.get("primary_metric", 0.0))

                if samples <= 0:
                    # Skip missing/empty evaluation slices
                    continue

                is_native = (rid == norm_native_id)
                if is_native:
                    deg = 0.0
                else:
                    deg = calculate_regime_degradation(native_metric, metric, higher_is_better=higher_is_better)

                cursor = conn.execute(
                    """
                    INSERT INTO regime_evaluations (
                        model_name, signature_hash, tested_regime_id, tested_regime_hash,
                        is_native_regime, sample_count, primary_metric, regime_degradation_pct,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        str(model_name).strip(),
                        str(signature_hash).strip(),
                        rid,
                        rhash,
                        1 if is_native else 0,
                        samples,
                        metric,
                        deg,
                        now_iso,
                    ),
                )
                inserted_ids.append(cursor.lastrowid or 0)
        return inserted_ids
    finally:
        conn.close()


def get_regime_evaluations_for_model(
    data_dir: str,
    signature_hash: str,
) -> list[dict[str, Any]]:
    """Retrieve all regime evaluation records for an experiment signature."""
    init_analysis_db(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        rows = conn.execute(
            """
            SELECT * FROM regime_evaluations
            WHERE signature_hash = ?
            ORDER BY is_native_regime DESC, tested_regime_id;
            """,
            (str(signature_hash).strip(),),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_regime_evaluations_for_regime(
    data_dir: str,
    tested_regime_id: str,
    *,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Retrieve all evaluations tested under a specific market regime across all models."""
    init_analysis_db(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        rows = conn.execute(
            """
            SELECT * FROM regime_evaluations
            WHERE tested_regime_id = ?
            ORDER BY primary_metric DESC
            LIMIT ?;
            """,
            (str(tested_regime_id).upper().strip(), int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def summarize_regime_feature_affinity(
    data_dir: str,
    regime_id: str,
) -> dict[str, Any]:
    """Compute descriptive empirical aggregation of feature population ratios for models evaluated in a regime."""
    init_analysis_db(data_dir)
    norm_rid = str(regime_id).upper().strip()

    conn = connect_analysis_db(data_dir)
    try:
        rows = conn.execute(
            """
            SELECT re.primary_metric, re.is_native_regime,
                   fse.total_features, fse.base_pipeline_count, fse.registry_feature_count,
                   fse.experimental_feature_count, fse.deprecated_feature_count,
                   fse.experimental_dependency_ratio
            FROM regime_evaluations re
            JOIN feature_set_evaluations fse ON re.signature_hash = fse.signature_hash
            WHERE re.tested_regime_id = ?;
            """,
            (norm_rid,),
        ).fetchall()

        if not rows:
            return {
                "regime_id": norm_rid,
                "models_evaluated_count": 0,
                "avg_total_features": 0.0,
                "avg_base_features_count": 0.0,
                "avg_registry_features_count": 0.0,
                "avg_experimental_features_count": 0.0,
                "avg_experimental_dependency_ratio": 0.0,
            }

        n = len(rows)
        avg_tot = sum(r["total_features"] for r in rows) / n
        avg_base = sum(r["base_pipeline_count"] for r in rows) / n
        avg_reg = sum(r["registry_feature_count"] for r in rows) / n
        avg_exp = sum(r["experimental_feature_count"] for r in rows) / n
        avg_exp_ratio = sum(r["experimental_dependency_ratio"] for r in rows) / n

        return {
            "regime_id": norm_rid,
            "models_evaluated_count": n,
            "avg_total_features": round(avg_tot, 2),
            "avg_base_features_count": round(avg_base, 2),
            "avg_registry_features_count": round(avg_reg, 2),
            "avg_experimental_features_count": round(avg_exp, 2),
            "avg_experimental_dependency_ratio": round(avg_exp_ratio, 4),
        }
    finally:
        conn.close()
