"""Deterministic Experiment Identity & Canonical Deduplication (Phase 4D.2).

Provides pure canonicalization, deterministic SHA-256 experiment signature hashing,
and atomic check-and-register database services for `<data_dir>/analysis.db`.

Invariants:
1. Pure determinism: identical mathematical experiments always produce identical SHA-256 hashes.
2. Cross-platform floating point quantization (6 decimal places).
3. Non-semantic field exclusion: timestamps, execution paths, process IDs, and model names
   are never part of the signature hash.
4. Atomic database deduplication: concurrent workers safely resolve duplicate experiments.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import sqlite3
from typing import Any

from chain_replay_ml.model_taxonomy.enums import TaskType
from chain_replay_ml.model_taxonomy.specs import ModelContextKey
from .db import connect_analysis_db, init_analysis_db


def canonical_context_key(
    *,
    market: str,
    sampling_interval_sec: int,
    task_type: str,
    prediction_horizon: str,
    regime_id: str,
) -> str:
    """Format canonical context key string using ModelContextKey."""
    tt = TaskType.from_str(task_type)
    ck = ModelContextKey(
        market=str(market).upper().strip(),
        sampling_interval_sec=int(sampling_interval_sec),
        task_type=tt,
        prediction_horizon=str(prediction_horizon).strip(),
        regime_id=str(regime_id).strip(),
    )
    return ck.canonical_key_str()


def _normalize_value(val: Any) -> Any:
    """Recursively normalize data structures for deterministic serialization."""
    if isinstance(val, Enum):
        return val.value
    if isinstance(val, dict):
        return {str(k).strip(): _normalize_value(v) for k, v in sorted(val.items())}
    if isinstance(val, (list, tuple, set)):
        if all(isinstance(x, str) for x in val):
            # String collections (e.g. features) are sorted & deduplicated
            return sorted(list(set(str(x).strip() for x in val)))
        return [_normalize_value(x) for x in val]
    if isinstance(val, float):
        # Quantize to 6 decimal places to prevent float precision drift
        return round(val, 6)
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, (int, bool)) or val is None:
        return val
    return str(val).strip()


def canonicalize_json(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Convert a dictionary payload into normalized dictionary and deterministic canonical JSON string."""
    normalized = _normalize_value(payload)
    if not isinstance(normalized, dict):
        raise TypeError("Payload must normalize to a dictionary.")
    canonical_str = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return canonical_str, normalized


def compute_subcomponent_hash(data: Any) -> str:
    """Compute deterministic SHA-256 hash for a subcomponent (features, hyperparams, walk_forward)."""
    norm = _normalize_value(data)
    can_str = json.dumps(norm, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(can_str.encode("utf-8")).hexdigest()


def build_canonical_experiment_payload(
    *,
    market: str,
    sampling_interval_sec: int,
    task_type: str,
    prediction_horizon: str,
    regime_id: str,
    regime_definition_hash: str,
    dataset_snapshot_hash: str,
    features: list[str] | set[str] | tuple[str, ...],
    algorithm: str,
    hyperparameters: dict[str, Any] | None = None,
    walk_forward_config: dict[str, Any] | None = None,
    random_seed: int = 42,
    transformations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct the canonical experiment payload dictionary excluding non-semantic fields."""
    norm_features = sorted(list(set(str(f).strip() for f in (features or []))))
    norm_market = str(market or "NIFTY").upper().strip()
    norm_interval = int(sampling_interval_sec or 3)
    norm_task = str(task_type or "DIRECTION_CLASSIFIER").upper().strip()
    norm_horizon = str(prediction_horizon or "5m").strip()
    norm_regime = str(regime_id or "R000").upper().strip()
    norm_algo = str(algorithm or "xgboost").lower().strip()

    context_key = canonical_context_key(
        market=norm_market,
        sampling_interval_sec=norm_interval,
        task_type=norm_task,
        prediction_horizon=norm_horizon,
        regime_id=norm_regime,
    )

    payload: dict[str, Any] = {
        "context_key": context_key,
        "market": norm_market,
        "sampling_interval_sec": norm_interval,
        "task_type": norm_task,
        "prediction_horizon": norm_horizon,
        "regime_id": norm_regime,
        "regime_definition_hash": str(regime_definition_hash or "").strip(),
        "dataset_snapshot_hash": str(dataset_snapshot_hash or "").strip(),
        "features": norm_features,
        "algorithm": norm_algo,
        "hyperparameters": _normalize_value(hyperparameters or {}),
        "walk_forward_config": _normalize_value(walk_forward_config or {}),
        "random_seed": int(random_seed or 42),
    }

    if transformations:
        payload["transformations"] = _normalize_value(transformations)

    return payload


def compute_experiment_signature(
    experiment_spec: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    """Compute the deterministic (signature_hash, canonical_json, normalized_payload) for an experiment spec.
    
    The experiment_spec can contain either top-level experiment attributes or structured configuration.
    Non-semantic fields (e.g. timestamps, model_name, training_time) are cleanly excluded.
    """
    # Extract training / config / lifecycle fields safely
    cfg = experiment_spec.get("config") if isinstance(experiment_spec.get("config"), dict) else experiment_spec

    market = str(cfg.get("market") or experiment_spec.get("market") or "NIFTY")
    interval = int(cfg.get("sampling_interval_sec") or cfg.get("sample_interval_sec") or experiment_spec.get("sampling_interval_sec") or 3)
    task_type = str(cfg.get("task_type") or experiment_spec.get("task_type") or "DIRECTION_CLASSIFIER")
    horizon = str(cfg.get("prediction_horizon") or cfg.get("horizon") or experiment_spec.get("prediction_horizon") or "5m")
    
    # Regime
    reg_dict = experiment_spec.get("regime") if isinstance(experiment_spec.get("regime"), dict) else {}
    regime_id = str(reg_dict.get("regime_id") or cfg.get("regime_id") or experiment_spec.get("regime_id") or "R000")
    regime_def_hash = str(reg_dict.get("definition_hash") or reg_dict.get("regime_definition_hash") or experiment_spec.get("regime_definition_hash") or "")

    # Lineage hashes
    dataset_hash = str(cfg.get("dataset_snapshot_hash") or experiment_spec.get("dataset_snapshot_hash") or "")
    features = list(cfg.get("features") or experiment_spec.get("features") or [])
    algorithm = str(cfg.get("algorithm") or experiment_spec.get("algorithm") or "xgboost")
    
    # Hyperparameters
    hyperparams = cfg.get("hyperparameters") or cfg.get("algorithm_parameters") or experiment_spec.get("hyperparameters") or {}
    if not hyperparams:
        # Extract algorithm-specific parameters if flattened
        hyperparams = {
            k: v for k, v in cfg.items()
            if any(k.startswith(pfx) for pfx in ("xgb_", "lgb_", "cat_", "rf_", "lr_", "n_estimators", "max_depth", "learning_rate"))
        }

    # Walk forward config
    wf_config = cfg.get("walk_forward") or cfg.get("walk_forward_config") or (cfg.get("split") or {}).get("walk_forward") or experiment_spec.get("walk_forward_config") or {}
    random_seed = int(cfg.get("random_seed") or cfg.get("seed") or experiment_spec.get("random_seed") or 42)
    transformations = cfg.get("transformations") or experiment_spec.get("transformations") or None

    payload = build_canonical_experiment_payload(
        market=market,
        sampling_interval_sec=interval,
        task_type=task_type,
        prediction_horizon=horizon,
        regime_id=regime_id,
        regime_definition_hash=regime_def_hash,
        dataset_snapshot_hash=dataset_hash,
        features=features,
        algorithm=algorithm,
        hyperparameters=hyperparams,
        walk_forward_config=wf_config,
        random_seed=random_seed,
        transformations=transformations,
    )

    canonical_json, normalized_payload = canonicalize_json(payload)
    sig_hash = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return sig_hash, canonical_json, normalized_payload


# ==============================================================================
# Database Operations (Atomic Deduplication & Query Services)
# ==============================================================================

def check_experiment_exists(data_dir: str, signature_hash: str) -> dict[str, Any] | None:
    """Check if an experiment signature already exists in analysis.db without modifying state."""
    init_analysis_db(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        row = conn.execute(
            """
            SELECT signature_hash, context_key, market, sampling_interval_sec, task_type,
                   prediction_horizon, regime_id, regime_definition_hash, dataset_snapshot_hash,
                   feature_set_hash, algorithm, hyperparameters_hash, walk_forward_hash,
                   random_seed, canonical_payload_json, first_executed_at, execution_count,
                   latest_model_name
            FROM experiment_signatures
            WHERE signature_hash = ?;
            """,
            (signature_hash,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def register_or_get_experiment(
    data_dir: str,
    experiment_spec: dict[str, Any],
    *,
    model_name: str,
    executed_at: str | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Atomically register a new experiment signature or retrieve and update an existing one.
    
    Returns:
        (is_new, record_dict):
        - is_new = True: Experiment was unseen and newly registered.
        - is_new = False: Experiment already existed (duplicate detected; execution_count incremented).
    """
    init_analysis_db(data_dir)
    sig_hash, canonical_json, norm_payload = compute_experiment_signature(experiment_spec)
    now_iso = executed_at or datetime.now(timezone.utc).isoformat()
    clean_model_name = str(model_name or "unnamed_model").strip()

    feat_hash = compute_subcomponent_hash(norm_payload["features"])
    hparam_hash = compute_subcomponent_hash(norm_payload["hyperparameters"])
    wf_hash = compute_subcomponent_hash(norm_payload["walk_forward_config"])

    conn = connect_analysis_db(data_dir)
    try:
        # Atomic registration transaction
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO experiment_signatures (
                    signature_hash, context_key, market, sampling_interval_sec, task_type,
                    prediction_horizon, regime_id, regime_definition_hash, dataset_snapshot_hash,
                    feature_set_hash, algorithm, hyperparameters_hash, walk_forward_hash,
                    random_seed, canonical_payload_json, first_executed_at, execution_count,
                    latest_model_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
                ON CONFLICT(signature_hash) DO UPDATE SET
                    execution_count = experiment_signatures.execution_count + 1,
                    latest_model_name = excluded.latest_model_name;
                """,
                (
                    sig_hash,
                    norm_payload["context_key"],
                    norm_payload["market"],
                    norm_payload["sampling_interval_sec"],
                    norm_payload["task_type"],
                    norm_payload["prediction_horizon"],
                    norm_payload["regime_id"],
                    norm_payload["regime_definition_hash"],
                    norm_payload["dataset_snapshot_hash"],
                    feat_hash,
                    norm_payload["algorithm"],
                    hparam_hash,
                    wf_hash,
                    norm_payload["random_seed"],
                    canonical_json,
                    now_iso,
                    clean_model_name,
                ),
            )

            # Retrieve the current record state
            row = conn.execute(
                "SELECT * FROM experiment_signatures WHERE signature_hash = ?;",
                (sig_hash,),
            ).fetchone()
            record = dict(row) if row else {}
            is_new = (record.get("execution_count") == 1 and record.get("first_executed_at") == now_iso)
            return is_new, record
    finally:
        conn.close()


def get_experiment_by_signature(data_dir: str, signature_hash: str) -> dict[str, Any] | None:
    """Retrieve full experiment record by signature hash."""
    return check_experiment_exists(data_dir, signature_hash)


def list_experiments_for_context(data_dir: str, context_key: str) -> list[dict[str, Any]]:
    """List all registered experiment signatures for a given ModelContextKey."""
    init_analysis_db(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        rows = conn.execute(
            """
            SELECT * FROM experiment_signatures
            WHERE context_key = ?
            ORDER BY first_executed_at DESC;
            """,
            (context_key,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()
