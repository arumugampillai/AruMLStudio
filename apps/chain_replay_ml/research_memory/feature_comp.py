"""Feature Composition & Population Dependency Auditing (Phase 4D.4).

Evaluates the population composition (Base Pipeline PL_0001, Canonical Registry,
Experimental PL_0002+, Deprecated/Retired, and Unknown) of a model's feature set
and records empirical composition metrics in `<data_dir>/analysis.db`.

Invariants:
1. Strict Metadata-Based Classification: Uses authoritative Schema & Feature Registry
   metadata. Never guesses population from naming conventions.
2. Unknown Classification: Features unmapped in authoritative metadata are categorized
   explicitly as "UNKNOWN".
3. Empirical Ratios: Calculates experimental_dependency_ratio as a descriptive metric,
   not an automated rejection/governance decision.
4. Lineage Integrity: Every feature composition evaluation links to `experiment_signatures`.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import sqlite3
from typing import Any

from chain_replay_ml.dataset_builder.schema_registry import load_schema_registry
from .db import connect_analysis_db, init_analysis_db
from .signature import compute_subcomponent_hash


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_feature_population(
    feature_name: str,
    *,
    schema: dict[str, Any] | None = None,
) -> str:
    """Classify a single feature into its authoritative population.
    
    Categories:
    - 'BASE': Belongs to Base Pipeline PL_0001 (is_base=True or project_id='PL_0001')
    - 'DEPRECATED': Marked as deprecated or retired in the registry
    - 'REGISTRY': Active Canonical Registry feature (FR0001..FR0206)
    - 'EXPERIMENTAL': Active Experimental feature from a candidate project (PL_0002+)
    - 'UNKNOWN': Not found in authoritative schema or registry
    """
    clean_name = str(feature_name or "").strip()
    if not clean_name:
        return "UNKNOWN"

    reg = schema if schema is not None else load_schema_registry()
    cols = reg.get("columns") or {}
    meta = cols.get(clean_name)

    if not meta or not isinstance(meta, dict):
        return "UNKNOWN"

    status = str(meta.get("status") or "ACTIVE").upper().strip()
    if status in ("DEPRECATED", "RETIRED", "DISABLED"):
        return "DEPRECATED"

    # Base Pipeline check
    is_base = bool(meta.get("is_base") or meta.get("base_pipeline"))
    proj_id = str(meta.get("project_id") or meta.get("pipeline_id") or "").upper().strip()
    if is_base or proj_id == "PL_0001":
        return "BASE"

    if proj_id and proj_id != "PL_0001":
        return "EXPERIMENTAL"

    if status == "EXPERIMENTAL":
        return "EXPERIMENTAL"

    return "REGISTRY"


def analyze_feature_set_composition(
    features: list[str] | set[str] | tuple[str, ...],
    *,
    top_features: list[dict[str, Any]] | None = None,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze the complete feature set composition and population breakdown.
    
    Returns structured composition summary with counts, ratios, and categorized lists.
    """
    reg_schema = schema if schema is not None else load_schema_registry()
    clean_features = sorted(list(set(str(f).strip() for f in (features or []) if str(f).strip())))
    total = len(clean_features)

    base_feats: list[str] = []
    reg_feats: list[str] = []
    exp_feats: list[str] = []
    dep_feats: list[str] = []
    unk_feats: list[str] = []

    for f in clean_features:
        pop = classify_feature_population(f, schema=reg_schema)
        if pop == "BASE":
            base_feats.append(f)
        elif pop == "DEPRECATED":
            dep_feats.append(f)
        elif pop == "EXPERIMENTAL":
            exp_feats.append(f)
        elif pop == "REGISTRY":
            reg_feats.append(f)
        else:
            unk_feats.append(f)

    exp_ratio = round(len(exp_feats) / total, 4) if total > 0 else 0.0
    base_ratio = round(len(base_feats) / total, 4) if total > 0 else 0.0
    reg_ratio = round(len(reg_feats) / total, 4) if total > 0 else 0.0

    feat_hash = compute_subcomponent_hash(clean_features)

    categorized_inventory = {
        "base_features": base_feats,
        "registry_features": reg_feats,
        "experimental_features": exp_feats,
        "deprecated_features": dep_feats,
        "unknown_features": unk_feats,
    }

    # Format top 10 features
    top_10 = top_features[:10] if top_features else []

    return {
        "feature_set_hash": feat_hash,
        "total_features": total,
        "base_pipeline_count": len(base_feats),
        "registry_feature_count": len(reg_feats),
        "experimental_feature_count": len(exp_feats),
        "deprecated_feature_count": len(dep_feats),
        "unknown_feature_count": len(unk_feats),
        "experimental_dependency_ratio": exp_ratio,
        "base_dependency_ratio": base_ratio,
        "registry_dependency_ratio": reg_ratio,
        "categorized_inventory": categorized_inventory,
        "top_10_features": top_10,
        "top_10_features_json": json.dumps(top_10, sort_keys=True),
        "features_list_json": json.dumps(categorized_inventory, sort_keys=True),
    }


def record_feature_set_evaluation(
    data_dir: str,
    *,
    signature_hash: str,
    features: list[str] | set[str] | tuple[str, ...],
    top_features: list[dict[str, Any]] | None = None,
    schema: dict[str, Any] | None = None,
) -> int:
    """Analyze and record a feature set evaluation row into `<data_dir>/analysis.db`.
    
    Returns:
        The autoincrement `feature_eval_id`.
    """
    init_analysis_db(data_dir)
    comp = analyze_feature_set_composition(features, top_features=top_features, schema=schema)
    now_iso = _utc_now_iso()

    conn = connect_analysis_db(data_dir)
    try:
        with conn:
            cursor = conn.execute(
                """
                INSERT INTO feature_set_evaluations (
                    signature_hash, feature_set_hash, total_features,
                    base_pipeline_count, registry_feature_count, experimental_feature_count,
                    deprecated_feature_count, experimental_dependency_ratio,
                    top_10_features_json, features_list_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    str(signature_hash).strip(),
                    comp["feature_set_hash"],
                    comp["total_features"],
                    comp["base_pipeline_count"],
                    comp["registry_feature_count"],
                    comp["experimental_feature_count"],
                    comp["deprecated_feature_count"],
                    comp["experimental_dependency_ratio"],
                    comp["top_10_features_json"],
                    comp["features_list_json"],
                    now_iso,
                ),
            )
            return cursor.lastrowid or 0
    finally:
        conn.close()


def get_feature_set_evaluation(
    data_dir: str,
    signature_hash: str,
) -> dict[str, Any] | None:
    """Retrieve the feature set evaluation record for an experiment signature."""
    init_analysis_db(data_dir)
    conn = connect_analysis_db(data_dir)
    try:
        row = conn.execute(
            """
            SELECT * FROM feature_set_evaluations
            WHERE signature_hash = ?
            ORDER BY feature_eval_id DESC
            LIMIT 1;
            """,
            (str(signature_hash).strip(),),
        ).fetchone()
        if not row:
            return None
        res = dict(row)
        if res.get("features_list_json"):
            try:
                res["features_inventory"] = json.loads(res["features_list_json"])
            except Exception:
                res["features_inventory"] = {}
        if res.get("top_10_features_json"):
            try:
                res["top_10_features"] = json.loads(res["top_10_features_json"])
            except Exception:
                res["top_10_features"] = []
        return res
    finally:
        conn.close()
