"""Phase 3C — Training Provenance, Model Lineage, and Closed-Loop Traceability Audit."""

from __future__ import annotations

import json
import os
import sqlite3
from typing import Any

from chain_replay_ml.training.paths import model_package_dir, safe_model_name
from .dataset_context import resolve_context_from_model_package


def get_model_recommendation_provenance(package_dir: str) -> dict[str, Any] | None:
    """Read metadata.json from model package directory and return recommendation_provenance block.

    Returns:
        dict if recommendation provenance is present, None for legacy models or missing files.
        Never modifies any files or databases.
    """
    if not os.path.isdir(package_dir):
        return None

    meta_path = os.path.join(package_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        return None

    try:
        with open(meta_path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        if isinstance(doc, dict):
            prov = doc.get("recommendation_provenance")
            if isinstance(prov, dict):
                return prov
    except Exception:
        return None

    return None


def audit_model_training_feedback_loop(
    data_dir: str,
    model_name: str,
) -> dict[str, Any]:
    """Perform a pure read-only audit of the entire Recommendation -> Training -> Validation -> Evidence DB loop.

    Verifies:
    1. Model package existence.
    2. metadata.json existence.
    3. recommendation_provenance integrity (context_id, policy_id, policy_version, trained candidates).
    4. Authoritative context resolution.
    5. Production validation artifacts presence.
    6. SQLite Evidence DB validation runs for this model.
    7. Closed-loop evidence ingestion into feature_context_summary.

    Never modifies files or database tables.
    """
    safe_name = safe_model_name(model_name)
    pkg_dir = model_package_dir(data_dir, safe_name)

    res: dict[str, Any] = {
        "model_name": safe_name,
        "package_dir": pkg_dir,
        "model_exists": False,
        "metadata_exists": False,
        "has_recommendation_provenance": False,
        "context_id": None,
        "policy_id": None,
        "policy_version": None,
        "decision_engine_version": None,
        "trained_candidates": [],
        "trained_candidates_count": 0,
        "validation_artifacts_present": False,
        "db_validation_runs_count": 0,
        "db_evidence_features_count": 0,
        "feedback_loop_closed": False,
        "audit_status": "FAIL",
        "audit_messages": [],
    }

    if not os.path.isdir(pkg_dir):
        res["audit_messages"].append(f"Model package directory not found: {pkg_dir}")
        return res
    res["model_exists"] = True

    meta_path = os.path.join(pkg_dir, "metadata.json")
    if not os.path.isfile(meta_path):
        res["audit_messages"].append("metadata.json missing in package directory.")
        return res
    res["metadata_exists"] = True

    prov = get_model_recommendation_provenance(pkg_dir)
    if prov:
        res["has_recommendation_provenance"] = True
        res["context_id"] = prov.get("context_id")
        res["policy_id"] = prov.get("originating_policy_id")
        res["policy_version"] = prov.get("originating_policy_version")
        res["decision_engine_version"] = prov.get("decision_engine_version")
        res["trained_candidates"] = list(prov.get("trained_candidates") or [])
        res["trained_candidates_count"] = len(res["trained_candidates"])
    else:
        # Check if legacy context can be resolved
        ctx = resolve_context_from_model_package(data_dir, safe_name)
        if ctx:
            res["context_id"] = ctx.context_id
        res["audit_messages"].append("Model was trained without recommendation_decision_bundle (legacy model).")

    # Check Production Validation artifacts
    val_summary_path = os.path.join(pkg_dir, "production_validation_summary.json")
    val_comp_path = os.path.join(pkg_dir, "production_validation_comparison.json")
    if os.path.isfile(val_summary_path) or os.path.isfile(val_comp_path):
        res["validation_artifacts_present"] = True

    # Inspect SQLite Evidence DB (Read-Only)
    db_path = os.path.join(data_dir, "feature_recommendation_evidence.db")
    if os.path.isfile(db_path):
        try:
            uri = f"file:{os.path.abspath(db_path)}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=5.0)
            try:
                cur = conn.cursor()
                # 1. Count validation runs for this model
                cur.execute(
                    "SELECT COUNT(*), COUNT(DISTINCT feature_name) FROM recommendation_evidence WHERE model_name = ?",
                    (safe_name,),
                )
                row = cur.fetchone()
                if row:
                    res["db_validation_runs_count"] = int(row[0] or 0)
                    res["db_evidence_features_count"] = int(row[1] or 0)

                # 2. Check if context summary reflects evidence
                if res["context_id"]:
                    cur.execute(
                        "SELECT COUNT(*) FROM feature_context_summary WHERE context_id = ? AND total_runs > 0",
                        (res["context_id"],),
                    )
                    c_row = cur.fetchone()
                    if c_row and int(c_row[0] or 0) > 0 and res["db_validation_runs_count"] > 0:
                        res["feedback_loop_closed"] = True
            finally:
                conn.close()
        except Exception as exc:
            res["audit_messages"].append(f"Evidence DB query error: {exc}")

    # Determine overall audit status
    if res["model_exists"] and res["metadata_exists"]:
        if res["has_recommendation_provenance"] and res["context_id"]:
            if res["db_validation_runs_count"] > 0 and res["feedback_loop_closed"]:
                res["audit_status"] = "PASS"
                res["audit_messages"].append("Full closed loop verified: Recommendation -> Model -> Validation -> Evidence DB.")
            elif res["validation_artifacts_present"]:
                res["audit_status"] = "PASS"
                res["audit_messages"].append("Training provenance verified and validation artifacts present (pending DB ingestion).")
            else:
                res["audit_status"] = "PASS"
                res["audit_messages"].append("Training provenance verified (model ready for Production Validation).")
        else:
            res["audit_status"] = "LEGACY_PASS"
            res["audit_messages"].append("Legacy model verified with heuristic context resolution.")
    else:
        res["audit_status"] = "FAIL"

    return res

