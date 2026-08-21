"""Authoritative Feature Partitioning & Discovery Provenance Service for AruMLStudio.

Enforces strict mutually-exclusive tripartite categorization across all studios:
1. Baseline Features: Features belonging to the authoritative Base Pipeline (PL_0001).
2. Registry Features: Permanent Feature Registry features ONLY (never including Base Pipeline).
3. Experimental Features: Genuine Autonomous Discovery Pipeline features (DF_* / DP_* provenance) ONLY.

Invariant:
    Baseline Features ∩ Registry Features ∩ Experimental Features = ∅
"""

from __future__ import annotations

from enum import Enum
import json
import os
from typing import Any, Sequence


class FeatureCategory(str, Enum):
    BASELINE = "BASELINE"
    REGISTRY = "REGISTRY"
    EXPERIMENTAL = "EXPERIMENTAL"


def is_synthetic_or_experimental(feature_name: str) -> bool:
    """Determine whether a feature name represents an explicitly synthetic / discovery feature name."""
    fn = str(feature_name or "").strip().lower()
    return fn.startswith("df_") or fn.startswith("synth_")


def resolve_feature_partition_sets(
    data_dir: str,
    dataset_name: str | None = None,
    campaign_id: str | None = None,
) -> tuple[set[str], set[str], set[str], dict[str, dict[str, Any]]]:
    """Resolve mutually-exclusive sets: (baseline_set, registry_set, experimental_set, discovery_provenance_map).

    Returns:
        tuple[set[str], set[str], set[str], dict[str, dict[str, Any]]]:
            baseline_set: Authoritative PL_0001 Base Pipeline features.
            registry_set: Permanent Feature Registry features ONLY (excluding baseline).
            experimental_set: Genuine Autonomous Discovery Pipeline features.
            discovery_provenance_map: Dict mapping feature_name -> discovery metadata (pipeline_id, formula, etc.).
    """
    baseline_set: set[str] = set()
    registry_set: set[str] = set()
    experimental_set: set[str] = set()
    discovery_provenance_map: dict[str, dict[str, Any]] = {}

    clean_dir = str(data_dir or "").strip()

    # Auto-resolve dataset_name from analysis.db if not provided
    if clean_dir and not dataset_name:
        try:
            from chain_replay_ml.research_memory.db import connect_analysis_db
            conn = connect_analysis_db(clean_dir)
            if campaign_id:
                row = conn.execute("SELECT config_json FROM overnight_campaigns WHERE campaign_id = ?;", (campaign_id,)).fetchone()
            else:
                row = conn.execute("SELECT config_json FROM overnight_campaigns ORDER BY start_time_iso DESC LIMIT 1;").fetchone()
            if row and row["config_json"]:
                c_cfg = json.loads(row["config_json"])
                dataset_name = c_cfg.get("dataset_name")
        except Exception:
            pass

    # 1. Inspect Dataset Metadata JSON if available for authoritative baseline & registry export lists
    if clean_dir:
        try:
            meta_paths = []
            if dataset_name:
                meta_paths.extend([
                    os.path.join(clean_dir, "datasets", f"{dataset_name}.json"),
                    os.path.join(clean_dir, f"{dataset_name}.json"),
                ])
            
            ds_dir = os.path.join(clean_dir, "datasets")
            if os.path.isdir(ds_dir):
                all_json = [
                    os.path.join(ds_dir, f) for f in os.listdir(ds_dir)
                    if f.endswith(".json") and not f.endswith(".expected.json")
                ]
                all_json.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                meta_paths.extend(all_json)

            for cp in meta_paths:
                if os.path.isfile(cp):
                    with open(cp, "r", encoding="utf-8") as fh:
                        meta = json.load(fh)
                    base_exp = meta.get("base_pipeline_export_features", [])
                    if isinstance(base_exp, list) and base_exp:
                        for f in base_exp:
                            if f:
                                baseline_set.add(str(f).strip())
                    reg_exp = meta.get("registry_export_features", [])
                    if isinstance(reg_exp, list) and reg_exp:
                        for f in reg_exp:
                            if f:
                                registry_set.add(str(f).strip())
                    if baseline_set or registry_set:
                        break
        except Exception:
            pass

    # 2. Inspect pipeline_registry_store.json for PL_0001 / Base Pipeline features
    if clean_dir and os.path.isdir(clean_dir):
        try:
            from chain_replay_ml.dataset_builder.pipeline_registry_store import load_store as load_pr_store
            pr_doc = load_pr_store(clean_dir)
            pipelines = pr_doc.get("pipelines", {})
            for pid, pdata in pipelines.items():
                if str(pid).upper() == "PL_0001" or str(pdata.get("type", "")).lower() in ("base", "existing"):
                    for f in pdata.get("feature_names", []):
                        if f:
                            baseline_set.add(str(f).strip())
        except Exception:
            pass

    # 3. Inspect feature_registry_store.json for Permanent Feature Registry features
    if clean_dir and os.path.isdir(clean_dir):
        try:
            from chain_replay_ml.dataset_builder.feature_registry_store import load_store as load_fr_store
            fr_doc = load_fr_store(clean_dir)
            identities = fr_doc.get("feature_identities", {})
            for fid, item in identities.items():
                name = item.get("name") or item.get("canonical_name")
                if name:
                    registry_set.add(str(name).strip())
                code = item.get("feature_code") or fid
                if code and not str(code).startswith("FR_"):
                    registry_set.add(str(code).strip())
            for f_name in fr_doc.get("features", {}).keys():
                if f_name:
                    registry_set.add(str(f_name).strip())
        except Exception:
            pass

    # 4. Inspect analysis.db for genuine Autonomous Discovery Pipeline features (DF_*)
    if clean_dir and os.path.isdir(clean_dir):
        try:
            from chain_replay_ml.research_memory.db import connect_analysis_db
            conn = connect_analysis_db(clean_dir)
            query = "SELECT feature_id, pipeline_id, feature_name, formula_expression, formula_hash, generator_strategy, generation_discovered, lifecycle_status, evidence_score FROM discovery_pipeline_features"
            params: list[Any] = []
            if campaign_id:
                query += " WHERE pipeline_id IN (SELECT pipeline_id FROM discovery_pipelines WHERE campaign_id = ?)"
                params.append(campaign_id)
            rows = conn.execute(query, params).fetchall()
            for r in rows:
                fn = str(r["feature_name"]).strip()
                experimental_set.add(fn)
                discovery_provenance_map[fn] = {
                    "feature_id": r["feature_id"],
                    "pipeline_id": r["pipeline_id"],
                    "formula_expression": r["formula_expression"],
                    "formula_hash": r["formula_hash"],
                    "generator_strategy": r["generator_strategy"],
                    "generation_discovered": r["generation_discovered"],
                    "lifecycle_status": r["lifecycle_status"],
                    "evidence_score": r["evidence_score"],
                }
        except Exception:
            pass

    # 5. Strict Mutually Exclusive Disjoint Enforcement:
    # Permanent Registry features MUST NEVER contain Base Pipeline features
    registry_set = registry_set - baseline_set
    # Baseline & Registry features MUST NEVER contain genuine Experimental features
    baseline_set = baseline_set - experimental_set
    registry_set = registry_set - experimental_set

    return baseline_set, registry_set, experimental_set, discovery_provenance_map


def classify_feature(
    feature_name: str,
    baseline_set: set[str],
    registry_set: set[str],
    experimental_set: set[str] | None = None,
) -> FeatureCategory:
    """Classify a single feature into exactly ONE of the three mutually-exclusive categories."""
    fn = str(feature_name or "").strip()
    if not fn:
        return FeatureCategory.BASELINE

    # Priority 1: Genuine Autonomous Discovery Pipeline features (DF_* or in discovery_pipeline_features)
    if (experimental_set and fn in experimental_set) or is_synthetic_or_experimental(fn):
        return FeatureCategory.EXPERIMENTAL

    # Priority 2: Base Pipeline (PL_0001) features
    if fn in baseline_set:
        return FeatureCategory.BASELINE

    # Priority 3: Permanent Feature Registry features
    if fn in registry_set:
        return FeatureCategory.REGISTRY

    # Priority 4: Default non-synthetic dataset features to BASELINE
    return FeatureCategory.BASELINE


def get_campaign_discovery_pipeline_info(
    data_dir: str,
    campaign_id: str | None = None,
) -> dict[str, Any]:
    """Retrieve authoritative Discovery Pipeline metadata and snapshot summary for a campaign."""
    clean_dir = str(data_dir or "").strip()
    if not clean_dir:
        return {"pipelines": [], "primary_pipeline_id": None, "total_pipelines": 0}

    try:
        from chain_replay_ml.research_memory.db import connect_analysis_db
        conn = connect_analysis_db(clean_dir)
        if campaign_id:
            rows = conn.execute(
                "SELECT pipeline_id, campaign_id, context_key, dataset_name, base_feature_count, active_features_count, current_generation, current_snapshot_hash, status FROM discovery_pipelines WHERE campaign_id = ? ORDER BY created_at DESC;",
                (campaign_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT pipeline_id, campaign_id, context_key, dataset_name, base_feature_count, active_features_count, current_generation, current_snapshot_hash, status FROM discovery_pipelines ORDER BY created_at DESC;",
            ).fetchall()

        pipelines = [dict(r) for r in rows]
        primary_id = pipelines[0]["pipeline_id"] if pipelines else None
        return {
            "pipelines": pipelines,
            "primary_pipeline_id": primary_id,
            "total_pipelines": len(pipelines),
            "primary_pipeline": pipelines[0] if pipelines else None,
        }
    except Exception:
        return {"pipelines": [], "primary_pipeline_id": None, "total_pipelines": 0}


def partition_feature_records(
    records: Sequence[Any],
    data_dir: str,
    dataset_name: str | None = None,
    campaign_id: str | None = None,
    name_accessor: Any = None,
) -> tuple[list[Any], list[Any], list[Any], dict[str, dict[str, Any]]]:
    """Partition a list of feature records/names into (registry_list, baseline_list, experimental_list, provenance_map).

    Returns:
        tuple[list[Any], list[Any], list[Any], dict[str, dict[str, Any]]]:
            registry_list: Permanent Feature Registry features ONLY.
            baseline_list: Base Pipeline (PL_0001) features ONLY.
            experimental_list: Genuine Autonomous Discovery Pipeline (DF_*) features ONLY.
            provenance_map: Dict mapping feature_name -> discovery metadata.
    """
    baseline_set, registry_set, experimental_set, prov_map = resolve_feature_partition_sets(
        data_dir, dataset_name, campaign_id
    )

    registry_list: list[Any] = []
    baseline_list: list[Any] = []
    experimental_list: list[Any] = []

    for item in records:
        if name_accessor is not None:
            fn = name_accessor(item)
        elif hasattr(item, "feature_name"):
            fn = item.feature_name
        elif isinstance(item, dict):
            fn = item.get("feature_name") or item.get("name") or str(item)
        else:
            fn = str(item)

        cat = classify_feature(fn, baseline_set, registry_set, experimental_set)
        if cat == FeatureCategory.REGISTRY:
            registry_list.append(item)
        elif cat == FeatureCategory.BASELINE:
            baseline_list.append(item)
        else:
            experimental_list.append(item)

    return registry_list, baseline_list, experimental_list, prov_map
