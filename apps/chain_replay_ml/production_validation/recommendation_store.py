"""Cumulative Feature Recommendation history + summary (JSON under data_dir).

Production Validation never mutates Pipeline Features or the Feature Registry.
``update_registry_recommendations`` only appends/upserts history rows and
rebuilds the aggregated summary used by Pipeline/Registry UIs later.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

STORAGE = "feature_recommendation_history.json"
VALID_RECOMMENDATIONS = frozenset({"KEEP", "WATCH", "REMOVE"})
DEFAULT_PERSIST = frozenset({"KEEP", "WATCH", "REMOVE"})

STAR_GLYPHS = {
    5: "★★★★★",
    4: "★★★★☆",
    3: "★★★☆☆",
    2: "★★☆☆☆",
    1: "★☆☆☆☆",
}


def storage_path(data_dir: str) -> str:
    root = str(data_dir or "").strip()
    if not root:
        return ""
    return os.path.join(root, STORAGE)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _empty_doc() -> dict[str, Any]:
    return {
        "version": 1,
        "updated_at": None,
        "entries": [],
        "ignored": {},
        "summary": {"by_feature": {}},
    }


def load_recommendation_store(data_dir: str) -> dict[str, Any]:
    path = storage_path(data_dir)
    if not path or not os.path.isfile(path):
        return _empty_doc()
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError, TypeError):
        return _empty_doc()
    if not isinstance(doc, dict):
        return _empty_doc()
    doc.setdefault("version", 1)
    doc.setdefault("entries", [])
    doc.setdefault("ignored", {})
    doc.setdefault("summary", {"by_feature": {}})
    if not isinstance(doc["entries"], list):
        doc["entries"] = []
    if not isinstance(doc["ignored"], dict):
        doc["ignored"] = {}
    if not isinstance(doc.get("summary"), dict):
        doc["summary"] = {"by_feature": {}}
    doc["summary"].setdefault("by_feature", {})
    return doc


def save_recommendation_store(data_dir: str, doc: dict[str, Any]) -> dict[str, Any]:
    path = storage_path(data_dir)
    if not path:
        raise ValueError("data_dir is required")
    payload = dict(doc)
    payload["version"] = int(payload.get("version") or 1)
    payload["updated_at"] = _utc_now()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        fh.write("\n")
    return payload


def compute_recommendation_strength(
    *,
    remove_models: int,
    watch_models: int,
    keep_models: int,
) -> int:
    """Advisory 1–5 rating weighted by unique model counts (not raw runs)."""
    rm = max(0, int(remove_models or 0))
    wm = max(0, int(watch_models or 0))
    km = max(0, int(keep_models or 0))
    total = rm + wm + km
    if total == 0:
        return 1

    if rm >= wm and rm >= km:
        # REMOVE leads
        if rm >= 3 and km == 0 and wm <= 1:
            return 5
        if rm >= 2 and km == 0:
            return 4
        if rm >= 2 and km > 0:
            return 3
        if rm == 1 and km == 0 and wm == 0:
            return 3
        if rm >= 1 and (km > 0 or wm > 0):
            return 3
        return 3

    if wm >= rm and wm >= km:
        return 2 if rm == 0 else 3

    # KEEP leads
    return 1 if rm == 0 else 3


def recommendation_strength_stars(stars: int) -> str:
    return STAR_GLYPHS.get(max(1, min(5, int(stars))), STAR_GLYPHS[1])


def _normalize_recommendation(value: Any) -> str | None:
    text = str(value or "").strip().upper()
    return text if text in VALID_RECOMMENDATIONS else None


def _feature_id_map(data_dir: str) -> dict[str, str]:
    try:
        from chain_replay_ml.dataset_builder.feature_registry_store import load_store

        store = load_store(data_dir)
    except Exception:
        return {}
    raw = store.get("feature_ids") if isinstance(store, dict) else None
    if not isinstance(raw, dict):
        return {}
    return {
        str(name).strip(): str(fid).strip()
        for name, fid in raw.items()
        if str(name).strip() and str(fid).strip()
    }


def _domain_map(data_dir: str) -> dict[str, str]:
    """Best-effort name → domain from Feature Registry identities/catalog."""
    out: dict[str, str] = {}
    try:
        from chain_replay_ml.dataset_builder.feature_registry_store import load_store

        store = load_store(data_dir)
        identities = store.get("feature_identities") or {}
        ids = store.get("feature_ids") or {}
        if isinstance(identities, dict) and isinstance(ids, dict):
            for name, fid in ids.items():
                ident = identities.get(fid) or {}
                domain = str(ident.get("domain") or ident.get("group") or "").strip()
                if domain:
                    out[str(name).strip()] = domain
    except Exception:
        pass
    return out


def rebuild_summary(doc: dict[str, Any], *, domains: dict[str, str] | None = None) -> dict[str, Any]:
    """Aggregate unique-model + run counts per feature from history entries."""
    by_feature: dict[str, dict[str, Any]] = {}
    model_sets: dict[str, dict[str, set[str]]] = {}

    for entry in doc.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        rec = _normalize_recommendation(entry.get("recommendation"))
        name = str(entry.get("feature_name") or "").strip()
        if not rec or not name:
            continue
        model = str(entry.get("model_name") or "").strip()
        row = by_feature.get(name)
        if row is None:
            row = {
                "feature_id": str(entry.get("feature_id") or "").strip(),
                "feature_name": name,
                "domain": (domains or {}).get(name, ""),
                "remove_runs": 0,
                "remove_models": 0,
                "watch_runs": 0,
                "watch_models": 0,
                "keep_runs": 0,
                "keep_models": 0,
                "last_recommendation": rec,
                "last_model": model,
                "last_date": str(entry.get("generated_date") or ""),
                "recommendation_strength": 1,
                "recommendation_strength_stars": STAR_GLYPHS[1],
            }
            by_feature[name] = row
            model_sets[name] = {"REMOVE": set(), "WATCH": set(), "KEEP": set()}
        else:
            fid = str(entry.get("feature_id") or "").strip()
            if fid and not row.get("feature_id"):
                row["feature_id"] = fid
            if domains and domains.get(name) and not row.get("domain"):
                row["domain"] = domains[name]

        key = rec.lower()
        row[f"{key}_runs"] = int(row.get(f"{key}_runs") or 0) + 1
        if model:
            model_sets[name][rec].add(model)

        last_date = str(row.get("last_date") or "")
        gen = str(entry.get("generated_date") or "")
        if gen >= last_date:
            row["last_recommendation"] = rec
            row["last_model"] = model
            row["last_date"] = gen

    for name, row in by_feature.items():
        sets = model_sets[name]
        row["remove_models"] = len(sets["REMOVE"])
        row["watch_models"] = len(sets["WATCH"])
        row["keep_models"] = len(sets["KEEP"])
        stars = compute_recommendation_strength(
            remove_models=row["remove_models"],
            watch_models=row["watch_models"],
            keep_models=row["keep_models"],
        )
        row["recommendation_strength"] = stars
        row["recommendation_strength_stars"] = recommendation_strength_stars(stars)

    summary = {"by_feature": by_feature}
    doc["summary"] = summary
    return summary


def list_recommendation_history(
    data_dir: str,
    *,
    feature_name: str | None = None,
    recommendation: str | None = None,
    model_name: str | None = None,
) -> list[dict[str, Any]]:
    doc = load_recommendation_store(data_dir)
    feat = str(feature_name or "").strip()
    rec_filter = _normalize_recommendation(recommendation) if recommendation else None
    model = str(model_name or "").strip()
    rows: list[dict[str, Any]] = []
    for entry in doc.get("entries") or []:
        if not isinstance(entry, dict):
            continue
        if feat and str(entry.get("feature_name") or "").strip() != feat:
            continue
        if model and str(entry.get("model_name") or "").strip() != model:
            continue
        if rec_filter and _normalize_recommendation(entry.get("recommendation")) != rec_filter:
            continue
        rows.append(dict(entry))
    rows.sort(key=lambda r: str(r.get("generated_date") or ""), reverse=True)
    return rows


def get_recommendation_summary(
    data_dir: str,
    *,
    rebuild: bool = False,
) -> dict[str, Any]:
    doc = load_recommendation_store(data_dir)
    by_feature = (doc.get("summary") or {}).get("by_feature")
    if rebuild or not isinstance(by_feature, dict) or not by_feature:
        domains = _domain_map(data_dir)
        rebuild_summary(doc, domains=domains)
        save_recommendation_store(data_dir, doc)
    return {
        "updated_at": doc.get("updated_at"),
        "features": list((doc.get("summary") or {}).get("by_feature", {}).values()),
        "ignored": dict(doc.get("ignored") or {}),
    }


def recommended_for_removal(
    data_dir: str,
    *,
    min_remove_runs: int = 1,
    include_ignored: bool = False,
    context_id: str | None = None,
) -> list[dict[str, Any]]:
    """Features with REMOVE history for Pipeline Features / Registry dialogs."""
    summary = get_recommendation_summary(data_dir)
    ignored = summary.get("ignored") or {}
    rows: list[dict[str, Any]] = []
    for feat in summary.get("features") or []:
        if not isinstance(feat, dict):
            continue
        name = str(feat.get("feature_name") or "").strip()
        if not name:
            continue
        if int(feat.get("remove_runs") or 0) < int(min_remove_runs):
            continue
        if not include_ignored and name in ignored:
            continue
        rows.append(dict(feat))
    rows.sort(
        key=lambda r: (
            -int(r.get("remove_models") or 0),
            -int(r.get("remove_runs") or 0),
            str(r.get("feature_name") or ""),
        )
    )
    return rows


def get_population_recommendations(
    data_dir: str,
    *,
    population: str = "registry",  # 'registry', 'base_pipeline', 'experimental', 'all'
    context_id: str | None = None,
    include_legacy_unknown: bool = False,
) -> list[dict[str, Any]]:
    """Retrieve population-specific recommendations and health metrics from Evidence DB."""
    from .dataset_context import LEGACY_UNKNOWN_CONTEXT_ID
    from .evidence_store import get_connection
    from .recommendation_policy import (
        RecommendationPolicy,
        compute_evidence_confidence,
        compute_model_consensus,
        compute_recency_staleness,
        load_recommendation_policy,
    )

    root = str(data_dir or "").strip()
    if not root:
        return []

    try:
        conn = get_connection(root)
    except Exception:
        return []

    try:
        policy = load_recommendation_policy(root, context_id=context_id)
        pop = str(population or "registry").strip().lower()
        if pop == "experimental":
            query = """
                SELECT 
                    e.lineage_id, e.context_id, e.pipeline_id, e.pipeline_snapshot_id,
                    e.feature_name, e.feature_identity_key, e.total_runs, e.keep_runs,
                    e.watch_runs, e.remove_runs, e.unique_models_count,
                    e.consecutive_keep_count, e.consecutive_remove_count,
                    e.lineage_evidence_score, e.lifecycle_status, e.last_recommendation,
                    e.last_validated_at,
                    COALESCE(c.lifecycle_status, 'active') as context_status
                FROM experimental_lineage_summary e
                LEFT JOIN feature_context_summary c 
                    ON e.context_id = c.context_id 
                    AND c.feature_source = 'experimental' 
                    AND e.feature_name = c.feature_name
                WHERE 1=1
            """
            params: list[Any] = []
            if context_id:
                query += " AND e.context_id = ?"
                params.append(context_id)
            elif not include_legacy_unknown:
                query += " AND e.context_id != ?"
                params.append(LEGACY_UNKNOWN_CONTEXT_ID)

            query += " ORDER BY e.lifecycle_status DESC, e.lineage_evidence_score DESC, e.feature_name ASC"
            cur = conn.execute(query, params)
            rows = [dict(row) for row in cur.fetchall()]
            return _enrich_intelligence_metrics(rows, conn, population="experimental", context_id=context_id, policy=policy)

        elif pop == "base_pipeline":
            query = """
                SELECT * FROM feature_context_summary
                WHERE feature_source = 'base_pipeline'
            """
            params = []
            if context_id:
                query += " AND context_id = ?"
                params.append(context_id)
            elif not include_legacy_unknown:
                query += " AND context_id != ?"
                params.append(LEGACY_UNKNOWN_CONTEXT_ID)

            query += " ORDER BY evidence_score DESC, keep_runs DESC, feature_name ASC"
            cur = conn.execute(query, params)
            rows = [dict(r) for r in cur.fetchall()]
            for rank, r in enumerate(rows, start=1):
                r["priority_rank"] = rank
            return _enrich_intelligence_metrics(rows, conn, population="base_pipeline", context_id=context_id, policy=policy)

        elif pop == "registry":
            query = """
                SELECT * FROM feature_context_summary
                WHERE feature_source = 'registry'
            """
            params = []
            if context_id:
                query += " AND context_id = ?"
                params.append(context_id)
            elif not include_legacy_unknown:
                query += " AND context_id != ?"
                params.append(LEGACY_UNKNOWN_CONTEXT_ID)

            query += " ORDER BY evidence_score ASC, remove_runs DESC, feature_name ASC"
            cur = conn.execute(query, params)
            rows = [dict(row) for row in cur.fetchall()]
            return _enrich_intelligence_metrics(rows, conn, population="registry", context_id=context_id, policy=policy)

        else:  # all
            query = "SELECT * FROM feature_context_summary WHERE 1=1"
            params = []
            if context_id:
                query += " AND context_id = ?"
                params.append(context_id)
            elif not include_legacy_unknown:
                query += " AND context_id != ?"
                params.append(LEGACY_UNKNOWN_CONTEXT_ID)
            query += " ORDER BY feature_source, evidence_score DESC"
            cur = conn.execute(query, params)
            rows = [dict(row) for row in cur.fetchall()]
            return _enrich_intelligence_metrics(rows, conn, population="all", context_id=context_id, policy=policy)
    finally:
        conn.close()


def _enrich_intelligence_metrics(
    rows: list[dict[str, Any]],
    conn: Any,
    *,
    population: str,
    context_id: str | None = None,
    policy: Any = None,
) -> list[dict[str, Any]]:
    if not rows:
        return []

    from .recommendation_policy import (
        compute_context_generalization,
        compute_evidence_confidence,
        compute_model_consensus,
        compute_recency_staleness,
        compute_score_volatility,
        derive_risk_badges,
    )

    # 1. Fetch raw evidence for historical reconstruction & consensus
    ev_query = """
        SELECT context_id, feature_name, feature_source, pipeline_id, pipeline_snapshot_id,
               model_name, recommendation, run_timestamp, evidence_id
        FROM recommendation_evidence
    """
    params = []
    if context_id:
        ev_query += " WHERE context_id = ?"
        params.append(context_id)
    ev_query += " ORDER BY run_timestamp ASC"

    try:
        cur_ev = conn.execute(ev_query, params)
        raw_ev = [dict(r) for r in cur_ev.fetchall()]
    except Exception:
        raw_ev = []

    ev_by_target: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for r in raw_ev:
        cid = r.get("context_id")
        fn = r.get("feature_name")
        pid = r.get("pipeline_id")
        snap = r.get("pipeline_snapshot_id")
        if population == "experimental" and pid and snap:
            key = (cid, pid, snap, fn)
        else:
            key = (cid, fn)
        ev_by_target.setdefault(key, []).append(r)

    # 2. Level 1 Comparable Context Resolution for Generalization
    level1_comparable_cids: list[str] = []
    comparable_summaries: dict[str, dict[str, Any]] = {}
    try:
        cur_ctxs = conn.execute("SELECT context_id, market, sampling_interval_sec, sliding_window, feature_project_id FROM dataset_contexts;").fetchall()
        all_ctxs = {c["context_id"]: dict(c) for c in cur_ctxs}
        
        target_ctx = all_ctxs.get(context_id) if context_id else None
        if target_ctx:
            m = str(target_ctx.get("market") or "").upper()
            win = str(target_ctx.get("sliding_window") or "").lower()
            fpid = str(target_ctx.get("feature_project_id") or "").lower()
            target_sec = int(target_ctx.get("sampling_interval_sec") or 0)
            
            for c_id, c_data in all_ctxs.items():
                if (
                    c_id != context_id
                    and str(c_data.get("market") or "").upper() == m
                    and str(c_data.get("sliding_window") or "").lower() == win
                    and str(c_data.get("feature_project_id") or "").lower() == fpid
                    and int(c_data.get("sampling_interval_sec") or 0) != target_sec
                ):
                    level1_comparable_cids.append(c_id)

        # Pre-load context summaries across all relevant contexts
        if level1_comparable_cids or context_id:
            relevant_cids = list(set(([context_id] if context_id else []) + level1_comparable_cids))
            placeholders = ",".join("?" for _ in relevant_cids)
            if population == "experimental":
                q_sum = f"SELECT context_id, feature_name, lineage_evidence_score as evidence_score, last_recommendation, dominant_recommendation FROM experimental_lineage_summary WHERE context_id IN ({placeholders})"
            else:
                q_sum = f"SELECT context_id, feature_name, evidence_score, last_recommendation FROM feature_context_summary WHERE context_id IN ({placeholders})"
            
            cur_sum = conn.execute(q_sum, relevant_cids).fetchall()
            for s_row in cur_sum:
                cid = s_row["context_id"]
                fn = s_row["feature_name"]
                comparable_summaries.setdefault(cid, {})[fn] = dict(s_row)
    except Exception:
        level1_comparable_cids = []
        comparable_summaries = {}

    for r in rows:
        cid = r.get("context_id")
        fn = r.get("feature_name")
        pid = r.get("pipeline_id")
        snap = r.get("pipeline_snapshot_id")

        if population == "experimental" and pid and snap:
            key = (cid, pid, snap, fn)
            runs = int(r.get("total_runs") or 0)
            score_val = float(r.get("lineage_evidence_score") or 0.0)
        else:
            key = (cid, fn)
            runs = int(r.get("total_runs") or 0)
            score_val = float(r.get("evidence_score") or 0.0)

        models = int(r.get("unique_models_count") or 0)

        # 1. Evidence Confidence (Phase 2A)
        conf = compute_evidence_confidence(runs=runs, models=models, policy=policy)
        r["evidence_confidence"] = conf
        r["confidence_pct"] = round(conf * 100.0, 1)
        r["confidence_display"] = f"{conf * 100.0:.1f}%" if runs > 0 else "—"

        # 2. Staleness & Freshness (Phase 2A)
        staleness = compute_recency_staleness(r.get("last_validated_at"))
        r["staleness_seconds"] = staleness["staleness_seconds"]
        r["staleness_days"] = staleness["staleness_days"]
        r["freshness_label"] = staleness["freshness_label"]
        r["freshness_display"] = staleness["display_text"]

        # 3. Model Consensus (Phase 2A)
        feat_ev_rows = ev_by_target.get(key, [])
        consensus = compute_model_consensus(feat_ev_rows)
        r["model_consensus"] = consensus
        r["consensus_ratio"] = consensus["consensus_ratio"]
        r["is_consensus_tie"] = consensus["is_tie"]
        r["dominant_recommendation"] = consensus["dominant_recommendation"]
        r["consensus_display"] = consensus["display_text"]

        # 4. Operational Priority Score (Phase 2A)
        op_score = round(score_val * conf, 2)
        r["operational_priority_score"] = op_score

        # 5. Score Stability & Volatility (Phase 2B)
        vol_res = compute_score_volatility(feat_ev_rows, policy=policy)
        r["score_volatility"] = vol_res["volatility_score"]
        r["score_range"] = vol_res["score_range"]
        r["direction_flips"] = vol_res["direction_flips"]
        r["stability_label"] = vol_res["stability_label"]
        r["stability_display"] = vol_res["display_text"]
        r["score_trajectory"] = vol_res["score_trajectory"]

        # 6. Level 1 Cross-Context Generalization (Phase 2B)
        gen_res = compute_context_generalization(
            primary_context_id=cid or "",
            feature_name=fn or "",
            context_summaries_by_cid=comparable_summaries,
            level1_comparable_context_ids=level1_comparable_cids,
        )
        r["generalization_score"] = gen_res["generalization_score"]
        r["generalization_label"] = gen_res["generalization_label"]
        r["generalization_display"] = gen_res["display_text"]
        r["comparable_context_count"] = gen_res["comparable_context_count"]

        # 7. Explicit Risk Badges (Phase 2B)
        risk_badges = derive_risk_badges(
            evidence_score=score_val,
            is_consensus_tie=bool(r.get("is_consensus_tie", False)),
            freshness_label=str(r.get("freshness_label") or ""),
            stability_label=str(r.get("stability_label") or ""),
        )
        r["risk_badges"] = risk_badges
        r["risk_badges_display"] = " ".join(f"[{b}]" for b in risk_badges) if risk_badges else "—"

    if population == "base_pipeline":
        sorted_for_advisory = sorted(
            rows,
            key=lambda item: (
                -float(item.get("operational_priority_score") or 0.0),
                -int(item.get("keep_runs") or 0),
                str(item.get("feature_name") or ""),
            ),
        )
        for adv_idx, item in enumerate(sorted_for_advisory, start=1):
            item["advisory_rank"] = adv_idx

    return rows


def ignore_recommendation(
    data_dir: str,
    feature_name: str,
    *,
    scope: str = "all",
    reason: str | None = None,
) -> dict[str, Any]:
    name = str(feature_name or "").strip()
    if not name:
        raise ValueError("feature_name is required")
    doc = load_recommendation_store(data_dir)
    ignored = dict(doc.get("ignored") or {})
    ignored[name] = {
        "feature_name": name,
        "ignored_at": _utc_now(),
        "scope": str(scope or "all").strip() or "all",
        "reason": str(reason or "").strip() or None,
    }
    doc["ignored"] = ignored
    save_recommendation_store(data_dir, doc)
    return ignored[name]


def unignore_recommendation(data_dir: str, feature_name: str) -> bool:
    name = str(feature_name or "").strip()
    if not name:
        raise ValueError("feature_name is required")
    doc = load_recommendation_store(data_dir)
    ignored = dict(doc.get("ignored") or {})
    if name not in ignored:
        return False
    del ignored[name]
    doc["ignored"] = ignored
    save_recommendation_store(data_dir, doc)
    return True


def ensure_run_id(meta: dict[str, Any]) -> str:
    existing = str(meta.get("run_id") or meta.get("production_validation_run_id") or "").strip()
    if existing:
        return existing
    run_id = str(uuid.uuid4())
    meta["run_id"] = run_id
    return run_id


def _patch_run_meta_run_id(package_dir: str, run_id: str) -> None:
    path = os.path.join(package_dir, "production_validation", "run_meta.json")
    if not os.path.isfile(path):
        return
    try:
        with open(path, encoding="utf-8") as fh:
            meta = json.load(fh)
    except (OSError, json.JSONDecodeError, TypeError):
        return
    if not isinstance(meta, dict):
        return
    if str(meta.get("run_id") or "").strip() == run_id:
        return
    meta["run_id"] = run_id
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, default=str)


def _next_entry_id(entries: Sequence[dict[str, Any]]) -> int:
    return max((int(e.get("id") or 0) for e in entries if isinstance(e, dict)), default=0) + 1


def _upsert_entry(
    entries: list[dict[str, Any]],
    *,
    feature_id: str,
    feature_name: str,
    model_name: str,
    recommendation: str,
    generated_date: str,
    run_id: str,
    recommendation_detail: str | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Replace same (run_id, model, feature) row; returns (entries, inserted_new)."""
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        if (
            str(entry.get("production_validation_run_id") or "").strip() == run_id
            and str(entry.get("model_name") or "").strip() == model_name
            and str(entry.get("feature_name") or "").strip() == feature_name
        ):
            updated = dict(entry)
            updated.update(
                {
                    "feature_id": feature_id or entry.get("feature_id") or "",
                    "feature_name": feature_name,
                    "model_name": model_name,
                    "recommendation": recommendation,
                    "generated_date": generated_date,
                    "production_validation_run_id": run_id,
                }
            )
            if recommendation_detail is not None:
                updated["recommendation_detail"] = recommendation_detail
            entries[idx] = updated
            return entries, False

    entry = {
        "id": _next_entry_id(entries),
        "feature_id": feature_id,
        "feature_name": feature_name,
        "model_name": model_name,
        "recommendation": recommendation,
        "generated_date": generated_date,
        "production_validation_run_id": run_id,
    }
    if recommendation_detail:
        entry["recommendation_detail"] = recommendation_detail
    entries.append(entry)
    return entries, True


def update_registry_recommendations(
    data_dir: str,
    *,
    model_name: str,
    package_dir: str | None = None,
    recommendations: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Persist PV recommendations for ``model_name`` into the cumulative store.

    Loads the model's latest Production Validation comparison artifacts.
    Upserts by (run_id, model, feature) so repeated button clicks do not
    double-count the same run.

    Defaults to persisting KEEP/WATCH/REMOVE so summary strength (unique-model
    REMOVE vs KEEP history) stays meaningful. Pass ``recommendations={"REMOVE"}``
    to match a REMOVE-only write.
    """
    from chain_replay_ml.production_validation.writer import load_validation_artifacts
    from chain_replay_ml.training.paths import model_package_dir, safe_model_name

    root = str(data_dir or "").strip()
    if not root:
        raise ValueError("data_dir is required")
    model = str(model_name or "").strip()
    if not model:
        raise ValueError("model_name is required")

    safe = safe_model_name(model)
    pkg = str(package_dir or "").strip() or model_package_dir(root, safe)
    artifacts = load_validation_artifacts(pkg)
    if not artifacts:
        raise FileNotFoundError(
            f"No production validation artifacts for model '{model}'"
        )

    allowed = {
        r
        for r in (
            _normalize_recommendation(x)
            for x in (recommendations if recommendations is not None else DEFAULT_PERSIST)
        )
        if r
    }
    if not allowed:
        raise ValueError("recommendations must include KEEP, WATCH, and/or REMOVE")

    meta = dict(artifacts.get("meta") or {})
    run_id = ensure_run_id(meta)
    _patch_run_meta_run_id(pkg, run_id)

    generated_date = str(
        meta.get("generated_at")
        or meta.get("computed_at")
        or artifacts.get("summary", {}).get("computed_at")
        or _utc_now()
    )
    # Prefer human model name from meta when present
    display_model = str(meta.get("model_name") or model).strip() or safe

    # Ensure legacy migration runs on existing pre-existing JSON before new entries are added
    try:
        from .recommendation_migration import migrate_legacy_recommendation_json
        migrate_legacy_recommendation_json(root)
    except Exception:
        pass

    id_map = _feature_id_map(root)
    domains = _domain_map(root)
    doc = load_recommendation_store(root)
    entries = [e for e in (doc.get("entries") or []) if isinstance(e, dict)]

    inserted = 0
    updated = 0
    skipped = 0
    for row in artifacts.get("rows") or []:
        if not isinstance(row, dict):
            continue
        rec = _normalize_recommendation(row.get("recommendation"))
        if not rec or rec not in allowed:
            skipped += 1
            continue
        feat = str(row.get("feature") or row.get("feature_name") or "").strip()
        if not feat:
            skipped += 1
            continue
        detail = row.get("recommendation_detail")
        entries, is_new = _upsert_entry(
            entries,
            feature_id=id_map.get(feat, ""),
            feature_name=feat,
            model_name=display_model,
            recommendation=rec,
            generated_date=generated_date,
            run_id=run_id,
            recommendation_detail=str(detail) if detail is not None else None,
        )
        if is_new:
            inserted += 1
        else:
            updated += 1

    doc["entries"] = entries
    rebuild_summary(doc, domains=domains)
    save_recommendation_store(root, doc)

    # --- SQLite Evidence DB Write Path (Canonical Evidence Store) ---
    sqlite_res: dict[str, Any] = {}
    try:
        from .dataset_context import resolve_context_from_model_package, resolve_context_or_legacy
        from .evidence_store import append_validation_evidence, get_connection
        from .recommendation_policy import load_recommendation_policy

        context = resolve_context_from_model_package(root, model) or resolve_context_or_legacy(root, model)
        policy = load_recommendation_policy(root, context_id=context.context_id)
        conn = get_connection(root)
        try:
            cfg_path = os.path.join(pkg, "config.json")
            model_cfg: dict[str, Any] = {}
            if os.path.isfile(cfg_path):
                try:
                    with open(cfg_path, encoding="utf-8") as cfh:
                        model_cfg = json.load(cfh)
                except Exception:
                    model_cfg = {}

            model_pid = str(model_cfg.get("pipeline_id") or "").strip().upper() or None
            model_snap = str(model_cfg.get("pipeline_snapshot_id") or "").strip() or None
            target_col = str(model_cfg.get("target") or "")

            ds_meta = model_cfg.get("dataset_metadata")
            if not isinstance(ds_meta, dict):
                ds_name = model_cfg.get("dataset")
                if ds_name:
                    from chain_replay_ml.dataset_builder.writer import _safe_filename, datasets_dir
                    ds_meta_path = os.path.join(datasets_dir(root), f"{_safe_filename(ds_name)}.json")
                    if os.path.isfile(ds_meta_path):
                        try:
                            with open(ds_meta_path, "r", encoding="utf-8") as dsfh:
                                ds_meta = json.load(dsfh)
                        except Exception:
                            ds_meta = {}
            if not isinstance(ds_meta, dict):
                ds_meta = {}

            from chain_replay_ml.diagnostics_studio.feature_partition import partition_diagnostic_rows
            partition = partition_diagnostic_rows(
                artifacts.get("rows") or [],
                data_dir=root,
                dataset_metadata=ds_meta,
            )
            reg_features = {str(r.get("feature") or "").strip() for r in partition.registry_rows}
            base_features = {str(r.get("feature") or "").strip() for r in partition.base_pipeline_rows}
            exp_features = {str(r.get("feature") or "").strip() for r in partition.experimental_rows}

            if not model_pid:
                ds_name_str = str(ds_meta.get("dataset_name") or model_cfg.get("dataset") or "")
                if "PL" in ds_name_str:
                    import re
                    m_pl = re.search(r"PL_?(\d+)", ds_name_str, re.IGNORECASE)
                    if m_pl:
                        model_pid = f"PL_{int(m_pl.group(1)):04d}"
            if not model_pid and exp_features:
                model_pid = "PL_EXP"
            if not model_snap:
                model_snap = "snap_v1"

            evidence_rows: list[dict[str, Any]] = []
            for row in artifacts.get("rows") or []:
                if not isinstance(row, dict):
                    continue
                rec = _normalize_recommendation(row.get("recommendation"))
                if not rec or rec not in allowed:
                    continue
                feat = str(row.get("feature") or row.get("feature_name") or "").strip()
                if not feat:
                    continue

                if feat in base_features:
                    f_source = "base_pipeline"
                elif feat in exp_features:
                    f_source = "experimental"
                elif feat in reg_features:
                    f_source = "registry"
                else:
                    f_source = "registry"

                evidence_rows.append(
                    {
                        "evidence_id": f"ev_{run_id}_{safe}_{feat}",
                        "feature_name": feat,
                        "feature_source": f_source,
                        "pipeline_id": model_pid if f_source == "experimental" else None,
                        "pipeline_snapshot_id": model_snap if f_source == "experimental" else None,
                        "recommendation": rec,
                        "validation_run_id": run_id,
                        "model_name": display_model,
                        "target_column": target_col,
                        "holdout_rank": row.get("holdout_rank"),
                        "unseen_rank": row.get("unseen_rank"),
                        "rank_change": row.get("rank_change"),
                        "relative_imp_drop": row.get("relative_imp_drop"),
                        "drift_severity": row.get("drift_severity"),
                        "evidence_detail_json": row.get("recommendation_detail"),
                        "run_timestamp": generated_date,
                    }
                )

            if evidence_rows:
                sqlite_res = append_validation_evidence(
                    conn,
                    context=context,
                    evidence_rows=evidence_rows,
                    policy=policy,
                )
        finally:
            conn.close()
    except Exception as exc:
        sqlite_res = {"error": str(exc)}

    return {
        "ok": True,
        "model_name": display_model,
        "production_validation_run_id": run_id,
        "inserted": inserted,
        "updated": updated,
        "skipped": skipped,
        "persisted_recommendations": sorted(allowed),
        "entry_count": len(entries),
        "feature_count": len((doc.get("summary") or {}).get("by_feature") or {}),
        "artifacts_dir": artifacts.get("artifacts_dir"),
        "sqlite_evidence": sqlite_res,
    }


__all__ = [
    "STORAGE",
    "VALID_RECOMMENDATIONS",
    "compute_recommendation_strength",
    "ensure_run_id",
    "get_recommendation_summary",
    "ignore_recommendation",
    "list_recommendation_history",
    "load_recommendation_store",
    "recommendation_strength_stars",
    "recommended_for_removal",
    "rebuild_summary",
    "save_recommendation_store",
    "storage_path",
    "unignore_recommendation",
    "update_registry_recommendations",
    "persist_validation_evidence",
]

persist_validation_evidence = update_registry_recommendations
