"""Versioned stage artifacts — the only allowed contract between stages.

Principle
---------
Every stage publishes an immutable, versioned artifact. The next stage
consumes *only* that artifact (by id). Stages must not read each other's
mutable working tables (feature_profiles, family_review, ...) as inputs.

Pipeline (artifact kinds)
-------------------------
  analysis_dataset
       |
  correlation
       |
  hca_families
       |
  mutual_information   --+
  permutation          --+-> discovery_rating
       |                 |
  discovery_bundle  <----+   (frozen Discovery Complete)
       |
  experiment_hypothesis      (feature-set hypothesis)
       |
  experiment_result          (Train + Holdout + WF + SHAP + Validation)
       |
  champion
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Sequence

from .analysis_lab_store import _AnalysisDb, _now_iso

KIND_ANALYSIS_DATASET = "analysis_dataset"
KIND_CORRELATION = "correlation"
KIND_HCA_FAMILIES = "hca_families"
KIND_MUTUAL_INFORMATION = "mutual_information"
KIND_PERMUTATION = "permutation"
KIND_DISCOVERY_RATING = "discovery_rating"
KIND_DISCOVERY_BUNDLE = "discovery_bundle"
KIND_EXPERIMENT_HYPOTHESIS = "experiment_hypothesis"
KIND_EXPERIMENT_RESULT = "experiment_result"
KIND_CHAMPION = "champion"

ARTIFACT_KINDS: tuple[str, ...] = (
    KIND_ANALYSIS_DATASET,
    KIND_CORRELATION,
    KIND_HCA_FAMILIES,
    KIND_MUTUAL_INFORMATION,
    KIND_PERMUTATION,
    KIND_DISCOVERY_RATING,
    KIND_DISCOVERY_BUNDLE,
    KIND_EXPERIMENT_HYPOTHESIS,
    KIND_EXPERIMENT_RESULT,
    KIND_CHAMPION,
)

ARTIFACT_CONSUMES: dict[str, tuple[str, ...]] = {
    KIND_ANALYSIS_DATASET: (),
    KIND_CORRELATION: (KIND_ANALYSIS_DATASET,),
    KIND_HCA_FAMILIES: (KIND_CORRELATION,),
    KIND_MUTUAL_INFORMATION: (KIND_ANALYSIS_DATASET,),
    KIND_PERMUTATION: (KIND_ANALYSIS_DATASET,),
    KIND_DISCOVERY_RATING: (
        KIND_CORRELATION,
        KIND_HCA_FAMILIES,
        KIND_MUTUAL_INFORMATION,
        KIND_PERMUTATION,
    ),
    KIND_DISCOVERY_BUNDLE: (
        KIND_HCA_FAMILIES,
        KIND_DISCOVERY_RATING,
    ),
    KIND_EXPERIMENT_HYPOTHESIS: (KIND_DISCOVERY_BUNDLE,),
    KIND_EXPERIMENT_RESULT: (KIND_EXPERIMENT_HYPOTHESIS,),
    KIND_CHAMPION: (KIND_EXPERIMENT_RESULT,),
}

SCHEMA_VERSIONS: dict[str, str] = {k: "1" for k in ARTIFACT_KINDS}


def ensure_artifacts_schema(conn: Any) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS stage_artifacts (
            artifact_id TEXT NOT NULL PRIMARY KEY,
            run_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            schema_version TEXT NOT NULL,
            version INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            parent_ids_json TEXT NOT NULL DEFAULT '[]',
            payload_json TEXT NOT NULL,
            label TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_stage_artifacts_run_kind
        ON stage_artifacts(run_id, kind, version DESC)
        """
    )


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _content_hash(payload: dict[str, Any], parent_ids: Sequence[str]) -> str:
    """Full SHA-256 hex — used as the public fingerprint."""
    blob = _canonical_json({"parents": list(parent_ids), "payload": payload})
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _next_version(conn: Any, run_id: str, kind: str) -> int:
    row = conn.execute(
        """
        SELECT MAX(version) AS v FROM stage_artifacts
        WHERE run_id = ? AND kind = ?
        """,
        (run_id, kind),
    ).fetchone()
    cur = int(row["v"] or 0) if row else 0
    return cur + 1


def _make_artifact_id(kind: str, version: int, stamp: str) -> str:
    """Human-readable ids for key handoff artifacts; others stay kind-vN-hex."""
    day = str(stamp or "")[:10].replace("-", "") or "00000000"
    if kind == KIND_DISCOVERY_BUNDLE:
        return f"db_{day}_{version:03d}"
    if kind == KIND_EXPERIMENT_HYPOTHESIS:
        return f"hyp_{day}_{version:03d}"
    if kind == KIND_EXPERIMENT_RESULT:
        return f"res_{day}_{version:03d}"
    if kind == KIND_CHAMPION:
        return f"champ_{day}_{version:03d}"
    return f"{kind}-v{version}-{uuid.uuid4().hex[:8]}"


def publish_artifact(
    data_dir: str,
    run_id: str,
    kind: str,
    payload: dict[str, Any],
    *,
    parent_ids: Sequence[str] | None = None,
    label: str = "",
    reuse_identical: bool = True,
) -> dict[str, Any]:
    """Publish an immutable versioned artifact. Returns the artifact record."""
    rid = str(run_id or "").strip()
    k = str(kind or "").strip()
    if not rid:
        raise ValueError("run_id is required")
    if k not in ARTIFACT_KINDS:
        raise ValueError(f"Unknown artifact kind {k!r}")
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")

    parents = [str(p) for p in (parent_ids or []) if str(p).strip()]
    if parents:
        loaded_parents = [load_artifact(data_dir, p) for p in parents]
        if any(p is None for p in loaded_parents):
            raise ValueError("One or more parent artifact ids are unknown")
        allowed = set(ARTIFACT_CONSUMES.get(k) or ())
        if allowed:
            got = {str(p.get("kind")) for p in loaded_parents if p}
            unexpected = got - allowed
            if unexpected:
                raise ValueError(
                    f"Artifact {k} cannot consume kinds {sorted(unexpected)}; "
                    f"allowed={sorted(allowed)}"
                )

    schema = SCHEMA_VERSIONS.get(k, "1")
    body = dict(payload)
    body.setdefault("kind", k)
    body.setdefault("schema_version", schema)
    fingerprint = _content_hash(body, parents)
    stamp = _now_iso()

    with _AnalysisDb(data_dir) as conn:
        ensure_artifacts_schema(conn)
        if reuse_identical:
            prev = conn.execute(
                """
                SELECT * FROM stage_artifacts
                WHERE run_id = ? AND kind = ?
                ORDER BY version DESC LIMIT 1
                """,
                (rid, k),
            ).fetchone()
            if prev and str(prev["content_hash"]) == fingerprint:
                return _row_to_artifact(dict(prev))

        version = _next_version(conn, rid, k)
        aid = _make_artifact_id(k, version, stamp)
        # Avoid PK collision if same day/version somehow reused
        existing = conn.execute(
            "SELECT 1 FROM stage_artifacts WHERE artifact_id = ?",
            (aid,),
        ).fetchone()
        if existing:
            aid = f"{aid}_{uuid.uuid4().hex[:6]}"
        conn.execute(
            """
            INSERT INTO stage_artifacts (
                artifact_id, run_id, kind, schema_version, version,
                content_hash, parent_ids_json, payload_json, label, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                aid,
                rid,
                k,
                schema,
                version,
                fingerprint,
                _canonical_json(parents),
                _canonical_json(body),
                str(label or "").strip() or f"{k} v{version}",
                stamp,
            ),
        )
    return load_artifact(data_dir, aid) or {"artifact_id": aid}


def load_artifact(data_dir: str, artifact_id: str) -> dict[str, Any] | None:
    aid = str(artifact_id or "").strip()
    if not aid:
        return None
    with _AnalysisDb(data_dir) as conn:
        ensure_artifacts_schema(conn)
        row = conn.execute(
            "SELECT * FROM stage_artifacts WHERE artifact_id = ?",
            (aid,),
        ).fetchone()
        return _row_to_artifact(dict(row)) if row else None


def latest_artifact(
    data_dir: str, run_id: str, kind: str
) -> dict[str, Any] | None:
    rid = str(run_id or "").strip()
    k = str(kind or "").strip()
    if not rid or not k:
        return None
    with _AnalysisDb(data_dir) as conn:
        ensure_artifacts_schema(conn)
        row = conn.execute(
            """
            SELECT * FROM stage_artifacts
            WHERE run_id = ? AND kind = ?
            ORDER BY version DESC LIMIT 1
            """,
            (rid, k),
        ).fetchone()
        return _row_to_artifact(dict(row)) if row else None


def list_artifacts(
    data_dir: str,
    run_id: str,
    *,
    kind: str | None = None,
) -> list[dict[str, Any]]:
    rid = str(run_id or "").strip()
    if not rid:
        return []
    with _AnalysisDb(data_dir) as conn:
        ensure_artifacts_schema(conn)
        if kind:
            rows = conn.execute(
                """
                SELECT * FROM stage_artifacts
                WHERE run_id = ? AND kind = ?
                ORDER BY kind, version
                """,
                (rid, str(kind)),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM stage_artifacts
                WHERE run_id = ?
                ORDER BY created_at, kind, version
                """,
                (rid,),
            ).fetchall()
    return [_row_to_artifact(dict(r)) for r in rows]


def artifact_lineage(
    data_dir: str, artifact_id: str
) -> list[dict[str, Any]]:
    """Ancestors first, this artifact last."""
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()

    def _walk(aid: str) -> None:
        if aid in seen:
            return
        seen.add(aid)
        art = load_artifact(data_dir, aid)
        if not art:
            return
        for pid in art.get("parent_ids") or []:
            _walk(str(pid))
        chain.append(art)

    _walk(str(artifact_id or "").strip())
    return chain


def _row_to_artifact(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    try:
        item["parent_ids"] = json.loads(str(item.get("parent_ids_json") or "[]"))
    except Exception:
        item["parent_ids"] = []
    try:
        item["payload"] = json.loads(str(item.get("payload_json") or "{}"))
    except Exception:
        item["payload"] = {}
    fp = str(item.get("content_hash") or "")
    item["fingerprint"] = fp
    item["fingerprint_short"] = (fp[:12] + "…") if len(fp) > 12 else fp
    item["artifact_ref"] = (
        f"{item.get('kind')}@v{item.get('version')} "
        f"({item.get('artifact_id')})"
    )
    card = dict((item.get("payload") or {}).get("card") or {})
    item["card"] = card
    return item


def verify_artifact_fingerprint(
    data_dir: str,
    artifact_id: str,
    expected_fingerprint: str,
) -> dict[str, Any]:
    """Ensure stored artifact still matches the fingerprint captured at bind time."""
    art = require_artifact(data_dir, artifact_id)
    got = str(art.get("fingerprint") or art.get("content_hash") or "")
    want = str(expected_fingerprint or "").strip()
    if not want:
        raise ValueError(
            f"No fingerprint recorded for artifact {artifact_id!r} — "
            "re-create the experiment from a frozen discovery_bundle."
        )
    if got != want:
        raise ValueError(
            f"Artifact fingerprint mismatch for {artifact_id!r}.\n"
            f"Expected: {want}\n"
            f"Actual:   {got}\n"
            "The bundle was modified or replaced — freeze Discovery again "
            "and create a new experiment."
        )
    return art


def format_artifact_card(art: dict[str, Any]) -> str:
    """Human-readable card for Discovery Bundle / Hypothesis / Result."""
    card = dict(art.get("card") or (art.get("payload") or {}).get("card") or {})
    lines = [
        f"ID:           {art.get('artifact_id')}",
        f"Fingerprint:  {art.get('fingerprint_short') or art.get('fingerprint')}",
    ]
    if card.get("dataset"):
        lines.append(f"Dataset:      {card.get('dataset')}")
    if card.get("n_features") is not None:
        lines.append(f"Features:     {card.get('n_features')}")
    if card.get("n_families") is not None:
        lines.append(f"Families:     {card.get('n_families')}")
    if card.get("n_experiment_reps") is not None:
        lines.append(f"Exp reps:     {card.get('n_experiment_reps')}")
    if art.get("kind"):
        lines.insert(0, f"Kind:         {art.get('kind')}")
    return "\n".join(lines)


def require_artifact(
    data_dir: str,
    artifact_id: str,
    *,
    expected_kind: str | None = None,
) -> dict[str, Any]:
    """Load artifact or raise — sole stage input gate."""
    art = load_artifact(data_dir, artifact_id)
    if not art:
        raise ValueError(
            f"Required stage artifact not found: {artifact_id!r}. "
            "Re-run the previous stage to publish it."
        )
    if expected_kind and str(art.get("kind")) != str(expected_kind):
        raise ValueError(
            f"Expected artifact kind {expected_kind!r}, "
            f"got {art.get('kind')!r} ({artifact_id})"
        )
    return art


def publish_discovery_bundle(
    data_dir: str,
    run_id: str,
    *,
    min_size: int = 2,
    label: str = "",
    feature_selection: dict[str, Any] | None = None,
    final_feature_dataset: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Freeze Discovery Complete into an immutable discovery_bundle.

    This snapshot is the only legal input for experiment hypotheses.
    Working tables may change later; the bundle does not.

    ``feature_selection`` / ``final_feature_dataset`` freeze *how* the feature
    set was built (HCA Top-N vs Corr+Perm, thresholds, etc.).
    """
    from .analysis_family_review import (
        FILTER_ALL,
        current_experiment_reps,
        discovery_readiness,
        load_families_with_reviews,
    )
    from .analysis_feature_selection import (
        STRATEGY_HCA,
        build_final_feature_dataset,
        build_selection_config,
        normalize_strategy,
    )
    from .analysis_hca import load_families

    rid = str(run_id or "").strip()
    sel_cfg = build_selection_config(
        **{
            k: v
            for k, v in dict(feature_selection or {}).items()
            if k
            in (
                "strategy",
                "representative_policy",
                "top_n",
                "correlation_threshold",
                "permutation_threshold",
                "min_family_size",
            )
        }
    ) if feature_selection else build_selection_config(STRATEGY_HCA)
    sid = normalize_strategy(sel_cfg.get("strategy"))

    ready = discovery_readiness(
        data_dir, rid, min_size=min_size, strategy=sid
    )
    if not ready.get("ready_to_create"):
        raise ValueError(
            "Cannot freeze discovery_bundle — Discovery is not ready.\n"
            + str(ready.get("banner_text") or "")
        )

    # Resolve Final Feature Dataset (authoritative selected feature list)
    ffd = dict(final_feature_dataset or {})
    if not ffd.get("features"):
        ffd = build_final_feature_dataset(
            data_dir, rid, sel_cfg, min_family_size=min_size
        )
    sel_cfg = dict(ffd.get("feature_selection") or sel_cfg)

    families = load_families(data_dir, rid, min_size=min_size)
    reviews = load_families_with_reviews(
        data_dir, rid, min_size=min_size, status_filter=FILTER_ALL
    )
    reps = dict(ffd.get("family_reps") or {}) or current_experiment_reps(
        data_dir, rid, min_size=min_size
    )

    family_rows: list[dict[str, Any]] = list(ffd.get("families") or [])
    if not family_rows and sid == STRATEGY_HCA:
        for fam in reviews:
            fid = str(fam.get("family_id") or "")
            members = list(fam.get("members") or [])
            family_rows.append(
                {
                    "family_id": fid,
                    "family_label": fam.get("family_label"),
                    "size": fam.get("size") or len(members),
                    "members": members,
                    "suggested_representative": fam.get("suggested_representative"),
                    "experiment_representative": reps.get(fid),
                    "representative": reps.get(fid),
                    "confidence": fam.get("confidence"),
                    "score_gap": fam.get("score_gap"),
                    "review_status": fam.get("review_status"),
                    "suggested_score": fam.get("suggested_score"),
                }
            )

    rating_rows: list[dict[str, Any]] = []
    with _AnalysisDb(data_dir) as conn:
        rows = conn.execute(
            """
            SELECT feature_name, feature_score, rating_score, rating_action,
                   rating_confidence, rating_reason, recommendation, reason,
                   rating_mi_pct, rating_perm_pct, rating_abs_corr,
                   rating_family_id, rating_family_label
            FROM feature_profiles
            WHERE run_id = ?
            """,
            (rid,),
        ).fetchall()
        for r in rows:
            rating_rows.append(dict(r))

    dataset_label = ""
    n_features_ds = None
    try:
        with _AnalysisDb(data_dir) as conn:
            run = conn.execute(
                "SELECT dataset_id, dataset_hash FROM analysis_runs WHERE run_id = ?",
                (rid,),
            ).fetchone()
            if run:
                ds = conn.execute(
                    "SELECT dataset_id, name, features, dataset_hash FROM datasets WHERE dataset_id = ?",
                    (str(run["dataset_id"]),),
                ).fetchone()
                if ds:
                    dataset_label = (
                        f"{ds['name'] or ds['dataset_id']} "
                        f"(hash={(str(ds['dataset_hash']) or '')[:12]})"
                    )
                    n_features_ds = ds["features"]
    except Exception:
        pass

    parent_ids: list[str] = []
    for kind in (KIND_HCA_FAMILIES, KIND_DISCOVERY_RATING):
        prev = latest_artifact(data_dir, rid, kind)
        if prev:
            parent_ids.append(str(prev["artifact_id"]))

    selected_features = list(ffd.get("features") or list(reps.values()))
    payload = {
        "run_id": rid,
        "n_families": len(family_rows) if family_rows else len(families),
        "n_experiment_reps": len(reps),
        "n_selected_features": len(selected_features),
        "family_reps": dict(reps),
        "selected_features": selected_features,
        "families": family_rows,
        "discovery_ratings": rating_rows,
        "feature_selection": sel_cfg,
        "final_feature_dataset": {
            "kind": "final_feature_dataset",
            "count": len(selected_features),
            "features": selected_features,
            "hash": ffd.get("hash"),
            "pipeline": ffd.get("pipeline"),
            "n_input_features": ffd.get("n_input_features"),
            "feature_selection": sel_cfg,
        },
        "feature_set": ffd.get("feature_set"),
        "readiness": {
            "complete": bool(ready.get("complete")),
            "ready_to_create": bool(ready.get("ready_to_create")),
            "n_needs_review": ready.get("n_needs_review"),
            "strategy": sid,
        },
        "card": {
            "dataset": dataset_label or rid,
            "n_features": int(n_features_ds)
            if n_features_ds is not None
            else len(rating_rows),
            "n_families": len(family_rows) if family_rows else len(families),
            "n_experiment_reps": len(reps),
            "n_selected_features": len(selected_features),
            "strategy": sel_cfg.get("strategy_short") or sid,
            "representative_policy": sel_cfg.get("representative_policy_label"),
        },
        "frozen_at": _now_iso(),
    }
    return publish_artifact(
        data_dir,
        rid,
        KIND_DISCOVERY_BUNDLE,
        payload,
        parent_ids=parent_ids,
        label=label or "Discovery Complete bundle",
    )


def publish_module_artifact(
    data_dir: str,
    run_id: str,
    module_id: str,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Map a completed Analysis module to a stage artifact (lineage + summary)."""
    mid = str(module_id or "").strip()
    kind_map = {
        "correlation": KIND_CORRELATION,
        "hca": KIND_HCA_FAMILIES,
        "mutual_information": KIND_MUTUAL_INFORMATION,
        "permutation": KIND_PERMUTATION,
        "feature_scorecard": KIND_DISCOVERY_RATING,
    }
    kind = kind_map.get(mid)
    if not kind:
        return None

    parents: list[str] = []
    for pk in ARTIFACT_CONSUMES.get(kind) or ():
        prev = latest_artifact(data_dir, run_id, pk)
        if prev:
            parents.append(str(prev["artifact_id"]))

    payload = {
        "module_id": mid,
        "summary": dict(summary or {}),
        "published_at": _now_iso(),
    }
    return publish_artifact(
        data_dir,
        run_id,
        kind,
        payload,
        parent_ids=parents,
        label=f"{mid} completed",
    )


__all__ = [
    "ARTIFACT_CONSUMES",
    "ARTIFACT_KINDS",
    "KIND_ANALYSIS_DATASET",
    "KIND_CHAMPION",
    "KIND_CORRELATION",
    "KIND_DISCOVERY_BUNDLE",
    "KIND_DISCOVERY_RATING",
    "KIND_EXPERIMENT_HYPOTHESIS",
    "KIND_EXPERIMENT_RESULT",
    "KIND_HCA_FAMILIES",
    "KIND_MUTUAL_INFORMATION",
    "KIND_PERMUTATION",
    "SCHEMA_VERSIONS",
    "artifact_lineage",
    "ensure_artifacts_schema",
    "format_artifact_card",
    "latest_artifact",
    "list_artifacts",
    "load_artifact",
    "publish_artifact",
    "publish_discovery_bundle",
    "publish_module_artifact",
    "require_artifact",
    "verify_artifact_fingerprint",
]
