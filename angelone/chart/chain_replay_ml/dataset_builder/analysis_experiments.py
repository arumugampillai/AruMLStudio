"""Experiment Manager — feature-set snapshots after Discovery.

An experiment freezes a *feature-set snapshot* (one Experiment Representative
per HCA family). Training produces a model; Validation scores belong to that
model / experiment_result — not to the snapshot itself.

**Stage contract:** create_experiment consumes only a versioned
``discovery_bundle`` artifact — never live family_review / profile tables.

Lifecycle (owned by Experiment Manager — one Train action):

  Created → Training → Model Produced → Validated → Champion
             Train → Model Package → Holdout → WF → SHAP → Validation
             → experiment_result
"""
from __future__ import annotations

import json
import os
from typing import Any, Callable

from .analysis_artifacts import (
    KIND_DISCOVERY_BUNDLE,
    KIND_EXPERIMENT_HYPOTHESIS,
    KIND_EXPERIMENT_RESULT,
    KIND_CHAMPION,
    format_artifact_card,
    latest_artifact,
    load_artifact,
    publish_artifact,
    publish_discovery_bundle,
    require_artifact,
    verify_artifact_fingerprint,
)
from .analysis_lab_store import _AnalysisDb, _now_iso

ProgressCb = Callable[[float, str], None]

# Explicit lifecycle statuses (scores attach after Model Produced / Validated)
STATUS_CREATED = "Created"
STATUS_TRAINING = "Training"
STATUS_MODEL_PRODUCED = "Model Produced"
STATUS_VALIDATED = "Validated"
STATUS_CHAMPION = "Champion"
STATUS_ARCHIVED = "Archived"

# Backward-compatible aliases
STATUS_READY = STATUS_CREATED
STATUS_COMPLETED = STATUS_VALIDATED
STATUS_DRAFT = STATUS_CREATED
STATUS_TRAINED = STATUS_MODEL_PRODUCED

VALIDATION_PENDING = "Pending"
VALIDATION_GOOD = "Good"
VALIDATION_BEST = "Best"
VALIDATION_WORSE = "Worse"
VALIDATION_UNSTABLE = "Unstable"

_LEGACY_EXP_STATUS = {
    "draft": STATUS_CREATED,
    "Created": STATUS_CREATED,
    "Ready": STATUS_CREATED,
    "ready": STATUS_CREATED,
    "Training": STATUS_TRAINING,
    "training": STATUS_TRAINING,
    "trained": STATUS_MODEL_PRODUCED,
    "Model Produced": STATUS_MODEL_PRODUCED,
    "Completed": STATUS_VALIDATED,
    "completed": STATUS_VALIDATED,
    "validated": STATUS_VALIDATED,
    "Validated": STATUS_VALIDATED,
    "Champion": STATUS_CHAMPION,
    "archived": STATUS_ARCHIVED,
    "Archived": STATUS_ARCHIVED,
}


def normalize_experiment_status(status: str | None) -> str:
    st = str(status or "").strip()
    if st in _LEGACY_EXP_STATUS:
        return _LEGACY_EXP_STATUS[st]
    return st or STATUS_CREATED


def display_experiment_status(exp: dict[str, Any]) -> str:
    """Status for UI — Champion overrides Validated when promoted."""
    if int(exp.get("is_champion") or 0):
        return STATUS_CHAMPION
    return normalize_experiment_status(exp.get("status"))


def ensure_experiments_schema(conn: Any) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            experiment_id TEXT NOT NULL,
            run_id TEXT NOT NULL,
            name TEXT,
            notes TEXT,
            status TEXT,
            model_name TEXT,
            holdout_score REAL,
            walk_forward_score REAL,
            validation_label TEXT,
            validation_summary TEXT,
            created_at TEXT,
            updated_at TEXT,
            PRIMARY KEY (experiment_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS experiment_family_reps (
            experiment_id TEXT NOT NULL,
            family_id TEXT NOT NULL,
            family_label TEXT,
            representative TEXT NOT NULL,
            PRIMARY KEY (experiment_id, family_id)
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_experiments_run
        ON experiments(run_id)
        """
    )
    cols = {
        str(r[1])
        for r in conn.execute("PRAGMA table_info(experiments)").fetchall()
    }
    for col, typ in (
        ("discovery_bundle_id", "TEXT"),
        ("discovery_bundle_fingerprint", "TEXT"),
        ("hypothesis_artifact_id", "TEXT"),
        ("result_artifact_id", "TEXT"),
        ("is_champion", "INTEGER"),
        ("parent_experiment_id", "TEXT"),
        ("variant_changes", "TEXT"),
        ("train_device", "TEXT"),
        ("shap_device", "TEXT"),
        ("device_label", "TEXT"),
    ):
        if col not in cols:
            conn.execute(f"ALTER TABLE experiments ADD COLUMN {col} {typ}")


def _next_experiment_id(conn: Any, run_id: str) -> str:
    row = conn.execute(
        """
        SELECT experiment_id FROM experiments
        WHERE run_id = ?
        ORDER BY experiment_id DESC
        LIMIT 1
        """,
        (run_id,),
    ).fetchone()
    n = 1
    if row:
        raw = str(row["experiment_id"] or "")
        digits = ""
        if "-" in raw:
            digits = "".join(ch for ch in raw.split("-")[1] if ch.isdigit())
        try:
            n = int(digits or "0") + 1
        except ValueError:
            n = 1
    return f"Exp-{n:03d}"


def create_experiment(
    data_dir: str,
    run_id: str,
    *,
    name: str = "",
    notes: str = "",
    family_reps: dict[str, str] | None = None,
    discovery_bundle_id: str | None = None,
    freeze_discovery: bool = True,
    min_size: int = 2,
    parent_experiment_id: str | None = None,
    variant_changes: list[dict[str, Any]] | None = None,
    feature_selection: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a feature-set snapshot from a frozen discovery_bundle only.

    Does not read live family_review / feature_profiles for its input.
    If ``discovery_bundle_id`` is omitted, reuses the latest frozen bundle;
    only freezes a new one when none exists and ``freeze_discovery`` is True.
    ``family_reps`` may override reps *within* the bundle's family membership
    (for variants); unknown families are rejected.
    ``feature_selection`` is applied when freezing a new bundle.
    """
    del min_size  # membership comes from the bundle
    rid = str(run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required")

    bundle_id = str(discovery_bundle_id or "").strip()
    if feature_selection and freeze_discovery and not discovery_bundle_id:
        # New strategy config → freeze a fresh Final Feature Dataset
        bundle = publish_discovery_bundle(
            data_dir, rid, feature_selection=feature_selection
        )
        bundle_id = str(bundle["artifact_id"])
    elif not bundle_id:
        latest = latest_artifact(data_dir, rid, KIND_DISCOVERY_BUNDLE)
        if latest:
            bundle_id = str(latest["artifact_id"])
        elif freeze_discovery:
            bundle = publish_discovery_bundle(
                data_dir, rid, feature_selection=feature_selection
            )
            bundle_id = str(bundle["artifact_id"])
        else:
            raise ValueError(
                "No discovery_bundle artifact — freeze Discovery Complete first."
            )

    bundle = require_artifact(
        data_dir, bundle_id, expected_kind=KIND_DISCOVERY_BUNDLE
    )
    bundle_fp = str(bundle.get("fingerprint") or bundle.get("content_hash") or "")
    payload = dict(bundle.get("payload") or {})
    if str(payload.get("run_id") or rid) != rid:
        # allow missing run_id in payload; reject mismatch
        if payload.get("run_id") and str(payload.get("run_id")) != rid:
            raise ValueError("discovery_bundle run_id does not match")

    sel_cfg = dict(
        feature_selection
        or payload.get("feature_selection")
        or {}
    )
    selected_from_bundle = [
        str(f)
        for f in (
            payload.get("selected_features")
            or (payload.get("final_feature_dataset") or {}).get("features")
            or []
        )
        if str(f).strip()
    ]

    base_reps = {
        str(k): str(v)
        for k, v in dict(payload.get("family_reps") or {}).items()
        if str(k).strip() and str(v).strip()
    }
    fam_meta = {
        str(f.get("family_id")): f
        for f in (payload.get("families") or [])
        if f.get("family_id")
    }
    # Flat strategies: synthesize singleton families from selected features
    if not base_reps and selected_from_bundle:
        for i, feat in enumerate(selected_from_bundle):
            fid = f"flat_{i:04d}"
            base_reps[fid] = feat
            fam_meta[fid] = {
                "family_id": fid,
                "family_label": feat,
                "members": [feat],
                "representative": feat,
                "representatives": [feat],
            }
    if not base_reps:
        raise ValueError(
            "discovery_bundle has no family_reps / selected_features — "
            "re-freeze Discovery Complete with a Feature Selection Strategy."
        )

    reps = dict(base_reps)
    if family_reps:
        for fid, rep in family_reps.items():
            fid_s = str(fid).strip()
            rep_s = str(rep).strip()
            if fid_s not in fam_meta and fid_s not in base_reps:
                raise ValueError(
                    f"Family {fid_s!r} is not in discovery_bundle — "
                    "cannot invent families outside the frozen artifact."
                )
            members = list((fam_meta.get(fid_s) or {}).get("members") or [])
            if members and rep_s not in members:
                raise ValueError(
                    f"{rep_s!r} is not a member of family {fid_s} in the "
                    "discovery_bundle artifact."
                )
            reps[fid_s] = rep_s

    # Authoritative train list: bundle selected_features (Top-N) else primary reps
    selected_features = list(selected_from_bundle)
    if not selected_features:
        # Expand Top-N representatives stored on family rows
        for fid, rep in reps.items():
            meta = fam_meta.get(fid) or {}
            extra = [
                str(x)
                for x in (meta.get("representatives") or [])
                if str(x).strip()
            ]
            if extra:
                for x in extra:
                    if x not in selected_features:
                        selected_features.append(x)
            elif rep and rep not in selected_features:
                selected_features.append(str(rep))
    if not selected_features:
        selected_features = [str(v) for v in reps.values() if v]

    parent_id = str(parent_experiment_id or "").strip() or None
    changes_list = [dict(c) for c in (variant_changes or [])]
    changes_json = json.dumps(changes_list) if changes_list else None

    stamp = _now_iso()
    with _AnalysisDb(data_dir) as conn:
        ensure_experiments_schema(conn)
        eid = _next_experiment_id(conn, rid)
        display = str(name or "").strip() or eid
        conn.execute(
            """
            INSERT INTO experiments (
                experiment_id, run_id, name, notes, status,
                model_name, holdout_score, walk_forward_score,
                validation_label, validation_summary,
                created_at, updated_at,
                discovery_bundle_id, discovery_bundle_fingerprint,
                hypothesis_artifact_id, result_artifact_id, is_champion,
                parent_experiment_id, variant_changes
            ) VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, ?, NULL, ?, ?, ?, ?, NULL, NULL, 0, ?, ?)
            """,
            (
                eid,
                rid,
                display,
                str(notes or "").strip() or None,
                STATUS_CREATED,
                VALIDATION_PENDING,
                stamp,
                stamp,
                bundle_id,
                bundle_fp,
                parent_id,
                changes_json,
            ),
        )
        for fid, rep in reps.items():
            fam = fam_meta.get(fid) or {}
            label = str(fam.get("family_label") or fid)
            conn.execute(
                """
                INSERT INTO experiment_family_reps (
                    experiment_id, family_id, family_label, representative
                ) VALUES (?, ?, ?, ?)
                """,
                (eid, fid, label, str(rep)),
            )

    # Publish experiment_hypothesis artifact — sole train input later
    display_name = str(name or "").strip() or eid
    families_payload = [
        {
            "family_id": fid,
            "family_label": (fam_meta.get(fid) or {}).get("family_label")
            or fid,
            "representative": rep,
            "representatives": list(
                (fam_meta.get(fid) or {}).get("representatives")
                or ([rep] if rep else [])
            ),
            "members": list((fam_meta.get(fid) or {}).get("members") or []),
        }
        for fid, rep in reps.items()
    ]
    feature_set = build_feature_set(
        families=families_payload,
        features=selected_features,
        discovery_bundle_id=bundle_id,
        parent_experiment_id=parent_id,
        variant_changes=changes_list,
    )
    if sel_cfg:
        feature_set["feature_selection"] = sel_cfg
    hyp = publish_artifact(
        data_dir,
        rid,
        KIND_EXPERIMENT_HYPOTHESIS,
        {
            "experiment_id": eid,
            "name": display_name,
            "family_reps": reps,
            "families": families_payload,
            "selected_features": selected_features,
            "feature_set": feature_set,
            "feature_selection": sel_cfg or None,
            "final_feature_dataset": {
                "count": len(selected_features),
                "features": selected_features,
                "hash": feature_set.get("hash"),
                "feature_selection": sel_cfg or None,
            },
            "discovery_bundle_id": bundle_id,
            "discovery_bundle_fingerprint": bundle_fp,
            "parent_experiment_id": parent_id,
            "variant_changes": changes_list,
            "card": {
                "dataset": (payload.get("card") or {}).get("dataset"),
                "n_features": len(selected_features),
                "n_families": len(reps),
                "n_experiment_reps": len(reps),
                "feature_set_hash": str(feature_set.get("hash") or "")[:16],
                "strategy": (sel_cfg or {}).get("strategy_short")
                or (sel_cfg or {}).get("strategy"),
                "representative_policy": (sel_cfg or {}).get(
                    "representative_policy_label"
                ),
            },
        },
        parent_ids=[bundle_id],
        label=f"Feature-set Snapshot {eid}",
    )
    with _AnalysisDb(data_dir) as conn:
        ensure_experiments_schema(conn)
        conn.execute(
            """
            UPDATE experiments
            SET hypothesis_artifact_id = ?, updated_at = ?
            WHERE experiment_id = ?
            """,
            (hyp["artifact_id"], _now_iso(), eid),
        )
    return load_experiment(data_dir, eid) or {"experiment_id": eid}


def experiment_family_options(
    data_dir: str,
    experiment_id: str,
) -> list[dict[str, Any]]:
    """Families + member lists from the parent's frozen discovery_bundle."""
    base = load_experiment(data_dir, experiment_id)
    if not base:
        raise ValueError(f"Unknown experiment {experiment_id!r}")
    bid = str(base.get("discovery_bundle_id") or "").strip()
    if not bid:
        raise ValueError(f"{experiment_id} has no discovery_bundle_id")
    bundle = require_artifact(
        data_dir, bid, expected_kind=KIND_DISCOVERY_BUNDLE
    )
    payload = dict(bundle.get("payload") or {})
    fam_meta = {
        str(f.get("family_id")): f
        for f in (payload.get("families") or [])
        if f.get("family_id")
    }
    out: list[dict[str, Any]] = []
    for r in base.get("family_reps") or []:
        fid = str(r.get("family_id") or "")
        cur = str(r.get("representative") or "")
        members = [
            str(m)
            for m in list((fam_meta.get(fid) or {}).get("members") or [])
            if m
        ]
        if not members and cur:
            members = [cur]
        alts = [m for m in members if m != cur]
        out.append(
            {
                "family_id": fid,
                "family_label": r.get("family_label") or fid,
                "current_representative": cur,
                "members": members,
                "alternate_members": alts,
                "can_change": len(alts) > 0,
            }
        )
    return out


def clone_experiment_variant(
    data_dir: str,
    experiment_id: str,
    *,
    changes: dict[str, str] | list[dict[str, Any]],
    name: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Create a variant that differs from parent in ≥1 HCA family representative.

    ``changes`` maps family_id → new representative (or a list of
    ``{family_id, representative}``). Rejects identical snapshots.
    Stores ``parent_experiment_id`` and family-level old→new changes.
    """
    base = load_experiment(data_dir, experiment_id)
    if not base:
        raise ValueError(f"Unknown experiment {experiment_id!r}")
    parent_id = str(base.get("experiment_id") or experiment_id)
    parent_reps = {
        str(r["family_id"]): str(r["representative"])
        for r in base.get("family_reps") or []
    }
    if not parent_reps:
        raise ValueError(f"{parent_id} has no family representatives")

    options = {
        str(o["family_id"]): o
        for o in experiment_family_options(data_dir, parent_id)
    }

    if isinstance(changes, dict):
        raw_items = [
            {"family_id": str(k), "representative": str(v)}
            for k, v in changes.items()
        ]
    else:
        raw_items = [dict(c) for c in (changes or [])]

    if not raw_items:
        raise ValueError(
            "Select at least one HCA family and choose a new representative."
        )

    change_rows: list[dict[str, Any]] = []
    new_reps = dict(parent_reps)
    for item in raw_items:
        fid = str(item.get("family_id") or "").strip()
        new_rep = str(
            item.get("representative") or item.get("new_representative") or ""
        ).strip()
        if not fid or not new_rep:
            raise ValueError("Each change needs family_id and representative")
        if fid not in parent_reps:
            raise ValueError(f"Family {fid!r} is not in parent {parent_id}")
        opt = options.get(fid) or {}
        members = list(opt.get("members") or [])
        if members and new_rep not in members:
            raise ValueError(
                f"{new_rep!r} is not a member of family {fid} "
                f"({opt.get('family_label') or fid})"
            )
        old_rep = parent_reps[fid]
        if new_rep == old_rep:
            continue  # not a real change
        new_reps[fid] = new_rep
        change_rows.append(
            {
                "family_id": fid,
                "family_label": str(
                    opt.get("family_label")
                    or next(
                        (
                            r.get("family_label")
                            for r in (base.get("family_reps") or [])
                            if str(r.get("family_id")) == fid
                        ),
                        fid,
                    )
                ),
                "old_representative": old_rep,
                "new_representative": new_rep,
            }
        )

    if not change_rows:
        raise ValueError(
            "At least one representative must differ from the parent. "
            "Pick a different member for a selected family."
        )
    if new_reps == parent_reps:
        raise ValueError(
            "Resulting feature-set snapshot is identical to the parent — "
            "refusing to create a duplicate variant."
        )

    change_summary = "; ".join(
        f"{c['family_label']}: {c['old_representative']} → {c['new_representative']}"
        for c in change_rows
    )
    note = str(notes or "").strip() or (
        f"Variant of {parent_id}: {change_summary}"
    )
    display = str(name or "").strip() or f"{parent_id}-var"

    return create_experiment(
        data_dir,
        str(base["run_id"]),
        name=display,
        notes=note,
        family_reps=new_reps,
        discovery_bundle_id=str(base.get("discovery_bundle_id") or ""),
        freeze_discovery=not bool(base.get("discovery_bundle_id")),
        parent_experiment_id=parent_id,
        variant_changes=change_rows,
    )


def clone_experiment_with_rep(
    data_dir: str,
    experiment_id: str,
    *,
    family_id: str,
    representative: str,
    name: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Backward-compatible single-family variant (must differ from parent)."""
    return clone_experiment_variant(
        data_dir,
        experiment_id,
        changes={str(family_id): str(representative)},
        name=name,
        notes=notes,
    )


def update_experiment_metrics(
    data_dir: str,
    experiment_id: str,
    *,
    model_name: str | None = None,
    holdout_score: float | None = None,
    walk_forward_score: float | None = None,
    validation_label: str | None = None,
    validation_summary: str | None = None,
    status: str | None = None,
    train_device: str | None = None,
    shap_device: str | None = None,
    device_label: str | None = None,
) -> dict[str, Any]:
    """Attach trained-model / validation metrics to an experiment."""
    eid = str(experiment_id or "").strip()
    if not eid:
        raise ValueError("experiment_id is required")
    stamp = _now_iso()
    with _AnalysisDb(data_dir) as conn:
        ensure_experiments_schema(conn)
        row = conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?",
            (eid,),
        ).fetchone()
        if not row:
            raise ValueError(f"Unknown experiment {eid!r}")
        cur = dict(row)
        new_model = (
            str(model_name).strip()
            if model_name is not None
            else cur.get("model_name")
        )
        new_hold = (
            float(holdout_score)
            if holdout_score is not None
            else cur.get("holdout_score")
        )
        new_wf = (
            float(walk_forward_score)
            if walk_forward_score is not None
            else cur.get("walk_forward_score")
        )
        new_label = (
            str(validation_label).strip()
            if validation_label is not None
            else cur.get("validation_label")
        ) or VALIDATION_PENDING
        new_sum = (
            str(validation_summary).strip()
            if validation_summary is not None
            else cur.get("validation_summary")
        )
        new_status = normalize_experiment_status(
            str(status).strip()
            if status is not None
            else cur.get("status")
        ) or STATUS_CREATED
        if new_hold is not None or new_wf is not None:
            if new_label and new_label != VALIDATION_PENDING:
                new_status = STATUS_VALIDATED
            elif new_status in (STATUS_CREATED, STATUS_TRAINING):
                new_status = STATUS_MODEL_PRODUCED
        elif new_model and new_status == STATUS_CREATED:
            new_status = STATUS_MODEL_PRODUCED

        new_train_device = (
            str(train_device).strip()
            if train_device is not None
            else cur.get("train_device")
        )
        new_shap_device = (
            str(shap_device).strip()
            if shap_device is not None
            else cur.get("shap_device")
        )
        new_device_label = (
            str(device_label).strip()
            if device_label is not None
            else cur.get("device_label")
        )
        if not new_device_label and new_train_device:
            new_device_label = new_train_device

        conn.execute(
            """
            UPDATE experiments
            SET model_name = ?,
                holdout_score = ?,
                walk_forward_score = ?,
                validation_label = ?,
                validation_summary = ?,
                status = ?,
                train_device = ?,
                shap_device = ?,
                device_label = ?,
                updated_at = ?
            WHERE experiment_id = ?
            """,
            (
                new_model,
                new_hold,
                new_wf,
                new_label,
                new_sum,
                new_status,
                new_train_device,
                new_shap_device,
                new_device_label,
                stamp,
                eid,
            ),
        )
    return load_experiment(data_dir, eid) or {}


def request_train_experiment(
    data_dir: str,
    experiment_id: str,
    *,
    target: str = "",
    on_progress: ProgressCb | None = None,
) -> dict[str, Any]:
    """Run Experiment Manager lifecycle (Train→…→experiment_result)."""
    from .analysis_experiment_lifecycle import run_experiment_lifecycle

    return run_experiment_lifecycle(
        data_dir,
        experiment_id,
        target=target,
        on_progress=on_progress,
    )


def features_fingerprint(features: list[str] | tuple[str, ...]) -> str:
    """Stable SHA-256 over the sorted selected-feature list."""
    import hashlib

    blob = "\n".join(sorted(str(f) for f in features)).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def build_feature_set(
    family_reps: list[dict[str, Any]] | None = None,
    *,
    features: list[str] | None = None,
    families: list[dict[str, Any]] | None = None,
    discovery_bundle_id: str | None = None,
    parent_experiment_id: str | None = None,
    variant_changes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Self-contained feature-set record (do not re-derive from Discovery later).

    Shape::

        {
          "count": 30,
          "hash": "a4d8c0…",
          "features": ["spot_price", …],
          "families": [
            {"family_id", "family_label", "representative", "changed": bool}
          ],
          "discovery_bundle_id": "…",
          "parent_experiment_id": "…",
          "changed_families": ["IV", "Price"],
        }
    """
    changed_ids: set[str] = set()
    changed_labels: list[str] = []
    for c in variant_changes or []:
        fid = str(c.get("family_id") or "").strip()
        label = str(c.get("family_label") or fid).strip()
        if fid:
            changed_ids.add(fid)
        if label and label not in changed_labels:
            changed_labels.append(label)

    fam_rows: list[dict[str, Any]] = []
    if families:
        for f in families:
            fid = str(f.get("family_id") or "").strip()
            if not fid:
                continue
            rep = str(f.get("representative") or "").strip()
            if not rep:
                continue
            fam_rows.append(
                {
                    "family_id": fid,
                    "family_label": str(f.get("family_label") or fid),
                    "representative": rep,
                    "changed": fid in changed_ids,
                    "members": list(f.get("members") or []),
                }
            )
    elif family_reps:
        for r in family_reps:
            fid = str(r.get("family_id") or "").strip()
            rep = str(r.get("representative") or "").strip()
            if not fid or not rep:
                continue
            fam_rows.append(
                {
                    "family_id": fid,
                    "family_label": str(r.get("family_label") or fid),
                    "representative": rep,
                    "changed": fid in changed_ids,
                    "members": list(r.get("members") or []),
                }
            )

    feat_list = [
        str(f).strip()
        for f in (features or [])
        if str(f).strip()
    ]
    if not feat_list:
        feat_list = list(
            dict.fromkeys(
                str(r["representative"]) for r in fam_rows if r.get("representative")
            )
        )
    # Keep family order for features when derived from families
    if fam_rows and features is None:
        feat_list = [str(r["representative"]) for r in fam_rows]

    return {
        "count": len(feat_list),
        "hash": features_fingerprint(feat_list) if feat_list else "",
        "features": feat_list,
        "families": fam_rows,
        "discovery_bundle_id": discovery_bundle_id or None,
        "parent_experiment_id": parent_experiment_id or None,
        "changed_families": changed_labels,
        "n_changed_families": len(changed_labels),
    }


def build_parent_diff(
    *,
    parent_experiment_id: str | None,
    parent_family_reps: list[dict[str, Any]] | None,
    current_family_reps: list[dict[str, Any]] | None,
    variant_changes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Side-by-side parent vs current representatives (self-contained)."""
    parent_map: dict[str, dict[str, Any]] = {}
    for r in parent_family_reps or []:
        fid = str(r.get("family_id") or "").strip()
        if fid:
            parent_map[fid] = {
                "family_id": fid,
                "family_label": str(r.get("family_label") or fid),
                "representative": str(r.get("representative") or ""),
            }
    current_map: dict[str, dict[str, Any]] = {}
    for r in current_family_reps or []:
        fid = str(r.get("family_id") or "").strip()
        if fid:
            current_map[fid] = {
                "family_id": fid,
                "family_label": str(r.get("family_label") or fid),
                "representative": str(r.get("representative") or ""),
            }

    # Prefer explicit variant_changes; else compute from maps
    changes: list[dict[str, Any]] = []
    if variant_changes:
        for c in variant_changes:
            changes.append(
                {
                    "family_id": str(c.get("family_id") or ""),
                    "family_label": str(
                        c.get("family_label") or c.get("family_id") or ""
                    ),
                    "old_representative": str(c.get("old_representative") or ""),
                    "new_representative": str(c.get("new_representative") or ""),
                }
            )
    else:
        for fid, cur in current_map.items():
            old = parent_map.get(fid)
            old_rep = str((old or {}).get("representative") or "")
            new_rep = str(cur.get("representative") or "")
            if old is not None and old_rep != new_rep:
                changes.append(
                    {
                        "family_id": fid,
                        "family_label": cur.get("family_label") or fid,
                        "old_representative": old_rep,
                        "new_representative": new_rep,
                    }
                )

    rows: list[dict[str, Any]] = []
    all_ids = list(
        dict.fromkeys(list(parent_map.keys()) + list(current_map.keys()))
    )
    changed_by_id = {str(c.get("family_id")): c for c in changes}
    for fid in all_ids:
        p = parent_map.get(fid) or {}
        c = current_map.get(fid) or {}
        label = (
            c.get("family_label")
            or p.get("family_label")
            or fid
        )
        old_rep = str(p.get("representative") or "")
        new_rep = str(c.get("representative") or "")
        changed = fid in changed_by_id or (
            bool(old_rep) and bool(new_rep) and old_rep != new_rep
        )
        rows.append(
            {
                "family_id": fid,
                "family_label": label,
                "parent_representative": old_rep or None,
                "representative": new_rep or None,
                "changed": changed,
            }
        )

    return {
        "parent_experiment_id": parent_experiment_id or None,
        "n_changed_families": len(changes),
        "changes": changes,
        "rows": rows,
    }


def format_experiment_details(
    exp: dict[str, Any],
    *,
    feature_set: dict[str, Any] | None = None,
    parent_diff: dict[str, Any] | None = None,
    result_payload: dict[str, Any] | None = None,
) -> str:
    """Complete Experiment Details card (self-contained record)."""
    eid = exp.get("experiment_id") or "—"
    st = display_experiment_status(exp)
    status = normalize_experiment_status(exp.get("status"))
    trained = status in (
        STATUS_MODEL_PRODUCED,
        STATUS_VALIDATED,
        STATUS_CHAMPION,
        STATUS_TRAINED,
    ) or bool(exp.get("result_artifact_id") or exp.get("model_name"))
    validated = status in (STATUS_VALIDATED, STATUS_CHAMPION) or (
        str(exp.get("validation_label") or "")
        not in ("", VALIDATION_PENDING)
        and bool(exp.get("result_artifact_id"))
    )
    is_champ = bool(int(exp.get("is_champion") or 0)) or status == STATUS_CHAMPION

    fs = dict(feature_set or exp.get("feature_set") or {})
    if not fs.get("features"):
        fs = build_feature_set(
            exp.get("family_reps") or [],
            features=(result_payload or {}).get("features")
            if result_payload
            else None,
            discovery_bundle_id=str(exp.get("discovery_bundle_id") or "") or None,
            parent_experiment_id=str(exp.get("parent_experiment_id") or "")
            or None,
            variant_changes=exp.get("variant_changes_list")
            or _parse_variant_changes(exp.get("variant_changes")),
        )

    features = [str(f) for f in (fs.get("features") or []) if str(f).strip()]
    families = list(fs.get("families") or [])
    if not families and exp.get("family_reps"):
        families = [
            {
                "family_id": r.get("family_id"),
                "family_label": r.get("family_label") or r.get("family_id"),
                "representative": r.get("representative"),
                "changed": False,
            }
            for r in (exp.get("family_reps") or [])
        ]

    diff = dict(parent_diff or exp.get("parent_diff") or {})
    if not diff and exp.get("parent_experiment_id"):
        diff = build_parent_diff(
            parent_experiment_id=str(exp.get("parent_experiment_id") or "")
            or None,
            parent_family_reps=exp.get("parent_family_reps"),
            current_family_reps=exp.get("family_reps"),
            variant_changes=exp.get("variant_changes_list")
            or _parse_variant_changes(exp.get("variant_changes")),
        )

    rp = dict(result_payload or exp.get("result_payload") or {})
    hold = exp.get("holdout_score")
    if hold is None:
        hold = rp.get("holdout_r2")
    wf = exp.get("walk_forward_score")
    if wf is None:
        wf = rp.get("walk_forward_r2")
    rmse = rp.get("holdout_rmse")
    val_label = exp.get("validation_label") or rp.get("validation_label") or "—"

    lines: list[str] = [
        f"Experiment: {eid}",
        f"Name: {exp.get('name') or eid}",
        "",
        "Status",
        f"  {'✓' if trained else '○'} Trained",
        f"  {'✓' if validated else '○'} Validated",
        f"  {'✓' if is_champ else '○'} Champion",
        "",
        "Metrics",
        f"  Holdout R²        {float(hold):.5f}" if hold is not None else "  Holdout R²        —",
        f"  Walk-forward R²   {float(wf):.5f}" if wf is not None else "  Walk-forward R²   —",
        f"  Holdout RMSE      {float(rmse):.5f}" if rmse is not None else "  Holdout RMSE      —",
        f"  Validation        {val_label}",
        "",
        "Final Feature Set",
        "────────────────────────────────────",
    ]
    if features:
        for i, feat in enumerate(features, start=1):
            lines.append(f"{i}. {feat}")
        lines.append(f"Total Features: {len(features)}")
    else:
        lines.append("(none)")
        lines.append("Total Features: 0")

    changed_fams = list(fs.get("changed_families") or [])
    if not changed_fams and diff.get("changes"):
        changed_fams = [
            str(c.get("family_label") or c.get("family_id") or "")
            for c in (diff.get("changes") or [])
            if c.get("family_label") or c.get("family_id")
        ]
    hash_full = str(fs.get("hash") or rp.get("features_fingerprint") or "")
    hash_short = (hash_full[:12] + "…") if len(hash_full) > 12 else (hash_full or "—")

    lines.extend(
        [
            "",
            "Feature Set Summary",
            f"  Selected Features : {fs.get('count') if fs.get('count') is not None else len(features)}",
            f"  Discovery Bundle   : {exp.get('discovery_bundle_id') or fs.get('discovery_bundle_id') or '—'}",
            f"  Feature Set Hash   : {hash_short}",
            f"  Created From       : {exp.get('parent_experiment_id') or '—'}",
            f"  Changed Families   : {', '.join(changed_fams) if changed_fams else '—'}",
            (
                f"  Model columns match: "
                f"{'Yes' if (rp.get('feature_names_match') if rp else fs.get('feature_names_match')) is not False else 'No'}"
                f" ({len(features)} selected"
                + (
                    f" = {len(rp.get('model_feature_names') or features)} model"
                    if rp or fs.get("model_feature_names")
                    else ""
                )
                + ")"
            ),
        ]
    )
    # Feature Selection provenance (how the set was built)
    sel_cfg = dict(
        rp.get("feature_selection")
        or fs.get("feature_selection")
        or exp.get("feature_selection")
        or {}
    )
    if sel_cfg:
        from .analysis_feature_selection import format_selection_summary

        lines.append("")
        lines.append(
            format_selection_summary(sel_cfg, n_features=len(features))
        )

    lines.extend(
        [
            "",
            "Feature Family Summary",
            f"  {'Family':<22} {'Representative':<36}",
            "  " + "-" * 58,
        ]
    )
    if families:
        for fam in families:
            label = str(fam.get("family_label") or fam.get("family_id") or "")
            rep = str(fam.get("representative") or "")
            mark = "  ← Changed" if fam.get("changed") else ""
            lines.append(f"  {label:<22} {rep:<36}{mark}".rstrip())
    else:
        lines.append("  (none)")

    parent_id = diff.get("parent_experiment_id") or exp.get("parent_experiment_id")
    if parent_id:
        lines.extend(["", "Changes from Parent", f"  Parent: {parent_id}", ""])
        parent_rows = list(diff.get("rows") or [])
        if parent_rows:
            lines.append("  Parent Feature Set")
            for row in parent_rows:
                label = str(row.get("family_label") or row.get("family_id") or "")
                prep = row.get("parent_representative") or "—"
                lines.append(f"    {label:<20} {prep}")
            lines.append("  ↓")
            lines.append("  Current Feature Set")
            for row in parent_rows:
                label = str(row.get("family_label") or row.get("family_id") or "")
                crep = row.get("representative") or "—"
                mark = "  ← Changed" if row.get("changed") else ""
                lines.append(f"    {label:<20} {crep}{mark}")
        elif diff.get("changes"):
            for c in diff.get("changes") or []:
                label = c.get("family_label") or c.get("family_id")
                lines.append(
                    f"  {label}: {c.get('old_representative')} → "
                    f"{c.get('new_representative')}  ← Changed"
                )
        else:
            lines.append("  (no family changes vs parent)")
    else:
        lines.extend(["", "Changes from Parent", "  (baseline — no parent)"])

    model_name = exp.get("model_name") or rp.get("model_name") or "—"
    model_path = rp.get("model_path") or ""
    train_dev = (
        exp.get("device_label")
        or exp.get("train_device")
        or rp.get("device_label")
        or rp.get("train_device")
        or "—"
    )
    shap_dev = exp.get("shap_device") or rp.get("shap_device") or "—"
    lines.extend(
        [
            "",
            "Model Information",
            f"  Target             {rp.get('target') or '—'}",
            f"  Model package      {model_name}",
            f"  Model path         {model_path or '—'}",
            f"  Train device       {train_dev}",
            f"  SHAP device        {shap_dev}",
            f"  Result artifact    {exp.get('result_artifact_id') or '—'}",
            f"  Snapshot artifact  {exp.get('hypothesis_artifact_id') or '—'}",
        ]
    )
    if is_champ:
        lines.extend(
            [
                "",
                "Champion Status",
                "  ★ Champion for this Discovery Bundle / run",
            ]
        )
    elif trained and not validated:
        lines.extend(["", "Champion Status", "  (train/validate first)"])
    else:
        lines.extend(["", "Champion Status", "  Not promoted"])

    if exp.get("fingerprint_ok") is False:
        lines.append(
            f"\n⚠ Fingerprint check failed: {exp.get('fingerprint_error') or ''}"
        )
    return "\n".join(lines)


def format_hypothesis_text(exp: dict[str, Any]) -> str:
    """Backward-compatible alias → full Experiment Details card."""
    return format_experiment_details(exp)


def _parse_variant_changes(raw: Any) -> list[dict[str, Any]]:
    if raw is None or raw == "":
        return []
    if isinstance(raw, list):
        return [dict(c) for c in raw]
    try:
        data = json.loads(str(raw))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if isinstance(data, list):
        return [dict(c) for c in data]
    return []


def _load_parent_family_reps(
    data_dir: str, parent_experiment_id: str | None
) -> list[dict[str, Any]]:
    pid = str(parent_experiment_id or "").strip()
    if not pid or not data_dir:
        return []
    try:
        with _AnalysisDb(data_dir) as conn:
            ensure_experiments_schema(conn)
            rows = conn.execute(
                """
                SELECT family_id, family_label, representative
                FROM experiment_family_reps
                WHERE experiment_id = ?
                ORDER BY family_label, family_id
                """,
                (pid,),
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception:
        return []


def _enrich_experiment(
    item: dict[str, Any],
    *,
    verify_bundle: bool = False,
    data_dir: str = "",
    load_result: bool = True,
) -> dict[str, Any]:
    out = dict(item)
    out["status"] = normalize_experiment_status(out.get("status"))
    if int(out.get("is_champion") or 0):
        out["status"] = STATUS_CHAMPION
    out["status_display"] = display_experiment_status(out)
    out["variant_changes_list"] = _parse_variant_changes(
        out.get("variant_changes")
    )

    result_payload: dict[str, Any] = {}
    if load_result and data_dir and out.get("result_artifact_id"):
        try:
            art = load_artifact(data_dir, str(out["result_artifact_id"]))
            result_payload = dict((art or {}).get("payload") or {})
            out["result_payload"] = result_payload
        except Exception:
            result_payload = {}

    feature_set = dict(result_payload.get("feature_set") or {})
    if not feature_set.get("features"):
        feature_set = build_feature_set(
            out.get("family_reps") or [],
            features=result_payload.get("features") or result_payload.get(
                "selected_features"
            ),
            discovery_bundle_id=str(out.get("discovery_bundle_id") or "") or None,
            parent_experiment_id=str(out.get("parent_experiment_id") or "")
            or None,
            variant_changes=out.get("variant_changes_list"),
        )
    out["feature_set"] = feature_set
    out["feature_selection"] = dict(
        result_payload.get("feature_selection")
        or feature_set.get("feature_selection")
        or {}
    ) or None

    parent_reps = _load_parent_family_reps(
        data_dir, out.get("parent_experiment_id")
    )
    out["parent_family_reps"] = parent_reps
    parent_diff = dict(result_payload.get("parent_diff") or {})
    if not parent_diff.get("rows") and not parent_diff.get("changes"):
        parent_diff = build_parent_diff(
            parent_experiment_id=str(out.get("parent_experiment_id") or "")
            or None,
            parent_family_reps=parent_reps,
            current_family_reps=out.get("family_reps"),
            variant_changes=out.get("variant_changes_list"),
        )
    out["parent_diff"] = parent_diff

    out["fingerprint_ok"] = None
    bid = out.get("discovery_bundle_id")
    bfp = out.get("discovery_bundle_fingerprint")
    if verify_bundle and data_dir and bid and bfp:
        try:
            verify_artifact_fingerprint(data_dir, str(bid), str(bfp))
            out["fingerprint_ok"] = True
        except Exception as exc:
            out["fingerprint_ok"] = False
            out["fingerprint_error"] = str(exc)

    out["hypothesis_text"] = format_experiment_details(
        out,
        feature_set=feature_set,
        parent_diff=parent_diff,
        result_payload=result_payload or None,
    )
    out["details_text"] = out["hypothesis_text"]
    return out


def load_experiment(
    data_dir: str,
    experiment_id: str,
    *,
    verify_bundle: bool = False,
) -> dict[str, Any] | None:
    eid = str(experiment_id or "").strip()
    if not eid:
        return None
    with _AnalysisDb(data_dir) as conn:
        ensure_experiments_schema(conn)
        row = conn.execute(
            "SELECT * FROM experiments WHERE experiment_id = ?",
            (eid,),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        reps = conn.execute(
            """
            SELECT family_id, family_label, representative
            FROM experiment_family_reps
            WHERE experiment_id = ?
            ORDER BY family_label, family_id
            """,
            (eid,),
        ).fetchall()
        item["family_reps"] = [dict(r) for r in reps]
        return _enrich_experiment(
            item, verify_bundle=verify_bundle, data_dir=data_dir
        )


def list_experiments(data_dir: str, run_id: str) -> list[dict[str, Any]]:
    rid = str(run_id or "").strip()
    if not rid:
        return []
    with _AnalysisDb(data_dir) as conn:
        ensure_experiments_schema(conn)
        rows = conn.execute(
            """
            SELECT * FROM experiments
            WHERE run_id = ?
            ORDER BY experiment_id
            """,
            (rid,),
        ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            reps = conn.execute(
                """
                SELECT family_id, family_label, representative
                FROM experiment_family_reps
                WHERE experiment_id = ?
                ORDER BY family_label, family_id
                """,
                (item["experiment_id"],),
            ).fetchall()
            item["family_reps"] = [dict(r) for r in reps]
            out.append(
                _enrich_experiment(
                    item,
                    verify_bundle=False,
                    data_dir=data_dir,
                    load_result=False,
                )
            )
        return out


def delete_experiment(data_dir: str, experiment_id: str) -> dict[str, Any]:
    """Delete a not-yet-trained experiment (Created only).

    Rejects Training / Model Produced / Validated / Champion and any row that
    already has a model or experiment_result.
    """
    eid = str(experiment_id or "").strip()
    if not eid:
        raise ValueError("experiment_id is required")
    exp = load_experiment(data_dir, eid)
    if not exp:
        raise ValueError(f"Unknown experiment {eid!r}")
    st = display_experiment_status(exp)
    if st != STATUS_CREATED:
        raise ValueError(
            f"Only Created (not executed) experiments can be deleted "
            f"(status={st}). {eid} already progressed."
        )
    if exp.get("model_name") or exp.get("result_artifact_id"):
        raise ValueError(
            f"{eid} has a model / experiment_result — cannot delete."
        )
    if int(exp.get("is_champion") or 0):
        raise ValueError(f"{eid} is Champion — cannot delete.")

    with _AnalysisDb(data_dir) as conn:
        ensure_experiments_schema(conn)
        conn.execute(
            "DELETE FROM experiment_family_reps WHERE experiment_id = ?",
            (eid,),
        )
        conn.execute(
            "DELETE FROM experiments WHERE experiment_id = ?",
            (eid,),
        )
    return {
        "experiment_id": eid,
        "deleted": True,
        "message": f"Deleted {eid} (Created snapshot, not executed)",
    }


def format_champion_bundle_card(bundle: dict[str, Any] | None) -> str:
    """Human-readable Champion Bundle card for UI / export."""
    if not bundle:
        return "No Champion Bundle yet — promote a Validated experiment."
    payload = dict(bundle.get("payload") or bundle)
    hold = payload.get("holdout_score")
    wf = payload.get("walk_forward_score")
    model_file = (
        payload.get("model_file")
        or (
            os.path.basename(str(payload.get("model_path") or ""))
            if payload.get("model_path")
            else ""
        )
        or "—"
    )
    lines = [
        "Champion Bundle",
        f"Name                 {payload.get('name') or '—'}",
        f"Artifact             {bundle.get('artifact_id') or payload.get('artifact_id') or '—'}",
        f"Experiment           {payload.get('experiment_id') or '—'}",
        f"Discovery Bundle     {payload.get('discovery_bundle_id') or '—'}",
        f"Selected Features    {payload.get('n_selected_features') if payload.get('n_selected_features') is not None else '—'}",
        f"Families             {payload.get('n_families') if payload.get('n_families') is not None else '—'}",
        f"Feature Set Hash     {(str((payload.get('feature_set') or {}).get('hash') or payload.get('features_fingerprint') or '')[:12] + '…') if (payload.get('feature_set') or {}).get('hash') or payload.get('features_fingerprint') else '—'}",
        f"Model                {model_file}",
        f"Model package        {payload.get('model_name') or '—'}",
        f"Train device         {payload.get('device_label') or payload.get('train_device') or '—'}",
        f"SHAP device          {payload.get('shap_device') or '—'}",
        (
            f"Holdout              {float(hold):.5f}"
            if hold is not None
            else "Holdout              —"
        ),
        (
            f"Walk-forward         {float(wf):.5f}"
            if wf is not None
            else "Walk-forward         —"
        ),
        f"Validation           {payload.get('validation_label') or '—'}",
        f"Promoted             {payload.get('promoted_at') or '—'}",
    ]
    if payload.get("model_path"):
        lines.append(f"Model path           {payload.get('model_path')}")
    feats = list(
        (payload.get("feature_set") or {}).get("features")
        or payload.get("selected_features")
        or []
    )
    if feats:
        lines.append("")
        lines.append("Final Feature Set")
        lines.append("────────────────────────────────────")
        for i, feat in enumerate(feats, start=1):
            lines.append(f"{i}. {feat}")
        lines.append(f"Total Features: {len(feats)}")
    return "\n".join(lines)


def load_champion_bundle(
    data_dir: str,
    run_id: str = "",
    *,
    artifact_id: str | None = None,
) -> dict[str, Any] | None:
    """Load latest (or specific) Champion Bundle artifact for a run."""
    if artifact_id:
        art = load_artifact(data_dir, str(artifact_id))
        if art and str(art.get("kind")) == KIND_CHAMPION:
            return art
        return None
    rid = str(run_id or "").strip()
    if not rid:
        return None
    return latest_artifact(data_dir, rid, KIND_CHAMPION)


def promote_champion(
    data_dir: str,
    experiment_id: str,
) -> dict[str, Any]:
    """Promote a Validated experiment into a reusable Champion Bundle.

    Downstream steps (backtest, live prediction, deploy) should load this
    artifact — not scrape experiment rows.
    """
    eid = str(experiment_id or "").strip()
    exp = load_experiment(data_dir, eid, verify_bundle=True)
    if not exp:
        raise ValueError(f"Unknown experiment {eid!r}")
    st = str(exp.get("status") or "")
    if st not in (STATUS_VALIDATED, STATUS_CHAMPION):
        raise ValueError(
            f"{eid} must be Validated before Champion promotion "
            f"(status={exp.get('status')})"
        )
    if exp.get("fingerprint_ok") is False:
        raise ValueError(
            f"{eid} discovery_bundle fingerprint check failed — "
            "cannot promote."
        )
    rid = str(exp["run_id"])
    result_id = str(exp.get("result_artifact_id") or "").strip()
    if not result_id:
        raise ValueError(f"{eid} has no experiment_result artifact")

    result_art = load_artifact(data_dir, result_id)
    result_payload = dict((result_art or {}).get("payload") or {})
    feature_set = dict(result_payload.get("feature_set") or {})
    features = [
        str(f)
        for f in (
            feature_set.get("features")
            or result_payload.get("selected_features")
            or result_payload.get("features")
            or []
        )
        if str(f).strip()
    ]
    if not features:
        features = [
            str(r.get("representative"))
            for r in (exp.get("family_reps") or [])
            if r.get("representative")
        ]
    if not feature_set.get("features"):
        feature_set = build_feature_set(
            exp.get("family_reps") or [],
            features=features,
            discovery_bundle_id=str(exp.get("discovery_bundle_id") or "")
            or None,
            parent_experiment_id=str(exp.get("parent_experiment_id") or "")
            or None,
            variant_changes=exp.get("variant_changes_list")
            or _parse_variant_changes(exp.get("variant_changes")),
        )
    parent_diff = dict(result_payload.get("parent_diff") or {})
    if not parent_diff:
        parent_diff = build_parent_diff(
            parent_experiment_id=str(exp.get("parent_experiment_id") or "")
            or None,
            parent_family_reps=_load_parent_family_reps(
                data_dir, exp.get("parent_experiment_id")
            ),
            current_family_reps=exp.get("family_reps"),
            variant_changes=exp.get("variant_changes_list")
            or _parse_variant_changes(exp.get("variant_changes")),
        )

    model_name = str(
        exp.get("model_name") or result_payload.get("model_name") or ""
    ).strip()
    model_path = ""
    model_file = ""
    if model_name:
        try:
            from chain_replay_ml.training.paths import model_artifact_paths

            paths = model_artifact_paths(data_dir, model_name)
            for key in ("model_ubj", "model_json", "tuned_model_ubj"):
                p = str(paths.get(key) or "")
                if p and os.path.isfile(p):
                    model_path = p
                    model_file = os.path.basename(p)
                    break
            if not model_path:
                model_path = str(paths.get("model_ubj") or "")
                model_file = os.path.basename(model_path) if model_path else ""
        except Exception:
            model_path = ""
            model_file = ""

    stamp = _now_iso()
    day = str(stamp)[:10] or "0000-00-00"
    bundle_name = f"Champion-{day}"

    with _AnalysisDb(data_dir) as conn:
        ensure_experiments_schema(conn)
        # Demote prior champions back to Validated
        conn.execute(
            """
            UPDATE experiments
            SET is_champion = 0,
                status = CASE WHEN status = ? THEN ? ELSE status END,
                updated_at = ?
            WHERE run_id = ?
            """,
            (STATUS_CHAMPION, STATUS_VALIDATED, stamp, rid),
        )
        conn.execute(
            """
            UPDATE experiments
            SET is_champion = 1, status = ?, updated_at = ?
            WHERE experiment_id = ?
            """,
            (STATUS_CHAMPION, stamp, eid),
        )

    payload = {
        "bundle_kind": "champion_bundle",
        "name": bundle_name,
        "experiment_id": eid,
        "result_artifact_id": result_id,
        "discovery_bundle_id": exp.get("discovery_bundle_id"),
        "discovery_bundle_fingerprint": exp.get(
            "discovery_bundle_fingerprint"
        ),
        "feature_set": feature_set,
        "feature_selection": dict(
            result_payload.get("feature_selection")
            or feature_set.get("feature_selection")
            or {}
        )
        or None,
        "selected_features": features,
        "model_feature_names": list(
            result_payload.get("model_feature_names")
            or feature_set.get("model_feature_names")
            or features
        ),
        "feature_names_match": bool(
            result_payload.get("feature_names_match", True)
        ),
        "n_selected_features": len(features),
        "n_families": len(
            feature_set.get("families") or exp.get("family_reps") or []
        ),
        "family_reps": [
            {
                "family_id": r.get("family_id"),
                "family_label": r.get("family_label"),
                "representative": r.get("representative"),
                "changed": r.get("changed"),
            }
            for r in (
                feature_set.get("families")
                or [
                    {
                        "family_id": r.get("family_id"),
                        "family_label": r.get("family_label"),
                        "representative": r.get("representative"),
                    }
                    for r in (exp.get("family_reps") or [])
                ]
            )
        ],
        "parent_diff": parent_diff,
        "parent_experiment_id": exp.get("parent_experiment_id"),
        "model_name": model_name or None,
        "model_path": model_path or result_payload.get("model_path") or None,
        "model_file": model_file or None,
        "train_device": (
            result_payload.get("train_device")
            or exp.get("train_device")
        ),
        "shap_device": (
            result_payload.get("shap_device")
            or exp.get("shap_device")
        ),
        "device_label": (
            result_payload.get("device_label")
            or exp.get("device_label")
            or result_payload.get("train_device")
            or exp.get("train_device")
        ),
        "executed_device": result_payload.get("executed_device"),
        "gpu_name": result_payload.get("gpu_name"),
        "device_fallback_reason": result_payload.get("device_fallback_reason"),
        "holdout_score": exp.get("holdout_score"),
        "walk_forward_score": exp.get("walk_forward_score"),
        "holdout_rmse": result_payload.get("holdout_rmse"),
        "validation_label": exp.get("validation_label"),
        "target": result_payload.get("target"),
        "features_fingerprint": (
            feature_set.get("hash")
            or result_payload.get("features_fingerprint")
        ),
        "promoted_at": stamp,
        "card": {
            "name": bundle_name,
            "dataset": rid,
            "experiment": eid,
            "discovery_bundle": exp.get("discovery_bundle_id"),
            "n_features": len(features),
            "n_families": len(
                feature_set.get("families") or exp.get("family_reps") or []
            ),
            "feature_set_hash": str(feature_set.get("hash") or "")[:16],
            "model": model_file or model_name or None,
            "device": (
                result_payload.get("device_label")
                or exp.get("device_label")
                or result_payload.get("train_device")
                or exp.get("train_device")
            ),
            "holdout": exp.get("holdout_score"),
            "walk_forward": exp.get("walk_forward_score"),
        },
    }
    champ = publish_artifact(
        data_dir,
        rid,
        KIND_CHAMPION,
        payload,
        parent_ids=[result_id],
        label=bundle_name,
        reuse_identical=False,
    )
    # Attach formatted card for UI consumers
    bundle_view = dict(champ)
    bundle_view["payload"] = payload
    card_text = format_champion_bundle_card(bundle_view)
    return {
        "experiment_id": eid,
        "champion_artifact_id": champ["artifact_id"],
        "champion_bundle_id": champ["artifact_id"],
        "name": bundle_name,
        "fingerprint": champ.get("fingerprint"),
        "selected_features": features,
        "n_selected_features": len(features),
        "model_name": model_name or None,
        "model_path": model_path or None,
        "model_file": model_file or None,
        "card_text": card_text,
        "message": (
            f"Promoted {eid} as {bundle_name} · {champ['artifact_id']}"
        ),
        "experiment": load_experiment(data_dir, eid, verify_bundle=True),
        "bundle": bundle_view,
    }


def _fmt_metric(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return "—"


def _variant_change_cols(exp: dict[str, Any]) -> tuple[str, str, str]:
    changes = exp.get("variant_changes_list") or []
    if not changes:
        if str(exp.get("name") or "").startswith("Auto-baseline") or not exp.get(
            "parent_experiment_id"
        ):
            return ("— (baseline)", "—", "—")
        return ("—", "—", "—")
    # Show first change; note if multiple
    c0 = changes[0]
    fam = str(c0.get("family_label") or c0.get("family_id") or "—")
    if len(changes) > 1:
        fam = f"{fam} (+{len(changes) - 1})"
    return (
        fam,
        str(c0.get("old_representative") or "—"),
        str(c0.get("new_representative") or "—"),
    )


def compare_experiments(
    data_dir: str,
    run_id: str,
    *,
    focus_family_id: str | None = None,
    baseline_experiment_id: str | None = None,
) -> list[dict[str, Any]]:
    """Rows for Experiment Comparison table."""
    focus = str(focus_family_id or "").strip() or None
    rows_out: list[dict[str, Any]] = []
    experiments = list_experiments(data_dir, run_id)

    scored = [e for e in experiments if e.get("holdout_score") is not None]
    best_id = None
    if scored:
        best_id = max(
            scored,
            key=lambda e: (
                float(e.get("holdout_score") or -1e9),
                float(e.get("walk_forward_score") or -1e9),
            ),
        )["experiment_id"]

    baseline_id = str(baseline_experiment_id or "").strip() or None
    if not baseline_id:
        for e in experiments:
            if str(e.get("name") or "") == "Auto-baseline":
                baseline_id = str(e.get("experiment_id"))
                break
    baseline_hold = None
    if baseline_id:
        for e in experiments:
            if str(e.get("experiment_id")) == baseline_id:
                baseline_hold = e.get("holdout_score")
                break

    for e in experiments:
        reps = {str(r["family_id"]): r for r in (e.get("family_reps") or [])}
        if focus and focus in reps:
            focus_rep = str(reps[focus].get("representative") or "—")
            focus_label = str(reps[focus].get("family_label") or focus)
        else:
            focus_rep = "—"
            focus_label = ""
            if not focus and reps:
                first = next(iter(reps.values()))
                focus_rep = str(first.get("representative") or "—")
                focus_label = str(
                    first.get("family_label") or first.get("family_id")
                )

        label = str(e.get("validation_label") or VALIDATION_PENDING)
        if best_id and e.get("holdout_score") is not None:
            if e["experiment_id"] == best_id and label == VALIDATION_PENDING:
                label = VALIDATION_BEST
            elif (
                e["experiment_id"] != best_id
                and label == VALIDATION_PENDING
                and len(scored) > 1
            ):
                label = VALIDATION_WORSE

        changed_family, old_rep, new_rep = _variant_change_cols(e)
        hold = e.get("holdout_score")
        delta = None
        delta_txt = "—"
        if hold is not None and baseline_hold is not None:
            delta = float(hold) - float(baseline_hold)
            delta_txt = f"{delta:+.4f}"

        # Families changed vs Auto-baseline feature-set
        n_fam_changed = 0
        if baseline_id:
            base_exp = next(
                (
                    x
                    for x in experiments
                    if str(x.get("experiment_id")) == baseline_id
                ),
                None,
            )
            if base_exp:
                base_reps = {
                    str(r["family_id"]): str(r["representative"])
                    for r in (base_exp.get("family_reps") or [])
                }
                cur_reps = {
                    str(r["family_id"]): str(r["representative"])
                    for r in (e.get("family_reps") or [])
                }
                n_fam_changed = sum(
                    1
                    for fid, rep in cur_reps.items()
                    if base_reps.get(fid) != rep
                )
        elif e.get("variant_changes_list"):
            n_fam_changed = len(e.get("variant_changes_list") or [])

        summary_reps = ", ".join(
            f"{r.get('family_label') or r.get('family_id')}→"
            f"{r.get('representative')}"
            for r in (e.get("family_reps") or [])[:4]
        )
        rows_out.append(
            {
                "experiment_id": e.get("experiment_id"),
                "name": e.get("name") or e.get("experiment_id"),
                "status": e.get("status_display") or e.get("status"),
                "model_name": e.get("model_name") or "—",
                "focus_family_id": focus,
                "focus_family_label": focus_label,
                "focus_representative": focus_rep,
                "changed_family": changed_family,
                "old_rep": old_rep,
                "new_rep": new_rep,
                "families_changed": n_fam_changed,
                "reps_summary": summary_reps or "—",
                "holdout_score": e.get("holdout_score"),
                "holdout": _fmt_metric(e.get("holdout_score")),
                "walk_forward_score": e.get("walk_forward_score"),
                "walk_forward": _fmt_metric(e.get("walk_forward_score")),
                "delta_vs_baseline": delta,
                "delta_vs_baseline_txt": delta_txt,
                "validation_label": label,
                "notes": e.get("notes") or "",
                "family_reps": e.get("family_reps") or [],
                "variant_changes_list": e.get("variant_changes_list") or [],
                "parent_experiment_id": e.get("parent_experiment_id"),
                "is_champion": int(e.get("is_champion") or 0),
                "discovery_bundle_id": e.get("discovery_bundle_id"),
                "discovery_bundle_fingerprint": e.get(
                    "discovery_bundle_fingerprint"
                ),
                "result_artifact_id": e.get("result_artifact_id"),
                "validation_summary": e.get("validation_summary") or "",
                "train_device": e.get("train_device") or "",
                "shap_device": e.get("shap_device") or "",
                "device_label": e.get("device_label")
                or e.get("train_device")
                or "—",
            }
        )
    return rows_out


def experiment_candidate_sets_preview(
    data_dir: str,
    run_id: str,
    *,
    min_size: int = 2,
) -> list[dict[str, Any]]:
    """Preview working-table picks (UI only). Create uses discovery_bundle."""
    from .analysis_family_review import (
        FILTER_ALL,
        load_families_with_reviews,
    )

    out: list[dict[str, Any]] = []
    for fam in load_families_with_reviews(
        data_dir, run_id, min_size=min_size, status_filter=FILTER_ALL
    ):
        rep = (
            str(fam.get("experiment_representative") or "").strip()
            or str(fam.get("suggested_representative") or "").strip()
        )
        out.append(
            {
                "family_id": fam.get("family_id"),
                "family_label": fam.get("family_label"),
                "size": fam.get("size"),
                "experiment_representative": rep or None,
                "suggested_representative": fam.get("suggested_representative"),
                "confidence": fam.get("confidence"),
                "review_status": fam.get("review_status"),
            }
        )
    return out


def discovery_bundle_card(
    data_dir: str,
    run_id: str,
) -> dict[str, Any]:
    """Card fields for the Discovery Bundle that experiments bind to."""
    from .analysis_artifacts import KIND_DISCOVERY_BUNDLE, latest_artifact

    rid = str(run_id or "").strip()
    empty = {
        "present": False,
        "artifact_id": None,
        "dataset": None,
        "n_families": 0,
        "n_experiment_reps": 0,
        "created_at": None,
        "n_experiments": 0,
        "card_text": (
            "Discovery Bundle\n"
            "  (none yet — Freeze Discovery Bundle after Discovery Complete)"
        ),
    }
    if not rid:
        return empty
    art = latest_artifact(data_dir, rid, KIND_DISCOVERY_BUNDLE)
    if not art:
        return empty
    payload = dict(art.get("payload") or {})
    card = dict(payload.get("card") or {})
    bid = str(art.get("artifact_id") or "")
    experiments = list_experiments(data_dir, rid)
    n_bound = sum(
        1
        for e in experiments
        if str(e.get("discovery_bundle_id") or "") == bid
    )
    n_families = int(
        card.get("n_families")
        if card.get("n_families") is not None
        else len(payload.get("families") or payload.get("family_reps") or {})
    )
    n_reps = int(
        card.get("n_experiment_reps")
        if card.get("n_experiment_reps") is not None
        else len(payload.get("family_reps") or {})
    )
    created = str(art.get("created_at") or payload.get("frozen_at") or "—")
    dataset = str(card.get("dataset") or "—")
    lines = [
        "Discovery Bundle",
        f"ID             {bid}",
        f"Dataset        {dataset}",
        f"Families       {n_families}",
        f"Representatives {n_reps}",
        f"Created        {created}",
        f"Experiments    {n_bound}",
    ]
    return {
        "present": True,
        "artifact_id": bid,
        "dataset": dataset,
        "n_families": n_families,
        "n_experiment_reps": n_reps,
        "created_at": created,
        "n_experiments": n_bound,
        "fingerprint": art.get("fingerprint"),
        "card_text": "\n".join(lines),
    }


def platform_workflow_summary(
    data_dir: str,
    run_id: str,
) -> dict[str, Any]:
    """Where-am-I strip: Discovery → Bundle → Experiments → Models → Champion."""
    from .analysis_family_review import discovery_readiness

    rid = str(run_id or "").strip()
    ready = discovery_readiness(data_dir, rid) if rid else {
        "complete": False,
        "headline": "Discovery  —",
        "latest_discovery_bundle_id": None,
        "banner_text": "Discovery  —  Load a dataset and run Stage 1 modules.",
    }
    bundle = discovery_bundle_card(data_dir, rid) if rid else {
        "present": False,
        "artifact_id": None,
        "n_experiments": 0,
        "card_text": "",
    }
    experiments = list_experiments(data_dir, rid) if rid else []
    n_exp = len(experiments)
    n_models = sum(
        1
        for e in experiments
        if e.get("model_name") or e.get("result_artifact_id")
    )
    champ = next(
        (e for e in experiments if int(e.get("is_champion") or 0)),
        None,
    )
    champ_id = str(champ.get("experiment_id")) if champ else None
    bundle_id = bundle.get("artifact_id") or ready.get("latest_discovery_bundle_id")

    disc_line = (
        "✓ Discovery Complete"
        if ready.get("complete")
        else str(ready.get("headline") or "Discovery  Incomplete")
    )
    lines = [
        "Discovery",
        "────────────────────────",
        disc_line,
        "",
        "Bundle",
        str(bundle_id or "(not frozen)"),
        "↓",
        "Experiments",
        str(n_exp),
        "↓",
        "Models",
        str(n_models),
        "↓",
        "Champion",
        str(champ_id or "—"),
    ]
    # Keep Discovery detail under the strip for module/family status
    detail = str(ready.get("detail") or "").strip()
    banner = "\n".join(lines)
    if detail and not ready.get("complete"):
        banner = f"{banner}\n\n{detail}"
    elif ready.get("complete") and bundle.get("present"):
        banner = (
            f"{banner}\n\n"
            f"{n_exp} experiment(s) · {n_models} model(s)"
            + (f" · Champion {champ_id}" if champ_id else "")
        )

    return {
        "discovery_complete": bool(ready.get("complete")),
        "bundle_id": bundle_id,
        "n_experiments": n_exp,
        "n_models": n_models,
        "champion_id": champ_id,
        "bundle_card": bundle,
        "banner_text": banner,
        "discovery": ready,
    }


def strategy_comparison_table(
    data_dir: str,
    run_id: str,
) -> list[dict[str, Any]]:
    """Compare experiments by Feature Selection Strategy (evidence table).

    Columns: Strategy | Final Features | Holdout | Walk-forward | Validation
    """
    from .analysis_feature_selection import compare_strategy_rows

    rows_in: list[dict[str, Any]] = []
    for e in list_experiments(data_dir, run_id):
        if e.get("holdout_score") is None and not e.get("result_artifact_id"):
            continue
        sel = dict(e.get("feature_selection") or {})
        if not sel and e.get("result_artifact_id"):
            try:
                art = load_artifact(data_dir, str(e["result_artifact_id"]))
                sel = dict((art or {}).get("payload") or {}).get(
                    "feature_selection"
                ) or {}
            except Exception:
                sel = {}
        fs = dict(e.get("feature_set") or {})
        rows_in.append(
            {
                "experiment_id": e.get("experiment_id"),
                "feature_selection": sel,
                "n_features": fs.get("count")
                or len(fs.get("features") or e.get("family_reps") or []),
                "holdout_score": e.get("holdout_score"),
                "walk_forward_score": e.get("walk_forward_score"),
                "validation_label": e.get("validation_label"),
                "is_champion": e.get("is_champion"),
            }
        )
    return compare_strategy_rows(rows_in)


__all__ = [
    "STATUS_ARCHIVED",
    "STATUS_CHAMPION",
    "STATUS_COMPLETED",
    "STATUS_CREATED",
    "STATUS_DRAFT",
    "STATUS_MODEL_PRODUCED",
    "STATUS_READY",
    "STATUS_TRAINED",
    "STATUS_TRAINING",
    "STATUS_VALIDATED",
    "VALIDATION_BEST",
    "VALIDATION_GOOD",
    "VALIDATION_PENDING",
    "VALIDATION_UNSTABLE",
    "VALIDATION_WORSE",
    "clone_experiment_variant",
    "clone_experiment_with_rep",
    "compare_experiments",
    "strategy_comparison_table",
    "create_experiment",
    "delete_experiment",
    "discovery_bundle_card",
    "display_experiment_status",
    "ensure_experiments_schema",
    "experiment_candidate_sets_preview",
    "experiment_family_options",
    "format_champion_bundle_card",
    "format_experiment_details",
    "format_hypothesis_text",
    "build_feature_set",
    "build_parent_diff",
    "features_fingerprint",
    "list_experiments",
    "load_champion_bundle",
    "load_experiment",
    "normalize_experiment_status",
    "platform_workflow_summary",
    "promote_champion",
    "request_train_experiment",
    "update_experiment_metrics",
]
