"""Persist / load Experiment Planner artifacts (Recommendation Engine output).

Generated recommendations live in ``planner.json`` (regenerable).
User-managed status / notes live in ``experiment_state.json`` (preserved across Compute).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from chain_replay_ml.recommendation_engine.config import (
    DEFAULT_EXPERIMENT_STATUS,
    EXPERIMENT_STATE_SCHEMA_VERSION,
    EXPERIMENT_STATUSES,
    MODEL_EXPERIMENT_CATEGORIES,
)
from chain_replay_ml.recommendation_engine.rules import (
    evidence_score_from_unit,
    unit_from_evidence_score,
)

# UI / disk folder name — distinct from compute package ``recommendation_engine``.
ARTIFACT_DIRNAME = "experiment_planner"
STATE_FILENAME = "experiment_state.json"
# Legacy sidecar (migrated on load into experiment_state.json).
LEGACY_STATUS_FILENAME = "status.json"

ARTIFACT_FILES = (
    "planner.json",
    "summary.json",
    "run_meta.json",
)


def studio_artifacts_dir(package_dir: str) -> str:
    path = os.path.join(package_dir, ARTIFACT_DIRNAME)
    os.makedirs(path, exist_ok=True)
    return path


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_expected_benefit(raw: Any) -> dict[str, str]:
    """Legacy benefit object (schema ≤2). Prefer ``hypothesis`` in schema 3+."""
    if isinstance(raw, dict):
        return {
            "model_stability": str(raw.get("model_stability") or "unknown"),
            "prediction_accuracy": str(raw.get("prediction_accuracy") or "unknown"),
            "training_speed": str(raw.get("training_speed") or "unknown"),
            "summary": str(raw.get("summary") or ""),
        }
    if raw is None:
        return {
            "model_stability": "unknown",
            "prediction_accuracy": "unknown",
            "training_speed": "unknown",
            "summary": "",
        }
    text = str(raw)
    return {
        "model_stability": "unknown",
        "prediction_accuracy": "unknown",
        "training_speed": "unknown",
        "summary": text,
    }


def _normalize_affected_features(raw: Any) -> list[Any]:
    if not isinstance(raw, list):
        return []
    out: list[Any] = []
    for item in raw:
        if isinstance(item, dict):
            name = str(item.get("feature") or "").strip()
            if not name:
                continue
            obj = dict(item)
            obj["feature"] = name
            out.append(obj)
        else:
            name = str(item).strip()
            if name:
                out.append({"feature": name})
    return out


def _normalize_hypothesis(raw: dict[str, Any]) -> str:
    """Ensure hypothesis string; fall back to legacy expected_benefit summary."""
    hyp = str(raw.get("hypothesis") or "").strip()
    if hyp:
        return hyp
    benefit = raw.get("expected_benefit")
    if isinstance(benefit, dict):
        return str(benefit.get("summary") or "").strip()
    if benefit is not None:
        return str(benefit).strip()
    return ""


def _normalize_status(raw: Any) -> str:
    text = str(raw or "").strip()
    if text in EXPERIMENT_STATUSES:
        return text
    return DEFAULT_EXPERIMENT_STATUS


def _normalize_effort(raw: Any) -> str:
    text = str(raw or "").strip()
    if text in ("Easy", "Medium", "High"):
        return text
    return "Medium"


def _normalize_next_steps(raw: Any) -> list[str]:
    if not isinstance(raw, list):
        return []
    return [str(s).strip() for s in raw if str(s).strip()]


def _normalize_findings(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(dict(item))
    return out


def _normalize_recommendations(raw: Any, findings: list[dict[str, Any]]) -> list[str]:
    if isinstance(raw, list) and raw:
        return [str(s).strip() for s in raw if str(s).strip()]
    return [
        str(f.get("recommendation") or "").strip()
        for f in findings
        if str(f.get("recommendation") or "").strip()
    ]


def normalize_suggestion(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize a suggestion/experiment for UI.

    Schema 5: ``findings`` / ``recommendations``, ``created_from``,
    ``generated_at``, ``planner_version``, plus schema-4 fields.
    Accepts legacy ``confidence`` (0–1) and ``expected_benefit`` objects/strings.
    """
    s = dict(raw)
    es = s.get("evidence_score")
    conf = s.get("confidence")
    if es is not None:
        try:
            f_es = float(es)
        except (TypeError, ValueError):
            f_es = None
        if f_es is not None:
            # Prefer 0–100 integers; treat bare 0–1 floats as legacy unit.
            if isinstance(es, int) or f_es > 1.0:
                s["evidence_score"] = int(round(max(0.0, min(100.0, f_es))))
            elif conf is not None:
                unit = unit_from_evidence_score(conf)
                s["evidence_score"] = evidence_score_from_unit(unit or 0.0)
            else:
                s["evidence_score"] = evidence_score_from_unit(f_es)
    elif conf is not None:
        unit = unit_from_evidence_score(conf)
        s["evidence_score"] = evidence_score_from_unit(unit or 0.0)

    if s.get("confidence") is None and s.get("evidence_score") is not None:
        try:
            s["confidence"] = round(float(s["evidence_score"]) / 100.0, 3)
        except (TypeError, ValueError):
            pass

    bullets = s.get("reason_bullets")
    if not isinstance(bullets, list) or not bullets:
        reason = str(s.get("reason") or "").strip()
        if reason:
            if " · " in reason:
                bullets = [p.strip() for p in reason.split(" · ") if p.strip()]
            elif "; " in reason:
                bullets = [p.strip() for p in reason.split("; ") if p.strip()]
            else:
                bullets = [reason]
        else:
            bullets = []
    else:
        bullets = [str(b).strip() for b in bullets if str(b).strip()]
    s["reason_bullets"] = bullets
    if not s.get("reason") and bullets:
        s["reason"] = " · ".join(bullets)

    hyp = _normalize_hypothesis(s)
    s["hypothesis"] = hyp
    # Keep legacy expected_benefit for older loaders; prefer hypothesis in UI.
    benefit = _normalize_expected_benefit(s.get("expected_benefit"))
    if hyp and not str(benefit.get("summary") or "").strip():
        benefit["summary"] = hyp
    s["expected_benefit"] = benefit

    fam = s.get("family")
    if fam is not None and str(fam).strip():
        s["family"] = str(fam).strip()
    else:
        # Infer from evidence or id suffix R1_...__IV
        ev = s.get("evidence")
        if isinstance(ev, dict) and ev.get("family"):
            s["family"] = str(ev["family"]).strip()
        else:
            sid = str(s.get("id") or "")
            if "__" in sid:
                s["family"] = sid.rsplit("__", 1)[-1]
            else:
                s["family"] = None

    s["affected_features"] = _normalize_affected_features(s.get("affected_features"))

    # Evidence: ensure rule id/name keys when recoverable.
    ev = s.get("evidence")
    if isinstance(ev, dict):
        ev = dict(ev)
        if not ev.get("rule_name") and ev.get("rule"):
            ev["rule_name"] = ev["rule"]
        if not ev.get("rule_id"):
            sid = str(s.get("id") or "")
            ev["rule_id"] = sid.split("__", 1)[0] if sid else None
        if s.get("family") and not ev.get("family"):
            ev["family"] = s["family"]
        agg = ev.get("aggregate")
        if isinstance(agg, dict):
            agg = dict(agg)
            if "highest_risk_feature" not in agg:
                # Derive from top_contributors / affected_features when missing.
                feats = (
                    ev.get("top_contributors")
                    if isinstance(ev.get("top_contributors"), list)
                    else s.get("affected_features")
                )
                best_name = None
                best_risk = None
                if isinstance(feats, list):
                    for item in feats:
                        if not isinstance(item, dict):
                            continue
                        name = str(item.get("feature") or "").strip()
                        risk = item.get("risk_score")
                        if not name:
                            continue
                        try:
                            risk_f = float(risk) if risk is not None else None
                        except (TypeError, ValueError):
                            risk_f = None
                        if risk_f is None:
                            if best_name is None:
                                best_name = name
                            continue
                        if best_risk is None or risk_f > best_risk:
                            best_risk = risk_f
                            best_name = name
                agg["highest_risk_feature"] = best_name
                agg["highest_risk_score"] = best_risk
            ev["aggregate"] = agg
        if ev.get("feature_count") is None:
            matched = ev.get("matched_features")
            if isinstance(matched, list):
                ev["feature_count"] = len(matched)
            else:
                ev["feature_count"] = len(s.get("affected_features") or [])
        s["evidence"] = ev

    # Schema 4/5 presentation metadata (defaults for older artifacts).
    exp_id = str(s.get("experiment_id") or "").strip()
    if exp_id:
        s["experiment_id"] = exp_id
    else:
        s["experiment_id"] = None

    s["status"] = _normalize_status(s.get("status"))
    s["estimated_effort"] = _normalize_effort(s.get("estimated_effort"))
    s["expected_experiment"] = str(s.get("expected_experiment") or "").strip()
    s["suggested_next_steps"] = _normalize_next_steps(s.get("suggested_next_steps"))
    findings = _normalize_findings(s.get("findings"))
    s["findings"] = findings
    s["recommendations"] = _normalize_recommendations(s.get("recommendations"), findings)

    created = s.get("created_from")
    s["created_from"] = str(created).strip() if created is not None and str(created).strip() else None
    generated = s.get("generated_at")
    s["generated_at"] = (
        str(generated).strip() if generated is not None and str(generated).strip() else None
    )
    pver = s.get("planner_version")
    s["planner_version"] = (
        str(pver).strip() if pver is not None and str(pver).strip() else None
    )

    scope = str(s.get("experiment_scope") or "").strip().lower()
    if scope not in ("model", "feature"):
        cat = str(s.get("category") or "").strip()
        scope = "model" if cat in MODEL_EXPERIMENT_CATEGORIES else "feature"
    s["experiment_scope"] = scope

    return s


# ---------------------------------------------------------------------------
# experiment_state.json — user-managed status / notes
# ---------------------------------------------------------------------------


def experiment_state_path(package_dir: str) -> str:
    return os.path.join(package_dir, ARTIFACT_DIRNAME, STATE_FILENAME)


def status_sidecar_path(package_dir: str) -> str:
    """Legacy path; prefer ``experiment_state_path``."""
    return os.path.join(package_dir, ARTIFACT_DIRNAME, LEGACY_STATUS_FILENAME)


def _empty_state() -> dict[str, Any]:
    return {
        "schema_version": EXPERIMENT_STATE_SCHEMA_VERSION,
        "experiments": {},
    }


def _normalize_note_entry(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    text = str(raw.get("text") or raw.get("note") or raw.get("reason") or "").strip()
    action = str(raw.get("action") or "add_notes").strip() or "add_notes"
    at = str(raw.get("at") or raw.get("updated_at") or "").strip() or None
    status = raw.get("status")
    entry: dict[str, Any] = {"action": action, "text": text}
    if at:
        entry["at"] = at
    if status is not None and str(status).strip():
        entry["status"] = _normalize_status(status)
    return entry


def _normalize_state_entry(experiment_id: str, raw: Any) -> dict[str, Any]:
    eid = str(experiment_id or "").strip()
    if isinstance(raw, str):
        # Legacy status.json value → entry
        return {
            "experiment_id": eid,
            "internal_id": None,
            "status": _normalize_status(raw),
            "notes": [],
            "updated_at": None,
        }
    if not isinstance(raw, dict):
        return {
            "experiment_id": eid,
            "internal_id": None,
            "status": DEFAULT_EXPERIMENT_STATUS,
            "notes": [],
            "updated_at": None,
        }
    notes_raw = raw.get("notes") if isinstance(raw.get("notes"), list) else []
    notes: list[dict[str, Any]] = []
    for n in notes_raw:
        entry = _normalize_note_entry(n)
        if entry is not None:
            notes.append(entry)
    # Promote legacy completed_note / rejected_reason into notes if notes empty.
    for key, action in (
        ("completed_note", "mark_complete"),
        ("rejected_reason", "reject"),
        ("note", "add_notes"),
    ):
        text = str(raw.get(key) or "").strip()
        if text and not any(n.get("text") == text for n in notes):
            notes.append(
                {
                    "action": action,
                    "text": text,
                    "at": str(raw.get("updated_at") or "") or None,
                    "status": _normalize_status(raw.get("status")),
                }
            )
    internal = raw.get("internal_id") or raw.get("id")
    return {
        "experiment_id": eid,
        "internal_id": str(internal).strip() if internal else None,
        "status": _normalize_status(raw.get("status")),
        "notes": notes,
        "updated_at": str(raw.get("updated_at") or "").strip() or None,
    }


def _migrate_legacy_status_file(package_dir: str) -> dict[str, Any]:
    """Load legacy ``status.json`` map into experiment_state shape."""
    path = status_sidecar_path(package_dir)
    if not os.path.isfile(path):
        return _empty_state()
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return _empty_state()
    if not isinstance(doc, dict):
        return _empty_state()
    statuses = doc.get("statuses") if isinstance(doc.get("statuses"), dict) else doc
    if not isinstance(statuses, dict):
        return _empty_state()
    experiments: dict[str, Any] = {}
    for key, val in statuses.items():
        k = str(key).strip()
        if not k:
            continue
        experiments[k] = _normalize_state_entry(k, val)
    return {
        "schema_version": EXPERIMENT_STATE_SCHEMA_VERSION,
        "experiments": experiments,
    }


def load_experiment_state(package_dir: str) -> dict[str, Any]:
    """Load ``experiment_state.json`` (migrate from ``status.json`` if needed)."""
    path = experiment_state_path(package_dir)
    if os.path.isfile(path):
        try:
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        except (OSError, json.JSONDecodeError):
            doc = None
        if isinstance(doc, dict):
            experiments_raw = doc.get("experiments")
            if not isinstance(experiments_raw, dict):
                # Accept flat map of id → entry/status
                experiments_raw = {
                    k: v
                    for k, v in doc.items()
                    if k not in ("schema_version", "statuses")
                }
            experiments: dict[str, Any] = {}
            for key, val in experiments_raw.items():
                k = str(key).strip()
                if k:
                    experiments[k] = _normalize_state_entry(k, val)
            return {
                "schema_version": int(
                    doc.get("schema_version") or EXPERIMENT_STATE_SCHEMA_VERSION
                ),
                "experiments": experiments,
            }
    # Migrate legacy status.json
    migrated = _migrate_legacy_status_file(package_dir)
    if migrated.get("experiments"):
        try:
            save_experiment_state(package_dir, migrated)
        except OSError:
            pass
    return migrated


def save_experiment_state(package_dir: str, state: dict[str, Any]) -> str:
    """Persist user experiment state (does not touch planner.json)."""
    out_dir = studio_artifacts_dir(package_dir)
    path = os.path.join(out_dir, STATE_FILENAME)
    experiments_raw = state.get("experiments") if isinstance(state, dict) else {}
    if not isinstance(experiments_raw, dict):
        experiments_raw = {}
    clean: dict[str, Any] = {}
    for key, val in experiments_raw.items():
        k = str(key).strip()
        if not k:
            continue
        clean[k] = _normalize_state_entry(k, val)
    payload = {
        "schema_version": EXPERIMENT_STATE_SCHEMA_VERSION,
        "experiments": clean,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return path


def apply_experiment_state(
    suggestions: list[dict[str, Any]],
    state: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Merge user state onto suggestions by ``experiment_id`` (then internal id)."""
    experiments = (
        state.get("experiments") if isinstance(state, dict) else None
    ) or {}
    if not isinstance(experiments, dict) or not experiments:
        return suggestions

    by_internal: dict[str, dict[str, Any]] = {}
    for entry in experiments.values():
        if not isinstance(entry, dict):
            continue
        iid = str(entry.get("internal_id") or "").strip()
        if iid and iid not in by_internal:
            by_internal[iid] = entry

    out: list[dict[str, Any]] = []
    for s in suggestions:
        row = dict(s)
        eid = str(row.get("experiment_id") or "").strip()
        iid = str(row.get("id") or "").strip()
        entry = None
        if eid and eid in experiments:
            entry = experiments[eid]
        elif iid and iid in by_internal:
            entry = by_internal[iid]
        if isinstance(entry, dict):
            row["status"] = _normalize_status(entry.get("status"))
            row["state_notes"] = list(entry.get("notes") or [])
            row["state_updated_at"] = entry.get("updated_at")
        out.append(row)
    return out


def sync_experiment_state_on_recompute(
    package_dir: str,
    suggestions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Preserve ``experiment_state.json`` across planner recompute.

    - Match by ``experiment_id``; rematch by ``internal_id`` when EXP ids shift.
    - Mark prior entries whose id no longer appears as ``Superseded``.
    - Does **not** overwrite planner.json with user status.
    """
    state = load_experiment_state(package_dir)
    old_experiments: dict[str, Any] = dict(state.get("experiments") or {})

    active_ids = {
        str(s.get("experiment_id") or "").strip()
        for s in suggestions
        if str(s.get("experiment_id") or "").strip()
    }
    # Map internal_id → new experiment_id
    internal_to_new: dict[str, str] = {}
    for s in suggestions:
        eid = str(s.get("experiment_id") or "").strip()
        iid = str(s.get("id") or "").strip()
        if eid and iid:
            internal_to_new[iid] = eid

    new_experiments: dict[str, Any] = {}
    remapped_old_ids: set[str] = set()

    # Rematch old entries onto new EXP ids via internal_id when possible.
    for old_eid, entry in old_experiments.items():
        if not isinstance(entry, dict):
            continue
        status = _normalize_status(entry.get("status"))
        if status == "Superseded":
            # Keep historical superseded rows
            new_experiments[old_eid] = _normalize_state_entry(old_eid, entry)
            continue
        iid = str(entry.get("internal_id") or "").strip()
        if old_eid in active_ids:
            # Same EXP id still present — refresh internal_id from suggestion
            refreshed = dict(entry)
            for s in suggestions:
                if str(s.get("experiment_id") or "").strip() == old_eid:
                    refreshed["internal_id"] = str(s.get("id") or "") or refreshed.get(
                        "internal_id"
                    )
                    break
            new_experiments[old_eid] = _normalize_state_entry(old_eid, refreshed)
            remapped_old_ids.add(old_eid)
            continue
        if iid and iid in internal_to_new:
            new_eid = internal_to_new[iid]
            if new_eid not in new_experiments:
                moved = dict(entry)
                moved["experiment_id"] = new_eid
                moved["internal_id"] = iid
                new_experiments[new_eid] = _normalize_state_entry(new_eid, moved)
            remapped_old_ids.add(old_eid)
            continue
        # Unmatched → Superseded (history)
        superseded = dict(entry)
        superseded["status"] = "Superseded"
        superseded["updated_at"] = _utc_now()
        notes = list(superseded.get("notes") or [])
        notes.append(
            {
                "at": _utc_now(),
                "action": "superseded",
                "text": "Experiment id no longer present after planner recompute.",
                "status": "Superseded",
            }
        )
        superseded["notes"] = notes
        new_experiments[old_eid] = _normalize_state_entry(old_eid, superseded)

    # Ensure every active suggestion has a state stub (default Not Started).
    for s in suggestions:
        eid = str(s.get("experiment_id") or "").strip()
        if not eid:
            continue
        if eid in new_experiments:
            continue
        new_experiments[eid] = _normalize_state_entry(
            eid,
            {
                "experiment_id": eid,
                "internal_id": str(s.get("id") or "") or None,
                "status": DEFAULT_EXPERIMENT_STATUS,
                "notes": [],
                "updated_at": None,
            },
        )

    state = {
        "schema_version": EXPERIMENT_STATE_SCHEMA_VERSION,
        "experiments": new_experiments,
    }
    save_experiment_state(package_dir, state)
    return state


def record_experiment_action(
    package_dir: str,
    *,
    experiment_id: str,
    action: str,
    status: str | None = None,
    note: str | None = None,
    internal_id: str | None = None,
) -> dict[str, Any]:
    """Apply a manual status action and append to notes history."""
    eid = str(experiment_id or "").strip()
    if not eid:
        raise ValueError("experiment_id required")
    state = load_experiment_state(package_dir)
    experiments = dict(state.get("experiments") or {})
    entry = dict(experiments.get(eid) or {})
    entry["experiment_id"] = eid
    if internal_id:
        entry["internal_id"] = str(internal_id).strip()
    elif not entry.get("internal_id"):
        entry["internal_id"] = None

    action_key = str(action or "").strip()
    note_text = str(note or "").strip()
    now = _utc_now()

    if action_key == "mark_in_progress":
        new_status = "In Progress"
    elif action_key == "mark_complete":
        new_status = "Completed"
    elif action_key == "reject":
        if not note_text:
            raise ValueError("Reject requires a reason/note")
        new_status = "Rejected"
    elif action_key == "add_notes":
        if not note_text:
            raise ValueError("Add Notes requires text")
        new_status = _normalize_status(entry.get("status") or status)
    else:
        raise ValueError(f"Unknown action: {action_key}")

    if status is not None and action_key == "add_notes":
        # add_notes does not change status unless explicitly passed (unused)
        pass
    else:
        entry["status"] = new_status

    notes = list(entry.get("notes") or [])
    notes.append(
        {
            "at": now,
            "action": action_key,
            "text": note_text,
            "status": entry["status"],
        }
    )
    entry["notes"] = notes
    entry["updated_at"] = now
    experiments[eid] = _normalize_state_entry(eid, entry)
    state = {
        "schema_version": EXPERIMENT_STATE_SCHEMA_VERSION,
        "experiments": experiments,
    }
    save_experiment_state(package_dir, state)
    return state


# ---------------------------------------------------------------------------
# Backward-compatible status helpers (thin wrappers over experiment_state)
# ---------------------------------------------------------------------------


def load_experiment_statuses(package_dir: str) -> dict[str, str]:
    """Load status map keyed by experiment_id (from experiment_state)."""
    state = load_experiment_state(package_dir)
    experiments = state.get("experiments") or {}
    out: dict[str, str] = {}
    if not isinstance(experiments, dict):
        return out
    for key, entry in experiments.items():
        k = str(key).strip()
        if not k:
            continue
        if isinstance(entry, dict):
            out[k] = _normalize_status(entry.get("status"))
        else:
            out[k] = _normalize_status(entry)
    return out


def save_experiment_statuses(package_dir: str, statuses: dict[str, str]) -> str:
    """Persist a flat status map into ``experiment_state.json`` (merge)."""
    state = load_experiment_state(package_dir)
    experiments = dict(state.get("experiments") or {})
    now = _utc_now()
    for key, val in (statuses or {}).items():
        k = str(key).strip()
        if not k:
            continue
        prev = dict(experiments.get(k) or {})
        prev["status"] = _normalize_status(val)
        prev["updated_at"] = now
        if "notes" not in prev:
            prev["notes"] = []
        experiments[k] = _normalize_state_entry(k, prev)
    state = {
        "schema_version": EXPERIMENT_STATE_SCHEMA_VERSION,
        "experiments": experiments,
    }
    return save_experiment_state(package_dir, state)


def apply_status_overrides(
    suggestions: list[dict[str, Any]],
    statuses: dict[str, str],
) -> list[dict[str, Any]]:
    """Merge flat status map onto suggestions (compat). Prefer apply_experiment_state."""
    if not statuses:
        return suggestions
    fake_state = {
        "experiments": {
            k: {"status": v, "notes": []} for k, v in statuses.items()
        }
    }
    return apply_experiment_state(suggestions, fake_state)


def write_studio_artifacts(
    package_dir: str,
    *,
    suggestions: list[dict[str, Any]],
    summary: dict[str, Any],
    run_meta: dict[str, Any],
) -> str:
    out = studio_artifacts_dir(package_dir)
    # Strip user status from generated planner (always default in artifact).
    clean_suggestions: list[dict[str, Any]] = []
    for s in suggestions:
        row = dict(s)
        row["status"] = DEFAULT_EXPERIMENT_STATUS
        row.pop("state_notes", None)
        row.pop("state_updated_at", None)
        clean_suggestions.append(row)
    payloads = {
        "planner.json": {"suggestions": clean_suggestions},
        "summary.json": summary,
        "run_meta.json": run_meta,
    }
    for name, doc in payloads.items():
        with open(os.path.join(out, name), "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, default=str)
    # Preserve / sync user state (mark Superseded for missing ids).
    sync_experiment_state_on_recompute(package_dir, clean_suggestions)
    return out


def load_studio_artifacts(package_dir: str) -> dict[str, Any] | None:
    out = os.path.join(package_dir, ARTIFACT_DIRNAME)
    planner_path = os.path.join(out, "planner.json")
    if not os.path.isfile(planner_path):
        return None
    with open(planner_path, encoding="utf-8") as fh:
        planner = json.load(fh)
    if not isinstance(planner, dict):
        return None
    suggestions = planner.get("suggestions")
    if not isinstance(suggestions, list):
        return None
    suggestions = [
        normalize_suggestion(s) for s in suggestions if isinstance(s, dict)
    ]
    # Assign display IDs for older schema-3 artifacts missing experiment_id.
    missing_ids = any(not s.get("experiment_id") for s in suggestions)
    if missing_ids:
        for i, s in enumerate(suggestions, start=1):
            if not s.get("experiment_id"):
                s["experiment_id"] = f"EXP-{i:03d}"

    state = load_experiment_state(package_dir)
    suggestions = apply_experiment_state(suggestions, state)
    statuses = {
        eid: str(entry.get("status") or DEFAULT_EXPERIMENT_STATUS)
        for eid, entry in (state.get("experiments") or {}).items()
        if isinstance(entry, dict)
    }

    summary: dict[str, Any] = {}
    summary_path = os.path.join(out, "summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as fh:
            loaded = json.load(fh)
            if isinstance(loaded, dict):
                summary = loaded
    # Alias bridge for older summary.json
    if (
        "highest_evidence_suggestion" not in summary
        and isinstance(summary.get("highest_confidence_suggestion"), dict)
    ):
        top = dict(summary["highest_confidence_suggestion"])
        if top.get("evidence_score") is None and top.get("confidence") is not None:
            unit = unit_from_evidence_score(top.get("confidence"))
            if unit is not None:
                top["evidence_score"] = evidence_score_from_unit(unit)
        summary["highest_evidence_suggestion"] = top

    meta: dict[str, Any] = {}
    meta_path = os.path.join(out, "run_meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            loaded = json.load(fh)
            if isinstance(loaded, dict):
                meta = loaded

    return {
        "artifacts_dir": out,
        "suggestions": suggestions,
        "summary": summary,
        "meta": meta,
        "statuses": statuses,
        "experiment_state": state,
    }
