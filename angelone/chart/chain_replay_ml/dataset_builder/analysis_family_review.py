"""Feature Family Review — choose Experiment Representatives after Discovery.

Design
------
HCA builds families and candidate lists only — never selects a representative.

Discovery Rating ranks members, suggests a candidate, computes confidence
from the score gap, and classifies:

  Suggested Default    — clear Discovery leader (default for next experiment)
  Review Recommended   — moderate gap
  Decision Required    — near-tie / low confidence

A pick here is an *Experiment Representative* — which candidate to test first —
not a permanent / universal winner. Final family winners come from training +
validation across experiments (Experiment Manager / Comparison).

Family Review is an *exception queue*. Default filter = Needs Review
(Review Recommended + Decision Required). Researchers may still override any
Suggested Default when starting an experiment.
"""
from __future__ import annotations

import json
from typing import Any, Sequence

from .analysis_hca import ensure_hca_schema, load_families
from .analysis_lab_store import _AnalysisDb, _now_iso

# System classification (set by Discovery Rating suggestions)
STATUS_SUGGESTED_DEFAULT = "Suggested Default"
STATUS_REVIEW_RECOMMENDED = "Review Recommended"
STATUS_DECISION_REQUIRED = "Decision Required"

# Researcher decisions (manual save) — experiment-scoped, not permanent
STATUS_FOR_EXPERIMENT = "For Experiment"
STATUS_DEFERRED = "Deferred"
STATUS_NEEDS_MORE_DATA = "Needs more data"

# Backward-compatible aliases (older UI / DB rows)
STATUS_AUTO_ACCEPTED = STATUS_SUGGESTED_DEFAULT
STATUS_ACCEPTED = STATUS_FOR_EXPERIMENT

SYSTEM_STATUSES = (
    STATUS_SUGGESTED_DEFAULT,
    STATUS_REVIEW_RECOMMENDED,
    STATUS_DECISION_REQUIRED,
)

MANUAL_STATUSES = (
    STATUS_FOR_EXPERIMENT,
    STATUS_DEFERRED,
    STATUS_NEEDS_MORE_DATA,
)

REVIEW_STATUSES = SYSTEM_STATUSES + MANUAL_STATUSES

# UI filter buckets
FILTER_NEEDS_REVIEW = "Needs Review"
FILTER_SUGGESTED_DEFAULT = "Suggested Default"
FILTER_AUTO_ACCEPTED = FILTER_SUGGESTED_DEFAULT  # alias
FILTER_ALL = "All Families"
FILTER_CHOICES = (FILTER_NEEDS_REVIEW, FILTER_SUGGESTED_DEFAULT, FILTER_ALL)

# Legacy status strings still present in older analysis.db rows
_LEGACY_STATUS_MAP = {
    "Auto Accepted": STATUS_SUGGESTED_DEFAULT,
    "Accepted": STATUS_FOR_EXPERIMENT,
}

NEEDS_REVIEW_STATUSES = frozenset(
    {STATUS_REVIEW_RECOMMENDED, STATUS_DECISION_REQUIRED}
)

REASON_INTERPRETABILITY = "Interpretability"
REASON_DATA_AVAILABILITY = "Data availability"
REASON_STABILITY = "Stability"
REASON_PERFORMANCE = "Performance"
REASON_OTHER = "Other"
REASON_AUTO = "Auto"

REVIEW_REASON_CHOICES = (
    REASON_INTERPRETABILITY,
    REASON_DATA_AVAILABILITY,
    REASON_STABILITY,
    REASON_PERFORMANCE,
    REASON_OTHER,
)

# Score-gap confidence thresholds (Discovery Rating 0–100 scale)
GAP_HIGH = 15.0
GAP_MEDIUM = 5.0


def normalize_review_status(status: str | None) -> str:
    st = str(status or "").strip()
    return _LEGACY_STATUS_MAP.get(st, st)


def ensure_family_review_schema(conn: Any) -> None:
    ensure_hca_schema(conn)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS family_review (
            run_id TEXT NOT NULL,
            family_id TEXT NOT NULL,
            selected_representative TEXT,
            reason_code TEXT,
            reason_text TEXT,
            status TEXT,
            updated_at TEXT,
            PRIMARY KEY (run_id, family_id)
        )
        """
    )
    cols = {
        str(r[1])
        for r in conn.execute("PRAGMA table_info(family_review)").fetchall()
    }
    for col, typ in (
        ("suggested_representative", "TEXT"),
        ("suggested_score", "REAL"),
        ("second_score", "REAL"),
        ("score_gap", "REAL"),
        ("confidence", "TEXT"),
        ("decision_source", "TEXT"),  # auto | manual
        ("overridden", "INTEGER"),
    ):
        if col not in cols:
            conn.execute(f"ALTER TABLE family_review ADD COLUMN {col} {typ}")


def _f(v: Any) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    return x


def confidence_from_gap(gap: float | None) -> str:
    if gap is None:
        return "Low"
    g = float(gap)
    if g >= GAP_HIGH:
        return "High"
    if g >= GAP_MEDIUM:
        return "Medium"
    return "Low"


def status_from_confidence(confidence: str) -> str:
    c = str(confidence or "").strip()
    if c == "High":
        return STATUS_SUGGESTED_DEFAULT
    if c == "Medium":
        return STATUS_REVIEW_RECOMMENDED
    return STATUS_DECISION_REQUIRED


def rank_family_members_by_score(
    members: Sequence[str],
    scores: dict[str, float],
) -> list[tuple[str, float]]:
    ranked: list[tuple[str, float]] = []
    for m in members:
        name = str(m)
        sc = scores.get(name)
        if sc is None:
            continue
        ranked.append((name, float(sc)))
    ranked.sort(key=lambda x: (-x[1], x[0]))
    # Members without scores go last with 0.0 for visibility
    missing = [str(m) for m in members if str(m) not in scores]
    for name in sorted(missing):
        ranked.append((name, 0.0))
    return ranked


def suggest_family_representative(
    members: Sequence[str],
    scores: dict[str, float],
) -> dict[str, Any]:
    """Suggest default experiment candidate + confidence from Discovery scores."""
    ranked = rank_family_members_by_score(members, scores)
    if not ranked:
        return {
            "suggested_representative": None,
            "suggested_score": None,
            "second_score": None,
            "score_gap": None,
            "confidence": "Low",
            "status": STATUS_DECISION_REQUIRED,
            "ranked": [],
        }
    top_name, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else None
    if len(ranked) == 1:
        gap = 100.0  # singleton — clear
    elif second_score is None:
        gap = float(top_score)
    else:
        gap = float(top_score) - float(second_score)
    conf = confidence_from_gap(gap)
    status = status_from_confidence(conf)
    return {
        "suggested_representative": top_name,
        "suggested_score": float(top_score),
        "second_score": float(second_score) if second_score is not None else None,
        "score_gap": float(gap),
        "confidence": conf,
        "status": status,
        "ranked": ranked,
    }


def _load_discovery_scores(data_dir: str, run_id: str) -> dict[str, float]:
    with _AnalysisDb(data_dir) as conn:
        rows = conn.execute(
            """
            SELECT feature_name, feature_score, rating_score
            FROM feature_profiles
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()
    out: dict[str, float] = {}
    for r in rows:
        name = str(r["feature_name"] or "")
        sc = _f(r["feature_score"])
        if sc is None:
            sc = _f(r["rating_score"])
        if name and sc is not None:
            out[name] = float(sc)
    return out


def apply_discovery_suggestions(
    data_dir: str,
    run_id: str,
    *,
    min_size: int = 2,
    overwrite_manual: bool = False,
) -> dict[str, Any]:
    """After Discovery Rating: suggest experiment defaults; flag near-ties.

    Preserves researcher manual experiment picks unless ``overwrite_manual``.
    Never claims a permanent family winner.
    """
    rid = str(run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required")

    families = load_families(data_dir, rid, min_size=int(min_size))
    if not families:
        return {
            "n_families": 0,
            "n_auto_accepted": 0,
            "n_needs_review": 0,
            "message": "No multi-member families — run HCA first.",
        }

    scores = _load_discovery_scores(data_dir, rid)
    stamp = _now_iso()
    n_auto = 0
    n_review = 0
    n_skipped_manual = 0

    with _AnalysisDb(data_dir) as conn:
        ensure_family_review_schema(conn)
        existing = {
            str(r["family_id"]): dict(r)
            for r in conn.execute(
                "SELECT * FROM family_review WHERE run_id = ?",
                (rid,),
            ).fetchall()
        }

        for fam in families:
            fid = str(fam.get("family_id") or "")
            members = list(fam.get("members") or [])
            if not fid or not members:
                continue

            prev = existing.get(fid) or {}
            prev_source = str(prev.get("decision_source") or "")
            prev_status = normalize_review_status(prev.get("status"))
            if (
                not overwrite_manual
                and prev_source == "manual"
                and prev_status in MANUAL_STATUSES
            ):
                n_skipped_manual += 1
                continue

            sug = suggest_family_representative(members, scores)
            suggested = sug["suggested_representative"]
            status = str(sug["status"])
            conf = str(sug["confidence"])
            if status == STATUS_SUGGESTED_DEFAULT:
                n_auto += 1
                selected = suggested
                reason_code = REASON_AUTO
                reason_text = (
                    f"Suggested Default for next experiment · "
                    f"Discovery confidence {conf} · "
                    f"score gap {sug['score_gap']:.1f} · "
                    f"not a permanent winner — compare after training"
                    if sug.get("score_gap") is not None
                    else (
                        f"Suggested Default for next experiment · "
                        f"Discovery confidence {conf} · "
                        f"not a permanent winner — compare after training"
                    )
                )
                source = "auto"
                overridden = 0
            else:
                n_review += 1
                # Do not commit experiment pick for ambiguous families
                selected = None
                reason_code = None
                reason_text = (
                    f"Suggested {suggested} · confidence {conf} · "
                    f"gap {sug['score_gap']:.1f} · "
                    f"near-tie — pick Experiment Representative to test first"
                    if suggested and sug.get("score_gap") is not None
                    else (
                        f"Suggested {suggested} · pick Experiment Representative"
                        if suggested
                        else "No scores"
                    )
                )
                source = "auto"
                overridden = 0

            conn.execute(
                """
                INSERT INTO family_review (
                    run_id, family_id, selected_representative,
                    reason_code, reason_text, status, updated_at,
                    suggested_representative, suggested_score, second_score,
                    score_gap, confidence, decision_source, overridden
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id, family_id) DO UPDATE SET
                    selected_representative = excluded.selected_representative,
                    reason_code = excluded.reason_code,
                    reason_text = excluded.reason_text,
                    status = excluded.status,
                    updated_at = excluded.updated_at,
                    suggested_representative = excluded.suggested_representative,
                    suggested_score = excluded.suggested_score,
                    second_score = excluded.second_score,
                    score_gap = excluded.score_gap,
                    confidence = excluded.confidence,
                    decision_source = excluded.decision_source,
                    overridden = excluded.overridden
                """,
                (
                    rid,
                    fid,
                    selected,
                    reason_code,
                    reason_text,
                    status,
                    stamp,
                    suggested,
                    sug.get("suggested_score"),
                    sug.get("second_score"),
                    sug.get("score_gap"),
                    conf,
                    source,
                    overridden,
                ),
            )

    msg = (
        f"Family suggestions · {len(families)} families · "
        f"Suggested Default {n_auto} · Needs Review {n_review}"
        + (f" · kept {n_skipped_manual} manual" if n_skipped_manual else "")
    )
    return {
        "n_families": len(families),
        "n_auto_accepted": n_auto,
        "n_needs_review": n_review,
        "n_skipped_manual": n_skipped_manual,
        "message": msg,
    }


def upsert_family_review(
    data_dir: str,
    run_id: str,
    family_id: str,
    *,
    selected_representative: str | None = None,
    experiment_representative: str | None = None,
    reason_code: str | None = None,
    reason_text: str | None = None,
    status: str | None = None,
) -> dict[str, Any]:
    """Set Experiment Representative for one family (manual, not permanent)."""
    rid = str(run_id or "").strip()
    fid = str(family_id or "").strip()
    if not rid or not fid:
        raise ValueError("run_id and family_id are required")

    rep = (
        str(experiment_representative or selected_representative or "").strip()
        or None
    )
    code = str(reason_code or "").strip() or None
    text = str(reason_text or "").strip() or None
    st = normalize_review_status(
        str(status or STATUS_FOR_EXPERIMENT).strip() or STATUS_FOR_EXPERIMENT
    )
    if st not in MANUAL_STATUSES:
        # Allow confirming suggested default as For Experiment
        if st in SYSTEM_STATUSES and rep:
            st = STATUS_FOR_EXPERIMENT
        else:
            raise ValueError(
                f"Manual status must be one of {MANUAL_STATUSES}, got {st!r}"
            )
    if code and code not in REVIEW_REASON_CHOICES and code != REASON_AUTO:
        if text:
            code = REASON_OTHER
        else:
            raise ValueError(
                f"reason_code must be one of {REVIEW_REASON_CHOICES}"
            )

    stamp = _now_iso()
    with _AnalysisDb(data_dir) as conn:
        ensure_family_review_schema(conn)
        if rep:
            row = conn.execute(
                """
                SELECT 1 FROM feature_family_members
                WHERE run_id = ? AND family_id = ? AND feature = ?
                """,
                (rid, fid, rep),
            ).fetchone()
            if not row:
                raise ValueError(f"{rep!r} is not a member of family {fid}")

        prev = conn.execute(
            """
            SELECT suggested_representative FROM family_review
            WHERE run_id = ? AND family_id = ?
            """,
            (rid, fid),
        ).fetchone()
        suggested = (
            str(prev["suggested_representative"] or "")
            if prev
            else ""
        )
        overridden = 1 if (rep and suggested and rep != suggested) else 0

        conn.execute(
            """
            INSERT INTO family_review (
                run_id, family_id, selected_representative,
                reason_code, reason_text, status, updated_at,
                decision_source, overridden
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'manual', ?)
            ON CONFLICT(run_id, family_id) DO UPDATE SET
                selected_representative = excluded.selected_representative,
                reason_code = excluded.reason_code,
                reason_text = excluded.reason_text,
                status = excluded.status,
                updated_at = excluded.updated_at,
                decision_source = 'manual',
                overridden = excluded.overridden
            """,
            (rid, fid, rep, code, text, st, stamp, overridden),
        )
        saved = conn.execute(
            """
            SELECT * FROM family_review
            WHERE run_id = ? AND family_id = ?
            """,
            (rid, fid),
        ).fetchone()
        return _enrich_review_row(dict(saved)) if saved else {}


def _enrich_review_row(row: dict[str, Any]) -> dict[str, Any]:
    item = dict(row)
    item["status"] = normalize_review_status(item.get("status"))
    rep = item.get("selected_representative")
    item["experiment_representative"] = rep
    return item


def load_family_reviews(
    data_dir: str, run_id: str
) -> dict[str, dict[str, Any]]:
    with _AnalysisDb(data_dir) as conn:
        ensure_family_review_schema(conn)
        rows = conn.execute(
            "SELECT * FROM family_review WHERE run_id = ?",
            (run_id,),
        ).fetchall()
    return {str(r["family_id"]): _enrich_review_row(dict(r)) for r in rows}


def load_family_review(
    data_dir: str, run_id: str, family_id: str
) -> dict[str, Any] | None:
    with _AnalysisDb(data_dir) as conn:
        ensure_family_review_schema(conn)
        row = conn.execute(
            """
            SELECT * FROM family_review
            WHERE run_id = ? AND family_id = ?
            """,
            (run_id, family_id),
        ).fetchone()
        return _enrich_review_row(dict(row)) if row else None


def _matches_filter(status: str | None, filt: str) -> bool:
    st = normalize_review_status(status)
    f = str(filt or FILTER_NEEDS_REVIEW).strip()
    if f == FILTER_ALL or f.upper() == "ALL":
        return True
    if f in (FILTER_SUGGESTED_DEFAULT, "Auto Accepted"):
        return st == STATUS_SUGGESTED_DEFAULT
    # Needs Review (default)
    if f == FILTER_NEEDS_REVIEW or f.upper() == "NEEDS REVIEW":
        return st in NEEDS_REVIEW_STATUSES
    return st == f or normalize_review_status(f) == st


def load_families_with_reviews(
    data_dir: str,
    run_id: str,
    *,
    min_size: int = 2,
    status_filter: str = FILTER_NEEDS_REVIEW,
) -> list[dict[str, Any]]:
    """Families enriched with review / suggestion fields."""
    reviews = load_family_reviews(data_dir, run_id)
    out: list[dict[str, Any]] = []
    for fam in load_families(data_dir, run_id, min_size=min_size):
        item = dict(fam)
        rev = reviews.get(str(fam.get("family_id") or "")) or {}
        item["review"] = rev
        item["selected_representative"] = rev.get("selected_representative")
        item["experiment_representative"] = rev.get("experiment_representative")
        item["suggested_representative"] = rev.get("suggested_representative")
        item["review_status"] = rev.get("status")
        item["review_reason_code"] = rev.get("reason_code")
        item["review_reason_text"] = rev.get("reason_text")
        item["confidence"] = rev.get("confidence")
        item["score_gap"] = rev.get("score_gap")
        item["suggested_score"] = rev.get("suggested_score")
        item["second_score"] = rev.get("second_score")
        item["decision_source"] = rev.get("decision_source")
        item["overridden"] = rev.get("overridden")
        if not _matches_filter(item.get("review_status"), status_filter):
            continue
        out.append(item)
    return out


def lookup_family_for_feature(
    data_dir: str, run_id: str, feature_name: str
) -> dict[str, Any] | None:
    """Return family + review context for a feature, or None."""
    feat = str(feature_name or "").strip()
    if not feat:
        return None
    with _AnalysisDb(data_dir) as conn:
        ensure_family_review_schema(conn)
        row = conn.execute(
            """
            SELECT m.family_id, f.family_label, f.size, f.members_json,
                   f.candidate_reps_json, f.avg_corr, f.max_corr
            FROM feature_family_members m
            JOIN feature_families f
              ON f.run_id = m.run_id AND f.family_id = m.family_id
            WHERE m.run_id = ? AND m.feature = ?
            """,
            (run_id, feat),
        ).fetchone()
        if not row:
            return None
        item = dict(row)
        try:
            item["members"] = json.loads(str(item.get("members_json") or "[]"))
        except Exception:
            item["members"] = []
        try:
            item["candidates"] = json.loads(
                str(item.get("candidate_reps_json") or "[]")
            )
        except Exception:
            item["candidates"] = []
        rev = load_family_review(data_dir, run_id, str(item["family_id"])) or {}
        item["review"] = rev
        item["selected_representative"] = rev.get("selected_representative")
        item["experiment_representative"] = rev.get("experiment_representative")
        item["suggested_representative"] = rev.get("suggested_representative")
        item["confidence"] = rev.get("confidence")
        item["score_gap"] = rev.get("score_gap")
        item["suggested_score"] = rev.get("suggested_score")
        item["second_score"] = rev.get("second_score")
        item["review_status"] = rev.get("status")
        return item


def format_family_context_text(
    data_dir: str,
    run_id: str,
    family_or_feature: dict[str, Any] | str,
    *,
    scores: dict[str, float] | None = None,
) -> str:
    """Human-readable family context for Scorecard / Family Review."""
    if isinstance(family_or_feature, str):
        fam = lookup_family_for_feature(data_dir, run_id, family_or_feature)
    else:
        fam = family_or_feature
    if not fam:
        return "No HCA family found for this feature.\n"
    if scores is None:
        scores = _load_discovery_scores(data_dir, run_id)

    label = fam.get("family_label") or fam.get("family_id")
    members = list(fam.get("members") or [])
    ranked = rank_family_members_by_score(members, scores)
    suggested = fam.get("suggested_representative")
    selected = (
        fam.get("experiment_representative")
        or fam.get("selected_representative")
    )
    conf = fam.get("confidence") or "—"
    gap = fam.get("score_gap")
    status = fam.get("review_status") or "—"

    lines = [
        f"Family                       {label} ({fam.get('family_id')})",
        f"Members                      {fam.get('size') or len(members)}",
        f"Suggested (Discovery)        {suggested or '—'}",
        f"Experiment Representative    {selected or '(none yet)'}",
        f"Confidence                   {conf}",
        f"Score gap                    {float(gap):.0f}"
        if gap is not None
        else "Score gap                    —",
        f"Family status                {status}",
        "",
        "Candidate representatives (Discovery Scores — not final winners):",
    ]
    for name, sc in ranked[:8]:
        mark = ""
        if suggested and name == suggested:
            mark = "  ← suggested"
        if selected and name == selected:
            mark = "  ← experiment"
        lines.append(f"  {name:<40} Score {sc:.0f}{mark}")
    lines.append("")
    if conf in ("Low", "Medium") or str(status) in NEEDS_REVIEW_STATUSES:
        lines.append(
            "Why an experiment pick is needed:\n"
            "  Top candidates have similar Discovery Scores.\n"
            "  Choose which representative to test in the next experiment —\n"
            "  training / validation decide the winner, not Discovery alone."
        )
    else:
        lines.append(
            "Clear Discovery leader → Suggested Default for the next experiment.\n"
            "  Still not permanent — compare alternative candidates after training."
        )
    return "\n".join(lines)


def sync_scorecard_family_links(
    data_dir: str,
    run_id: str,
    *,
    min_size: int = 2,
) -> dict[str, Any]:
    """Mark ambiguous-family members as REVIEW FAMILY on the Discovery Scorecard.

    Links each feature to its HCA family so the UI can open Family Review.
    Does not change KEEP / MERGE / RETIRE for features outside Needs Review
    families (except attaching family_id when present).
    """
    from .analysis_feature_rating import ACTION_REVIEW, ensure_rating_schema

    rid = str(run_id or "").strip()
    scores = _load_discovery_scores(data_dir, rid)
    families = load_families_with_reviews(
        data_dir, rid, min_size=min_size, status_filter=FILTER_ALL
    )
    stamp = _now_iso()
    n_linked = 0
    n_review = 0

    with _AnalysisDb(data_dir) as conn:
        ensure_rating_schema(conn)
        ensure_family_review_schema(conn)
        for fam in families:
            fid = str(fam.get("family_id") or "")
            label = str(fam.get("family_label") or fid)
            members = list(fam.get("members") or [])
            status = str(fam.get("review_status") or "")
            needs = status in NEEDS_REVIEW_STATUSES
            suggested = fam.get("suggested_representative")
            conf = str(fam.get("confidence") or "Medium")
            gap = fam.get("score_gap")
            ranked = rank_family_members_by_score(members, scores)
            top_lines = ", ".join(
                f"{n}={s:.0f}" for n, s in ranked[:3]
            )

            if needs:
                reason = (
                    f"Review within {label} · "
                    f"Suggested {suggested or '—'} · "
                    f"Confidence {conf}"
                    + (f" · gap {float(gap):.0f}" if gap is not None else "")
                    + (f" · candidates {top_lines}" if top_lines else "")
                    + " · Open Family Review to choose Experiment Representative"
                )
            else:
                reason = None

            for feat in members:
                n_linked += 1
                if needs:
                    n_review += 1
                    conn.execute(
                        """
                        UPDATE feature_profiles
                        SET recommendation = ?,
                            rating_action = ?,
                            rating_confidence = ?,
                            rating_reason = ?,
                            reason = ?,
                            rating_family_id = ?,
                            rating_family_label = ?,
                            updated_at = ?
                        WHERE run_id = ? AND feature_name = ?
                        """,
                        (
                            ACTION_REVIEW,
                            ACTION_REVIEW,
                            conf,
                            reason,
                            reason,
                            fid,
                            label,
                            stamp,
                            rid,
                            feat,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE feature_profiles
                        SET rating_family_id = ?,
                            rating_family_label = ?,
                            updated_at = ?
                        WHERE run_id = ? AND feature_name = ?
                        """,
                        (fid, label, stamp, rid, feat),
                    )

    return {
        "n_features_linked": n_linked,
        "n_review_family": n_review,
        "n_families": len(families),
    }


def review_summary(
    data_dir: str, run_id: str, *, min_size: int = 2
) -> dict[str, Any]:
    # Load all (no filter) for counts
    reviews = load_family_reviews(data_dir, run_id)
    families = load_families(data_dir, run_id, min_size=min_size)
    counts = {
        STATUS_SUGGESTED_DEFAULT: 0,
        STATUS_REVIEW_RECOMMENDED: 0,
        STATUS_DECISION_REQUIRED: 0,
        STATUS_FOR_EXPERIMENT: 0,
        STATUS_DEFERRED: 0,
        STATUS_NEEDS_MORE_DATA: 0,
        "Unreviewed": 0,
    }
    for fam in families:
        fid = str(fam.get("family_id") or "")
        rev = reviews.get(fid) or {}
        st = normalize_review_status(rev.get("status"))
        if st in counts:
            counts[st] += 1
        else:
            counts["Unreviewed"] += 1
    needs = (
        counts[STATUS_REVIEW_RECOMMENDED] + counts[STATUS_DECISION_REQUIRED]
    )
    return {
        "n_families": len(families),
        "counts": counts,
        "n_needs_review": needs,
        "n_auto_accepted": counts[STATUS_SUGGESTED_DEFAULT],
        "n_for_experiment": counts[STATUS_FOR_EXPERIMENT],
        "n_selected": sum(
            1
            for fam in families
            if (reviews.get(str(fam.get("family_id") or "")) or {}).get(
                "selected_representative"
            )
            or (reviews.get(str(fam.get("family_id") or "")) or {}).get(
                "experiment_representative"
            )
        ),
    }


def current_experiment_reps(
    data_dir: str,
    run_id: str,
    *,
    min_size: int = 2,
) -> dict[str, str]:
    """Family_id → experiment representative from Family Review state."""
    out: dict[str, str] = {}
    for fam in load_families_with_reviews(
        data_dir, run_id, min_size=min_size, status_filter=FILTER_ALL
    ):
        fid = str(fam.get("family_id") or "")
        if not fid:
            continue
        rep = (
            str(fam.get("experiment_representative") or "").strip()
            or str(fam.get("suggested_representative") or "").strip()
        )
        if rep:
            out[fid] = rep
    return out


def discovery_readiness(
    data_dir: str,
    run_id: str,
    *,
    min_size: int = 2,
    strategy: str | None = None,
) -> dict[str, Any]:
    """Answer: is Discovery complete enough to create an Experiment?

    Required modules depend on Feature Selection Strategy. HCA strategies
    also need an Experiment Representative per multi-member family.
    """
    from .analysis_feature_selection import (
        STRATEGY_HCA,
        STRATEGY_LABELS,
        normalize_strategy,
        required_modules_for_strategy,
    )
    from .analysis_lab_store import STATUS_COMPLETED, module_statuses

    rid = str(run_id or "").strip()
    sid = normalize_strategy(strategy)
    required = required_modules_for_strategy(sid)
    statuses = {
        str(r.get("module_id") or ""): str(r.get("status") or "")
        for r in (module_statuses(data_dir, rid) if rid else [])
    }
    modules_ok = {
        mid: statuses.get(mid) == STATUS_COMPLETED for mid in required
    }
    missing_modules = [m for m, ok in modules_ok.items() if not ok]

    uses_hca = sid == STRATEGY_HCA
    families = (
        load_families(data_dir, rid, min_size=min_size)
        if rid and uses_hca
        else []
    )
    n_families = len(families)
    reps = (
        current_experiment_reps(data_dir, rid, min_size=min_size)
        if rid and uses_hca
        else {}
    )
    n_assigned = sum(
        1 for f in families if str(f.get("family_id") or "") in reps
    )
    n_reps = len(reps)

    summary = (
        review_summary(data_dir, rid, min_size=min_size)
        if rid and uses_hca
        else {"n_needs_review": 0}
    )
    n_needs = int(summary.get("n_needs_review") or 0)

    unassigned = [
        str(f.get("family_id"))
        for f in families
        if str(f.get("family_id") or "") not in reps
    ]
    strat_label = STRATEGY_LABELS.get(sid, sid)
    if uses_hca:
        complete = (
            bool(rid)
            and not missing_modules
            and n_families > 0
            and n_assigned == n_families
            and n_needs == 0
        )
        ready_to_create = (
            bool(rid)
            and not missing_modules
            and n_families > 0
            and n_assigned == n_families
        )
    else:
        complete = bool(rid) and not missing_modules
        ready_to_create = complete

    if complete:
        headline = "Discovery  ✓ Complete"
        if uses_hca:
            detail = (
                f"Strategy: {strat_label}\n"
                f"{n_assigned} / {n_families} Families Assigned\n"
                f"{n_reps} Experiment Representatives\n"
                f"Ready to freeze discovery_bundle → create Experiment."
            )
        else:
            detail = (
                f"Strategy: {strat_label}\n"
                f"Required modules complete.\n"
                f"Ready to freeze Final Feature Dataset → create Experiment."
            )
    elif ready_to_create and n_needs > 0:
        headline = "Discovery  Almost ready"
        detail = (
            f"Strategy: {strat_label}\n"
            f"{n_assigned} / {n_families} Families Assigned\n"
            f"{n_needs} familie(s) still Needs Review "
            f"(Suggested Default covers them — confirm or override).\n"
            f"You can create an Experiment, or finish Family Review first."
        )
    elif missing_modules:
        headline = "Discovery  Incomplete"
        detail = (
            f"Strategy: {strat_label}\n"
            f"Pending modules: {', '.join(missing_modules)}\n"
            + (
                f"Families assigned {n_assigned} / {n_families}"
                if uses_hca
                else "Finish required Analysis modules first."
            )
        )
    elif uses_hca and n_families == 0:
        headline = "Discovery  Incomplete"
        detail = (
            f"Strategy: {strat_label}\n"
            "Run HCA (Feature Families) after Correlation."
        )
    elif uses_hca:
        headline = "Discovery  Incomplete"
        detail = (
            f"Strategy: {strat_label}\n"
            f"{n_assigned} / {n_families} Families Assigned\n"
            f"{len(unassigned)} familie(s) still need an Experiment Representative.\n"
            f"Open Family Review (Needs Review filter)."
        )
    else:
        headline = "Discovery  Incomplete"
        detail = f"Strategy: {strat_label}\nFinish required Analysis modules."

    try:
        from .analysis_artifacts import KIND_DISCOVERY_BUNDLE, latest_artifact

        bundle = latest_artifact(data_dir, rid, KIND_DISCOVERY_BUNDLE) if rid else None
        bundle_id = str(bundle["artifact_id"]) if bundle else None
        if bundle_id and complete:
            if uses_hca:
                detail = (
                    f"Strategy: {strat_label}\n"
                    f"{n_assigned} / {n_families} Families Assigned\n"
                    f"{n_reps} Experiment Representatives\n"
                    f"Frozen artifact: {bundle_id}\n"
                    f"Ready to create Experiment (hypothesis)."
                )
            else:
                detail = (
                    f"Strategy: {strat_label}\n"
                    f"Frozen artifact: {bundle_id}\n"
                    f"Ready to create Experiment (hypothesis)."
                )
            headline = "Discovery  ✓ Complete"
    except Exception:
        bundle_id = None

    return {
        "complete": complete,
        "ready_to_create": ready_to_create,
        "headline": headline,
        "detail": detail,
        "strategy": sid,
        "n_families": n_families,
        "n_assigned": n_assigned,
        "n_experiment_reps": n_reps,
        "n_needs_review": n_needs,
        "unassigned_family_ids": unassigned,
        "modules_ok": modules_ok,
        "missing_modules": missing_modules,
        "banner_text": f"{headline}\n{detail}",
        "latest_discovery_bundle_id": bundle_id,
    }


__all__ = [
    "FILTER_ALL",
    "FILTER_AUTO_ACCEPTED",
    "FILTER_CHOICES",
    "FILTER_NEEDS_REVIEW",
    "FILTER_SUGGESTED_DEFAULT",
    "GAP_HIGH",
    "GAP_MEDIUM",
    "MANUAL_STATUSES",
    "NEEDS_REVIEW_STATUSES",
    "REASON_AUTO",
    "REASON_DATA_AVAILABILITY",
    "REASON_INTERPRETABILITY",
    "REASON_OTHER",
    "REASON_PERFORMANCE",
    "REASON_STABILITY",
    "REVIEW_REASON_CHOICES",
    "REVIEW_STATUSES",
    "STATUS_ACCEPTED",
    "STATUS_AUTO_ACCEPTED",
    "STATUS_DECISION_REQUIRED",
    "STATUS_DEFERRED",
    "STATUS_FOR_EXPERIMENT",
    "STATUS_NEEDS_MORE_DATA",
    "STATUS_REVIEW_RECOMMENDED",
    "STATUS_SUGGESTED_DEFAULT",
    "SYSTEM_STATUSES",
    "apply_discovery_suggestions",
    "confidence_from_gap",
    "current_experiment_reps",
    "discovery_readiness",
    "ensure_family_review_schema",
    "format_family_context_text",
    "load_families_with_reviews",
    "load_family_review",
    "load_family_reviews",
    "lookup_family_for_feature",
    "normalize_review_status",
    "rank_family_members_by_score",
    "review_summary",
    "status_from_confidence",
    "suggest_family_representative",
    "sync_scorecard_family_links",
    "upsert_family_review",
]
