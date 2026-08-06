"""Feature Rating — Discovery (Stage 1) and Validation (Stage 2).

Stage 1 — Discovery Rating
--------------------------
Uses ONLY Correlation + Mutual Information + Permutation.
Never imports SHAP. Never writes SHAP into reasons / recommendations.
Persists to discovery fields: feature_score, recommendation, rating_*.

Stage 2 — Validation Rating
---------------------------
Uses SHAP (+ later Holdout / Walk-forward / Confidence).
Persists to validation_* columns only — never overwrites Discovery Rating.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from .analysis_correlation import load_top_pairs
from .analysis_feature_roles import ROLE_PREDICTOR, classify_feature_role
from .analysis_lab_store import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    _AnalysisDb,
    _now_iso,
    module_statuses,
    set_module_status,
)
from .analysis_permutation import load_permutation_results

ProgressCb = Callable[[float, str], None]
CancelCb = Callable[[], bool]

STAGE_DISCOVERY = "discovery"
STAGE_VALIDATION = "validation"
RATING_STAGES = (STAGE_DISCOVERY, STAGE_VALIDATION)

ACTION_KEEP = "KEEP"
ACTION_REVIEW = "REVIEW FAMILY"
ACTION_MERGE = "MERGE CANDIDATE"
ACTION_RETIRE = "RETIRE CANDIDATE"

# Validation actions (model quality — not feature discovery)
VAL_PRODUCTION_READY = "PRODUCTION READY"
VAL_NEEDS_REVIEW = "NEEDS REVIEW"
VAL_UNSTABLE = "UNSTABLE"

ACTION_FILTER_ALIASES = {
    "KEEP": ACTION_KEEP,
    "REVIEW": ACTION_REVIEW,
    "REVIEW FAMILY": ACTION_REVIEW,
    "FAMILY DECISION REQUIRED": ACTION_REVIEW,
    "MERGE": ACTION_MERGE,
    "MERGE CANDIDATE": ACTION_MERGE,
    "RETIRE": ACTION_RETIRE,
    "RETIRE CANDIDATE": ACTION_RETIRE,
}

RATING_PROFILE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("rating_action", "TEXT"),
    ("rating_confidence", "TEXT"),
    ("rating_reason", "TEXT"),
    ("rating_score", "REAL"),
    ("rating_stars", "TEXT"),
    ("rating_peer", "TEXT"),
    ("rating_abs_corr", "REAL"),
    ("rating_mi_pct", "REAL"),
    ("rating_perm_pct", "REAL"),
    ("rating_model", "TEXT"),
    ("rating_target", "TEXT"),
    ("rating_stage", "TEXT"),
    ("rating_family_id", "TEXT"),
    ("rating_family_label", "TEXT"),
    # Stage 2 — never mixed into Discovery Scorecard / Recommendations
    ("validation_score", "REAL"),
    ("validation_stars", "TEXT"),
    ("validation_action", "TEXT"),
    ("validation_confidence", "TEXT"),
    ("validation_reason", "TEXT"),
    ("validation_shap_pct", "REAL"),
    ("validation_model", "TEXT"),
)


def ensure_rating_schema(conn: Any) -> None:
    from .analysis_feature_profiles import ensure_feature_profiles_schema

    ensure_feature_profiles_schema(conn)
    cols = {
        str(r[1])
        for r in conn.execute("PRAGMA table_info(feature_profiles)").fetchall()
    }
    for col, typ in RATING_PROFILE_COLUMNS:
        if col not in cols:
            conn.execute(f"ALTER TABLE feature_profiles ADD COLUMN {col} {typ}")


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


def rating_stars(score: float | None) -> str:
    if score is None:
        return "—"
    s = float(score)
    if s >= 90:
        return "★★★★★"
    if s >= 75:
        return "★★★★☆"
    if s >= 55:
        return "★★★☆☆"
    if s >= 35:
        return "★★☆☆☆"
    if s >= 15:
        return "★☆☆☆☆"
    return "☆☆☆☆☆"


def _band(pct: float | None) -> str:
    if pct is None:
        return "unknown"
    if pct >= 80:
        return "high"
    if pct >= 40:
        return "medium"
    return "low"


def _fmt_pct_label(name: str, pct: float | None) -> str:
    if pct is None:
        return f"{name} n/a"
    band = _band(pct).capitalize()
    return f"{band} {name} ({pct:.0f}th percentile)"


def _fmt_corr(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:.4f}"


def _fmt_perm(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.6f}"


def normalize_rating_stage(stage: str | None) -> str:
    s = str(stage or "").strip().lower()
    if s in ("validation", "model_validation", "production", "final"):
        return STAGE_VALIDATION
    return STAGE_DISCOVERY


def shap_module_completed(data_dir: str, run_id: str) -> bool:
    for r in module_statuses(data_dir, run_id):
        if str(r.get("module_id") or "") == "shap" and str(
            r.get("status") or ""
        ) == STATUS_COMPLETED:
            return True
    return False


def resolve_rating_stage(
    data_dir: str,
    run_id: str,
    *,
    stage: str | None = None,
    prefer_validation_if_shap: bool = False,
) -> str:
    """Default is always Discovery. Validation only when explicitly requested."""
    del data_dir, run_id, prefer_validation_if_shap  # discovery-first; no auto-upgrade
    if stage is not None and str(stage).strip():
        return normalize_rating_stage(stage)
    return STAGE_DISCOVERY


def _best_corr_lookup(
    data_dir: str, run_id: str
) -> dict[str, dict[str, Any]]:
    rows = load_top_pairs(data_dir, run_id, limit=500_000, min_abs=0.0)
    best: dict[str, dict[str, Any]] = {}
    for r in rows:
        a = str(r.get("feature_a") or "")
        b = str(r.get("feature_b") or "")
        corr = _f(r.get("correlation"))
        if corr is None:
            continue
        abs_c = abs(corr)
        for feat, peer in ((a, b), (b, a)):
            if not feat:
                continue
            cur = best.get(feat)
            if cur is None or abs_c > float(cur["abs_corr"]):
                best[feat] = {
                    "abs_corr": abs_c,
                    "peer": peer,
                    "correlation": corr,
                }
    return best


def _perm_value_map(
    data_dir: str, run_id: str, model_name: str, target: str
) -> dict[str, float]:
    if not model_name or not target:
        return {}
    out: dict[str, float] = {}
    for r in load_permutation_results(data_dir, run_id, model_name, target):
        name = str(r.get("feature_name") or "")
        imp = _f(r.get("importance"))
        if name and imp is not None:
            out[name] = float(imp)
    return out


def _discovery_score(
    *,
    mi_pct: float | None,
    perm_pct: float | None,
    abs_corr: float | None,
    coverage: float | None,
) -> float:
    """Discovery composite — Corr penalty + MI + Permutation only."""
    parts: list[tuple[float, float]] = []
    if mi_pct is not None:
        parts.append((0.45, float(mi_pct)))
    if perm_pct is not None:
        parts.append((0.40, float(perm_pct)))
    if not parts:
        return 0.0
    wsum = sum(w for w, _ in parts)
    base = sum(w * v for w, v in parts) / wsum

    penalty = 0.0
    if abs_corr is not None:
        c = float(abs_corr)
        if c >= 0.999:
            penalty += 25.0
        elif c >= 0.95:
            penalty += 12.0
        elif c >= 0.85:
            penalty += 5.0

    if coverage is not None:
        cov = float(coverage)
        cov_frac = cov / 100.0 if cov > 1.5 else cov
        if cov_frac < 0.90:
            penalty += 8.0 * (1.0 - cov_frac)

    return max(0.0, min(100.0, base - penalty))


def decide_discovery_action(
    *,
    mi_pct: float | None,
    perm_pct: float | None,
    abs_corr: float | None,
    perm_mean: float | None,
    coverage: float | None,
) -> tuple[str, str, str]:
    """Discovery decisions — Correlation / MI / Permutation only. No SHAP."""
    mi_b = _band(mi_pct)
    pe_b = _band(perm_pct)
    corr = float(abs_corr) if abs_corr is not None else None
    cov_ok = True
    if coverage is not None:
        cov = float(coverage)
        cov_frac = cov / 100.0 if cov > 1.5 else cov
        cov_ok = cov_frac >= 0.90

    metrics_bit = (
        (f"High correlation {_fmt_corr(corr)}" if corr is not None and corr >= 0.85 else "")
        + (f" · {_fmt_pct_label('Mutual Information', mi_pct)}" if mi_pct is not None else "")
        + (f" · {_fmt_pct_label('Permutation', perm_pct)}" if perm_pct is not None else "")
    ).lstrip(" ·")

    # MERGE: near-duplicate + weak permutation
    if corr is not None and corr >= 0.999 and pe_b == "low":
        reason = (
            f"High correlation {_fmt_corr(corr)} · "
            f"{_fmt_pct_label('Mutual Information', mi_pct)} · "
            f"Permutation {_fmt_perm(perm_mean)} · "
            f"Competes with similar features in the same HCA family."
        )
        return ACTION_MERGE, "High", reason

    # RETIRE: very low MI + near-zero perm, coverage OK
    mi_very_low = mi_pct is not None and float(mi_pct) < 15.0
    perm_near_zero = perm_mean is not None and abs(float(perm_mean)) < 1e-6
    if not perm_near_zero and perm_mean is None and perm_pct is not None:
        perm_near_zero = float(perm_pct) < 5.0
    if mi_very_low and perm_near_zero and cov_ok:
        reason = (
            f"{_fmt_pct_label('Mutual Information', mi_pct)} · "
            f"Permutation ≈ 0 ({_fmt_perm(perm_mean)}) · "
            f"Coverage OK"
        )
        return ACTION_RETIRE, "High", reason

    # KEEP: high MI + high Perm + not a duplicate
    no_dup = corr is None or corr < 0.95
    if mi_b == "high" and pe_b == "high" and no_dup:
        reason = (
            f"{_fmt_pct_label('Mutual Information', mi_pct)} · "
            f"{_fmt_pct_label('Permutation', perm_pct)} · "
            f"Low redundancy"
        )
        return ACTION_KEEP, "High", reason

    highs = sum(1 for b in (mi_b, pe_b) if b == "high")
    soft_ok = highs >= 1 and (
        mi_b == "high" or pe_b == "high"
    ) and mi_b != "low" and pe_b != "low" and (
        corr is None or corr < 0.999
    )
    if soft_ok:
        reason = metrics_bit or (
            f"{_fmt_pct_label('Mutual Information', mi_pct)} · "
            f"{_fmt_pct_label('Permutation', perm_pct)}"
        )
        return ACTION_KEEP, "Medium", reason

    # REVIEW FAMILY — always a family decision, never isolated feature review
    if corr is not None and corr >= 0.85:
        reason = (
            f"High correlation {_fmt_corr(corr)} · "
            f"{_fmt_pct_label('Mutual Information', mi_pct)} · "
            f"{_fmt_pct_label('Permutation', perm_pct)} · "
            f"Competes with similar features in the same HCA family."
        )
        return ACTION_REVIEW, "Medium", reason

    conf = "Low" if highs == 0 else "Medium"
    reason = (
        (metrics_bit or "Mixed Discovery signals")
        + " · Competes with similar features in the same HCA family."
    )
    return ACTION_REVIEW, conf, reason


def decide_action(
    *,
    mi_pct: float | None,
    shap_pct: float | None = None,
    perm_pct: float | None,
    abs_corr: float | None,
    shap_share: float | None = None,
    perm_mean: float | None,
    coverage: float | None,
    stage: str = STAGE_DISCOVERY,
) -> tuple[str, str, str]:
    """Backward-compatible wrapper. Discovery path ignores SHAP entirely."""
    stage = normalize_rating_stage(stage)
    if stage == STAGE_DISCOVERY:
        return decide_discovery_action(
            mi_pct=mi_pct,
            perm_pct=perm_pct,
            abs_corr=abs_corr,
            perm_mean=perm_mean,
            coverage=coverage,
        )
    return _decide_validation_action(
        mi_pct=mi_pct,
        shap_pct=shap_pct,
        perm_pct=perm_pct,
        abs_corr=abs_corr,
        shap_share=shap_share,
        perm_mean=perm_mean,
        coverage=coverage,
    )


def _fmt_shap_share(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{100.0 * float(v):.2f}%"


def _fmt_pct(pct: float | None) -> str:
    if pct is None:
        return "n/a"
    return f"{pct:.0f}th percentile"


def _decide_validation_action(
    *,
    mi_pct: float | None,
    shap_pct: float | None,
    perm_pct: float | None,
    abs_corr: float | None,
    shap_share: float | None,
    perm_mean: float | None,
    coverage: float | None,
) -> tuple[str, str, str]:
    """Stage 2 model-quality signals (SHAP allowed)."""
    mi_b = _band(mi_pct)
    pe_b = _band(perm_pct)
    sh_b = _band(shap_pct)
    corr = float(abs_corr) if abs_corr is not None else None

    shap_near_zero = shap_share is not None and float(shap_share) < 0.001
    if not shap_near_zero and shap_share is None and shap_pct is not None:
        shap_near_zero = float(shap_pct) < 5.0
    perm_near_zero = perm_mean is not None and abs(float(perm_mean)) < 1e-6
    if not perm_near_zero and perm_mean is None and perm_pct is not None:
        perm_near_zero = float(perm_pct) < 5.0

    if shap_near_zero and perm_near_zero and sh_b == "low" and pe_b == "low":
        return (
            VAL_UNSTABLE,
            "High",
            f"SHAP ≈ 0 ({_fmt_shap_share(shap_share)}) · "
            f"Permutation ≈ 0 ({_fmt_perm(perm_mean)}) · "
            f"Model barely uses this feature",
        )

    if sh_b == "high" and pe_b == "high":
        return (
            VAL_PRODUCTION_READY,
            "High",
            f"SHAP {_fmt_pct(shap_pct)} · "
            f"Permutation {_fmt_pct(perm_pct)} · "
            f"Model uses this feature consistently",
        )

    highs = sum(1 for b in (mi_b, sh_b, pe_b) if b == "high")
    if highs >= 2:
        return (
            VAL_PRODUCTION_READY,
            "Medium",
            f"SHAP {_fmt_pct(shap_pct)} · "
            f"Permutation {_fmt_pct(perm_pct)} · "
            f"MI {_fmt_pct(mi_pct)}",
        )

    reason = (
        f"SHAP {_fmt_pct(shap_pct)} · "
        f"Permutation {_fmt_pct(perm_pct)} · "
        f"MI {_fmt_pct(mi_pct)}"
        + (f" · Corr {_fmt_corr(corr)}" if corr is not None else "")
    )
    return VAL_NEEDS_REVIEW, "Medium" if highs else "Low", reason


def _validation_score(
    *,
    mi_pct: float | None,
    shap_pct: float | None,
    perm_pct: float | None,
) -> float:
    parts: list[tuple[float, float]] = []
    if shap_pct is not None:
        parts.append((0.45, float(shap_pct)))
    if perm_pct is not None:
        parts.append((0.30, float(perm_pct)))
    if mi_pct is not None:
        parts.append((0.25, float(mi_pct)))
    if not parts:
        return 0.0
    wsum = sum(w for w, _ in parts)
    return max(0.0, min(100.0, sum(w * v for w, v in parts) / wsum))


def _overall_score(
    *,
    mi_pct: float | None,
    shap_pct: float | None,
    perm_pct: float | None,
    abs_corr: float | None,
    coverage: float | None,
    stage: str = STAGE_DISCOVERY,
) -> float:
    stage = normalize_rating_stage(stage)
    if stage == STAGE_DISCOVERY:
        return _discovery_score(
            mi_pct=mi_pct,
            perm_pct=perm_pct,
            abs_corr=abs_corr,
            coverage=coverage,
        )
    return _validation_score(mi_pct=mi_pct, shap_pct=shap_pct, perm_pct=perm_pct)


def _resolve_model_target(
    profiles: Sequence[dict[str, Any]],
    *,
    model_name: str,
    target: str,
) -> tuple[str, str]:
    model = str(model_name or "").strip()
    tgt = str(target or "").strip()
    if not model:
        for p in profiles:
            model = str(
                p.get("rating_model")
                or p.get("permutation_model")
                or p.get("validation_model")
                or ""
            ).strip()
            if model:
                break
    if not tgt:
        for p in profiles:
            tgt = str(
                p.get("rating_target")
                or p.get("mi_target")
                or p.get("permutation_target")
                or ""
            ).strip()
            if tgt:
                break
    return model, tgt


def _abs_corr_for_profile(
    p: dict[str, Any], corr_lk: dict[str, dict[str, Any]]
) -> tuple[float | None, dict[str, Any]]:
    name = str(p.get("feature_name") or "")
    corr_info = corr_lk.get(name) or {}
    abs_corr = _f(corr_info.get("abs_corr"))
    if abs_corr is None:
        try:
            import json

            top = json.loads(str(p.get("top_corr_json") or "[]"))
            if isinstance(top, list) and top:
                abs_corr = max(
                    abs(_f(t.get("correlation")) or 0.0) for t in top
                )
                peer = str((top[0] or {}).get("feature") or "") or None
                corr_info = {
                    "abs_corr": abs_corr,
                    "peer": peer,
                    "correlation": _f((top[0] or {}).get("correlation")),
                }
        except Exception:
            pass
    return abs_corr, corr_info


def run_feature_rating(
    data_dir: str,
    run_id: str,
    *,
    model_name: str = "",
    target: str = "",
    stage: str | None = None,
    on_progress: ProgressCb | None = None,
    should_cancel: CancelCb | None = None,
) -> dict[str, Any]:
    """Run Discovery or Validation rating. Discovery never touches SHAP."""
    rid = str(run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required")

    rating_stage = normalize_rating_stage(stage) if stage else STAGE_DISCOVERY
    if rating_stage == STAGE_DISCOVERY:
        return _run_discovery_rating(
            data_dir,
            rid,
            model_name=model_name,
            target=target,
            on_progress=on_progress,
            should_cancel=should_cancel,
        )
    return _run_validation_rating(
        data_dir,
        rid,
        model_name=model_name,
        target=target,
        on_progress=on_progress,
        should_cancel=should_cancel,
    )


def _run_discovery_rating(
    data_dir: str,
    run_id: str,
    *,
    model_name: str = "",
    target: str = "",
    on_progress: ProgressCb | None = None,
    should_cancel: CancelCb | None = None,
) -> dict[str, Any]:
    """Stage 1 — Correlation + MI + Permutation only. No SHAP import."""
    rid = run_id
    started = _now_iso()
    t0 = __import__("time").perf_counter()
    set_module_status(
        data_dir,
        rid,
        "feature_scorecard",
        STATUS_RUNNING,
        started_at=started,
        message="Feature Discovery: combining Correlation / MI / Permutation…",
    )
    try:
        if on_progress:
            on_progress(0.02, "Loading profiles (Feature Discovery)…")

        with _AnalysisDb(data_dir) as conn:
            ensure_rating_schema(conn)
            raw = conn.execute(
                "SELECT * FROM feature_profiles WHERE run_id = ?",
                (rid,),
            ).fetchall()
            profiles = [dict(r) for r in raw]

        predictors = [
            p
            for p in profiles
            if classify_feature_role(str(p.get("feature_name") or ""))
            == ROLE_PREDICTOR
        ]
        if not predictors:
            raise ValueError(
                "No predictor profiles found — rebuild Feature Profiles first."
            )

        model, tgt = _resolve_model_target(
            predictors, model_name=model_name, target=target
        )
        if on_progress:
            on_progress(0.08, "Loading correlation / permutation lookups…")

        corr_lk = _best_corr_lookup(data_dir, rid)
        perm_vals = _perm_value_map(data_dir, rid, model, tgt)

        n = len(predictors)
        summary = {
            ACTION_KEEP: 0,
            ACTION_REVIEW: 0,
            ACTION_MERGE: 0,
            ACTION_RETIRE: 0,
        }
        stamp = _now_iso()

        with _AnalysisDb(data_dir) as conn:
            ensure_rating_schema(conn)
            for i, p in enumerate(predictors):
                if should_cancel and should_cancel():
                    raise RuntimeError("cancelled")
                name = str(p.get("feature_name") or "")
                if not name:
                    continue

                mi_pct = _f(p.get("mi_percentile"))
                perm_pct = _f(p.get("permutation_percentile"))
                coverage = _f(p.get("coverage"))
                abs_corr, corr_info = _abs_corr_for_profile(p, corr_lk)

                score = _discovery_score(
                    mi_pct=mi_pct,
                    perm_pct=perm_pct,
                    abs_corr=abs_corr,
                    coverage=coverage,
                )
                action, confidence, reason = decide_discovery_action(
                    mi_pct=mi_pct,
                    perm_pct=perm_pct,
                    abs_corr=abs_corr,
                    perm_mean=perm_vals.get(name)
                    if name in perm_vals
                    else _f(p.get("permutation_importance")),
                    coverage=coverage,
                )
                summary[action] = summary.get(action, 0) + 1
                stars = rating_stars(score)
                peer = str(corr_info.get("peer") or "") or None

                conn.execute(
                    """
                    UPDATE feature_profiles
                    SET feature_score = ?,
                        recommendation = ?,
                        reason = ?,
                        rating_action = ?,
                        rating_confidence = ?,
                        rating_reason = ?,
                        rating_score = ?,
                        rating_stars = ?,
                        rating_peer = ?,
                        rating_abs_corr = ?,
                        rating_mi_pct = ?,
                        rating_perm_pct = ?,
                        rating_model = ?,
                        rating_target = ?,
                        rating_stage = ?,
                        updated_at = ?
                    WHERE run_id = ? AND feature_name = ?
                    """,
                    (
                        float(score),
                        action,
                        reason,
                        action,
                        confidence,
                        reason,
                        float(score),
                        stars,
                        peer,
                        abs_corr,
                        mi_pct,
                        perm_pct,
                        model or None,
                        tgt or None,
                        STAGE_DISCOVERY,
                        stamp,
                        rid,
                        name,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO recommendations(run_id, feature, action, reason)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(run_id, feature) DO UPDATE SET
                        action = excluded.action,
                        reason = excluded.reason
                    """,
                    (rid, name, action, reason),
                )
                if on_progress and (i % 25 == 0 or i + 1 == n):
                    on_progress((i + 1) / max(n, 1), f"Rating {i + 1}/{n}: {name}")

        elapsed = round(max(__import__("time").perf_counter() - t0, 0.0), 3)
        msg = (
            f"Feature Discovery · Rated {n} predictors · "
            f"KEEP {summary[ACTION_KEEP]} · "
            f"REVIEW FAMILY {summary[ACTION_REVIEW]} · "
            f"MERGE {summary[ACTION_MERGE]} · "
            f"RETIRE {summary[ACTION_RETIRE]}"
        )
        family_suggestions: dict[str, Any] | None = None
        try:
            from .analysis_family_review import (
                apply_discovery_suggestions,
                sync_scorecard_family_links,
            )

            if on_progress:
                on_progress(0.97, "Suggesting family representatives…")
            family_suggestions = apply_discovery_suggestions(data_dir, rid)
            if family_suggestions.get("message"):
                msg = f"{msg} · {family_suggestions['message']}"
            link_out = sync_scorecard_family_links(data_dir, rid)
            if link_out.get("n_review_family"):
                msg = (
                    f"{msg} · REVIEW FAMILY linked for "
                    f"{link_out['n_review_family']} features"
                )
        except Exception as fam_exc:
            family_suggestions = {"error": str(fam_exc)}
            msg = f"{msg} · Family suggestions skipped: {fam_exc}"

        set_module_status(
            data_dir,
            rid,
            "feature_scorecard",
            STATUS_COMPLETED,
            started_at=started,
            finished_at=_now_iso(),
            elapsed_sec=elapsed,
            message=msg,
        )
        try:
            from .analysis_artifacts import publish_module_artifact

            art = publish_module_artifact(
                data_dir,
                rid,
                "feature_scorecard",
                summary={
                    "n_features": n,
                    "KEEP": summary.get(ACTION_KEEP, 0),
                    "REVIEW_FAMILY": summary.get(ACTION_REVIEW, 0),
                    "MERGE": summary.get(ACTION_MERGE, 0),
                    "RETIRE": summary.get(ACTION_RETIRE, 0),
                },
            )
            if art:
                msg = f"{msg} · artifact {art.get('artifact_id')}"
                set_module_status(
                    data_dir,
                    rid,
                    "feature_scorecard",
                    STATUS_COMPLETED,
                    started_at=started,
                    finished_at=_now_iso(),
                    elapsed_sec=elapsed,
                    message=msg,
                )
        except Exception:
            pass
        if on_progress:
            on_progress(1.0, "Discovery rating complete")
        return {
            "run_id": rid,
            "model_name": model,
            "target": tgt,
            "stage": STAGE_DISCOVERY,
            "stage_label": "Feature Discovery",
            "n_features": n,
            "summary": summary,
            "message": msg,
            "family_suggestions": family_suggestions,
            "modules_used": {
                "correlation": bool(corr_lk),
                "mutual_information": any(
                    _f(p.get("mi_percentile")) is not None for p in predictors
                ),
                "shap": False,
                "permutation": any(
                    _f(p.get("permutation_percentile")) is not None
                    for p in predictors
                ),
            },
        }
    except Exception as exc:
        set_module_status(
            data_dir,
            rid,
            "feature_scorecard",
            STATUS_FAILED,
            started_at=started,
            finished_at=_now_iso(),
            message=str(exc),
        )
        raise


def _run_validation_rating(
    data_dir: str,
    run_id: str,
    *,
    model_name: str = "",
    target: str = "",
    on_progress: ProgressCb | None = None,
    should_cancel: CancelCb | None = None,
) -> dict[str, Any]:
    """Stage 2 — SHAP-aware validation. Does NOT overwrite Discovery fields."""
    # Lazy import — Discovery path must never load analysis_shap
    from .analysis_shap import load_shap_results

    rid = run_id
    started = _now_iso()
    t0 = __import__("time").perf_counter()
    if on_progress:
        on_progress(0.02, "Loading profiles (Model Validation)…")

    with _AnalysisDb(data_dir) as conn:
        ensure_rating_schema(conn)
        raw = conn.execute(
            "SELECT * FROM feature_profiles WHERE run_id = ?",
            (rid,),
        ).fetchall()
        profiles = [dict(r) for r in raw]

    predictors = [
        p
        for p in profiles
        if classify_feature_role(str(p.get("feature_name") or ""))
        == ROLE_PREDICTOR
    ]
    if not predictors:
        raise ValueError("No predictor profiles found.")

    model, tgt = _resolve_model_target(
        predictors, model_name=model_name, target=target
    )
    if not model:
        raise ValueError("Model name required for Validation Rating (SHAP).")

    shap_rows = load_shap_results(data_dir, rid, model)
    total_shap = sum(
        float(r.get("importance") or 0.0) for r in shap_rows
    ) or 0.0
    shap_share = {
        str(r.get("feature")): (
            float(r.get("importance") or 0.0) / total_shap
            if total_shap > 0
            else 0.0
        )
        for r in shap_rows
        if r.get("feature")
    }
    shap_pct_map = {
        str(r.get("feature")): _f(r.get("percentile"))
        for r in shap_rows
        if r.get("feature")
    }
    perm_vals = _perm_value_map(data_dir, rid, model, tgt)
    corr_lk = _best_corr_lookup(data_dir, rid)

    n = len(predictors)
    summary = {
        VAL_PRODUCTION_READY: 0,
        VAL_NEEDS_REVIEW: 0,
        VAL_UNSTABLE: 0,
    }
    stamp = _now_iso()

    with _AnalysisDb(data_dir) as conn:
        ensure_rating_schema(conn)
        for i, p in enumerate(predictors):
            if should_cancel and should_cancel():
                raise RuntimeError("cancelled")
            name = str(p.get("feature_name") or "")
            if not name:
                continue
            mi_pct = _f(p.get("mi_percentile"))
            perm_pct = _f(p.get("permutation_percentile"))
            shap_pct = shap_pct_map.get(name)
            if shap_pct is None:
                shap_pct = _f(p.get("shap_percentile"))
            abs_corr, _ = _abs_corr_for_profile(p, corr_lk)
            coverage = _f(p.get("coverage"))

            score = _validation_score(
                mi_pct=mi_pct, shap_pct=shap_pct, perm_pct=perm_pct
            )
            action, confidence, reason = _decide_validation_action(
                mi_pct=mi_pct,
                shap_pct=shap_pct,
                perm_pct=perm_pct,
                abs_corr=abs_corr,
                shap_share=shap_share.get(name),
                perm_mean=perm_vals.get(name)
                if name in perm_vals
                else _f(p.get("permutation_importance")),
                coverage=coverage,
            )
            summary[action] = summary.get(action, 0) + 1
            # Write ONLY validation_* — Discovery Scorecard stays clean
            conn.execute(
                """
                UPDATE feature_profiles
                SET validation_score = ?,
                    validation_stars = ?,
                    validation_action = ?,
                    validation_confidence = ?,
                    validation_reason = ?,
                    validation_shap_pct = ?,
                    validation_model = ?,
                    updated_at = ?
                WHERE run_id = ? AND feature_name = ?
                """,
                (
                    float(score),
                    rating_stars(score),
                    action,
                    confidence,
                    reason,
                    shap_pct,
                    model,
                    stamp,
                    rid,
                    name,
                ),
            )
            if on_progress and (i % 25 == 0 or i + 1 == n):
                on_progress((i + 1) / max(n, 1), f"Validation {i + 1}/{n}")

    elapsed = round(max(__import__("time").perf_counter() - t0, 0.0), 3)
    msg = (
        f"Model Validation · {n} predictors · "
        f"READY {summary[VAL_PRODUCTION_READY]} · "
        f"NEEDS REVIEW {summary[VAL_NEEDS_REVIEW]} · "
        f"UNSTABLE {summary[VAL_UNSTABLE]} · {elapsed}s"
    )
    if on_progress:
        on_progress(1.0, "Validation rating complete")
    return {
        "run_id": rid,
        "model_name": model,
        "target": tgt,
        "stage": STAGE_VALIDATION,
        "stage_label": "Model Validation",
        "n_features": n,
        "summary": summary,
        "message": msg,
        "modules_used": {"shap": True, "permutation": bool(perm_vals)},
    }


def load_feature_ratings(
    data_dir: str,
    run_id: str,
    *,
    action_filter: str | Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Discovery recommendations only (never validation / SHAP actions)."""
    with _AnalysisDb(data_dir) as conn:
        ensure_rating_schema(conn)
        rows = conn.execute(
            """
            SELECT feature_name, category, source, cluster_id, coverage,
                   feature_score, recommendation, reason,
                   rating_action, rating_confidence, rating_reason,
                   rating_score, rating_stars, rating_peer, rating_abs_corr,
                   rating_mi_pct, rating_perm_pct,
                   rating_model, rating_target, rating_stage,
                   rating_family_id, rating_family_label,
                   mi_percentile, permutation_percentile,
                   mi_score, permutation_importance,
                   feature_role
            FROM feature_profiles
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        item = dict(r)
        name = str(item.get("feature_name") or "")
        if classify_feature_role(name) != ROLE_PREDICTOR:
            continue
        out.append(item)

    if action_filter:
        wanted: set[str] = set()
        items = (
            [action_filter]
            if isinstance(action_filter, str)
            else list(action_filter)
        )
        for a in items:
            key = str(a or "").strip().upper()
            if not key or key == "ALL":
                continue
            wanted.add(ACTION_FILTER_ALIASES.get(key, key))
        if wanted:
            out = [
                r
                for r in out
                if str(r.get("rating_action") or r.get("recommendation") or "")
                in wanted
            ]

    out.sort(
        key=lambda r: (
            -float(r.get("feature_score") or r.get("rating_score") or 0.0),
            str(r.get("feature_name") or ""),
        )
    )
    return out


def format_score_cell(row: dict[str, Any]) -> str:
    score = _f(
        row.get("feature_score")
        if row.get("feature_score") is not None
        else row.get("rating_score")
    )
    if score is None:
        return "Pending"
    action = str(row.get("rating_action") or row.get("recommendation") or "").strip()
    stars = str(row.get("rating_stars") or rating_stars(score))
    if action:
        return f"{score:.0f} {stars} {action}"
    return f"{score:.0f} {stars}"


def current_rating_stage(data_dir: str, run_id: str) -> str:
    rows = load_feature_ratings(data_dir, run_id)
    for r in rows:
        st = str(r.get("rating_stage") or "").strip()
        if st:
            return normalize_rating_stage(st)
    return STAGE_DISCOVERY


__all__ = [
    "ACTION_KEEP",
    "ACTION_MERGE",
    "ACTION_RETIRE",
    "ACTION_REVIEW",
    "STAGE_DISCOVERY",
    "STAGE_VALIDATION",
    "VAL_NEEDS_REVIEW",
    "VAL_PRODUCTION_READY",
    "VAL_UNSTABLE",
    "current_rating_stage",
    "decide_action",
    "decide_discovery_action",
    "ensure_rating_schema",
    "format_score_cell",
    "load_feature_ratings",
    "normalize_rating_stage",
    "rating_stars",
    "resolve_rating_stage",
    "run_feature_rating",
    "shap_module_completed",
]
