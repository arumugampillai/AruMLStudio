"""Model Improvement Lab — suggestions for the next training experiment."""

from __future__ import annotations

from typing import Any

from .feature_research import list_research_features
from .store import ModelLabStore

# Suggestion labels — experiment guidance, not facts
REC_STRONG_PROMOTE = "Strong Promote"
REC_PROMOTE = "Promote"
REC_REVIEW = "Review"
REC_WATCH = "Watch"
REC_CANDIDATE_REMOVE = "Candidate Remove"

# Backward-compatible aliases (older tests / callers)
REC_KEEP = REC_WATCH
REC_REMOVE = REC_CANDIDATE_REMOVE
REC_UNSTABLE = REC_REVIEW
REC_ADD = REC_PROMOTE

EVIDENCE_HIGH = "High"
EVIDENCE_MEDIUM = "Medium"
EVIDENCE_LOW = "Low"


def _pct(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def research_score_from_spread(spread: float | None, *, max_spread: float) -> float:
    """Map Hit% tertile |spread| to 0–100 research score."""
    if spread is None or max_spread <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * float(spread) / float(max_spread)))


def _stability_flags(tertile: dict[str, Any] | None) -> list[str]:
    """Detect contradictory Low vs High quality signals."""
    if not tertile:
        return []
    lo = tertile.get("low") or {}
    hi = tertile.get("high") or {}
    flags: list[str] = []
    lo_hit, hi_hit = _pct(lo.get("hit_rate")), _pct(hi.get("hit_rate"))
    lo_dd, hi_dd = _pct(lo.get("avg_dd_before_target")), _pct(hi.get("avg_dd_before_target"))
    lo_dir, hi_dir = _pct(lo.get("direction_accuracy")), _pct(hi.get("direction_accuracy"))
    lo_mae, hi_mae = _pct(lo.get("mae")), _pct(hi.get("mae"))

    if lo_hit is not None and hi_hit is not None and hi_hit > lo_hit + 0.03:
        if lo_dd is not None and hi_dd is not None and hi_dd > lo_dd * 1.25 and (hi_dd - lo_dd) > 0.5:
            flags.append("hit_up_dd_up")
        if lo_dir is not None and hi_dir is not None and hi_dir < lo_dir - 0.05:
            flags.append("hit_up_dir_down")
        if lo_mae is not None and hi_mae is not None and hi_mae > lo_mae * 1.25:
            flags.append("hit_up_mae_up")
    return flags


def classify_suggestion_evidence(
    *,
    total_rows: int,
    tertile: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Evidence High / Medium / Low for a suggestion.

    Based on prediction row count and tertile-band coverage.
    Trading-day stability and program consistency come later.
    """
    lo = (tertile or {}).get("low") or {}
    hi = (tertile or {}).get("high") or {}
    lo_n = int(lo.get("rows") or 0)
    hi_n = int(hi.get("rows") or 0)
    band_n = lo_n + hi_n
    coverage = (band_n / total_rows) if total_rows > 0 else None

    if total_rows >= 5_000 and band_n >= 200 and (coverage or 0) >= 0.40:
        level = EVIDENCE_HIGH
    elif total_rows >= 1_000 and band_n >= 50:
        level = EVIDENCE_MEDIUM
    else:
        level = EVIDENCE_LOW

    return {
        "evidence": level,
        "rows": total_rows,
        "band_rows": band_n,
        "coverage": coverage,
    }


def build_structured_evidence(
    *,
    research_score: float,
    model_rank: int | None,
    tertile: dict[str, Any] | None,
    evidence_meta: dict[str, Any],
) -> dict[str, Any]:
    """Compact evidence values for the detail / reason panel."""
    lo = (tertile or {}).get("low") or {}
    hi = (tertile or {}).get("high") or {}
    lo_hit, hi_hit = _pct(lo.get("hit_rate")), _pct(hi.get("hit_rate"))
    lo_mae, hi_mae = _pct(lo.get("mae")), _pct(hi.get("mae"))
    lo_dd, hi_dd = _pct(lo.get("avg_dd_before_target")), _pct(hi.get("avg_dd_before_target"))

    hit_imp = None
    if lo_hit is not None and hi_hit is not None:
        hit_imp = float(hi_hit) - float(lo_hit)
    mae_delta = None
    if lo_mae is not None and hi_mae is not None:
        mae_delta = float(hi_mae) - float(lo_mae)
    dd_delta = None
    if lo_dd is not None and hi_dd is not None:
        dd_delta = float(hi_dd) - float(lo_dd)

    cov = evidence_meta.get("coverage")
    return {
        "Research Score": round(float(research_score), 1),
        "Model Rank": model_rank if model_rank is not None else "—",
        "Hit Improvement": (
            f"{100.0 * hit_imp:+.1f}%" if hit_imp is not None else "—"
        ),
        "MAE": f"{mae_delta:+.2f}" if mae_delta is not None else "—",
        "DD": f"{dd_delta:+.2f}" if dd_delta is not None else "—",
        "Coverage": f"{100.0 * float(cov):.0f}%" if cov is not None else "—",
        "Prediction Rows": int(evidence_meta.get("rows") or 0),
        # raw for tests / sorting
        "_hit_improvement": hit_imp,
        "_mae_delta": mae_delta,
        "_dd_delta": dd_delta,
        "_coverage": cov,
    }


def recommend_action(
    *,
    research_score: float,
    model_rank: int | None,
    n_features: int,
    unstable: bool,
    in_model: bool,
) -> str:
    """
    Map research vs model-rank posture to a next-experiment suggestion.

    Model Rank: lower = more trusted by training/RFE.
    Research Score: higher = stronger prediction-behavior signal.
    """
    if unstable and research_score >= 40:
        return REC_REVIEW

    if not in_model:
        if research_score >= 90:
            return REC_STRONG_PROMOTE
        if research_score >= 70:
            return REC_PROMOTE
        return REC_WATCH

    mid_rank = max(1, n_features // 2)
    very_strong = research_score >= 90
    strong = research_score >= 70
    weak = research_score <= 30
    poor_rank = model_rank is not None and int(model_rank) >= max(mid_rank, 20)
    good_rank = model_rank is not None and int(model_rank) <= max(10, mid_rank // 2)

    if weak and (good_rank or model_rank is not None):
        return REC_CANDIDATE_REMOVE
    if weak:
        return REC_CANDIDATE_REMOVE
    if very_strong and poor_rank:
        return REC_STRONG_PROMOTE
    if strong and poor_rank:
        return REC_PROMOTE
    if strong and not poor_rank:
        return REC_WATCH  # already trusted + research-supported → observe
    return REC_WATCH


def format_evidence_text(evidence: dict[str, Any], recommendation: str) -> str:
    """Structured evidence block + short conclusion (no prose paragraphs)."""
    keys = (
        "Research Score",
        "Model Rank",
        "Hit Improvement",
        "MAE",
        "DD",
        "Coverage",
    )
    lines = [f"{k}\t{evidence.get(k, '—')}" for k in keys]
    lines.append("")
    lines.append(str(recommendation))
    return "\n".join(lines)


def compute_model_improvement(db_path: str) -> dict[str, Any]:
    """
    Model Improvement Lab payload.

    Suggestions for the next experiment using Research Score vs Model Rank,
    with Evidence (High/Medium/Low) and structured metrics.
    """
    empty: dict[str, Any] = {
        "available": False,
        "error": None,
        "features": [],
        "summary": {},
        "answers": {},
    }
    try:
        catalog = list_research_features(db_path)
        if catalog.get("error"):
            return {**empty, "error": catalog["error"]}
        feats = list(catalog.get("features") or [])
        if not feats:
            return {**empty, "error": "No researchable features in the prediction dataset."}

        spreads = [float(f["hit_spread"]) for f in feats if f.get("hit_spread") is not None]
        max_spread = max(spreads) if spreads else 0.0
        total_rows = int(catalog.get("total_predictions") or 0)

        selected: set[str] = set()
        ranking_extra: list[dict[str, Any]] = []
        with ModelLabStore(db_path) as store:
            if total_rows <= 0:
                total_rows = int(store.prediction_row_count() or 0)
            info = store.read_info()
            if info and isinstance(info.selected_features_snapshot, list):
                selected = {str(x).strip() for x in info.selected_features_snapshot if str(x).strip()}
            snap = info.feature_ranking_snapshot if info else None
            if isinstance(snap, dict):
                for row in snap.get("rows") or []:
                    if not isinstance(row, dict):
                        continue
                    name = str(row.get("feature") or "").strip()
                    if not name:
                        continue
                    if name not in {f.get("feature") for f in feats}:
                        ranking_extra.append(row)

        if not selected:
            selected = {str(f.get("feature") or "") for f in feats}

        n_features = len(feats)
        rows_out: list[dict[str, Any]] = []
        for f in feats:
            name = str(f.get("feature") or "")
            spread = f.get("hit_spread")
            score = research_score_from_spread(
                float(spread) if spread is not None else None,
                max_spread=max_spread,
            )
            model_rank = f.get("model_rank")
            if model_rank is None:
                model_rank = f.get("feature_rank")
            try:
                model_rank_i = int(model_rank) if model_rank is not None else None
            except (TypeError, ValueError):
                model_rank_i = None

            tertile = f.get("tertile") if isinstance(f.get("tertile"), dict) else None
            flags = _stability_flags(tertile)
            unstable = bool(flags) and score >= 40
            in_model = name in selected
            rec = recommend_action(
                research_score=score,
                model_rank=model_rank_i,
                n_features=n_features,
                unstable=unstable,
                in_model=in_model,
            )
            ev_meta = classify_suggestion_evidence(total_rows=total_rows, tertile=tertile)
            structured = build_structured_evidence(
                research_score=score,
                model_rank=model_rank_i,
                tertile=tertile,
                evidence_meta=ev_meta,
            )
            rows_out.append(
                {
                    "feature": name,
                    "research_score": round(score, 1),
                    "research_rank": f.get("research_rank") or f.get("rank"),
                    "model_rank": model_rank_i,
                    "hit_spread": spread,
                    "recommendation": rec,
                    "evidence": ev_meta["evidence"],
                    "evidence_meta": ev_meta,
                    "structured_evidence": structured,
                    "reason": format_evidence_text(structured, rec),
                    "unstable_flags": flags,
                    "in_model": in_model,
                }
            )

        pri = {
            REC_CANDIDATE_REMOVE: 0,
            REC_STRONG_PROMOTE: 1,
            REC_PROMOTE: 2,
            REC_REVIEW: 3,
            REC_WATCH: 4,
        }
        rows_out.sort(
            key=lambda r: (
                pri.get(str(r.get("recommendation")), 9),
                {"High": 0, "Medium": 1, "Low": 2}.get(str(r.get("evidence")), 3),
                -float(r.get("research_score") or 0),
                int(r.get("model_rank") or 10_000),
            )
        )

        def _names(rec: str) -> list[str]:
            return [str(r["feature"]) for r in rows_out if r.get("recommendation") == rec]

        summary = {
            "total": len(rows_out),
            "strong_promote": len(_names(REC_STRONG_PROMOTE)),
            "promote": len(_names(REC_PROMOTE)),
            "review": len(_names(REC_REVIEW)),
            "watch": len(_names(REC_WATCH)),
            "candidate_remove": len(_names(REC_CANDIDATE_REMOVE)),
            # aliases
            "keep": len(_names(REC_WATCH)),
            "remove": len(_names(REC_CANDIDATE_REMOVE)),
            "unstable": len(_names(REC_REVIEW)),
            "add": len(_names(REC_PROMOTE)) + len(_names(REC_STRONG_PROMOTE)),
        }
        answers = {
            "strong_promote": _names(REC_STRONG_PROMOTE)[:20],
            "promote": _names(REC_PROMOTE)[:20],
            "review": _names(REC_REVIEW)[:20],
            "watch": _names(REC_WATCH)[:20],
            "candidate_remove": _names(REC_CANDIDATE_REMOVE)[:20],
            "remove": _names(REC_CANDIDATE_REMOVE)[:20],
            "hurt_quality": _names(REC_CANDIDATE_REMOVE)[:20],
            "keep": _names(REC_WATCH)[:20],
            "unstable": _names(REC_REVIEW)[:20],
        }

        return {
            "available": True,
            "error": None,
            "features": rows_out,
            "summary": summary,
            "answers": answers,
            "max_hit_spread": max_spread,
            "ranking_extra_count": len(ranking_extra),
            "total_predictions": total_rows,
        }
    except Exception as exc:
        return {**empty, "error": str(exc)}
