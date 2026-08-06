"""Research Programs — hypothesis playbooks over the prediction dataset."""

from __future__ import annotations

from typing import Any

from .research_dashboard import _load_feature_map, _rate
from .store import ModelLabStore

DEFAULT_LIMIT = 1000
LIMIT_CHOICES = (100, 500, 1000, 5000)
COMPARE_DATASET = "dataset"
COMPARE_WORST = "worst"
COMPARE_BEST = "best"
COMPARE_COUNTERPART = "counterpart"


def _programs_catalog() -> list[dict[str, Any]]:
    """Static catalog — each program answers a concrete research question."""
    return [
        {
            "id": "top_success",
            "title": "Top Success Research",
            "hypothesis": "Top predictions by lowest MAE — why do they win?",
            "answer": "Why did they succeed?",
            "kind": "ranked",
            "configurable_limit": True,
            "compare_options": [COMPARE_DATASET, COMPARE_WORST],
        },
        {
            "id": "worst_predictions",
            "title": "Worst Predictions",
            "hypothesis": "Highest-MAE predictions — what went wrong?",
            "answer": "Why were they bad?",
            "kind": "ranked",
            "configurable_limit": True,
            "compare_options": [COMPARE_DATASET, COMPARE_BEST],
        },
        {
            "id": "best_vs_worst",
            "title": "Best vs Worst MAE",
            "hypothesis": "Compare lowest-MAE cohort with highest-MAE cohort.",
            "answer": "How do best and worst predictions differ?",
            "kind": "compare",
            "configurable_limit": True,
            "compare_options": [COMPARE_COUNTERPART, COMPARE_DATASET],
        },
        {
            "id": "fast_target",
            "title": "Fast Target",
            "hypothesis": "Target reached in under 20 seconds.",
            "answer": "What conditions produce very fast targets?",
            "kind": "filter",
            "compare_options": [COMPARE_DATASET],
        },
        {
            "id": "slow_target",
            "title": "Slow Target",
            "hypothesis": "Time to Target over 180 seconds.",
            "answer": "What conditions produce slow targets?",
            "kind": "filter",
            "compare_options": [COMPARE_DATASET],
        },
        {
            "id": "no_drawdown",
            "title": "No Drawdown (Dream Trades)",
            "hypothesis": "DD before Target = 0 — dream trades.",
            "answer": "What feature values appear in dream trades?",
            "kind": "filter",
            "compare_options": [COMPARE_DATASET],
        },
        {
            "id": "huge_drawdown",
            "title": "Huge Drawdown",
            "hypothesis": "DD before Target > 10.",
            "answer": "What market conditions produce large drawdowns?",
            "kind": "filter",
            "compare_options": [COMPARE_DATASET],
        },
        {
            "id": "hit_low_mae",
            "title": "Hit + Low MAE",
            "hypothesis": "Target hit with the lowest absolute error.",
            "answer": "What makes predictions both accurate and profitable?",
            "kind": "ranked",
            "configurable_limit": True,
            "compare_options": [COMPARE_DATASET, COMPARE_WORST],
        },
        {
            "id": "dir_correct_miss",
            "title": "Direction Correct, Target Miss",
            "hypothesis": "Direction right but target missed.",
            "answer": "Why was direction right but the target missed?",
            "kind": "filter",
            "compare_options": [COMPARE_DATASET],
        },
        {
            "id": "hit_dir_wrong",
            "title": "Target Hit, Direction Wrong",
            "hypothesis": "Target hit but direction wrong.",
            "answer": "What explains hit-with-wrong-direction edge cases?",
            "kind": "filter",
            "compare_options": [COMPARE_DATASET],
        },
        {
            "id": "premium_bands",
            "title": "Premium Bands",
            "hypothesis": "Compare ₹15–30 vs ₹30–50 vs ₹50–100 current LTP.",
            "answer": "How do outcomes and features differ across premium bands?",
            "kind": "bands",
            "compare_options": [COMPARE_DATASET, COMPARE_COUNTERPART],
        },
    ]


def list_research_programs() -> list[dict[str, Any]]:
    return list(_programs_catalog())


def _safe_ident(name: str) -> bool:
    return bool(name) and name.replace("_", "").isalnum()


def _ranked_id_sql(order_col: str, direction: str, *, limit: int, extra_where: str = "") -> str:
    if not _safe_ident(order_col) or direction not in ("ASC", "DESC"):
        return ""
    where = f'WHERE "{order_col}" IS NOT NULL'
    if extra_where.strip():
        where += f" AND ({extra_where})"
    return (
        f'id IN (SELECT id FROM prediction_dataset {where} '
        f'ORDER BY "{order_col}" {direction} LIMIT {int(limit)})'
    )


def _program_specs(limit: int) -> dict[str, dict[str, Any]]:
    lim = max(1, int(limit))
    return {
        "top_success": {
            "label": f"Top {lim} by lowest MAE",
            "where_sql": _ranked_id_sql("absolute_error", "ASC", limit=lim),
            "where_args": [],
            "cohorts": None,
            "counterpart": {
                "title": f"Worst {lim} (highest MAE)",
                "where_sql": _ranked_id_sql("absolute_error", "DESC", limit=lim),
                "where_args": [],
            },
        },
        "worst_predictions": {
            "label": f"Top {lim} by highest MAE",
            "where_sql": _ranked_id_sql("absolute_error", "DESC", limit=lim),
            "where_args": [],
            "cohorts": None,
            "counterpart": {
                "title": f"Best {lim} (lowest MAE)",
                "where_sql": _ranked_id_sql("absolute_error", "ASC", limit=lim),
                "where_args": [],
            },
        },
        "best_vs_worst": {
            "label": f"Best {lim} vs Worst {lim} MAE",
            "where_sql": "",
            "where_args": [],
            "cohorts": [
                {
                    "key": "best",
                    "title": f"Best {lim} (lowest MAE)",
                    "where_sql": _ranked_id_sql("absolute_error", "ASC", limit=lim),
                    "where_args": [],
                },
                {
                    "key": "worst",
                    "title": f"Worst {lim} (highest MAE)",
                    "where_sql": _ranked_id_sql("absolute_error", "DESC", limit=lim),
                    "where_args": [],
                },
            ],
        },
        "fast_target": {
            "label": "Target reached · T→Target < 20 s",
            "where_sql": (
                "target_reached = 1 AND time_to_target IS NOT NULL "
                "AND time_to_target >= 0 AND time_to_target < ?"
            ),
            "where_args": [20.0],
            "cohorts": None,
        },
        "slow_target": {
            "label": "T→Target > 180 s",
            "where_sql": "time_to_target IS NOT NULL AND time_to_target > ?",
            "where_args": [180.0],
            "cohorts": None,
        },
        "no_drawdown": {
            "label": "DD before Target = 0 (and target reached)",
            "where_sql": "target_reached = 1 AND dd_before_target IS NOT NULL AND dd_before_target = ?",
            "where_args": [0.0],
            "cohorts": None,
        },
        "huge_drawdown": {
            "label": "DD before Target > 10",
            "where_sql": "dd_before_target IS NOT NULL AND dd_before_target > ?",
            "where_args": [10.0],
            "cohorts": None,
        },
        "hit_low_mae": {
            "label": f"Hit + lowest MAE · top {lim}",
            "where_sql": _ranked_id_sql(
                "absolute_error",
                "ASC",
                limit=lim,
                extra_where="target_reached = 1",
            ),
            "where_args": [],
            "cohorts": None,
            "counterpart": {
                "title": f"Hit + highest MAE · top {lim}",
                "where_sql": _ranked_id_sql(
                    "absolute_error",
                    "DESC",
                    limit=lim,
                    extra_where="target_reached = 1",
                ),
                "where_args": [],
            },
        },
        "dir_correct_miss": {
            "label": "Direction correct · Target miss",
            "where_sql": "direction_correct = 1 AND target_reached = 0",
            "where_args": [],
            "cohorts": None,
        },
        "hit_dir_wrong": {
            "label": "Target hit · Direction wrong",
            "where_sql": "target_reached = 1 AND direction_correct = 0",
            "where_args": [],
            "cohorts": None,
        },
        "premium_bands": {
            "label": "Premium bands by current LTP",
            "where_sql": "",
            "where_args": [],
            "cohorts": [
                {
                    "key": "band_15_30",
                    "title": "₹15–30",
                    "where_sql": "current_ltp IS NOT NULL AND current_ltp >= ? AND current_ltp < ?",
                    "where_args": [15.0, 30.0],
                },
                {
                    "key": "band_30_50",
                    "title": "₹30–50",
                    "where_sql": "current_ltp IS NOT NULL AND current_ltp >= ? AND current_ltp < ?",
                    "where_args": [30.0, 50.0],
                },
                {
                    "key": "band_50_100",
                    "title": "₹50–100",
                    "where_sql": "current_ltp IS NOT NULL AND current_ltp >= ? AND current_ltp < ?",
                    "where_args": [50.0, 100.0],
                },
            ],
        },
    }


def _where_clause(where_sql: str) -> str:
    if where_sql.strip():
        return f" WHERE ({where_sql})"
    return ""


def _sql_cohort_metrics(
    store: ModelLabStore,
    *,
    where_sql: str,
    where_args: list[Any],
) -> dict[str, Any]:
    cols = store._prediction_table_columns()
    where = _where_clause(where_sql)
    args = list(where_args)

    parts = ["COUNT(*) AS n"]
    if "target_reached" in cols:
        parts.append("SUM(CASE WHEN target_reached IS NOT NULL THEN 1 ELSE 0 END) AS n_hit")
        parts.append("SUM(CASE WHEN target_reached = 1 THEN 1 ELSE 0 END) AS hit")
    if "direction_correct" in cols:
        parts.append("SUM(CASE WHEN direction_correct IS NOT NULL THEN 1 ELSE 0 END) AS n_dir")
        parts.append("SUM(CASE WHEN direction_correct = 1 THEN 1 ELSE 0 END) AS dir_h")
    if "absolute_error" in cols:
        parts.append("AVG(CASE WHEN absolute_error IS NOT NULL THEN absolute_error END) AS mae")
    if "dd_before_target" in cols:
        parts.append(
            "AVG(CASE WHEN dd_before_target IS NOT NULL THEN dd_before_target END) AS avg_dd"
        )
    if "time_to_target" in cols:
        parts.append(
            "AVG(CASE WHEN time_to_target IS NOT NULL AND time_to_target >= 0 "
            "THEN time_to_target END) AS avg_ttt"
        )
    if "premium_error_pct" in cols:
        parts.append(
            "AVG(CASE WHEN premium_error_pct IS NOT NULL THEN premium_error_pct END) AS prem"
        )
    if "current_ltp" in cols:
        parts.append("AVG(CASE WHEN current_ltp IS NOT NULL THEN current_ltp END) AS avg_ltp")

    sql = f"SELECT {', '.join(parts)} FROM prediction_dataset{where}"
    row = store.conn.execute(sql, args).fetchone()
    if not row:
        return {"rows": 0}

    aliases = [p.rsplit(" AS ", 1)[-1].strip() if " AS " in p else p for p in parts]

    def _get(alias: str) -> Any:
        try:
            return row[aliases.index(alias)]
        except (ValueError, IndexError):
            return None

    n = int(_get("n") or 0)
    n_hit, hit = _get("n_hit"), _get("hit")
    n_dir, dir_h = _get("n_dir"), _get("dir_h")
    return {
        "rows": n,
        "hit_rate": _rate(float(hit or 0), float(n_hit or 0)) if n_hit is not None else None,
        "direction_accuracy": (
            _rate(float(dir_h or 0), float(n_dir or 0)) if n_dir is not None else None
        ),
        "mae": float(_get("mae")) if _get("mae") is not None else None,
        "avg_dd_before_target": float(_get("avg_dd")) if _get("avg_dd") is not None else None,
        "avg_time_to_target": float(_get("avg_ttt")) if _get("avg_ttt") is not None else None,
        "premium_mae": float(_get("prem")) if _get("prem") is not None else None,
        "avg_current_ltp": float(_get("avg_ltp")) if _get("avg_ltp") is not None else None,
    }


def _sql_feature_means(
    store: ModelLabStore,
    feature_pairs: list[tuple[str, str]],
    *,
    where_sql: str,
    where_args: list[Any],
) -> dict[str, float | None]:
    stats = _sql_feature_stats(store, feature_pairs, where_sql=where_sql, where_args=where_args)
    return {k: v.get("mean") for k, v in stats.items()}


def _sql_feature_stats(
    store: ModelLabStore,
    feature_pairs: list[tuple[str, str]],
    *,
    where_sql: str,
    where_args: list[Any],
) -> dict[str, dict[str, Any]]:
    """Per-feature AVG + non-null count inside a cohort filter."""
    from .prediction_feature_store import PredictionFeatureStore

    access = PredictionFeatureStore.from_store(store)
    if access.is_referenced():
        return access.sql_feature_avg(
            feature_pairs, where_sql=where_sql, where_args=where_args
        )
    cols = store._prediction_table_columns()
    where = _where_clause(where_sql)
    out: dict[str, dict[str, Any]] = {}
    for name, col in feature_pairs:
        if col not in cols or not _safe_ident(col):
            out[name] = {"mean": None, "n": 0}
            continue
        if where:
            sql = (
                f'SELECT AVG("{col}"), COUNT("{col}") FROM prediction_dataset{where} '
                f'AND "{col}" IS NOT NULL'
            )
        else:
            sql = (
                f'SELECT AVG("{col}"), COUNT("{col}") FROM prediction_dataset '
                f'WHERE "{col}" IS NOT NULL'
            )
        row = store.conn.execute(sql, list(where_args)).fetchone()
        mean = float(row[0]) if row and row[0] is not None else None
        n = int(row[1] or 0) if row else 0
        out[name] = {"mean": mean, "n": n}
    return out


def format_feature_difference(cohort: float, compare: float) -> dict[str, Any]:
    """Absolute + relative difference for Why tables."""
    delta = float(cohort) - float(compare)
    pct = None
    if abs(float(compare)) > 1e-12:
        pct = 100.0 * delta / abs(float(compare))
    if pct is not None and abs(pct) >= 1.0:
        arrow = "↓" if pct < 0 else "↑"
        display = f"{arrow}{abs(pct):.0f}%"
    else:
        sign = "+" if delta >= 0 else ""
        display = f"{sign}{delta:.4g}"
    return {
        "delta": delta,
        "pct": pct,
        "display": display,
        "sort_key": abs(pct) if pct is not None else abs(delta),
    }


def classify_evidence(
    *,
    effect_pct: float | None,
    effect_abs: float | None,
    rows_affected: int,
    cohort_rows: int,
    total_rows: int,
) -> dict[str, Any]:
    """
    Evidence strength from effect size + sample support.

    Effect: Strong (|Δ|≥40%), Moderate (≥15%), Weak (else).
    Confidence: High (n≥200), Medium (n≥50), Low (else).
    Overall evidence: Strong only when effect Strong and confidence High;
    Weak when sample Low or effect Weak; else Moderate.
    """
    n = max(0, int(rows_affected))
    cohort_n = max(0, int(cohort_rows))
    total_n = max(0, int(total_rows))

    # Prefer relative effect; fall back to absolute normalized heuristically
    mag = abs(float(effect_pct)) if effect_pct is not None else None
    if mag is None and effect_abs is not None:
        mag = min(100.0, abs(float(effect_abs)) * 100.0)

    if mag is not None and mag >= 40.0:
        effect = "Strong"
    elif mag is not None and mag >= 15.0:
        effect = "Moderate"
    else:
        effect = "Weak"

    if n >= 200:
        confidence = "High"
    elif n >= 50:
        confidence = "Medium"
    else:
        confidence = "Low"

    if effect == "Strong" and confidence == "High":
        evidence = "Strong"
    elif effect == "Weak" or confidence == "Low":
        evidence = "Weak"
    else:
        evidence = "Moderate"

    # Coverage: share of cohort with this feature; also vs full dataset
    coverage_cohort = (n / cohort_n) if cohort_n else None
    coverage_dataset = (n / total_n) if total_n else None
    # Display coverage: prefer cohort coverage when cohort is defined
    coverage = coverage_cohort if coverage_cohort is not None else coverage_dataset

    return {
        "effect": effect,
        "confidence": confidence,
        "evidence": evidence,
        "rows_affected": n,
        "coverage": coverage,
        "coverage_dataset": coverage_dataset,
        "coverage_cohort": coverage_cohort,
    }


def build_why_table(
    cohort_stats: dict[str, dict[str, Any]],
    compare_means: dict[str, float | None],
    feature_pairs: list[tuple[str, str]],
    *,
    cohort_rows: int,
    total_rows: int,
    top_n: int = 12,
    min_sort: float = 5.0,
) -> list[dict[str, Any]]:
    """Largest feature differences vs compare baseline, with evidence tags."""
    rows: list[dict[str, Any]] = []
    for name, col in feature_pairs:
        st = cohort_stats.get(name) or {}
        c = st.get("mean")
        b = compare_means.get(name)
        if c is None or b is None:
            continue
        diff = format_feature_difference(float(c), float(b))
        n = int(st.get("n") or 0)
        ev = classify_evidence(
            effect_pct=diff.get("pct"),
            effect_abs=diff.get("delta"),
            rows_affected=n,
            cohort_rows=cohort_rows,
            total_rows=total_rows,
        )
        rows.append(
            {
                "feature": name,
                "column": col,
                "cohort_mean": float(c),
                "compare_mean": float(b),
                "difference": diff["display"],
                "delta": diff["delta"],
                "pct": diff["pct"],
                "sort_key": diff["sort_key"],
                "rows_affected": ev["rows_affected"],
                "coverage": ev["coverage"],
                "effect": ev["effect"],
                "confidence": ev["confidence"],
                "evidence": ev["evidence"],
            }
        )
    rows.sort(
        key=lambda r: (
            {"Strong": 0, "Moderate": 1, "Weak": 2}.get(str(r.get("evidence")), 3),
            -float(r["sort_key"]),
        )
    )
    filtered = [r for r in rows if float(r["sort_key"]) >= min_sort]
    return (filtered or rows)[:top_n]


def build_program_conclusions(
    *,
    metrics: dict[str, Any],
    compare_metrics: dict[str, Any] | None,
    why_rows: list[dict[str, Any]],
    program_id: str,
) -> dict[str, Any]:
    """Narrative bullets classified by evidence strength."""
    bullets: list[dict[str, Any]] = []
    cmp = compare_metrics or {}
    cohort_rows = int(metrics.get("rows") or 0)
    total_rows = int(metrics.get("total_rows") or 0)

    def _add(
        kind: str,
        title: str,
        *,
        detail: str = "",
        evidence: str = "Moderate",
        rows_affected: int | None = None,
        difference: str | None = None,
    ) -> None:
        bullets.append(
            {
                "kind": kind,
                "title": title,
                "detail": detail,
                "evidence": evidence,
                "rows_affected": rows_affected,
                "difference": difference,
                "text": title,  # compat
            }
        )

    def _outcome_evidence(rows: int) -> str:
        if rows >= 200:
            return "Strong"
        if rows >= 50:
            return "Moderate"
        return "Weak"

    def _lower_better(key: str, label: str, *, material: float = 0.05) -> None:
        a, b = metrics.get(key), cmp.get(key)
        if a is None or b is None:
            return
        if float(a) < float(b) and (
            float(a) < float(b) * (1.0 - material)
            or (float(b) - float(a)) > material * max(abs(float(b)), 1.0)
        ):
            _add(
                "good",
                f"Smaller {label}",
                detail=f"{float(a):.3g} vs {float(b):.3g}",
                evidence=_outcome_evidence(cohort_rows),
                rows_affected=cohort_rows,
            )
        elif float(a) > float(b) and (
            float(a) > float(b) * (1.0 + material)
            or (float(a) - float(b)) > material * max(abs(float(b)), 1.0)
        ):
            _add(
                "warn",
                f"Larger {label}",
                detail=f"{float(a):.3g} vs {float(b):.3g}",
                evidence=_outcome_evidence(cohort_rows),
                rows_affected=cohort_rows,
            )

    def _higher_better(key: str, label: str, *, material_pp: float = 0.02) -> None:
        a, b = metrics.get(key), cmp.get(key)
        if a is None or b is None:
            return
        if float(a) > float(b) + material_pp:
            _add(
                "good",
                f"Higher {label}",
                detail=f"{100 * float(a):.1f}% vs {100 * float(b):.1f}%",
                evidence=_outcome_evidence(cohort_rows),
                rows_affected=cohort_rows,
            )
        elif float(a) < float(b) - material_pp:
            _add(
                "warn",
                f"Lower {label}",
                detail=f"{100 * float(a):.1f}% vs {100 * float(b):.1f}%",
                evidence=_outcome_evidence(cohort_rows),
                rows_affected=cohort_rows,
            )

    if cmp:
        _higher_better("hit_rate", "Hit Rate")
        _higher_better("direction_accuracy", "Direction Accuracy")
        _lower_better("mae", "MAE", material=0.02)
        _lower_better("avg_dd_before_target", "drawdown", material=0.05)
        a_ttt, b_ttt = metrics.get("avg_time_to_target"), cmp.get("avg_time_to_target")
        if a_ttt is not None and b_ttt is not None and float(b_ttt) > 0:
            if float(a_ttt) < float(b_ttt) * 0.9:
                _add(
                    "good",
                    "Faster target hit",
                    detail=f"{float(a_ttt):.1f}s vs {float(b_ttt):.1f}s",
                    evidence=_outcome_evidence(cohort_rows),
                    rows_affected=cohort_rows,
                )
            elif float(a_ttt) > float(b_ttt) * 1.1:
                _add(
                    "warn",
                    "Slower target hit",
                    detail=f"{float(a_ttt):.1f}s vs {float(b_ttt):.1f}s",
                    evidence=_outcome_evidence(cohort_rows),
                    rows_affected=cohort_rows,
                )

    if not cmp:
        if metrics.get("avg_dd_before_target") is not None and float(metrics["avg_dd_before_target"]) <= 0.01:
            _add("good", "Near-zero drawdown before target", evidence=_outcome_evidence(cohort_rows), rows_affected=cohort_rows)
        if metrics.get("hit_rate") is not None and float(metrics["hit_rate"]) >= 0.9:
            _add(
                "good",
                "Very high Hit Rate",
                detail=f"{100 * float(metrics['hit_rate']):.1f}%",
                evidence=_outcome_evidence(cohort_rows),
                rows_affected=cohort_rows,
            )

    ltp = metrics.get("avg_current_ltp")
    if ltp is not None:
        v = float(ltp)
        band = None
        if 15 <= v < 30:
            band = "low-premium options (≈ ₹15–30)"
        elif 30 <= v < 50:
            band = "medium-premium options (≈ ₹30–50)"
        elif 50 <= v < 100:
            band = "higher-premium options (≈ ₹50–100)"
        if band:
            ev = classify_evidence(
                effect_pct=25.0,
                effect_abs=None,
                rows_affected=cohort_rows,
                cohort_rows=cohort_rows,
                total_rows=total_rows,
            )
            _add(
                "good" if ev["evidence"] != "Weak" else "warn",
                f"Mostly {band}",
                evidence=str(ev["evidence"]),
                rows_affected=cohort_rows,
            )

    for row in why_rows[:6]:
        feat = str(row.get("feature") or "")
        if not feat:
            continue
        pretty = feat.replace("_", " ")
        pct = row.get("pct")
        direction = "Higher" if (pct is not None and float(pct) > 0) or (
            pct is None and float(row.get("delta") or 0) > 0
        ) else "Lower"
        evidence = str(row.get("evidence") or "Moderate")
        n = int(row.get("rows_affected") or 0)
        kind = "good"
        if evidence == "Weak":
            kind = "warn"
        elif program_id not in ("top_success", "hit_low_mae", "no_drawdown", "fast_target") and kind == "good":
            kind = "info"
        _add(
            kind if kind != "info" else "good",
            f"{direction} {pretty}",
            detail=str(row.get("difference") or ""),
            evidence=evidence,
            rows_affected=n,
            difference=str(row.get("difference") or ""),
        )

    # De-dupe by title
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for b in bullets:
        t = str(b.get("title") or b.get("text") or "")
        if t in seen:
            continue
        seen.add(t)
        uniq.append(b)

    if not uniq:
        uniq = [
            {
                "kind": "info",
                "title": "No strong contrasts detected yet",
                "detail": "",
                "evidence": "Weak",
                "rows_affected": cohort_rows,
                "text": "No strong contrasts detected yet",
            }
        ]

    # Strong first, then moderate, weak last
    order = {"Strong": 0, "Moderate": 1, "Weak": 2}
    uniq.sort(key=lambda b: order.get(str(b.get("evidence")), 3))

    blocks: list[str] = []
    for b in uniq[:8]:
        ev = str(b.get("evidence") or "Moderate")
        mark = "✓" if b.get("kind") == "good" else ("⚠" if b.get("kind") == "warn" else "·")
        lines = [f"{mark} {ev} evidence", str(b.get("title") or "")]
        if b.get("difference"):
            lines.append(f"({b['difference']})")
        elif b.get("detail"):
            lines.append(str(b["detail"]))
        n = b.get("rows_affected")
        if n is not None:
            if ev == "Weak" and int(n) < 50:
                lines.append(f"Only {int(n):,} rows")
            else:
                lines.append(f"Rows affected: {int(n):,}")
        blocks.append("\n".join(lines))

    return {"bullets": uniq[:8], "text": "\n------------------\n".join(blocks)}


def _why_column_labels(compare_to: str, program_id: str) -> tuple[str, str]:
    if compare_to == COMPARE_WORST:
        return "Success", "Worst"
    if compare_to == COMPARE_BEST:
        return "Worst", "Best"
    if compare_to == COMPARE_COUNTERPART:
        if program_id == "best_vs_worst":
            return "Cohort", "Other"
        return "Cohort", "Counterpart"
    return "Cohort", "Overall"


def _resolve_compare_to(program_id: str, compare_to: str | None, meta: dict[str, Any]) -> str:
    options = list(meta.get("compare_options") or [COMPARE_DATASET])
    if compare_to and compare_to in options:
        return compare_to
    # Friendly aliases
    if compare_to == COMPARE_WORST and COMPARE_WORST in options:
        return COMPARE_WORST
    if compare_to == COMPARE_BEST and COMPARE_BEST in options:
        return COMPARE_BEST
    return options[0] if options else COMPARE_DATASET


def _counterpart_means_for_compare(
    store: ModelLabStore,
    *,
    program_id: str,
    spec: dict[str, Any],
    compare_to: str,
    feature_pairs: list[tuple[str, str]],
    cohort_index: int,
    cohorts_raw: list[dict[str, Any]],
) -> tuple[dict[str, float | None], dict[str, Any], str]:
    """
    Returns (compare_feature_means, compare_metrics, compare_label).
    """
    dataset_means = _sql_feature_means(store, feature_pairs, where_sql="", where_args=[])
    dataset_metrics = _sql_cohort_metrics(store, where_sql="", where_args=[])

    if compare_to == COMPARE_DATASET:
        return dataset_means, dataset_metrics, "Whole Dataset"

    counterpart = spec.get("counterpart")
    if compare_to in (COMPARE_WORST, COMPARE_BEST) and counterpart:
        means = _sql_feature_means(
            store,
            feature_pairs,
            where_sql=str(counterpart["where_sql"]),
            where_args=list(counterpart.get("where_args") or []),
        )
        metrics = _sql_cohort_metrics(
            store,
            where_sql=str(counterpart["where_sql"]),
            where_args=list(counterpart.get("where_args") or []),
        )
        return means, metrics, str(counterpart.get("title") or "Counterpart")

    if compare_to == COMPARE_COUNTERPART and len(cohorts_raw) >= 2:
        other = cohorts_raw[1 - cohort_index] if cohort_index < 2 else cohorts_raw[0]
        means = _sql_feature_means(
            store,
            feature_pairs,
            where_sql=str(other["where_sql"]),
            where_args=list(other.get("where_args") or []),
        )
        metrics = _sql_cohort_metrics(
            store,
            where_sql=str(other["where_sql"]),
            where_args=list(other.get("where_args") or []),
        )
        return means, metrics, str(other.get("title") or "Counterpart")

    return dataset_means, dataset_metrics, "Whole Dataset"


def _cohort_payload(
    store: ModelLabStore,
    *,
    title: str,
    label: str,
    where_sql: str,
    where_args: list[Any],
    feature_pairs: list[tuple[str, str]],
    compare_means: dict[str, float | None],
    compare_metrics: dict[str, Any],
    compare_label: str,
    compare_to: str,
    program_id: str,
    why_col_a: str,
    why_col_b: str,
) -> dict[str, Any]:
    metrics = _sql_cohort_metrics(store, where_sql=where_sql, where_args=where_args)
    stats = _sql_feature_stats(
        store, feature_pairs, where_sql=where_sql, where_args=where_args
    )
    total = store.prediction_row_count()
    n = int(metrics.get("rows") or 0)
    coverage = (n / total) if total else None
    metrics = {**metrics, "coverage": coverage, "total_rows": total}
    cmp_metrics = {**compare_metrics, "total_rows": total}
    why_rows = build_why_table(
        stats,
        compare_means,
        feature_pairs,
        cohort_rows=n,
        total_rows=total,
    )
    conclusions = build_program_conclusions(
        metrics=metrics,
        compare_metrics=cmp_metrics,
        why_rows=why_rows,
        program_id=program_id,
    )
    return {
        "title": title,
        "label": label,
        "where_sql": where_sql,
        "where_args": list(where_args),
        "metrics": metrics,
        "executive_summary": metrics,
        "why_rows": why_rows,
        "why_columns": {"cohort": why_col_a, "compare": why_col_b},
        "compare_to": compare_to,
        "compare_label": compare_label,
        "conclusions": conclusions,
        "feature_profile": why_rows,
    }


def run_research_program(
    db_path: str,
    program_id: str,
    *,
    limit: int = DEFAULT_LIMIT,
    compare_to: str | None = None,
) -> dict[str, Any]:
    """
    Run one Research Program.

    Returns Executive Summary metrics, Why-difference table, conclusions,
    and explorer filter SQL. ``compare_to`` selects Overall dataset vs counterpart
    (Worst/Best) for the Why table.
    """
    empty: dict[str, Any] = {
        "available": False,
        "error": None,
        "program_id": program_id,
        "program": None,
        "cohorts": [],
    }
    meta = next((p for p in _programs_catalog() if p["id"] == program_id), None)
    if meta is None:
        return {**empty, "error": f"Unknown program: {program_id}"}

    lim = int(limit) if limit else DEFAULT_LIMIT
    if lim not in LIMIT_CHOICES:
        # allow custom positive ints
        lim = max(1, lim)

    specs = _program_specs(lim)
    spec = specs.get(program_id)
    if spec is None:
        return {**empty, "error": f"No runner for program: {program_id}"}

    resolved_compare = _resolve_compare_to(program_id, compare_to, meta)
    why_a, why_b = _why_column_labels(resolved_compare, program_id)
    # Friendly column names for top success vs worst
    if program_id == "top_success" and resolved_compare == COMPARE_DATASET:
        why_a, why_b = "Success", "Overall"
    elif program_id == "top_success" and resolved_compare == COMPARE_WORST:
        why_a, why_b = "Success", "Worst"
    elif program_id == "worst_predictions" and resolved_compare == COMPARE_DATASET:
        why_a, why_b = "Worst", "Overall"
    elif program_id == "worst_predictions" and resolved_compare == COMPARE_BEST:
        why_a, why_b = "Worst", "Best"

    try:
        with ModelLabStore(db_path) as store:
            store.ensure_prediction_schema()
            cols = store._prediction_table_columns()
            if store.prediction_row_count() <= 0:
                return {**empty, "error": "Prediction dataset is empty.", "program": meta}
            pairs = _load_feature_map(store, cols)

            raw_cohorts: list[dict[str, Any]]
            if spec.get("cohorts"):
                raw_cohorts = list(spec["cohorts"])
            else:
                raw_cohorts = [
                    {
                        "key": "primary",
                        "title": str(meta["title"]),
                        "where_sql": str(spec["where_sql"]),
                        "where_args": list(spec.get("where_args") or []),
                        "label": str(spec["label"]),
                    }
                ]

            cohorts_out: list[dict[str, Any]] = []
            for i, c in enumerate(raw_cohorts):
                compare_means, compare_metrics, compare_label = _counterpart_means_for_compare(
                    store,
                    program_id=program_id,
                    spec=spec,
                    compare_to=resolved_compare,
                    feature_pairs=pairs,
                    cohort_index=i,
                    cohorts_raw=raw_cohorts,
                )
                # For multi-cohort + dataset compare, use dataset; for counterpart use other
                col_a, col_b = why_a, why_b
                if program_id == "best_vs_worst" and resolved_compare == COMPARE_COUNTERPART:
                    col_a = "Best" if i == 0 else "Worst"
                    col_b = "Worst" if i == 0 else "Best"
                elif program_id == "premium_bands":
                    col_a = str(c.get("title") or "Band")
                    col_b = compare_label if resolved_compare == COMPARE_DATASET else "Other band"

                cohorts_out.append(
                    _cohort_payload(
                        store,
                        title=str(c.get("title") or meta["title"]),
                        label=str(c.get("label") or c.get("title") or spec.get("label") or ""),
                        where_sql=str(c["where_sql"]),
                        where_args=list(c.get("where_args") or []),
                        feature_pairs=pairs,
                        compare_means=compare_means,
                        compare_metrics=compare_metrics,
                        compare_label=compare_label,
                        compare_to=resolved_compare,
                        program_id=program_id,
                        why_col_a=col_a,
                        why_col_b=col_b,
                    )
                )

        return {
            "available": True,
            "error": None,
            "program_id": program_id,
            "program": meta,
            "hypothesis": meta.get("hypothesis"),
            "answer": meta.get("answer"),
            "limit": lim,
            "compare_to": resolved_compare,
            "compare_options": list(meta.get("compare_options") or [COMPARE_DATASET]),
            "label": spec.get("label"),
            "cohorts": cohorts_out,
            "primary": cohorts_out[0] if cohorts_out else None,
        }
    except Exception as exc:
        return {**empty, "error": str(exc), "program": meta}
