"""Feature Research laboratory — per-feature analysis on prediction rows."""

from __future__ import annotations

import math
import sqlite3
import statistics
from typing import Any

from .prediction_feature_store import PredictionFeatureStore
from .research_dashboard import (
    _bucket_metrics,
    _f,
    _load_feature_map,
    _mean,
    _median,
    _percentile,
    _rate,
)
from .store import ModelLabStore


def _snapshot_feature_ranks(info: Any) -> dict[str, int]:
    """RFE final ranks copied into the lab at creation (prediction-lab metadata)."""
    rank_order: dict[str, int] = {}
    snap = getattr(info, "feature_ranking_snapshot", None) if info is not None else None
    if not isinstance(snap, dict):
        return rank_order
    rows = snap.get("rows") or []
    if not isinstance(rows, list):
        return rank_order
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            continue
        name = str(row.get("feature") or "").strip()
        if not name:
            continue
        fr = row.get("final_rank")
        try:
            rank_order[name] = int(fr) if fr is not None else i + 1
        except (TypeError, ValueError):
            rank_order[name] = i + 1
    return rank_order


def _compute_feature_tertiles(
    db_path: str,
    *,
    data_dir: str | None = None,
) -> list[dict[str, Any]]:
    """
    Per-feature low/mid/high tertile outcome stats.

    Separate from Research Dashboard rebuild — Feature Research owns this workload.
    """
    with ModelLabStore(db_path) as store:
        store.ensure_prediction_schema()
        cols = store._prediction_table_columns()
        feat_pairs = _load_feature_map(store, cols)
        if not feat_pairs:
            return []

        base_cols = [
            c
            for c in (
                "direction_correct",
                "target_reached",
                "time_to_target",
                "dd_before_target",
                "absolute_error",
                "premium_error_pct",
            )
            if c in cols
        ]
        access = PredictionFeatureStore.from_store(store, data_dir=data_dir)
        records: list[dict[str, Any]] = []
        if access.is_referenced():
            try:
                records = access.fetch_rows(
                    outcome_cols=base_cols,
                    feature_names=[n for n, _ in feat_pairs],
                )
            except (FileNotFoundError, sqlite3.Error, OSError):
                return []
        else:
            feat_cols = [c for _, c in feat_pairs]
            select_cols = base_cols + [c for c in feat_cols if c not in base_cols]
            if not select_cols:
                return []
            col_sql = ", ".join(f'"{c}"' for c in select_cols)
            raw_rows = store.conn.execute(
                f"SELECT {col_sql} FROM prediction_dataset"
            ).fetchall()
            idx = {c: i for i, c in enumerate(select_cols)}
            for row in raw_rows:
                records.append({c: row[idx[c]] for c in select_cols})

        for rec in records:
            for k in ("direction_correct", "target_reached"):
                if rec.get(k) is not None:
                    try:
                        rec[k] = int(rec[k])
                    except (TypeError, ValueError):
                        rec[k] = None

    features_out: list[dict[str, Any]] = []
    for feat_name, feat_col in feat_pairs:
        pairs: list[tuple[float, dict[str, Any]]] = []
        for r in records:
            v = _f(r.get(feat_name))
            if v is None:
                v = _f(r.get(feat_col))
            if v is None:
                continue
            pairs.append((v, r))
        if len(pairs) < 9:
            continue
        pairs.sort(key=lambda x: x[0])
        vals = [p[0] for p in pairs]
        t1 = _percentile(vals, 100.0 / 3.0)
        t2 = _percentile(vals, 200.0 / 3.0)
        if t1 is None or t2 is None:
            continue
        low_rows: list[dict[str, Any]] = []
        mid_rows: list[dict[str, Any]] = []
        high_rows: list[dict[str, Any]] = []
        for v, r in pairs:
            if v <= t1:
                low_rows.append(r)
            elif v <= t2:
                mid_rows.append(r)
            else:
                high_rows.append(r)
        features_out.append(
            {
                "feature": feat_name,
                "column": feat_col,
                "thresholds": {"low_max": t1, "high_min": t2},
                "low": _bucket_metrics(low_rows),
                "medium": _bucket_metrics(mid_rows),
                "high": _bucket_metrics(high_rows),
                "n": len(pairs),
            }
        )
    return features_out


def list_research_features(
    db_path: str,
    *,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """
    Ranked feature catalog for the Feature Research tab.

    Research Rank prefers larger tertile Hit% spread (High − Low).
    Feature Rank is the RFE final rank from the lab snapshot.
    Does not rebuild the Research Dashboard.
    """
    empty = {"available": False, "error": None, "features": [], "total_predictions": 0}
    try:
        tertile_rows = _compute_feature_tertiles(db_path, data_dir=data_dir)
        tertile_by_name = {
            str(f.get("feature") or ""): f
            for f in tertile_rows
            if f.get("feature")
        }
        with ModelLabStore(db_path) as store:
            store.ensure_prediction_schema()
            cols = store._prediction_table_columns()
            pairs = _load_feature_map(store, cols)
            n = store.prediction_row_count()
            rank_order = _snapshot_feature_ranks(store.read_info())

        catalog: list[dict[str, Any]] = []
        for name, col in pairs:
            tert = tertile_by_name.get(name) or {}
            lo = (tert.get("low") or {}).get("hit_rate")
            hi = (tert.get("high") or {}).get("hit_rate")
            spread = None
            if lo is not None and hi is not None:
                spread = abs(float(hi) - float(lo))
            feature_rank = rank_order.get(name)
            catalog.append(
                {
                    "feature": name,
                    "column": col,
                    "hit_spread": spread,
                    "feature_rank": feature_rank,
                    "model_rank": feature_rank,
                    # Alias kept for older callers
                    "ranking_rank": feature_rank,
                    "tertile": tert or None,
                }
            )

        # Sort: larger Hit% spread first, then Feature Rank, then name
        catalog.sort(
            key=lambda r: (
                -(float(r["hit_spread"]) if r.get("hit_spread") is not None else -1.0),
                int(r["feature_rank"]) if r.get("feature_rank") is not None else 10_000,
                str(r["feature"]).lower(),
            )
        )
        for i, row in enumerate(catalog, start=1):
            row["research_rank"] = i
            row["rank"] = i  # alias: Research Rank in the list UI

        return {
            "available": n > 0 and bool(catalog),
            "error": None,
            "features": catalog,
            "total_predictions": n,
        }
    except Exception as exc:
        return {**empty, "error": str(exc)}


def _std(vals: list[float]) -> float | None:
    if len(vals) < 2:
        return 0.0 if vals else None
    return float(statistics.pstdev(vals))


def _histogram(vals: list[float], *, bins: int = 24) -> list[dict[str, Any]]:
    if not vals:
        return []
    lo = min(vals)
    hi = max(vals)
    if hi <= lo:
        return [{"lo": lo, "hi": hi, "count": len(vals), "center": lo}]
    n_bins = max(5, min(int(bins), 40))
    width = (hi - lo) / n_bins
    counts = [0] * n_bins
    for v in vals:
        idx = int((v - lo) / width)
        if idx >= n_bins:
            idx = n_bins - 1
        if idx < 0:
            idx = 0
        counts[idx] += 1
    out: list[dict[str, Any]] = []
    for i, c in enumerate(counts):
        a = lo + i * width
        b = lo + (i + 1) * width
        out.append({"lo": a, "hi": b, "count": c, "center": (a + b) / 2.0})
    return out


def _equal_frequency_bins(
    pairs: list[tuple[float, dict[str, Any]]],
    *,
    n_bins: int = 10,
) -> list[dict[str, Any]]:
    """Equal-count value bins with bucket outcome metrics."""
    if not pairs:
        return []
    pairs = sorted(pairs, key=lambda x: x[0])
    n = len(pairs)
    bins_n = max(3, min(int(n_bins), n // 5 if n >= 15 else max(2, n // 3)))
    bins_n = max(2, min(bins_n, n))
    out: list[dict[str, Any]] = []
    for i in range(bins_n):
        start = int(round(i * n / bins_n))
        end = int(round((i + 1) * n / bins_n))
        if end <= start:
            continue
        chunk = pairs[start:end]
        rows = [r for _, r in chunk]
        vals = [v for v, _ in chunk]
        m = _bucket_metrics(rows)
        out.append(
            {
                "lo": vals[0],
                "hi": vals[-1],
                **m,
            }
        )
    return out


def _pick_best_worst(
    bins: list[dict[str, Any]],
    *,
    min_rows: int,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    eligible = [b for b in bins if int(b.get("rows") or 0) >= min_rows and b.get("hit_rate") is not None]
    if not eligible:
        return None, None
    best = max(eligible, key=lambda b: float(b["hit_rate"]))
    worst = min(eligible, key=lambda b: float(b["hit_rate"]))
    return best, worst


def analyze_feature(db_path: str, feature_name: str) -> dict[str, Any]:
    """
    Full Feature Analysis laboratory payload for one selected feature.
    """
    empty: dict[str, Any] = {
        "available": False,
        "error": None,
        "feature": feature_name,
        "column": None,
    }
    name = str(feature_name or "").strip()
    if not name:
        return {**empty, "error": "feature required"}

    try:
        with ModelLabStore(db_path) as store:
            store.ensure_prediction_schema()
            cols = store._prediction_table_columns()
            pairs = _load_feature_map(store, cols)
            col = None
            for feat, sf in pairs:
                if feat == name:
                    col = sf
                    break
            if col is None:
                # Allow raw column name
                if name in cols:
                    col = name
                elif f"sf_{name}" in cols:
                    col = f"sf_{name}"
            if not col and not any(f == name for f, _ in pairs):
                return {**empty, "error": f"Feature column not found: {name}"}

            outcome_need = [
                c
                for c in (
                    "target_reached",
                    "direction_correct",
                    "absolute_error",
                    "dd_before_target",
                    "time_to_target",
                    "premium_error_pct",
                    "maximum_drawdown",
                    "maximum_profit",
                )
                if c in cols
            ]
            from .prediction_feature_store import PredictionFeatureStore

            access = PredictionFeatureStore.from_store(store)
            if access.is_referenced() or col not in cols:
                raw_dicts = access.fetch_rows(
                    outcome_cols=outcome_need,
                    feature_names=[name],
                    require_feature_nonnull=name,
                )
                total_rows = int(store.prediction_row_count() or 0)
                feature_rank = _snapshot_feature_ranks(store.read_info()).get(name)
                records = []
                values = []
                pairs_vr = []
                for rec in raw_dicts:
                    v = _f(rec.get(name))
                    if v is None:
                        continue
                    for k in ("direction_correct", "target_reached"):
                        if rec.get(k) is not None:
                            try:
                                rec[k] = int(rec[k])
                            except (TypeError, ValueError):
                                rec[k] = None
                    # Keep feature under both name and physical for downstream
                    rec[col or name] = v
                    records.append(rec)
                    values.append(v)
                    pairs_vr.append((v, rec))
            else:
                need = [c for c in (col, *outcome_need) if c in cols]
                sql_cols = ", ".join(f'"{c}"' for c in need)
                sql = (
                    f'SELECT {sql_cols} FROM prediction_dataset '
                    f'WHERE "{col}" IS NOT NULL'
                )
                raw = store.conn.execute(sql).fetchall()
                idx = {c: i for i, c in enumerate(need)}
                total_rows = int(store.prediction_row_count() or 0)
                feature_rank = _snapshot_feature_ranks(store.read_info()).get(name)

                records = []
                values = []
                pairs_vr = []
                for row in raw:
                    v = _f(row[idx[col]])
                    if v is None:
                        continue
                    rec: dict[str, Any] = {}
                    for c in need:
                        rec[c] = row[idx[c]]
                    for k in ("direction_correct", "target_reached"):
                        if rec.get(k) is not None:
                            try:
                                rec[k] = int(rec[k])
                            except (TypeError, ValueError):
                                rec[k] = None
                    records.append(rec)
                    values.append(v)
                    pairs_vr.append((v, rec))

        if len(values) < 9:
            missing = max(0, total_rows - len(values))
            return {
                **empty,
                "available": False,
                "error": f"Not enough non-null values ({len(values)})",
                "column": col,
                "n": len(values),
                "rows": len(values),
                "rows_analyzed": len(values),
                "total_rows": total_rows,
                "missing_values": missing,
                "coverage": (len(values) / total_rows) if total_rows else None,
                "feature_rank": feature_rank,
                "model_rank": feature_rank,
                "research_rank": None,
            }

        values_sorted = sorted(values)
        stats = {
            "n": len(values),
            "minimum": values_sorted[0],
            "maximum": values_sorted[-1],
            "average": _mean(values),
            "median": _median(values),
            "std_dev": _std(values),
        }

        t1 = _percentile(values_sorted, 100.0 / 3.0)
        t2 = _percentile(values_sorted, 200.0 / 3.0)
        low_rows: list[dict[str, Any]] = []
        mid_rows: list[dict[str, Any]] = []
        high_rows: list[dict[str, Any]] = []
        for v, r in pairs_vr:
            if t1 is not None and v <= t1:
                low_rows.append(r)
            elif t2 is not None and v <= t2:
                mid_rows.append(r)
            else:
                high_rows.append(r)

        tertiles = {
            "thresholds": {"low_max": t1, "high_min": t2},
            "low": _bucket_metrics(low_rows),
            "medium": _bucket_metrics(mid_rows),
            "high": _bucket_metrics(high_rows),
        }

        hist = _histogram(values_sorted, bins=24)
        bins = _equal_frequency_bins(pairs_vr, n_bins=10)
        min_rows = max(20, int(0.05 * len(values)))
        best, worst = _pick_best_worst(bins, min_rows=min_rows)

        # Mode-ish peak from densest histogram bin
        peak = max(hist, key=lambda b: int(b["count"])) if hist else None

        compare = {
            "low": tertiles["low"],
            "high": tertiles["high"],
            "delta_hit_rate": None,
            "delta_dir": None,
            "delta_dd": None,
        }
        if tertiles["low"].get("hit_rate") is not None and tertiles["high"].get("hit_rate") is not None:
            compare["delta_hit_rate"] = float(tertiles["high"]["hit_rate"]) - float(
                tertiles["low"]["hit_rate"]
            )
        if (
            tertiles["low"].get("direction_accuracy") is not None
            and tertiles["high"].get("direction_accuracy") is not None
        ):
            compare["delta_dir"] = float(tertiles["high"]["direction_accuracy"]) - float(
                tertiles["low"]["direction_accuracy"]
            )
        if (
            tertiles["low"].get("avg_dd_before_target") is not None
            and tertiles["high"].get("avg_dd_before_target") is not None
        ):
            compare["delta_dd"] = float(tertiles["high"]["avg_dd_before_target"]) - float(
                tertiles["low"]["avg_dd_before_target"]
            )
        if tertiles["low"].get("mae") is not None and tertiles["high"].get("mae") is not None:
            compare["delta_mae"] = float(tertiles["high"]["mae"]) - float(tertiles["low"]["mae"])

        analyzed_n = len(values)
        missing_values = max(0, int(total_rows) - analyzed_n)
        coverage = (analyzed_n / total_rows) if total_rows else None
        research_rank = None
        try:
            for feat in list_research_features(db_path).get("features") or []:
                if feat.get("feature") == name:
                    research_rank = feat.get("research_rank") or feat.get("rank")
                    if feature_rank is None:
                        feature_rank = feat.get("feature_rank") or feat.get("model_rank")
                    break
        except Exception:
            research_rank = None

        def _with_coverage(bucket: dict[str, Any] | None) -> dict[str, Any] | None:
            if not bucket:
                return bucket
            rows_n = bucket.get("rows")
            try:
                rn = int(rows_n) if rows_n is not None else None
            except (TypeError, ValueError):
                rn = None
            cov = (rn / analyzed_n) if rn is not None and analyzed_n else None
            return {**bucket, "coverage": cov}

        best = _with_coverage(best)
        worst = _with_coverage(worst)
        compare = {
            **compare,
            "low": _with_coverage(compare.get("low")) or compare.get("low"),
            "high": _with_coverage(compare.get("high")) or compare.get("high"),
        }
        conclusion = build_research_conclusion(name, compare)

        return {
            "available": True,
            "error": None,
            "feature": name,
            "column": col,
            "feature_rank": feature_rank,
            "model_rank": feature_rank,
            "research_rank": research_rank,
            "rows": analyzed_n,
            "rows_analyzed": analyzed_n,
            "total_rows": total_rows,
            "missing_values": missing_values,
            "coverage": coverage,
            "stats": stats,
            "tertiles": tertiles,
            "histogram": hist,
            "peak_bin": peak,
            "bins": bins,
            "best_range": best,
            "worst_range": worst,
            "compare": compare,
            "conclusion": conclusion,
            "filters": {
                "low": {"column": col, "op": "lte", "value": t1},
                "medium": {"column": col, "op": "between", "lo": t1, "hi": t2},
                "high": {"column": col, "op": "gt", "value": t2},
                "best": (
                    {
                        "column": col,
                        "op": "between_inclusive",
                        "lo": best.get("lo"),
                        "hi": best.get("hi"),
                    }
                    if best
                    else None
                ),
                "worst": (
                    {
                        "column": col,
                        "op": "between_inclusive",
                        "lo": worst.get("lo"),
                        "hi": worst.get("hi"),
                    }
                    if worst
                    else None
                ),
            },
        }
    except Exception as exc:
        return {**empty, "error": str(exc)}


def build_research_conclusion(
    feature_name: str,
    compare: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Narrative Research Conclusion from Low vs High tertile compare.

    Positive bullets (✓) = outcomes that improve; warnings (⚠) = trade-offs.
    Score is 1–5 stars, derived from Hit / MAE / DD gains vs Direction cost.
    """
    empty = {
        "bullets": [],
        "score": None,
        "score_stars": "—",
        "preferred": None,
        "text": "Not enough Low vs High contrast to form a conclusion.",
    }
    cmp = compare or {}
    lo = cmp.get("low") or {}
    hi = cmp.get("high") or {}
    name = str(feature_name or "feature").strip() or "feature"

    lo_hit = lo.get("hit_rate")
    hi_hit = hi.get("hit_rate")
    if lo_hit is None and hi_hit is None:
        return empty

    # Prefer the tertile with better hit rate; MAE then as tie-breaker.
    preferred = "high"
    if lo_hit is not None and hi_hit is not None:
        if float(hi_hit) > float(lo_hit) + 1e-12:
            preferred = "high"
        elif float(lo_hit) > float(hi_hit) + 1e-12:
            preferred = "low"
        else:
            lo_mae, hi_mae = lo.get("mae"), hi.get("mae")
            if lo_mae is not None and hi_mae is not None and float(lo_mae) < float(hi_mae):
                preferred = "low"
            elif lo_mae is not None and hi_mae is not None and float(hi_mae) < float(lo_mae):
                preferred = "high"
    elif hi_hit is not None:
        preferred = "high"
    else:
        preferred = "low"

    pref_band = hi if preferred == "high" else lo
    other_band = lo if preferred == "high" else hi
    side_word = "Higher" if preferred == "high" else "Lower"
    values_word = "High values" if preferred == "high" else "Low values"

    def _pp(delta: float) -> str:
        return f"{100.0 * delta:+.1f}%"

    def _num(v: Any) -> str:
        try:
            return f"{float(v):.2f}"
        except (TypeError, ValueError):
            return str(v)

    bullets: list[dict[str, Any]] = []
    good = 0
    bad = 0

    d_hit = cmp.get("delta_hit_rate")
    if d_hit is None and lo_hit is not None and hi_hit is not None:
        d_hit = float(hi_hit) - float(lo_hit)
    if d_hit is not None:
        hit_effect = float(d_hit) if preferred == "high" else -float(d_hit)
        if abs(hit_effect) >= 0.005:
            if hit_effect > 0:
                bullets.append(
                    {
                        "kind": "good",
                        "text": (
                            f"{side_word} {name} improves Hit Rate "
                            f"({_pp(hit_effect)})."
                        ),
                    }
                )
                good += 1 if hit_effect >= 0.01 else 0
                if hit_effect >= 0.03:
                    good += 1
            else:
                bullets.append(
                    {
                        "kind": "warn",
                        "text": (
                            f"{side_word} {name} lowers Hit Rate "
                            f"({_pp(hit_effect)})."
                        ),
                    }
                )
                bad += 1

    lo_mae, hi_mae = lo.get("mae"), hi.get("mae")
    if lo_mae is not None and hi_mae is not None:
        pref_mae = float(pref_band["mae"]) if pref_band.get("mae") is not None else None
        other_mae = float(other_band["mae"]) if other_band.get("mae") is not None else None
        if pref_mae is not None and other_mae is not None and other_mae > pref_mae + 1e-9:
            bullets.append(
                {
                    "kind": "good",
                    "text": (
                        f"{values_word} reduce MAE from {_num(other_mae)} "
                        f"to {_num(pref_mae)}."
                    ),
                }
            )
            good += 1
            if (other_mae - pref_mae) / max(abs(other_mae), 1e-9) >= 0.25:
                good += 1
        elif pref_mae is not None and other_mae is not None and pref_mae > other_mae + 1e-9:
            bullets.append(
                {
                    "kind": "warn",
                    "text": (
                        f"{values_word} raise MAE from {_num(other_mae)} "
                        f"to {_num(pref_mae)}."
                    ),
                }
            )
            bad += 1

    if lo.get("avg_dd_before_target") is not None and hi.get("avg_dd_before_target") is not None:
        pref_dd = pref_band.get("avg_dd_before_target")
        other_dd = other_band.get("avg_dd_before_target")
        if pref_dd is not None and other_dd is not None:
            pref_dd_f, other_dd_f = float(pref_dd), float(other_dd)
            if other_dd_f > pref_dd_f + 1e-9:
                bullets.append(
                    {
                        "kind": "good",
                        "text": (
                            f"{values_word} reduce Drawdown from {_num(other_dd_f)} "
                            f"to {_num(pref_dd_f)}."
                        ),
                    }
                )
                good += 1
            elif pref_dd_f > other_dd_f + 1e-9:
                bullets.append(
                    {
                        "kind": "warn",
                        "text": (
                            f"{values_word} increase Drawdown from {_num(other_dd_f)} "
                            f"to {_num(pref_dd_f)}."
                        ),
                    }
                )
                bad += 1

    d_dir = cmp.get("delta_dir")
    lo_dir, hi_dir = lo.get("direction_accuracy"), hi.get("direction_accuracy")
    if d_dir is None and lo_dir is not None and hi_dir is not None:
        d_dir = float(hi_dir) - float(lo_dir)
    if d_dir is not None:
        dir_effect = float(d_dir) if preferred == "high" else -float(d_dir)
        if abs(dir_effect) >= 0.01:
            if dir_effect > 0:
                bullets.append(
                    {
                        "kind": "good",
                        "text": (
                            f"Direction accuracy improves by "
                            f"{abs(100.0 * dir_effect):.1f}%."
                        ),
                    }
                )
                good += 1
            else:
                bullets.append(
                    {
                        "kind": "warn",
                        "text": (
                            f"Direction accuracy decreases by "
                            f"{abs(100.0 * dir_effect):.1f}%."
                        ),
                    }
                )
                bad += 1

    if not bullets:
        return empty

    score = 3 + min(2, good) - min(2, bad)
    score = max(1, min(5, score))
    stars = "★" * score + "☆" * (5 - score)

    lines = [f"{'✓' if b.get('kind') == 'good' else '⚠'} {b.get('text')}" for b in bullets]
    lines.append("")
    lines.append("Overall Research Score:")
    lines.append(stars)

    return {
        "bullets": bullets,
        "score": score,
        "score_stars": stars,
        "preferred": preferred,
        "text": "\n".join(lines),
    }


def filter_label_from_spec(
    spec: dict[str, Any] | None,
    *,
    feature: str | None = None,
) -> str:
    """Human-readable Applied Filter line, e.g. ``feat ∈ [214.8, 267.5]``."""
    if not spec:
        return ""
    col = str(spec.get("column") or "").strip()
    name = str(feature or "").strip()
    if not name:
        name = col[3:] if col.startswith("sf_") else col
    if not name:
        return ""

    def _n(v: Any) -> str:
        if v is None:
            return "—"
        try:
            f = float(v)
            if abs(f) >= 100 or (abs(f) >= 1 and abs(f - round(f)) < 1e-9):
                return f"{f:.4g}"
            return f"{f:.4g}"
        except (TypeError, ValueError):
            return str(v)

    op = str(spec.get("op") or "")
    if op == "lte":
        return f"{name} ≤ {_n(spec.get('value'))}"
    if op == "gt":
        return f"{name} > {_n(spec.get('value'))}"
    if op == "between":
        return f"{name} ∈ ({_n(spec.get('lo'))}, {_n(spec.get('hi'))}]"
    if op == "between_inclusive":
        return f"{name} ∈ [{_n(spec.get('lo'))}, {_n(spec.get('hi'))}]"
    return name


def filter_sql_from_spec(
    spec: dict[str, Any] | None,
    *,
    referenced: bool = False,
) -> tuple[str, list[Any]]:
    """Translate a Feature Analysis filter into (where_sql, args)."""
    if not spec:
        return "", []
    col = str(spec.get("column") or "").strip()
    if not col or not col.replace("_", "").isalnum():
        return "", []
    op = str(spec.get("op") or "")
    quoted = f'm."{col}"' if referenced else f'"{col}"'
    if op == "lte":
        return f"{quoted} IS NOT NULL AND {quoted} <= ?", [spec.get("value")]
    if op == "gt":
        return f"{quoted} IS NOT NULL AND {quoted} > ?", [spec.get("value")]
    if op == "between":
        # medium: low_max < x <= high_min (match tertile split)
        return (
            f"{quoted} IS NOT NULL AND {quoted} > ? AND {quoted} <= ?",
            [spec.get("lo"), spec.get("hi")],
        )
    if op == "between_inclusive":
        return (
            f"{quoted} IS NOT NULL AND {quoted} >= ? AND {quoted} <= ?",
            [spec.get("lo"), spec.get("hi")],
        )
    return "", []
