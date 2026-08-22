"""Correlation analysis engine for Phase 2 Analysis Lab.

Computes pairwise Pearson correlations on an analysis parquet, persists pairs
into analysis.db, builds threshold clusters, and serves Summary / Top Pairs /
Matrix / Clusters views.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Any, Callable, Sequence

from .analysis_lab_store import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    _AnalysisDb,
    _now_iso,
    resolve_parquet_path,
    set_module_status,
)
from .analysis_feature_roles import is_predictor, predictor_columns

ProgressCb = Callable[[dict[str, Any]], None]

logger = logging.getLogger(__name__)

# Cap rows for Pearson on large analysis datasets. Full-frame loads of multi-million
# row / 300+ column parquets routinely OOM (same class of failure as post-transform
# No-Null). 150k is plenty for stable pairwise correlations.
DEFAULT_CORR_MAX_ROWS = 150_000


def _correlation_read_columns(
    parquet_path: str,
    sidecar: dict[str, Any] | None,
) -> list[str] | None:
    """Predictor + day columns for Engine column prune (None = read all)."""
    try:
        import pyarrow.parquet as pq
    except Exception:
        return None
    try:
        names = [str(n) for n in pq.read_schema(parquet_path).names]
    except Exception:
        return None
    if not names:
        return None
    preds = predictor_columns(names, sidecar=sidecar or {})
    day_cols = [c for c in ("trading_day", "date", "session_day") if c in names]
    cols = list(dict.fromkeys([*day_cols, *preds]))
    return cols or None


def _load_parquet_for_correlation_legacy(
    parquet_path: str,
    *,
    max_rows: int | None,
    progress: Callable[[float, str], None] | None = None,
) -> Any:
    """Legacy pyarrow row-group / pandas load (Phase 3A Pandas fallback)."""
    import pandas as pd

    try:
        import pyarrow.parquet as pq
    except Exception:
        df = pd.read_parquet(parquet_path)
        if max_rows is not None and len(df) > int(max_rows):
            if progress:
                progress(0.08, f"Sampling {int(max_rows):,} of {len(df):,} rows…")
            df = df.sample(n=int(max_rows), random_state=42)
        return df

    pf = pq.ParquetFile(parquet_path)
    total = int(pf.metadata.num_rows) if pf.metadata is not None else 0
    if progress:
        progress(0.04, f"Opening parquet · {total:,} rows…")

    if max_rows is None or total <= int(max_rows):
        if progress:
            progress(0.08, f"Loading {total:,} rows…")
        return pf.read().to_pandas()

    target = int(max_rows)
    n_rg = int(pf.num_row_groups)
    rng = __import__("random").Random(42)
    order = list(range(n_rg))
    rng.shuffle(order)

    tables: list[Any] = []
    rows = 0
    for i, rg in enumerate(order):
        t = pf.read_row_group(rg)
        tables.append(t)
        rows += int(t.num_rows)
        if progress and (i == 0 or (i + 1) % 4 == 0 or rows >= target):
            progress(
                0.04 + 0.06 * min(1.0, rows / max(target, 1)),
                f"Sampling row groups · {min(rows, target):,}/{target:,}…",
            )
        if rows >= target:
            break

    import pyarrow as pa

    table = pa.concat_tables(tables) if tables else pf.read()
    df = table.to_pandas()
    if len(df) > target:
        df = df.sample(n=target, random_state=42)
    if progress:
        progress(0.10, f"Using {len(df):,} of {total:,} rows for correlation…")
    return df


def _load_parquet_for_correlation_engine(
    parquet_path: str,
    *,
    max_rows: int | None,
    columns: Sequence[str] | None,
    progress: Callable[[float, str], None] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Dataset Engine retrieval for Correlation (Phase 3A)."""
    from chain_replay_ml.dataset_engine import SampleSpec, query_dataset
    from chain_replay_ml.training.load_backend import measure_span

    if progress:
        progress(0.04, "Loading via Dataset Engine…")

    sample = None
    if max_rows is not None:
        sample = SampleSpec(max_rows=int(max_rows), seed=42)

    def _load():
        return query_dataset(
            parquet_path,
            columns=list(columns) if columns else None,
            sample=sample,
            parquet_path=parquet_path,
        )

    result, span = measure_span(_load)
    from chain_replay_ml.frame_backend import arrow_table_to_pandas

    df, frame_bridge = arrow_table_to_pandas(result.table, via_polars=True)
    stats = result.stats
    load_metrics: dict[str, Any] = {
        "backend": "dataset_engine",
        "frame_bridge": frame_bridge,
        **span,
        "rows_returned": int(stats.rows_returned or len(df)),
        "columns_returned": int(len(df.columns)),
        "partitions_scanned": stats.partitions_scanned,
        "partitions_pruned": stats.partitions_pruned,
        "engine_execution_time_sec": stats.execution_time_sec,
        "engine_extra": dict(stats.extra or {}),
        "engine_fallback": False,
    }
    if progress:
        progress(0.10, f"Engine loaded {len(df):,} rows × {len(df.columns)} cols…")
    return df, load_metrics


def _load_parquet_for_correlation(
    parquet_path: str,
    *,
    max_rows: int | None,
    progress: Callable[[float, str], None] | None = None,
    sidecar: dict[str, Any] | None = None,
    force_backend: str | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Load analysis parquet for Correlation (Engine preferred, Pandas fallback).

    Phase 3A: replace retrieval only. Pearson math stays unchanged.
    Env ``ARUNEO_DATASET_ENGINE`` matches Model Builder (``auto`` / ``on`` / ``off``).
    """
    from chain_replay_ml.training.load_backend import (
        measure_span,
        resolve_training_load_backend,
    )

    backend = force_backend or resolve_training_load_backend()
    columns = _correlation_read_columns(parquet_path, sidecar)

    if backend == "dataset_engine":
        try:
            df, load_metrics = _load_parquet_for_correlation_engine(
                parquet_path,
                max_rows=max_rows,
                columns=columns,
                progress=progress,
            )
            logger.info(
                "dataset_load consumer=correlation backend=%s load_time_sec=%s "
                "peak_rss_mb=%s rows=%s cols=%s fallback=%s",
                load_metrics.get("backend"),
                load_metrics.get("load_time_sec"),
                load_metrics.get("peak_rss_mb"),
                load_metrics.get("rows_returned"),
                load_metrics.get("columns_returned"),
                False,
            )
            return df, load_metrics
        except Exception as exc:
            logger.warning(
                "dataset_load consumer=correlation engine fallback reason=%s",
                exc,
            )

    def _legacy():
        return _load_parquet_for_correlation_legacy(
            parquet_path, max_rows=max_rows, progress=progress
        )

    df, span = measure_span(_legacy)
    load_metrics = {
        "backend": "pandas",
        **span,
        "rows_returned": int(len(df)),
        "columns_returned": int(len(df.columns)),
        "partitions_scanned": None,
        "engine_fallback": backend == "dataset_engine",
    }
    if load_metrics["engine_fallback"]:
        load_metrics["engine_fallback_reason"] = "engine_load_failed"
    logger.info(
        "dataset_load consumer=correlation backend=%s load_time_sec=%s "
        "peak_rss_mb=%s rows=%s cols=%s fallback=%s",
        load_metrics.get("backend"),
        load_metrics.get("load_time_sec"),
        load_metrics.get("peak_rss_mb"),
        load_metrics.get("rows_returned"),
        load_metrics.get("columns_returned"),
        bool(load_metrics.get("engine_fallback")),
    )
    return df, load_metrics


def _family_label(name: str) -> str:
    n = str(name or "").lower()
    if any(tok in n for tok in ("iv_", "_iv", "current_iv", "vega", "volga", "vanna", "implied", "svi_", "sabr_", "surface_", "vrp_")):
        return "IV"
    if any(tok in n for tok in ("oi", "open_interest")):
        return "OI"
    if any(tok in n for tok in ("delta", "gamma", "theta", "charm", "speed", "rho", "color", "zomma", "ultima")):
        return "Greeks"
    if any(tok in n for tok in ("volume", "vwap", "turnover")):
        return "Volume"
    if any(tok in n for tok in ("_lag_", "time_to", "session", "minute")):
        return "Time"
    if any(tok in n for tok in ("zscore", "momentum", "_change_", "_return_", "slope")):
        return "Momentum"
    if any(
        tok in n
        for tok in ("spot", "ltp", "ema", "premium", "price", "fut", "underlying", "straddle")
    ):
        return "Price"
    return "Other"


def _is_feature_col(name: str) -> bool:
    """Backward-compatible name: scorable predictors only."""
    return is_predictor(name)


def _zero_variance_mask(series: Any) -> bool:
    """True when a series has no usable variance for Pearson correlation."""
    import pandas as pd

    s = pd.to_numeric(series, errors="coerce")
    if int(s.notna().sum()) < 2:
        return True
    if int(s.nunique(dropna=True)) <= 1:
        return True
    try:
        return float(s.std(ddof=0) or 0.0) == 0.0
    except Exception:
        return True


def _yield_gil() -> None:
    """Briefly release the GIL so Tk / other threads can schedule."""
    time.sleep(0)


def _constant_feature_columns(num: Any) -> tuple[list[str], list[str]]:
    """Split numeric frame into (keep, excluded_constant) with vectorized screens.

    Matches ``_zero_variance_mask`` semantics per column without a Python loop
    over every feature (avoids multi-second GIL holds that freeze the Studio UI).
    """
    import pandas as pd

    if num is None or getattr(num, "empty", True):
        return [], []
    cols = [str(c) for c in num.columns]
    if not cols:
        return [], []
    notna = num.notna().sum(axis=0)
    nunique = num.nunique(dropna=True)
    try:
        std = num.std(ddof=0)
    except Exception:
        std = pd.Series(0.0, index=num.columns)
    keep: list[str] = []
    excluded: list[str] = []
    for c in cols:
        try:
            if int(notna.get(c, 0) or 0) < 2:
                excluded.append(c)
                continue
            if int(nunique.get(c, 0) or 0) <= 1:
                excluded.append(c)
                continue
            try:
                s_val = float(std.get(c))
            except (TypeError, ValueError):
                s_val = float("nan")
            if (s_val or 0.0) == 0.0:
                excluded.append(c)
                continue
            keep.append(c)
        except Exception:
            excluded.append(c)
    return keep, excluded


def _day_constant_feature_columns(
    num: Any,
    keep: Sequence[str],
    days: Any,
) -> tuple[list[str], list[str]]:
    """Drop columns that are constant inside every trading day (vectorized).

    Equivalent to per-column ``groupby(days).nunique(dropna=True).max() <= 1``.
    """
    import pandas as pd

    if not keep:
        return [], []
    frame = num[list(keep)]
    try:
        within = frame.groupby(days, sort=False).nunique(dropna=True)
        max_within = within.max(axis=0)
    except Exception:
        # Fall back to per-column path on exotic group keys.
        keep2: list[str] = []
        excluded: list[str] = []
        for c in keep:
            try:
                within_c = num[c].groupby(days).nunique(dropna=True)
                if len(within_c) > 0 and int(within_c.max()) <= 1:
                    excluded.append(str(c))
                    continue
            except Exception:
                pass
            keep2.append(str(c))
        return keep2, excluded

    keep2 = []
    excluded = []
    for c in keep:
        try:
            mx = max_within.get(c)
            if mx is not None and pd.notna(mx) and int(mx) <= 1:
                excluded.append(str(c))
            else:
                keep2.append(str(c))
        except Exception:
            keep2.append(str(c))
    return keep2, excluded


def compute_correlation_frame(
    parquet_path: str,
    *,
    max_rows: int | None = DEFAULT_CORR_MAX_ROWS,
    progress: ProgressCb | None = None,
    started_at: float | None = None,
    force_backend: str | None = None,
    compute_backend: str | None = None,
) -> tuple[Any, list[str]]:
    """Return (corr DataFrame, feature column names).

    Constant and day-constant features are excluded (Pearson is undefined when
    variance is zero). Day-constants — e.g. ``days_to_expiry`` or session
    ``spot_low`` that only change between trading days — are dropped so they
    cannot form spurious |r|=1.0 pairs on multi-day datasets.

    ``max_rows`` defaults to ``DEFAULT_CORR_MAX_ROWS`` (150_000) so multi-million-row
    analysis datasets do not load the full parquet into RAM. Pass ``max_rows=None``
    only when you intentionally want all rows.

    ``force_backend``: ``dataset_engine`` | ``pandas`` for parity tests; default
    follows ``ARUNEO_DATASET_ENGINE``.

    ``compute_backend``: Pearson engine preference ``auto`` | ``cpu`` | ``gpu``
    (see ``analytics.correlation``). Default ``auto`` → GPU when RAPIDS is
    available, else CPU. CPU remains the safe default on Windows.
    """
    import json
    import os

    import pandas as pd

    t0 = float(started_at) if started_at is not None else time.perf_counter()

    def _tick(frac: float, message: str, **extra: Any) -> None:
        if not progress:
            return
        payload = {
            "frac": max(0.0, min(1.0, float(frac))),
            "elapsed": max(time.perf_counter() - t0, 0.0),
            "message": str(message),
            **extra,
        }
        progress(payload)

    sidecar: dict[str, Any] = {}
    base, _ = os.path.splitext(parquet_path)
    js = base + ".json"
    if os.path.isfile(js):
        try:
            with open(js, encoding="utf-8") as f:
                doc = json.load(f)
            if isinstance(doc, dict):
                sidecar = doc
        except Exception:
            sidecar = {}

    _tick(0.02, "Loading analysis parquet…")
    df, load_metrics = _load_parquet_for_correlation(
        parquet_path,
        max_rows=max_rows,
        progress=lambda frac, msg: _tick(frac, msg),
        sidecar=sidecar,
        force_backend=force_backend,
    )
    _tick(0.12, f"Preparing predictors · {len(df):,} rows…")
    cols = predictor_columns([str(c) for c in df.columns], sidecar=sidecar)
    # Keep numeric only
    num = df[cols].apply(pd.to_numeric, errors="coerce")
    _yield_gil()

    n_cols = max(len(list(num.columns)), 1)
    _tick(0.15, f"Screening constants · {n_cols} features…", done=0, total=n_cols)
    keep, excluded_constant = _constant_feature_columns(num)
    _yield_gil()
    _tick(
        0.27,
        f"Constant screen done · kept {len(keep)} / {n_cols}",
        done=n_cols,
        total=n_cols,
    )
    if not keep:
        raise ValueError("No numeric feature columns available for correlation")

    day_col = next(
        (c for c in ("trading_day", "date", "session_day") if c in df.columns),
        None,
    )
    if day_col is not None and keep:
        # Vectorized day-constant screen (one groupby for all columns).
        days = df[day_col]
        n_keep = max(len(keep), 1)
        _tick(
            0.30,
            f"Screening day-constants · {n_keep} features…",
            done=0,
            total=n_keep,
        )
        keep2, day_excl = _day_constant_feature_columns(num, keep, days)
        excluded_constant.extend(day_excl)
        keep = keep2
        _yield_gil()
        _tick(
            0.45,
            f"Day-constant screen done · kept {len(keep)} / {n_keep}",
            done=n_keep,
            total=n_keep,
        )
        if not keep:
            raise ValueError(
                "No features with within-day variance available for correlation "
                f"(excluded {len(excluded_constant)} constant/day-constant column(s))"
            )

    _tick(
        0.50,
        f"Computing Pearson matrix · {len(keep)} features × {len(df):,} rows…",
        done=0,
        total=len(keep),
    )
    from chain_replay_ml.analytics.correlation import CorrelationEngine

    _yield_gil()
    compute_result = CorrelationEngine(
        preference=compute_backend or "auto"
    ).compute(num[keep], min_periods=2)
    _yield_gil()
    corr = compute_result.matrix
    _tick(
        0.88,
        f"Matrix ready · {len(keep)} features · {compute_result.backend_used.upper()}",
        done=len(keep),
        total=len(keep),
        compute_backend=compute_result.backend_used,
    )
    try:
        corr.attrs["excluded_constant_features"] = list(dict.fromkeys(excluded_constant))
        corr.attrs["demeaned_by"] = None
        corr.attrs["excluded_day_constants"] = bool(day_col)
        corr.attrs["dataset_load"] = dict(load_metrics)
        corr.attrs["compute_backend"] = compute_result.to_meta()
    except Exception:
        pass
    return corr, list(keep)


def compare_correlation_matrices(
    parquet_path: str,
    *,
    max_rows: int | None = None,
    rtol: float = 1e-9,
    atol: float = 1e-8,
) -> dict[str, Any]:
    """Compare Pearson matrices from Engine vs Pandas loads (full-frame preferred).

    Use ``max_rows=None`` (or a cap ≥ file rows) so both backends see the same
    rows. Engine Phase-1 sampling is ``LIMIT``; legacy sampling is row-group
    shuffle — those paths intentionally diverge when the file is larger than
    ``max_rows``.
    """
    import numpy as np

    corr_eng, feats_eng = compute_correlation_frame(
        parquet_path, max_rows=max_rows, force_backend="dataset_engine"
    )
    corr_pd, feats_pd = compute_correlation_frame(
        parquet_path, max_rows=max_rows, force_backend="pandas"
    )
    common = [f for f in feats_eng if f in feats_pd]
    report: dict[str, Any] = {
        "features_engine": feats_eng,
        "features_pandas": feats_pd,
        "features_common": common,
        "features_match": feats_eng == feats_pd,
        "engine_load": dict((getattr(corr_eng, "attrs", None) or {}).get("dataset_load") or {}),
        "pandas_load": dict((getattr(corr_pd, "attrs", None) or {}).get("dataset_load") or {}),
    }
    if not common:
        report["matrices_close"] = False
        report["ok"] = False
        return report
    a = corr_eng.loc[common, common].to_numpy(dtype=float)
    b = corr_pd.loc[common, common].to_numpy(dtype=float)
    close = bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))
    report["matrices_close"] = close
    report["max_abs_diff"] = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
    report["ok"] = bool(report["features_match"] and close)
    return report


def _pair_rows(corr: Any) -> list[tuple[str, str, float]]:
    """Upper-triangle pairs from a Pearson matrix (numpy path; yields GIL)."""
    import numpy as np

    cols = [str(c) for c in corr.columns]
    n = len(cols)
    if n < 2:
        return []
    try:
        mat = np.asarray(corr.to_numpy(dtype=float), dtype=float)
    except Exception:
        mat = None
    out: list[tuple[str, str, float]] = []
    if mat is None or mat.shape != (n, n):
        for i, a in enumerate(cols):
            for j in range(i + 1, n):
                val = corr.iloc[i, j]
                try:
                    r = float(val)
                except (TypeError, ValueError):
                    continue
                if math.isnan(r):
                    continue
                out.append((a, cols[j], r))
            if i and i % 32 == 0:
                _yield_gil()
        return out

    # Vectorized upper triangle — far less Python overhead than iloc nested loops.
    ii, jj = np.triu_indices(n, k=1)
    vals = mat[ii, jj]
    # Process in chunks so long pair builds release the GIL for the Tk UI thread.
    chunk = 8192
    total = int(vals.shape[0])
    for start in range(0, total, chunk):
        end = min(start + chunk, total)
        for k in range(start, end):
            r = float(vals[k])
            if r != r:  # NaN
                continue
            out.append((cols[int(ii[k])], cols[int(jj[k])], r))
        _yield_gil()
    return out


def _union_find_clusters(
    features: Sequence[str],
    pairs: Sequence[tuple[str, str, float]],
    *,
    threshold: float = 0.95,
) -> dict[str, list[str]]:
    parent = {f: f for f in features}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b, r in pairs:
        if abs(r) >= float(threshold):
            union(a, b)

    groups: dict[str, list[str]] = {}
    for f in features:
        groups.setdefault(find(f), []).append(f)
    return groups


def _name_cluster(members: list[str]) -> str:
    votes: dict[str, int] = {}
    for m in members:
        fam = _family_label(m)
        votes[fam] = votes.get(fam, 0) + 1
    best = max(votes.items(), key=lambda kv: (kv[1], kv[0]))[0]
    return f"{best} Family"


def _representative(members: list[str], pairs: Sequence[tuple[str, str, float]]) -> tuple[str, float]:
    """Pick member with highest mean |r| to others in the cluster."""
    if len(members) == 1:
        return members[0], 1.0
    member_set = set(members)
    scores: dict[str, list[float]] = {m: [] for m in members}
    best_pair = 0.0
    for a, b, r in pairs:
        if a in member_set and b in member_set:
            ar = abs(r)
            best_pair = max(best_pair, ar)
            scores[a].append(ar)
            scores[b].append(ar)
    ranked = sorted(
        members,
        key=lambda m: (
            -(sum(scores[m]) / len(scores[m]) if scores[m] else 0.0),
            m,
        ),
    )
    return ranked[0], best_pair


def persist_correlation_results(
    data_dir: str,
    run_id: str,
    *,
    corr: Any,
    features: Sequence[str],
    pairs: Sequence[tuple[str, str, float]],
    cluster_threshold: float = 0.95,
) -> dict[str, Any]:
    """Write correlation + clusters + summary meta for a run."""
    n_feat = len(features)
    n_pairs = len(pairs)
    ge95 = sum(1 for _, _, r in pairs if abs(r) >= 0.95)
    ge99 = sum(1 for _, _, r in pairs if abs(r) >= 0.99)

    groups = _union_find_clusters(features, pairs, threshold=cluster_threshold)
    # Prefer multi-member clusters for "top cluster"
    named: list[dict[str, Any]] = []
    for members in groups.values():
        members = sorted(members)
        label = _name_cluster(members)
        rep, high = _representative(members, pairs)
        named.append(
            {
                "cluster": label,
                "members": members,
                "size": len(members),
                "representative": rep,
                "highest_correlation": high,
            }
        )
    # Disambiguate duplicate family labels
    label_counts: dict[str, int] = {}
    for item in named:
        base = str(item["cluster"])
        label_counts[base] = label_counts.get(base, 0) + 1
    seen: dict[str, int] = {}
    for item in named:
        base = str(item["cluster"])
        if label_counts[base] > 1:
            seen[base] = seen.get(base, 0) + 1
            item["cluster"] = f"{base} #{seen[base]}"

    multi = [c for c in named if int(c["size"]) > 1]
    multi.sort(key=lambda c: (-int(c["size"]), -float(c["highest_correlation"])))
    top = multi[0] if multi else (named[0] if named else None)

    excluded_constant: list[str] = []
    demeaned_by = None
    try:
        attrs = getattr(corr, "attrs", None) or {}
        raw_ex = attrs.get("excluded_constant_features") or []
        excluded_constant = [str(x) for x in raw_ex if str(x).strip()]
        demeaned_by = attrs.get("demeaned_by")
    except Exception:
        excluded_constant = []
        demeaned_by = None

    summary = {
        "features_analysed": n_feat,
        "pairs": n_pairs,
        "pairs_ge_095": ge95,
        "pairs_ge_099": ge99,
        "top_cluster": str(top["cluster"]) if top else "",
        "top_cluster_size": int(top["size"]) if top else 0,
        "cluster_count": len([c for c in named if int(c["size"]) > 1]),
        "cluster_threshold": float(cluster_threshold),
        "excluded_constant_features": excluded_constant,
        "excluded_constant_count": len(excluded_constant),
        "demeaned_by": demeaned_by,
        "excluded_day_constants": bool(
            (getattr(corr, "attrs", None) or {}).get("excluded_day_constants")
        ),
    }
    load_meta = (getattr(corr, "attrs", None) or {}).get("dataset_load")
    if isinstance(load_meta, dict) and load_meta:
        summary["dataset_load"] = dict(load_meta)
    compute_meta = (getattr(corr, "attrs", None) or {}).get("compute_backend")
    if isinstance(compute_meta, dict) and compute_meta:
        summary["compute_backend"] = dict(compute_meta)

    with _AnalysisDb(data_dir) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS correlation_summary (
                run_id TEXT PRIMARY KEY,
                features_analysed INTEGER,
                pairs INTEGER,
                pairs_ge_095 INTEGER,
                pairs_ge_099 INTEGER,
                top_cluster TEXT,
                top_cluster_size INTEGER,
                cluster_count INTEGER,
                cluster_threshold REAL,
                summary_json TEXT,
                created_at TEXT
            )
            """
        )
        conn.execute("DELETE FROM correlation WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM clusters WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM correlation_summary WHERE run_id = ?", (run_id,))

        # Chunked inserts so large pair lists release the GIL between batches.
        pair_rows = [(run_id, a, b, float(r)) for a, b, r in pairs]
        insert_sql = """
            INSERT INTO correlation (run_id, feature_a, feature_b, correlation)
            VALUES (?, ?, ?, ?)
            """
        chunk = 5000
        for start in range(0, len(pair_rows), chunk):
            conn.executemany(insert_sql, pair_rows[start : start + chunk])
            _yield_gil()
        cluster_rows: list[tuple[Any, ...]] = []
        for item in named:
            for m in item["members"]:
                cluster_rows.append(
                    (
                        run_id,
                        m,
                        item["cluster"],
                        1 if m == item["representative"] else 0,
                    )
                )
        conn.executemany(
            """
            INSERT INTO clusters (run_id, feature, cluster, representative)
            VALUES (?, ?, ?, ?)
            """,
            cluster_rows,
        )
        conn.execute(
            """
            INSERT INTO correlation_summary (
                run_id, features_analysed, pairs, pairs_ge_095, pairs_ge_099,
                top_cluster, top_cluster_size, cluster_count, cluster_threshold,
                summary_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                n_feat,
                n_pairs,
                ge95,
                ge99,
                summary["top_cluster"],
                summary["top_cluster_size"],
                summary["cluster_count"],
                cluster_threshold,
                json.dumps(
                    {
                        "clusters": named[:200],
                        "excluded_constant_features": excluded_constant[:200],
                        "excluded_constant_count": len(excluded_constant),
                        "demeaned_by": demeaned_by,
                        "compute_backend": (
                            dict(summary["compute_backend"])
                            if isinstance(summary.get("compute_backend"), dict)
                            else None
                        ),
                        "dataset_load": (
                            dict(summary["dataset_load"])
                            if isinstance(summary.get("dataset_load"), dict)
                            else None
                        ),
                    },
                    separators=(",", ":"),
                ),
                _now_iso(),
            ),
        )

    # Correlation Insights — recommendations only; never mutates features.
    from .analysis_correlation_insights import (
        build_correlation_insights,
        persist_correlation_insights,
    )

    insights = build_correlation_insights(clusters=named, pairs=pairs)
    insight_n = persist_correlation_insights(data_dir, run_id, insights)
    summary["insights_count"] = insight_n
    summary["duplicate_candidates"] = sum(
        1 for i in insights if i.get("recommendation") == "Duplicate Candidate"
    )
    summary["review_clusters"] = sum(
        1 for i in insights if i.get("recommendation") == "Review"
    )
    return summary


def run_correlation_analysis(
    data_dir: str,
    run_id: str,
    dataset: dict[str, Any],
    *,
    max_rows: int | None = DEFAULT_CORR_MAX_ROWS,
    cluster_threshold: float = 0.95,
    progress: ProgressCb | None = None,
    compute_backend: str | None = None,
) -> dict[str, Any]:
    """Full Correlation module: compute, persist, mark Completed.

    Samples up to ``DEFAULT_CORR_MAX_ROWS`` by default to avoid OOM on large
    analysis datasets. Pass ``max_rows=None`` for a full-frame run.

    ``compute_backend``: ``auto`` | ``cpu`` | ``gpu`` for Pearson matrix engine.
    """
    path = resolve_parquet_path(data_dir, dataset)
    if not path or not os.path.isfile(path):
        raise FileNotFoundError(f"Dataset parquet not found: {path!r}")

    started = _now_iso()
    t0 = time.perf_counter()

    def _tick(frac: float, message: str, **extra: Any) -> None:
        if not progress:
            return
        progress(
            {
                "frac": max(0.0, min(1.0, float(frac))),
                "elapsed": max(time.perf_counter() - t0, 0.0),
                "message": str(message),
                **extra,
            }
        )

    set_module_status(
        data_dir,
        run_id,
        "correlation",
        STATUS_RUNNING,
        started_at=started,
        message="Computing correlation matrix…",
    )
    _tick(0.01, "Starting Correlation…")
    try:
        corr, features = compute_correlation_frame(
            path,
            max_rows=max_rows,
            progress=progress,
            started_at=t0,
            compute_backend=compute_backend,
        )
        _tick(0.90, "Building pair list…")
        pairs = _pair_rows(corr)
        _tick(0.93, f"Persisting {len(pairs):,} pairs + clusters…")
        summary = persist_correlation_results(
            data_dir,
            run_id,
            corr=corr,
            features=features,
            pairs=pairs,
            cluster_threshold=cluster_threshold,
        )
        _tick(0.99, "Writing Correlation Insights…")
        elapsed = max(time.perf_counter() - t0, 0.0)
        set_module_status(
            data_dir,
            run_id,
            "correlation",
            STATUS_COMPLETED,
            started_at=started,
            finished_at=_now_iso(),
            elapsed_sec=round(elapsed, 3),
            message=(
                f"{summary['features_analysed']} features · "
                f"{summary['pairs']:,} pairs · "
                f"|r|>=0.95: {summary['pairs_ge_095']:,}"
            ),
        )
        _tick(
            1.0,
            f"Done · {summary['features_analysed']} features · "
            f"{summary['pairs']:,} pairs · {elapsed:.1f}s",
        )
        return summary
    except Exception as exc:
        set_module_status(
            data_dir,
            run_id,
            "correlation",
            STATUS_FAILED,
            started_at=started,
            finished_at=_now_iso(),
            elapsed_sec=round(max(time.perf_counter() - t0, 0.0), 3),
            message=str(exc),
        )
        raise


def load_correlation_summary(data_dir: str, run_id: str) -> dict[str, Any] | None:
    with _AnalysisDb(data_dir) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS correlation_summary (
                run_id TEXT PRIMARY KEY,
                features_analysed INTEGER,
                pairs INTEGER,
                pairs_ge_095 INTEGER,
                pairs_ge_099 INTEGER,
                top_cluster TEXT,
                top_cluster_size INTEGER,
                cluster_count INTEGER,
                cluster_threshold REAL,
                summary_json TEXT,
                created_at TEXT
            )
            """
        )
        row = conn.execute(
            "SELECT * FROM correlation_summary WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return dict(row) if row else None


def load_top_pairs(
    data_dir: str,
    run_id: str,
    *,
    limit: int = 500,
    min_abs: float = 0.0,
) -> list[dict[str, Any]]:
    with _AnalysisDb(data_dir) as conn:
        rows = conn.execute(
            """
            SELECT feature_a, feature_b, correlation
            FROM correlation
            WHERE run_id = ? AND ABS(correlation) >= ?
            ORDER BY ABS(correlation) DESC, feature_a, feature_b
            LIMIT ?
            """,
            (run_id, float(min_abs), int(limit)),
        ).fetchall()
        return [dict(r) for r in rows]


def load_clusters(data_dir: str, run_id: str) -> list[dict[str, Any]]:
    with _AnalysisDb(data_dir) as conn:
        rows = conn.execute(
            """
            SELECT feature, cluster, representative
            FROM clusters
            WHERE run_id = ?
            ORDER BY cluster, feature
            """,
            (run_id,),
        ).fetchall()
    by: dict[str, dict[str, Any]] = {}
    for r in rows:
        c = str(r["cluster"])
        slot = by.setdefault(
            c,
            {"cluster": c, "members": [], "representative": "", "size": 0},
        )
        feat = str(r["feature"])
        slot["members"].append(feat)
        if int(r["representative"] or 0):
            slot["representative"] = feat
        slot["size"] = len(slot["members"])
    # Attach highest |r| within cluster from summary_json if present
    summary = load_correlation_summary(data_dir, run_id) or {}
    extra = {}
    try:
        payload = json.loads(str(summary.get("summary_json") or "{}"))
        for item in payload.get("clusters") or []:
            extra[str(item.get("cluster"))] = item
    except Exception:
        extra = {}
    out = list(by.values())
    for item in out:
        meta = extra.get(item["cluster"]) or {}
        item["highest_correlation"] = float(meta.get("highest_correlation") or 0.0)
        if not item["representative"]:
            item["representative"] = meta.get("representative") or (
                item["members"][0] if item["members"] else ""
            )
    out.sort(key=lambda c: (-int(c["size"]), str(c["cluster"])))
    return out


def load_matrix_slice(
    data_dir: str,
    run_id: str,
    features: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Return nested dict matrix[a][b] = r for requested features (incl. diagonal)."""
    feats = [str(f).strip() for f in features if str(f).strip()]
    mat: dict[str, dict[str, float]] = {f: {f: 1.0} for f in feats}
    if len(feats) < 2:
        return mat
    with _AnalysisDb(data_dir) as conn:
        # Fetch all pairs among the subset
        placeholders = ",".join("?" * len(feats))
        rows = conn.execute(
            f"""
            SELECT feature_a, feature_b, correlation
            FROM correlation
            WHERE run_id = ?
              AND feature_a IN ({placeholders})
              AND feature_b IN ({placeholders})
            """,
            [run_id, *feats, *feats],
        ).fetchall()
    for r in rows:
        a, b = str(r["feature_a"]), str(r["feature_b"])
        val = float(r["correlation"])
        mat.setdefault(a, {})[b] = val
        mat.setdefault(b, {})[a] = val
    return mat


def list_correlated_features(data_dir: str, run_id: str) -> list[str]:
    with _AnalysisDb(data_dir) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT feature FROM (
                SELECT feature_a AS feature FROM correlation WHERE run_id = ?
                UNION
                SELECT feature_b AS feature FROM correlation WHERE run_id = ?
            )
            ORDER BY feature
            """,
            (run_id, run_id),
        ).fetchall()
        return [str(r["feature"]) for r in rows]


__all__ = [
    "compute_correlation_frame",
    "load_clusters",
    "load_correlation_summary",
    "load_matrix_slice",
    "load_top_pairs",
    "list_correlated_features",
    "persist_correlation_results",
    "run_correlation_analysis",
]
