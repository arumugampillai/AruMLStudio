"""Trading-day metadata — compute once on ingest, UI reads only.

Tables
------
- ``master_dataset_day_metadata`` — one row per trading day
- ``master_dataset_column_metadata`` — one row per day × column
- ``master_dataset_gap_metadata`` — gap events per day
"""

from __future__ import annotations

import json
import math
import statistics
from datetime import datetime, timezone
from typing import Any, Iterable

DEFAULT_META_COLUMNS: tuple[str, ...] = (
    "trading_day",
    "market",
    "expiry",
    "timestamp",
    "strike",
    "option_type",
    "token",
    "symbol",
    "master_row_id",
)

FEATURE_FAMILIES: tuple[str, ...] = (
    "Base",
    "Derived",
    "Market Structure",
    "Greeks",
    "Volatility",
    "Order Book",
    "Prediction",
    "Target",
    "Meta",
    "Other",
)

_GROUP_TO_FAMILY: dict[str, str] = {
    "price": "Base",
    "greeks": "Greeks",
    "iv": "Volatility",
    "iv_zscore": "Volatility",
    "iv_ema_ratio": "Volatility",
    "market_microstructure": "Order Book",
    "oi": "Market Structure",
    "chain": "Market Structure",
    "moneyness": "Market Structure",
    "atm_straddle": "Market Structure",
    "atm6_ltp": "Market Structure",
    "dgt_reiv": "Prediction",
    "time": "Meta",
}

_DAY_TABLE = "master_dataset_day_metadata"
_COL_TABLE = "master_dataset_column_metadata"
_GAP_TABLE = "master_dataset_gap_metadata"

_DAY_EXTRA_COLUMNS: dict[str, str] = {
    "observed_interval_sec": "REAL",
    "trading_duration_sec": "REAL",
    "token_count": "INTEGER",
    "expiry": "TEXT",
    "spot_min": "REAL",
    "spot_max": "REAL",
    "avg_iv": "REAL",
    "avg_spread": "REAL",
    "expected_samples": "INTEGER",
    "actual_unique_timestamps": "INTEGER",
    "missing_timestamps": "INTEGER",
    "gap_triggered": "INTEGER",
    "gap_ignored": "INTEGER",
    "gap_filled": "INTEGER",
    "dataset_version": "TEXT",
    "registry_version": "TEXT",
    "feature_engine_version": "TEXT",
    "gap_policy_version": "TEXT",
    "imported_at": "TEXT",
    "build_duration_sec": "REAL",
    "metadata_generated_at": "TEXT",
    "healthy_features": "INTEGER",
    "sparse_features": "INTEGER",
    "expected_empty_features": "INTEGER",
    "unexpected_empty_features": "INTEGER",
    "status_counts_json": "TEXT",
}

_COL_EXTRA_COLUMNS: dict[str, str] = {
    "status": "TEXT",
    "feature_family": "TEXT",
    "reason": "TEXT",
    "availability": "TEXT",
    "source": "TEXT",
    "expected_empty": "INTEGER",
    "required_flag": "INTEGER",
    "can_be_empty": "INTEGER",
}


_DDL = f"""
CREATE TABLE IF NOT EXISTS {_DAY_TABLE} (
    trading_day            TEXT PRIMARY KEY,
    total_rows             INTEGER NOT NULL,
    total_columns          INTEGER NOT NULL,
    registry_features      INTEGER NOT NULL,
    meta_columns           INTEGER NOT NULL,
    first_timestamp        REAL,
    last_timestamp         REAL,
    average_coverage       REAL,
    complete_features      INTEGER NOT NULL DEFAULT 0,
    partial_features       INTEGER NOT NULL DEFAULT 0,
    empty_features         INTEGER NOT NULL DEFAULT 0,
    warmup_features        INTEGER NOT NULL DEFAULT 0,
    gap_policy_sec         REAL,
    sampling_interval_sec  REAL,
    gap_events             INTEGER NOT NULL DEFAULT 0,
    largest_gap_sec        REAL,
    rows_affected_by_gaps  INTEGER NOT NULL DEFAULT 0,
    coverage_loss_pct      REAL,
    duplicate_timestamps   INTEGER NOT NULL DEFAULT 0,
    out_of_order_timestamps INTEGER NOT NULL DEFAULT 0,
    constant_features      INTEGER NOT NULL DEFAULT 0,
    infinite_values        INTEGER NOT NULL DEFAULT 0,
    nan_values             INTEGER NOT NULL DEFAULT 0,
    registry_expected      INTEGER NOT NULL DEFAULT 0,
    registry_found         INTEGER NOT NULL DEFAULT 0,
    registry_missing       INTEGER NOT NULL DEFAULT 0,
    registry_unexpected    INTEGER NOT NULL DEFAULT 0,
    duplicate_columns      INTEGER NOT NULL DEFAULT 0,
    health_score           REAL,
    health_components_json TEXT,
    build_version          TEXT,
    created_at             TEXT NOT NULL,
    updated_at             TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS {_COL_TABLE} (
    trading_day     TEXT NOT NULL,
    feature         TEXT NOT NULL,
    column_type     TEXT,
    non_null        INTEGER NOT NULL DEFAULT 0,
    null_count      INTEGER NOT NULL DEFAULT 0,
    coverage_pct    REAL,
    first_valid_ts  REAL,
    last_valid_ts   REAL,
    warmup_rows     INTEGER NOT NULL DEFAULT 0,
    is_constant     INTEGER NOT NULL DEFAULT 0,
    infinite_count  INTEGER NOT NULL DEFAULT 0,
    nan_count       INTEGER NOT NULL DEFAULT 0,
    notes           TEXT,
    PRIMARY KEY (trading_day, feature)
);

CREATE TABLE IF NOT EXISTS {_GAP_TABLE} (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    trading_day      TEXT NOT NULL,
    token            TEXT,
    start_ts         REAL NOT NULL,
    end_ts           REAL NOT NULL,
    gap_seconds      REAL NOT NULL,
    missing_samples  INTEGER,
    action           TEXT,
    UNIQUE (trading_day, token, start_ts, end_ts)
);

CREATE INDEX IF NOT EXISTS idx_day_meta_col_day ON {_COL_TABLE}(trading_day);
CREATE INDEX IF NOT EXISTS idx_day_meta_gap_day ON {_GAP_TABLE}(trading_day);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_null(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip().lower() in ("", "nan", "none", "null"):
        return True
    return False


def _is_inf(value: Any) -> bool:
    try:
        return isinstance(value, float) and math.isinf(value)
    except Exception:
        return False


def _migrate_extra(conn: Any, table: str, extras: dict[str, str]) -> None:
    existing = {
        str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()
    }
    for col, sql_type in extras.items():
        if col not in existing:
            conn.execute(f'ALTER TABLE {table} ADD COLUMN "{col}" {sql_type}')


def ensure_day_metadata_tables(conn: Any) -> None:
    conn.executescript(_DDL)
    _migrate_extra(conn, _DAY_TABLE, _DAY_EXTRA_COLUMNS)
    _migrate_extra(conn, _COL_TABLE, _COL_EXTRA_COLUMNS)
    conn.commit()


def delete_day_metadata(conn: Any, trading_day: str) -> None:
    td = str(trading_day)
    ensure_day_metadata_tables(conn)
    conn.execute(f"DELETE FROM {_COL_TABLE} WHERE trading_day = ?", (td,))
    conn.execute(f"DELETE FROM {_GAP_TABLE} WHERE trading_day = ?", (td,))
    conn.execute(f"DELETE FROM {_DAY_TABLE} WHERE trading_day = ?", (td,))


def feature_family_map_from_registry(registry: dict[str, Any] | None) -> dict[str, str]:
    """Map feature name → UI family from feature registry groups."""
    out: dict[str, str] = {}
    if not isinstance(registry, dict):
        return out
    groups = registry.get("groups") or {}
    if not isinstance(groups, dict):
        return out
    for gid, gdoc in groups.items():
        family = _GROUP_TO_FAMILY.get(str(gid), "Derived")
        feats = []
        if isinstance(gdoc, dict):
            feats = gdoc.get("features") or []
        elif isinstance(gdoc, list):
            feats = gdoc
        for f in feats:
            name = str(f.get("name") if isinstance(f, dict) else f).strip()
            if name:
                out[name] = family
    return out


def _resolve_family(
    name: str,
    *,
    meta_set: set[str],
    family_by_name: dict[str, str],
    missing: bool = False,
) -> str:
    if name in meta_set:
        return "Meta"
    if missing:
        return family_by_name.get(name, "Other")
    if name.startswith("future_") or name.endswith("_target"):
        return "Target"
    if name in family_by_name:
        return family_by_name[name]
    low = name.lower()
    if any(k in low for k in ("delta", "gamma", "theta", "vega", "charm", "vanna", "vomma")):
        return "Greeks"
    if "iv" in low or "vol" in low:
        return "Volatility"
    if any(k in low for k in ("bid", "ask", "book", "depth", "spread")):
        return "Order Book"
    if any(k in low for k in ("wall", "oi", "pcr", "straddle", "gex")):
        return "Market Structure"
    if "pred" in low or "reiv" in low:
        return "Prediction"
    return "Derived"


def _status_and_notes(
    *,
    nn: int,
    n: int,
    cov: float,
    is_warmup: bool,
    is_const: bool,
    is_meta: bool,
    registry_missing: bool = False,
) -> tuple[str, str]:
    """Legacy helper kept for callers; new path uses classify_population."""
    if registry_missing:
        return "Registry Missing", "Expected by registry but absent from samples"
    if nn == 0:
        return "Empty", "No non-null values"
    if is_warmup and cov >= 50.0:
        return "Warm-up", "Rolling window initialization"
    if cov < 50.0:
        return "Sparse", "Signal occurs conditionally"
    if is_const and not is_meta:
        return "Constant", "Expected constant"
    if cov < 100.0:
        return "Partial", f"Coverage {cov:.2f}%"
    return "Healthy", "Fully populated"


def _fmt_clock(ts: float | None) -> str | None:
    if ts is None:
        return None
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
    except (OverflowError, OSError, ValueError, TypeError):
        return None


def _detect_gaps_per_token(
    rows: list[dict[str, Any]],
    *,
    gap_max_sec: float,
    sampling_interval_sec: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_token: dict[str, list[float]] = {}
    for r in rows:
        tok = str(r.get("token") or "")
        ts = r.get("timestamp")
        if ts is None:
            continue
        try:
            by_token.setdefault(tok, []).append(float(ts))
        except (TypeError, ValueError):
            continue

    events: list[dict[str, Any]] = []
    largest = 0.0
    affected = 0
    triggered_n = ignored_n = 0
    missing_total = 0
    deltas: list[float] = []
    step = max(float(sampling_interval_sec or 1.0), 0.001)
    threshold = float(gap_max_sec or 0.0)
    expected = 0
    actual_unique = 0

    for tok, stamps in by_token.items():
        stamps = sorted(set(stamps))
        actual_unique += len(stamps)
        if len(stamps) >= 2:
            span = stamps[-1] - stamps[0]
            expected += int(round(span / step)) + 1
        elif stamps:
            expected += 1
        for i in range(1, len(stamps)):
            dt = stamps[i] - stamps[i - 1]
            if dt <= 0:
                continue
            deltas.append(dt)
            largest = max(largest, dt)
            missing = max(int(round(dt / step)) - 1, 0)
            if threshold > 0 and dt > threshold:
                triggered_n += 1
                affected += missing
                missing_total += missing
                events.append(
                    {
                        "token": tok,
                        "start_ts": stamps[i - 1],
                        "end_ts": stamps[i],
                        "gap_seconds": round(dt, 3),
                        "missing_samples": missing,
                        "action": "Gap policy triggered",
                    }
                )
            elif dt > step * 1.5 and missing > 0:
                ignored_n += 1
                missing_total += missing
                events.append(
                    {
                        "token": tok,
                        "start_ts": stamps[i - 1],
                        "end_ts": stamps[i],
                        "gap_seconds": round(dt, 3),
                        "missing_samples": missing,
                        "action": "Ignored",
                    }
                )

    observed = float(statistics.median(deltas)) if deltas else float(step)
    summary = {
        "largest_gap_sec": round(largest, 3) if largest else 0.0,
        "rows_affected": affected,
        "gap_triggered": triggered_n,
        "gap_ignored": ignored_n,
        "gap_filled": 0,
        "expected_samples": expected,
        "actual_unique_timestamps": actual_unique,
        "missing_timestamps": max(0, expected - actual_unique),
        "missing_samples_from_gaps": missing_total,
        "observed_interval_sec": round(observed, 4),
        "token_count": len(by_token),
    }
    events.sort(key=lambda e: (-float(e["gap_seconds"]), e.get("start_ts") or 0))
    return events[:500], summary


def _timestamp_quality(rows: list[dict[str, Any]]) -> tuple[int, int]:
    by_token: dict[str, list[float]] = {}
    for r in rows:
        tok = str(r.get("token") or "")
        ts = r.get("timestamp")
        if ts is None:
            continue
        try:
            by_token.setdefault(tok, []).append(float(ts))
        except (TypeError, ValueError):
            continue
    dup = ooo = 0
    for stamps in by_token.values():
        seen: set[float] = set()
        prev: float | None = None
        for t in stamps:
            if t in seen:
                dup += 1
            seen.add(t)
            if prev is not None and t < prev:
                ooo += 1
            prev = t
    return dup, ooo


def _day_market_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    spot_min = spot_max = None
    iv_sum = iv_n = 0.0
    spread_sum = spread_n = 0.0
    expiry = None
    for r in rows:
        if expiry is None and r.get("expiry") not in (None, ""):
            expiry = str(r.get("expiry"))
        spot = r.get("spot")
        if not _is_null(spot):
            try:
                sv = float(spot)
                spot_min = sv if spot_min is None else min(spot_min, sv)
                spot_max = sv if spot_max is None else max(spot_max, sv)
            except (TypeError, ValueError):
                pass
        for iv_key in ("current_iv", "atm_iv_ce", "atm_iv_pe"):
            iv = r.get(iv_key)
            if not _is_null(iv):
                try:
                    iv_sum += float(iv)
                    iv_n += 1
                    break
                except (TypeError, ValueError):
                    pass
        sp = r.get("bid_ask_spread")
        if not _is_null(sp):
            try:
                spread_sum += float(sp)
                spread_n += 1
            except (TypeError, ValueError):
                pass
    return {
        "expiry": expiry,
        "spot_min": spot_min,
        "spot_max": spot_max,
        "avg_iv": round(iv_sum / iv_n, 6) if iv_n else None,
        "avg_spread": round(spread_sum / spread_n, 6) if spread_n else None,
    }


def compute_day_metadata_from_rows(
    rows: list[dict[str, Any]],
    *,
    trading_day: str,
    registry_features: list[str],
    meta_columns: Iterable[str] | None = None,
    gap_max_sec: float = 20.0,
    sampling_interval_sec: float = 3.0,
    build_version: str | None = None,
    category_by_name: dict[str, str] | None = None,
    family_by_name: dict[str, str] | None = None,
    expectation_by_name: dict[str, dict[str, Any]] | None = None,
    ingestion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"No rows for trading day {trading_day}")

    from .feature_expectation import (
        classify_population,
        expectation_from_registry_entry,
    )

    meta_cols = [c for c in (meta_columns or DEFAULT_META_COLUMNS)]
    meta_set = set(meta_cols)
    reg_list = list(dict.fromkeys(str(f) for f in registry_features if str(f).strip()))
    reg_set = set(reg_list)
    categories = dict(category_by_name or {})
    families = dict(family_by_name or {})
    expectations = dict(expectation_by_name or {})
    ingest = dict(ingestion or {})

    present_keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            sk = str(k)
            if sk not in seen:
                seen.add(sk)
                present_keys.append(sk)

    ordered_cols: list[str] = []
    for group in (meta_cols, reg_list, present_keys):
        for c in group:
            if c in seen and c not in ordered_cols:
                ordered_cols.append(c)

    n = len(rows)
    non_null = {c: 0 for c in ordered_cols}
    nan_count = {c: 0 for c in ordered_cols}
    inf_count = {c: 0 for c in ordered_cols}
    first_valid: dict[str, float | None] = {c: None for c in ordered_cols}
    last_valid: dict[str, float | None] = {c: None for c in ordered_cols}
    warmup_rows = {c: 0 for c in ordered_cols}
    seen_valid = {c: False for c in ordered_cols}
    sample_vals: dict[str, list[Any]] = {c: [] for c in ordered_cols}

    first_ts: float | None = None
    last_ts: float | None = None

    for r in rows:
        ts_raw = r.get("timestamp")
        try:
            ts = float(ts_raw) if ts_raw is not None else None
        except (TypeError, ValueError):
            ts = None
        if ts is not None:
            first_ts = ts if first_ts is None else min(first_ts, ts)
            last_ts = ts if last_ts is None else max(last_ts, ts)

        for c in ordered_cols:
            v = r.get(c)
            if _is_null(v):
                if isinstance(v, float) and math.isnan(v):
                    nan_count[c] += 1
                if not seen_valid[c]:
                    warmup_rows[c] += 1
                continue
            if _is_inf(v):
                inf_count[c] += 1
            non_null[c] += 1
            seen_valid[c] = True
            if ts is not None:
                if first_valid[c] is None:
                    first_valid[c] = ts
                last_valid[c] = ts
            if len(sample_vals[c]) < 2 and v not in sample_vals[c]:
                sample_vals[c].append(v)

    gap_events, gap_summary = _detect_gaps_per_token(
        rows,
        gap_max_sec=float(gap_max_sec),
        sampling_interval_sec=float(sampling_interval_sec),
    )
    dup_ts, ooo_ts = _timestamp_quality(rows)
    market = _day_market_stats(rows)

    # Futures feed empty if every futures_* column present is all-null.
    futures_cols = [c for c in ordered_cols if str(c).startswith("futures_")]
    futures_feed_empty = bool(futures_cols) and all(non_null.get(c, 0) == 0 for c in futures_cols)

    columns_out: list[dict[str, Any]] = []
    status_counts: dict[str, int] = {}
    complete = partial = empty = warmup_feat = constant_feat = healthy = sparse = 0
    expected_empty_n = unexpected_empty_n = 0
    cov_sum = 0.0
    cov_n = 0
    total_inf = total_nan = 0

    for c in ordered_cols:
        nn = non_null[c]
        nulls = n - nn
        cov = (100.0 * nn / n) if n else 0.0
        is_const = nn > 0 and len(sample_vals[c]) == 1
        wu = int(warmup_rows[c])
        is_warmup = bool(wu > 0 and nn > 0 and wu == nulls)
        exp = expectations.get(c) or expectation_from_registry_entry(c)
        pop = classify_population(
            name=c,
            coverage_pct=cov,
            non_null=nn,
            is_warmup=is_warmup,
            is_constant=is_const,
            is_meta=c in meta_set,
            registry_missing=False,
            expectation=exp,
            futures_feed_empty=futures_feed_empty,
        )
        status = str(pop["status"])
        reason = str(pop["reason"])
        status_counts[status] = status_counts.get(status, 0) + 1
        if pop.get("expected_empty"):
            expected_empty_n += 1
        if status == "Unexpected Empty":
            unexpected_empty_n += 1
            empty += 1
        elif status in ("Empty", "Expected Empty"):
            empty += 1
        elif status == "Warm-up":
            warmup_feat += 1
        elif status == "Sparse":
            sparse += 1
            partial += 1
        elif status == "Constant":
            constant_feat += 1
            complete += 1
        elif status == "Partial":
            partial += 1
        elif status == "Healthy":
            healthy += 1
            complete += 1

        total_inf += inf_count[c]
        total_nan += nan_count[c]
        if c not in meta_set:
            cov_sum += cov
            cov_n += 1

        family = _resolve_family(c, meta_set=meta_set, family_by_name=families)
        pol = str(categories.get(c) or "").lower()
        if c in meta_set:
            ctype = "Meta"
        elif pol in ("raw", "base"):
            ctype = "Base"
        elif pol in ("rolling", "lookback", "cumulative"):
            ctype = "Rolling"
        elif pol == "derived":
            ctype = "Derived"
        elif pol == "target":
            ctype = "Target"
        else:
            ctype = family

        columns_out.append(
            {
                "feature": c,
                "column_type": ctype,
                "feature_family": family,
                "status": status,
                "reason": reason,
                "availability": pop.get("availability"),
                "source": pop.get("source") or exp.get("source"),
                "expected_empty": int(bool(pop.get("expected_empty"))),
                "required_flag": int(bool(pop.get("required"))),
                "can_be_empty": int(bool(pop.get("can_be_empty"))),
                "non_null": nn,
                "null_count": nulls,
                "coverage_pct": round(cov, 4),
                "first_valid_ts": first_valid[c],
                "last_valid_ts": last_valid[c],
                "warmup_rows": wu if is_warmup else 0,
                "is_constant": int(is_const),
                "infinite_count": inf_count[c],
                "nan_count": nan_count[c],
                "notes": reason,
            }
        )

    missing_reg = [c for c in reg_list if c not in seen]
    for c in missing_reg:
        exp = expectations.get(c) or expectation_from_registry_entry(c)
        pop = classify_population(
            name=c,
            coverage_pct=0.0,
            non_null=0,
            is_warmup=False,
            is_constant=False,
            is_meta=False,
            registry_missing=True,
            expectation=exp,
            futures_feed_empty=futures_feed_empty,
        )
        status = str(pop["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
        if pop.get("expected_empty"):
            expected_empty_n += 1
        columns_out.append(
            {
                "feature": c,
                "column_type": "Feature",
                "feature_family": _resolve_family(
                    c, meta_set=meta_set, family_by_name=families, missing=True
                ),
                "status": status,
                "reason": pop.get("reason"),
                "availability": pop.get("availability"),
                "source": pop.get("source") or exp.get("source"),
                "expected_empty": int(bool(pop.get("expected_empty"))),
                "required_flag": int(bool(pop.get("required"))),
                "can_be_empty": int(bool(pop.get("can_be_empty"))),
                "non_null": 0,
                "null_count": n,
                "coverage_pct": 0.0,
                "first_valid_ts": None,
                "last_valid_ts": None,
                "warmup_rows": 0,
                "is_constant": 0,
                "infinite_count": 0,
                "nan_count": 0,
                "notes": pop.get("reason"),
            }
        )

    found_reg = [c for c in reg_list if c in seen]
    unexpected = [
        c for c in present_keys if c not in meta_set and c not in reg_set and not c.startswith("__")
    ]
    avg_cov = round(cov_sum / cov_n, 4) if cov_n else 0.0
    coverage_loss = round(
        100.0 * int(gap_summary["rows_affected"]) / max(n, 1), 4
    )
    duration = (
        round(float(last_ts) - float(first_ts), 3)
        if first_ts is not None and last_ts is not None
        else None
    )

    components = _health_components(
        average_coverage=avg_cov,
        gap_events_triggered=int(gap_summary["gap_triggered"]),
        largest_gap_sec=float(gap_summary["largest_gap_sec"]),
        gap_max_sec=float(gap_max_sec),
        coverage_loss_pct=coverage_loss,
        registry_missing=len(missing_reg),
        registry_unexpected=len(unexpected),
        duplicate_timestamps=dup_ts,
        out_of_order=ooo_ts,
        empty_features=empty,
        feature_count=max(cov_n, 1),
    )
    score = round(
        0.40 * components["coverage"]
        + 0.20 * components["gap_quality"]
        + 0.20 * components["registry"]
        + 0.10 * components["timestamp_quality"]
        + 0.10 * components["null_quality"],
        2,
    )

    now = _utc_now()
    day_doc = {
        "trading_day": str(trading_day),
        "total_rows": n,
        "total_columns": len(ordered_cols),
        "registry_features": len(reg_list),
        "meta_columns": len([c for c in ordered_cols if c in meta_set]),
        "first_timestamp": first_ts,
        "last_timestamp": last_ts,
        "trading_duration_sec": duration,
        "sampling_interval_sec": float(sampling_interval_sec),
        "observed_interval_sec": gap_summary["observed_interval_sec"],
        "gap_policy_sec": float(gap_max_sec),
        "average_coverage": avg_cov,
        "complete_features": complete,
        "partial_features": partial,
        "empty_features": empty,
        "warmup_features": warmup_feat,
        "healthy_features": healthy,
        "sparse_features": sparse,
        "expected_empty_features": expected_empty_n,
        "unexpected_empty_features": unexpected_empty_n,
        "gap_events": int(gap_summary["gap_triggered"]),
        "largest_gap_sec": gap_summary["largest_gap_sec"],
        "rows_affected_by_gaps": int(gap_summary["rows_affected"]),
        "coverage_loss_pct": coverage_loss,
        "expected_samples": int(gap_summary["expected_samples"]),
        "actual_unique_timestamps": int(gap_summary["actual_unique_timestamps"]),
        "missing_timestamps": int(gap_summary["missing_timestamps"]),
        "gap_triggered": int(gap_summary["gap_triggered"]),
        "gap_ignored": int(gap_summary["gap_ignored"]),
        "gap_filled": int(gap_summary["gap_filled"]),
        "duplicate_timestamps": dup_ts,
        "out_of_order_timestamps": ooo_ts,
        "constant_features": constant_feat,
        "infinite_values": total_inf,
        "nan_values": total_nan,
        "token_count": int(gap_summary["token_count"]),
        "expiry": market.get("expiry"),
        "spot_min": market.get("spot_min"),
        "spot_max": market.get("spot_max"),
        "avg_iv": market.get("avg_iv"),
        "avg_spread": market.get("avg_spread"),
        "registry_expected": len(reg_list),
        "registry_found": len(found_reg),
        "registry_missing": len(missing_reg),
        "registry_unexpected": len(unexpected),
        "duplicate_columns": len(present_keys) - len(set(present_keys)),
        "health_score": score,
        "health_components": components,
        "registry_missing_names": missing_reg,
        "registry_unexpected_names": unexpected[:50],
        "status_counts": status_counts,
        "build_version": build_version or ingest.get("dataset_version") or "",
        "dataset_version": ingest.get("dataset_version") or build_version or "",
        "registry_version": str(ingest.get("registry_version") or ""),
        "feature_engine_version": str(ingest.get("feature_engine_version") or ""),
        "gap_policy_version": str(ingest.get("gap_policy_version") or ""),
        "imported_at": ingest.get("imported_at") or now,
        "build_duration_sec": ingest.get("build_duration_sec"),
        "metadata_generated_at": now,
    }
    return {"day": day_doc, "columns": columns_out, "gaps": gap_events}


def _health_components(
    *,
    average_coverage: float,
    gap_events_triggered: int,
    largest_gap_sec: float,
    gap_max_sec: float,
    coverage_loss_pct: float,
    registry_missing: int,
    registry_unexpected: int,
    duplicate_timestamps: int,
    out_of_order: int,
    empty_features: int,
    feature_count: int,
) -> dict[str, float]:
    coverage = max(0.0, min(100.0, float(average_coverage)))
    if gap_events_triggered <= 0:
        gap_quality = 100.0
    else:
        over = max(0.0, float(largest_gap_sec) - float(gap_max_sec or 0.0))
        gap_quality = max(
            0.0,
            100.0
            - min(60.0, gap_events_triggered * 8.0)
            - min(30.0, over)
            - min(20.0, coverage_loss_pct * 100),
        )
    if registry_missing == 0 and registry_unexpected == 0:
        registry = 100.0
    else:
        registry = max(
            0.0, 100.0 - registry_missing * 10.0 - min(30.0, registry_unexpected * 2.0)
        )
    if duplicate_timestamps == 0 and out_of_order == 0:
        timestamp_quality = 100.0
    else:
        timestamp_quality = max(
            0.0,
            100.0
            - min(50.0, duplicate_timestamps * 0.01)
            - min(50.0, out_of_order * 0.5),
        )
    empty_ratio = empty_features / max(feature_count, 1)
    null_quality = max(0.0, 100.0 - empty_ratio * 100.0)
    return {
        "coverage": round(coverage, 2),
        "gap_quality": round(gap_quality, 2),
        "registry": round(registry, 2),
        "timestamp_quality": round(timestamp_quality, 2),
        "null_quality": round(null_quality, 2),
    }


def persist_day_metadata(conn: Any, payload: dict[str, Any]) -> None:
    ensure_day_metadata_tables(conn)
    day = payload["day"]
    td = str(day["trading_day"])
    now = _utc_now()
    delete_day_metadata(conn, td)

    comps = dict(day.get("health_components") or {})
    comps["registry_missing_names"] = day.get("registry_missing_names") or []
    comps["registry_unexpected_names"] = day.get("registry_unexpected_names") or []

    day_row = (
        td,
        int(day["total_rows"]),
        int(day["total_columns"]),
        int(day["registry_features"]),
        int(day["meta_columns"]),
        day.get("first_timestamp"),
        day.get("last_timestamp"),
        day.get("average_coverage"),
        int(day["complete_features"]),
        int(day["partial_features"]),
        int(day["empty_features"]),
        int(day["warmup_features"]),
        day.get("gap_policy_sec"),
        day.get("sampling_interval_sec"),
        int(day["gap_events"]),
        day.get("largest_gap_sec"),
        int(day.get("rows_affected_by_gaps") or 0),
        day.get("coverage_loss_pct"),
        int(day.get("duplicate_timestamps") or 0),
        int(day.get("out_of_order_timestamps") or 0),
        int(day.get("constant_features") or 0),
        int(day.get("infinite_values") or 0),
        int(day.get("nan_values") or 0),
        int(day.get("registry_expected") or 0),
        int(day.get("registry_found") or 0),
        int(day.get("registry_missing") or 0),
        int(day.get("registry_unexpected") or 0),
        int(day.get("duplicate_columns") or 0),
        day.get("health_score"),
        json.dumps(comps, ensure_ascii=False),
        day.get("build_version") or "",
        now,
        now,
        day.get("observed_interval_sec"),
        day.get("trading_duration_sec"),
        day.get("token_count"),
        day.get("expiry"),
        day.get("spot_min"),
        day.get("spot_max"),
        day.get("avg_iv"),
        day.get("avg_spread"),
        day.get("expected_samples"),
        day.get("actual_unique_timestamps"),
        day.get("missing_timestamps"),
        day.get("gap_triggered"),
        day.get("gap_ignored"),
        day.get("gap_filled"),
        day.get("dataset_version"),
        day.get("registry_version"),
        day.get("feature_engine_version"),
        day.get("gap_policy_version"),
        day.get("imported_at"),
        day.get("build_duration_sec"),
        day.get("metadata_generated_at") or now,
        day.get("healthy_features"),
        day.get("sparse_features"),
        day.get("expected_empty_features"),
        day.get("unexpected_empty_features"),
        json.dumps(day.get("status_counts") or {}, ensure_ascii=False),
    )
    day_cols = (
        "trading_day, total_rows, total_columns, registry_features, meta_columns, "
        "first_timestamp, last_timestamp, average_coverage, "
        "complete_features, partial_features, empty_features, warmup_features, "
        "gap_policy_sec, sampling_interval_sec, gap_events, largest_gap_sec, "
        "rows_affected_by_gaps, coverage_loss_pct, "
        "duplicate_timestamps, out_of_order_timestamps, "
        "constant_features, infinite_values, nan_values, "
        "registry_expected, registry_found, registry_missing, registry_unexpected, "
        "duplicate_columns, health_score, health_components_json, "
        "build_version, created_at, updated_at, "
        "observed_interval_sec, trading_duration_sec, token_count, expiry, "
        "spot_min, spot_max, avg_iv, avg_spread, "
        "expected_samples, actual_unique_timestamps, missing_timestamps, "
        "gap_triggered, gap_ignored, gap_filled, "
        "dataset_version, registry_version, feature_engine_version, gap_policy_version, "
        "imported_at, build_duration_sec, metadata_generated_at, "
        "healthy_features, sparse_features, "
        "expected_empty_features, unexpected_empty_features, status_counts_json"
    )
    placeholders = ",".join("?" * len(day_row))
    conn.execute(
        f"INSERT INTO {_DAY_TABLE} ({day_cols}) VALUES ({placeholders})",
        day_row,
    )

    col_rows = [
        (
            td,
            str(c["feature"]),
            c.get("column_type"),
            int(c.get("non_null") or 0),
            int(c.get("null_count") or 0),
            c.get("coverage_pct"),
            c.get("first_valid_ts"),
            c.get("last_valid_ts"),
            int(c.get("warmup_rows") or 0),
            int(c.get("is_constant") or 0),
            int(c.get("infinite_count") or 0),
            int(c.get("nan_count") or 0),
            c.get("notes") or c.get("reason"),
            c.get("status"),
            c.get("feature_family"),
            c.get("reason"),
            c.get("availability"),
            c.get("source"),
            int(c.get("expected_empty") or 0),
            int(c.get("required_flag") or 0),
            int(c.get("can_be_empty") or 0),
        )
        for c in (payload.get("columns") or [])
    ]
    if col_rows:
        conn.executemany(
            f"""
            INSERT INTO {_COL_TABLE} (
                trading_day, feature, column_type, non_null, null_count, coverage_pct,
                first_valid_ts, last_valid_ts, warmup_rows, is_constant,
                infinite_count, nan_count, notes, status, feature_family,
                reason, availability, source, expected_empty, required_flag, can_be_empty
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            col_rows,
        )

    gap_rows = [
        (
            td,
            g.get("token"),
            float(g["start_ts"]),
            float(g["end_ts"]),
            float(g["gap_seconds"]),
            g.get("missing_samples"),
            g.get("action"),
        )
        for g in (payload.get("gaps") or [])
    ]
    if gap_rows:
        conn.executemany(
            f"""
            INSERT INTO {_GAP_TABLE} (
                trading_day, token, start_ts, end_ts, gap_seconds, missing_samples, action
            ) VALUES (?,?,?,?,?,?,?)
            """,
            gap_rows,
        )
    conn.commit()


def build_and_persist_day_metadata(
    conn: Any,
    rows: list[dict[str, Any]],
    *,
    trading_day: str,
    registry_features: list[str],
    meta_columns: Iterable[str] | None = None,
    gap_max_sec: float = 20.0,
    sampling_interval_sec: float = 3.0,
    build_version: str | None = None,
    category_by_name: dict[str, str] | None = None,
    family_by_name: dict[str, str] | None = None,
    expectation_by_name: dict[str, dict[str, Any]] | None = None,
    ingestion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = compute_day_metadata_from_rows(
        rows,
        trading_day=trading_day,
        registry_features=registry_features,
        meta_columns=meta_columns,
        gap_max_sec=gap_max_sec,
        sampling_interval_sec=sampling_interval_sec,
        build_version=build_version,
        category_by_name=category_by_name,
        family_by_name=family_by_name,
        expectation_by_name=expectation_by_name,
        ingestion=ingestion,
    )
    persist_day_metadata(conn, payload)
    return payload


def load_day_overview(conn: Any, trading_day: str) -> dict[str, Any] | None:
    ensure_day_metadata_tables(conn)
    cur = conn.execute(
        f"SELECT * FROM {_DAY_TABLE} WHERE trading_day = ?",
        (str(trading_day),),
    )
    row = cur.fetchone()
    if not row or not cur.description:
        return None
    keys = [d[0] for d in cur.description]
    doc = dict(zip(keys, row))
    for blob_key, out_key in (
        ("health_components_json", "health_components"),
        ("status_counts_json", "status_counts"),
    ):
        raw = doc.pop(blob_key, None)
        try:
            doc[out_key] = json.loads(raw) if raw else {}
        except (TypeError, json.JSONDecodeError):
            doc[out_key] = {}
    return doc


def load_column_metadata(conn: Any, trading_day: str) -> list[dict[str, Any]]:
    ensure_day_metadata_tables(conn)
    cur = conn.execute(
        f"""
        SELECT feature, column_type, feature_family, status, reason, availability,
               source, expected_empty, required_flag, can_be_empty,
               non_null, null_count, coverage_pct, first_valid_ts, last_valid_ts,
               warmup_rows, is_constant, infinite_count, nan_count, notes
        FROM {_COL_TABLE}
        WHERE trading_day = ?
        ORDER BY feature
        """,
        (str(trading_day),),
    )
    keys = [d[0] for d in cur.description]
    rows = [dict(zip(keys, r)) for r in cur.fetchall()]
    for r in rows:
        if not r.get("status"):
            notes = str(r.get("notes") or "")
            r["status"] = {
                "✓": "Healthy",
                "Empty": "Empty",
                "Warm-up": "Warm-up",
                "Sparse": "Sparse",
                "Partial": "Partial",
                "Constant": "Constant",
            }.get(notes, notes or "Healthy")
        if not r.get("reason"):
            r["reason"] = r.get("notes") or ""
        if not r.get("feature_family"):
            r["feature_family"] = r.get("column_type") or "Other"
        if not r.get("availability"):
            if r.get("expected_empty") or r.get("status") in ("Expected Empty",):
                r["availability"] = "Optional"
            elif r.get("status") in ("Unexpected Empty",):
                r["availability"] = "Unavailable"
            elif r.get("status") == "Registry Missing":
                r["availability"] = "Optional"
            else:
                r["availability"] = "Available"
        if not r.get("source"):
            r["source"] = "Unknown"
    return rows


def load_gap_metadata(conn: Any, trading_day: str) -> list[dict[str, Any]]:
    ensure_day_metadata_tables(conn)
    cur = conn.execute(
        f"""
        SELECT token, start_ts, end_ts, gap_seconds, missing_samples, action
        FROM {_GAP_TABLE}
        WHERE trading_day = ?
        ORDER BY gap_seconds DESC, start_ts
        """,
        (str(trading_day),),
    )
    keys = [d[0] for d in cur.description]
    return [dict(zip(keys, r)) for r in cur.fetchall()]


def list_days_with_metadata(conn: Any) -> list[str]:
    ensure_day_metadata_tables(conn)
    rows = conn.execute(
        f"SELECT trading_day FROM {_DAY_TABLE} ORDER BY trading_day"
    ).fetchall()
    return [str(r[0]) for r in rows]


def rebuild_day_metadata_from_samples(
    conn: Any,
    trading_day: str,
    *,
    registry_features: list[str],
    meta_columns: Iterable[str] | None = None,
    gap_max_sec: float = 20.0,
    sampling_interval_sec: float = 3.0,
    build_version: str | None = None,
    category_by_name: dict[str, str] | None = None,
    family_by_name: dict[str, str] | None = None,
    expectation_by_name: dict[str, dict[str, Any]] | None = None,
    ingestion: dict[str, Any] | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    td = str(trading_day)
    sample_cols = [
        str(r[1]) for r in conn.execute("PRAGMA table_info(samples)").fetchall()
    ]
    if not sample_cols:
        raise RuntimeError("samples table has no columns")
    col_sql = ", ".join(f'"{c}"' for c in sample_cols)
    sql = f"SELECT {col_sql} FROM samples WHERE trading_day = ?"
    params: list[Any] = [td]
    if limit is not None:
        sql += " LIMIT ?"
        params.append(int(limit))
    cur = conn.execute(sql, params)
    keys = [d[0] for d in cur.description]
    rows = [dict(zip(keys, r)) for r in cur.fetchall()]
    if not rows:
        raise RuntimeError(f"No samples for trading day {td}")
    return build_and_persist_day_metadata(
        conn,
        rows,
        trading_day=td,
        registry_features=registry_features,
        meta_columns=meta_columns,
        gap_max_sec=gap_max_sec,
        sampling_interval_sec=sampling_interval_sec,
        build_version=build_version,
        category_by_name=category_by_name,
        family_by_name=family_by_name,
        expectation_by_name=expectation_by_name,
        ingestion=ingestion,
    )


def format_clock(ts: float | None) -> str:
    return _fmt_clock(ts) or "—"
