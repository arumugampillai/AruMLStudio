"""Read-only Reward/Risk label validation for persisted prediction rows."""

from __future__ import annotations

import math
import statistics
from typing import Any

from .prediction_schema import compute_rr_hit_labels, horizon_sec_from_target
from .research_dashboard import PREMIUM_BANDS
from .store import ModelLabStore

_HORIZON_TOL_SEC = 1.0
_TS_TOL_SEC = 0.5
_RR_EPS = 1e-9


def _pct(part: int | float, total: int | float) -> float | None:
    if total is None or float(total) <= 0:
        return None
    return 100.0 * float(part) / float(total)


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    p = max(0.0, min(100.0, float(p)))
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return sorted_vals[lo]
    w = k - lo
    return sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w


def _check_label_consistency(
    *,
    target_hits: int,
    rr_1_1: int | None,
    rr_2_3: int | None,
    rr_1_2: int,
    rr_1_3: int,
    rr_1_4: int,
) -> dict[str, Any]:
    """
    Monotonic RR counts:
      Target Hit ≥ RR 1:1 ≥ RR 2:3 ≥ RR 1:2 ≥ RR 1:3 ≥ RR 1:4
    When extended labels (1:1 / 2:3) are not yet populated, only the legacy
    RR 1:2 ≥ RR 1:3 ≥ RR 1:4 chain is checked.
    """
    chain: list[tuple[str, int]] = [("Target Hit", target_hits)]
    if rr_1_1 is not None:
        chain.append(("RR 1:1", rr_1_1))
    if rr_2_3 is not None:
        chain.append(("RR 2:3", rr_2_3))
    chain.extend(
        [
            ("RR 1:2", rr_1_2),
            ("RR 1:3", rr_1_3),
            ("RR 1:4", rr_1_4),
        ]
    )
    for i in range(1, len(chain)):
        prev_label, prev_n = chain[i - 1]
        label, n = chain[i]
        if n > prev_n:
            return {
                "ok": False,
                "message": f"{label} exceeds {prev_label}.",
            }
    return {"ok": True, "message": "RR label consistency passed"}


def _window_bounds_ok(
    *,
    ts: float | None,
    exit_at: float | None,
    at: float | None,
) -> bool:
    if ts is None or exit_at is None or at is None:
        return True
    try:
        t0 = float(ts)
        t1 = float(exit_at)
        t = float(at)
    except (TypeError, ValueError):
        return False
    return (t0 - _TS_TOL_SEC) <= t <= (t1 + _TS_TOL_SEC)


def _exit_matches_horizon(
    *,
    ts: float | None,
    exit_at: float | None,
    horizon_sec: float,
) -> bool:
    if ts is None or exit_at is None:
        return False
    try:
        return abs(float(exit_at) - (float(ts) + float(horizon_sec))) <= _HORIZON_TOL_SEC
    except (TypeError, ValueError):
        return False


def _reward_risk_ratio(profit: Any, drawdown: Any) -> float | None:
    """Reward/Risk = maximum_profit / maximum_drawdown (when risk > 0)."""
    try:
        p = float(profit)
        d = float(drawdown)
    except (TypeError, ValueError):
        return None
    if p != p or d != d:  # NaN
        return None
    if d <= _RR_EPS:
        return None
    if p < 0:
        return None
    return p / d


def _band_label_for_premium(ltp: float | None) -> str | None:
    if ltp is None:
        return None
    try:
        x = float(ltp)
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    for label, lo, hi in PREMIUM_BANDS:
        if lo is not None and x < float(lo):
            continue
        if hi is not None and x >= float(hi):
            continue
        return label
    return None


def _hit_rate_row(
    *,
    n: int,
    target_hits: int,
    rr_1_1: int = 0,
    rr_2_3: int = 0,
    rr_1_2: int = 0,
    rr_1_3: int = 0,
    rr_1_4: int = 0,
) -> dict[str, Any]:
    return {
        "rows": n,
        "target_hit_pct": _pct(target_hits, n),
        "rr_1_1_pct": _pct(rr_1_1, n),
        "rr_2_3_pct": _pct(rr_2_3, n),
        "rr_1_2_pct": _pct(rr_1_2, n),
        "rr_1_3_pct": _pct(rr_1_3, n),
        "rr_1_4_pct": _pct(rr_1_4, n),
        "target_hits": target_hits,
        "rr_1_1": rr_1_1,
        "rr_2_3": rr_2_3,
        "rr_1_2": rr_1_2,
        "rr_1_3": rr_1_3,
        "rr_1_4": rr_1_4,
    }


def load_rr_validation_report(db_path: str) -> dict[str, Any]:
    """
    Summarize persisted RR labels and run consistency / window checks.

    Read-only — never mutates prediction_dataset.
    """
    empty: dict[str, Any] = {
        "available": False,
        "error": None,
        "total_rows": 0,
        "labeled_rows": 0,
        "summary": [],
        "consistency": {"ok": False, "message": "No prediction rows"},
        "outcome_window": {"ok": False, "message": "No prediction rows"},
        "class_balance": [],
        "reward_risk": {
            "avg": None,
            "median": None,
            "p95": None,
            "n": 0,
        },
        "premium_bands": [],
        "trading_days": [],
        "target_column": None,
        "horizon_sec": None,
    }
    try:
        with ModelLabStore(db_path) as store:
            store.ensure_prediction_schema()
            cols = set(store._prediction_table_columns())
            needed = {
                "target_reached",
                "rr_1_2_hit",
                "rr_1_3_hit",
                "rr_1_4_hit",
                "maximum_profit",
                "maximum_drawdown",
            }
            has_extended_rr = "rr_1_1_hit" in cols and "rr_2_3_hit" in cols
            if not needed.issubset(cols):
                missing = sorted(needed - cols)
                return {
                    **empty,
                    "error": f"Missing columns: {', '.join(missing)}",
                }

            tables = {
                str(r[0])
                for r in store.conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "prediction_dataset" not in tables:
                return empty

            row = store.conn.execute(
                """
                SELECT
                    COUNT(*) AS total_rows,
                    SUM(CASE WHEN target_reached = 1 THEN 1 ELSE 0 END) AS target_hits,
                    SUM(CASE WHEN rr_1_2_hit = 1 THEN 1 ELSE 0 END) AS rr_1_2,
                    SUM(CASE WHEN rr_1_3_hit = 1 THEN 1 ELSE 0 END) AS rr_1_3,
                    SUM(CASE WHEN rr_1_4_hit = 1 THEN 1 ELSE 0 END) AS rr_1_4,
                    SUM(CASE WHEN rr_1_2_hit IS NOT NULL THEN 1 ELSE 0 END) AS labeled_rows,
                    SUM(CASE WHEN rr_1_2_hit = 0 THEN 1 ELSE 0 END) AS rr_1_2_neg,
                    SUM(CASE WHEN rr_1_3_hit = 0 THEN 1 ELSE 0 END) AS rr_1_3_neg,
                    SUM(CASE WHEN rr_1_4_hit = 0 THEN 1 ELSE 0 END) AS rr_1_4_neg,
                    SUM(CASE WHEN rr_1_1_hit = 1 THEN 1 ELSE 0 END) AS rr_1_1,
                    SUM(CASE WHEN rr_2_3_hit = 1 THEN 1 ELSE 0 END) AS rr_2_3,
                    SUM(CASE WHEN rr_1_1_hit = 0 THEN 1 ELSE 0 END) AS rr_1_1_neg,
                    SUM(CASE WHEN rr_2_3_hit = 0 THEN 1 ELSE 0 END) AS rr_2_3_neg,
                    SUM(CASE WHEN rr_1_1_hit IS NOT NULL THEN 1 ELSE 0 END) AS rr_1_1_labeled
                FROM prediction_dataset
                """
            ).fetchone()
            total = int(row[0] or 0) if row else 0
            target_hits = int(row[1] or 0) if row else 0
            rr_1_2 = int(row[2] or 0) if row else 0
            rr_1_3 = int(row[3] or 0) if row else 0
            rr_1_4 = int(row[4] or 0) if row else 0
            labeled_rows = int(row[5] or 0) if row else 0
            rr_1_2_neg = int(row[6] or 0) if row else 0
            rr_1_3_neg = int(row[7] or 0) if row else 0
            rr_1_4_neg = int(row[8] or 0) if row else 0
            rr_1_1 = int(row[9] or 0) if row else 0
            rr_2_3 = int(row[10] or 0) if row else 0
            rr_1_1_neg = int(row[11] or 0) if row else 0
            rr_2_3_neg = int(row[12] or 0) if row else 0
            rr_1_1_labeled = int(row[13] or 0) if row else 0
            use_extended = bool(has_extended_rr and rr_1_1_labeled > 0)

            pred_sum = store.read_prediction_summary() or {}
            target_col = str(pred_sum.get("target_column") or "").strip() or None
            try:
                horizon = horizon_sec_from_target(target_col) if target_col else None
            except ValueError:
                horizon = None

            if total <= 0:
                return {
                    **empty,
                    "target_column": target_col,
                    "horizon_sec": horizon,
                }

            summary = [
                {
                    "metric": "Total Predictions",
                    "count": total,
                    "pct": 100.0 if total else None,
                },
                {
                    "metric": "Target Hit",
                    "count": target_hits,
                    "pct": _pct(target_hits, total),
                },
            ]
            if use_extended:
                summary.extend(
                    [
                        {
                            "metric": "RR 1:1 Hit",
                            "count": rr_1_1,
                            "pct": _pct(rr_1_1, total),
                        },
                        {
                            "metric": "RR 2:3 Hit",
                            "count": rr_2_3,
                            "pct": _pct(rr_2_3, total),
                        },
                    ]
                )
            summary.extend(
                [
                    {
                        "metric": "RR 1:2 Hit",
                        "count": rr_1_2,
                        "pct": _pct(rr_1_2, total),
                    },
                    {
                        "metric": "RR 1:3 Hit",
                        "count": rr_1_3,
                        "pct": _pct(rr_1_3, total),
                    },
                    {
                        "metric": "RR 1:4 Hit",
                        "count": rr_1_4,
                        "pct": _pct(rr_1_4, total),
                    },
                ]
            )

            consistency = _check_label_consistency(
                target_hits=target_hits,
                rr_1_1=rr_1_1 if use_extended else None,
                rr_2_3=rr_2_3 if use_extended else None,
                rr_1_2=rr_1_2,
                rr_1_3=rr_1_3,
                rr_1_4=rr_1_4,
            )

            # Outcome window + stored-vs-recomputed RR checks (labeled rows only).
            window_fail = 0
            recompute_fail = 0
            missing_exit = 0
            if horizon is None:
                outcome_window = {
                    "ok": False,
                    "message": (
                        "Cannot validate outcome window — "
                        "prediction summary has no resolvable target_column horizon."
                    ),
                }
            else:
                sample_sql = """
                    SELECT timestamp, exit_at, target_reached, target_reached_at,
                           maximum_profit, maximum_drawdown,
                           max_profit_at, max_drawdown_at,
                           rr_1_2_hit, rr_1_3_hit, rr_1_4_hit,
                           rr_1_1_hit, rr_2_3_hit
                    FROM prediction_dataset
                    WHERE rr_1_2_hit IS NOT NULL
                """
                for r in store.conn.execute(sample_sql).fetchall():
                    ts, exit_at, target_reached, target_at = r[0], r[1], r[2], r[3]
                    max_profit, max_dd = r[4], r[5]
                    mfe_at, mae_at = r[6], r[7]
                    stored_legacy = (r[8], r[9], r[10])
                    stored_ext = (r[11], r[12])

                    if not _exit_matches_horizon(ts=ts, exit_at=exit_at, horizon_sec=horizon):
                        if exit_at is None:
                            missing_exit += 1
                        window_fail += 1
                        continue

                    if not _window_bounds_ok(ts=ts, exit_at=exit_at, at=mfe_at):
                        window_fail += 1
                    if not _window_bounds_ok(ts=ts, exit_at=exit_at, at=mae_at):
                        window_fail += 1
                    if int(target_reached or 0) == 1 and not _window_bounds_ok(
                        ts=ts, exit_at=exit_at, at=target_at
                    ):
                        window_fail += 1

                    expected = compute_rr_hit_labels(
                        target_reached=int(target_reached) if target_reached is not None else None,
                        maximum_profit=max_profit,
                        maximum_drawdown=max_dd,
                    )
                    mismatch = (
                        stored_legacy[0] != expected["rr_1_2_hit"]
                        or stored_legacy[1] != expected["rr_1_3_hit"]
                        or stored_legacy[2] != expected["rr_1_4_hit"]
                    )
                    if use_extended and stored_ext[0] is not None and stored_ext[1] is not None:
                        mismatch = mismatch or (
                            stored_ext[0] != expected["rr_1_1_hit"]
                            or stored_ext[1] != expected["rr_2_3_hit"]
                        )
                    if mismatch:
                        recompute_fail += 1

                if labeled_rows <= 0:
                    outcome_window = {
                        "ok": False,
                        "message": "RR labels not computed yet — rebuild prediction days.",
                    }
                elif window_fail > 0 or recompute_fail > 0:
                    parts = []
                    if window_fail:
                        parts.append(f"{window_fail:,} row(s) outside horizon window")
                    if recompute_fail:
                        parts.append(f"{recompute_fail:,} row(s) mismatch stored vs recomputed RR")
                    if missing_exit:
                        parts.append(f"{missing_exit:,} missing exit_at")
                    outcome_window = {
                        "ok": False,
                        "message": "Outcome window mismatch. "
                        + "; ".join(parts)
                        + ". Labels may be invalid.",
                    }
                else:
                    outcome_window = {
                        "ok": True,
                        "message": "Outcome window validated",
                    }

            # Class balance — Positive % is the key classifier decision metric.
            class_balance = []
            balance_specs: list[tuple[str, int, int]] = []
            if use_extended:
                balance_specs.extend(
                    [
                        ("RR 1:1", rr_1_1, rr_1_1_neg),
                        ("RR 2:3", rr_2_3, rr_2_3_neg),
                    ]
                )
            balance_specs.extend(
                [
                    ("RR 1:2", rr_1_2, rr_1_2_neg),
                    ("RR 1:3", rr_1_3, rr_1_3_neg),
                    ("RR 1:4", rr_1_4, rr_1_4_neg),
                ]
            )
            for label, pos_count, neg_count in balance_specs:
                denom = pos_count + neg_count
                class_balance.append(
                    {
                        "label": label,
                        "positive": pos_count,
                        "negative": neg_count,
                        "positive_pct": _pct(pos_count, denom),
                    }
                )

            # Reward/Risk ratio distribution (profit / drawdown).
            rr_ratios: list[float] = []
            ratio_sql = """
                SELECT maximum_profit, maximum_drawdown
                FROM prediction_dataset
                WHERE maximum_profit IS NOT NULL
                  AND maximum_drawdown IS NOT NULL
            """
            for profit, dd in store.conn.execute(ratio_sql).fetchall():
                ratio = _reward_risk_ratio(profit, dd)
                if ratio is not None:
                    rr_ratios.append(float(ratio))
            rr_ratios.sort()
            reward_risk = {
                "avg": float(statistics.fmean(rr_ratios)) if rr_ratios else None,
                "median": float(statistics.median(rr_ratios)) if rr_ratios else None,
                "p95": _percentile(rr_ratios, 95.0),
                "n": len(rr_ratios),
            }

            # Premium band RR hit rates.
            has_ltp = "current_ltp" in cols
            band_acc: dict[str, dict[str, int]] = {
                label: {"n": 0, "t": 0, "r11": 0, "r23": 0, "r2": 0, "r3": 0, "r4": 0}
                for label, _, _ in PREMIUM_BANDS
            }
            if has_ltp:
                band_sql = """
                    SELECT current_ltp, target_reached,
                           rr_1_1_hit, rr_2_3_hit,
                           rr_1_2_hit, rr_1_3_hit, rr_1_4_hit
                    FROM prediction_dataset
                    WHERE rr_1_2_hit IS NOT NULL
                """
                for ltp, th, r11, r23, r2, r3, r4 in store.conn.execute(band_sql).fetchall():
                    try:
                        prem = float(ltp) if ltp is not None else None
                    except (TypeError, ValueError):
                        prem = None
                    band = _band_label_for_premium(prem)
                    if band is None or band not in band_acc:
                        continue
                    acc = band_acc[band]
                    acc["n"] += 1
                    if int(th or 0) == 1:
                        acc["t"] += 1
                    if int(r11 or 0) == 1:
                        acc["r11"] += 1
                    if int(r23 or 0) == 1:
                        acc["r23"] += 1
                    if int(r2 or 0) == 1:
                        acc["r2"] += 1
                    if int(r3 or 0) == 1:
                        acc["r3"] += 1
                    if int(r4 or 0) == 1:
                        acc["r4"] += 1

            premium_bands = []
            for label, _, _ in PREMIUM_BANDS:
                acc = band_acc[label]
                if acc["n"] <= 0:
                    continue
                rates = _hit_rate_row(
                    n=acc["n"],
                    target_hits=acc["t"],
                    rr_1_1=acc["r11"],
                    rr_2_3=acc["r23"],
                    rr_1_2=acc["r2"],
                    rr_1_3=acc["r3"],
                    rr_1_4=acc["r4"],
                )
                premium_bands.append({"band": label, **rates})

            # Trading day breakdown.
            day_rows = store.conn.execute(
                """
                SELECT
                    trading_day,
                    COUNT(*) AS n,
                    SUM(CASE WHEN target_reached = 1 THEN 1 ELSE 0 END) AS t,
                    SUM(CASE WHEN rr_1_1_hit = 1 THEN 1 ELSE 0 END) AS r11,
                    SUM(CASE WHEN rr_2_3_hit = 1 THEN 1 ELSE 0 END) AS r23,
                    SUM(CASE WHEN rr_1_2_hit = 1 THEN 1 ELSE 0 END) AS r2,
                    SUM(CASE WHEN rr_1_3_hit = 1 THEN 1 ELSE 0 END) AS r3,
                    SUM(CASE WHEN rr_1_4_hit = 1 THEN 1 ELSE 0 END) AS r4
                FROM prediction_dataset
                WHERE rr_1_2_hit IS NOT NULL
                GROUP BY trading_day
                ORDER BY trading_day
                """
            ).fetchall()
            trading_days = []
            for day, n, t, r11, r23, r2, r3, r4 in day_rows:
                rates = _hit_rate_row(
                    n=int(n or 0),
                    target_hits=int(t or 0),
                    rr_1_1=int(r11 or 0),
                    rr_2_3=int(r23 or 0),
                    rr_1_2=int(r2 or 0),
                    rr_1_3=int(r3 or 0),
                    rr_1_4=int(r4 or 0),
                )
                trading_days.append(
                    {
                        "trading_day": str(day or ""),
                        **rates,
                    }
                )

            return {
                "available": True,
                "error": None,
                "total_rows": total,
                "labeled_rows": labeled_rows,
                "summary": summary,
                "consistency": consistency,
                "outcome_window": outcome_window,
                "class_balance": class_balance,
                "reward_risk": reward_risk,
                "premium_bands": premium_bands,
                "trading_days": trading_days,
                "target_column": target_col,
                "horizon_sec": horizon,
            }
    except Exception as exc:
        return {**empty, "error": str(exc)}
