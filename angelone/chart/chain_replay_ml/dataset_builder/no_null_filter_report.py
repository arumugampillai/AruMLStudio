"""Diagnostic No-Null filter report for Master Dataset Builder (read-only).

Generates a text analysis of Master → ATM → LTP → No-Null stages and
per-feature NULL / exclusive-removal impact. Never modifies data, schema,
or the build pipeline.
"""

from __future__ import annotations

import sqlite3
import time
from typing import Any, Callable, Sequence

ProgressFn = Callable[[str], None]


def _pct(removed: int, before: int) -> str:
    if before <= 0:
        return "—"
    return f"{100.0 * removed / before:.2f}%"


def _fmt_stage_line(
    name: str,
    *,
    before: int,
    after: int,
) -> str:
    removed = max(before - after, 0)
    return (
        f"  {name:<28} "
        f"rows={after:>10,}  "
        f"removed={removed:>10,}  "
        f"({_pct(removed, before)})"
    )


def _ist_hms(ts: float) -> str:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return datetime.fromtimestamp(float(ts), ZoneInfo("Asia/Kolkata")).strftime(
        "%H:%M:%S"
    )


def _infer_strike_step(strikes: list[float]) -> float:
    uniq = sorted({float(s) for s in strikes if s is not None})
    diffs = [uniq[i] - uniq[i - 1] for i in range(1, len(uniq)) if uniq[i] > uniq[i - 1]]
    if not diffs:
        return 50.0
    diffs.sort()
    return float(diffs[len(diffs) // 2])


def _gap_duration_bucket(gap_sec: float) -> str:
    g = float(gap_sec)
    if g <= 30:
        return "21-30 sec"
    if g <= 60:
        return "31-60 sec"
    if g <= 120:
        return "61-120 sec"
    if g <= 600:
        return "2-10 min"
    return ">10 min"


_GAP_BUCKET_ORDER = (
    "21-30 sec",
    "31-60 sec",
    "61-120 sec",
    "2-10 min",
    ">10 min",
)


def _classify_token_disappearance(
    *,
    prev_ts: float,
    curr_ts: float,
    strike: float | None,
    prev_dist: float | None,
    curr_dist: float | None,
    mid_spot: float | None,
    other_rows_in_gap: int,
    atm_band: int,
    strike_step: float,
) -> str:
    """Best-effort reason the token left the sample stream during a gap."""
    gap = float(curr_ts) - float(prev_ts)
    # Same calendar session normally; multi-hour silence with no peers → data hole.
    if other_rows_in_gap <= 0:
        if gap >= 4 * 3600:
            return "Session boundary"
        return "Replay gap"

    band = max(0, int(atm_band))
    step = float(strike_step) if strike_step and strike_step > 0 else 50.0
    mid_dist: float | None = None
    if strike is not None and mid_spot is not None and mid_spot > 0:
        atm = round(float(mid_spot) / step) * step
        mid_dist = abs(float(strike) - atm) / step

    if mid_dist is not None and mid_dist > band:
        return f"Left ATM +/-{band} band"

    # At the band edge before/after while peers continued → ATM membership flicker.
    edge = float(band)
    if (
        (prev_dist is not None and abs(float(prev_dist)) >= edge)
        or (curr_dist is not None and abs(float(curr_dist)) >= edge)
    ):
        return f"Left ATM +/-{band} band"

    if mid_dist is not None and mid_dist <= band:
        return "Token not selected"

    return "Unknown"


def _format_ema_gap_reset_check(
    conn: sqlite3.Connection,
    *,
    where_sql: str,
    params: list[Any],
    feature: str = "ltp_ema300",
    ema_period: int = 300,
    gap_max_sec: float = 20.0,
    trading_day: str | None = None,
    atm_band: int | None = 10,
    max_gap_rows: int = 500,
) -> list[str]:
    """Attribute NULL streaks and log every gap reset with duration + reason."""
    from collections import Counter, defaultdict

    cols = {r[1] for r in conn.execute("PRAGMA table_info(samples)")}
    if feature not in cols:
        return [f"EMA gap-reset check: {feature} not in samples"]

    has_strike = "strike" in cols
    has_spot = "spot" in cols
    has_dist = "strike_distance_from_atm" in cols
    select_bits = ["token", "timestamp", f'"{feature}"']
    if has_strike:
        select_bits.append("strike")
    if has_spot:
        select_bits.append("spot")
    if has_dist:
        select_bits.append("strike_distance_from_atm")

    rows = conn.execute(
        f"SELECT {', '.join(select_bits)} FROM samples "
        f"WHERE {where_sql} ORDER BY token, timestamp",
        list(params or []),
    ).fetchall()

    # Per-token timeline: (ts, is_null, strike, spot, dist)
    by_tok: dict[str, list[tuple[float, bool, float | None, float | None, float | None]]] = (
        defaultdict(list)
    )
    all_strikes: list[float] = []
    # Global timeline for peer presence / mid-gap spot (same filtered set).
    global_ts: list[tuple[float, float | None]] = []  # (ts, spot)

    for row in rows:
        tok = str(row[0])
        ts = float(row[1])
        is_null = row[2] is None
        idx = 3
        strike = float(row[idx]) if has_strike and row[idx] is not None else None
        if has_strike:
            idx += 1
        spot = float(row[idx]) if has_spot and row[idx] is not None else None
        if has_spot:
            idx += 1
        dist = float(row[idx]) if has_dist and row[idx] is not None else None
        if strike is not None:
            all_strikes.append(strike)
        by_tok[tok].append((ts, is_null, strike, spot, dist))
        global_ts.append((ts, spot))

    global_ts.sort(key=lambda x: x[0])
    global_times = [t for t, _ in global_ts]
    strike_step = _infer_strike_step(all_strikes)
    band = int(atm_band) if atm_band is not None and int(atm_band) >= 0 else 10

    total_null = sum(1 for seq in by_tok.values() for _, n, *_ in seq if n)
    if total_null <= 0:
        return [
            f"EMA gap-reset check ({feature}): 0 NULLs — nothing to attribute"
        ]

    def _peer_count_and_mid_spot(lo: float, hi: float) -> tuple[int, float | None]:
        """Rows with lo < ts < hi and a spot near the midpoint."""
        import bisect

        left = bisect.bisect_right(global_times, lo)
        right = bisect.bisect_left(global_times, hi)
        n = max(0, right - left)
        if n <= 0:
            return 0, None
        mid = (lo + hi) / 2.0
        # Nearest sample index in (lo, hi) to mid.
        j = bisect.bisect_left(global_times, mid, lo=left, hi=right)
        best_spot: float | None = None
        best_d = float("inf")
        for k in (j - 1, j, j + 1):
            if left <= k < right:
                ts_k, sp = global_ts[k]
                if sp is None:
                    continue
                d = abs(ts_k - mid)
                if d < best_d:
                    best_d = d
                    best_spot = sp
        if best_spot is None:
            for k in range(left, right):
                sp = global_ts[k][1]
                if sp is not None:
                    return n, sp
        return n, best_spot

    reason_rows: Counter[str] = Counter()
    reason_events: Counter[str] = Counter()
    disappear_counts: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()
    bucket_warmup: Counter[str] = Counter()

    gap_log: list[tuple[str, float, float, float, float, int, str]] = []
    # (token, curr_ts, prev_ts, curr_ts, gap, warmup_lost, disappear_reason)

    for tok, seq in by_tok.items():
        # NULL attribution (session_start / gap_reset)
        i = 0
        while i < len(seq):
            if i == 0:
                reset_reason = "session_start"
            elif seq[i][0] - seq[i - 1][0] > gap_max_sec:
                reset_reason = "gap_reset"
            else:
                i += 1
                continue
            j = i
            null_n = 0
            while j < len(seq):
                if j > i and (seq[j][0] - seq[j - 1][0]) > gap_max_sec:
                    break
                if not seq[j][1]:
                    break
                null_n += 1
                j += 1
            if null_n:
                reason_rows[reset_reason] += null_n
                reason_events[reset_reason] += 1
            i = max(i + 1, j)

        # Every sample-stream gap > gap_max_sec
        for k in range(1, len(seq)):
            prev = seq[k - 1]
            curr = seq[k]
            gap = curr[0] - prev[0]
            if gap <= gap_max_sec:
                continue
            # Warm-up rows lost = consecutive NULLs starting at curr until ready / next gap
            warmup = 0
            j = k
            while j < len(seq):
                if j > k and (seq[j][0] - seq[j - 1][0]) > gap_max_sec:
                    break
                if not seq[j][1]:
                    break
                warmup += 1
                j += 1

            peers, mid_spot = _peer_count_and_mid_spot(prev[0], curr[0])
            disappear = _classify_token_disappearance(
                prev_ts=prev[0],
                curr_ts=curr[0],
                strike=prev[2] if prev[2] is not None else curr[2],
                prev_dist=prev[4],
                curr_dist=curr[4],
                mid_spot=mid_spot,
                other_rows_in_gap=peers,
                atm_band=band,
                strike_step=strike_step,
            )
            disappear_counts[disappear] += 1
            bucket = _gap_duration_bucket(gap)
            bucket_counts[bucket] += 1
            bucket_warmup[bucket] += warmup
            gap_log.append(
                (tok, curr[0], prev[0], curr[0], gap, warmup, disappear)
            )

    gap_log.sort(key=lambda r: (r[1], r[0]))
    n_tok = len(by_tok)
    expected_one = n_tok * int(ema_period)
    gap_events = len(gap_log)

    lines = [
        f"EMA gap-reset check ({feature})",
        "-" * 72,
        f"Trading day: {trading_day or '(filtered set)'}",
        f"Tokens: {n_tok}   NULL rows: {total_null:,}",
        f"gap_max_sec: {gap_max_sec:g}   EMA period: {ema_period}   "
        f"ATM band: +/-{band}   strike_step: {strike_step:g}",
        f"Expected if one warm-up/token: {expected_one:,}",
        f"Excess vs one warm-up/token: {total_null - expected_one:+,}",
        f"Sample-stream gaps > {gap_max_sec:g}s (resets): {gap_events}",
        "",
        "NULL rows by controller reset reason:",
    ]
    for reason, n in reason_rows.most_common():
        lines.append(
            f"  {reason:<20} rows={n:>8,}  events={reason_events[reason]:>4}"
        )
    lines.append(
        f"  {'TOTAL':<20} rows={sum(reason_rows.values()):>8,}"
    )

    lines.append("")
    lines.append("Gap duration summary (resets only):")
    lines.append(
        f"  {'Gap Duration':<14} {'Count':>7}  {'Warm-up Rows Lost':>18}"
    )
    for label in _GAP_BUCKET_ORDER:
        c = bucket_counts.get(label, 0)
        w = bucket_warmup.get(label, 0)
        if c or w:
            lines.append(f"  {label:<14} {c:>7}  {w:>18,}")
    lines.append(
        f"  {'TOTAL':<14} {gap_events:>7}  {sum(bucket_warmup.values()):>18,}"
    )

    lines.append("")
    lines.append("Why token disappeared (best-effort):")
    if disappear_counts:
        for reason, n in disappear_counts.most_common():
            lines.append(f"  {reason:<28} {n:>5}")
    else:
        lines.append("  (no gaps above gap_max_sec)")

    lines.append("")
    lines.append(
        "Per-gap reset log "
        "(Token | Timestamp | Previous Sample | Current Sample | "
        "Gap (sec) | Reset? | Warm-up lost | Reason):"
    )
    header = (
        f"  {'Token':<10} {'Timestamp':<10} {'Prev':<10} {'Curr':<10} "
        f"{'Gap':>8} {'Reset?':<7} {'WU':>5}  Reason"
    )
    lines.append(header)
    shown = gap_log[: max(0, int(max_gap_rows))]
    for tok, _cts, prev_ts, curr_ts, gap, warmup, disappear in shown:
        lines.append(
            f"  {tok:<10} {_ist_hms(curr_ts):<10} {_ist_hms(prev_ts):<10} "
            f"{_ist_hms(curr_ts):<10} {gap:>8.0f} {'Yes':<7} {warmup:>5}  "
            f"{disappear}"
        )
    if len(gap_log) > len(shown):
        lines.append(
            f"  ... truncated {len(gap_log) - len(shown)} more gaps "
            f"(max_gap_rows={max_gap_rows})"
        )

    lines.append("")
    if reason_rows.get("gap_reset", 0) > 0:
        short = bucket_counts.get("21-30 sec", 0) + bucket_counts.get(
            "31-60 sec", 0
        )
        long_ = (
            bucket_counts.get("2-10 min", 0) + bucket_counts.get(">10 min", 0)
        )
        atm_left = sum(
            n
            for r, n in disappear_counts.items()
            if r.startswith("Left ATM")
        )
        lines.append(
            "Verdict: excess NULLs are from gap_max_sec resets (token sample "
            "stream paused > gap limit, then EMA warm-up restarted)."
        )
        if gap_events:
            pct_short = 100.0 * short / gap_events
            pct_atm = 100.0 * atm_left / gap_events
            lines.append(
                f"Policy hint: {pct_short:.0f}% of resets are <=60s; "
                f"{100.0 * long_ / gap_events:.0f}% are >=2 min; "
                f"{pct_atm:.0f}% classified as ATM-band exit. "
                "Short ATM dropouts -> consider preserving EMA across brief "
                "absences; multi-minute gaps -> reset remains appropriate."
            )
    else:
        lines.append(
            "Verdict: NULLs match session-start warm-up only (no gap resets)."
        )
    return lines


def build_no_null_filter_report_text(
    *,
    db_path: str,
    trading_day: str | None = None,
    selected_days: Sequence[str] | None = None,
    all_days: bool = False,
    token: str | None = None,
    atm_band_filter: int | None = None,
    premium_min: float | None = None,
    premium_max: float | None = None,
    delta_min: float | None = None,
    delta_max: float | None = None,
    top_n: int = 25,
    chart_dir: str | None = None,
    on_progress: ProgressFn | None = None,
) -> str:
    """Return a multi-line No-Null diagnostics report (read-only)."""
    import numpy as np
    import pandas as pd

    from .master_status import _sample_filter_where
    from .nullable_features import (
        NULLABLE_FEATURE_LIST,
        mandatory_columns_for_step2,
    )
    from .transformations.lag_ui import (
        META_SKIP_COLUMNS,
        canonical_registry_feature_names,
    )

    def _prog(msg: str) -> None:
        if on_progress is not None:
            try:
                on_progress(msg)
            except Exception:
                pass

    t0 = time.monotonic()
    lines: list[str] = [
        "=" * 72,
        "No-Null Data Filter Report (diagnostics only — no data modified)",
        "=" * 72,
    ]

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only=ON")
        db_cols = [r[1] for r in conn.execute("PRAGMA table_info(samples)")]
        if not db_cols:
            return "No-Null Data Filter Report: samples table has no columns."

        days = [str(d).strip() for d in (selected_days or []) if str(d).strip()]
        td = str(trading_day or "").strip() or None
        tok = str(token or "").strip() or None

        def _count(where: str, params: list[Any]) -> int:
            row = conn.execute(
                f"SELECT COUNT(*) FROM samples WHERE {where}",
                list(params or []),
            ).fetchone()
            return int(row[0] or 0) if row else 0

        _prog("No-Null report: counting Master rows…")
        where_master, p_master = _sample_filter_where(
            trading_day=td,
            selected_days=days or None,
            token=tok,
            all_days=all_days,
            column_names=db_cols,
        )
        n_master = _count(where_master, p_master)

        where_atm, p_atm = _sample_filter_where(
            trading_day=td,
            selected_days=days or None,
            token=tok,
            all_days=all_days,
            atm_band_filter=atm_band_filter,
            column_names=db_cols,
        )
        _prog("No-Null report: counting after ATM…")
        n_atm = _count(where_atm, p_atm)

        where_ltp, p_ltp = _sample_filter_where(
            trading_day=td,
            selected_days=days or None,
            token=tok,
            all_days=all_days,
            atm_band_filter=atm_band_filter,
            premium_min=premium_min,
            premium_max=premium_max,
            delta_min=delta_min,
            delta_max=delta_max,
            column_names=db_cols,
        )
        _prog("No-Null report: counting after LTP / Delta…")
        n_ltp = _count(where_ltp, p_ltp)

        lines.append(f"Master DB: {db_path}")
        scope_bits = []
        if all_days:
            scope_bits.append("all_days")
        elif td:
            scope_bits.append(f"day={td}")
        elif days:
            scope_bits.append(f"days={len(days)}")
        if tok:
            scope_bits.append(f"token={tok}")
        if atm_band_filter is not None:
            scope_bits.append(f"ATM±{atm_band_filter}")
        if premium_min is not None and premium_max is not None:
            scope_bits.append(f"LTP {premium_min:g}–{premium_max:g}")
        lines.append("Scope: " + (", ".join(scope_bits) if scope_bits else "(full samples)"))
        lines.append("")
        lines.append("Filter stages")
        lines.append("-" * 72)
        lines.append(_fmt_stage_line("Master dataset", before=n_master, after=n_master))
        lines.append(_fmt_stage_line("After ATM filter", before=n_master, after=n_atm))
        lines.append(_fmt_stage_line("After LTP / Delta", before=n_atm, after=n_ltp))

        if n_ltp <= 0:
            lines.append(_fmt_stage_line("After No-Null Step 2", before=n_ltp, after=0))
            lines.append("")
            lines.append("Final surviving rows: 0")
            lines.append("(No rows remain after ATM/LTP filters — No-Null not run.)")
            lines.append(f"Elapsed: {time.monotonic() - t0:.1f}s")
            lines.append("=" * 72)
            return "\n".join(lines)

        registry_all = list(canonical_registry_feature_names())
        registry = [c for c in registry_all if c in db_cols]
        missing = [c for c in registry_all if c not in db_cols]

        # Step 1 on LTP-filtered set via SQL (empty columns)
        _prog(
            f"No-Null report: Step 1 scanning {len(registry)} Registry features…"
        )
        from .non_null_filter import discover_kept_columns_step1

        nn_scope = [
            c
            for c in db_cols
            if c in META_SKIP_COLUMNS or c in set(registry)
        ]
        kept_step1, dropped_step1 = discover_kept_columns_step1(
            conn, nn_scope, where_ltp, p_ltp
        )
        kept_registry = [c for c in registry if c in set(kept_step1)]
        mandatory = mandatory_columns_for_step2(kept_registry)
        nullable_present = [c for c in NULLABLE_FEATURE_LIST if c in kept_registry]

        if mandatory:
            null_sql = " AND ".join(f'"{c}" IS NOT NULL' for c in mandatory)
            where_nn = f"({where_ltp}) AND ({null_sql})"
            _prog("No-Null report: counting after No-Null Step 2…")
            n_nn = _count(where_nn, p_ltp)
        else:
            n_nn = n_ltp

        lines.append(
            _fmt_stage_line("After No-Null Step 2", before=n_ltp, after=n_nn)
        )
        lines.append(
            f"  Step 1 empty columns dropped: {len(dropped_step1)}"
            + (f" ({', '.join(dropped_step1[:12])}"
               + ("…" if len(dropped_step1) > 12 else "")
               + ")" if dropped_step1 else "")
        )
        if nullable_present:
            lines.append(
                "  Nullable Feature List ignored in Step 2: "
                + ", ".join(nullable_present)
            )
        lines.append("")
        lines.append(
            f"Final surviving rows: {n_nn:,}  "
            f"(of {n_master:,} Master · {_pct(n_master - n_nn, n_master)} total removed)"
        )
        lines.append("")

        # Per-feature attribution on LTP-filtered partition
        _prog(
            f"No-Null report: loading {len(mandatory)} mandatory columns "
            f"for NULL attribution ({n_ltp:,} rows)…"
        )
        load_cols = list(dict.fromkeys([*kept_registry]))
        # Prefer mandatory + nullable registry for null listing
        report_feats = list(
            dict.fromkeys([*kept_registry, *[c for c in dropped_step1 if c in registry]])
        )
        load_cols = [c for c in report_feats if c in db_cols]
        if not load_cols:
            lines.append("(No Registry feature columns to attribute.)")
            lines.append(f"Elapsed: {time.monotonic() - t0:.1f}s")
            lines.append("=" * 72)
            return "\n".join(lines)

        col_sql = ", ".join(f'"{c}"' for c in load_cols)
        df = pd.read_sql_query(
            f"SELECT {col_sql} FROM samples WHERE {where_ltp}",
            conn,
            params=list(p_ltp),
        )
        n_rows = len(df)
        _prog("No-Null report: computing exclusive removals…")

        feature_rows: list[dict[str, Any]] = []
        mand_in_df = [c for c in mandatory if c in df.columns]
        if mand_in_df:
            mat = np.column_stack([df[c].notna().to_numpy() for c in mand_in_df])
            complete = mat.all(axis=1)
            baseline = int(complete.sum())
        else:
            mat = None
            complete = np.ones(n_rows, dtype=bool)
            baseline = n_rows

        for feat in load_cols:
            n_null = int(df[feat].isna().sum())
            if n_null <= 0 and feat not in dropped_step1:
                continue
            exclusive = 0
            marginal = 0
            on_list = feat in NULLABLE_FEATURE_LIST
            drop100 = feat in dropped_step1 or n_null >= n_rows
            if (
                not on_list
                and not drop100
                and mat is not None
                and feat in mand_in_df
            ):
                col_idx = mand_in_df.index(feat)
                feat_null = ~mat[:, col_idx]
                if len(mand_in_df) > 1:
                    others_ok = mat[
                        :, np.arange(len(mand_in_df)) != col_idx
                    ].all(axis=1)
                else:
                    others_ok = np.ones(n_rows, dtype=bool)
                exclusive = int((feat_null & others_ok).sum())
                without = mat[
                    :, np.arange(len(mand_in_df)) != col_idx
                ].all(axis=1)
                marginal = int(without.sum()) - baseline
            feature_rows.append(
                {
                    "feature": feat,
                    "null_rows": n_null if not drop100 else n_rows,
                    "exclusive": exclusive,
                    "marginal": marginal,
                    "on_nullable_list": on_list,
                    "step1_dropped": drop100,
                }
            )

        feature_rows.sort(
            key=lambda r: (
                -int(r["exclusive"]),
                -int(r["marginal"]),
                -int(r["null_rows"]),
                r["feature"],
            )
        )

        with_null = [r for r in feature_rows if r["null_rows"] > 0]
        lines.append(
            f"Features containing NULL values: {len(with_null)} "
            f"(of {len(registry)} Registry present"
            + (f"; {len(missing)} missing from DB" if missing else "")
            + ")"
        )
        lines.append("-" * 72)
        lines.append(
            f"{'Feature':<42} {'NULL rows':>10} {'Exclusive':>10} "
            f"{'Marginal':>10} {'Flags':<16}"
        )
        for r in with_null:
            flags = []
            if r["on_nullable_list"]:
                flags.append("nullable")
            if r["step1_dropped"]:
                flags.append("step1-drop")
            lines.append(
                f"{r['feature']:<42} {r['null_rows']:>10,} {r['exclusive']:>10,} "
                f"{r['marginal']:>10,} {','.join(flags) or '—':<16}"
            )

        lines.append("")
        lines.append(f"Top row-removing features (by exclusive impact, top {top_n})")
        lines.append("-" * 72)
        top = [r for r in with_null if r["exclusive"] > 0][:top_n]
        if not top:
            lines.append("  (no exclusive killers — removals are overlapping NULLs)")
            # Still show top by null count among mandatory
            for r in with_null[: min(top_n, 15)]:
                if r["on_nullable_list"] or r["step1_dropped"]:
                    continue
                lines.append(
                    f"  {r['feature']:<40} null={r['null_rows']:,} "
                    f"(overlap only)"
                )
        else:
            for i, r in enumerate(top, 1):
                lines.append(
                    f"  {i:>2}. {r['feature']:<38} "
                    f"exclusive={r['exclusive']:,}  "
                    f"null={r['null_rows']:,}  "
                    f"if-nullable=+{r['marginal']:,}"
                )

        lines.append("")
        lines.append(f"Final surviving row count: {n_nn:,}")

        # Historical bar check when a single day is in scope
        hist_day = td
        if not hist_day and len(days) == 1:
            hist_day = days[0]
        if hist_day:
            _prog("No-Null report: Historical Bar Check…")
            try:
                from .historic_spot_ema_context import format_historic_bar_check

                lines.append("")
                lines.extend(
                    format_historic_bar_check(hist_day, chart_dir=chart_dir)
                )
            except Exception as exc:
                lines.append("")
                lines.append(f"Historical Bar Check failed: {exc}")

            _prog("No-Null report: EMA gap-reset check (ltp_ema300)…")
            try:
                lines.append("")
                lines.extend(
                    _format_ema_gap_reset_check(
                        conn,
                        where_sql=where_ltp,
                        params=p_ltp,
                        feature="ltp_ema300",
                        ema_period=300,
                        gap_max_sec=20.0,
                        trading_day=hist_day,
                        atm_band=(
                            int(atm_band_filter)
                            if atm_band_filter is not None
                            else 10
                        ),
                    )
                )
            except Exception as exc:
                lines.append("")
                lines.append(f"EMA gap-reset check failed: {exc}")

        lines.append("")
        lines.append(f"Elapsed: {time.monotonic() - t0:.1f}s")
        lines.append("=" * 72)
        return "\n".join(lines)
    finally:
        conn.close()


__all__ = ["build_no_null_filter_report_text"]
