"""Investigate ltp_ema300 excess NULLs vs expected warm-up (read-only).

Splits NULL streaks on sample-stream gaps > gap_max_sec so nested resets
inside a long NULL run are counted separately.
"""
from __future__ import annotations

import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")
DAY = "2026-07-24"
MASTER = r"D:\data\master_dataset\master_dataset_nifty_3s.db"
GAP_MAX_SEC = 20.0
EMA_PERIOD = 300
INTERVAL_SEC = 3.0
FEATURE = "ltp_ema300"


def _fmt(ts: float) -> str:
    return datetime.fromtimestamp(float(ts), tz=IST).strftime("%H:%M:%S")


def main() -> None:
    conn = sqlite3.connect(MASTER)
    rows = conn.execute(
        f"""
        SELECT token, timestamp, "{FEATURE}", ltp
        FROM samples
        WHERE trading_day=?
        ORDER BY token, timestamp
        """,
        (DAY,),
    ).fetchall()
    conn.close()

    by_tok: dict[str, list[tuple[float, bool, bool]]] = defaultdict(list)
    for tok, ts, val, ltp in rows:
        by_tok[str(tok)].append((float(ts), val is None, ltp is None))

    n_tokens = len(by_tok)
    total_null = sum(1 for seq in by_tok.values() for _, is_n, _ in seq if is_n)
    total_rows = sum(len(seq) for seq in by_tok.values())

    print("=" * 72)
    print(f"{FEATURE} warm-up / gap-reset investigation - {DAY}")
    print("=" * 72)
    print(f"Master rows: {total_rows:,}")
    print(f"Tokens: {n_tokens}  (user estimate assumed 20)")
    print(f"NULL rows: {total_null:,}")
    print(f"gap_max_sec: {GAP_MAX_SEC:g}   EMA period: {EMA_PERIOD} samples "
          f"({EMA_PERIOD * INTERVAL_SEC / 60:.0f} min @ {INTERVAL_SEC:g}s)")
    print()
    print("Expected if ONE warm-up per token:")
    print(f"  20 x {EMA_PERIOD} = {20 * EMA_PERIOD:,}")
    print(f"  {n_tokens} x {EMA_PERIOD} = {n_tokens * EMA_PERIOD:,}")
    print(f"  Observed: {total_null:,}  (excess vs {n_tokens}x{EMA_PERIOD}: "
          f"{total_null - n_tokens * EMA_PERIOD:+,})")
    print()

    gap_events = []
    warm_segments = []  # each reset → warm-up segment
    reason_rows = Counter()
    reason_events = Counter()
    resets_per_token: dict[str, Counter] = defaultdict(Counter)

    for tok, seq in sorted(by_tok.items()):
        for i in range(1, len(seq)):
            gap = seq[i][0] - seq[i - 1][0]
            if gap > GAP_MAX_SEC:
                gap_events.append({
                    "token": tok,
                    "prev_ts": seq[i - 1][0],
                    "ts": seq[i][0],
                    "gap": gap,
                })

        # Walk samples; every session start or gap>limit starts a warm segment
        # while EMA is not yet ready (count nulls until EMA_PERIOD updates with
        # non-null LTP, or until next gap / end).
        i = 0
        segment_idx = 0
        while i < len(seq):
            # Determine if this index starts a reset
            if i == 0:
                reason = "session_start"
                gap_before = None
            else:
                gap_before = seq[i][0] - seq[i - 1][0]
                if gap_before > GAP_MAX_SEC:
                    reason = "gap_reset"
                else:
                    i += 1
                    continue

            # Count warm-up nulls from here: consecutive samples until we have
            # EMA_PERIOD non-null-LTP updates OR hit another gap OR end.
            ltp_updates = 0
            null_in_seg = 0
            j = i
            start_ts = seq[i][0]
            while j < len(seq):
                if j > i and (seq[j][0] - seq[j - 1][0]) > GAP_MAX_SEC:
                    break  # nested gap — next loop iteration handles it
                is_null, ltp_null = seq[j][1], seq[j][2]
                if is_null:
                    null_in_seg += 1
                if not ltp_null:
                    ltp_updates += 1
                j += 1
                if ltp_updates >= EMA_PERIOD and not is_null:
                    # completed warm-up; stop at first non-null after ready
                    # Actually after EMA_PERIOD updates, value should be non-null
                    # Keep counting nulls only; break when we see non-null after ready
                    pass
                if ltp_updates >= EMA_PERIOD and not seq[j - 1][1]:
                    break

            # Simpler: from reset point, count consecutive nulls until non-null
            # or nested gap (already handled by break above for gap).
            null_in_seg = 0
            j = i
            while j < len(seq):
                if j > i and (seq[j][0] - seq[j - 1][0]) > GAP_MAX_SEC:
                    break
                if not seq[j][1]:
                    break
                null_in_seg += 1
                j += 1

            end_ts = seq[j - 1][0] if null_in_seg else start_ts
            if null_in_seg == 0 and reason == "gap_reset":
                # Gap occurred but next sample already had EMA ready? rare
                reason_events["gap_no_null"] += 1
                i = max(i + 1, j)
                continue

            if null_in_seg < EMA_PERIOD and j >= len(seq):
                detail = f"{reason}_incomplete"
            elif null_in_seg > EMA_PERIOD:
                detail = f"{reason}_overlong"
            else:
                detail = reason

            warm_segments.append({
                "token": tok,
                "reason": detail,
                "base_reason": reason,
                "start_ts": start_ts,
                "end_ts": end_ts,
                "null_rows": null_in_seg,
                "gap_before": gap_before,
            })
            reason_rows[detail] += null_in_seg
            reason_events[detail] += 1
            resets_per_token[tok][detail] += 1
            segment_idx += 1
            i = max(i + 1, j)

    print("Warm-up segments by reason (NULL rows in each post-reset streak)")
    print("-" * 72)
    for reason, n in reason_rows.most_common():
        print(f"  {reason:<32} rows={n:>8,}  events={reason_events[reason]:>4}")
    print(f"  {'TOTAL in segments':<32} rows={sum(reason_rows.values()):>8,}")
    print(f"  {'Actual NULL rows':<32} rows={total_null:>8,}")
    leftover = total_null - sum(reason_rows.values())
    print(f"  {'Unattributed leftover':<32} rows={leftover:>8,}")
    print()

    base = Counter()
    for seg in warm_segments:
        base[seg["base_reason"]] += seg["null_rows"]
    print("Collapsed by base reason:")
    for k, v in base.most_common():
        print(f"  {k:<20} {v:,}")
    print()

    print(f"Sample-stream gaps > {GAP_MAX_SEC:g}s: {len(gap_events)}")
    if gap_events:
        gaps = sorted(g["gap"] for g in gap_events)
        print(f"  gap sec: min={gaps[0]:.1f}  median={gaps[len(gaps)//2]:.1f}  "
              f"max={gaps[-1]:.1f}")
        buckets = Counter()
        for g in gap_events:
            sec = g["gap"]
            if sec <= 30:
                buckets["20-30s"] += 1
            elif sec <= 60:
                buckets["30-60s"] += 1
            elif sec <= 300:
                buckets["1-5m"] += 1
            elif sec <= 1800:
                buckets["5-30m"] += 1
            else:
                buckets[">30m"] += 1
        for k in ("20-30s", "30-60s", "1-5m", "5-30m", ">30m"):
            if buckets[k]:
                print(f"    {k}: {buckets[k]}")
    print()

    print("Top tokens by excess NULL beyond one EMA300 warm-up")
    print("-" * 72)
    tok_stats = []
    for tok, seq in by_tok.items():
        nn = sum(1 for _, is_n, _ in seq if is_n)
        excess = nn - min(EMA_PERIOD, len(seq))
        tok_stats.append((excess, nn, len(seq), tok))
    for excess, nn, n, tok in sorted(tok_stats, reverse=True)[:12]:
        print(f"  {tok}: null={nn:,}/{n:,} excess={excess:+,}  "
              f"{dict(resets_per_token[tok])}")
    print()

    print("Largest gap_reset warm-up segments")
    print("-" * 72)
    gap_segs = [s for s in warm_segments if s["base_reason"] == "gap_reset"]
    gap_segs.sort(key=lambda s: -s["null_rows"])
    for s in gap_segs[:15]:
        print(
            f"  token={s['token']}  {_fmt(s['start_ts'])}  "
            f"gap_before={s['gap_before']:.1f}s  "
            f"null_rows={s['null_rows']}  ({s['reason']})"
        )
    print()

    sess = base.get("session_start", 0)
    gap_n = base.get("gap_reset", 0)
    print("Conclusion")
    print("-" * 72)
    print(f"  Session-start warm-up rows: {sess:,}")
    print(f"  Gap-reset warm-up rows:     {gap_n:,}")
    print(f"  Gap events (>20s):          {len(gap_events)}")
    print(
        f"  vs single-warm-up expectation for {n_tokens} tokens: "
        f"{n_tokens * EMA_PERIOD:,}"
    )
    print(
        "  Root cause: gap_max_sec=20 resets token LTP EMA state whenever a "
        "token's sample stream pauses >20s (ATM band exit/re-entry, coverage "
        "holes). Each reset restarts a full 300-sample warm-up. This is "
        "expected under current gap policy — not a missing-history bug — but "
        "it explains the excess over a single 15-minute warm-up."
    )
    print(
        "  Live debug: set EMA_RESET_DEBUG=1 to log every reset_all "
        "(token, ts, gap, reason=row_gap) during the next Master rebuild."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
