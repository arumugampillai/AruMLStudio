"""Stop-path audit: raw ticks (reference) vs Prediction Dataset execution (~3s)."""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

_IST = ZoneInfo("Asia/Kolkata")

EXECUTION_MODEL_LABEL = "3-second Prediction Dataset"
EXECUTION_MODEL_DETAIL = (
    "The Strategy Simulator executes only on Prediction Dataset samples "
    "(typically every ~3 seconds). Entries, stop-loss, target, hold-time and exits "
    "are evaluated only on prediction samples. Raw exchange ticks are for "
    "reference/debug only and are not executable by this simulator. "
    "This is expected simulator behavior, not a bug."
)


def _num(v: Any) -> float | None:
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def _fmt_ts(ts: float | None) -> str:
    if ts is None:
        return "—"
    try:
        return datetime.fromtimestamp(float(ts), tz=_IST).strftime("%H:%M:%S.%f")[:-3] + " IST"
    except (OSError, OverflowError, ValueError):
        return str(ts)


def _position_state(ts: float, entry_ts: float, exit_ts: float) -> str:
    if ts < entry_ts - 1e-9:
        return "pre_entry"
    if abs(ts - entry_ts) <= 1e-6:
        return "entry"
    if abs(ts - exit_ts) <= 0.05:
        return "exit"
    if entry_ts < ts < exit_ts:
        return "open"
    return "flat"


def analyze_stop_path(
    *,
    raw_ticks: list[tuple[float, float]],
    sample_marks: list[tuple[float, float]],
    entry_ts: float,
    exit_ts: float,
    stop_price: float,
    exit_price: float | None,
    direction: str,
    exit_reason: str = "",
) -> dict[str, Any]:
    """
    Build separate raw-tick and prediction-sample timelines for stop audit.

    Simulator execution decisions use prediction samples only; raw ticks are reference.
    """
    direction = str(direction or "long").lower()
    exit_reason = str(exit_reason or "")

    def _breached(ltp: float) -> bool:
        if direction == "short":
            return ltp >= float(stop_price) - 1e-12
        return ltp <= float(stop_price) + 1e-12

    first_tick_breach_ts = None
    first_tick_breach_ltp = None
    for ts, ltp in raw_ticks:
        if ts < float(entry_ts):
            continue
        if _breached(ltp):
            first_tick_breach_ts = ts
            first_tick_breach_ltp = ltp
            break

    first_sample_breach_ts = None
    first_sample_breach_ltp = None
    for ts, ltp in sample_marks:
        if ts < float(entry_ts):
            continue
        if _breached(ltp):
            first_sample_breach_ts = ts
            first_sample_breach_ltp = ltp
            break

    exit_p = float(exit_price) if exit_price is not None else None
    lo = min(float(stop_price), float(exit_p if exit_p is not None else stop_price))
    hi = max(float(stop_price), float(exit_p if exit_p is not None else stop_price))
    ticks_between = 0
    for ts, ltp in raw_ticks:
        if ts < float(entry_ts) or ts > float(exit_ts):
            continue
        if lo < ltp < hi:
            ticks_between += 1

    open_ticks = [(ts, ltp) for ts, ltp in raw_ticks if float(entry_ts) <= ts <= float(exit_ts)]
    max_gap = None
    for i in range(1, len(open_ticks)):
        gap = open_ticks[i][0] - open_ticks[i - 1][0]
        max_gap = gap if max_gap is None else max(max_gap, gap)

    sample_intervals = []
    for i in range(1, len(sample_marks)):
        if sample_marks[i][0] >= float(entry_ts):
            sample_intervals.append(sample_marks[i][0] - sample_marks[i - 1][0])
    median_sample_sec = None
    if sample_intervals:
        s = sorted(sample_intervals)
        median_sample_sec = s[len(s) // 2]
    cadence = median_sample_sec if median_sample_sec is not None else 3.0

    material_through = (
        exit_p is not None
        and (
            (direction != "short" and exit_p < float(stop_price) * 0.99)
            or (direction == "short" and exit_p > float(stop_price) * 1.01)
        )
    )

    raw_crossed_before_sim = (
        first_tick_breach_ts is not None
        and (
            first_sample_breach_ts is None
            or first_tick_breach_ts + 0.05 < first_sample_breach_ts
        )
    )

    # Path classification (wording emphasizes execution model, not "miss"/"bug"
    # unless samples never hit stop while exit_reason is not stop).
    path_kind = "aligned"
    path_label = "Stop path aligned on prediction samples"
    path_detail = (
        "Raw ticks and prediction samples agree on stop timing within the "
        f"~{cadence:.1f}s Prediction Dataset execution model."
    )

    if not raw_ticks:
        path_kind = "no_ticks_in_window"
        path_label = "No raw ticks in window"
        path_detail = "No raw exchange ticks found for this token between entry−30s and exit."
    elif material_through and ticks_between == 0:
        path_kind = "genuine_gap"
        path_label = "Genuine exchange gap (no intermediate ticks)"
        path_detail = (
            f"No raw ticks with LTP between stop ₹{stop_price:.4f} and exit ₹{exit_p:.4f}. "
            "Price jumped through the stop on the tape — exchange gap risk."
        )
    elif (
        material_through
        and first_tick_breach_ts is not None
        and first_sample_breach_ts is None
        and exit_reason != "stop"
    ):
        path_kind = "stop_bug"
        path_label = "Possible stop-execution bug"
        path_detail = (
            f"Raw ticks crossed stop at {_fmt_ts(first_tick_breach_ts)} "
            f"but no prediction sample was at/beyond stop and exit_reason={exit_reason!r}."
        )
    elif material_through and raw_crossed_before_sim and first_sample_breach_ts is not None:
        path_kind = "sampled_exit_after_raw_cross"
        path_label = f"Execution Model: {EXECUTION_MODEL_LABEL}"
        path_detail = (
            f"Raw tick crossed stop at {_fmt_ts(first_tick_breach_ts)} "
            f"(₹{first_tick_breach_ltp:.4f}). Simulator exit executed on the next "
            f"prediction sample at {_fmt_ts(first_sample_breach_ts)} "
            f"(₹{first_sample_breach_ltp:.4f}). "
            f"Median prediction sample interval ≈ {cadence:.2f}s. "
            "Any additional loss between samples is expected under this execution model — not a bug."
        )
    elif material_through and ticks_between > 0:
        path_kind = "sampled_exit_print_through"
        path_label = f"Execution Model: {EXECUTION_MODEL_LABEL}"
        path_detail = (
            f"First prediction sample at/beyond stop was already ₹{first_sample_breach_ltp:.4f} "
            f"(stop ₹{stop_price:.4f}). Intermediate raw ticks ({ticks_between}) exist for "
            "reference only — the simulator does not execute on them."
        )

    # Legacy alias keys kept for older tests / callers.
    diagnosis = {
        "aligned": "ok",
        "no_ticks_in_window": "no_ticks_in_window",
        "genuine_gap": "genuine_gap",
        "stop_bug": "stop_bug",
        "sampled_exit_after_raw_cross": "sample_interval_miss",
        "sampled_exit_print_through": "sample_print_through",
    }.get(path_kind, path_kind)

    raw_rows: list[dict[str, Any]] = []
    seen_raw_cross = False
    for ts, ltp in raw_ticks:
        note = ""
        if ts >= float(entry_ts) - 1e-9 and _breached(ltp) and not seen_raw_cross:
            note = "Raw Tick Stop Cross (Reference)"
            seen_raw_cross = True
        raw_rows.append({
            "timestamp": ts,
            "time_label": _fmt_ts(ts),
            "rel_sec": round(ts - float(entry_ts), 3),
            "live_ltp": round(ltp, 4),
            "stop_price": round(float(stop_price), 4),
            "position_state": _position_state(ts, float(entry_ts), float(exit_ts)),
            "note": note,
            "executable": False,
        })

    sim_rows: list[dict[str, Any]] = []
    seen_sim_stop = False
    prev_sample_ts = None
    for ts, ltp in sample_marks:
        decision = ""
        if ts >= float(entry_ts) - 1e-9 and _breached(ltp) and not seen_sim_stop:
            decision = "Stop check (sample)"
            seen_sim_stop = True
        if exit_reason == "stop" and abs(ts - float(exit_ts)) <= 0.05:
            # Exit fills at stop price; sample LTP is trigger evidence only.
            decision = "Simulator Exit Sample (fill @ stop)"
        dt_sample = None
        if prev_sample_ts is not None:
            dt_sample = round(ts - prev_sample_ts, 3)
        prev_sample_ts = ts
        sim_rows.append({
            "timestamp": ts,
            "time_label": _fmt_ts(ts),
            "rel_sec": round(ts - float(entry_ts), 3),
            "live_ltp": round(ltp, 4),
            "stop_price": round(float(stop_price), 4),
            "position_state": _position_state(ts, float(entry_ts), float(exit_ts)),
            "decision": decision,
            "dt_from_prev_sample_sec": dt_sample,
            "executable": True,
        })

    # Backward-compatible mixed rows (prefer separate timelines in UI).
    mixed_rows: list[dict[str, Any]] = []
    for r in raw_rows:
        mixed_rows.append({
            **r,
            "source": "tick",
            "exit_trigger": (
                "RAW_TICK_STOP_CROSS_REF" if r.get("note") else ""
            ),
        })
    for r in sim_rows:
        trig = ""
        if r.get("decision") and "Exit" in str(r.get("decision")):
            trig = "SIMULATOR_EXIT_SAMPLE"
        elif r.get("decision") == "Stop check (sample)":
            trig = "SIM_STOP_CHECK"
        mixed_rows.append({
            **{k: v for k, v in r.items() if k != "decision"},
            "source": "sim_sample",
            "exit_trigger": trig,
            "note": r.get("decision") or "",
        })
    mixed_rows.sort(key=lambda r: (float(r["timestamp"]), 0 if r["source"] == "tick" else 1))

    return {
        "raw_tick_rows": raw_rows,
        "sim_sample_rows": sim_rows,
        "rows": mixed_rows,  # legacy
        "path_kind": path_kind,
        "diagnosis": diagnosis,
        "diagnosis_label": path_label,
        "diagnosis_detail": path_detail,
        "execution_model_label": EXECUTION_MODEL_LABEL,
        "execution_model_detail": EXECUTION_MODEL_DETAIL,
        "execution_type": f"Prediction Sample (~{cadence:.0f}s)",
        "raw_tick_stop_cross_ts": first_tick_breach_ts,
        "raw_tick_stop_cross_ltp": first_tick_breach_ltp,
        "raw_tick_stop_cross_label": _fmt_ts(first_tick_breach_ts),
        "simulator_exit_sample_ts": first_sample_breach_ts if exit_reason == "stop" else (
            float(exit_ts) if exit_reason == "stop" else first_sample_breach_ts
        ),
        "simulator_exit_sample_ltp": (
            first_sample_breach_ltp if first_sample_breach_ltp is not None else exit_p
        ),
        "simulator_exit_sample_label": _fmt_ts(
            first_sample_breach_ts if first_sample_breach_ts is not None else (
                float(exit_ts) if exit_reason == "stop" else None
            )
        ),
        "raw_crossed_before_sim_exit": bool(raw_crossed_before_sim and exit_reason == "stop"),
        "first_tick_stop_breach_ts": first_tick_breach_ts,
        "first_tick_stop_breach_ltp": first_tick_breach_ltp,
        "first_sample_stop_breach_ts": first_sample_breach_ts,
        "first_sample_stop_breach_ltp": first_sample_breach_ltp,
        "ticks_between_stop_and_exit": ticks_between,
        "max_tick_gap_sec_while_open": round(max_gap, 3) if max_gap is not None else None,
        "median_sample_interval_sec": round(median_sample_sec, 3) if median_sample_sec is not None else None,
        "tick_count": len(raw_ticks),
        "sample_count": len(sample_marks),
    }


def build_stop_replay(
    *,
    chart_dir: str,
    trade: dict[str, Any],
    lab_db_path: str | None = None,
    pre_entry_sec: float = 30.0,
) -> dict[str, Any]:
    """
    Build stop-path audit from entry-30s through exit.

    Raw exchange ticks are reference-only. Simulator decisions use Prediction Dataset samples.
    """
    token = str(trade.get("token") or "").strip()
    day = str(trade.get("trading_day") or "").strip()
    entry_ts = _num(trade.get("entry_ts"))
    exit_ts = _num(trade.get("exit_ts"))
    entry_price = _num(trade.get("entry_price"))
    exit_price = _num(trade.get("exit_price"))
    stop_price = _num(trade.get("stop_price"))
    stop_pct = _num(trade.get("stop_loss_pct")) or 0.0
    direction = str(trade.get("direction") or "long").lower()
    exit_reason = str(trade.get("exit_reason") or "")

    if entry_price and stop_price is None and stop_pct > 0:
        if direction == "short":
            stop_price = entry_price * (1.0 + stop_pct / 100.0)
        else:
            stop_price = entry_price * (1.0 - stop_pct / 100.0)

    empty: dict[str, Any] = {
        "ok": False,
        "error": None,
        "token": token,
        "trading_day": day,
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_price": stop_price,
        "direction": direction,
        "exit_reason": exit_reason,
        "window_start_ts": None,
        "window_end_ts": exit_ts,
        "tick_count": 0,
        "sample_count": 0,
        "rows": [],
        "raw_tick_rows": [],
        "sim_sample_rows": [],
        "path_kind": "no_data",
        "diagnosis": "no_data",
        "diagnosis_label": "No data",
        "diagnosis_detail": "Missing trade timestamps/token/day.",
        "execution_model_label": EXECUTION_MODEL_LABEL,
        "execution_model_detail": EXECUTION_MODEL_DETAIL,
        "execution_type": "Prediction Sample (~3s)",
        "raw_tick_stop_cross_ts": None,
        "raw_tick_stop_cross_ltp": None,
        "simulator_exit_sample_ts": None,
        "simulator_exit_sample_ltp": None,
        "raw_crossed_before_sim_exit": False,
        "first_tick_stop_breach_ts": None,
        "first_tick_stop_breach_ltp": None,
        "first_sample_stop_breach_ts": None,
        "first_sample_stop_breach_ltp": None,
        "ticks_between_stop_and_exit": 0,
        "max_tick_gap_sec_while_open": None,
        "median_sample_interval_sec": None,
        "tick_db_path": None,
    }

    if not token or not day or entry_ts is None or exit_ts is None or stop_price is None:
        empty["error"] = "trade missing token/day/entry_ts/exit_ts/stop_price"
        return empty

    window_start = float(entry_ts) - float(pre_entry_sec)
    window_end = float(exit_ts)
    empty["window_start_ts"] = window_start
    empty["window_end_ts"] = window_end

    from tick_data_paths import replay_db_path

    tick_db = replay_db_path(chart_dir, day)
    empty["tick_db_path"] = tick_db
    if not tick_db or not os.path.isfile(tick_db):
        empty["error"] = f"tick DB not found for {day}"
        empty["path_kind"] = "no_tick_db"
        empty["diagnosis"] = "no_tick_db"
        empty["diagnosis_label"] = "Tick DB missing"
        empty["diagnosis_detail"] = (
            f"Could not find angel_market_{day}.db under the configured tick data dir."
        )
        return empty

    from chain_replay_ml.ticks import load_tick_timelines

    with sqlite3.connect(tick_db) as conn:
        timelines = load_tick_timelines(conn, [token], window_start, window_end)
    tl = timelines.get(token)
    raw_ticks: list[tuple[float, float]] = []
    if tl is not None and tl.timestamps:
        for ts, paise in zip(tl.timestamps, tl.ltps_paise):
            raw_ticks.append((float(ts), float(paise) / 100.0))

    sample_marks: list[tuple[float, float]] = []
    if lab_db_path and os.path.isfile(lab_db_path):
        try:
            from chain_replay_ml.model_lab.store import ModelLabStore

            with ModelLabStore(lab_db_path) as store:
                store.ensure_prediction_schema()
                rows = store.conn.execute(
                    """
                    SELECT timestamp, current_ltp
                    FROM prediction_dataset
                    WHERE token = ?
                      AND trading_day = ?
                      AND timestamp >= ?
                      AND timestamp <= ?
                      AND current_ltp IS NOT NULL
                    ORDER BY timestamp ASC
                    """,
                    (token, day, window_start, window_end),
                ).fetchall()
                for r in rows:
                    ts = _num(r[0])
                    ltp = _num(r[1])
                    if ts is not None and ltp is not None:
                        sample_marks.append((ts, ltp))
        except Exception as exc:
            empty["sample_load_error"] = str(exc)

    analyzed = analyze_stop_path(
        raw_ticks=raw_ticks,
        sample_marks=sample_marks,
        entry_ts=float(entry_ts),
        exit_ts=float(exit_ts),
        stop_price=float(stop_price),
        exit_price=exit_price,
        direction=direction,
        exit_reason=exit_reason,
    )

    return {
        "ok": True,
        "error": None,
        "token": token,
        "trading_day": day,
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_price": stop_price,
        "direction": direction,
        "exit_reason": exit_reason,
        "window_start_ts": window_start,
        "window_end_ts": window_end,
        "tick_db_path": tick_db,
        **analyzed,
    }
