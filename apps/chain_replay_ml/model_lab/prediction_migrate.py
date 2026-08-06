"""One-shot migrations for Model Lab prediction datasets."""

from __future__ import annotations

import gc
from typing import Any, Callable

from chain_replay_ml.prediction_meta.outcomes import (
    compute_path_outcomes,
    prepare_path_outcome_timelines,
)
from chain_replay_ml.ticks import TickTimeline

from .prediction_schema import horizon_sec_from_target
from .store import ModelLabStore

ProgressFn = Callable[[dict[str, Any]], None]


def _load_day_timelines(data_dir: str, trading_day: str, *, market: str) -> dict[str, TickTimeline]:
    try:
        from chain_replay_ml.prediction_meta.builder import _load_timelines_for_day

        return _load_timelines_for_day(
            data_dir,
            trading_day,
            market=market or "NIFTY",
            step_sec=3,
        )
    except Exception:
        return {}


def backfill_time_to_target(
    data_dir: str,
    lab_db_path: str,
    *,
    on_progress: ProgressFn | None = None,
) -> dict[str, Any]:
    """
    Fill path event absolute timestamps + Time To Target + DD-before-target
    for existing rows via compute_path_outcomes.

    Updates:
      time_to_target, target_reached, target_reached_at,
      max_profit_at, max_drawdown_at, exit_at,
      time_to_max_profit, time_to_max_drawdown,
      dd_before_target, time_to_dd_before_target

    Does not change Current / Predicted / Actual / Max Profit / Max DD amounts
    unless those were already stored (MFE/MAE values stay as originally written).
    """
    def emit(**kw: Any) -> None:
        if on_progress:
            on_progress(kw)

    with ModelLabStore(lab_db_path) as store:
        store.ensure_prediction_schema()
        lab = store.read_info()
        if lab is None:
            return {"ok": False, "error": "Model Lab info not found", "updated": 0}
        cols = store._prediction_table_columns()
        needed = (
            "time_to_target",
            "target_reached",
            "target_reached_at",
            "max_profit_at",
            "max_drawdown_at",
            "exit_at",
            "time_to_max_profit",
            "time_to_max_drawdown",
            "dd_before_target",
            "time_to_dd_before_target",
        )
        missing_cols = [c for c in needed if c not in cols]
        if missing_cols:
            return {
                "ok": False,
                "error": f"Missing columns: {', '.join(missing_cols)}",
                "updated": 0,
            }

        days = [
            str(r[0])
            for r in store.conn.execute(
                """
                SELECT DISTINCT trading_day FROM prediction_dataset
                WHERE trading_day IS NOT NULL AND trading_day != ''
                ORDER BY trading_day
                """
            ).fetchall()
        ]
        pending = store.conn.execute(
            """
            SELECT COUNT(*) FROM prediction_dataset
            WHERE exit_at IS NULL
               OR time_to_target IS NULL
               OR dd_before_target IS NULL
            """
        ).fetchone()
        pending_n = int(pending[0] or 0) if pending else 0

    if pending_n <= 0:
        return {"ok": True, "updated": 0, "days": len(days), "message": "Already filled"}

    updated = 0
    missed = 0
    summary_horizon: float | None = None
    with ModelLabStore(lab_db_path) as store:
        summary = store.read_prediction_summary() or {}
        tc_sum = str(summary.get("target_column") or "").strip()
        if tc_sum:
            summary_horizon = horizon_sec_from_target(tc_sum)

    emit(
        phase="migrate",
        message=f"Backfilling path timestamps · {pending_n:,} rows",
        days_total=len(days),
        days_done=0,
        updated=0,
    )

    for i, day in enumerate(days):
        with ModelLabStore(lab_db_path) as store:
            rows = store.conn.execute(
                """
                SELECT id, timestamp, token, current_ltp, predicted_future_ltp,
                       market, target_column
                FROM prediction_dataset
                WHERE trading_day = ?
                  AND (
                        exit_at IS NULL
                     OR time_to_target IS NULL
                     OR dd_before_target IS NULL
                  )
                """,
                (day,),
            ).fetchall()

        if not rows:
            emit(days_done=i + 1, days_total=len(days), current_day=day, updated=updated)
            continue

        horizon = summary_horizon
        market = "NIFTY"
        for r in rows:
            m = str(r[5] or "").strip().upper()
            if m:
                market = m
                break

        if horizon is None:
            for r in rows:
                tc = str(r[6] or "").strip()
                if tc:
                    horizon = horizon_sec_from_target(tc)
                    break
        if horizon is None:
            raise ValueError(
                f"Cannot migrate path outcomes for {day}: "
                "no resolvable target_column / prediction horizon"
            )

        timelines = _load_day_timelines(data_dir, day, market=market)
        prepare_path_outcome_timelines(timelines)
        batch: list[tuple[Any, ...]] = []

        for row_id, ts, token, entry, predicted, _mkt, _tgt in rows:
            try:
                ts_f = float(ts)
                entry_f = float(entry) if entry is not None else None
                pred_f = float(predicted) if predicted is not None else None
            except (TypeError, ValueError):
                continue

            if entry_f is None or entry_f <= 0:
                continue

            tl = timelines.get(str(token or ""))
            path = compute_path_outcomes(
                tl,
                ts=ts_f,
                entry_ltp=entry_f,
                horizon_sec=horizon,
                predicted_ltp=pred_f,
            )

            ttt = path.get("time_to_target")
            target_at = path.get("target_reached_at")
            if ttt is not None and (
                target_at is None and isinstance(ttt, (int, float)) and float(ttt) < 0
            ):
                missed += 1

            hit_flag: int | None = None
            if ttt is not None:
                hit_flag = 1 if (target_at is not None or float(ttt) >= 0) else 0

            batch.append(
                (
                    ttt,
                    target_at,
                    hit_flag,
                    path.get("max_profit_at"),
                    path.get("max_drawdown_at"),
                    path.get("exit_at"),
                    path.get("time_to_max_profit"),
                    path.get("time_to_max_drawdown"),
                    path.get("dd_before_target"),
                    path.get("time_to_dd_before_target"),
                    int(row_id),
                )
            )

        if batch:
            with ModelLabStore(lab_db_path) as store:
                store.conn.executemany(
                    """
                    UPDATE prediction_dataset SET
                        time_to_target = ?,
                        target_reached_at = ?,
                        target_reached = ?,
                        max_profit_at = ?,
                        max_drawdown_at = ?,
                        exit_at = ?,
                        time_to_max_profit = COALESCE(?, time_to_max_profit),
                        time_to_max_drawdown = COALESCE(?, time_to_max_drawdown),
                        dd_before_target = ?,
                        time_to_dd_before_target = ?
                    WHERE id = ?
                    """,
                    batch,
                )
                store.conn.commit()
            updated += len(batch)

        del timelines
        gc.collect()
        emit(
            phase="migrate",
            message=f"Path timestamps · {day}",
            current_day=day,
            days_done=i + 1,
            days_total=len(days),
            updated=updated,
            missed=missed,
        )

    if updated > 0:
        try:
            from chain_replay_ml.model_lab.research_dashboard import (
                refresh_research_dashboard_cache,
            )

            refresh_research_dashboard_cache(lab_db_path, force=True)
        except Exception:
            pass

    return {
        "ok": True,
        "updated": updated,
        "missed": missed,
        "days": len(days),
    }
