"""Compare models for one replay trading day from SQLite session cache (no live scoring)."""

from __future__ import annotations

import glob
import os
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable

from chain_replay_ml.recompute_2_1_ratio import (
    _outcome_summary,
    _trade_net_pnl_rs,
    run_zero_brokerage_simulation,
    simulate_positions,
)
from chain_replay_ml.replay_session_store import (
    ARTIFACT_MODEL_AUDIT,
    ARTIFACT_POSITION_LIMITS,
    ARTIFACT_SCORING_COVERAGE,
    ARTIFACT_SUMMARY,
    ARTIFACT_TRADE_REPORT,
    KNOWN_ARTIFACTS,
    ensure_replay_session,
    list_artifacts,
    list_replay_sessions_for_day,
    list_replay_sessions_for_model,
    load_artifact_json,
    save_artifact_json,
)
from chain_replay_ml.training.registry import get_trained_model, list_trained_models

_POSITION_LIMIT_VALUES = (1, 2, 3, 5, 10, 30, 0)

_POSITION_LABELS = {
    0: "Unconstrained",
    1: "1 Position",
    2: "2 Positions",
    3: "3 Positions",
    5: "5 Positions",
    10: "10 Positions",
    30: "30 Positions",
}

_MIN_CHAIN_TICKS_MODEL_DAY = 500_000


def position_limit_label(position_limit: int) -> str:
    n = int(position_limit)
    return _POSITION_LABELS.get(n, f"{n} Positions" if n > 0 else "Unconstrained")


_STEP_LABELS = {
    "resolve_expiry": "Resolving expiry",
    "scoring_coverage": "Checking scoring coverage",
    "load_day_context": "Loading day context",
    "build_day_rows": "Building feature rows",
    "to_dataframe": "Converting to dataframe",
    "model_predict": "Running model predict",
    "build_scored_frame": "Building scored frame",
    "simulate_trades": "Simulating trades",
    "model_audit": "Building model audit",
    "persist_session": "Saving replay session",
}


def _emit_progress(
    on_progress: Callable[[dict[str, Any]], None] | None,
    payload: dict[str, Any],
) -> None:
    if on_progress:
        on_progress(payload)


def _step_message(step: str, *, trading_day: str = "", detail: str = "") -> str:
    label = _STEP_LABELS.get(step, step.replace("_", " ").title())
    day = str(trading_day or "").strip()
    if day and detail:
        return f"{day} · {label} · {detail}"
    if day:
        return f"{day} · {label}"
    return label if not detail else f"{label} · {detail}"


def _model_validation_metrics(
    model_row: dict[str, Any] | None,
) -> tuple[float | None, float | None, float | None]:
    if not model_row:
        return None, None, None
    prod = dict(model_row.get("production_metrics") or {})
    metrics = dict(model_row.get("metrics") or {})
    rmse = prod.get("rmse") if prod.get("rmse") is not None else metrics.get("rmse")
    mae = prod.get("mae") if prod.get("mae") is not None else metrics.get("mae")
    composite = (
        prod.get("composite_score")
        if prod.get("composite_score") is not None
        else metrics.get("composite_score")
    )
    return (
        round(float(rmse), 4) if rmse is not None else None,
        round(float(mae), 4) if mae is not None else None,
        round(float(composite), 4) if composite is not None else None,
    )


def _registry_size_fields(registry_row: dict[str, Any] | None) -> dict[str, Any]:
    if not registry_row:
        return {"tds": None, "fc": None}
    tds = registry_row.get("rows")
    fc = registry_row.get("feature_count")
    return {
        "tds": int(tds) if tds is not None else None,
        "fc": int(fc) if fc is not None else None,
    }


def _normalize_trades(trades: list[dict[str, Any]], trading_day: str) -> list[dict[str, Any]]:
    day = str(trading_day or "").strip()
    return [{**t, "fold_date": t.get("fold_date") or day} for t in trades]


_TIMEOUT_FLAT_EPS_RS = 1.0


def _outcome_wltp_counts(trades: list[dict[str, Any]], *, qty: int = 65) -> tuple[int, int, int, int, int]:
    """Match Report exit audit: target, SL, timeout split by net P&L."""
    wins = losses = timeout_profit = timeout_loss = timeout_flat = 0
    for t in trades:
        outcome = t.get("outcome_type")
        if outcome == "target":
            wins += 1
        elif outcome == "sl":
            losses += 1
        elif outcome == "timeout":
            net = _trade_net_pnl_rs(t, qty)
            if net > _TIMEOUT_FLAT_EPS_RS:
                timeout_profit += 1
            elif net < -_TIMEOUT_FLAT_EPS_RS:
                timeout_loss += 1
            else:
                timeout_flat += 1
    return wins, losses, timeout_profit, timeout_loss, timeout_flat


def _format_wltp(
    wins: int,
    losses: int,
    timeout_profit: int,
    timeout_loss: int,
    timeout_flat: int = 0,
) -> str:
    base = f"{wins}W - {losses}L - {timeout_profit}TP - {timeout_loss}TL"
    if timeout_flat > 0:
        return f"{base} - {timeout_flat}F"
    return base


def _trade_day_metrics(entered_trades: list[dict[str, Any]], *, qty: int = 65) -> dict[str, Any]:
    summary = _outcome_summary(entered_trades, qty=qty)
    n = int(summary.get("count") or 0)
    wins, losses, timeout_profit, timeout_loss, timeout_flat = _outcome_wltp_counts(entered_trades, qty=qty)
    timeouts = timeout_profit + timeout_loss + timeout_flat
    if n <= 0:
        return {
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "timeouts": 0,
            "timeout_profit": 0,
            "timeout_loss": 0,
            "timeout_flat": 0,
            "wl_t": "0W - 0L - 0TP - 0TL",
            "win_pct": None,
            "net_pnl": 0.0,
            "pf": None,
            "max_capital": 0.0,
            "max_drawdown": 0.0,
            "max_dd_pct": None,
            "dd": 0.0,
            "dd_pct": 0.0,
            "avg_score": None,
            "target_pct": None,
        }

    effective_wins = wins + timeout_profit
    scores = [float(t.get("score") or 0.0) for t in entered_trades if t.get("score") is not None]
    sim = run_zero_brokerage_simulation(entered_trades, qty=qty)
    peak_capital = round(float(sim.get("peak_capital") or 0.0), 2)
    max_dd = round(float(sim.get("max_dd_rs") or 0.0), 2)
    max_dd_pct = round(max_dd / peak_capital * 100.0, 2) if peak_capital > 0 else None
    return {
        "trades": n,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "timeout_profit": timeout_profit,
        "timeout_loss": timeout_loss,
        "timeout_flat": timeout_flat,
        "wl_t": _format_wltp(wins, losses, timeout_profit, timeout_loss, timeout_flat),
        "win_pct": round(effective_wins / n * 100.0, 2),
        "net_pnl": round(float(summary.get("net_pnl") or 0.0), 2),
        "pf": summary.get("pf"),
        "max_capital": peak_capital,
        "max_drawdown": max_dd,
        "max_dd_pct": max_dd_pct,
        "dd": max_dd,
        "dd_pct": max_dd_pct if max_dd_pct is not None else round(float(sim.get("max_dd_pct") or 0.0), 2),
        "avg_score": round(sum(scores) / len(scores), 3) if scores else None,
        "target_pct": summary.get("target_pct"),
    }


def build_position_limit_snapshot(
    candidates: list[dict[str, Any]],
    *,
    trading_day: str = "",
    limits: tuple[int, ...] = _POSITION_LIMIT_VALUES,
    qty: int = 65,
) -> list[dict[str, Any]]:
    """Same logic as Report tab position-limit table (simulate from all ML signals)."""
    rows: list[dict[str, Any]] = []
    for lim in limits:
        concurrent = int(lim) if int(lim) > 0 else 9999
        entered = simulate_positions(candidates, concurrent)
        metrics = _trade_day_metrics(_normalize_trades(entered, trading_day), qty=qty)
        rows.append({
            "position_limit": int(lim),
            "limit": position_limit_label(int(lim)),
            **metrics,
        })
    return rows


def _metrics_for_position_limit(
    candidates: list[dict[str, Any]],
    *,
    position_limit: int,
    trading_day: str,
    snapshot: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    pos = int(position_limit)
    if candidates:
        concurrent = pos if pos > 0 else 9999
        entered = simulate_positions(candidates, concurrent)
        return {
            "position_limit": pos,
            "limit": position_limit_label(pos),
            **_trade_day_metrics(_normalize_trades(entered, trading_day)),
        }
    if snapshot:
        for row in snapshot:
            if int(row.get("position_limit", -1)) == pos:
                return dict(row)
    return None


def _load_cached_candidates(
    data_dir: str,
    session_id: str,
    trading_day: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]] | None]:
    trade_doc = load_artifact_json(data_dir, session_id, ARTIFACT_TRADE_REPORT) or {}
    candidates = list(trade_doc.get("trades") or [])
    snapshot_doc = load_artifact_json(data_dir, session_id, ARTIFACT_POSITION_LIMITS)
    snapshot = list(snapshot_doc.get("rows") or []) if isinstance(snapshot_doc, dict) else None
    if candidates:
        return candidates, snapshot
    return [], snapshot


def _row_from_cached_session(
    data_dir: str,
    session: dict[str, Any],
    *,
    position_limit: int,
    registry_row: dict[str, Any] | None,
) -> dict[str, Any]:
    name = str(session.get("model_name") or "").strip()
    trading_day = str(session.get("trading_day") or "").strip()
    rmse, mae, composite = _model_validation_metrics(registry_row)
    session_id = str(session.get("session_id") or "")
    row: dict[str, Any] = {
        "model_name": name,
        "algorithm": (registry_row or {}).get("algorithm"),
        "rmse": rmse,
        "mae": mae,
        "composite": composite,
        **_registry_size_fields(registry_row),
        "trades": None,
        "win_pct": None,
        "net_pnl": None,
        "pf": None,
        "dd": None,
        "dd_pct": None,
        "avg_score": None,
        "target_pct": None,
        "status": "no_cache",
        "cache_state": None,
        "cached_position_limit": None,
        "source": "sqlite_cache",
        "error": None,
    }

    artifacts = list_artifacts(data_dir, session_id)
    ready_names = {a["artifact_name"] for a in artifacts if a.get("status") == "ready"}
    if ARTIFACT_TRADE_REPORT not in ready_names:
        row["status"] = "no_cache"
        row["error"] = "No cached trade report — open this model in replay first"
        return row

    candidates, snapshot = _load_cached_candidates(data_dir, session_id, trading_day)
    if not candidates and not snapshot:
        row["status"] = "no_cache"
        row["error"] = "Cached trade report is empty"
        return row

    metrics = _metrics_for_position_limit(
        candidates,
        position_limit=int(position_limit),
        trading_day=trading_day,
        snapshot=snapshot,
    )
    if not metrics:
        row["status"] = "no_data"
        row["error"] = "No metrics for this position limit"
        return row

    row.update({
        k: metrics.get(k)
        for k in (
            "trades", "wins", "losses", "timeouts", "timeout_profit", "timeout_loss", "timeout_flat", "wl_t",
            "win_pct", "net_pnl", "pf",
            "max_capital", "max_drawdown", "max_dd_pct", "dd", "dd_pct",
            "avg_score", "target_pct",
        )
    })
    row["status"] = "cached"
    row["cache_state"] = "ready"
    row["error"] = None
    return row


def compare_models_for_trading_day(
    data_dir: str,
    *,
    date_str: str,
    underlying: str = "NIFTY",
    expiry: str | None = None,
    position_limit: int = 1,
    model_names: list[str] | None = None,
    include_registry_without_cache: bool = True,
) -> dict[str, Any]:
    """Read cached replay sessions from SQLite — no model scoring."""
    t0 = time.perf_counter()
    pos = int(position_limit)
    ul = str(underlying or "NIFTY").strip()
    exp = str(expiry or "").strip() or None

    sessions = list_replay_sessions_for_day(
        data_dir,
        trading_day=str(date_str).strip(),
        underlying=ul,
        expiry=exp,
    )
    if model_names:
        wanted = {str(n).strip() for n in model_names if str(n).strip()}
        sessions = [s for s in sessions if str(s.get("model_name") or "") in wanted]

    session_by_name = {str(s.get("model_name") or ""): s for s in sessions}
    rows: list[dict[str, Any]] = []
    for session in sessions:
        name = str(session.get("model_name") or "").strip()
        if not name:
            continue
        registry_row = get_trained_model(data_dir, name)
        rows.append(_row_from_cached_session(
            data_dir,
            session,
            position_limit=pos,
            registry_row=registry_row,
        ))

    if include_registry_without_cache:
        cached_names = set(session_by_name)
        for model in list_trained_models(data_dir, lightweight=True):
            if str(model.get("status") or "").lower() != "ready":
                continue
            name = str(model.get("model_name") or "").strip()
            if not name or name in cached_names:
                continue
            if model_names and name not in set(model_names):
                continue
            rmse, mae, composite = _model_validation_metrics(model)
            rows.append({
                "model_name": name,
                "algorithm": model.get("algorithm"),
                "rmse": rmse,
                "mae": mae,
                "composite": composite,
                **_registry_size_fields(get_trained_model(data_dir, name)),
                "trades": None,
                "win_pct": None,
                "net_pnl": None,
                "pf": None,
                "dd": None,
                "dd_pct": None,
                "avg_score": None,
                "target_pct": None,
                "status": "not_cached",
                "cache_state": None,
                "cached_position_limit": None,
                "source": "registry",
                "error": "Not in replay cache for this day — open in replay to populate",
            })

    def _sort_key(r: dict[str, Any]) -> tuple:
        status_rank = {"cached": 0, "partial": 1, "limit_mismatch": 2, "no_cache": 3, "not_cached": 4}
        pnl = r.get("net_pnl")
        return (
            status_rank.get(str(r.get("status") or ""), 9),
            -(float(pnl) if pnl is not None else float("-inf")),
            str(r.get("model_name") or ""),
        )

    rows.sort(key=_sort_key)
    cached_count = sum(1 for r in rows if r.get("status") == "cached")
    elapsed = round(time.perf_counter() - t0, 3)
    return {
        "trading_day": str(date_str).strip(),
        "underlying": ul,
        "expiry": exp,
        "expiry_requested": exp,
        "position_limit": pos,
        "position_label": position_limit_label(pos),
        "rows": rows,
        "model_count": len(rows),
        "cached_count": cached_count,
        "session_count": len(sessions),
        "source": "sqlite_cache",
        "timing_sec": elapsed,
    }


_REPLAY_DB_DAY_RE = re.compile(r"^angel_market_(\d{4}-\d{2}-\d{2})\.db$")


def _replay_db_search_dirs(data_dir: str) -> list[str]:
    dirs = [data_dir, os.path.join(data_dir, "old")]
    return [d for d in dirs if os.path.isdir(d)]


def list_trading_days_with_tick_db(data_dir: str) -> list[str]:
    """Sorted trading days (desc) that have an angel_market_<day>.db with ticks."""
    import sqlite3

    days: set[str] = set()
    for base in _replay_db_search_dirs(data_dir):
        pattern = os.path.join(base, "angel_market_*.db")
        for path in glob.glob(pattern):
            base_name = os.path.basename(path)
            m = _REPLAY_DB_DAY_RE.match(base_name)
            if not m:
                continue
            day = m.group(1)
            try:
                if os.path.getsize(path) <= 0:
                    continue
                conn = sqlite3.connect(path)
                try:
                    row = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='ticks'"
                    ).fetchone()
                    if not row:
                        continue
                    count = conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
                    if int(count or 0) > 0:
                        days.add(day)
                finally:
                    conn.close()
            except (OSError, sqlite3.Error):
                continue
    return sorted(days, reverse=True)


def chain_tick_count_for_day(data_dir: str, trading_day: str, underlying: str) -> int:
    """Option-chain (CE/PE) tick count for one index on a trading day — excludes SPOT."""
    day = str(trading_day or "").strip()
    ul = str(underlying or "NIFTY").strip().upper()
    if not day:
        return 0

    try:
        from storage.market_db_inventory_cache import load_cache

        cache = load_cache(data_dir)
        entry = (cache or {}).get("databases", {}).get(day)
        if entry and isinstance(entry.get("rows"), list):
            total = 0
            saw_rows = False
            for row in entry["rows"]:
                if not isinstance(row, dict):
                    continue
                saw_rows = True
                kind = str(row.get("kind") or "").lower()
                expiry = str(row.get("expiry") or "").strip().upper()
                if kind in ("spot", "all") or expiry == "SPOT":
                    continue
                if str(row.get("index") or "").strip().upper() != ul:
                    continue
                total += int(row.get("tick_count") or 0)
            if saw_rows:
                return total
    except Exception:
        pass

    import sqlite3

    from chain_replay_ml.export_atm_pipeline import replay_db_path
    from chain_replay_ml.replay_feature_scoring import chart_dir_from_data_dir

    db_path = replay_db_path(chart_dir_from_data_dir(data_dir), day)
    if not db_path or not os.path.isfile(db_path):
        return 0
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        try:
            has_meta = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='token_day_meta'"
            ).fetchone()
            if not has_meta:
                return 0
            row = conn.execute(
                """
                SELECT COUNT(t.id) AS tick_count
                FROM token_day_meta m
                INNER JOIN ticks t ON t.token = m.token
                WHERE m.as_of_date = ?
                  AND m.name = ?
                  AND m.option_type IN ('CE', 'PE')
                  AND m.expiry_date IS NOT NULL AND m.expiry_date != ''
                """,
                (day, ul),
            ).fetchone()
            return int(row[0] or 0) if row else 0
        finally:
            conn.close()
    except sqlite3.Error:
        return 0


def filter_days_by_min_chain_ticks(
    data_dir: str,
    days: list[str],
    underlying: str,
    *,
    min_ticks: int = _MIN_CHAIN_TICKS_MODEL_DAY,
) -> tuple[list[str], dict[str, int], list[dict[str, Any]]]:
    """Keep only days with enough option-chain ticks for the selected index."""
    eligible: list[str] = []
    counts: dict[str, int] = {}
    skipped: list[dict[str, Any]] = []
    floor = max(0, int(min_ticks))
    for day in days:
        n = chain_tick_count_for_day(data_dir, day, underlying)
        counts[day] = n
        if n >= floor:
            eligible.append(day)
        else:
            skipped.append({"trading_day": day, "tick_count": n, "min_ticks": floor})
    return eligible, counts, skipped


def persist_replay_session_artifacts(
    data_dir: str,
    *,
    session_id: str,
    position_limit: int,
    trades: list,
    scored_rows: int,
    scoring_coverage: dict,
    model_audit: dict | None,
    timing: dict,
    expiry_scoring: str | None,
    trading_day: str = "",
) -> None:
    save_artifact_json(
        data_dir,
        session_id,
        ARTIFACT_TRADE_REPORT,
        {"trades": trades, "scored_rows": int(scored_rows)},
        compute_ms=(timing.get("steps") or {}).get("simulate_trades"),
    )
    save_artifact_json(
        data_dir,
        session_id,
        ARTIFACT_POSITION_LIMITS,
        {"rows": build_position_limit_snapshot(trades, trading_day=str(trading_day or "").strip())},
        compute_ms=(timing.get("steps") or {}).get("simulate_trades"),
    )
    save_artifact_json(
        data_dir,
        session_id,
        ARTIFACT_SCORING_COVERAGE,
        scoring_coverage,
        compute_ms=(timing.get("steps") or {}).get("scoring_coverage"),
    )
    if model_audit is not None:
        save_artifact_json(
            data_dir,
            session_id,
            ARTIFACT_MODEL_AUDIT,
            model_audit,
            compute_ms=(timing.get("steps") or {}).get("model_audit"),
        )
    artifacts = list_artifacts(data_dir, session_id)
    ready = sum(
        1 for a in artifacts
        if a.get("status") == "ready" and a.get("artifact_name") in KNOWN_ARTIFACTS
    )
    cache_state = "complete" if ready >= len(KNOWN_ARTIFACTS) else ("partial" if ready else "none")
    save_artifact_json(
        data_dir,
        session_id,
        ARTIFACT_SUMMARY,
        {
            "position_limit": int(position_limit),
            "resolved_expiry": expiry_scoring,
            "timing": timing,
            "cache_state": cache_state,
            "scored_rows": int(scored_rows),
        },
    )


def compute_replay_day_for_model(
    data_dir: str,
    *,
    model_name: str,
    trading_day: str,
    underlying: str = "NIFTY",
    expiry_hint: str | None = None,
    position_limit: int = 1,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Score one trading day, persist replay session artifacts, return metrics row."""
    from chain_replay_ml.recompute_2_1_ratio import compute_model_audit, run_experiment_backtest_from_scored_frame
    from chain_replay_ml.registry_backtest import build_registry_scored_frame, model_scoring_coverage
    from chain_replay_ml.replay_feature_scoring import chart_dir_from_data_dir, resolve_scoring_expiry
    from chain_replay_ml.replay_scoring_cache import get_cached_token_timelines, replay_timeline_cache_key
    from chain_replay_ml.training.registry import get_trained_model

    def emit(event: str, **kwargs: Any) -> None:
        _emit_progress(on_progress, {"event": event, "trading_day": day, **kwargs})

    def cancelled() -> bool:
        return bool(cancel_event and cancel_event.is_set())

    name = str(model_name or "").strip()
    day = str(trading_day or "").strip()
    ul = str(underlying or "NIFTY").strip().upper()
    pos = int(position_limit)
    registry_row = get_trained_model(data_dir, name)
    if not registry_row:
        return {
            "trading_day": day,
            "status": "error",
            "error": f"Model not found: {name}",
        }

    t0 = time.perf_counter()
    emit("DAY_STEP", step="resolve_expiry", message=_step_message("resolve_expiry", trading_day=day))
    chart_dir = chart_dir_from_data_dir(data_dir)
    expiry_resolution = resolve_scoring_expiry(
        chart_dir,
        day,
        str(expiry_hint or "").strip() or None,
        underlying=ul,
    )
    expiry_scoring = str(expiry_resolution.get("resolved_expiry") or "").strip()
    if not expiry_scoring:
        row = {
            "trading_day": day,
            "underlying": ul,
            "expiry": None,
            "status": "no_db",
            "error": expiry_resolution.get("reason") or "No tick database or expiry for this day",
        }
        emit("DAY_DONE", row=row)
        return row

    step_times: dict[str, float] = {}
    try:
        if cancelled():
            return {"trading_day": day, "status": "cancelled", "error": "Cancelled"}

        emit("DAY_STEP", step="scoring_coverage", message=_step_message("scoring_coverage", trading_day=day))
        t_step = time.perf_counter()
        scoring_coverage = model_scoring_coverage(
            data_dir,
            name,
            day,
            expiry_hint=expiry_scoring,
            underlying=ul,
            expiry_resolution=expiry_resolution,
        )
        step_times["scoring_coverage"] = round(time.perf_counter() - t_step, 3)

        emit("DAY_STEP", step="build_scored_frame", message=_step_message("build_scored_frame", trading_day=day))
        t_step = time.perf_counter()
        scored_step_times: dict[str, float] = {}
        scoring_diagnostics: dict[str, Any] = {}
        heartbeat_stop = threading.Event()

        def heartbeat() -> None:
            while not heartbeat_stop.wait(2.0):
                emit(
                    "DAY_STEP",
                    step="build_scored_frame",
                    elapsed_sec=round(time.perf_counter() - t0, 1),
                    message=_step_message(
                        "build_scored_frame",
                        trading_day=day,
                        detail=f"{round(time.perf_counter() - t_step, 0):.0f}s elapsed",
                    ),
                )

        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True, name=f"mad-heartbeat-{day}")
        heartbeat_thread.start()
        try:
            if on_progress is not None:
                def scored_step(name: str, detail: str) -> None:
                    emit(
                        "DAY_STEP",
                        step=name,
                        elapsed_sec=round(time.perf_counter() - t0, 1),
                        message=_step_message(name, trading_day=day, detail=detail),
                    )

                scored_df = build_registry_scored_frame(
                    data_dir,
                    name,
                    day,
                    expiry_hint=expiry_scoring,
                    parallel_features=False,
                    step_times=scored_step_times,
                    scoring_diagnostics=scoring_diagnostics,
                    on_step_progress=scored_step,
                )
            else:
                scored_df = build_registry_scored_frame(
                    data_dir,
                    name,
                    day,
                    expiry_hint=expiry_scoring,
                    parallel_features=os.getenv("REPLAY_PARALLEL_FEATURES", "1") == "1",
                    step_times=scored_step_times,
                    scoring_diagnostics=scoring_diagnostics,
                )
        finally:
            heartbeat_stop.set()
        for sub_step, sec in scored_step_times.items():
            emit(
                "DAY_STEP",
                step=sub_step,
                elapsed_sec=round(float(sec), 1),
                message=_step_message(sub_step, trading_day=day, detail=f"{sec}s"),
            )
        step_times.update(scored_step_times)
        step_times["build_scored_frame"] = round(time.perf_counter() - t_step, 3)
        if scoring_diagnostics:
            scoring_coverage = dict(scoring_coverage)
            scoring_coverage["scoring_diagnostics"] = scoring_diagnostics

        if cancelled():
            return {"trading_day": day, "status": "cancelled", "error": "Cancelled"}

        tl_key = replay_timeline_cache_key(data_dir, day, expiry_scoring)
        token_timelines = get_cached_token_timelines(tl_key)
        emit("DAY_STEP", step="simulate_trades", message=_step_message("simulate_trades", trading_day=day))
        t_step = time.perf_counter()
        trades = run_experiment_backtest_from_scored_frame(day, scored_df, timelines=token_timelines)
        step_times["simulate_trades"] = round(time.perf_counter() - t_step, 3)

        emit("DAY_STEP", step="model_audit", message=_step_message("model_audit", trading_day=day))
        t_step = time.perf_counter()
        model_audit = compute_model_audit(
            day,
            name,
            pos,
            expiry_hint=expiry_scoring,
            scored_df=scored_df,
        )
        step_times["model_audit"] = round(time.perf_counter() - t_step, 3)

        emit("DAY_STEP", step="persist_session", message=_step_message("persist_session", trading_day=day))
        session_row = ensure_replay_session(
            data_dir,
            model_name=name,
            trading_day=day,
            expiry=expiry_scoring,
            underlying=ul,
        )
        session_id = str(session_row.get("session_id") or "")
        timing = {
            "elapsed_sec": round(time.perf_counter() - t0, 2),
            "cache_hit": False,
            "disk_cache_hit": False,
            "scored_rows": len(scored_df),
            "steps": step_times,
        }
        if session_id:
            persist_replay_session_artifacts(
                data_dir,
                session_id=session_id,
                position_limit=pos,
                trades=trades,
                scored_rows=len(scored_df),
                scoring_coverage=scoring_coverage,
                model_audit=model_audit,
                timing=timing,
                expiry_scoring=expiry_scoring,
                trading_day=day,
            )

        row = _row_from_cached_session(
            data_dir,
            session_row,
            position_limit=pos,
            registry_row=registry_row,
        )
        row["trading_day"] = day
        row["expiry"] = expiry_scoring
        row["underlying"] = ul
        row["status"] = "computed" if row.get("status") == "cached" else row.get("status", "computed")
        row["source"] = "live_scoring"
        row["timing_sec"] = timing["elapsed_sec"]
        emit("DAY_DONE", row=row, elapsed_sec=timing["elapsed_sec"])
        return row
    except Exception as exc:
        row = {
            "trading_day": day,
            "underlying": ul,
            "expiry": expiry_scoring or None,
            "status": "error",
            "error": str(exc),
            "timing_sec": round(time.perf_counter() - t0, 2),
        }
        emit("DAY_DONE", row=row)
        return row


def _row_for_day_from_cache(
    data_dir: str,
    *,
    model_name: str,
    trading_day: str,
    underlying: str,
    position_limit: int,
    registry_row: dict[str, Any] | None,
    session: dict[str, Any] | None,
) -> dict[str, Any]:
    if session:
        row = _row_from_cached_session(
            data_dir,
            session,
            position_limit=int(position_limit),
            registry_row=registry_row,
        )
        row["trading_day"] = trading_day
        row["expiry"] = session.get("expiry")
        row["underlying"] = underlying
        return row
    rmse, mae, composite = _model_validation_metrics(registry_row)
    return {
        "model_name": model_name,
        "trading_day": trading_day,
        "underlying": underlying,
        "expiry": None,
        "algorithm": (registry_row or {}).get("algorithm"),
        "rmse": rmse,
        "mae": mae,
        "composite": composite,
        **_registry_size_fields(registry_row),
        "trades": None,
        "win_pct": None,
        "net_pnl": None,
        "pf": None,
        "dd": None,
        "dd_pct": None,
        "avg_score": None,
        "target_pct": None,
        "status": "not_cached",
        "cache_state": None,
        "source": "registry",
        "error": "Not in replay cache — enable Compute missing or open in replay",
    }


def compare_model_across_trading_days(
    data_dir: str,
    *,
    model_name: str,
    underlying: str = "NIFTY",
    position_limit: int = 1,
    fill_missing: bool = False,
    max_workers: int = 1,
    max_compute_days: int = 30,
    date_from: str | None = None,
    date_to: str | None = None,
    on_progress: Callable[[dict[str, Any]], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """One model × many trading days. Cache-first; optional parallel scoring for gaps."""
    t0 = time.perf_counter()
    name = str(model_name or "").strip()
    ul = str(underlying or "NIFTY").strip().upper()
    pos = int(position_limit)
    if not name:
        raise ValueError("model_name is required")

    def emit(event: str, **kwargs: Any) -> None:
        _emit_progress(on_progress, {"event": event, **kwargs})

    emit("PROGRESS", phase="init", message="Loading model registry…")
    registry_row = get_trained_model(data_dir, name)
    if not registry_row:
        raise ValueError(f"Model not found: {name}")

    emit("PROGRESS", phase="scan_days", message="Scanning tick databases…")
    all_days = list_trading_days_with_tick_db(data_dir)
    if date_from:
        all_days = [d for d in all_days if d >= str(date_from).strip()]
    if date_to:
        all_days = [d for d in all_days if d <= str(date_to).strip()]

    emit(
        "PROGRESS",
        phase="tick_filter",
        message=f"Keeping days with ≥ {_MIN_CHAIN_TICKS_MODEL_DAY:,} {ul} chain ticks…",
    )
    all_days, tick_counts, skipped_low_tick_days = filter_days_by_min_chain_ticks(
        data_dir,
        all_days,
        ul,
        min_ticks=_MIN_CHAIN_TICKS_MODEL_DAY,
    )

    emit("PROGRESS", phase="cache_scan", message=f"Reading replay cache for {len(all_days)} trading days…")
    sessions = list_replay_sessions_for_model(data_dir, model_name=name, underlying=ul)
    session_by_day = {str(s.get("trading_day") or ""): s for s in sessions}

    rows: list[dict[str, Any]] = []
    to_compute: list[str] = []
    for day in all_days:
        session = session_by_day.get(day)
        row = _row_for_day_from_cache(
            data_dir,
            model_name=name,
            trading_day=day,
            underlying=ul,
            position_limit=pos,
            registry_row=registry_row,
            session=session,
        )
        row["chain_tick_count"] = tick_counts.get(day)
        if row.get("status") not in ("cached", "computed") and fill_missing:
            to_compute.append(day)
        rows.append(row)

    cached_count = sum(1 for r in rows if r.get("status") in ("cached", "computed"))
    emit(
        "ROWS",
        rows=rows,
        day_count=len(rows),
        cached_count=cached_count,
        not_cached_count=sum(1 for r in rows if r.get("status") == "not_cached"),
        to_compute=len(to_compute),
        message=f"{cached_count} cached · {len(to_compute)} to compute" if fill_missing else f"{cached_count} cached",
    )

    computed_count = 0
    compute_errors = 0
    if fill_missing and to_compute and not (cancel_event and cancel_event.is_set()):
        workers = max(1, min(int(max_workers), 4))
        batch = to_compute[: max(1, int(max_compute_days))]
        row_by_day = {str(r.get("trading_day") or ""): r for r in rows}
        total_batch = len(batch)

        def _compute_one(day: str, index: int) -> tuple[str, dict[str, Any]]:
            if cancel_event and cancel_event.is_set():
                return day, {"trading_day": day, "status": "cancelled", "error": "Cancelled"}
            emit(
                "DAY_STARTED",
                trading_day=day,
                index=index,
                total=total_batch,
                message=f"Day {index}/{total_batch}: {day}",
            )
            row_by_day[day] = {
                **row_by_day.get(day, {"trading_day": day}),
                "status": "computing",
                "compute_step": "starting",
                "error": None,
            }
            emit("ROW_UPDATE", trading_day=day, row=row_by_day[day])

            def day_progress(payload: dict[str, Any]) -> None:
                event = str(payload.get("event") or "")
                if event == "DAY_STEP":
                    row_by_day[day] = {
                        **row_by_day.get(day, {"trading_day": day}),
                        "status": "computing",
                        "compute_step": payload.get("step"),
                        "compute_message": payload.get("message"),
                        "compute_elapsed_sec": payload.get("elapsed_sec"),
                    }
                    emit("ROW_UPDATE", trading_day=day, row=row_by_day[day])
                emit(event, **{k: v for k, v in payload.items() if k != "event"})

            computed = compute_replay_day_for_model(
                data_dir,
                model_name=name,
                trading_day=day,
                underlying=ul,
                position_limit=pos,
                on_progress=day_progress,
                cancel_event=cancel_event,
            )
            return day, computed

        emit(
            "PROGRESS",
            phase="compute",
            message=f"Computing {total_batch} missing days ({workers} workers)…",
            to_compute=total_batch,
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(_compute_one, day, idx + 1): day
                for idx, day in enumerate(batch)
            }
            for fut in as_completed(futures):
                if cancel_event and cancel_event.is_set():
                    break
                day = futures[fut]
                try:
                    _, computed = fut.result()
                except Exception as exc:
                    computed = {
                        "trading_day": day,
                        "status": "error",
                        "error": str(exc),
                    }
                row_by_day[day] = computed
                emit("ROW_UPDATE", trading_day=day, row=computed)
                if computed.get("status") in ("cached", "computed"):
                    computed_count += 1
                elif computed.get("status") == "error":
                    compute_errors += 1
        rows = [row_by_day.get(str(r.get("trading_day") or ""), r) for r in rows]

    def _sort_key(r: dict[str, Any]) -> str:
        return str(r.get("trading_day") or "")

    rows.sort(key=_sort_key, reverse=True)
    cached_count = sum(1 for r in rows if r.get("status") in ("cached", "computed"))
    not_cached_count = sum(1 for r in rows if r.get("status") == "not_cached")
    elapsed = round(time.perf_counter() - t0, 3)
    rmse, mae, composite = _model_validation_metrics(registry_row)
    return {
        "model_name": name,
        "underlying": ul,
        "position_limit": pos,
        "position_label": position_limit_label(pos),
        "rmse": rmse,
        "mae": mae,
        "composite": composite,
        **_registry_size_fields(registry_row),
        "rows": rows,
        "day_count": len(rows),
        "cached_count": cached_count,
        "not_cached_count": not_cached_count,
        "computed_count": computed_count,
        "compute_errors": compute_errors,
        "compute_queued": len(to_compute) if fill_missing else 0,
        "compute_skipped": max(0, len(to_compute) - int(max_compute_days)) if fill_missing else 0,
        "fill_missing": bool(fill_missing),
        "max_workers": int(max_workers) if fill_missing else 0,
        "min_chain_ticks": _MIN_CHAIN_TICKS_MODEL_DAY,
        "skipped_low_ticks_count": len(skipped_low_tick_days),
        "skipped_low_tick_days": skipped_low_tick_days,
        "source": "sqlite_cache" if not fill_missing else "cache_and_compute",
        "timing_sec": elapsed,
    }
