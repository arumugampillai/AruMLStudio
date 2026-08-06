"""Replay strategy rules on saved prediction rows — no inference."""

from __future__ import annotations

import bisect
from typing import Any

from chain_replay_ml.strategy_registry.schema import normalize_strategy_config

_BATCH = 500


def _num(v: Any) -> float | None:
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def _calc_fees(entry: float, exit_p: float, qty: int, fees_mode: str) -> float:
    """
    Per-trade fees.

    - ``zero``: no charges
    - anything else (default ``rupee_charges``): statutory charges only
      (STT/exchange/SEBI/stamp/GST) — **zero-brokerage plan** (no ₹40 flat)
    """
    if fees_mode == "zero":
        return 0.0
    try:
        from chain_replay_ml.capital_simulation import calculate_zero_brokerage_charges

        return float(calculate_zero_brokerage_charges(entry, exit_p, qty))
    except Exception:
        return 0.0


def _atm_distance_ok(row: dict[str, Any], atm_band: int) -> bool:
    if atm_band <= 0:
        return True
    spot = _num(row.get("spot"))
    strike = _num(row.get("strike"))
    if spot is None or strike is None:
        return True
    step = 50.0 if strike > 1000 else 100.0
    return abs(strike - spot) <= atm_band * step


def _entry_signal(row: dict[str, Any], cfg: dict[str, Any]) -> bool:
    entry = cfg["entry"]
    ltp = _num(row.get("ltp"))
    if ltp is None or ltp <= 0:
        return False
    if not (entry["premium_min"] <= ltp <= entry["premium_max"]):
        return False
    if not _atm_distance_ok(row, int(entry.get("atm_band") or 0)):
        return False
    opt = str(row.get("option_type") or "").upper()
    allowed = [str(x).upper() for x in (entry.get("option_types") or [])]
    if allowed and opt and opt not in allowed:
        return False
    use_regression = bool(entry.get("use_regression", True))
    if use_regression:
        pred = _num(row.get("predicted_ltp"))
        if pred is None:
            return False
        direction = str(entry.get("direction") or "long").lower()
        # Signed predicted move in the trade direction (long: up is +, short: down is +).
        raw_move_pct = ((pred - ltp) / ltp) * 100.0
        predicted_move_pct = raw_move_pct if direction != "short" else -raw_move_pct
        if predicted_move_pct <= 0:
            return False
        min_move = float(entry.get("minimum_predicted_move_pct") or 0.0)
        if min_move > 0 and predicted_move_pct < min_move:
            return False
    conf = cfg["confidence"]
    if conf.get("use_model_confidence"):
        row_conf = _num(row.get("confidence"))
        if row_conf is not None and row_conf < float(conf.get("min_signal_strength") or 0):
            return False
    return True


def _mark_price(row: dict[str, Any]) -> float | None:
    """Live mark at this timestamp — never use actual/future horizon LTP."""
    return _num(row.get("ltp")) if _num(row.get("ltp")) is not None else _num(row.get("current_ltp"))


def _group_path_series(
    path_rows: list[dict[str, Any]],
) -> dict[tuple[str, str], tuple[list[dict[str, Any]], list[float]]]:
    """Group mark-path rows by (day, token) with parallel timestamp lists for bisect."""
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in sorted(
        path_rows,
        key=lambda r: (_num(r.get("timestamp")) or 0.0, int(r.get("row_index") or 0)),
    ):
        key = (str(row.get("trading_day") or ""), str(row.get("token") or ""))
        by_key.setdefault(key, []).append(row)
    out: dict[tuple[str, str], tuple[list[dict[str, Any]], list[float]]] = {}
    for key, series in by_key.items():
        ts_list = [_num(r.get("timestamp")) or 0.0 for r in series]
        out[key] = (series, ts_list)
    return out


def _forward_path_after_entry(
    series: list[dict[str, Any]],
    ts_list: list[float],
    entry_row: dict[str, Any],
) -> list[dict[str, Any]]:
    """Samples strictly after the entry row on the mark path (not classifier-thinned)."""
    if not series:
        return []
    pid = entry_row.get("prediction_id")
    if pid is not None:
        for i, r in enumerate(series):
            if r.get("prediction_id") == pid:
                return series[i + 1 :]
    entry_ts = _num(entry_row.get("timestamp"))
    if entry_ts is None:
        return []
    entry_ri = int(entry_row.get("row_index") or 0)
    idx = bisect.bisect_left(ts_list, entry_ts)
    while idx < len(series) and ts_list[idx] == entry_ts:
        if int(series[idx].get("row_index") or 0) <= entry_ri:
            idx += 1
            continue
        break
    return series[idx:]


def _simulate_trade(
    *,
    entry_row: dict[str, Any],
    forward_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    strategy_run_id: str,
    strategy_version_id: str,
    prediction_run_id: str,
    fold_number: int | None,
    qty: int,
) -> dict[str, Any] | None:
    entry_price = _num(entry_row.get("ltp"))
    entry_ts = _num(entry_row.get("timestamp"))
    if entry_price is None or entry_ts is None or entry_price <= 0:
        return None

    target_pct = float(cfg["target"]["target_profit_pct"])
    stop_pct = float(cfg["stop"]["stop_loss_pct"])
    max_hold = int(cfg["hold_time"]["max_hold_sec"])
    direction = str(cfg["entry"].get("direction") or "long").lower()
    use_regression = bool(cfg["entry"].get("use_regression", True))
    # Without regression: %-target only (never use predicted LTP as exit target).
    use_predicted_ltp = bool(cfg["target"].get("use_predicted_ltp")) and use_regression

    if direction == "long":
        stop_price = entry_price * (1.0 - stop_pct / 100.0)
    else:
        stop_price = entry_price * (1.0 + stop_pct / 100.0)

    if use_predicted_ltp:
        target_price = _num(entry_row.get("predicted_ltp"))
        if target_price is None or target_price <= 0:
            return None
        # Predicted target must be on the profitable side of entry for this direction.
        if direction == "long" and target_price <= entry_price:
            return None
        if direction == "short" and target_price >= entry_price:
            return None
    elif direction == "long":
        target_price = entry_price * (1.0 + target_pct / 100.0)
    else:
        target_price = entry_price * (1.0 - target_pct / 100.0)

    # Same-timestamp entry/exit is not a valid trade. exit_reason stays unset until
    # target/stop/max_hold/end_of_path is decided from forward samples only.
    exit_row: dict[str, Any] | None = None
    exit_reason: str | None = None
    max_fav = 0.0
    max_adv = 0.0
    time_to_first_profit_sec: float | None = None
    lowest_mark = entry_price
    highest_mark = entry_price
    lowest_unrealized = 0.0
    exit_sample_index = 0  # 1-based count of prediction samples after entry until exit

    for row in forward_rows:
        ts = _num(row.get("timestamp"))
        if ts is None or ts <= entry_ts:
            continue
        hold = float(ts - entry_ts)
        # Live path only — actual_ltp is the horizon label, not the mark.
        mark = _mark_price(row)
        if mark is None:
            continue
        exit_sample_index += 1
        if direction == "long":
            lowest_mark = min(lowest_mark, mark)
            highest_mark = max(highest_mark, mark)
            unreal = (mark - entry_price) * qty
        else:
            lowest_mark = min(lowest_mark, mark)
            highest_mark = max(highest_mark, mark)
            unreal = (entry_price - mark) * qty
        lowest_unrealized = min(lowest_unrealized, unreal)

        move_pct = ((mark - entry_price) / entry_price) * 100.0
        if direction == "short":
            move_pct = -move_pct
        max_fav = max(max_fav, move_pct)
        max_adv = min(max_adv, move_pct)
        if time_to_first_profit_sec is None and move_pct > 0:
            time_to_first_profit_sec = hold

        # Evaluate stop/target on every forward sample (including hold >= max_hold).
        # Time exit only when the hold window is reached and levels were not hit.
        if direction == "long":
            if mark >= target_price:
                exit_row = row
                exit_reason = "target"
                break
            if mark <= stop_price:
                exit_row = row
                exit_reason = "stop"
                break
        else:
            if mark <= target_price:
                exit_row = row
                exit_reason = "target"
                break
            if mark >= stop_price:
                exit_row = row
                exit_reason = "stop"
                break

        exit_row = row
        if hold >= max_hold:
            exit_reason = "max_hold"
            break

    # No usable forward path (empty series, all non-positive Δt, or no marks).
    if exit_row is None or exit_sample_index == 0:
        return None

    if exit_reason is None:
        # Path exhausted before max_hold without target/stop — not a time exit.
        exit_reason = "end_of_path"

    sample_exit_ltp = _mark_price(exit_row) or entry_price
    exit_ts = _num(exit_row.get("timestamp")) or entry_ts
    holding = max(0.0, exit_ts - entry_ts)

    if exit_reason == "max_hold" and holding < float(max_hold):
        raise AssertionError(
            f"max_hold exit requires holding_seconds >= max_hold_sec "
            f"({holding} < {max_hold})"
        )
    if exit_reason == "max_hold":
        # Stop/target are evaluated on the max_hold sample before time exit.
        if direction == "long" and sample_exit_ltp <= stop_price:
            raise AssertionError(
                f"max_hold exit mark {sample_exit_ltp} is at/beyond stop {stop_price}"
            )
        if direction == "short" and sample_exit_ltp >= stop_price:
            raise AssertionError(
                f"max_hold exit mark {sample_exit_ltp} is at/beyond stop {stop_price}"
            )

    # Limit fills: sample LTP is evidence the level was crossed. Default fills at
    # the configured target/stop price so a 1:1 strategy stays 1:1. Optional
    # fill_at_sample_ltp applies to BOTH target and stop (never mixed).
    fill_at_sample = bool(
        cfg.get("execution", {}).get("fill_at_sample_ltp")
        or cfg.get("stop", {}).get("gap_fill_at_sample_ltp")  # legacy alias
        or cfg.get("target", {}).get("fill_at_sample_ltp")
    )
    stop_trigger_ltp: float | None = None
    target_trigger_ltp: float | None = None
    gap_beyond_stop = False

    if exit_reason == "stop":
        stop_trigger_ltp = sample_exit_ltp
        if fill_at_sample:
            exit_price = sample_exit_ltp
            if direction == "long" and exit_price < stop_price - 1e-9:
                gap_beyond_stop = True
            if direction == "short" and exit_price > stop_price + 1e-9:
                gap_beyond_stop = True
        else:
            exit_price = stop_price
            gap_beyond_stop = False
            # Cap floating trough at stop fill — sample print-through is not realizable.
            if direction == "long":
                stop_unreal = (stop_price - entry_price) * qty
                lowest_unrealized = max(lowest_unrealized, stop_unreal)
            else:
                stop_unreal = (entry_price - stop_price) * qty
                lowest_unrealized = max(lowest_unrealized, stop_unreal)
    elif exit_reason == "target":
        target_trigger_ltp = sample_exit_ltp
        if fill_at_sample:
            exit_price = sample_exit_ltp
        else:
            exit_price = target_price
    else:
        # max_hold / end_of_path: exit at last sample mark
        exit_price = sample_exit_ltp

    if direction == "long":
        gross = (exit_price - entry_price) * qty
        ret_pct = ((exit_price - entry_price) / entry_price) * 100.0
    else:
        gross = (entry_price - exit_price) * qty
        ret_pct = ((entry_price - exit_price) / entry_price) * 100.0

    fees = _calc_fees(entry_price, exit_price, qty, str(cfg["execution"].get("fees_mode") or "rupee_charges"))
    net = gross - fees
    stop_risk = entry_price * qty * (stop_pct / 100.0)

    return {
        "trade_id": f"{strategy_run_id}:{entry_row.get('prediction_id')}",
        "strategy_run_id": strategy_run_id,
        "prediction_run_id": prediction_run_id,
        "fold_id": entry_row.get("fold_id"),
        "fold_number": fold_number,
        "entry_prediction_id": entry_row.get("prediction_id"),
        "exit_prediction_id": exit_row.get("prediction_id"),
        "strategy_version_id": strategy_version_id,
        "trading_day": entry_row.get("trading_day"),
        "token": entry_row.get("token"),
        "strike": entry_row.get("strike"),
        "option_type": entry_row.get("option_type"),
        "direction": direction,
        "entry_ts": entry_ts,
        "exit_ts": exit_ts,
        "entry_price": round(entry_price, 4),
        "exit_price": round(exit_price, 4),
        "stop_price": round(stop_price, 4),
        "target_price": round(target_price, 4),
        "use_predicted_ltp": use_predicted_ltp,
        "qty": qty,
        "gross_pnl": round(gross, 2),
        "fees": round(fees, 2),
        "net_pnl": round(net, 2),
        "return_pct": round(ret_pct, 4),
        "holding_seconds": round(holding, 3),
        "exit_reason": exit_reason,
        "stop_loss_pct": stop_pct,
        "stop_risk_rupees": round(stop_risk, 2),
        "expected_stop_loss_rupees": round(stop_risk, 2),
        "lowest_mark_price": round(lowest_mark, 4),
        "highest_mark_price": round(highest_mark, 4),
        "lowest_unrealized_pnl": round(lowest_unrealized, 2),
        "stop_trigger_ltp": round(stop_trigger_ltp, 4) if stop_trigger_ltp is not None else None,
        "target_trigger_ltp": (
            round(target_trigger_ltp, 4) if target_trigger_ltp is not None else None
        ),
        "sample_exit_ltp": round(sample_exit_ltp, 4),
        "fill_at_sample_ltp": fill_at_sample,
        "exit_sample_index": int(exit_sample_index),
        "exit_row_index": (
            int(exit_row["row_index"])
            if exit_row.get("row_index") is not None
            else None
        ),
        "gap_beyond_stop": gap_beyond_stop,
        "max_favorable_pct": round(max_fav, 4),
        "max_adverse_pct": round(max_adv, 4),
        "time_to_first_profit_sec": (
            round(time_to_first_profit_sec, 3)
            if time_to_first_profit_sec is not None
            else None
        ),
    }


def normalize_execution_rules(rules: dict[str, Any] | None) -> dict[str, Any]:
    """
    Strategy Simulator-only execution constraints.

    Never applied by Confidence Label Builder (forced-entry replay).
    """
    src = rules if isinstance(rules, dict) else {}
    enabled = bool(src.get("enabled"))
    try:
        max_open = int(src.get("max_open_positions") or 0)
    except (TypeError, ValueError):
        max_open = 0
    return {
        "enabled": enabled,
        "max_open_positions": max(0, max_open),
        "one_position_per_symbol": bool(src.get("one_position_per_symbol")),
    }


def _empty_sim_stats() -> dict[str, int]:
    return {
        "predictions_evaluated": 0,
        "signals": 0,
        "skipped": 0,
        "trades": 0,
        "no_signal": 0,
        "blocked_open": 0,
        "skipped_cadence": 0,
        "skipped_no_path": 0,
        "skipped_max_positions": 0,
        "skipped_same_symbol": 0,
        "candidate_signals": 0,
    }


def _count_open_at(open_exits: list[float], ts: float) -> int:
    return sum(1 for exit_ts in open_exits if exit_ts > ts)


def simulate_fold_rows(
    rows: list[dict[str, Any]],
    *,
    cfg: dict[str, Any],
    strategy_run_id: str,
    strategy_version_id: str,
    prediction_run_id: str,
    fold_number: int | None = None,
    execution_rules: dict[str, Any] | None = None,
    path_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Simulate trades for one fold's prediction rows.

    Pipeline (Strategy Simulator):
      rows → Strategy entry → Candidate signals → Execution Rules → Trades

    ``rows`` are entry candidates (may be classifier-filtered).
    ``path_rows`` is the mark path for stop/target/hold (defaults to ``rows``).
    Classifier must gate entries only — never thin the exit path.

    ``execution_rules`` is Simulator-only and must not be passed into
    ``simulate_forced_entry_outcomes`` (label builder).
    """
    cfg = normalize_strategy_config(cfg)
    rules = normalize_execution_rules(execution_rules)
    pos = cfg["position_size"]
    qty = int(pos.get("lots") or 1) * int(pos.get("qty_per_lot") or 65)
    cadence = int(cfg["entry"].get("entry_cadence_sec") or 1)

    sorted_rows = sorted(
        rows,
        key=lambda r: (_num(r.get("timestamp")) or 0.0, int(r.get("row_index") or 0)),
    )

    path_by_key = _group_path_series(path_rows if path_rows is not None else sorted_rows)

    stats = _empty_sim_stats()
    stats["predictions_evaluated"] = len(sorted_rows)
    trades: list[dict[str, Any]] = []
    last_entry_ts: dict[tuple[str, str], float] = {}
    # Legacy: one open position per (day, token) when Execution Rules are off.
    open_until: dict[tuple[str, str], float] = {}
    # Execution Rules state
    open_exits: list[float] = []
    symbol_open_until: dict[str, float] = {}

    for row in sorted_rows:
        key = (str(row.get("trading_day") or ""), str(row.get("token") or ""))
        token = str(row.get("token") or "")
        ts = _num(row.get("timestamp")) or 0.0

        if not rules["enabled"]:
            # Backward-compatible default: block re-entry while same token is open.
            if open_until.get(key, 0) > ts:
                stats["blocked_open"] += 1
                continue

        if not _entry_signal(row, cfg):
            stats["no_signal"] += 1
            continue

        stats["signals"] += 1

        last_ts = last_entry_ts.get(key)
        if last_ts is not None and ts - last_ts < cadence:
            stats["skipped"] += 1
            stats["skipped_cadence"] += 1
            continue

        # Candidate = passed Strategy entry (+ cadence); Execution Rules apply next.
        stats["candidate_signals"] += 1

        if rules["enabled"]:
            if rules["one_position_per_symbol"] and symbol_open_until.get(token, 0) > ts:
                stats["skipped"] += 1
                stats["skipped_same_symbol"] += 1
                continue
            max_open = int(rules["max_open_positions"] or 0)
            if max_open > 0 and _count_open_at(open_exits, ts) >= max_open:
                stats["skipped"] += 1
                stats["skipped_max_positions"] += 1
                continue

        series_pack = path_by_key.get(key)
        if series_pack is None:
            forward: list[dict[str, Any]] = []
        else:
            series, ts_list = series_pack
            forward = _forward_path_after_entry(series, ts_list, row)

        trade = _simulate_trade(
            entry_row=row,
            forward_rows=forward,
            cfg=cfg,
            strategy_run_id=strategy_run_id,
            strategy_version_id=strategy_version_id,
            prediction_run_id=prediction_run_id,
            fold_number=fold_number,
            qty=qty,
        )
        if not trade:
            stats["skipped"] += 1
            stats["skipped_no_path"] += 1
            continue
        trades.append(trade)
        stats["trades"] += 1
        last_entry_ts[key] = ts
        exit_ts = float(trade.get("exit_ts") or ts)
        if rules["enabled"]:
            open_exits.append(exit_ts)
            if rules["one_position_per_symbol"]:
                symbol_open_until[token] = exit_ts
        else:
            open_until[key] = exit_ts

    return trades, stats


def simulate_prediction_rows(
    rows: list[dict[str, Any]],
    *,
    cfg: dict[str, Any],
    strategy_run_id: str,
    strategy_version_id: str,
    prediction_run_id: str,
    fold_numbers: dict[str, int] | None = None,
    execution_rules: dict[str, Any] | None = None,
    path_rows: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Simulate across rows, optionally grouped by fold_id.

    ``path_rows`` (optional) is the full mark path for exits. When omitted, ``rows``
    is used for both entry candidates and the mark path.
    """
    if not rows:
        return [], _empty_sim_stats()

    by_fold: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        fid = str(row.get("fold_id") or "")
        by_fold.setdefault(fid, []).append(row)

    path_by_fold: dict[str, list[dict[str, Any]]] | None = None
    if path_rows is not None:
        path_by_fold = {}
        for row in path_rows:
            path_by_fold.setdefault(str(row.get("fold_id") or ""), []).append(row)

    if len(by_fold) <= 1:
        fid = next(iter(by_fold), "")
        fn = (fold_numbers or {}).get(fid)
        return simulate_fold_rows(
            rows,
            cfg=cfg,
            strategy_run_id=strategy_run_id,
            strategy_version_id=strategy_version_id,
            prediction_run_id=prediction_run_id,
            fold_number=fn,
            execution_rules=execution_rules,
            path_rows=path_rows,
        )

    all_trades: list[dict[str, Any]] = []
    total = _empty_sim_stats()
    fold_numbers = fold_numbers or {}
    for fid, fold_rows in by_fold.items():
        fold_path = None if path_by_fold is None else path_by_fold.get(fid, [])
        fold_trades, stats = simulate_fold_rows(
            fold_rows,
            cfg=cfg,
            strategy_run_id=strategy_run_id,
            strategy_version_id=strategy_version_id,
            prediction_run_id=prediction_run_id,
            fold_number=fold_numbers.get(fid),
            execution_rules=execution_rules,
            path_rows=fold_path,
        )
        all_trades.extend(fold_trades)
        for k in total:
            total[k] += int(stats.get(k) or 0)
    return all_trades, total


def simulate_forced_entry_outcomes(
    rows: list[dict[str, Any]],
    *,
    cfg: dict[str, Any],
    strategy_version_id: str = "",
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Confidence Label Builder mode: replay every prediction row as if entered.

    Ignores cadence and open-position blocking. Still records ``would_enter``
    from strategy entry filters (premium / ATM / direction / …).

    Returns continuous outcome rows suitable for TargetSpec derivation —
    not Strategy Simulator trade-list rows.
    """
    cfg = normalize_strategy_config(cfg)
    pos = cfg["position_size"]
    qty = int(pos.get("lots") or 1) * int(pos.get("qty_per_lot") or 65)

    sorted_rows = sorted(
        rows,
        key=lambda r: (_num(r.get("timestamp")) or 0.0, int(r.get("row_index") or 0)),
    )
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in sorted_rows:
        key = (str(row.get("trading_day") or ""), str(row.get("token") or ""))
        by_key.setdefault(key, []).append(row)

    stats = {
        "predictions_evaluated": len(sorted_rows),
        "outcomes": 0,
        "skipped_no_path": 0,
        "would_enter": 0,
        "would_not_enter": 0,
    }
    outcomes: list[dict[str, Any]] = []

    for i, row in enumerate(sorted_rows):
        key = (str(row.get("trading_day") or ""), str(row.get("token") or ""))
        would = bool(_entry_signal(row, cfg))
        if would:
            stats["would_enter"] += 1
        else:
            stats["would_not_enter"] += 1

        series = by_key.get(key, [])
        try:
            idx = series.index(row)
            forward = series[idx + 1 :]
        except ValueError:
            forward = sorted_rows[i + 1 :]

        trade = _simulate_trade(
            entry_row=row,
            forward_rows=forward,
            cfg=cfg,
            strategy_run_id="confidence_label",
            strategy_version_id=strategy_version_id,
            prediction_run_id="confidence_label",
            fold_number=None,
            qty=qty,
        )
        if not trade:
            stats["skipped_no_path"] += 1
            outcomes.append({
                "prediction_id": row.get("prediction_id"),
                "trading_day": row.get("trading_day"),
                "token": row.get("token"),
                "net_pnl": None,
                "gross_pnl": None,
                "return_pct": None,
                "max_adverse_pct": None,
                "max_favorable_pct": None,
                "holding_seconds": None,
                "time_to_first_profit_sec": None,
                "exit_reason": None,
                "fees": None,
                "would_enter": int(would),
            })
            continue

        outcomes.append({
            "prediction_id": trade.get("entry_prediction_id") or row.get("prediction_id"),
            "trading_day": trade.get("trading_day"),
            "token": trade.get("token"),
            "net_pnl": trade.get("net_pnl"),
            "gross_pnl": trade.get("gross_pnl"),
            "return_pct": trade.get("return_pct"),
            "max_adverse_pct": trade.get("max_adverse_pct"),
            "max_favorable_pct": trade.get("max_favorable_pct"),
            "holding_seconds": trade.get("holding_seconds"),
            "time_to_first_profit_sec": trade.get("time_to_first_profit_sec"),
            "exit_reason": trade.get("exit_reason"),
            "fees": trade.get("fees"),
            "would_enter": int(would),
        })
        stats["outcomes"] += 1

    return outcomes, stats
