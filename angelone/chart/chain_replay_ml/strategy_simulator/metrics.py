"""Aggregate trading metrics from simulated trades."""

from __future__ import annotations

from typing import Any


def _trade_sort_key(t: dict[str, Any]) -> tuple:
    return (t.get("exit_ts") or 0, t.get("trade_id") or "")


def _empty_max_drawdown_episode() -> dict[str, Any]:
    return {
        "method": "closed_trade_cumulative_equity_peak_to_trough",
        "uses_floating_pnl": False,
        "uses_executed_trades_only": True,
        "max_drawdown": 0.0,
        "peak_equity": 0.0,
        "trough_equity": 0.0,
        "peak_point": None,
        "trough_point": None,
        "peak_exit_ts": None,
        "trough_exit_ts": None,
        "peak_trading_day": None,
        "trough_trading_day": None,
        "peak_trade_id": None,
        "trough_trade_id": None,
        "peak_token": None,
        "trough_token": None,
    }


def build_equity_curve(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Cumulative realized equity from executed trades only (sorted by exit time).

    Equity starts at 0 and adds each trade's net_pnl at close. This is a
    closed-trade P&L curve — not floating mark-to-market of open positions.
    Skipped / candidate signals never appear.
    """
    curve: list[dict[str, Any]] = []
    cumulative = 0.0
    peak = 0.0
    for i, t in enumerate(sorted(trades, key=_trade_sort_key), start=1):
        pnl = float(t.get("net_pnl") or 0)
        cumulative += pnl
        peak = max(peak, cumulative)
        dd = peak - cumulative
        curve.append({
            "point": i,
            "trade_id": t.get("trade_id"),
            "trading_day": t.get("trading_day"),
            "token": t.get("token"),
            "exit_ts": t.get("exit_ts"),
            "entry_ts": t.get("entry_ts"),
            "entry_price": t.get("entry_price"),
            "exit_price": t.get("exit_price"),
            "exit_reason": t.get("exit_reason"),
            "qty": t.get("qty"),
            "net_pnl": round(pnl, 2),
            "equity": round(cumulative, 2),
            "peak": round(peak, 2),
            "drawdown": round(dd, 2),
            "is_max_dd_peak": False,
            "is_max_dd_trough": False,
        })
    return curve


def compute_average_drawdown(trades: list[dict[str, Any]]) -> float:
    """Mean peak-to-current drawdown across the closed-trade equity curve.

    Uses the same curve as Max Drawdown (closed trades only, no floating P&L);
    Max Drawdown is the worst single point on this same series.
    """
    curve = build_equity_curve(trades)
    if not curve:
        return 0.0
    return round(sum(float(p.get("drawdown") or 0.0) for p in curve) / len(curve), 2)


def compute_max_drawdown_episode(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Max Drawdown = peak-to-trough drop on the closed-trade cumulative equity curve.

    Not floating P&L of open positions. Only executed trades contribute.
    """
    episode = _empty_max_drawdown_episode()
    if not trades:
        return episode

    cumulative = 0.0
    peak = 0.0
    peak_meta: dict[str, Any] = {
        "point": 0,
        "exit_ts": None,
        "trading_day": None,
        "trade_id": None,
        "token": None,
        "equity": 0.0,
    }
    max_dd = 0.0
    trough_meta: dict[str, Any] = dict(peak_meta)

    for i, t in enumerate(sorted(trades, key=_trade_sort_key), start=1):
        pnl = float(t.get("net_pnl") or 0)
        cumulative += pnl
        if cumulative >= peak:
            peak = cumulative
            peak_meta = {
                "point": i,
                "exit_ts": t.get("exit_ts"),
                "trading_day": t.get("trading_day"),
                "trade_id": t.get("trade_id"),
                "token": t.get("token"),
                "equity": peak,
            }
        dd = peak - cumulative
        if dd > max_dd:
            max_dd = dd
            trough_meta = {
                "point": i,
                "exit_ts": t.get("exit_ts"),
                "trading_day": t.get("trading_day"),
                "trade_id": t.get("trade_id"),
                "token": t.get("token"),
                "equity": cumulative,
                "peak_point": peak_meta["point"],
                "peak_exit_ts": peak_meta["exit_ts"],
                "peak_trading_day": peak_meta["trading_day"],
                "peak_trade_id": peak_meta["trade_id"],
                "peak_token": peak_meta["token"],
                "peak_equity": peak_meta["equity"],
            }

    episode.update({
        "max_drawdown": round(max_dd, 2),
        "peak_equity": round(float(trough_meta.get("peak_equity", peak_meta["equity"]) or 0), 2),
        "trough_equity": round(float(trough_meta.get("equity") or 0), 2),
        "peak_point": trough_meta.get("peak_point", peak_meta["point"]) if max_dd > 0 else peak_meta["point"],
        "trough_point": trough_meta.get("point") if max_dd > 0 else None,
        "peak_exit_ts": trough_meta.get("peak_exit_ts", peak_meta["exit_ts"]) if max_dd > 0 else peak_meta["exit_ts"],
        "trough_exit_ts": trough_meta.get("exit_ts") if max_dd > 0 else None,
        "peak_trading_day": trough_meta.get("peak_trading_day", peak_meta["trading_day"]) if max_dd > 0 else peak_meta["trading_day"],
        "trough_trading_day": trough_meta.get("trading_day") if max_dd > 0 else None,
        "peak_trade_id": trough_meta.get("peak_trade_id", peak_meta["trade_id"]) if max_dd > 0 else peak_meta["trade_id"],
        "trough_trade_id": trough_meta.get("trade_id") if max_dd > 0 else None,
        "peak_token": trough_meta.get("peak_token", peak_meta["token"]) if max_dd > 0 else peak_meta["token"],
        "trough_token": trough_meta.get("token") if max_dd > 0 else None,
    })
    return episode


def annotate_equity_curve_max_dd(
    curve: list[dict[str, Any]],
    episode: dict[str, Any],
) -> list[dict[str, Any]]:
    """Mark the peak and trough points used for Max Drawdown on the curve."""
    peak_pt = episode.get("peak_point")
    trough_pt = episode.get("trough_point")
    out: list[dict[str, Any]] = []
    for p in curve:
        row = dict(p)
        row["is_max_dd_peak"] = peak_pt is not None and int(p.get("point") or 0) == int(peak_pt)
        row["is_max_dd_trough"] = (
            trough_pt is not None and int(p.get("point") or 0) == int(trough_pt)
        )
        out.append(row)
    return out


def compute_outcome_audit(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Win/loss averages, pre-fee gross sides, exit-reason mix, and target/stop fill quality.

    Default: both target and stop fill at configured prices (sample LTP is trigger
    evidence only). Optional fill_at_sample_ltp applies to both sides together.
    """
    empty = {
        "avg_winning_trade_net": 0.0,
        "avg_losing_trade_net": 0.0,
        "avg_winning_trade_gross": 0.0,
        "avg_losing_trade_gross": 0.0,
        "gross_profit_before_fees": 0.0,
        "gross_loss_before_fees": 0.0,
        "profit_factor_before_fees": None,
        "profit_factor_after_fees": None,
        "profit_factor_formula": (
            "sum(net_pnl of winners) / abs(sum(net_pnl of losers))"
        ),
        "net_profit_formula": "sum(gross_pnl) − total_fees",
        "total_fees": 0.0,
        "net_profit": 0.0,
        "gross_pnl_total": 0.0,
        "exit_reason_counts": {
            "target": 0,
            "stop": 0,
            "max_hold": 0,
            "end_of_path": 0,
            "other": 0,
        },
        "target_exact": 0,
        "target_above": 0,
        "target_below": 0,
        "stop_exact": 0,
        "stop_beyond_gap": 0,
        "stop_other": 0,
        "stop_trigger_beyond_sample": 0,
        "gap_beyond_stop_count": 0,
        "target_mean_return_pct": None,
        "stop_mean_return_pct": None,
        "hold_mean_return_pct": None,
        "asymmetry_note": "",
    }
    if not trades:
        return empty

    def _f(t: dict[str, Any], key: str, default: float = 0.0) -> float:
        try:
            v = t.get(key)
            return float(v) if v is not None else default
        except (TypeError, ValueError):
            return default

    wins = [t for t in trades if _f(t, "net_pnl") > 0]
    losses = [t for t in trades if _f(t, "net_pnl") <= 0]
    gpos = [t for t in trades if _f(t, "gross_pnl") > 0]
    gneg = [t for t in trades if _f(t, "gross_pnl") < 0]

    gp_before = sum(max(0.0, _f(t, "gross_pnl")) for t in trades)
    gl_before = abs(sum(min(0.0, _f(t, "gross_pnl")) for t in trades))
    gp_net = sum(_f(t, "net_pnl") for t in wins)
    gl_net = abs(sum(_f(t, "net_pnl") for t in losses))
    fees = sum(_f(t, "fees") for t in trades)
    net = sum(_f(t, "net_pnl") for t in trades)
    gross_total = sum(_f(t, "gross_pnl") for t in trades)

    reasons = {"target": 0, "stop": 0, "max_hold": 0, "end_of_path": 0, "other": 0}
    target_exact = target_above = target_below = 0
    stop_exact = stop_beyond = stop_other = 0
    stop_trigger_beyond = 0
    gap_count = 0
    target_rets: list[float] = []
    stop_rets: list[float] = []
    hold_rets: list[float] = []
    eps = 1e-4

    for t in trades:
        reason = str(t.get("exit_reason") or "")
        if reason in reasons:
            reasons[reason] += 1
        else:
            reasons["other"] += 1
        if t.get("gap_beyond_stop"):
            gap_count += 1

        entry = _f(t, "entry_price")
        exitp = _f(t, "exit_price")
        direction = str(t.get("direction") or "long").lower()
        tgt = _f(t, "target_price") if t.get("target_price") is not None else None
        stp = _f(t, "stop_price") if t.get("stop_price") is not None else None
        trig = t.get("stop_trigger_ltp")

        if trig is not None and stp is not None:
            try:
                trig_f = float(trig)
                if direction == "long" and trig_f < stp - eps:
                    stop_trigger_beyond += 1
                elif direction == "short" and trig_f > stp + eps:
                    stop_trigger_beyond += 1
            except (TypeError, ValueError):
                pass

        if reason == "target" and tgt is not None:
            target_rets.append(_f(t, "return_pct"))
            if abs(exitp - tgt) <= max(eps, abs(tgt) * 1e-6):
                target_exact += 1
            elif (direction != "short" and exitp > tgt + eps) or (
                direction == "short" and exitp < tgt - eps
            ):
                target_above += 1
            else:
                target_below += 1
        elif reason == "stop" and stp is not None:
            stop_rets.append(_f(t, "return_pct"))
            if abs(exitp - stp) <= max(eps, abs(stp) * 1e-6):
                stop_exact += 1
            elif (direction != "short" and exitp < stp - eps) or (
                direction == "short" and exitp > stp + eps
            ):
                stop_beyond += 1
            else:
                stop_other += 1
        elif reason == "max_hold":
            hold_rets.append(_f(t, "return_pct"))

    asymmetry = (
        "Default limit fills: target exits at Target Price and stop exits at Stop Price "
        "(sample LTP is trigger evidence only). A configured 1:1 (e.g. 3%/3%) therefore "
        "realizes ~1:1 before fees. Optional execution.fill_at_sample_ltp applies to "
        "both target and stop together — never mixed."
    )

    return {
        "avg_winning_trade_net": round(gp_net / len(wins), 2) if wins else 0.0,
        "avg_losing_trade_net": round(-gl_net / len(losses), 2) if losses else 0.0,
        "avg_winning_trade_gross": (
            round(sum(_f(t, "gross_pnl") for t in gpos) / len(gpos), 2) if gpos else 0.0
        ),
        "avg_losing_trade_gross": (
            round(sum(_f(t, "gross_pnl") for t in gneg) / len(gneg), 2) if gneg else 0.0
        ),
        "gross_profit_before_fees": round(gp_before, 2),
        "gross_loss_before_fees": round(gl_before, 2),
        "profit_factor_before_fees": (
            round(gp_before / gl_before, 4) if gl_before > 0 else None
        ),
        "profit_factor_after_fees": round(gp_net / gl_net, 4) if gl_net > 0 else None,
        "profit_factor_formula": (
            "sum(net_pnl of winners) / abs(sum(net_pnl of losers))"
        ),
        "net_profit_formula": "sum(gross_pnl) − total_fees",
        "total_fees": round(fees, 2),
        "net_profit": round(net, 2),
        "gross_pnl_total": round(gross_total, 2),
        "exit_reason_counts": reasons,
        "target_exact": target_exact,
        "target_above": target_above,
        "target_below": target_below,
        "stop_exact": stop_exact,
        "stop_beyond_gap": stop_beyond,
        "stop_other": stop_other,
        "stop_trigger_beyond_sample": stop_trigger_beyond,
        "gap_beyond_stop_count": gap_count,
        "target_mean_return_pct": (
            round(sum(target_rets) / len(target_rets), 4) if target_rets else None
        ),
        "stop_mean_return_pct": (
            round(sum(stop_rets) / len(stop_rets), 4) if stop_rets else None
        ),
        "hold_mean_return_pct": (
            round(sum(hold_rets) / len(hold_rets), 4) if hold_rets else None
        ),
        "asymmetry_note": asymmetry,
    }


def compute_trade_metrics(trades: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Performance metrics from executed trades only.

    Must never receive candidate/skipped signals — callers pass the simulator
    trade list after Execution Rules filtering.
    """
    empty_episode = _empty_max_drawdown_episode()
    empty = {
        "trade_count": 0,
        "profit": 0.0,
        "win_rate_pct": 0.0,
        "profit_factor": None,
        "max_drawdown": 0.0,
        "avg_return_pct": 0.0,
        "avg_holding_sec": 0.0,
        "wins": 0,
        "losses": 0,
        "gross_profit": 0.0,
        "gross_loss": 0.0,
        "gross_pnl_total": 0.0,
        "net_profit": 0.0,
        "total_fees": 0.0,
        "avg_trade_pnl": 0.0,
        "avg_winning_trade": 0.0,
        "avg_losing_trade": 0.0,
        "expectancy": 0.0,
        "average_drawdown": 0.0,
        "equity_curve_points": 0,
        "metrics_from_executed_trades_only": True,
        "profit_factor_formula": "sum(net_pnl of winning trades) / abs(sum(net_pnl of losing trades))",
        "net_profit_formula": "sum(gross_pnl) − sum(fees)  [= sum(net_pnl)]",
        "outcome_audit": compute_outcome_audit([]),
        "max_drawdown_episode": empty_episode,
        "max_dd_peak_equity": 0.0,
        "max_dd_trough_equity": 0.0,
        "max_dd_peak_exit_ts": None,
        "max_dd_trough_exit_ts": None,
        "max_dd_method": empty_episode["method"],
        "account_equity_max_drawdown": 0.0,
        "max_portfolio_drawdown_open_risk": 0.0,
        "max_theoretical_portfolio_risk": 0.0,
        "stop_loss_per_trade_rupees": 0.0,
        "max_open_positions_for_risk": 0,
        "observed_max_concurrent_open": 0,
        "portfolio_risk": {},
    }
    if not trades:
        return empty

    wins = [t for t in trades if float(t.get("net_pnl") or 0) > 0]
    losses = [t for t in trades if float(t.get("net_pnl") or 0) <= 0]
    gross_profit = sum(float(t.get("net_pnl") or 0) for t in wins)
    gross_loss = abs(sum(float(t.get("net_pnl") or 0) for t in losses))
    total_fees = sum(float(t.get("fees") or 0) for t in trades)
    profit = sum(float(t.get("net_pnl") or 0) for t in trades)
    returns = [float(t.get("return_pct") or 0) for t in trades]
    holds = [float(t.get("holding_seconds") or 0) for t in trades]

    episode = compute_max_drawdown_episode(trades)
    curve = annotate_equity_curve_max_dd(build_equity_curve(trades), episode)
    max_dd = float(episode.get("max_drawdown") or 0)

    pf = (gross_profit / gross_loss) if gross_loss > 0 else None
    gross_total = sum(
        float(t.get("gross_pnl") if t.get("gross_pnl") is not None else t.get("net_pnl") or 0)
        for t in trades
    )
    n = len(trades)
    avg_trade_pnl = round(gross_total / n, 2) if n else 0.0
    expectancy = round(profit / n, 2) if n else 0.0

    return {
        "trade_count": n,
        "profit": round(profit, 2),
        "net_profit": round(profit, 2),
        "gross_pnl_total": round(gross_total, 2),
        "win_rate_pct": round(len(wins) / n * 100.0, 2) if n else 0.0,
        "profit_factor": round(pf, 4) if pf is not None else None,
        "profit_factor_formula": "sum(net_pnl of winning trades) / abs(sum(net_pnl of losing trades))",
        "profit_factor_basis": "net_pnl_after_fees",
        "max_drawdown": round(max_dd, 2),
        "account_equity_max_drawdown": round(max_dd, 2),
        "avg_return_pct": round(sum(returns) / n, 4) if returns else 0.0,
        "avg_holding_sec": round(sum(holds) / n, 2) if holds else 0.0,
        "avg_trade_pnl": avg_trade_pnl,
        "expectancy": expectancy,
        "average_drawdown": compute_average_drawdown(trades),
        "wins": len(wins),
        "losses": len(losses),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "avg_winning_trade": round(gross_profit / len(wins), 2) if wins else 0.0,
        "avg_losing_trade": round(-gross_loss / len(losses), 2) if losses else 0.0,
        "total_fees": round(total_fees, 2),
        "net_profit_formula": "sum(gross_pnl) − sum(fees)  [= sum(net_pnl)]",
        "equity_curve_points": len(curve),
        "metrics_from_executed_trades_only": True,
        "max_drawdown_episode": episode,
        "max_dd_peak_equity": episode.get("peak_equity"),
        "max_dd_trough_equity": episode.get("trough_equity"),
        "max_dd_peak_exit_ts": episode.get("peak_exit_ts"),
        "max_dd_trough_exit_ts": episode.get("trough_exit_ts"),
        "max_dd_method": episode.get("method"),
        "max_portfolio_drawdown_open_risk": 0.0,
        "max_theoretical_portfolio_risk": 0.0,
        "stop_loss_per_trade_rupees": 0.0,
        "max_open_positions_for_risk": 0,
        "observed_max_concurrent_open": 0,
        "portfolio_risk": {},
        "outcome_audit": compute_outcome_audit(trades),
    }


def _num(v: Any) -> float | None:
    try:
        x = float(v)
        return x if x == x else None
    except (TypeError, ValueError):
        return None


def stop_loss_rupees_for_trade(trade: dict[str, Any], *, stop_loss_pct: float) -> float:
    """Rupee risk if stop is hit exactly at entry (excludes slippage/gaps)."""
    entry = _num(trade.get("entry_price")) or 0.0
    qty = int(trade.get("qty") or 0)
    if entry <= 0 or qty <= 0 or stop_loss_pct <= 0:
        return 0.0
    return entry * qty * (stop_loss_pct / 100.0)


def compute_stop_loss_per_trade_rupees(
    trades: list[dict[str, Any]],
    *,
    cfg: dict[str, Any],
) -> float:
    """Typical stop loss ₹/trade from strategy stop % and position size."""
    from chain_replay_ml.strategy_registry.schema import normalize_strategy_config

    cfg = normalize_strategy_config(cfg)
    stop_pct = float(cfg["stop"].get("stop_loss_pct") or 0)
    pos = cfg["position_size"]
    qty = int(pos.get("lots") or 1) * int(pos.get("qty_per_lot") or 65)
    if trades:
        risks = [stop_loss_rupees_for_trade(t, stop_loss_pct=stop_pct) for t in trades]
        risks = [r for r in risks if r > 0]
        if risks:
            return round(sum(risks) / len(risks), 2)
    entry = cfg["entry"]
    mid = (float(entry.get("premium_min") or 0) + float(entry.get("premium_max") or 0)) / 2.0
    if mid <= 0:
        mid = float(entry.get("premium_min") or entry.get("premium_max") or 0)
    return round(mid * qty * (stop_pct / 100.0), 2) if mid > 0 and qty > 0 and stop_pct > 0 else 0.0


def compute_theoretical_portfolio_risk(
    *,
    cfg: dict[str, Any],
    execution_rules: dict[str, Any] | None,
    trades: list[dict[str, Any]],
    observed_max_concurrent_open: int = 0,
) -> dict[str, Any]:
    """
    Maximum Theoretical Portfolio Risk =
      Maximum Open Positions × Stop Loss Per Trade
    (excludes slippage / gaps).
    """
    from chain_replay_ml.strategy_registry.schema import normalize_strategy_config
    from chain_replay_ml.strategy_simulator.engine import normalize_execution_rules

    norm_cfg = normalize_strategy_config(cfg)
    rules = normalize_execution_rules(execution_rules)
    stop_per = compute_stop_loss_per_trade_rupees(trades, cfg=norm_cfg)
    if rules.get("enabled") and int(rules.get("max_open_positions") or 0) > 0:
        n_open = int(rules["max_open_positions"])
        n_source = "execution_rules.max_open_positions"
    elif observed_max_concurrent_open > 0:
        n_open = int(observed_max_concurrent_open)
        n_source = "observed_max_concurrent_open"
    else:
        n_open = 1
        n_source = "default_one"
    theoretical = round(n_open * stop_per, 2)
    return {
        "max_open_positions": n_open,
        "max_open_positions_source": n_source,
        "stop_loss_per_trade_rupees": stop_per,
        "stop_loss_pct": float(norm_cfg["stop"].get("stop_loss_pct") or 0),
        "max_theoretical_portfolio_risk": theoretical,
        "formula": f"{n_open} × {stop_per:.2f} = {theoretical:.2f}",
        "excludes_slippage_gaps": True,
    }


def _unrealized_gross(trade: dict[str, Any], mark_price: float) -> float:
    entry = _num(trade.get("entry_price"))
    qty = int(trade.get("qty") or 0)
    if entry is None or entry <= 0 or qty <= 0:
        return 0.0
    direction = str(trade.get("direction") or "long").lower()
    if direction == "short":
        return (entry - mark_price) * qty
    return (mark_price - entry) * qty


def _expected_stop_rupees(trade: dict[str, Any]) -> float:
    expected = _num(trade.get("expected_stop_loss_rupees"))
    if expected is None:
        expected = _num(trade.get("stop_risk_rupees"))
    if expected is not None and expected > 0:
        return float(expected)
    entry = _num(trade.get("entry_price")) or 0.0
    qty = int(trade.get("qty") or 0)
    stop_pct = _num(trade.get("stop_loss_pct")) or 0.0
    if entry > 0 and qty > 0 and stop_pct > 0:
        return entry * qty * (stop_pct / 100.0)
    return 0.0


def _raw_max_floating_loss(trade: dict[str, Any]) -> float:
    """Worst unrealized loss ₹ from engine live-path trough (or MAE fallback)."""
    low = _num(trade.get("lowest_unrealized_pnl"))
    if low is not None:
        return abs(min(0.0, low))
    mae = _num(trade.get("max_adverse_pct"))
    entry = _num(trade.get("entry_price")) or 0.0
    qty = int(trade.get("qty") or 0)
    if mae is not None and mae < 0 and entry > 0 and qty > 0:
        return abs(entry * qty * (mae / 100.0))
    return 0.0


def _trade_audit_snapshot(
    trade: dict[str, Any],
    *,
    floating_loss_rupees: float | None = None,
    applied_cap: bool = False,
) -> dict[str, Any]:
    entry = _num(trade.get("entry_price")) or 0.0
    qty = int(trade.get("qty") or 0)
    stop_pct = _num(trade.get("stop_loss_pct")) or 0.0
    stop_price = _num(trade.get("stop_price"))
    if stop_price is None and entry > 0 and stop_pct > 0:
        direction = str(trade.get("direction") or "long").lower()
        if direction == "short":
            stop_price = entry * (1.0 + stop_pct / 100.0)
        else:
            stop_price = entry * (1.0 - stop_pct / 100.0)
    expected = _expected_stop_rupees(trade)
    raw_loss = (
        float(floating_loss_rupees)
        if floating_loss_rupees is not None
        else _raw_max_floating_loss(trade)
    )
    gap = bool(trade.get("gap_beyond_stop"))
    # Material exceed = more than 1% above configured stop (not tiny float noise).
    material_exceed = raw_loss > expected * 1.01 + 1e-6 if expected > 0 else raw_loss > 0
    return {
        "trade_id": trade.get("trade_id"),
        "token": trade.get("token"),
        "trading_day": trade.get("trading_day"),
        "entry_ts": trade.get("entry_ts"),
        "exit_ts": trade.get("exit_ts"),
        "entry_price": trade.get("entry_price"),
        "quantity": qty,
        "qty": qty,
        "position_value": round(entry * qty, 2) if entry > 0 and qty > 0 else 0.0,
        "stop_price": round(stop_price, 4) if stop_price is not None else None,
        "stop_loss_pct": stop_pct,
        "lowest_mark_price": trade.get("lowest_mark_price"),
        "lowest_price_reached": trade.get("lowest_mark_price"),
        "lowest_live_ltp_before_exit": trade.get("lowest_mark_price"),
        "stop_trigger_ltp": trade.get("stop_trigger_ltp"),
        "sample_exit_ltp": trade.get("sample_exit_ltp"),
        "exit_price": trade.get("exit_price"),
        "expected_stop_loss_rupees": round(expected, 2),
        "actual_maximum_floating_loss_rupees": round(raw_loss, 2),
        "lowest_unrealized_pnl": _num(trade.get("lowest_unrealized_pnl")),
        "exit_reason": trade.get("exit_reason"),
        "gap_beyond_stop": gap,
        "stop_cap_applied": applied_cap,
        "exceeds_configured_stop": material_exceed and not gap,
        "exceeds_stop_due_to_gap": material_exceed and gap,
        "incorrect_metric_if_uncapped_walk": False,
    }


def compute_max_portfolio_drawdown_open_risk(
    trades: list[dict[str, Any]],
    price_rows: list[dict[str, Any]] | None = None,
    *,
    execution_rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Max Portfolio DD (Open Risk) from stop-aware per-trade live troughs.

    With Maximum Open Positions = 1 this is the worst single-trade floating loss
    on the live LTP path (already stop-enforced in the engine).

    Floating loss is capped at the configured stop ₹ unless ``gap_beyond_stop``
    (modeled gap through the stop on the exit tick). A free price-walk that
    ignores stops is NOT used for the reported metric when max_open <= 1.
    """
    from chain_replay_ml.strategy_simulator.engine import normalize_execution_rules

    empty = {
        "max_portfolio_drawdown_open_risk": 0.0,
        "worst_floating_pnl": 0.0,
        "at_ts": None,
        "open_positions_at_worst": 0,
        "observed_max_concurrent_open": 0,
        "method": "per_trade_stop_aware_live_trough",
        "mark_price_field": "ltp",
        "worst_trade": None,
        "stop_enforced_before_open_risk": True,
        "incorrect_metric": None,
        "note": (
            "Max Portfolio DD (Open) uses engine live-LTP troughs with stop "
            "enforcement; capped at stop ₹ unless gap-through-stop."
        ),
    }
    if not trades:
        return empty

    rules = normalize_execution_rules(execution_rules)
    max_open_cap = (
        int(rules["max_open_positions"])
        if rules.get("enabled") and int(rules.get("max_open_positions") or 0) > 0
        else None
    )

    enriched: list[dict[str, Any]] = []
    for t in trades:
        row = dict(t)
        if not row.get("direction"):
            row["direction"] = "long"
        # Lift audit fields persisted under meta.
        meta = row.get("meta") if isinstance(row.get("meta"), dict) else {}
        for k in (
            "stop_price",
            "stop_loss_pct",
            "expected_stop_loss_rupees",
            "stop_risk_rupees",
            "lowest_mark_price",
            "lowest_unrealized_pnl",
            "gap_beyond_stop",
            "direction",
        ):
            if row.get(k) is None and meta.get(k) is not None:
                row[k] = meta[k]
        enriched.append(row)

    # Per-trade stop-aware loss (cap unless gap).
    scored: list[tuple[float, float, dict[str, Any], dict[str, Any]]] = []
    for tr in enriched:
        expected = _expected_stop_rupees(tr)
        raw = _raw_max_floating_loss(tr)
        gap = bool(tr.get("gap_beyond_stop"))
        if expected > 0 and raw > expected * 1.01 and not gap:
            # Should not happen if stop is enforced on live path — treat as
            # data/engine inconsistency: cap for the metric, flag on audit.
            loss = expected
            applied_cap = True
        elif expected > 0 and not gap:
            loss = min(raw, expected)
            applied_cap = raw > expected + 1e-9
        else:
            loss = raw  # gap allowed, or no stop configured
            applied_cap = False
        audit = _trade_audit_snapshot(tr, floating_loss_rupees=raw, applied_cap=applied_cap)
        if applied_cap and not gap:
            audit["exceeds_configured_stop"] = True
            audit["incorrect_metric_if_uncapped_walk"] = True
        scored.append((loss, raw, tr, audit))

    if not scored:
        return empty

    scored.sort(key=lambda x: x[0], reverse=True)
    best_loss, best_raw, best_tr, best_audit = scored[0]

    # Observed concurrency from non-overlapping intervals (entry_ts, exit_ts).
    intervals = sorted(
        (
            (_num(t.get("entry_ts")) or 0.0, _num(t.get("exit_ts")) or 0.0)
            for t in enriched
        ),
        key=lambda iv: iv[0],
    )
    max_concurrent = 0
    open_ends: list[float] = []
    for start, end in intervals:
        open_ends = [e for e in open_ends if e > start]
        open_ends.append(end)
        max_concurrent = max(max_concurrent, len(open_ends))

    single_slot = max_open_cap == 1 or max_concurrent <= 1
    if single_slot or max_open_cap == 1:
        risk = best_loss
        method = "per_trade_stop_aware_live_trough_max_open_1"
        incorrect = None
        if best_raw > best_loss * 1.01 + 1e-6 and not best_audit.get("gap_beyond_stop"):
            incorrect = (
                "Raw floating path exceeded configured stop without gap; "
                "reported Max Portfolio DD (Open) is capped at stop ₹. "
                "Uncapped walk values are not used."
            )
        return {
            "max_portfolio_drawdown_open_risk": round(risk, 2),
            "worst_floating_pnl": round(-risk, 2),
            "raw_worst_floating_loss": round(best_raw, 2),
            "at_ts": best_tr.get("entry_ts"),
            "open_positions_at_worst": 1,
            "observed_max_concurrent_open": max_concurrent,
            "method": method,
            "mark_price_field": "ltp",
            "worst_trade": best_audit,
            "stop_enforced_before_open_risk": True,
            "incorrect_metric": incorrect,
            "which_metric_was_wrong": (
                "max_portfolio_drawdown_open_risk (uncapped price-walk) was incorrect; "
                "corrected to stop-aware per-trade trough"
                if incorrect
                else None
            ),
            "note": empty["note"],
        }

    # Multi-open: worst sum of concurrent stop-capped legs by sweep.
    events: list[tuple[float, int, str]] = []  # ts, kind(0=exit,1=enter), tid
    by_id = {str(t.get("trade_id") or id(t)): t for t in enriched}
    loss_by_id = {
        str(t.get("trade_id") or id(t)): loss for loss, _raw, t, _a in scored
    }
    for t in enriched:
        tid = str(t.get("trade_id") or id(t))
        ets = _num(t.get("entry_ts"))
        xts = _num(t.get("exit_ts"))
        if ets is not None:
            events.append((ets, 1, tid))
        if xts is not None:
            events.append((xts, 0, tid))
    events.sort(key=lambda e: (e[0], e[1]))
    active: set[str] = set()
    worst_sum = 0.0
    worst_n = 0
    worst_ids: list[str] = []
    for _ts, kind, tid in events:
        if kind == 0:
            active.discard(tid)
        else:
            active.add(tid)
        total = sum(loss_by_id.get(i, 0.0) for i in active)
        if total > worst_sum:
            worst_sum = total
            worst_n = len(active)
            worst_ids = list(active)

    contributor = by_id.get(worst_ids[0]) if worst_ids else best_tr
    # Pick audit for largest leg in the worst basket.
    leg = best_audit
    if worst_ids:
        legs = [(loss_by_id.get(i, 0.0), i) for i in worst_ids]
        legs.sort(reverse=True)
        top_id = legs[0][1]
        for loss, _raw, t, audit in scored:
            if str(t.get("trade_id") or id(t)) == top_id:
                leg = audit
                contributor = t
                break

    return {
        "max_portfolio_drawdown_open_risk": round(worst_sum, 2),
        "worst_floating_pnl": round(-worst_sum, 2),
        "at_ts": contributor.get("entry_ts") if contributor else None,
        "open_positions_at_worst": worst_n,
        "observed_max_concurrent_open": max_concurrent,
        "method": "concurrent_stop_capped_open_legs",
        "mark_price_field": "ltp",
        "worst_trade": leg,
        "stop_enforced_before_open_risk": True,
        "incorrect_metric": None,
        "note": empty["note"],
    }


def attach_portfolio_risk_metrics(
    metrics: dict[str, Any],
    *,
    trades: list[dict[str, Any]],
    price_rows: list[dict[str, Any]] | None = None,
    cfg: dict[str, Any] | None = None,
    execution_rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach open-risk + theoretical portfolio risk (separate from account equity Max DD)."""
    out = dict(metrics)
    account_dd = float(out.get("account_equity_max_drawdown", out.get("max_drawdown")) or 0)
    out["account_equity_max_drawdown"] = round(account_dd, 2)
    out["max_drawdown"] = round(account_dd, 2)  # legacy alias = account equity Max DD

    open_risk = compute_max_portfolio_drawdown_open_risk(
        trades,
        price_rows,
        execution_rules=execution_rules,
    )
    theoretical = compute_theoretical_portfolio_risk(
        cfg=cfg or {},
        execution_rules=execution_rules,
        trades=trades,
        observed_max_concurrent_open=int(open_risk.get("observed_max_concurrent_open") or 0),
    )
    out["max_portfolio_drawdown_open_risk"] = open_risk["max_portfolio_drawdown_open_risk"]
    out["max_theoretical_portfolio_risk"] = theoretical["max_theoretical_portfolio_risk"]
    out["stop_loss_per_trade_rupees"] = theoretical["stop_loss_per_trade_rupees"]
    out["max_open_positions_for_risk"] = theoretical["max_open_positions"]
    out["observed_max_concurrent_open"] = open_risk["observed_max_concurrent_open"]
    out["worst_open_risk_trade"] = open_risk.get("worst_trade")
    out["open_risk_incorrect_metric_note"] = open_risk.get("incorrect_metric")
    out["portfolio_risk"] = {
        "account_equity_max_drawdown": out["account_equity_max_drawdown"],
        "account_equity_max_drawdown_method": out.get("max_dd_method"),
        "max_portfolio_drawdown_open_risk": open_risk,
        "max_theoretical_portfolio_risk": theoretical,
        "worst_open_risk_trade": open_risk.get("worst_trade"),
        "stop_enforced_before_open_risk": open_risk.get("stop_enforced_before_open_risk", True),
        "mark_price_field": open_risk.get("mark_price_field", "ltp"),
        "which_metric_was_wrong": open_risk.get("which_metric_was_wrong"),
    }
    return out


def compute_fold_metrics(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_fold: dict[str, list[dict[str, Any]]] = {}
    for t in trades:
        fid = str(t.get("fold_id") or "")
        by_fold.setdefault(fid, []).append(t)
    out: list[dict[str, Any]] = []
    for fid, fold_trades in sorted(by_fold.items(), key=lambda kv: kv[0]):
        m = compute_trade_metrics(fold_trades)
        m["fold_id"] = fid
        m["fold_number"] = fold_trades[0].get("fold_number") if fold_trades else None
        out.append(m)
    return out
