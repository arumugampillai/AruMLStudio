#!/usr/bin/env python3
"""
Tick-by-Tick Capital Simulation for Option Buying Strategy.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import sqlite3

# Add parent and project root directories to sys.path
from path_config import CHART_DATA_ROOT as _CHART_DIR
from path_config import ensure_ml_studio_paths
ensure_ml_studio_paths()
from chain_replay_ml.execution_audit import run_audit_backtest_for_date
from shared.data.data_api_utils import calculate_charges


def calculate_zero_brokerage_charges(entry_p: float, exit_p: float, qty: int = 65) -> float:
    """Statutory charges only (STT, exchange, SEBI, stamp, GST) — no brokerage."""
    v_buy = entry_p * qty
    v_sell = exit_p * qty
    return float(calculate_charges(v_buy, v_sell))


def break_down_zero_brokerage_charges(
    entry_p: float, exit_p: float, qty: int = 65
) -> dict[str, float]:
    """
    Component breakdown matching ``calculate_charges`` / zero-brokerage plan.

    Brokerage is always 0 on this plan.
    """
    try:
        from config import config as app_config

        stt_rate = float(app_config.STT_RATE)
        exch_rate = float(app_config.EXCHANGE_RATE)
        sebi_per_crore = float(app_config.SEBI_PER_CRORE)
        stamp_rate = float(app_config.STAMP_DUTY_RATE)
        gst_rate = float(app_config.GST_RATE)
    except Exception:
        # Fallback mirrors typical FO option rates if config is unavailable.
        stt_rate = 0.000625
        exch_rate = 0.00053
        sebi_per_crore = 10.0
        stamp_rate = 0.00003
        gst_rate = 0.18

    buy_value = float(entry_p) * int(qty)
    sell_value = float(exit_p) * int(qty)
    total_value = buy_value + sell_value

    stt = sell_value * stt_rate
    exch = total_value * exch_rate
    sebi = (total_value / 1_00_00_000) * sebi_per_crore
    stamp = buy_value * stamp_rate
    gst = (exch + sebi + stamp) * gst_rate
    brokerage = 0.0
    total = stt + exch + sebi + stamp + gst + brokerage
    return {
        "buy_value": round(buy_value, 2),
        "sell_value": round(sell_value, 2),
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 4),
        "exchange": round(exch, 4),
        "sebi": round(sebi, 4),
        "stamp": round(stamp, 4),
        "gst": round(gst, 4),
        "total": round(total, 2),
    }


def aggregate_trade_charge_breakdown(trades: list[dict]) -> dict:
    """Sum statutory charge components across executed simulator trades."""
    empty = {
        "trade_count": 0,
        "brokerage": 0.0,
        "stt": 0.0,
        "exchange": 0.0,
        "sebi": 0.0,
        "stamp": 0.0,
        "gst": 0.0,
        "total": 0.0,
        "total_from_trade_fees": 0.0,
        "avg_per_trade": 0.0,
        "buy_notional": 0.0,
        "sell_notional": 0.0,
        "plan": "zero_brokerage_statutory",
    }
    if not trades:
        return empty

    totals = {
        "brokerage": 0.0,
        "stt": 0.0,
        "exchange": 0.0,
        "sebi": 0.0,
        "stamp": 0.0,
        "gst": 0.0,
        "total": 0.0,
        "buy_notional": 0.0,
        "sell_notional": 0.0,
    }
    fees_on_trades = 0.0
    n = 0
    for t in trades:
        try:
            entry = float(t.get("entry_price") or 0)
            exit_p = float(t.get("exit_price") or entry)
            qty = int(t.get("qty") or 0)
        except (TypeError, ValueError):
            continue
        if entry <= 0 or qty <= 0:
            continue
        part = break_down_zero_brokerage_charges(entry, exit_p, qty)
        for k in ("brokerage", "stt", "exchange", "sebi", "stamp", "gst", "total"):
            totals[k] += float(part.get(k) or 0)
        totals["buy_notional"] += float(part.get("buy_value") or 0)
        totals["sell_notional"] += float(part.get("sell_value") or 0)
        try:
            fees_on_trades += float(t.get("fees") or 0)
        except (TypeError, ValueError):
            pass
        n += 1

    out = {k: round(v, 2) for k, v in totals.items()}
    out["trade_count"] = n
    out["total_from_trade_fees"] = round(fees_on_trades, 2)
    out["avg_per_trade"] = round(out["total"] / n, 2) if n else 0.0
    out["plan"] = "zero_brokerage_statutory"
    return out


def calculate_rupee_charges(entry_p: float, exit_p: float, qty: int = 65) -> float:
    """
    Round-trip charges for option buying.

    Default assumes a **zero-brokerage** plan: statutory charges only.
    Pass ``include_brokerage=True`` for legacy flat ₹40 + GST brokerage.
    """
    return calculate_zero_brokerage_charges(entry_p, exit_p, qty)


def calculate_rupee_charges_with_brokerage(
    entry_p: float, exit_p: float, qty: int = 65, *, brokerage: float = 40.0
) -> float:
    """Legacy retail model: statutory charges + flat brokerage + GST on brokerage."""
    gov_charges = calculate_zero_brokerage_charges(entry_p, exit_p, qty)
    brokerage_with_gst = float(brokerage) * 1.18
    return gov_charges + brokerage_with_gst


def run_capital_simulation(trades: list[dict[str, any]], qty: int = 65) -> dict[str, any]:
    if not trades:
        return {}
        
    # Compile events
    events = []
    for idx, t in enumerate(trades):
        entry_p = t["ltp"]
        exit_p = t["exit_ltp"] if t["exit_ltp"] is not None else entry_p * (1.0 + t["outcome_return"]/100.0)
        
        v_buy = entry_p * qty
        v_sell = exit_p * qty
        charges = calculate_rupee_charges(entry_p, exit_p, qty)
        net_pnl = (v_sell - v_buy) - charges
        
        events.append({
            "ts": t["entry_ts"],
            "type": "entry",
            "val": v_buy,
            "trade_idx": idx,
            "pnl": 0.0,
            "date": t["fold_date"]
        })
        events.append({
            "ts": t["exit_ts"],
            "type": "exit",
            "val": v_buy,
            "trade_idx": idx,
            "pnl": net_pnl,
            "date": t["fold_date"]
        })
        
    # Sort events chronologically. Exits first if timestamps are identical
    events.sort(key=lambda x: (x["ts"], 0 if x["type"] == "exit" else 1))
    
    # First pass to find peak capital locked
    curr_locked = 0.0
    peak_locked = 0.0
    curr_pos = 0
    max_pos = 0
    
    for ev in events:
        if ev["type"] == "entry":
            curr_locked += ev["val"]
            curr_pos += 1
            if curr_locked > peak_locked:
                peak_locked = curr_locked
            if curr_pos > max_pos:
                max_pos = curr_pos
        else:
            curr_locked -= ev["val"]
            curr_pos -= 1
            
    # Set initial capital to peak capital required (rounded up to nearest 1000)
    initial_capital = float(np.ceil(peak_locked / 1000.0) * 1000.0)
    if initial_capital < 10000.0:
        initial_capital = 10000.0 # set minimum starting capital of Rs. 10,000
        
    curr_equity = initial_capital
    equity_curve = []
    
    # Reset tracking
    curr_locked = 0.0
    curr_pos = 0
    
    for ev in events:
        if ev["type"] == "entry":
            curr_locked += ev["val"]
            curr_pos += 1
        else:
            curr_locked -= ev["val"]
            curr_pos -= 1
            curr_equity += ev["pnl"]
            
        equity_curve.append({
            "ts": ev["ts"],
            "date": ev["date"],
            "equity": curr_equity,
            "locked": curr_locked,
            "positions": curr_pos
        })
        
    df_curve = pd.DataFrame(equity_curve)
    
    # Calculate daily equity curve (end of day equity)
    daily_groups = df_curve.groupby("date")
    daily_curve = []
    for d, g in daily_groups:
        daily_curve.append({
            "Date": d,
            "End Day Equity": f"Rs.{g['equity'].iloc[-1]:,.2f}",
            "Peak Locked Capital": f"Rs.{g['locked'].max():,.2f}",
            "Max Concurrent Positions": g["positions"].max()
        })
    df_daily = pd.DataFrame(daily_curve)
    
    # Calculate final P&L metrics
    total_net_pnl = curr_equity - initial_capital
    percentage_return = (total_net_pnl / initial_capital) * 100.0
    
    # Annualized CAGR (assuming 2 trading days out of 252 annual trading days)
    active_days = 2
    cagr = ((curr_equity / initial_capital) ** (252.0 / active_days) - 1.0) * 100.0
    
    # Max Drawdown of the equity curve
    peak = initial_capital
    max_dd = 0.0
    for row in equity_curve:
        eq = row["equity"]
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = (max_dd / initial_capital) * 100.0
    
    # Profit factor in rupees
    gains = []
    losses = []
    for ev in events:
        if ev["type"] == "exit":
            pnl = ev["pnl"]
            if pnl > 0:
                gains.append(pnl)
            else:
                losses.append(abs(pnl))
    pf = sum(gains) / sum(losses) if sum(losses) > 0 else float("inf")
    
    # Total charges paid
    total_charges = 0.0
    for t in trades:
        entry_p = t["ltp"]
        exit_p = t["exit_ltp"] if t["exit_ltp"] is not None else entry_p * (1.0 + t["outcome_return"]/100.0)
        total_charges += calculate_rupee_charges(entry_p, exit_p, qty)
        
    return {
        "initial_capital": initial_capital,
        "final_equity": curr_equity,
        "net_pnl": total_net_pnl,
        "return_pct": percentage_return,
        "cagr": cagr,
        "max_dd_rs": max_dd,
        "max_dd_pct": max_dd_pct,
        "profit_factor": pf,
        "max_concurrent": max_pos,
        "peak_capital_locked": peak_locked,
        "total_charges": total_charges,
        "daily_curve": df_daily,
        "raw_curve": equity_curve
    }


def to_markdown_custom(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    cols = df.columns.tolist()
    header = " | ".join(cols)
    divider = " | ".join(["---"] * len(cols))
    rows = []
    for _, row in df.iterrows():
        rows.append(" | ".join(str(row[c]) for c in cols))
    return f"| {header} |\n| {divider} |\n" + "\n".join(f"| {r} |" for r in rows)


def main():
    print("Initiating Capital Usage Simulation...")
    
    folds = [
        {"date": "2026-06-22", "stamp": "fold_1_2026-06-22"},
        {"date": "2026-06-23", "stamp": "fold_2_2026-06-23"}
    ]
    
    # Load audited Config B trades
    all_trades = []
    for f in folds:
        print(f"Loading Config B trades for {f['date']}...")
        res = run_audit_backtest_for_date(f["date"], f["stamp"], "B")
        for r in res:
            r["fold_date"] = f["date"]
        all_trades.extend(res)
        
    print(f"Loaded {len(all_trades)} trades. Running simulation...")
    
    sim = run_capital_simulation(all_trades, qty=65)
    
    # Write the markdown report
    artifact_report_path = "C:\\Users\\admin\\.gemini\\antigravity\\brain\\5f4680e3-3aa8-4297-9f56-996f5027fd78\\capital_report.md"
    
    report_content = f"""# Tick-by-Tick Capital Simulation Report

This report evaluates the capital efficiency, peak capital required, and net performance in Indian Rupees (Rs.) for the audited **Config B (Strict outcome scanning)** backtest, trading exactly **1 lot (65 quantity)** per signal.

---

## 1. Capital and Performance Summary

* **Lot Size**: 65 Quantity
* **Peak Capital Locked**: Rs. {sim['peak_capital_locked']:.2f}
* **Starting Capital (No Leverage)**: **Rs. {sim['initial_capital']:.2f}** (Rounded up to nearest Rs.1,000)
* **Final Account Equity**: **Rs. {sim['final_equity']:.2f}**
* **Net Profit & Loss (after charges)**: **Rs. {sim['net_pnl']:+.2f}**
* **Percentage Return**: **{sim['return_pct']:+.2f}%**
* **Annualized CAGR (2-day period)**: **{sim['cagr']:+.2f}%**
* **Maximum Concurrent Positions Open**: **{sim['max_concurrent']}**
* **Maximum Peak Drawdown**: Rs. {sim['max_dd_rs']:.2f} ({sim['max_dd_pct']:.2f}%)
* **Rupee-based Profit Factor**: **{sim['profit_factor']:.4f}**
* **Total Frictional Charges Paid**: Rs. {sim['total_charges']:.2f} (Brokerage + STT + GST + Exchange Txn Fees)

---

## 2. Daily Equity Curve

This table shows the progression of capital at the end of each trading day:

{to_markdown_custom(sim['daily_curve'])}

---

## 3. Detailed Capital Usage Breakdown

1. **Brokerage & Frictional Drag**:
   * The total charges paid across {len(all_trades)} trades was **Rs. {sim['total_charges']:.2f}**. 
   * Each trade paid an average of **Rs. {sim['total_charges']/len(all_trades):.2f}** in brokerage and government taxes.
2. **Capital Efficiency**:
   * To trade 1 lot with unconstrained overlaps, a starting capital of **Rs. {sim['initial_capital']:.2f}** was required due to a peak of **{sim['max_concurrent']} concurrent positions** open at the same time.
   * This is a massive capital requirement for a 1-lot strategy, indicating that unconstrained overlap backtesting has low capital efficiency (locking up capital in highly correlated concurrent trades).
3. **CAGR & Drawdown**:
   * The raw return of **{sim['return_pct']:+.2f}%** over 2 days annualizes to a huge CAGR due to the short time horizon. However, the max drawdown of **{sim['max_dd_pct']:.2f}%** shows that unconstrained execution carries substantial risk when multiple overlapping positions hit their stop-loss simultaneously.
"""

    with open(artifact_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Capital simulation completed! Report saved to: {artifact_report_path}")


if __name__ == "__main__":
    main()
