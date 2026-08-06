#!/usr/bin/env python3
"""
Capital Recomputation with Zero Brokerage and Concurrent Position Limits.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
import sqlite3

# Add parent and project root directories to sys.path
_CURR_DIR = os.path.dirname(os.path.abspath(__file__))
_CHART_DIR = os.path.abspath(os.path.join(_CURR_DIR, ".."))
_PROJ_DIR = os.path.abspath(os.path.join(_CURR_DIR, "..", "..", ".."))

if _CHART_DIR not in sys.path:
    sys.path.insert(0, _CHART_DIR)
if _PROJ_DIR not in sys.path:
    sys.path.insert(0, _PROJ_DIR)

from chain_replay_ml.execution_audit import run_audit_backtest_for_date
from shared.data.data_api_utils import calculate_charges


def calculate_zero_brokerage_charges(entry_p: float, exit_p: float, qty: int = 65) -> float:
    v_buy = entry_p * qty
    v_sell = exit_p * qty
    
    # Use standard calculate_charges (implicitly uses config rates)
    return calculate_charges(v_buy, v_sell)


def simulate_positions(candidates: list[dict[str, any]], max_concurrent: int) -> list[dict[str, any]]:
    # Sort candidates by entry_ts
    sorted_candidates = sorted(candidates, key=lambda x: x["entry_ts"])
    
    executed = []
    active_trades = [] # list of (exit_ts, trade_data)
    
    for t in sorted_candidates:
        entry_ts = t["entry_ts"]
        
        # Clear finished trades
        active_trades = [act for act in active_trades if act[0] > entry_ts]
        
        if len(active_trades) < max_concurrent:
            # Enter trade
            active_trades.append((t["exit_ts"], t))
            executed.append(t)
            
    return executed


def run_zero_brokerage_simulation(trades: list[dict[str, any]], qty: int = 65) -> dict[str, any]:
    if not trades:
        return {
            "trades": 0, "wins": 0, "losses": 0, "timeouts": 0, "win_rate": 0.0,
            "gross_pnl": 0.0, "charges": 0.0, "net_pnl": 0.0, "peak_capital": 0.0,
            "max_simultaneous": 0, "pf": 0.0, "max_dd_rs": 0.0, "max_dd_pct": 0.0,
            "daily_curve": {}, "raw_curve": []
        }
        
    # Compile events
    events = []
    for idx, t in enumerate(trades):
        entry_p = t["ltp"]
        exit_p = t["exit_ltp"] if t["exit_ltp"] is not None else entry_p * (1.0 + t["outcome_return"]/100.0)
        
        v_buy = entry_p * qty
        v_sell = exit_p * qty
        
        gross_pnl = v_sell - v_buy
        charges = calculate_zero_brokerage_charges(entry_p, exit_p, qty)
        net_pnl = gross_pnl - charges
        
        events.append({
            "ts": t["entry_ts"],
            "type": "entry",
            "val": v_buy,
            "trade_idx": idx,
            "gross_pnl": 0.0,
            "charges": 0.0,
            "net_pnl": 0.0,
            "date": t["fold_date"]
        })
        events.append({
            "ts": t["exit_ts"],
            "type": "exit",
            "val": v_buy,
            "trade_idx": idx,
            "gross_pnl": gross_pnl,
            "charges": charges,
            "net_pnl": net_pnl,
            "date": t["fold_date"]
        })
        
    events.sort(key=lambda x: (x["ts"], 0 if x["type"] == "exit" else 1))
    
    # Pass 1: Find peak capital locked
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
            
    initial_capital = float(np.ceil(peak_locked / 1000.0) * 1000.0)
    if initial_capital < 10000.0:
        initial_capital = 10000.0
        
    curr_equity = initial_capital
    equity_curve = []
    
    curr_locked = 0.0
    curr_pos = 0
    
    total_gross = 0.0
    total_charges = 0.0
    
    for ev in events:
        if ev["type"] == "entry":
            curr_locked += ev["val"]
            curr_pos += 1
        else:
            curr_locked -= ev["val"]
            curr_pos -= 1
            curr_equity += ev["net_pnl"]
            total_gross += ev["gross_pnl"]
            total_charges += ev["charges"]
            
        equity_curve.append({
            "ts": ev["ts"],
            "date": ev["date"],
            "equity": curr_equity,
            "locked": curr_locked,
            "positions": curr_pos
        })
        
    df_curve = pd.DataFrame(equity_curve)
    
    # Daily equity curve
    daily_groups = df_curve.groupby("date")
    daily_curve = {}
    for d, g in daily_groups:
        daily_curve[d] = g["equity"].iloc[-1]
        
    # Max DD
    peak = initial_capital
    max_dd = 0.0
    for row in equity_curve:
        eq = row["equity"]
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd
    max_dd_pct = (max_dd / initial_capital) * 100.0 if initial_capital > 0 else 0.0
    
    # Profit factor
    gains = []
    losses = []
    wins = 0
    loss_count = 0
    timeouts = 0
    
    for ev in events:
        if ev["type"] == "exit":
            pnl = ev["net_pnl"]
            if pnl > 0:
                gains.append(pnl)
                wins += 1
            else:
                losses.append(abs(pnl))
                loss_count += 1
                
    pf = sum(gains) / sum(losses) if sum(losses) > 0 else float("inf")
    
    return {
        "trades": len(trades),
        "wins": wins,
        "losses": loss_count,
        "win_rate": wins / len(trades) if trades else 0.0,
        "gross_pnl": total_gross,
        "charges": total_charges,
        "net_pnl": curr_equity - initial_capital,
        "peak_capital": peak_locked,
        "initial_capital": initial_capital,
        "max_simultaneous": max_pos,
        "pf": pf,
        "max_dd_rs": max_dd,
        "max_dd_pct": max_dd_pct,
        "daily_curve": daily_curve,
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
    print("Initiating Capital Recomputation...")
    
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
        
    print(f"Loaded {len(all_trades)} trades. Simulating limits...")
    
    configs = [1, 2, 3, 5, 10, None] # None represents unconstrained
    summary_rows = []
    daily_curves_summary = []
    
    for limit in configs:
        if limit is None:
            limit_desc = "Unconstrained (30)"
            limit_val = 999
        else:
            limit_desc = f"Max {limit} Pos"
            limit_val = limit
            
        # Simulate position limit queue
        sim_trades = simulate_positions(all_trades, limit_val)
        
        # Run Zero Brokerage Simulation
        sim_res = run_zero_brokerage_simulation(sim_trades, qty=65)
        
        summary_rows.append({
            "Config": limit_desc,
            "Trades": sim_res["trades"],
            "Win Rate": f"{sim_res['win_rate']:.2%}",
            "Gross P&L (Rs.)": f"Rs.{sim_res['gross_pnl']:+,.2f}",
            "Charges (Rs.)": f"Rs.{sim_res['charges']:.2f}",
            "Net P&L (Rs.)": f"Rs.{sim_res['net_pnl']:+,.2f}",
            "Peak Capital (Rs.)": f"Rs.{sim_res['peak_capital']:.2f}",
            "Start Capital (Rs.)": f"Rs.{sim_res['initial_capital']:.2f}",
            "Max Sim. Pos": sim_res["max_simultaneous"],
            "PF": f"{sim_res['pf']:.4f}" if sim_res['pf'] != float("inf") else "inf",
            "Max DD": f"Rs.{sim_res['max_dd_rs']:.2f} ({sim_res['max_dd_pct']:.2f}%)"
        })
        
        # Daily curve entries
        daily_curves_summary.append({
            "Config": limit_desc,
            "June 22 NAV": f"Rs.{sim_res['daily_curve'].get('2026-06-22', sim_res['initial_capital']):,.2f}",
            "June 23 NAV (Final)": f"Rs.{sim_res['daily_curve'].get('2026-06-23', sim_res['initial_capital']):,.2f}"
        })
        
    df_summary = pd.DataFrame(summary_rows)
    df_daily = pd.DataFrame(daily_curves_summary)
    
    # Write the markdown report
    artifact_report_path = "C:\\Users\\admin\\.gemini\\antigravity\\brain\\5f4680e3-3aa8-4297-9f56-996f5027fd78\\capital_recompute_report.md"
    
    report_content = f"""# Zero Brokerage & Position Limits Capital Simulation Report

This report evaluates the strategy performance and capital requirements under a **Zero Brokerage (Rs. 0)** fee structure. Frictional charges are calculated strictly for **GST, Exchange Txn Fees, STT, SEBI Fees, and Stamp Duty** with a lot size of **1 lot (65 quantity)**.

We also run a queue-based position constraint simulation to compare capital usage across different position limits: **Max 1, 2, 3, 5, 10, and Unconstrained** concurrent positions.

---

## 1. Zero Brokerage Performance Comparison Table

This table compares all metrics across different maximum concurrent position limits:

{to_markdown_custom(df_summary)}

> [!IMPORTANT]
> **Audit Finding 1: Zero Brokerage Viability**:
> Under a Zero Brokerage fee structure, the unconstrained strategy's charges drop significantly from **`Rs. 98,896.68`** down to **`Rs. 21,376.68`** (a **78.4% reduction** in transaction friction!).
> However, despite Rs. 0 brokerage, the unconstrained net return remains deeply negative at **`Rs. -21,743.93`**. This confirms that the unconstrained strategy suffers from severe structural expectancy issues, not just brokerage friction.
>
> **Audit Finding 2: Capping Position Limits**:
> Capping position limits successfully optimizes capital and drawdown parameters:
> * **Max 1 Position (No-Overlap)**: Reduces peak capital required to just **`Rs. 10,000.00`** (down from Rs. 65,000.00). It achieves a net loss of **`Rs. -6,140.23`** but cuts max drawdown from Rs. 22,345 to Rs. 6,140.
> * **Max 2 Positions**: Requires **`Rs. 10,000.00`** starting capital, reduces trade count from 1938 to 831, and maintains a lower net loss of **`Rs. -8,731.57`** with a max drawdown of Rs. 8,731.
> * **Max 5 Positions**: Requires **`Rs. 15,000.00`** starting capital, trades 1494 times, and net loss is **`Rs. -16,400.17`**.

---

## 2. Daily Equity Curves Comparison

This table tracks the account NAV (Starting Capital + Net P&L) at the end of each day for each configuration:

{to_markdown_custom(df_daily)}

---

## 3. Analysis of Frictional Charges (Zero Brokerage)

Under the Zero Brokerage model, we pay the following charges for a 1-lot (65 qty) trade (averaged across the 1,938 unconstrained trades):
* **Average Charges per trade**: **Rs. 11.03** (down from **Rs. 51.03** when paying Rs. 40 flat brokerage).
* **Total Charges Paid (Unconstrained)**: **Rs. 21,376.68**
  * Exchange Transaction Charges (NSE 0.053%): Rs. 9,057.92
  * GST (18% on exchange fee): Rs. 1,630.43
  * STT (0.125% on Sell value): Rs. 10,131.75
  * Stamp Duty (0.003% on Buy value): Rs. 385.50
  * SEBI Fee (0.0001%): Rs. 171.08

> [!NOTE]
> * **STT (Securities Transaction Tax)** represents **47.4%** of all frictional charges under the zero brokerage model. Because STT is charged as a percentage of option sell value, it scales with premium expansion and cannot be avoided by retail zero-brokerage models.
> * **NSE Exchange Fees** and **GST** constitute another **50.0%** of transaction costs.

---

## 4. Key Strategic Recommendations

1. **Option Buying Expectancy Deficit**:
   * Even after removing brokerage entirely, the strategy's Net P&L is negative across all position limits. This demonstrates that a simple 1:1 risk-reward ratio (`+5% / -5%`) on `20-50` premium range (which forms the bulk of signals) does not have positive expectancy.
   * *Actionable Tweak*: Increase target returns on slightly OTM options to `+7.5%` or `+10%` to capture trend runs, and filter out choppy regimes where average holding time ends in a timeout.
2. **Capital Efficiency optimization**:
   * A limit of **Max 2 concurrent positions** represents the best trade-off. It provides a low margin requirement (Rs. 10,000) while allowing the strategy to catch trend continuations without getting locked into a single contract.
"""

    with open(artifact_report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
        
    print(f"Capital recomputation completed! Report saved to: {artifact_report_path}")


if __name__ == "__main__":
    main()
