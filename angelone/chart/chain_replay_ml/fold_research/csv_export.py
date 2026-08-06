"""Export fold research dashboard sections to a single CSV file."""

from __future__ import annotations

import csv
import io
from typing import Any, Iterable

from .service import _load_fold_trades


def _section(
    writer: csv.writer,
    title: str,
    headers: list[str],
    rows: Iterable[list[Any]],
) -> None:
    writer.writerow([title])
    writer.writerow(headers)
    for row in rows:
        writer.writerow(row)
    writer.writerow([])


def _overview_rows(detail: dict[str, Any]) -> list[list[Any]]:
    fold = detail.get("fold") or {}
    pq = detail.get("prediction_quality") or {}
    trading = detail.get("trading") or {}
    tm = trading.get("metrics") or {}
    market = detail.get("market_summary") or {}
    pred_run = detail.get("prediction_run") or {}

    mae = fold.get("mae") if fold.get("mae") is not None else pq.get("mae")
    direction = (
        fold.get("directional_accuracy_pct")
        if fold.get("directional_accuracy_pct") is not None
        else pq.get("directional_accuracy_pct")
    )

    rows = [
        ["Fold Number", fold.get("fold_number")],
        ["Fold ID", fold.get("fold_id")],
        ["Model", pred_run.get("model_id")],
        ["Prediction Run", pred_run.get("run_id")],
        ["Prediction Rows", fold.get("validation_rows") or pq.get("row_count")],
        ["Trades", trading.get("trade_count") or tm.get("trade_count")],
        ["Profit", tm.get("profit")],
        ["Win Rate %", tm.get("win_rate_pct")],
        ["MAE", mae],
        ["Direction %", direction],
        ["Profit Factor", tm.get("profit_factor")],
        ["Max Drawdown", tm.get("max_drawdown")],
        ["Wins", tm.get("wins")],
        ["Losses", tm.get("losses")],
        ["Avg Trade PnL", tm.get("avg_trade_pnl")],
        ["Expectancy", tm.get("expectancy")],
        ["Trading Days", ", ".join(market.get("trading_days") or [])],
        ["Spot Start", market.get("spot_start")],
        ["Spot End", market.get("spot_end")],
        ["Spot Trend %", market.get("spot_trend_pct")],
        ["Volatility Proxy %", market.get("volatility_proxy_pct")],
        ["Token Count", market.get("token_count")],
        ["Time Span (s)", market.get("timestamp_span_sec")],
    ]
    return rows


def _prediction_rows(detail: dict[str, Any]) -> tuple[list[list[Any]], list[list[Any]]]:
    pq = detail.get("prediction_quality") or {}
    metrics = [
        ["Rows", pq.get("row_count")],
        ["MAE", pq.get("mae")],
        ["RMSE", pq.get("rmse")],
        ["Median Error", pq.get("median_error")],
        ["P95 Error", pq.get("p95_error")],
        ["Max Error", pq.get("max_error")],
        ["Bias", pq.get("bias")],
        ["Bias %", pq.get("bias_pct")],
        ["Direction %", pq.get("directional_accuracy_pct")],
    ]
    calibration = [
        [
            b.get("bin"),
            b.get("count"),
            b.get("pred_return_avg_pct"),
            b.get("actual_return_avg_pct"),
            b.get("calibration_error_pct"),
        ]
        for b in (pq.get("calibration_buckets") or [])
    ]
    return metrics, calibration


def _trading_rows(detail: dict[str, Any], trades: list[dict[str, Any]]) -> tuple[list[list[Any]], list[list[Any]]]:
    tm = (detail.get("trading") or {}).get("metrics") or {}
    summary = [
        ["Trade Count", tm.get("trade_count")],
        ["Profit", tm.get("profit")],
        ["Win Rate %", tm.get("win_rate_pct")],
        ["Profit Factor", tm.get("profit_factor")],
        ["Max Drawdown", tm.get("max_drawdown")],
        ["Gross Profit", tm.get("gross_profit")],
        ["Gross Loss", tm.get("gross_loss")],
        ["Total Fees", tm.get("total_fees")],
    ]
    trade_rows = []
    for i, t in enumerate(trades, start=1):
        trade_rows.append([
            i,
            t.get("trade_id"),
            t.get("trading_day"),
            t.get("token"),
            t.get("entry_ts"),
            t.get("exit_ts"),
            t.get("entry_price"),
            t.get("exit_price"),
            t.get("net_pnl"),
            t.get("gross_pnl"),
            t.get("return_pct"),
            t.get("holding_seconds"),
            t.get("exit_reason"),
        ])
    return summary, trade_rows


def _error_rows(detail: dict[str, Any], *, limit: int = 1000) -> list[list[Any]]:
    explorer = detail.get("error_explorer") or {}
    out: list[list[Any]] = []
    for mode in ("absolute", "positive", "negative"):
        for rank, row in enumerate(explorer.get(mode) or [], start=1):
            if rank > limit:
                break
            out.append([
                mode,
                rank,
                row.get("prediction_id"),
                row.get("timestamp"),
                row.get("trading_day"),
                row.get("token"),
                row.get("ltp"),
                row.get("predicted_ltp"),
                row.get("actual_ltp"),
                row.get("prediction_error"),
                row.get("abs_error"),
                row.get("direction_correct"),
            ])
    return out


def _drift_rows(detail: dict[str, Any]) -> list[list[Any]]:
    drift = detail.get("feature_drift") or {}
    return [
        [
            r.get("feature"),
            r.get("train_mean"),
            r.get("validation_mean"),
            r.get("shift_pct"),
            r.get("severity"),
        ]
        for r in (drift.get("top_drifted") or [])
    ]


def _regime_rows(detail: dict[str, Any]) -> list[list[Any]]:
    reg = detail.get("regime_analysis") or {}
    rows = [
        [
            r.get("regime"),
            r.get("row_count"),
            r.get("mae"),
        ]
        for r in (reg.get("regimes") or [])
    ]
    if reg.get("volatility_regime"):
        rows.insert(0, ["Volatility Regime", reg.get("volatility_regime"), reg.get("volatility_proxy_pct")])
    return rows


def build_fold_research_csv(
    data_dir: str,
    detail: dict[str, Any],
    *,
    error_limit: int = 1000,
) -> str:
    """Build a multi-section CSV string from a get_fold_research() document."""
    if not detail.get("ok"):
        raise ValueError(detail.get("error") or "fold research not loaded")

    trades: list[dict[str, Any]] = []
    trading = detail.get("trading") or {}
    strat_id = trading.get("strategy_run_id")
    fold_id = (detail.get("fold") or {}).get("fold_id")
    if strat_id and fold_id:
        trades = _load_fold_trades(data_dir, strat_id, str(fold_id))
    else:
        trades = list(trading.get("trades") or [])

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")

    _section(writer, "Overview", ["metric", "value"], _overview_rows(detail))

    pred_metrics, calibration = _prediction_rows(detail)
    _section(writer, "Prediction", ["metric", "value"], pred_metrics)
    _section(
        writer,
        "Prediction Calibration",
        ["bucket", "count", "prediction_pct", "actual_pct", "difference_pct"],
        calibration,
    )

    trade_summary, trade_rows = _trading_rows(detail, trades)
    _section(writer, "Trading Summary", ["metric", "value"], trade_summary)
    _section(
        writer,
        "Trading",
        [
            "trade_num", "trade_id", "trading_day", "token",
            "entry_ts", "exit_ts", "entry_price", "exit_price",
            "net_pnl", "gross_pnl", "return_pct", "holding_seconds", "exit_reason",
        ],
        trade_rows,
    )

    _section(
        writer,
        "Error Explorer",
        [
            "mode", "rank", "prediction_id", "timestamp", "trading_day", "token",
            "ltp", "predicted_ltp", "actual_ltp", "error", "abs_error", "direction_correct",
        ],
        _error_rows(detail, limit=error_limit),
    )

    drift = detail.get("feature_drift") or {}
    _section(
        writer,
        "Feature Drift",
        ["feature", "train_mean", "validation_mean", "shift_pct", "severity"],
        _drift_rows(detail),
    )
    if drift.get("note"):
        writer.writerow(["Feature Drift Note", drift.get("note")])
        writer.writerow([])

    _section(writer, "Regime", ["regime", "rows", "mae"], _regime_rows(detail))
    reg = detail.get("regime_analysis") or {}
    if reg.get("note"):
        writer.writerow(["Regime Note", reg.get("note")])
        writer.writerow([])

    return buf.getvalue()
