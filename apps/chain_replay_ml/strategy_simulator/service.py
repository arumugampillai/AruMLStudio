"""Run strategy simulation on prediction runs or Research Lab prediction datasets."""

from __future__ import annotations

import copy
import json
import os
import uuid
from typing import Any

from chain_replay_ml.prediction_runs.store import PredictionRunStore
from chain_replay_ml.strategy_registry.schema import normalize_strategy_config
from chain_replay_ml.strategy_registry.service import get_strategy_version

from .engine import simulate_prediction_rows
from .lab_source import apply_classifier_filter, load_lab_prediction_rows_for_simulation
from .probability_filter import apply_probability_filter
from .triple_barrier_filter import MISSING_TB_REASON, TB_DISABLED, apply_tb_filter
from .metrics import (
    attach_portfolio_risk_metrics,
    compute_fold_metrics,
    compute_trade_metrics,
)
from .scoring import attach_strategy_score
from .paths import strategy_runs_dir
from .store import StrategyRunStore

_BATCH = 500


def _prediction_run_hash(run: dict[str, Any]) -> str:
    payload = {
        "dataset_fingerprint": run.get("dataset_fingerprint"),
        "feature_snapshot_hash": run.get("feature_snapshot_hash"),
        "walk_forward_config_hash": run.get("walk_forward_config_hash"),
        "training_config_hash": run.get("training_config_hash"),
    }
    return json.dumps(payload, sort_keys=True)


def _lab_dataset_hash(status: dict[str, Any]) -> str:
    payload = {
        "source": "model_lab_prediction_dataset",
        "dataset_hash": status.get("dataset_hash"),
        "row_count": status.get("row_count"),
        "trading_days": status.get("trading_days"),
        "start_day": status.get("start_day"),
        "end_day": status.get("end_day"),
        "parent_model_name": status.get("parent_model_name"),
    }
    return json.dumps(payload, sort_keys=True, default=str)


def _apply_config_overrides(cfg: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    out = normalize_strategy_config(copy.deepcopy(cfg))
    if not overrides:
        return out
    for section, values in overrides.items():
        if section not in out or not isinstance(values, dict):
            continue
        if not isinstance(out.get(section), dict):
            continue
        out[section].update(values)
    return normalize_strategy_config(out)


def _attach_pipeline_metrics(
    metrics: dict[str, Any],
    *,
    stats: dict[str, int],
    dataset_row_count: int | None = None,
    date_filtered_count: int | None = None,
    classifier_meta: dict[str, Any] | None = None,
    probability_meta: dict[str, Any] | None = None,
    tb_meta: dict[str, Any] | None = None,
    execution_rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Expose funnel counts so UI can distinguish predictions vs executed trades."""
    out = dict(metrics)
    evaluated = int(stats.get("predictions_evaluated") or 0)
    signals = int(stats.get("signals") or 0)
    skipped = int(stats.get("skipped") or 0)
    executed = int(stats.get("trades") or out.get("trade_count") or 0)
    wins = int(out.get("wins") or 0)
    losses = int(out.get("losses") or 0)
    dataset_n = int(dataset_row_count) if dataset_row_count is not None else evaluated
    date_n = int(date_filtered_count) if date_filtered_count is not None else evaluated
    clf = dict(classifier_meta or {})

    out["dataset_row_count"] = dataset_n
    out["date_filtered_count"] = date_n
    out["predictions_loaded"] = date_n
    out["classifier_filter"] = clf
    out["classifier_active"] = bool(clf.get("active"))
    out["classifier_label"] = clf.get("label") or "Disabled"
    out["classifier_kept"] = int(clf.get("rows_after") if clf.get("active") else date_n)
    out["classifier_removed"] = int(clf.get("rows_removed") or 0) if clf.get("active") else 0
    kept_n = int(out["classifier_kept"] or 0)
    removed_n = int(out["classifier_removed"] or 0)
    base_n = max(date_n, 1)
    trades_kept = int(clf.get("trades_kept") if clf.get("trades_kept") is not None else executed)
    trades_removed = int(clf.get("trades_removed") or 0) if clf.get("active") else 0
    out["classifier_summary"] = {
        "label": out["classifier_label"],
        "active": out["classifier_active"],
        "prediction_rows": date_n,
        "rows_kept": kept_n,
        "rows_removed": removed_n,
        "rows_kept_pct": round(100.0 * kept_n / base_n, 2) if date_n else 0.0,
        "rows_removed_pct": round(100.0 * removed_n / base_n, 2) if date_n else 0.0,
        "executed_trades_kept": trades_kept,
        "executed_trades_removed": trades_removed,
    }
    out["executed_trades_kept"] = trades_kept
    out["executed_trades_removed"] = trades_removed

    prob = dict(probability_meta or {})
    prob_active = bool(prob.get("active"))
    prob_before = int(prob.get("rows_before") if prob.get("rows_before") is not None else kept_n)
    prob_kept = int(prob.get("rows_after") if prob_active else prob_before)
    prob_removed = int(prob.get("rows_removed") or 0) if prob_active else 0
    prob_base = max(prob_before, 1)
    out["probability_filter"] = prob
    out["probability_filter_active"] = prob_active
    out["probability_filter_label"] = prob.get("label") or "Disabled"
    out["probability_filter_column"] = prob.get("column")
    out["probability_filter_threshold"] = prob.get("threshold")
    out["probability_kept"] = prob_kept
    out["probability_removed"] = prob_removed
    out["probability_summary"] = {
        "label": out["probability_filter_label"],
        "active": prob_active,
        "column": prob.get("column"),
        "threshold": prob.get("threshold"),
        "prediction_rows": prob_before,
        "rows_kept": prob_kept,
        "rows_removed": prob_removed,
        "rows_kept_pct": round(100.0 * prob_kept / prob_base, 2) if prob_before else 0.0,
        "rows_removed_pct": round(100.0 * prob_removed / prob_base, 2) if prob_before else 0.0,
        "rows_null": int(prob.get("rows_null") or 0),
        "executed_trades": executed,
    }

    tb = dict(tb_meta or {})
    tb_active = bool(tb.get("active"))
    tb_before = int(tb.get("rows_before") if tb.get("rows_before") is not None else prob_kept)
    tb_kept = int(tb.get("rows_after") if tb_active else tb_before)
    tb_removed = int(tb.get("rows_removed") or 0) if tb_active else 0
    out["tb_filter"] = tb
    out["tb_filter_active"] = tb_active
    out["tb_filter_label"] = tb.get("label") or TB_DISABLED
    out["tb_filter_class_id"] = tb.get("class_id")
    out["tb_filter_threshold"] = tb.get("threshold")
    out["tb_kept"] = tb_kept
    out["tb_removed"] = tb_removed
    out["tb_summary"] = {
        "label": out["tb_filter_label"],
        "active": tb_active,
        "tb_model_name": tb.get("tb_model_name"),
        "class_id": tb.get("class_id"),
        "threshold": tb.get("threshold"),
        "candidate_rows": tb_before,
        "trades_filtered": tb_removed,
        "rows_kept": tb_kept,
        "class_counts": tb.get("class_counts") or {},
        "avg_tb_probability": tb.get("avg_tb_probability"),
        "skipped_missing_count": int(tb.get("rows_null") or 0),
        "skipped_missing_reason": tb.get("skip_reason") or MISSING_TB_REASON,
        "probability_distribution": tb.get("probability_distribution") or {},
    }
    out["predictions_evaluated"] = evaluated
    out["signals_generated"] = signals
    out["signals_skipped"] = skipped
    out["skipped_signals"] = skipped  # legacy key
    out["signals"] = signals  # legacy key
    out["executed_trades"] = executed
    out["trade_count"] = executed
    out["winning_trades"] = wins
    out["losing_trades"] = losses
    out["ignored_predictions"] = max(0, evaluated - executed)
    out["no_signal"] = int(stats.get("no_signal") or 0)
    out["blocked_open"] = int(stats.get("blocked_open") or 0)
    out["skipped_cadence"] = int(stats.get("skipped_cadence") or 0)
    out["skipped_no_path"] = int(stats.get("skipped_no_path") or 0)
    out["candidate_signals"] = int(
        stats.get("candidate_signals") if stats.get("candidate_signals") is not None else signals
    )
    out["skipped_max_positions"] = int(stats.get("skipped_max_positions") or 0)
    out["skipped_same_symbol"] = int(stats.get("skipped_same_symbol") or 0)
    from .engine import normalize_execution_rules

    exe = normalize_execution_rules(execution_rules)
    out["execution_rules"] = exe
    equity_pts = int(out.get("equity_curve_points") if out.get("equity_curve_points") is not None else executed)
    out["equity_curve_points"] = equity_pts
    out["simulator_summary"] = {
        "prediction_rows": date_n,
        "classifier_kept": kept_n,
        "probability_kept": prob_kept,
        "candidate_signals": out["candidate_signals"],
        "executed_trades": executed,
        "skipped_max_positions": out["skipped_max_positions"],
        "skipped_same_symbol": out["skipped_same_symbol"],
        "equity_curve_points": equity_pts,
        "equity_matches_executed": equity_pts == executed,
        "execution_rules_enabled": bool(exe.get("enabled")),
    }
    out["metrics_debug"] = {
        "candidate_signals": out["candidate_signals"],
        "executed_trades": executed,
        "skipped_max_positions": out["skipped_max_positions"],
        "skipped_same_symbol": out["skipped_same_symbol"],
        "equity_curve_points": equity_pts,
        "equity_matches_executed": equity_pts == executed,
        "metrics_from_executed_trades_only": bool(
            out.get("metrics_from_executed_trades_only", True)
        ),
        "net_profit": out.get("net_profit"),
        "gross_profit": out.get("gross_profit"),
        "gross_loss": out.get("gross_loss"),
        "win_rate_pct": out.get("win_rate_pct"),
        "profit_factor": out.get("profit_factor"),
        "max_drawdown": out.get("max_drawdown"),
        "account_equity_max_drawdown": out.get("account_equity_max_drawdown"),
        "max_portfolio_drawdown_open_risk": out.get("max_portfolio_drawdown_open_risk"),
        "max_theoretical_portfolio_risk": out.get("max_theoretical_portfolio_risk"),
        "stop_loss_per_trade_rupees": out.get("stop_loss_per_trade_rupees"),
        "max_open_positions_for_risk": out.get("max_open_positions_for_risk"),
        "observed_max_concurrent_open": out.get("observed_max_concurrent_open"),
        "max_dd_method": out.get("max_dd_method"),
        "max_dd_peak_equity": out.get("max_dd_peak_equity"),
        "max_dd_trough_equity": out.get("max_dd_trough_equity"),
        "max_dd_peak_exit_ts": out.get("max_dd_peak_exit_ts"),
        "max_dd_trough_exit_ts": out.get("max_dd_trough_exit_ts"),
        "max_drawdown_episode": out.get("max_drawdown_episode"),
        "portfolio_risk": out.get("portfolio_risk"),
        "uses_floating_pnl_for_account_dd": False,
    }
    return out


def _finalize_simulation_run(
    *,
    data_dir: str,
    strategy_run_id: str,
    strategy_version_id: str,
    prediction_run_id: str,
    trades: list[dict[str, Any]],
    stats: dict[str, int],
    dataset_row_count: int | None = None,
    date_filtered_count: int | None = None,
    classifier_meta: dict[str, Any] | None = None,
    probability_meta: dict[str, Any] | None = None,
    tb_meta: dict[str, Any] | None = None,
    execution_rules: dict[str, Any] | None = None,
    price_rows: list[dict[str, Any]] | None = None,
    strategy_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    with StrategyRunStore(data_dir) as store:
        for i in range(0, len(trades), _BATCH):
            store.insert_trades_batch(trades[i : i + _BATCH])

        metrics = compute_trade_metrics(trades)
        metrics["fold_metrics"] = compute_fold_metrics(trades)
        metrics = attach_portfolio_risk_metrics(
            metrics,
            trades=trades,
            price_rows=price_rows,
            cfg=strategy_config,
            execution_rules=execution_rules,
        )
        metrics = _attach_pipeline_metrics(
            metrics,
            stats=stats,
            dataset_row_count=dataset_row_count,
            date_filtered_count=date_filtered_count,
            classifier_meta=classifier_meta,
            probability_meta=probability_meta,
            tb_meta=tb_meta,
            execution_rules=execution_rules,
        )
        metrics = attach_strategy_score(metrics, trades)

        store.finalize_run(
            strategy_run_id,
            trade_count=len(trades),
            signal_count=int(stats.get("signals") or 0),
            skipped_signals=int(stats.get("skipped") or 0),
            metrics=metrics,
        )

        pkg_dir = os.path.join(strategy_runs_dir(data_dir), strategy_run_id)
        os.makedirs(pkg_dir, exist_ok=True)
        with open(os.path.join(pkg_dir, "strategy_run.json"), "w", encoding="utf-8") as fh:
            json.dump({
                "strategy_run_id": strategy_run_id,
                "prediction_run_id": prediction_run_id,
                "strategy_version_id": strategy_version_id,
                "trade_count": len(trades),
                "metrics": metrics,
            }, fh, indent=2, default=str)

    return get_strategy_run_detail(data_dir, strategy_run_id) or {"strategy_run_id": strategy_run_id}


def run_strategy_simulation(
    data_dir: str,
    *,
    prediction_run_id: str,
    strategy_version_id: str,
    fold_id: str | None = None,
    execution_rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay strategy on saved prediction-run rows. No model inference."""
    version = get_strategy_version(data_dir, strategy_version_id)
    if not version:
        raise ValueError(f"strategy version not found: {strategy_version_id}")

    with PredictionRunStore(data_dir) as pred_store:
        pred_run = pred_store.get_run(prediction_run_id)
        if not pred_run:
            raise ValueError(f"prediction run not found: {prediction_run_id}")
        folds = pred_store.list_folds(prediction_run_id)
        fold_numbers = {str(f["fold_id"]): int(f["fold_number"]) for f in folds}
        if fold_id:
            rows = pred_store.list_all_rows(prediction_run_id, fold_id=fold_id)
            scope = "single_fold"
            fold_number = fold_numbers.get(fold_id)
        else:
            rows = pred_store.list_all_rows(prediction_run_id)
            scope = "all_folds"
            fold_number = None

    cfg = version.get("config") or {}
    strategy_run_id = uuid.uuid4().hex

    with StrategyRunStore(data_dir) as store:
        store.create_run({
            "strategy_run_id": strategy_run_id,
            "prediction_run_id": prediction_run_id,
            "strategy_id": version["strategy_id"],
            "strategy_version_id": strategy_version_id,
            "strategy_config_hash": version.get("config_hash"),
            "prediction_run_hash": _prediction_run_hash(pred_run),
            "model_id": pred_run.get("model_id"),
            "scope": scope,
            "fold_id": fold_id,
            "fold_number": fold_number,
            "meta": {
                "strategy_display_name": version.get("display_name"),
                "strategy_version_label": version.get("version_label"),
                "source": "prediction_run",
            },
        })

        trades, stats = simulate_prediction_rows(
            rows,
            cfg=cfg,
            strategy_run_id=strategy_run_id,
            strategy_version_id=strategy_version_id,
            prediction_run_id=prediction_run_id,
            fold_numbers=fold_numbers,
            execution_rules=execution_rules,
        )

    return _finalize_simulation_run(
        data_dir=data_dir,
        strategy_run_id=strategy_run_id,
        strategy_version_id=strategy_version_id,
        prediction_run_id=prediction_run_id,
        trades=trades,
        stats=stats,
        dataset_row_count=len(rows),
        execution_rules=execution_rules,
        price_rows=rows,
        strategy_config=cfg,
    )


def run_strategy_simulation_from_lab(
    data_dir: str,
    *,
    lab_db_path: str,
    strategy_version_id: str,
    trading_days: list[str] | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    config_overrides: dict[str, Any] | None = None,
    capital: float | None = None,
    confidence_classifier: str | None = None,
    classifier_keep_value: int = 1,
    probability_filter_column: str | None = None,
    probability_filter_threshold: float | None = None,
    probability_filter_label: str | None = None,
    probability_filter_member: str | None = None,
    tb_filter_enabled: bool = False,
    tb_class_id: int | None = None,
    tb_class_label: str | None = None,
    tb_threshold: float | None = None,
    tb_model_name: str | None = None,
    tb_class_labels: dict[int, str] | None = None,
    execution_rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay strategy on the Research Lab Prediction Dataset (single source of truth).

    Pipeline:
      Prediction Dataset → Classifier Filter (entries only) →
      Prediction Package Probability Filter (entries only) →
      Triple Barrier Filter (entries only, optional) → Strategy Rules →
      Execution Rules (optional) → Trades

    All entry filters are independent AND predicates on the same row set, so
    the order they are applied in does not change the final trade set — only
    which stage's funnel counts a removed row shows up under.

    The probability filter keeps entry rows where a Prediction Package ladder
    member scored at or above ``probability_filter_threshold`` — for example
    ``pred_prob_up_2pct_5m >= 0.60``.

    The Triple Barrier filter (disabled by default) keeps entry rows where the
    persisted ``tb_pred_class`` equals ``tb_class_id`` and ``tb_pred_probability``
    is at or above ``tb_threshold``. It only reads columns already written by
    the Prediction Dataset builder — it never runs TB inference and never
    writes back to the dataset. When ``tb_filter_enabled`` is False the stage
    is a no-op and results are identical to a run with no Triple Barrier
    arguments at all.

    Exit path (stop/target/hold) always walks the full date-filtered prediction
    series for the token — entry filters must not thin mark samples.

    ``execution_rules`` affect the Strategy Simulator only — never label builder.
    """
    from chain_replay_ml.model_lab.prediction_builder import prediction_dataset_status
    from chain_replay_ml.model_lab.store import ModelLabStore
    from .engine import normalize_execution_rules

    if not lab_db_path or not os.path.isfile(lab_db_path):
        raise ValueError(
            "Prediction Dataset not found — build predictions in the Prediction Dataset tab first."
        )

    version = get_strategy_version(data_dir, strategy_version_id)
    if not version:
        raise ValueError(f"strategy version not found: {strategy_version_id}")

    status = prediction_dataset_status(lab_db_path, light=True)
    row_count = int(status.get("row_count") or 0)
    if row_count <= 0:
        raise ValueError("Prediction Dataset is empty — generate predictions before simulating.")

    with ModelLabStore(lab_db_path) as store:
        lab = store.read_info()
        lab_uuid = str(getattr(lab, "lab_uuid", "") or "")
        model_id = str(
            getattr(lab, "parent_model_name", None)
            or status.get("parent_model_name")
            or ""
        ).strip()

    rows = load_lab_prediction_rows_for_simulation(
        lab_db_path,
        trading_days=trading_days,
        date_from=date_from,
        date_to=date_to,
        confidence_classifier=confidence_classifier,
    )
    if not rows:
        raise ValueError("No prediction rows match the selected date range.")
    date_filtered_count = len(rows)
    date_rows = list(rows)

    rows, classifier_meta = apply_classifier_filter(
        rows,
        confidence_classifier=confidence_classifier,
        keep_value=int(classifier_keep_value),
    )

    rows, probability_meta = apply_probability_filter(
        rows,
        column=probability_filter_column,
        threshold=probability_filter_threshold,
        label=probability_filter_label,
    )
    probability_meta = dict(probability_meta)
    probability_meta["member_key"] = str(probability_filter_member or "") or None

    rows, tb_meta = apply_tb_filter(
        rows,
        class_id=(int(tb_class_id) if tb_filter_enabled and tb_class_id is not None else None),
        threshold=tb_threshold,
        label=tb_class_label,
        class_labels=tb_class_labels,
    )
    tb_meta = dict(tb_meta)
    tb_meta["tb_model_name"] = str(tb_model_name or "") or None

    cfg = _apply_config_overrides(version.get("config") or {}, config_overrides)
    exe_rules = normalize_execution_rules(execution_rules)
    strategy_run_id = uuid.uuid4().hex
    # Synthetic id — strategy_runs.prediction_run_id is NOT NULL.
    prediction_source_id = f"lab:{lab_uuid or os.path.basename(lab_db_path)}"

    if trading_days:
        scope = "lab_day_list"
    elif date_from or date_to:
        scope = "lab_date_range"
    else:
        scope = "lab_all_days"

    with StrategyRunStore(data_dir) as store:
        store.create_run({
            "strategy_run_id": strategy_run_id,
            "prediction_run_id": prediction_source_id,
            "strategy_id": version["strategy_id"],
            "strategy_version_id": strategy_version_id,
            "strategy_config_hash": version.get("config_hash"),
            "prediction_run_hash": _lab_dataset_hash(status),
            "model_id": model_id,
            "scope": scope,
            "fold_id": None,
            "fold_number": None,
            "meta": {
                "source": "model_lab_prediction_dataset",
                "lab_db_path": lab_db_path,
                "lab_uuid": lab_uuid,
                "date_from": date_from,
                "date_to": date_to,
                "trading_days": list(trading_days) if trading_days else None,
                "rows_loaded": date_filtered_count,
                "rows_simulated": len(rows),
                "capital": capital,
                "confidence_classifier": confidence_classifier,
                "classifier_keep_value": int(classifier_keep_value),
                "classifier_filter": classifier_meta,
                "probability_filter": probability_meta,
                "classification_filter_label": probability_meta.get("label"),
                "classification_filter_threshold": probability_meta.get("threshold"),
                "tb_filter_enabled": bool(tb_filter_enabled),
                "tb_filter": tb_meta,
                "tb_model_name": tb_meta.get("tb_model_name"),
                "execution_rules": exe_rules,
                "strategy_display_name": version.get("display_name"),
                "strategy_version_label": version.get("version_label"),
                "config_overrides": config_overrides or {},
            },
        })

        trades, stats = simulate_prediction_rows(
            rows,
            cfg=cfg,
            strategy_run_id=strategy_run_id,
            strategy_version_id=strategy_version_id,
            prediction_run_id=prediction_source_id,
            fold_numbers=None,
            execution_rules=exe_rules,
            # Classifier gates entries only — stop/target/hold walk the full day path.
            path_rows=date_rows,
        )

        # Trades blocked by classifier = simulate the complement row set (not persisted).
        trades_removed = 0
        if classifier_meta.get("active") and int(classifier_meta.get("rows_removed") or 0) > 0:
            pred_col = str(classifier_meta.get("pred_col") or "")
            keep_v = int(classifier_meta.get("keep_value") or 1)
            removed_rows: list[dict[str, Any]] = []
            for row in date_rows:
                raw = row.get(pred_col) if pred_col else None
                if raw is None:
                    removed_rows.append(row)
                    continue
                try:
                    val = int(float(raw))
                except (TypeError, ValueError):
                    removed_rows.append(row)
                    continue
                if val != keep_v:
                    removed_rows.append(row)
            if removed_rows:
                blocked, _blocked_stats = simulate_prediction_rows(
                    removed_rows,
                    cfg=cfg,
                    strategy_run_id=f"{strategy_run_id}_blocked",
                    strategy_version_id=strategy_version_id,
                    prediction_run_id=prediction_source_id,
                    fold_numbers=None,
                    execution_rules=exe_rules,
                    path_rows=date_rows,
                )
                trades_removed = len(blocked)

        classifier_meta = dict(classifier_meta)
        classifier_meta["trades_kept"] = len(trades)
        classifier_meta["trades_removed"] = int(trades_removed)

    return _finalize_simulation_run(
        data_dir=data_dir,
        strategy_run_id=strategy_run_id,
        strategy_version_id=strategy_version_id,
        prediction_run_id=prediction_source_id,
        trades=trades,
        stats=stats,
        dataset_row_count=row_count,
        date_filtered_count=date_filtered_count,
        classifier_meta=classifier_meta,
        probability_meta=probability_meta,
        tb_meta=tb_meta,
        execution_rules=exe_rules,
        price_rows=rows,
        strategy_config=cfg,
    )


_COMPARISON_METRIC_KEYS: tuple[tuple[str, str], ...] = (
    ("total_trades", "trade_count"),
    ("win_rate_pct", "win_rate_pct"),
    ("net_profit", "net_profit"),
    ("average_profit", "avg_trade_pnl"),
    ("average_drawdown", "average_drawdown"),
    ("max_drawdown", "account_equity_max_drawdown"),
    ("sharpe", "sharpe_ratio"),
    ("expectancy", "expectancy"),
    ("profit_factor", "profit_factor"),
)


def _comparison_metric_snapshot(run: dict[str, Any] | None) -> dict[str, Any]:
    """Strategy-metric snapshot for the Baseline vs Filtered comparison table.

    No Precision/Recall here by design (Sim reports are strategy-outcome only).
    ``sharpe`` is included only when present on the metrics payload — Sharpe is
    not currently computed by this simulator, so it reports ``None`` until it
    is; callers should render that as "not available", not "0".
    """
    m = (run or {}).get("metrics") or {}
    out: dict[str, Any] = {}
    for label_key, metric_key in _COMPARISON_METRIC_KEYS:
        out[label_key] = m.get(metric_key)
    out["strategy_run_id"] = (run or {}).get("strategy_run_id")
    return out


def build_tb_comparison_payload(
    baseline_run: dict[str, Any] | None,
    filtered_run: dict[str, Any] | None,
) -> dict[str, Any]:
    """Side-by-side Baseline (TB off) vs Filtered (TB on) strategy metrics."""
    baseline = _comparison_metric_snapshot(baseline_run)
    filtered = _comparison_metric_snapshot(filtered_run)
    delta: dict[str, Any] = {}
    for label_key, _metric_key in _COMPARISON_METRIC_KEYS:
        b, f = baseline.get(label_key), filtered.get(label_key)
        try:
            delta[label_key] = round(float(f) - float(b), 4) if b is not None and f is not None else None
        except (TypeError, ValueError):
            delta[label_key] = None
    return {
        "baseline": baseline,
        "filtered": filtered,
        "delta": delta,
        "metric_order": [label_key for label_key, _ in _COMPARISON_METRIC_KEYS],
    }


def run_strategy_simulation_from_lab_with_tb_comparison(
    data_dir: str,
    *,
    lab_db_path: str,
    strategy_version_id: str,
    tb_filter_enabled: bool = False,
    tb_class_id: int | None = None,
    tb_class_label: str | None = None,
    tb_threshold: float | None = None,
    tb_model_name: str | None = None,
    tb_class_labels: dict[int, str] | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run the lab Strategy Simulator, auto dual-running Baseline vs TB-Filtered.

    When the Triple Barrier filter is disabled this is a thin pass-through to
    ``run_strategy_simulation_from_lab`` — a single run, identical results to
    today (Part A acceptance criterion). When enabled, it additionally runs a
    Baseline pass with the Triple Barrier filter switched off (every other
    filter/config held identical) and returns both runs plus a side-by-side
    comparison payload. No second button — the dual-run is automatic.
    """
    filtered_detail = run_strategy_simulation_from_lab(
        data_dir,
        lab_db_path=lab_db_path,
        strategy_version_id=strategy_version_id,
        tb_filter_enabled=tb_filter_enabled,
        tb_class_id=tb_class_id,
        tb_class_label=tb_class_label,
        tb_threshold=tb_threshold,
        tb_model_name=tb_model_name,
        tb_class_labels=tb_class_labels,
        **kwargs,
    )
    if not tb_filter_enabled:
        return {
            "ok": True,
            "mode": "single",
            "run": filtered_detail.get("run"),
            "baseline_run": None,
            "filtered_run": None,
            "comparison": None,
        }

    baseline_detail = run_strategy_simulation_from_lab(
        data_dir,
        lab_db_path=lab_db_path,
        strategy_version_id=strategy_version_id,
        tb_filter_enabled=False,
        tb_class_id=None,
        tb_class_label=None,
        tb_threshold=None,
        tb_model_name=None,
        tb_class_labels=None,
        **kwargs,
    )
    baseline_run = baseline_detail.get("run")
    filtered_run = filtered_detail.get("run")
    return {
        "ok": True,
        "mode": "comparison",
        "run": filtered_run,
        "baseline_run": baseline_run,
        "filtered_run": filtered_run,
        "comparison": build_tb_comparison_payload(baseline_run, filtered_run),
    }


def get_strategy_run_detail(data_dir: str, strategy_run_id: str) -> dict[str, Any] | None:
    with StrategyRunStore(data_dir) as store:
        run = store.get_run(strategy_run_id)
        if not run:
            return None
        run["trade_count_stored"] = store.count_trades(strategy_run_id)
        return {"ok": True, "run": run}


def list_strategy_runs(
    data_dir: str,
    *,
    prediction_run_id: str | None = None,
    strategy_id: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    with StrategyRunStore(data_dir) as store:
        return store.list_runs(
            prediction_run_id=prediction_run_id,
            strategy_id=strategy_id,
            limit=limit,
        )


def get_strategy_run_trades(
    data_dir: str,
    strategy_run_id: str,
    *,
    limit: int = 500,
    offset: int = 0,
) -> dict[str, Any]:
    with StrategyRunStore(data_dir) as store:
        run = store.get_run(strategy_run_id)
        if not run:
            return {"ok": False, "error": "strategy run not found"}
        total = store.count_trades(strategy_run_id)
        trades = store.list_trades(strategy_run_id, limit=limit, offset=offset)
        return {
            "ok": True,
            "run": run,
            "trades": trades,
            "total": total,
            "limit": limit,
            "offset": offset,
        }
