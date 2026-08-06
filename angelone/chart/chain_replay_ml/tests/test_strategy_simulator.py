"""Tests for Phase 3 strategy simulator."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from chain_replay_ml.prediction_runs.store import PredictionRunStore
from chain_replay_ml.prediction_runs.writer import PredictionRunWriter
from chain_replay_ml.strategy_registry import create_strategy, get_default_template
from chain_replay_ml.strategy_simulator import (
    get_strategy_run_detail,
    get_strategy_run_trades,
    list_strategy_runs,
    run_strategy_simulation,
)
from chain_replay_ml.strategy_simulator.engine import _entry_signal, simulate_fold_rows
from chain_replay_ml.strategy_simulator.metrics import compute_trade_metrics


class StrategySimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def _seed_prediction_run_with_series(self) -> tuple[str, str, str]:
        with PredictionRunStore(self.tmp) as store:
            run = store.create_run({
                "model_id": "SimModel_v1",
                "dataset_name": "MS_test",
                "target": "future_ltp_5m",
                "status": "completed",
            })
            run_id = run["run_id"]
            writer = PredictionRunWriter(store, run_id)
            # Token A: entry 20 -> actual rises to target (+10%)
            ctx = pd.DataFrame({
                "trading_day": ["2026-07-01"] * 4,
                "timestamp": [100.0, 103.0, 106.0, 109.0],
                "token": ["TOK_A"] * 4,
                "strike": [24000.0] * 4,
                "option_type": ["CE"] * 4,
                "spot": [24050.0] * 4,
                "ltp": [20.0, 20.5, 22.5, 23.0],
            })
            writer.write_fold_predictions(
                fold_number=1,
                fold_def={
                    "train": {"start": 0, "stop": 10, "rows": 10},
                    "validation": {"start": 10, "stop": 14, "rows": 4},
                },
                metrics={"mae": 1.0, "rmse": 1.2, "directional_accuracy_pct": 75.0},
                val_context=ctx,
                val_pred=np.array([21.0, 21.0, 24.0, 24.0]),
                val_y=pd.Series([20.0, 21.0, 22.5, 23.0]),
                baseline_ltp=pd.Series([19.5, 20.0, 21.0, 22.0]),
            )
            store.finalize_run(run_id, status="completed", prediction_count=4, fold_count=1)
            folds = store.list_folds(run_id)
            fold_id = folds[0]["fold_id"]

        strat = create_strategy(self.tmp, display_name="Sim Strat", config=get_default_template())
        version_id = strat["champion_version"]["version_id"]
        return run_id, fold_id, version_id

    def test_simulate_target_exit(self) -> None:
        cfg = get_default_template()
        cfg["entry"]["premium_min"] = 10
        cfg["entry"]["premium_max"] = 30
        cfg["target"]["target_profit_pct"] = 8.0
        rows = [
            {
                "prediction_id": "p0",
                "fold_id": "f1",
                "row_index": 0,
                "timestamp": 100.0,
                "trading_day": "2026-07-01",
                "token": "TOK_A",
                "ltp": 20.0,
                "predicted_ltp": 21.0,
                "actual_ltp": 20.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
            {
                "prediction_id": "p1",
                "fold_id": "f1",
                "row_index": 1,
                "timestamp": 103.0,
                "trading_day": "2026-07-01",
                "token": "TOK_A",
                "ltp": 20.5,
                "predicted_ltp": 21.0,
                "actual_ltp": 21.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
            {
                "prediction_id": "p2",
                "fold_id": "f1",
                "row_index": 2,
                "timestamp": 106.0,
                "trading_day": "2026-07-01",
                "token": "TOK_A",
                "ltp": 22.5,
                "predicted_ltp": 24.0,
                "actual_ltp": 22.5,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
        ]
        trades, stats = simulate_fold_rows(
            rows,
            cfg=cfg,
            strategy_run_id="sr1",
            strategy_version_id="sv1",
            prediction_run_id="pr1",
            fold_number=1,
        )
        self.assertGreaterEqual(stats["trades"], 1)
        self.assertEqual(trades[0]["exit_reason"], "target")

    def test_end_to_end_strategy_run(self) -> None:
        run_id, _fold_id, version_id = self._seed_prediction_run_with_series()
        detail = run_strategy_simulation(
            self.tmp,
            prediction_run_id=run_id,
            strategy_version_id=version_id,
        )
        run = detail["run"]
        self.assertEqual(run["prediction_run_id"], run_id)
        self.assertGreater(run["trade_count"], 0)
        metrics = run.get("metrics") or {}
        self.assertIn("profit", metrics)
        self.assertIn("win_rate_pct", metrics)

        runs = list_strategy_runs(self.tmp, prediction_run_id=run_id)
        self.assertEqual(len(runs), 1)

        trades_doc = get_strategy_run_trades(self.tmp, run["strategy_run_id"])
        self.assertTrue(trades_doc["ok"])
        self.assertGreater(trades_doc["total"], 0)

    def test_metrics_computation(self) -> None:
        trades = [
            {"net_pnl": 100.0, "gross_pnl": 105.0, "return_pct": 5.0, "holding_seconds": 10, "fees": 5, "exit_ts": 1},
            {"net_pnl": -50.0, "gross_pnl": -45.0, "return_pct": -2.5, "holding_seconds": 20, "fees": 5, "exit_ts": 2},
        ]
        m = compute_trade_metrics(trades)
        self.assertEqual(m["trade_count"], 2)
        self.assertEqual(m["wins"], 1)
        self.assertEqual(m["profit"], 50.0)
        self.assertEqual(m["net_profit"], 50.0)
        self.assertEqual(m["gross_pnl_total"], 60.0)
        self.assertEqual(m["win_rate_pct"], 50.0)

    def test_strategy_run_json_written(self) -> None:
        run_id, _, version_id = self._seed_prediction_run_with_series()
        detail = run_strategy_simulation(
            self.tmp,
            prediction_run_id=run_id,
            strategy_version_id=version_id,
        )
        path = os.path.join(
            self.tmp, "strategy_runs", detail["run"]["strategy_run_id"], "strategy_run.json",
        )
        self.assertTrue(os.path.isfile(path))

    def test_lab_row_mapping(self) -> None:
        from chain_replay_ml.strategy_simulator.lab_source import lab_row_to_engine_row

        mapped = lab_row_to_engine_row({
            "id": 7,
            "prediction_id": "p1",
            "trading_day": "2026-07-01",
            "timestamp": 100.0,
            "token": "TOK",
            "strike": 24000.0,
            "option_type": "CE",
            "current_spot": 24050.0,
            "current_ltp": 20.0,
            "predicted_future_ltp": 22.0,
            "actual_future_ltp": 21.5,
            "confidence_target_hit_pred": 1,
        })
        self.assertEqual(mapped["ltp"], 20.0)
        self.assertEqual(mapped["predicted_ltp"], 22.0)
        self.assertEqual(mapped["actual_ltp"], 21.5)
        self.assertEqual(mapped["spot"], 24050.0)
        self.assertEqual(mapped["fold_id"], "")
        self.assertEqual(mapped["confidence"], 1.0)

    def test_apply_classifier_filter_keeps_pred_one(self) -> None:
        from chain_replay_ml.strategy_simulator.lab_source import apply_classifier_filter

        rows = [
            {"prediction_id": "a", "confidence_target_hit_pred": 1},
            {"prediction_id": "b", "confidence_target_hit_pred": 0},
            {"prediction_id": "c", "confidence_target_hit_pred": 1},
            {"prediction_id": "d", "confidence_target_hit_pred": None},
        ]
        kept, meta = apply_classifier_filter(rows, confidence_classifier="target_hit", keep_value=1)
        self.assertEqual([r["prediction_id"] for r in kept], ["a", "c"])
        self.assertTrue(meta["active"])
        self.assertEqual(meta["rows_before"], 4)
        self.assertEqual(meta["rows_after"], 2)
        self.assertEqual(meta["rows_removed"], 2)
        self.assertEqual(meta["rows_null"], 1)

    def test_simulate_from_lab_prediction_dataset(self) -> None:
        from chain_replay_ml.model_lab.store import ModelLabStore
        from chain_replay_ml.strategy_simulator import run_strategy_simulation_from_lab

        lab_path = os.path.join(self.tmp, "lab.db")
        pred_rows = []
        for i, (ltp, pred, actual) in enumerate(
            [(20.0, 21.0, 20.0), (20.5, 21.0, 21.0), (22.5, 24.0, 22.5), (23.0, 24.0, 23.0)]
        ):
            pred_rows.append({
                "lab_uuid": "u1",
                "prediction_id": f"p{i}",
                "trading_day": "2026-07-01",
                "timestamp": float(100 + i * 3),
                "token": "TOK_A",
                "strike": 24000.0,
                "option_type": "CE",
                "current_spot": 24050.0,
                "current_ltp": ltp,
                "predicted_future_ltp": pred,
                "actual_future_ltp": actual,
                "master_row_id": i + 1,
            })
        with ModelLabStore(lab_path) as store:
            store._ensure_schema()
            store.ensure_prediction_schema()
            store.write_info(
                lab_uuid="u1",
                lab_id="lab1",
                lab_name="Sim Lab",
                parent_model_id="m1",
                parent_model_name="SimModel_v1",
                model_checksum=None,
                description=None,
                purpose=None,
                version=1,
                original_feature_count=2,
                selected_feature_count=2,
                training_rows=4,
                target="future_ltp_5m",
                algorithm="xgboost",
                dataset_snapshot={"dataset_name": "MS_test"},
                model_snapshot=None,
                training_config_snapshot=None,
                wf_snapshot=None,
                metrics_snapshot=None,
                selected_features_snapshot=["f1"],
                feature_ranking_snapshot=None,
                artifact_pointers={},
            )
            store.write_prediction_summary(
                lab_uuid="u1",
                status="ready",
                row_count=4,
                trading_days=1,
                start_day="2026-07-01",
                end_day="2026-07-01",
                target_column="future_ltp_5m",
                parent_dataset="MS_test",
                parent_model_name="SimModel_v1",
                created_at="2026-07-01T10:00:00+00:00",
            )
            store.insert_prediction_rows(pred_rows)
            store.ensure_build_days("u1", ["2026-07-01"])

        cfg = get_default_template()
        cfg["entry"]["premium_min"] = 10
        cfg["entry"]["premium_max"] = 30
        cfg["target"]["target_profit_pct"] = 8.0
        strat = create_strategy(self.tmp, display_name="Lab Sim Strat", config=cfg)
        version_id = strat["champion_version"]["version_id"]

        detail = run_strategy_simulation_from_lab(
            self.tmp,
            lab_db_path=lab_path,
            strategy_version_id=version_id,
        )
        self.assertTrue(detail.get("ok"))
        run = detail["run"]
        self.assertEqual(run.get("model_id"), "SimModel_v1")
        self.assertTrue(str(run.get("prediction_run_id") or "").startswith("lab:"))
        self.assertEqual((run.get("meta") or {}).get("source"), "model_lab_prediction_dataset")
        m = run.get("metrics") or {}
        self.assertEqual(m.get("dataset_row_count"), 4)
        self.assertEqual(m.get("predictions_evaluated"), 4)
        self.assertIn("signals_generated", m)
        self.assertIn("executed_trades", m)
        self.assertEqual(m.get("executed_trades"), m.get("trade_count"))
        self.assertGreaterEqual(int(run.get("trade_count") or 0), 0)
        self.assertIn("simulator_summary", m)
        self.assertIn("candidate_signals", m.get("simulator_summary") or {})

    def _seed_lab_with_probabilities(self, lab_path: str, probs: list[float | None]) -> None:
        from chain_replay_ml.model_lab.store import ModelLabStore

        series = [(20.0, 21.0, 20.0), (20.5, 21.0, 21.0), (22.5, 24.0, 22.5), (23.0, 24.0, 23.0)]
        pred_rows = []
        for i, (ltp, pred, actual) in enumerate(series):
            pred_rows.append({
                "lab_uuid": "u1",
                "prediction_id": f"p{i}",
                "trading_day": "2026-07-01",
                "timestamp": float(100 + i * 3),
                "token": "TOK_A",
                "strike": 24000.0,
                "option_type": "CE",
                "current_spot": 24050.0,
                "current_ltp": ltp,
                "predicted_future_ltp": pred,
                "actual_future_ltp": actual,
                "master_row_id": i + 1,
                "pred_prob_up_2pct_5m": probs[i],
            })
        with ModelLabStore(lab_path) as store:
            store._ensure_schema()
            store.ensure_prediction_schema()
            store.write_info(
                lab_uuid="u1",
                lab_id="lab1",
                lab_name="Sim Lab",
                parent_model_id="m1",
                parent_model_name="SimModel_v1",
                model_checksum=None,
                description=None,
                purpose=None,
                version=1,
                original_feature_count=2,
                selected_feature_count=2,
                training_rows=4,
                target="future_ltp_5m",
                algorithm="xgboost",
                dataset_snapshot={"dataset_name": "MS_test"},
                model_snapshot=None,
                training_config_snapshot=None,
                wf_snapshot=None,
                metrics_snapshot=None,
                selected_features_snapshot=["f1"],
                feature_ranking_snapshot=None,
                artifact_pointers={},
            )
            store.write_prediction_summary(
                lab_uuid="u1",
                status="ready",
                row_count=len(pred_rows),
                trading_days=1,
                start_day="2026-07-01",
                end_day="2026-07-01",
                target_column="future_ltp_5m",
                parent_dataset="MS_test",
                parent_model_name="SimModel_v1",
                created_at="2026-07-01T10:00:00+00:00",
            )
            store.insert_prediction_rows(pred_rows)
            store.ensure_build_days("u1", ["2026-07-01"])

    def test_lab_run_records_probability_filter(self) -> None:
        from chain_replay_ml.strategy_simulator import run_strategy_simulation_from_lab

        lab_path = os.path.join(self.tmp, "lab_prob.db")
        self._seed_lab_with_probabilities(lab_path, [0.81, 0.62, 0.30, None])

        cfg = get_default_template()
        cfg["entry"]["premium_min"] = 10
        cfg["entry"]["premium_max"] = 30
        cfg["target"]["target_profit_pct"] = 8.0
        strat = create_strategy(self.tmp, display_name="Prob Filter Strat", config=cfg)
        version_id = strat["champion_version"]["version_id"]

        detail = run_strategy_simulation_from_lab(
            self.tmp,
            lab_db_path=lab_path,
            strategy_version_id=version_id,
            probability_filter_column="pred_prob_up_2pct_5m",
            probability_filter_threshold=0.60,
            probability_filter_label="+2%",
            probability_filter_member="up_2pct",
        )
        self.assertTrue(detail.get("ok"))
        run = detail["run"]

        meta = run.get("meta") or {}
        self.assertEqual(meta.get("classification_filter_label"), "+2%")
        self.assertEqual(meta.get("classification_filter_threshold"), 0.60)
        pf = meta.get("probability_filter") or {}
        self.assertTrue(pf.get("active"))
        self.assertEqual(pf.get("column"), "pred_prob_up_2pct_5m")
        self.assertEqual(pf.get("member_key"), "up_2pct")
        # 0.81 and 0.62 pass; 0.30 and NULL are dropped.
        self.assertEqual(pf.get("rows_after"), 2)
        self.assertEqual(pf.get("rows_removed"), 2)

        m = run.get("metrics") or {}
        self.assertTrue(m.get("probability_filter_active"))
        self.assertEqual(m.get("probability_filter_label"), "+2%")
        self.assertEqual(m.get("probability_filter_threshold"), 0.60)
        self.assertEqual(m.get("probability_kept"), 2)
        self.assertEqual(m.get("probability_removed"), 2)
        self.assertEqual((m.get("probability_summary") or {}).get("rows_kept"), 2)
        self.assertEqual((m.get("simulator_summary") or {}).get("probability_kept"), 2)
        # Classifier Filter is untouched by the probability filter.
        self.assertFalse(m.get("classifier_active"))

    def test_lab_run_without_probability_filter_stays_disabled(self) -> None:
        from chain_replay_ml.strategy_simulator import run_strategy_simulation_from_lab

        lab_path = os.path.join(self.tmp, "lab_noprob.db")
        self._seed_lab_with_probabilities(lab_path, [0.81, 0.62, 0.30, None])

        cfg = get_default_template()
        cfg["entry"]["premium_min"] = 10
        cfg["entry"]["premium_max"] = 30
        cfg["target"]["target_profit_pct"] = 8.0
        strat = create_strategy(self.tmp, display_name="No Prob Strat", config=cfg)
        version_id = strat["champion_version"]["version_id"]

        detail = run_strategy_simulation_from_lab(
            self.tmp,
            lab_db_path=lab_path,
            strategy_version_id=version_id,
        )
        m = (detail["run"] or {}).get("metrics") or {}
        self.assertFalse(m.get("probability_filter_active"))
        self.assertEqual(m.get("probability_filter_label"), "Disabled")
        self.assertEqual(m.get("probability_kept"), 4)
        self.assertEqual(m.get("probability_removed"), 0)


class ProbabilityFilterTests(unittest.TestCase):
    """Prediction Package Filter — probability gate on entry rows."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def _rows(self) -> list[dict]:
        return [
            {"prediction_id": "a", "pred_prob_up_2pct_5m": 0.81},
            {"prediction_id": "b", "pred_prob_up_2pct_5m": 0.60},
            {"prediction_id": "c", "pred_prob_up_2pct_5m": 0.42},
            {"prediction_id": "d", "pred_prob_up_2pct_5m": None},
        ]

    def test_filter_keeps_rows_at_or_above_threshold(self) -> None:
        from chain_replay_ml.strategy_simulator import apply_probability_filter

        kept, meta = apply_probability_filter(
            self._rows(),
            column="pred_prob_up_2pct_5m",
            threshold=0.60,
            label="+2%",
        )
        self.assertEqual([r["prediction_id"] for r in kept], ["a", "b"])
        self.assertTrue(meta["active"])
        self.assertEqual(meta["label"], "+2%")
        self.assertEqual(meta["column"], "pred_prob_up_2pct_5m")
        self.assertEqual(meta["threshold"], 0.60)
        self.assertEqual(meta["rows_before"], 4)
        self.assertEqual(meta["rows_after"], 2)
        self.assertEqual(meta["rows_removed"], 2)
        self.assertEqual(meta["rows_null"], 1)

    def test_disabled_filter_is_passthrough(self) -> None:
        from chain_replay_ml.strategy_simulator import apply_probability_filter

        rows = self._rows()
        kept, meta = apply_probability_filter(rows, column=None, threshold=None)
        self.assertEqual(len(kept), len(rows))
        self.assertFalse(meta["active"])
        self.assertIsNone(meta["column"])
        self.assertIsNone(meta["threshold"])

    def test_threshold_removing_every_row_raises(self) -> None:
        from chain_replay_ml.strategy_simulator import apply_probability_filter

        with self.assertRaises(ValueError):
            apply_probability_filter(
                self._rows(),
                column="pred_prob_up_2pct_5m",
                threshold=0.99,
                label="+2%",
            )

    def test_all_null_column_raises_rebuild_hint(self) -> None:
        from chain_replay_ml.strategy_simulator import apply_probability_filter

        rows = [{"prediction_id": "a", "pred_prob_up_2pct_5m": None}]
        with self.assertRaises(ValueError) as ctx:
            apply_probability_filter(
                rows,
                column="pred_prob_up_2pct_5m",
                threshold=0.5,
                label="+2%",
            )
        self.assertIn("Rebuild the Prediction Dataset", str(ctx.exception))

    def test_lab_row_mapping_preserves_probability_columns(self) -> None:
        from chain_replay_ml.strategy_simulator.lab_source import lab_row_to_engine_row

        out = lab_row_to_engine_row({
            "prediction_id": "p1",
            "current_ltp": 20.0,
            "pred_prob_up_2pct_5m": 0.66,
            "pred_prob_up_3pct_5m": None,
        })
        self.assertEqual(out["pred_prob_up_2pct_5m"], 0.66)
        self.assertIsNone(out["pred_prob_up_3pct_5m"])

    def test_recommended_threshold_uses_best_composite(self) -> None:
        from chain_replay_ml.strategy_simulator.probability_filter import (
            recommended_threshold,
            threshold_composite_scores,
        )

        # Precision keeps climbing while recall fades — composite weights
        # precision highest, so 0.70 must win over the best-F1 row at 0.30.
        rows = [
            {"threshold": 0.30, "precision_pct": 60.1, "recall_pct": 58.2, "f1_pct": 59.2},
            {"threshold": 0.50, "precision_pct": 62.4, "recall_pct": 49.1, "f1_pct": 54.9},
            {"threshold": 0.70, "precision_pct": 73.8, "recall_pct": 40.0, "f1_pct": 51.8},
            {"threshold": 0.90, "precision_pct": 72.7, "recall_pct": 28.9, "f1_pct": 41.4},
        ]
        thr, criterion = recommended_threshold(rows, decision_threshold=0.5, roc_auc=0.83)
        self.assertEqual(thr, 0.70)
        self.assertIn("Composite", criterion)
        scores = threshold_composite_scores(rows, roc_auc=0.83)
        self.assertEqual(max(scores, key=lambda k: scores[k]), 0.70)

    def test_recommendation_falls_back_to_decision_threshold(self) -> None:
        from chain_replay_ml.strategy_simulator.probability_filter import recommended_threshold

        thr, criterion = recommended_threshold([], decision_threshold=0.55)
        self.assertEqual(thr, 0.55)
        self.assertIn("decision threshold", criterion)

        thr, criterion = recommended_threshold([], decision_threshold=None)
        self.assertEqual(thr, 0.50)

    def test_options_exclude_untrained_members(self) -> None:
        import json

        from chain_replay_ml.strategy_simulator import (
            probability_filter_labels,
            probability_filter_options,
        )

        def _write(name: str, target: str, prediction_type: str, trained_at: str) -> None:
            pkg = os.path.join(self.tmp, "models", name)
            os.makedirs(pkg, exist_ok=True)
            with open(os.path.join(pkg, "config.json"), "w", encoding="utf-8") as fh:
                json.dump({
                    "dataset": "MS_test",
                    "target": target,
                    "prediction_type": prediction_type,
                    "features": ["f1"],
                    "algorithm": "xgboost",
                }, fh)
            with open(os.path.join(pkg, "registry.json"), "w", encoding="utf-8") as fh:
                json.dump({"trained_at": trained_at}, fh)
            with open(os.path.join(pkg, "model.ubj"), "wb") as fh:
                fh.write(b"stub")

        _write("Anchor_5m", "future_ltp_5m", "regression", "2026-07-01T00:00:00+00:00")
        _write("Clf_2pct", "label_up_2pct_5m", "binary", "2026-07-01T01:00:00+00:00")
        _write("Clf_3pct", "label_up_3pct_5m", "binary", "2026-07-01T02:00:00+00:00")

        options = probability_filter_options(
            self.tmp,
            dataset="MS_test",
            anchor_target="future_ltp_5m",
            anchor_model_name="Anchor_5m",
        )
        self.assertEqual([o["label"] for o in options], ["+2% Probability", "+3% Probability"])
        self.assertEqual(
            [o["column"] for o in options],
            ["pred_prob_up_2pct_5m", "pred_prob_up_3pct_5m"],
        )
        # +4% … >6% were never trained, so they must not appear.
        self.assertEqual(
            probability_filter_labels(options),
            ["Disabled", "+2% Probability", "+3% Probability"],
        )


class ExecutionRulesTests(unittest.TestCase):
    """Phase 1 Execution Rules — Strategy Simulator only."""

    def _cfg(self) -> dict:
        cfg = get_default_template()
        cfg["entry"]["premium_min"] = 10
        cfg["entry"]["premium_max"] = 30
        cfg["entry"]["entry_cadence_sec"] = 1
        cfg["entry"]["atm_band"] = 0
        cfg["target"]["target_profit_pct"] = 8.0
        cfg["stop"]["stop_loss_pct"] = 50.0
        cfg["hold_time"]["max_hold_sec"] = 60
        return cfg

    def _pair(
        self,
        *,
        token: str,
        t0: float,
        prediction_id: str,
        hold_sec: float = 10.0,
    ) -> list[dict]:
        """Entry signal at t0 + forward row so the trade can exit."""
        return [
            {
                "prediction_id": f"{prediction_id}_e",
                "fold_id": "f1",
                "row_index": int(t0),
                "timestamp": t0,
                "trading_day": "2026-07-01",
                "token": token,
                "ltp": 20.0,
                "predicted_ltp": 22.0,
                "actual_ltp": 20.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
            {
                "prediction_id": f"{prediction_id}_x",
                "fold_id": "f1",
                "row_index": int(t0) + 1,
                "timestamp": t0 + hold_sec,
                "trading_day": "2026-07-01",
                "token": token,
                "ltp": 22.0,
                "predicted_ltp": 22.0,
                "actual_ltp": 22.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
        ]

    def test_max_open_positions_skips_when_full(self) -> None:
        # Three concurrent candidates; max open = 2 → third skipped.
        rows = (
            self._pair(token="A", t0=100.0, prediction_id="a", hold_sec=20.0)
            + self._pair(token="B", t0=101.0, prediction_id="b", hold_sec=20.0)
            + self._pair(token="C", t0=102.0, prediction_id="c", hold_sec=20.0)
        )
        trades, stats = simulate_fold_rows(
            rows,
            cfg=self._cfg(),
            strategy_run_id="sr1",
            strategy_version_id="sv1",
            prediction_run_id="pr1",
            fold_number=1,
            execution_rules={
                "enabled": True,
                "max_open_positions": 2,
                "one_position_per_symbol": False,
            },
        )
        self.assertEqual(stats["candidate_signals"], 3)
        self.assertEqual(stats["skipped_max_positions"], 1)
        self.assertEqual(stats["skipped_same_symbol"], 0)
        self.assertEqual(stats["trades"], 2)
        self.assertEqual(len(trades), 2)

    def test_one_position_per_symbol_skips_while_open(self) -> None:
        # Same symbol twice while first still open → second skipped; after close OK.
        rows = (
            self._pair(token="TOK", t0=100.0, prediction_id="a", hold_sec=10.0)
            + self._pair(token="TOK", t0=105.0, prediction_id="b", hold_sec=10.0)
            + self._pair(token="TOK", t0=120.0, prediction_id="c", hold_sec=10.0)
        )
        trades, stats = simulate_fold_rows(
            rows,
            cfg=self._cfg(),
            strategy_run_id="sr1",
            strategy_version_id="sv1",
            prediction_run_id="pr1",
            fold_number=1,
            execution_rules={
                "enabled": True,
                "max_open_positions": 10,
                "one_position_per_symbol": True,
            },
        )
        self.assertEqual(stats["candidate_signals"], 3)
        self.assertEqual(stats["skipped_same_symbol"], 1)
        self.assertEqual(stats["skipped_max_positions"], 0)
        self.assertEqual(stats["trades"], 2)
        self.assertEqual([t["token"] for t in trades], ["TOK", "TOK"])

    def test_disabled_keeps_legacy_one_open_per_token(self) -> None:
        rows = (
            self._pair(token="TOK", t0=100.0, prediction_id="a", hold_sec=10.0)
            + self._pair(token="TOK", t0=105.0, prediction_id="b", hold_sec=10.0)
        )
        trades, stats = simulate_fold_rows(
            rows,
            cfg=self._cfg(),
            strategy_run_id="sr1",
            strategy_version_id="sv1",
            prediction_run_id="pr1",
            fold_number=1,
            execution_rules={"enabled": False},
        )
        self.assertEqual(stats["trades"], 1)
        self.assertGreaterEqual(stats["blocked_open"], 1)
        self.assertEqual(stats["skipped_max_positions"], 0)
        self.assertEqual(stats["skipped_same_symbol"], 0)

    def test_forced_entry_ignores_execution_rules_api(self) -> None:
        """Label builder path has no execution_rules parameter — all rows labeled."""
        from chain_replay_ml.strategy_simulator.engine import simulate_forced_entry_outcomes
        import inspect

        sig = inspect.signature(simulate_forced_entry_outcomes)
        self.assertNotIn("execution_rules", sig.parameters)

        rows = (
            self._pair(token="A", t0=100.0, prediction_id="a", hold_sec=10.0)
            + self._pair(token="A", t0=105.0, prediction_id="b", hold_sec=10.0)
            + self._pair(token="B", t0=101.0, prediction_id="c", hold_sec=10.0)
        )
        outcomes, stats = simulate_forced_entry_outcomes(rows, cfg=self._cfg())
        # Forced entry evaluates every row (no skip for open position / max open).
        self.assertEqual(len(outcomes), len(rows))
        self.assertEqual(stats.get("predictions_evaluated"), len(rows))
        self.assertGreaterEqual(int(stats.get("outcomes") or 0), 1)

    def test_metrics_use_executed_trades_only_with_max_open_1(self) -> None:
        """Skipped candidates must not change Net/Gross/WR/PF/Max DD or equity points."""
        from chain_replay_ml.strategy_simulator.metrics import (
            build_equity_curve,
            compute_trade_metrics,
        )

        # A opens first; B and C are candidates while A is still open → skipped.
        rows = (
            self._pair(token="A", t0=100.0, prediction_id="a", hold_sec=30.0)
            + self._pair(token="B", t0=105.0, prediction_id="b", hold_sec=30.0)
            + self._pair(token="C", t0=110.0, prediction_id="c", hold_sec=30.0)
            + self._pair(token="D", t0=140.0, prediction_id="d", hold_sec=10.0)
        )
        trades, stats = simulate_fold_rows(
            rows,
            cfg=self._cfg(),
            strategy_run_id="sr1",
            strategy_version_id="sv1",
            prediction_run_id="pr1",
            fold_number=1,
            execution_rules={
                "enabled": True,
                "max_open_positions": 1,
                "one_position_per_symbol": True,
            },
        )
        self.assertGreaterEqual(stats["candidate_signals"], 3)
        self.assertGreaterEqual(stats["skipped_max_positions"], 1)
        self.assertEqual(stats["trades"], len(trades))
        self.assertLess(len(trades), stats["candidate_signals"])

        m = compute_trade_metrics(trades)
        curve = build_equity_curve(trades)
        self.assertEqual(m["equity_curve_points"], len(trades))
        self.assertEqual(len(curve), len(trades))
        self.assertTrue(m["metrics_from_executed_trades_only"])
        self.assertEqual(m["net_profit"], round(sum(float(t.get("net_pnl") or 0) for t in trades), 2))
        self.assertEqual(m["trade_count"], len(trades))
        # Recompute Max DD only from executed equity — must match metric.
        self.assertEqual(
            m["max_drawdown"],
            round(max((float(p.get("drawdown") or 0) for p in curve), default=0.0), 2),
        )
        self.assertFalse(m["max_drawdown_episode"]["uses_floating_pnl"])
        self.assertTrue(m["max_drawdown_episode"]["uses_executed_trades_only"])
        # Injecting a fake skipped trade into a copy must change metrics — proving
        # that metrics are a pure function of the executed list only.
        fake = dict(trades[0])
        fake["net_pnl"] = -9999.0
        fake["trade_id"] = "skipped_should_not_be_here"
        polluted = compute_trade_metrics(trades + [fake])
        self.assertNotEqual(polluted["net_profit"], m["net_profit"])
        self.assertNotEqual(polluted["max_drawdown"], m["max_drawdown"])
        self.assertEqual(polluted["equity_curve_points"], len(trades) + 1)

    def test_max_drawdown_is_peak_to_trough_closed_equity(self) -> None:
        from chain_replay_ml.strategy_simulator.metrics import (
            annotate_equity_curve_max_dd,
            build_equity_curve,
            compute_max_drawdown_episode,
            compute_trade_metrics,
        )

        # Equity path: 100 → 150 → 70 → 30. Max DD = 150 - 30 = 120.
        trades = [
            {"trade_id": "t1", "exit_ts": 10, "trading_day": "d1", "token": "A", "net_pnl": 100.0},
            {"trade_id": "t2", "exit_ts": 20, "trading_day": "d1", "token": "B", "net_pnl": 50.0},
            {"trade_id": "t3", "exit_ts": 30, "trading_day": "d2", "token": "C", "net_pnl": -80.0},
            {"trade_id": "t4", "exit_ts": 40, "trading_day": "d2", "token": "D", "net_pnl": -40.0},
        ]
        ep = compute_max_drawdown_episode(trades)
        self.assertEqual(ep["method"], "closed_trade_cumulative_equity_peak_to_trough")
        self.assertFalse(ep["uses_floating_pnl"])
        self.assertEqual(ep["max_drawdown"], 120.0)
        self.assertEqual(ep["peak_equity"], 150.0)
        self.assertEqual(ep["trough_equity"], 30.0)
        self.assertEqual(ep["peak_point"], 2)
        self.assertEqual(ep["trough_point"], 4)
        self.assertEqual(ep["peak_exit_ts"], 20)
        self.assertEqual(ep["trough_exit_ts"], 40)
        self.assertEqual(ep["peak_trading_day"], "d1")
        self.assertEqual(ep["trough_trading_day"], "d2")

        m = compute_trade_metrics(trades)
        self.assertEqual(m["max_drawdown"], 120.0)
        self.assertEqual(m["account_equity_max_drawdown"], 120.0)
        self.assertEqual(m["max_dd_peak_equity"], 150.0)
        self.assertEqual(m["max_dd_trough_equity"], 30.0)
        curve = annotate_equity_curve_max_dd(build_equity_curve(trades), ep)
        self.assertTrue(curve[1]["is_max_dd_peak"])
        self.assertTrue(curve[3]["is_max_dd_trough"])
        self.assertEqual(sum(1 for p in curve if p["is_max_dd_peak"]), 1)
        self.assertEqual(sum(1 for p in curve if p["is_max_dd_trough"]), 1)

    def test_theoretical_portfolio_risk_is_max_open_times_stop(self) -> None:
        from chain_replay_ml.strategy_simulator.metrics import (
            attach_portfolio_risk_metrics,
            compute_trade_metrics,
        )

        # entry 100 × qty 100 × stop 10% = ₹1,000/trade; max open 3 → ₹3,000
        cfg = get_default_template()
        cfg["stop"]["stop_loss_pct"] = 10.0
        cfg["position_size"]["lots"] = 1
        cfg["position_size"]["qty_per_lot"] = 100
        cfg["entry"]["premium_min"] = 100
        cfg["entry"]["premium_max"] = 100
        trades = [
            {
                "trade_id": "t1",
                "token": "A",
                "entry_ts": 1,
                "exit_ts": 5,
                "entry_price": 100.0,
                "qty": 100,
                "net_pnl": -50.0,
                "direction": "long",
            }
        ]
        m = attach_portfolio_risk_metrics(
            compute_trade_metrics(trades),
            trades=trades,
            price_rows=[],
            cfg=cfg,
            execution_rules={
                "enabled": True,
                "max_open_positions": 3,
                "one_position_per_symbol": True,
            },
        )
        self.assertEqual(m["stop_loss_per_trade_rupees"], 1000.0)
        self.assertEqual(m["max_open_positions_for_risk"], 3)
        self.assertEqual(m["max_theoretical_portfolio_risk"], 3000.0)
        self.assertNotEqual(
            m["account_equity_max_drawdown"],
            m["max_theoretical_portfolio_risk"],
        )

    def test_max_portfolio_open_risk_sums_simultaneous_unrealized(self) -> None:
        from chain_replay_ml.strategy_simulator.metrics import (
            attach_portfolio_risk_metrics,
            compute_max_portfolio_drawdown_open_risk,
            compute_trade_metrics,
        )

        trades = [
            {
                "trade_id": "a",
                "token": "A",
                "entry_ts": 0.0,
                "exit_ts": 10.0,
                "entry_price": 100.0,
                "qty": 10,
                "net_pnl": -50.0,
                "direction": "long",
                "stop_loss_pct": 4.0,
                "stop_price": 96.0,
                "expected_stop_loss_rupees": 40.0,
                "lowest_mark_price": 90.0,
                "lowest_unrealized_pnl": -100.0,
                "exit_reason": "stop",
                "gap_beyond_stop": True,  # gapped through stop to 90
            },
            {
                "trade_id": "b",
                "token": "B",
                "entry_ts": 2.0,
                "exit_ts": 10.0,
                "entry_price": 100.0,
                "qty": 10,
                "net_pnl": -50.0,
                "direction": "long",
                "stop_loss_pct": 4.0,
                "stop_price": 96.0,
                "expected_stop_loss_rupees": 40.0,
                "lowest_mark_price": 90.0,
                "lowest_unrealized_pnl": -100.0,
                "exit_reason": "stop",
                "gap_beyond_stop": True,
            },
        ]
        open_risk = compute_max_portfolio_drawdown_open_risk(
            trades,
            execution_rules={"enabled": True, "max_open_positions": 3},
        )
        # Two overlapping gap legs → 100 + 100
        self.assertEqual(open_risk["max_portfolio_drawdown_open_risk"], 200.0)
        self.assertEqual(open_risk["observed_max_concurrent_open"], 2)

        m = attach_portfolio_risk_metrics(
            compute_trade_metrics(trades),
            trades=trades,
            cfg=get_default_template(),
            execution_rules={"enabled": True, "max_open_positions": 3},
        )
        self.assertEqual(m["max_portfolio_drawdown_open_risk"], 200.0)
        self.assertIsNotNone(m.get("worst_open_risk_trade"))

    def test_open_risk_with_max_open_1_uses_stop_aware_trough(self) -> None:
        from chain_replay_ml.strategy_simulator.metrics import compute_max_portfolio_drawdown_open_risk

        trades = [
            {
                "trade_id": "worst",
                "token": "A",
                "entry_ts": 0.0,
                "exit_ts": 5.0,
                "entry_price": 100.0,
                "exit_price": 96.0,
                "qty": 65,
                "direction": "long",
                "stop_loss_pct": 6.0,
                "stop_price": 94.0,
                "expected_stop_loss_rupees": 390.0,
                "lowest_mark_price": 94.0,
                "lowest_unrealized_pnl": -390.0,
                "exit_reason": "stop",
                "net_pnl": -390.0,
                "gap_beyond_stop": False,
            },
            {
                "trade_id": "poison",
                "token": "B",
                "entry_ts": 10.0,
                "exit_ts": 15.0,
                "entry_price": 100.0,
                "exit_price": 50.0,
                "qty": 65,
                "direction": "long",
                "stop_loss_pct": 6.0,
                "stop_price": 94.0,
                "expected_stop_loss_rupees": 390.0,
                # Uncapped phantom trough — must NOT drive Max Portfolio DD when no gap.
                "lowest_mark_price": 50.0,
                "lowest_unrealized_pnl": -18047.25,
                "exit_reason": "max_hold",
                "net_pnl": -130.0,
                "gap_beyond_stop": False,
            },
        ]
        open_risk = compute_max_portfolio_drawdown_open_risk(
            trades,
            execution_rules={"enabled": True, "max_open_positions": 1, "one_position_per_symbol": True},
        )
        # Capped phantom trade at stop ₹390; worst reported open DD is 390.
        self.assertEqual(open_risk["max_portfolio_drawdown_open_risk"], 390.0)
        wt = open_risk["worst_trade"]
        self.assertIsNotNone(wt)
        self.assertEqual(wt["expected_stop_loss_rupees"], 390.0)
        self.assertEqual(wt["position_value"], 6500.0)
        self.assertTrue(wt.get("stop_cap_applied") or wt.get("trade_id") == "worst")
        # Metric cannot materially exceed stop without gap.
        self.assertLessEqual(
            open_risk["max_portfolio_drawdown_open_risk"],
            390.0 * 1.01 + 1e-6,
        )
    def test_simulate_trade_uses_live_ltp_not_actual_horizon(self) -> None:
        """Stop/target must use live ltp; actual_ltp is the horizon label only."""
        cfg = self._cfg()
        cfg["stop"]["stop_loss_pct"] = 4.0
        cfg["target"]["target_profit_pct"] = 50.0
        cfg["hold_time"]["max_hold_sec"] = 60
        cfg["entry"]["premium_min"] = 1
        cfg["entry"]["premium_max"] = 200
        cfg["position_size"]["qty_per_lot"] = 65
        cfg["position_size"]["lots"] = 1
        rows = [
            {
                "prediction_id": "e",
                "fold_id": "f1",
                "row_index": 0,
                "timestamp": 100.0,
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 100.0,
                "predicted_ltp": 110.0,
                "actual_ltp": 1.0,  # horizon label — must NOT trigger stop at entry path
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
            {
                "prediction_id": "x",
                "fold_id": "f1",
                "row_index": 1,
                "timestamp": 105.0,
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 96.0,  # live mark hits 4% stop
                "predicted_ltp": 90.0,  # not an entry signal
                "actual_ltp": 1.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
        ]
        trades, stats = simulate_fold_rows(
            rows,
            cfg=cfg,
            strategy_run_id="sr1",
            strategy_version_id="sv1",
            prediction_run_id="pr1",
            fold_number=1,
            execution_rules={"enabled": True, "max_open_positions": 1, "one_position_per_symbol": True},
        )
        self.assertEqual(stats["trades"], 1)
        self.assertEqual(trades[0]["exit_reason"], "stop")
        self.assertEqual(trades[0]["exit_price"], 96.0)
        self.assertEqual(trades[0]["stop_price"], 96.0)
        self.assertEqual(trades[0]["lowest_mark_price"], 96.0)
        self.assertEqual(trades[0]["lowest_unrealized_pnl"], -260.0)  # (96-100)*65
        self.assertFalse(trades[0]["gap_beyond_stop"])
        self.assertEqual(trades[0]["exit_sample_index"], 1)

    def test_stop_exit_fills_at_stop_price_not_sample_ltp(self) -> None:
        """Sample below stop is evidence only; protective fill is at stop price."""
        cfg = self._cfg()
        cfg["stop"]["stop_loss_pct"] = 6.0
        cfg["target"]["target_profit_pct"] = 50.0
        cfg["hold_time"]["max_hold_sec"] = 60
        cfg["entry"]["premium_min"] = 1
        cfg["entry"]["premium_max"] = 200
        cfg["position_size"]["qty_per_lot"] = 65
        cfg["position_size"]["lots"] = 1
        rows = [
            {
                "prediction_id": "e",
                "fold_id": "f1",
                "row_index": 0,
                "timestamp": 100.0,
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 159.0,
                "predicted_ltp": 170.0,
                "actual_ltp": 1.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
            {
                "prediction_id": "x",
                "fold_id": "f1",
                "row_index": 1,
                "timestamp": 103.0,
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 135.50,  # sample already through 6% stop
                "predicted_ltp": 120.0,
                "actual_ltp": 1.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
        ]
        trades, stats = simulate_fold_rows(
            rows,
            cfg=cfg,
            strategy_run_id="sr_stop_fill",
            strategy_version_id="sv1",
            prediction_run_id="pr1",
            fold_number=1,
            execution_rules={"enabled": True, "max_open_positions": 1, "one_position_per_symbol": True},
        )
        self.assertEqual(stats["trades"], 1)
        t = trades[0]
        stop = 159.0 * (1.0 - 0.06)
        self.assertEqual(t["exit_reason"], "stop")
        self.assertAlmostEqual(t["stop_price"], stop, places=4)
        self.assertAlmostEqual(t["exit_price"], stop, places=4)
        self.assertAlmostEqual(t["stop_trigger_ltp"], 135.50, places=4)
        self.assertAlmostEqual(t["sample_exit_ltp"], 135.50, places=4)
        self.assertFalse(t["gap_beyond_stop"])
        # Floating trough capped at stop fill, not sample 135.50.
        self.assertAlmostEqual(t["lowest_unrealized_pnl"], (stop - 159.0) * 65, places=2)
        self.assertAlmostEqual(t["expected_stop_loss_rupees"], 159.0 * 65 * 0.06, places=2)
        # Gross uses stop fill, not sample print-through.
        self.assertAlmostEqual(t["gross_pnl"], (stop - 159.0) * 65, places=2)

    def test_gap_fill_at_sample_ltp_when_explicitly_enabled(self) -> None:
        """Optional sample-fill mode exits both target and stop at sample LTP."""
        cfg = self._cfg()
        cfg["stop"]["stop_loss_pct"] = 4.0
        cfg["execution"]["fill_at_sample_ltp"] = True
        cfg["target"]["target_profit_pct"] = 50.0
        cfg["hold_time"]["max_hold_sec"] = 60
        cfg["entry"]["premium_min"] = 1
        cfg["entry"]["premium_max"] = 200
        cfg["position_size"]["qty_per_lot"] = 65
        cfg["position_size"]["lots"] = 1
        rows = [
            {
                "prediction_id": "e",
                "fold_id": "f1",
                "row_index": 0,
                "timestamp": 100.0,
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 100.0,
                "predicted_ltp": 110.0,
                "actual_ltp": 1.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
            {
                "prediction_id": "x",
                "fold_id": "f1",
                "row_index": 1,
                "timestamp": 103.0,
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 90.0,
                "predicted_ltp": 80.0,
                "actual_ltp": 1.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
        ]
        trades, _stats = simulate_fold_rows(
            rows,
            cfg=cfg,
            strategy_run_id="sr_gap",
            strategy_version_id="sv1",
            prediction_run_id="pr1",
            fold_number=1,
            execution_rules={"enabled": True, "max_open_positions": 1, "one_position_per_symbol": True},
        )
        self.assertEqual(trades[0]["exit_price"], 90.0)
        self.assertTrue(trades[0]["gap_beyond_stop"])
        self.assertTrue(trades[0]["fill_at_sample_ltp"])

    def test_target_exit_fills_at_target_price_not_sample_ltp(self) -> None:
        """Sample above target is evidence only; limit fill is at target price."""
        cfg = self._cfg()
        cfg["stop"]["stop_loss_pct"] = 50.0
        cfg["target"]["target_profit_pct"] = 3.0
        cfg["hold_time"]["max_hold_sec"] = 60
        cfg["entry"]["premium_min"] = 1
        cfg["entry"]["premium_max"] = 200
        cfg["position_size"]["qty_per_lot"] = 65
        cfg["position_size"]["lots"] = 1
        rows = [
            {
                "prediction_id": "e",
                "fold_id": "f1",
                "row_index": 0,
                "timestamp": 100.0,
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 100.0,
                "predicted_ltp": 110.0,
                "actual_ltp": 1.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
            {
                "prediction_id": "x",
                "fold_id": "f1",
                "row_index": 1,
                "timestamp": 103.0,
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 108.0,  # sample already through 3% target
                "predicted_ltp": 90.0,  # not an entry signal
                "actual_ltp": 1.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
        ]
        trades, stats = simulate_fold_rows(
            rows,
            cfg=cfg,
            strategy_run_id="sr_tgt_fill",
            strategy_version_id="sv1",
            prediction_run_id="pr1",
            fold_number=1,
            execution_rules={"enabled": True, "max_open_positions": 1, "one_position_per_symbol": True},
        )
        self.assertEqual(stats["trades"], 1)
        t = trades[0]
        target = 100.0 * 1.03
        self.assertEqual(t["exit_reason"], "target")
        self.assertAlmostEqual(t["target_price"], target, places=4)
        self.assertAlmostEqual(t["exit_price"], target, places=4)
        self.assertAlmostEqual(t["target_trigger_ltp"], 108.0, places=4)
        self.assertAlmostEqual(t["sample_exit_ltp"], 108.0, places=4)
        self.assertFalse(t["fill_at_sample_ltp"])
        self.assertAlmostEqual(t["return_pct"], 3.0, places=4)
        self.assertAlmostEqual(t["gross_pnl"], (target - 100.0) * 65, places=2)

    def test_one_to_one_target_and_stop_fills_are_symmetric(self) -> None:
        """3% target and 3% stop both fill at configured prices (true 1:1 before fees)."""
        cfg = self._cfg()
        cfg["stop"]["stop_loss_pct"] = 3.0
        cfg["target"]["target_profit_pct"] = 3.0
        cfg["hold_time"]["max_hold_sec"] = 60
        cfg["entry"]["premium_min"] = 1
        cfg["entry"]["premium_max"] = 200
        cfg["position_size"]["qty_per_lot"] = 65
        cfg["position_size"]["lots"] = 1

        def _rows(exit_ltp: float) -> list[dict]:
            return [
                {
                    "prediction_id": "e",
                    "fold_id": "f1",
                    "row_index": 0,
                    "timestamp": 100.0,
                    "trading_day": "2026-07-01",
                    "token": "TOK",
                    "ltp": 100.0,
                    "predicted_ltp": 110.0,
                    "actual_ltp": 1.0,
                    "spot": 24050.0,
                    "strike": 24000.0,
                    "option_type": "CE",
                },
                {
                    "prediction_id": "x",
                    "fold_id": "f1",
                    "row_index": 1,
                    "timestamp": 103.0,
                    "trading_day": "2026-07-01",
                    "token": "TOK",
                    "ltp": exit_ltp,
                    "predicted_ltp": 90.0,
                    "actual_ltp": 1.0,
                    "spot": 24050.0,
                    "strike": 24000.0,
                    "option_type": "CE",
                },
            ]

        win, _ = simulate_fold_rows(
            _rows(110.0),
            cfg=cfg,
            strategy_run_id="sr_sym_w",
            strategy_version_id="sv1",
            prediction_run_id="pr1",
            fold_number=1,
            execution_rules={"enabled": True, "max_open_positions": 1, "one_position_per_symbol": True},
        )
        loss, _ = simulate_fold_rows(
            _rows(90.0),
            cfg=cfg,
            strategy_run_id="sr_sym_l",
            strategy_version_id="sv1",
            prediction_run_id="pr1",
            fold_number=1,
            execution_rules={"enabled": True, "max_open_positions": 1, "one_position_per_symbol": True},
        )
        self.assertEqual(win[0]["exit_reason"], "target")
        self.assertEqual(loss[0]["exit_reason"], "stop")
        self.assertAlmostEqual(win[0]["return_pct"], 3.0, places=4)
        self.assertAlmostEqual(loss[0]["return_pct"], -3.0, places=4)
        self.assertAlmostEqual(win[0]["gross_pnl"], -loss[0]["gross_pnl"], places=2)


    def test_outcome_audit_separates_target_fill_through_from_stop_fill(self) -> None:
        from chain_replay_ml.strategy_simulator.metrics import compute_outcome_audit

        trades = [
            {
                "entry_price": 100.0,
                "exit_price": 103.0,  # filled at target
                "target_price": 103.0,
                "stop_price": 97.0,
                "gross_pnl": 195.0,
                "fees": 10.0,
                "net_pnl": 185.0,
                "return_pct": 3.0,
                "exit_reason": "target",
                "direction": "long",
                "target_trigger_ltp": 104.0,
                "qty": 65,
            },
            {
                "entry_price": 100.0,
                "exit_price": 97.0,
                "target_price": 103.0,
                "stop_price": 97.0,
                "gross_pnl": -195.0,
                "fees": 10.0,
                "net_pnl": -205.0,
                "return_pct": -3.0,
                "exit_reason": "stop",
                "direction": "long",
                "stop_trigger_ltp": 95.0,
                "gap_beyond_stop": False,
                "qty": 65,
            },
        ]
        audit = compute_outcome_audit(trades)
        self.assertEqual(audit["target_exact"], 1)
        self.assertEqual(audit["target_above"], 0)
        self.assertEqual(audit["stop_exact"], 1)
        self.assertEqual(audit["stop_trigger_beyond_sample"], 1)
        self.assertEqual(audit["exit_reason_counts"]["target"], 1)
        self.assertEqual(audit["exit_reason_counts"]["stop"], 1)
        self.assertIn("Target Price", audit["asymmetry_note"])


class StopReplayDiagnosisTests(unittest.TestCase):
    """Gap vs ~3s sample-miss vs stop-bug for Worst Portfolio DD trade."""

    def test_genuine_gap_no_ticks_between_stop_and_exit(self) -> None:
        from chain_replay_ml.strategy_simulator.stop_replay import analyze_stop_path

        entry_ts = 1_000_000.0
        # Jump: 151 → 135.50 with nothing between stop 150.776 and exit.
        raw_ticks = [
            (entry_ts - 10, 155.0),
            (entry_ts, 152.0),
            (entry_ts + 1.0, 151.0),
            (entry_ts + 1.05, 135.50),  # gap print
        ]
        samples = [
            (entry_ts, 152.0),
            (entry_ts + 3.0, 135.50),
        ]
        out = analyze_stop_path(
            raw_ticks=raw_ticks,
            sample_marks=samples,
            entry_ts=entry_ts,
            exit_ts=entry_ts + 3.0,
            stop_price=150.776,
            exit_price=135.50,
            direction="long",
            exit_reason="stop",
        )
        self.assertEqual(out["diagnosis"], "genuine_gap")
        self.assertEqual(out["ticks_between_stop_and_exit"], 0)
        self.assertTrue(
            any(r.get("note") == "Raw Tick Stop Cross (Reference)" for r in out["raw_tick_rows"])
        )
        self.assertTrue(any(r.get("position_state") == "pre_entry" for r in out["raw_tick_rows"]))

    def test_sample_interval_miss_when_ticks_cross_stop_before_sample(self) -> None:
        from chain_replay_ml.strategy_simulator.stop_replay import analyze_stop_path

        entry_ts = 2_000_000.0
        # Gradual walk through stop on ticks; sim only sees breach 3s later.
        raw_ticks = [
            (entry_ts, 100.0),
            (entry_ts + 0.5, 96.0),   # first tick breach of stop 97
            (entry_ts + 1.0, 95.5),
            (entry_ts + 1.5, 95.0),
            (entry_ts + 2.0, 94.5),
            (entry_ts + 3.0, 94.0),
        ]
        samples = [
            (entry_ts, 100.0),
            (entry_ts + 3.0, 94.0),  # first sample at/beyond stop
        ]
        out = analyze_stop_path(
            raw_ticks=raw_ticks,
            sample_marks=samples,
            entry_ts=entry_ts,
            exit_ts=entry_ts + 3.0,
            stop_price=97.0,
            exit_price=94.0,
            direction="long",
            exit_reason="stop",
        )
        self.assertEqual(out["diagnosis"], "sample_interval_miss")
        self.assertEqual(out["path_kind"], "sampled_exit_after_raw_cross")
        self.assertIn("Execution Model", out["diagnosis_label"])
        self.assertTrue(out["raw_crossed_before_sim_exit"])
        self.assertGreater(out["ticks_between_stop_and_exit"], 0)
        self.assertAlmostEqual(out["first_tick_stop_breach_ltp"], 96.0)
        self.assertAlmostEqual(out["first_sample_stop_breach_ltp"], 94.0)
        self.assertTrue(
            any("Exit" in str(r.get("decision") or "") for r in out["sim_sample_rows"])
        )
        # Timelines stay separate — no mixed tick+sample in either list.
        self.assertEqual(len(out["raw_tick_rows"]), len(raw_ticks))
        self.assertEqual(len(out["sim_sample_rows"]), len(samples))

    def test_stop_bug_when_ticks_breach_but_sample_never_and_not_stop_exit(self) -> None:
        from chain_replay_ml.strategy_simulator.stop_replay import analyze_stop_path

        entry_ts = 3_000_000.0
        raw_ticks = [
            (entry_ts, 100.0),
            (entry_ts + 1.0, 90.0),
            (entry_ts + 5.0, 88.0),
        ]
        samples = [
            (entry_ts, 100.0),
            (entry_ts + 3.0, 99.0),  # never at/beyond stop
            (entry_ts + 6.0, 98.5),
        ]
        out = analyze_stop_path(
            raw_ticks=raw_ticks,
            sample_marks=samples,
            entry_ts=entry_ts,
            exit_ts=entry_ts + 6.0,
            stop_price=97.0,
            exit_price=88.0,  # material print-through without a stop exit
            direction="long",
            exit_reason="horizon",
        )
        self.assertEqual(out["diagnosis"], "stop_bug")

    def test_no_forward_path_skips_trade_not_zero_hold_max_hold(self) -> None:
        """Entry with no later samples must not emit same-ts max_hold (hold=0)."""
        cfg = get_default_template()
        cfg["entry"]["premium_min"] = 1
        cfg["entry"]["premium_max"] = 200
        cfg["hold_time"]["max_hold_sec"] = 300
        cfg["target"]["target_profit_pct"] = 6.0
        cfg["stop"]["stop_loss_pct"] = 4.0
        rows = [
            {
                "prediction_id": "only",
                "fold_id": "f1",
                "row_index": 0,
                "timestamp": 100.0,
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 80.0,
                "predicted_ltp": 90.0,
                "actual_ltp": 80.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
        ]
        trades, stats = simulate_fold_rows(
            rows,
            cfg=cfg,
            strategy_run_id="sr_nopath",
            strategy_version_id="sv1",
            prediction_run_id="pr1",
            fold_number=1,
        )
        self.assertEqual(trades, [])
        self.assertEqual(stats["trades"], 0)
        self.assertEqual(stats["skipped_no_path"], 1)

    def test_path_exhausted_before_max_hold_is_end_of_path(self) -> None:
        """Early path end must not be labeled max_hold when hold < max_hold_sec."""
        cfg = get_default_template()
        cfg["entry"]["premium_min"] = 1
        cfg["entry"]["premium_max"] = 200
        cfg["hold_time"]["max_hold_sec"] = 300
        cfg["target"]["target_profit_pct"] = 50.0
        cfg["stop"]["stop_loss_pct"] = 50.0
        rows = [
            {
                "prediction_id": "e",
                "fold_id": "f1",
                "row_index": 0,
                "timestamp": 100.0,
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 80.0,
                "predicted_ltp": 90.0,
                "actual_ltp": 80.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
            {
                "prediction_id": "x1",
                "fold_id": "f1",
                "row_index": 1,
                "timestamp": 103.0,
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 81.0,
                "predicted_ltp": 70.0,
                "actual_ltp": 81.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
            {
                "prediction_id": "x2",
                "fold_id": "f1",
                "row_index": 2,
                "timestamp": 106.0,
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 80.5,
                "predicted_ltp": 70.0,
                "actual_ltp": 80.5,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
        ]
        trades, stats = simulate_fold_rows(
            rows,
            cfg=cfg,
            strategy_run_id="sr_eop",
            strategy_version_id="sv1",
            prediction_run_id="pr1",
            fold_number=1,
        )
        self.assertEqual(stats["trades"], 1)
        t = trades[0]
        self.assertEqual(t["exit_reason"], "end_of_path")
        self.assertAlmostEqual(t["holding_seconds"], 6.0, places=3)
        self.assertLess(t["holding_seconds"], 300)
        self.assertEqual(t["exit_sample_index"], 2)

    def test_max_hold_requires_configured_duration(self) -> None:
        """max_hold only after hold >= max_hold_sec; never on entry sample."""
        cfg = get_default_template()
        cfg["entry"]["premium_min"] = 1
        cfg["entry"]["premium_max"] = 200
        cfg["hold_time"]["max_hold_sec"] = 9
        cfg["target"]["target_profit_pct"] = 50.0
        cfg["stop"]["stop_loss_pct"] = 50.0
        rows = [
            {
                "prediction_id": "e",
                "fold_id": "f1",
                "row_index": 0,
                "timestamp": 100.0,
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 80.0,
                "predicted_ltp": 90.0,
                "actual_ltp": 80.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
            {
                "prediction_id": "x1",
                "fold_id": "f1",
                "row_index": 1,
                "timestamp": 103.0,
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 81.0,
                "predicted_ltp": 70.0,
                "actual_ltp": 81.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
            {
                "prediction_id": "x2",
                "fold_id": "f1",
                "row_index": 2,
                "timestamp": 109.0,  # hold == 9 == max_hold_sec
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 80.5,
                "predicted_ltp": 70.0,
                "actual_ltp": 80.5,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
        ]
        trades, _stats = simulate_fold_rows(
            rows,
            cfg=cfg,
            strategy_run_id="sr_mh",
            strategy_version_id="sv1",
            prediction_run_id="pr1",
            fold_number=1,
        )
        self.assertEqual(len(trades), 1)
        t = trades[0]
        self.assertEqual(t["exit_reason"], "max_hold")
        self.assertGreaterEqual(t["holding_seconds"], 9.0)
        self.assertEqual(t["exit_sample_index"], 2)
        self.assertNotEqual(t["entry_ts"], t["exit_ts"])

    def test_stop_wins_on_sample_past_max_hold(self) -> None:
        """Sparse gap past max_hold that is also through stop must exit stop."""
        cfg = get_default_template()
        cfg["entry"]["premium_min"] = 1
        cfg["entry"]["premium_max"] = 200
        cfg["hold_time"]["max_hold_sec"] = 30
        cfg["target"]["target_profit_pct"] = 50.0
        cfg["stop"]["stop_loss_pct"] = 4.0
        rows = [
            {
                "prediction_id": "e",
                "fold_id": "f1",
                "row_index": 0,
                "timestamp": 100.0,
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 100.0,
                "predicted_ltp": 110.0,
                "actual_ltp": 100.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
            {
                "prediction_id": "x",
                "fold_id": "f1",
                "row_index": 1,
                "timestamp": 160.0,  # hold 60 > max_hold, LTP through stop
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 90.0,
                "predicted_ltp": 70.0,
                "actual_ltp": 90.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
        ]
        trades, _ = simulate_fold_rows(
            rows,
            cfg=cfg,
            strategy_run_id="sr_gap_stop",
            strategy_version_id="sv1",
            prediction_run_id="pr1",
            fold_number=1,
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["exit_reason"], "stop")
        self.assertAlmostEqual(trades[0]["exit_price"], 96.0, places=4)

    def test_minimum_predicted_move_pct_skips_small_moves(self) -> None:
        """V2 magnitude gate: tiny predicted moves do not enter."""
        cfg = get_default_template()
        cfg["entry"]["premium_min"] = 1
        cfg["entry"]["premium_max"] = 200
        cfg["entry"]["entry_cadence_sec"] = 1
        cfg["entry"]["minimum_predicted_move_pct"] = 5.0
        cfg["hold_time"]["max_hold_sec"] = 60
        cfg["target"]["target_profit_pct"] = 50.0
        cfg["stop"]["stop_loss_pct"] = 50.0

        def _row(pid: str, ts: float, ltp: float, pred: float, token: str = "TOK") -> dict:
            return {
                "prediction_id": pid,
                "fold_id": "f1",
                "row_index": int(ts),
                "timestamp": ts,
                "trading_day": "2026-07-01",
                "token": token,
                "ltp": ltp,
                "predicted_ltp": pred,
                "actual_ltp": ltp,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            }

        # +1% predicted — below 5% gate
        small = [
            _row("s0", 100.0, 100.0, 101.0, "A"),
            _row("s1", 103.0, 100.5, 90.0, "A"),
        ]
        trades_small, stats_small = simulate_fold_rows(
            small,
            cfg=cfg,
            strategy_run_id="sr_min_small",
            strategy_version_id="sv1",
            prediction_run_id="pr1",
            fold_number=1,
        )
        self.assertEqual(trades_small, [])
        self.assertEqual(stats_small["trades"], 0)
        self.assertGreaterEqual(stats_small["no_signal"], 1)

        # +8% predicted — above 5% gate
        big = [
            _row("b0", 100.0, 100.0, 108.0, "B"),
            _row("b1", 103.0, 101.0, 90.0, "B"),
        ]
        trades_big, stats_big = simulate_fold_rows(
            big,
            cfg=cfg,
            strategy_run_id="sr_min_big",
            strategy_version_id="sv1",
            prediction_run_id="pr1",
            fold_number=1,
        )
        self.assertEqual(stats_big["trades"], 1)
        self.assertEqual(trades_big[0]["entry_price"], 100.0)

    def test_minimum_predicted_move_zero_keeps_v1_direction_only(self) -> None:
        """Default 0% keeps V1: any positive predicted direction enters."""
        cfg = get_default_template()
        cfg["entry"]["premium_min"] = 1
        cfg["entry"]["premium_max"] = 200
        cfg["entry"]["entry_cadence_sec"] = 1
        cfg["entry"]["minimum_predicted_move_pct"] = 0.0
        cfg["hold_time"]["max_hold_sec"] = 60
        cfg["target"]["target_profit_pct"] = 50.0
        cfg["stop"]["stop_loss_pct"] = 50.0
        rows = [
            {
                "prediction_id": "e",
                "fold_id": "f1",
                "row_index": 0,
                "timestamp": 100.0,
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 100.0,
                "predicted_ltp": 100.01,  # +0.01%
                "actual_ltp": 100.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
            {
                "prediction_id": "x",
                "fold_id": "f1",
                "row_index": 1,
                "timestamp": 103.0,
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 100.0,
                "predicted_ltp": 90.0,
                "actual_ltp": 100.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
        ]
        trades, stats = simulate_fold_rows(
            rows,
            cfg=cfg,
            strategy_run_id="sr_v1",
            strategy_version_id="sv1",
            prediction_run_id="pr1",
            fold_number=1,
        )
        self.assertEqual(stats["trades"], 1)
        self.assertEqual(len(trades), 1)

    def test_use_predicted_ltp_as_target_price(self) -> None:
        """When use_predicted_ltp=true, target is entry predicted_ltp (not %)."""
        cfg = get_default_template()
        cfg["entry"]["premium_min"] = 1
        cfg["entry"]["premium_max"] = 200
        cfg["entry"]["entry_cadence_sec"] = 1
        cfg["hold_time"]["max_hold_sec"] = 60
        cfg["target"]["target_profit_pct"] = 50.0  # ignored when flag on
        cfg["target"]["use_predicted_ltp"] = True
        cfg["stop"]["stop_loss_pct"] = 50.0
        rows = [
            {
                "prediction_id": "e",
                "fold_id": "f1",
                "row_index": 0,
                "timestamp": 100.0,
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 100.0,
                "predicted_ltp": 108.0,  # target = 108, not 150
                "actual_ltp": 100.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
            {
                "prediction_id": "x",
                "fold_id": "f1",
                "row_index": 1,
                "timestamp": 103.0,
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 109.0,  # crosses predicted target
                "predicted_ltp": 90.0,
                "actual_ltp": 109.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
        ]
        trades, stats = simulate_fold_rows(
            rows,
            cfg=cfg,
            strategy_run_id="sr_pred_tgt",
            strategy_version_id="sv1",
            prediction_run_id="pr1",
            fold_number=1,
        )
        self.assertEqual(stats["trades"], 1)
        t = trades[0]
        self.assertTrue(t["use_predicted_ltp"])
        self.assertAlmostEqual(t["target_price"], 108.0, places=4)
        self.assertEqual(t["exit_reason"], "target")
        self.assertAlmostEqual(t["exit_price"], 108.0, places=4)  # limit fill at target
        self.assertAlmostEqual(t["target_trigger_ltp"], 109.0, places=4)

    def test_classifier_filter_does_not_thin_exit_path(self) -> None:
        """Entry-only classifier rows must not hide stop crosses on the mark path."""
        cfg = get_default_template()
        cfg["entry"]["premium_min"] = 1
        cfg["entry"]["premium_max"] = 200
        cfg["hold_time"]["max_hold_sec"] = 300
        cfg["target"]["target_profit_pct"] = 50.0
        cfg["stop"]["stop_loss_pct"] = 4.0
        entry = {
            "prediction_id": "e",
            "fold_id": "",
            "row_index": 0,
            "timestamp": 100.0,
            "trading_day": "2026-07-01",
            "token": "TOK",
            "ltp": 100.0,
            "predicted_ltp": 110.0,
            "actual_ltp": 100.0,
            "spot": 24050.0,
            "strike": 24000.0,
            "option_type": "CE",
        }
        # Intermediate path sample crosses stop but is NOT an entry candidate.
        path_stop = {
            "prediction_id": "p_stop",
            "fold_id": "",
            "row_index": 1,
            "timestamp": 112.0,
            "trading_day": "2026-07-01",
            "token": "TOK",
            "ltp": 95.0,
            "predicted_ltp": 70.0,
            "actual_ltp": 95.0,
            "spot": 24050.0,
            "strike": 24000.0,
            "option_type": "CE",
        }
        later = {
            "prediction_id": "p_late",
            "fold_id": "",
            "row_index": 2,
            "timestamp": 400.0,
            "trading_day": "2026-07-01",
            "token": "TOK",
            "ltp": 80.0,
            "predicted_ltp": 70.0,
            "actual_ltp": 80.0,
            "spot": 24050.0,
            "strike": 24000.0,
            "option_type": "CE",
        }
        # Bug reproduction: entry candidates omit the stop sample.
        entry_rows = [entry, later]
        path_rows = [entry, path_stop, later]
        from chain_replay_ml.strategy_simulator.engine import simulate_prediction_rows

        trades, _ = simulate_prediction_rows(
            entry_rows,
            cfg=cfg,
            strategy_run_id="sr_clf_path",
            strategy_version_id="sv1",
            prediction_run_id="pr1",
            path_rows=path_rows,
        )
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["exit_reason"], "stop")
        self.assertAlmostEqual(trades[0]["holding_seconds"], 12.0, places=3)
        self.assertAlmostEqual(trades[0]["sample_exit_ltp"], 95.0, places=4)

    def test_entry_signal_requires_regression_direction_by_default(self) -> None:
        cfg = get_default_template()
        cfg["entry"]["premium_min"] = 1
        cfg["entry"]["premium_max"] = 200
        row_up = {
            "ltp": 100.0,
            "predicted_ltp": 110.0,
            "spot": 24050.0,
            "strike": 24000.0,
            "option_type": "CE",
        }
        row_down = {**row_up, "predicted_ltp": 90.0}
        row_missing_pred = {**row_up, "predicted_ltp": None}
        self.assertTrue(_entry_signal(row_up, cfg))
        self.assertFalse(_entry_signal(row_down, cfg))
        self.assertFalse(_entry_signal(row_missing_pred, cfg))

    def test_entry_signal_without_regression_skips_predicted_ltp_gates(self) -> None:
        cfg = get_default_template()
        cfg["entry"]["premium_min"] = 1
        cfg["entry"]["premium_max"] = 200
        cfg["entry"]["use_regression"] = False
        cfg["entry"]["minimum_predicted_move_pct"] = 50.0  # ignored when off
        row_down = {
            "ltp": 100.0,
            "predicted_ltp": 50.0,  # would fail direction + min-move with regression on
            "spot": 24050.0,
            "strike": 24000.0,
            "option_type": "CE",
        }
        row_no_pred = {
            "ltp": 100.0,
            "predicted_ltp": None,
            "spot": 24050.0,
            "strike": 24000.0,
            "option_type": "CE",
        }
        row_premium_fail = {**row_down, "ltp": 0.5}
        self.assertTrue(_entry_signal(row_down, cfg))
        self.assertTrue(_entry_signal(row_no_pred, cfg))
        self.assertFalse(_entry_signal(row_premium_fail, cfg))

    def test_without_regression_forces_pct_target_not_predicted_ltp(self) -> None:
        """use_predicted_ltp is ignored when use_regression is False."""
        cfg = get_default_template()
        cfg["entry"]["premium_min"] = 1
        cfg["entry"]["premium_max"] = 200
        cfg["entry"]["entry_cadence_sec"] = 1
        cfg["entry"]["use_regression"] = False
        cfg["hold_time"]["max_hold_sec"] = 60
        cfg["target"]["target_profit_pct"] = 8.0
        cfg["target"]["use_predicted_ltp"] = True
        cfg["stop"]["stop_loss_pct"] = 50.0
        rows = [
            {
                "prediction_id": "e",
                "fold_id": "f1",
                "row_index": 0,
                "timestamp": 100.0,
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 100.0,
                "predicted_ltp": 105.0,  # would be target if regression on
                "actual_ltp": 100.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
            {
                "prediction_id": "x",
                "fold_id": "f1",
                "row_index": 1,
                "timestamp": 103.0,
                "trading_day": "2026-07-01",
                "token": "TOK",
                "ltp": 109.0,  # crosses 8% target (108), not just predicted 105
                "predicted_ltp": 90.0,
                "actual_ltp": 109.0,
                "spot": 24050.0,
                "strike": 24000.0,
                "option_type": "CE",
            },
        ]
        trades, stats = simulate_fold_rows(
            rows,
            cfg=cfg,
            strategy_run_id="sr_no_reg_tgt",
            strategy_version_id="sv1",
            prediction_run_id="pr1",
            fold_number=1,
        )
        self.assertEqual(stats["trades"], 1)
        t = trades[0]
        self.assertFalse(t["use_predicted_ltp"])
        self.assertAlmostEqual(t["target_price"], 108.0, places=4)
        self.assertEqual(t["exit_reason"], "target")


if __name__ == "__main__":
    unittest.main()
