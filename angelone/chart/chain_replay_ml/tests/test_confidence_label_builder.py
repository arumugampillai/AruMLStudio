"""Tests for Confidence Label Builder + TargetSpec registry."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.model_lab.target_spec import (
    REPLAY_TARGET_SPECS,
    TARGET_SPEC_BY_KEY,
    derive_binary_labels,
    inference_columns_for_key,
)
from chain_replay_ml.strategy_simulator.engine import simulate_forced_entry_outcomes
from chain_replay_ml.strategy_registry.schema import normalize_strategy_config


class TargetSpecTests(unittest.TestCase):
    def test_market_and_replay_families(self) -> None:
        self.assertEqual(TARGET_SPEC_BY_KEY["target_hit"].family, "market")
        self.assertEqual(TARGET_SPEC_BY_KEY["trade_winner"].family, "replay_based")
        self.assertTrue(TARGET_SPEC_BY_KEY["profit_250"].is_replay_based)

    def test_derive_binary_labels_from_continuous(self) -> None:
        outcomes = [
            {
                "prediction_id": "a",
                "net_pnl": 320.0,
                "return_pct": 8.2,
                "time_to_first_profit_sec": 95.0,
                "exit_reason": "target",
            },
            {
                "prediction_id": "b",
                "net_pnl": -180.0,
                "return_pct": -4.5,
                "time_to_first_profit_sec": None,
                "exit_reason": "stop",
            },
            {
                "prediction_id": "c",
                "net_pnl": 90.0,
                "return_pct": 2.1,
                "time_to_first_profit_sec": 40.0,
                "exit_reason": "max_hold",
            },
        ]
        rows = {r["prediction_id"]: r for r in derive_binary_labels(outcomes)}
        self.assertEqual(rows["a"]["trade_winner"], 1)
        self.assertEqual(rows["a"]["profit_250"], 1)
        self.assertEqual(rows["a"]["return_5"], 1)
        self.assertEqual(rows["a"]["fast_120"], 1)
        self.assertEqual(rows["a"]["target_exit"], 1)

        self.assertEqual(rows["b"]["trade_winner"], 0)
        self.assertEqual(rows["b"]["profit_250"], 0)
        self.assertEqual(rows["b"]["fast_120"], None)
        self.assertEqual(rows["b"]["target_exit"], 0)

        self.assertEqual(rows["c"]["trade_winner"], 1)
        self.assertEqual(rows["c"]["profit_250"], 0)
        self.assertEqual(rows["c"]["return_5"], 0)
        self.assertEqual(rows["c"]["fast_60"], 1)
        self.assertEqual(rows["c"]["target_exit"], 0)

    def test_inference_column_naming(self) -> None:
        th = inference_columns_for_key("target_hit")
        self.assertEqual(th["pred"], "confidence_target_hit_pred")
        tw = inference_columns_for_key("trade_winner")
        self.assertEqual(tw["pred"], "confidence_trade_winner_pred")

    def test_manifest_includes_replay_targets(self) -> None:
        from chain_replay_ml.model_lab.confidence_manifest import (
            CONFIDENCE_TARGETS,
            COLUMN_BY_KEY,
        )

        keys = {t["key"] for t in CONFIDENCE_TARGETS}
        self.assertIn("target_hit", keys)
        self.assertIn("trade_winner", keys)
        self.assertIn("profit_250", keys)
        self.assertEqual(COLUMN_BY_KEY["trade_winner"], "trade_winner")
        self.assertGreaterEqual(len(REPLAY_TARGET_SPECS), 5)


class ForcedEntryReplayTests(unittest.TestCase):
    def test_forced_entry_produces_outcomes_for_every_row(self) -> None:
        cfg = normalize_strategy_config({
            "entry": {
                "direction": "long",
                "premium_min": 1.0,
                "premium_max": 100.0,
                "atm_band": 0,
                "entry_cadence_sec": 3,
                "option_types": ["CE", "PE"],
            },
            "target": {"target_profit_pct": 8.0},
            "stop": {"stop_loss_pct": 5.0},
            "hold_time": {"max_hold_sec": 30},
            "position_size": {"lots": 1, "qty_per_lot": 65},
            "execution": {"fees_mode": "zero"},
        })
        # Three ticks on same token — first is entry, later path hits +8%
        rows = [
            {
                "prediction_id": "p1",
                "timestamp": 1000.0,
                "trading_day": "2026-05-26",
                "token": "T1",
                "strike": 24000.0,
                "option_type": "CE",
                "spot": 24000.0,
                "ltp": 20.0,
                "predicted_ltp": 22.0,
                "actual_ltp": 20.0,
                "row_index": 1,
            },
            {
                "prediction_id": "p2",
                "timestamp": 1005.0,
                "trading_day": "2026-05-26",
                "token": "T1",
                "strike": 24000.0,
                "option_type": "CE",
                "spot": 24000.0,
                "ltp": 21.0,
                "predicted_ltp": 22.0,
                "actual_ltp": 21.6,  # +8% from 20
                "row_index": 2,
            },
            {
                "prediction_id": "p3",
                "timestamp": 1010.0,
                "trading_day": "2026-05-26",
                "token": "T1",
                "strike": 24000.0,
                "option_type": "CE",
                "spot": 24000.0,
                "ltp": 22.0,
                "predicted_ltp": 23.0,
                "actual_ltp": 22.0,
                "row_index": 3,
            },
        ]
        outcomes, stats = simulate_forced_entry_outcomes(
            rows, cfg=cfg, strategy_version_id="ver1"
        )
        self.assertEqual(len(outcomes), 3)
        self.assertEqual(stats["predictions_evaluated"], 3)
        self.assertGreaterEqual(stats["outcomes"], 1)
        o1 = next(o for o in outcomes if o["prediction_id"] == "p1")
        self.assertEqual(o1["exit_reason"], "target")
        self.assertGreater(o1["net_pnl"], 0)
        self.assertIsNotNone(o1["time_to_first_profit_sec"])
        binaries = derive_binary_labels(outcomes)
        b1 = next(b for b in binaries if b["prediction_id"] == "p1")
        self.assertEqual(b1["trade_winner"], 1)
        self.assertEqual(b1["target_exit"], 1)

class ConfidenceLabelBuilderPersistTests(unittest.TestCase):
    def test_builder_persists_meta_and_parquets(self) -> None:
        from chain_replay_ml.model_lab.confidence_label_builder import (
            run_confidence_label_builder,
            read_latest_label_run,
            load_replay_outcome_frames,
        )
        from chain_replay_ml.model_lab.store import ModelLabStore
        from chain_replay_ml.strategy_registry.service import create_strategy

        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            os.makedirs(data_dir)
            lab_path = os.path.join(tmp, "lab.db")
            with ModelLabStore(lab_path) as store:
                store._ensure_schema()
                store.ensure_prediction_schema()
                store.write_info(
                    lab_uuid="u1",
                    lab_id="lab1",
                    lab_name="Test Lab",
                    parent_model_id="m1",
                    parent_model_name="Future_LTP_5m_Test",
                    model_checksum=None,
                    description=None,
                    purpose=None,
                    version=1,
                    original_feature_count=2,
                    selected_feature_count=2,
                    training_rows=2,
                    target="future_ltp_5m",
                    algorithm="xgboost",
                    dataset_snapshot={"dataset_name": "MS_test"},
                    model_snapshot=None,
                    training_config_snapshot=None,
                    wf_snapshot=None,
                    metrics_snapshot=None,
                    selected_features_snapshot=["f1", "f2"],
                    feature_ranking_snapshot=None,
                    artifact_pointers={},
                )
                store.write_prediction_summary(
                    lab_uuid="u1",
                    status="ready",
                    row_count=2,
                    trading_days=1,
                    target_column="future_ltp_5m",
                    parent_dataset="MS_test",
                    parent_model_name="Future_LTP_5m_Test",
                    created_at="2026-07-16T10:00:00+00:00",
                    dataset_hash="abc123",
                )
                store.insert_prediction_rows(
                    [
                        {
                            "lab_uuid": "u1",
                            "prediction_id": "p1",
                            "trading_day": "2026-05-26",
                            "timestamp": 1000.0,
                            "token": "T1",
                            "strike": 24000.0,
                            "option_type": "CE",
                            "current_spot": 24000.0,
                            "current_ltp": 20.0,
                            "predicted_future_ltp": 22.0,
                            "actual_future_ltp": 20.0,
                        },
                        {
                            "lab_uuid": "u1",
                            "prediction_id": "p2",
                            "trading_day": "2026-05-26",
                            "timestamp": 1005.0,
                            "token": "T1",
                            "strike": 24000.0,
                            "option_type": "CE",
                            "current_spot": 24000.0,
                            "current_ltp": 21.0,
                            "predicted_future_ltp": 22.0,
                            "actual_future_ltp": 21.6,
                        },
                    ]
                )

            detail = create_strategy(
                data_dir,
                display_name="Label Test Strat",
                config={
                    "entry": {
                        "direction": "long",
                        "premium_min": 1.0,
                        "premium_max": 100.0,
                        "atm_band": 0,
                        "entry_cadence_sec": 1,
                        "option_types": ["CE", "PE"],
                    },
                    "target": {"target_profit_pct": 8.0},
                    "stop": {"stop_loss_pct": 5.0},
                    "hold_time": {"max_hold_sec": 30},
                    "position_size": {"lots": 1, "qty_per_lot": 65},
                    "execution": {"fees_mode": "zero"},
                },
            )
            version_id = str(
                (detail.get("champion_version") or {}).get("version_id")
                or detail.get("champion_version_id")
                or ""
            )
            self.assertTrue(version_id, f"no version in {list(detail.keys())}")

            result = run_confidence_label_builder(
                lab_path,
                data_dir=data_dir,
                strategy_version_id=version_id,
            )
            self.assertTrue(result.get("ok"), result)
            self.assertEqual(result.get("strategy_version_id"), version_id)
            self.assertTrue(result.get("strategy_config_hash"))
            self.assertEqual(result.get("replay_mode"), "forced_entry")

            latest = read_latest_label_run(lab_path)
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest["label_run_id"], result["label_run_id"])

            loaded = load_replay_outcome_frames(lab_path)
            self.assertTrue(loaded.get("ok"))
            self.assertGreater(len(loaded["outcomes"]), 0)
            self.assertIn("trade_winner", loaded["binary_labels"].columns)
            oc = loaded["outcomes"]
            self.assertIn("prediction_id", oc.columns)
            self.assertFalse(oc["prediction_id"].isna().all())
            self.assertTrue(set(oc["prediction_id"].dropna().astype(str)).issubset({"p1", "p2"}))
            self.assertIn("exit_reason", oc.columns)
            self.assertTrue(oc["exit_reason"].notna().any())
            for col in (
                "net_pnl",
                "gross_pnl",
                "return_pct",
                "max_adverse_pct",
                "max_favorable_pct",
                "holding_seconds",
                "fees",
                "would_enter",
            ):
                self.assertIn(col, oc.columns)


class ConfidenceLabelStalenessTests(unittest.TestCase):
    def _seed_lab_and_strategy(self, tmp: str) -> tuple[str, str, str]:
        from chain_replay_ml.model_lab.confidence_label_builder import (
            run_confidence_label_builder,
        )
        from chain_replay_ml.model_lab.store import ModelLabStore
        from chain_replay_ml.strategy_registry.service import create_strategy

        data_dir = os.path.join(tmp, "data")
        os.makedirs(data_dir)
        lab_path = os.path.join(tmp, "lab.db")
        with ModelLabStore(lab_path) as store:
            store._ensure_schema()
            store.ensure_prediction_schema()
            store.write_info(
                lab_uuid="u1",
                lab_id="lab1",
                lab_name="Test Lab",
                parent_model_id="m1",
                parent_model_name="Future_LTP_5m_Test",
                model_checksum=None,
                description=None,
                purpose=None,
                version=1,
                original_feature_count=2,
                selected_feature_count=2,
                training_rows=2,
                target="future_ltp_5m",
                algorithm="xgboost",
                dataset_snapshot={"dataset_name": "MS_test"},
                model_snapshot=None,
                training_config_snapshot=None,
                wf_snapshot=None,
                metrics_snapshot=None,
                selected_features_snapshot=["f1", "f2"],
                feature_ranking_snapshot=None,
                artifact_pointers={},
            )
            store.write_prediction_summary(
                lab_uuid="u1",
                status="ready",
                row_count=2,
                trading_days=1,
                target_column="future_ltp_5m",
                parent_dataset="MS_test",
                parent_model_name="Future_LTP_5m_Test",
                created_at="2026-07-16T10:00:00+00:00",
                dataset_hash="abc123",
            )
            store.insert_prediction_rows(
                [
                    {
                        "lab_uuid": "u1",
                        "prediction_id": "p1",
                        "trading_day": "2026-05-26",
                        "timestamp": 1000.0,
                        "token": "T1",
                        "strike": 24000.0,
                        "option_type": "CE",
                        "current_spot": 24000.0,
                        "current_ltp": 20.0,
                        "predicted_future_ltp": 22.0,
                        "actual_future_ltp": 20.0,
                    },
                    {
                        "lab_uuid": "u1",
                        "prediction_id": "p2",
                        "trading_day": "2026-05-26",
                        "timestamp": 1005.0,
                        "token": "T1",
                        "strike": 24000.0,
                        "option_type": "CE",
                        "current_spot": 24000.0,
                        "current_ltp": 21.0,
                        "predicted_future_ltp": 22.0,
                        "actual_future_ltp": 21.6,
                    },
                ]
            )

        detail = create_strategy(
            data_dir,
            display_name="Stale Test Strat",
            config={
                "entry": {
                    "direction": "long",
                    "premium_min": 1.0,
                    "premium_max": 100.0,
                    "atm_band": 0,
                    "entry_cadence_sec": 1,
                    "option_types": ["CE", "PE"],
                },
                "target": {"target_profit_pct": 8.0},
                "stop": {"stop_loss_pct": 5.0},
                "hold_time": {"max_hold_sec": 30},
                "position_size": {"lots": 1, "qty_per_lot": 65},
                "execution": {"fees_mode": "zero"},
            },
        )
        version_id = str(
            (detail.get("champion_version") or {}).get("version_id")
            or detail.get("champion_version_id")
            or ""
        )
        result = run_confidence_label_builder(
            lab_path,
            data_dir=data_dir,
            strategy_version_id=version_id,
        )
        self.assertTrue(result.get("ok"), result)
        return lab_path, data_dir, version_id

    def test_missing_labels_status(self) -> None:
        from chain_replay_ml.model_lab.confidence_label_builder import (
            assess_label_run_staleness,
            confidence_labels_status,
        )

        with tempfile.TemporaryDirectory() as tmp:
            lab_path = os.path.join(tmp, "empty.db")
            data_dir = os.path.join(tmp, "data")
            os.makedirs(data_dir)
            stale = assess_label_run_staleness(lab_path, data_dir=data_dir)
            self.assertFalse(stale["has_run"])
            self.assertEqual(stale["status"], "missing")
            self.assertTrue(stale["rebuild_recommended"])

            status = confidence_labels_status(lab_path, data_dir=data_dir)
            self.assertEqual(status["run_count"], 0)
            self.assertEqual(status["staleness"]["status"], "missing")

    def test_up_to_date_after_build(self) -> None:
        from chain_replay_ml.model_lab.confidence_label_builder import (
            assess_label_run_staleness,
            confidence_labels_status,
            list_confidence_label_runs,
        )

        with tempfile.TemporaryDirectory() as tmp:
            lab_path, data_dir, _vid = self._seed_lab_and_strategy(tmp)
            stale = assess_label_run_staleness(lab_path, data_dir=data_dir)
            self.assertTrue(stale["has_run"])
            self.assertTrue(stale["up_to_date"])
            self.assertEqual(stale["status"], "up_to_date")
            self.assertEqual(stale["status_display"], "Up to date")

            status = confidence_labels_status(lab_path, data_dir=data_dir)
            self.assertEqual(status["run_count"], 1)
            self.assertTrue(status["staleness"]["up_to_date"])
            runs = list_confidence_label_runs(lab_path)
            self.assertEqual(len(runs), 1)
            self.assertTrue(runs[0]["is_latest"])

    def test_stale_when_prediction_hash_changes(self) -> None:
        from chain_replay_ml.model_lab.confidence_label_builder import (
            assess_label_run_staleness,
        )
        from chain_replay_ml.model_lab.store import ModelLabStore

        with tempfile.TemporaryDirectory() as tmp:
            lab_path, data_dir, _vid = self._seed_lab_and_strategy(tmp)
            with ModelLabStore(lab_path) as store:
                store.write_prediction_summary(
                    lab_uuid="u1",
                    status="ready",
                    row_count=2,
                    trading_days=1,
                    target_column="future_ltp_5m",
                    parent_dataset="MS_test",
                    parent_model_name="Future_LTP_5m_Test",
                    created_at="2026-07-16T12:00:00+00:00",
                    dataset_hash="changed_hash_999",
                )
            stale = assess_label_run_staleness(lab_path, data_dir=data_dir)
            self.assertFalse(stale["up_to_date"])
            self.assertEqual(stale["status"], "out_of_date")
            self.assertTrue(
                any("Prediction Dataset changed" in r for r in stale["reasons"])
            )

    def test_stale_when_strategy_config_changes(self) -> None:
        from chain_replay_ml.model_lab.confidence_label_builder import (
            assess_label_run_staleness,
            read_latest_label_run,
        )
        from chain_replay_ml.strategy_registry.service import create_strategy_version

        with tempfile.TemporaryDirectory() as tmp:
            lab_path, data_dir, version_id = self._seed_lab_and_strategy(tmp)
            latest = read_latest_label_run(lab_path)
            assert latest is not None
            strategy_id = str(latest.get("strategy_id") or "")
            if not strategy_id:
                from chain_replay_ml.strategy_registry.service import get_strategy_version

                ver = get_strategy_version(data_dir, version_id)
                strategy_id = str((ver or {}).get("strategy_id") or "")
            self.assertTrue(strategy_id)

            create_strategy_version(
                data_dir,
                strategy_id=strategy_id,
                config={
                    "name": "Stale Test Strat",
                    "entry": {
                        "direction": "long",
                        "premium_min": 1.0,
                        "premium_max": 100.0,
                        "atm_band": 0,
                        "entry_cadence_sec": 1,
                        "option_types": ["CE", "PE"],
                    },
                    "target": {"target_profit_pct": 12.0},  # changed → new champion
                    "stop": {"stop_loss_pct": 5.0},
                    "hold_time": {"max_hold_sec": 30},
                    "position_size": {"lots": 1, "qty_per_lot": 65},
                    "execution": {"fees_mode": "zero"},
                },
                set_champion=True,
            )
            stale = assess_label_run_staleness(lab_path, data_dir=data_dir)
            self.assertFalse(stale["up_to_date"])
            self.assertEqual(stale["status"], "out_of_date")
            self.assertTrue(
                any("Strategy configuration changed" in r for r in stale["reasons"])
            )


if __name__ == "__main__":
    unittest.main()
