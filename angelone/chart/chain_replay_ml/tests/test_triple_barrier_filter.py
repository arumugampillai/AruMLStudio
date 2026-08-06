"""Unit tests for the Strategy Simulator Triple Barrier (TB) filter — Phase 2.

Covers:
  * TB filter pass/fail (class + threshold, NULL exclude)
  * TB disabled leaves results unchanged (same candidate set / same trades)
  * Class mapping resolved from metadata fixture (not hardcoded 0/1/2)
  * Comparison payload structure (Baseline vs Filtered) when enabled
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from chain_replay_ml.model_lab.store import ModelLabStore
from chain_replay_ml.strategy_registry import create_strategy, get_default_template
from chain_replay_ml.strategy_simulator import (
    apply_tb_filter,
    build_tb_comparison_payload,
    discover_tb_model_name,
    resolve_tb_class_options,
    run_strategy_simulation_from_lab,
    run_strategy_simulation_from_lab_with_tb_comparison,
    tb_filter_options,
)


class ApplyTbFilterTests(unittest.TestCase):
    """Pure predicate tests — no DB, no simulation."""

    def _rows(self) -> list[dict]:
        return [
            {"prediction_id": "a", "tb_pred_class": 0, "tb_pred_probability": 0.85},
            {"prediction_id": "b", "tb_pred_class": 0, "tb_pred_probability": 0.55},
            {"prediction_id": "c", "tb_pred_class": 1, "tb_pred_probability": 0.90},
            {"prediction_id": "d", "tb_pred_class": None, "tb_pred_probability": None},
            {"prediction_id": "e", "tb_pred_class": 0, "tb_pred_probability": None},
        ]

    def test_disabled_is_passthrough(self) -> None:
        rows = self._rows()
        kept, meta = apply_tb_filter(rows, class_id=None, threshold=0.6)
        self.assertEqual(kept, rows)
        self.assertFalse(meta["active"])
        self.assertIsNone(meta["class_id"])
        self.assertIsNone(meta["threshold"])
        self.assertEqual(meta["rows_before"], len(rows))
        self.assertEqual(meta["rows_after"], len(rows))

    def test_class_and_threshold_and_gate(self) -> None:
        kept, meta = apply_tb_filter(
            self._rows(),
            class_id=0,
            threshold=0.6,
            label="TP",
            class_labels={0: "TP", 1: "SL", 2: "TIME"},
        )
        # Only "a" is class 0 AND prob >= 0.60. "b" fails threshold, "c" fails
        # class, "d"/"e" are NULL and excluded regardless of threshold.
        self.assertEqual([r["prediction_id"] for r in kept], ["a"])
        self.assertTrue(meta["active"])
        self.assertEqual(meta["class_id"], 0)
        self.assertEqual(meta["label"], "TP")
        self.assertEqual(meta["threshold"], 0.6)
        self.assertEqual(meta["rows_before"], 5)
        self.assertEqual(meta["rows_after"], 1)
        self.assertEqual(meta["rows_removed"], 4)
        # "d" and "e" are the NULL rows — never guessed into a class.
        self.assertEqual(meta["rows_null"], 2)
        self.assertEqual(meta["skip_reason"], "Missing Triple Barrier prediction")
        # Class counts among candidates (excludes NULL rows).
        self.assertEqual(meta["class_counts"], {"TP": 2, "SL": 1})
        self.assertEqual(meta["avg_tb_probability"], 0.85)

    def test_null_rows_never_pass_even_at_zero_threshold(self) -> None:
        rows = [{"prediction_id": "z", "tb_pred_class": None, "tb_pred_probability": None}]
        with self.assertRaises(ValueError) as ctx:
            apply_tb_filter(rows, class_id=0, threshold=0.0)
        self.assertIn("Rebuild the Prediction Dataset", str(ctx.exception))

    def test_all_removed_raises(self) -> None:
        with self.assertRaises(ValueError):
            apply_tb_filter(self._rows(), class_id=0, threshold=0.99)

    def test_probability_distribution_buckets_selected_class_only(self) -> None:
        rows = [
            {"prediction_id": "a", "tb_pred_class": 0, "tb_pred_probability": 0.52},
            {"prediction_id": "b", "tb_pred_class": 0, "tb_pred_probability": 0.95},
            {"prediction_id": "c", "tb_pred_class": 1, "tb_pred_probability": 0.99},
        ]
        _kept, meta = apply_tb_filter(rows, class_id=0, threshold=0.0)
        dist = meta["probability_distribution"]
        self.assertEqual(dist["0.50-0.60"], 1)
        self.assertEqual(dist["0.90-1.00"], 1)
        # Class-1 row must not leak into the class-0 distribution.
        self.assertEqual(sum(dist.values()), 2)


class ResolveTbClassOptionsTests(unittest.TestCase):
    """Class labels must come from metadata, never literal ints in filter logic."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def _write_json(self, path: str, doc: dict) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)

    def test_resolves_custom_encoding_from_label_registry(self) -> None:
        """A non-canonical label set (not TP/SL/TIME) must round-trip exactly."""
        from chain_replay_ml.label_runs.paths import label_run_meta_path, label_run_parquet_path
        from chain_replay_ml.training.paths import model_artifact_paths

        model_name = "TBM_custom"
        run_id = "custom_run_1"

        paths = model_artifact_paths(self.tmp, model_name)
        self._write_json(
            paths["config_json"],
            {"algorithm": "xgboost", "features": ["f1", "f2"], "label_run_id": run_id},
        )

        meta_path = label_run_meta_path(self.tmp, run_id)
        self._write_json(
            meta_path,
            {
                "run_id": run_id,
                "strategy": "triple_barrier",
                "label_encoding": {"WIN": 0, "LOSS": 1, "TIMEOUT": 2},
            },
        )
        # get_label_run() requires both meta + parquet to exist.
        pq_path = label_run_parquet_path(self.tmp, run_id)
        os.makedirs(os.path.dirname(pq_path), exist_ok=True)
        with open(pq_path, "wb"):
            pass

        resolved = resolve_tb_class_options(self.tmp, model_name)
        self.assertTrue(resolved["ok"])
        self.assertEqual(resolved["source"], "label_run")
        self.assertEqual(
            resolved["classes"],
            [
                {"class_id": 0, "label": "WIN"},
                {"class_id": 1, "label": "LOSS"},
                {"class_id": 2, "label": "TIMEOUT"},
            ],
        )

    def test_falls_back_to_default_encoding_when_no_metadata(self) -> None:
        resolved = resolve_tb_class_options(self.tmp, "no_such_model")
        self.assertTrue(resolved["ok"])
        self.assertEqual(resolved["source"], "default_triple_barrier_encoding")
        labels = {c["label"] for c in resolved["classes"]}
        self.assertEqual(labels, {"TP", "SL", "TIME"})

    def test_empty_model_name_is_not_ok(self) -> None:
        resolved = resolve_tb_class_options(self.tmp, "")
        self.assertFalse(resolved["ok"])
        self.assertEqual(resolved["classes"], [])


class LabIntegrationTests(unittest.TestCase):
    """End-to-end through the Research Lab Prediction Dataset."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()

    def _seed_lab_with_tb(
        self,
        lab_path: str,
        *,
        tb_classes: list[int | None],
        tb_probs: list[float | None],
        tb_model_name: str | None = "TBM_test",
    ) -> None:
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
                "tb_model_name": tb_model_name,
                "tb_pred_class": tb_classes[i],
                "tb_pred_probability": tb_probs[i],
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

    def _make_strategy(self) -> str:
        cfg = get_default_template()
        cfg["entry"]["premium_min"] = 10
        cfg["entry"]["premium_max"] = 30
        cfg["target"]["target_profit_pct"] = 8.0
        strat = create_strategy(self.tmp, display_name="TB Filter Strat", config=cfg)
        return strat["champion_version"]["version_id"]

    def test_discover_tb_model_name_reads_persisted_column(self) -> None:
        lab_path = os.path.join(self.tmp, "lab_disc.db")
        self._seed_lab_with_tb(
            lab_path, tb_classes=[0, 0, 1, 2], tb_probs=[0.85, 0.55, 0.9, 0.4]
        )
        disc = discover_tb_model_name(lab_path)
        self.assertEqual(disc["tb_model_name"], "TBM_test")
        self.assertEqual(disc["distinct_count"], 1)

    def test_tb_filter_options_combines_discovery_and_metadata(self) -> None:
        lab_path = os.path.join(self.tmp, "lab_opts.db")
        self._seed_lab_with_tb(
            lab_path, tb_classes=[0, 0, 1, 2], tb_probs=[0.85, 0.55, 0.9, 0.4]
        )
        opts = tb_filter_options(self.tmp, lab_path)
        self.assertTrue(opts["available"])
        self.assertEqual(opts["tb_model_name"], "TBM_test")
        # No metadata on disk for TBM_test -> default encoding, still surfaced.
        self.assertEqual(opts["source"], "default_triple_barrier_encoding")
        labels = {c["label"] for c in opts["classes"]}
        self.assertEqual(labels, {"TP", "SL", "TIME"})

    def test_tb_disabled_matches_baseline_exactly(self) -> None:
        """Acceptance: TB disabled -> identical results to a run with no TB args."""
        lab_path = os.path.join(self.tmp, "lab_disabled.db")
        self._seed_lab_with_tb(
            lab_path, tb_classes=[0, 0, 1, None], tb_probs=[0.85, 0.55, 0.9, None]
        )
        version_id = self._make_strategy()

        plain = run_strategy_simulation_from_lab(
            self.tmp, lab_db_path=lab_path, strategy_version_id=version_id,
        )
        with_tb_off = run_strategy_simulation_from_lab(
            self.tmp,
            lab_db_path=lab_path,
            strategy_version_id=version_id,
            tb_filter_enabled=False,
            tb_class_id=0,
            tb_threshold=0.9,
        )
        m1 = (plain["run"] or {}).get("metrics") or {}
        m2 = (with_tb_off["run"] or {}).get("metrics") or {}
        self.assertEqual(m1.get("trade_count"), m2.get("trade_count"))
        self.assertEqual(m1.get("net_profit"), m2.get("net_profit"))
        self.assertFalse(m2.get("tb_filter_active"))
        self.assertEqual(m2.get("tb_kept"), m2.get("tb_summary", {}).get("candidate_rows"))

    def test_tb_enabled_excludes_null_and_records_summary(self) -> None:
        lab_path = os.path.join(self.tmp, "lab_enabled.db")
        # a,b match class 0; b below threshold; c is class 1; d is NULL.
        self._seed_lab_with_tb(
            lab_path, tb_classes=[0, 0, 1, None], tb_probs=[0.85, 0.55, 0.9, None]
        )
        version_id = self._make_strategy()

        detail = run_strategy_simulation_from_lab(
            self.tmp,
            lab_db_path=lab_path,
            strategy_version_id=version_id,
            tb_filter_enabled=True,
            tb_class_id=0,
            tb_threshold=0.6,
            tb_class_label="TP",
            tb_model_name="TBM_test",
            tb_class_labels={0: "TP", 1: "SL", 2: "TIME"},
        )
        m = (detail["run"] or {}).get("metrics") or {}
        self.assertTrue(m.get("tb_filter_active"))
        self.assertEqual(m.get("tb_filter_label"), "TP")
        # Only row "a" (class 0, prob 0.85 >= 0.60) survives out of 4 candidates.
        self.assertEqual(m.get("tb_kept"), 1)
        self.assertEqual(m.get("tb_removed"), 3)
        summary = m.get("tb_summary") or {}
        self.assertEqual(summary["skipped_missing_count"], 1)
        self.assertEqual(summary["skipped_missing_reason"], "Missing Triple Barrier prediction")
        self.assertEqual(summary["trades_filtered"], 3)
        self.assertEqual(summary["class_counts"], {"TP": 2, "SL": 1})
        self.assertEqual(summary["avg_tb_probability"], 0.85)

    def test_comparison_payload_structure_when_enabled(self) -> None:
        lab_path = os.path.join(self.tmp, "lab_cmp.db")
        self._seed_lab_with_tb(
            lab_path, tb_classes=[0, 0, 1, None], tb_probs=[0.85, 0.55, 0.9, None]
        )
        version_id = self._make_strategy()

        result = run_strategy_simulation_from_lab_with_tb_comparison(
            self.tmp,
            lab_db_path=lab_path,
            strategy_version_id=version_id,
            tb_filter_enabled=True,
            tb_class_id=0,
            tb_threshold=0.6,
            tb_class_label="TP",
            tb_model_name="TBM_test",
            tb_class_labels={0: "TP", 1: "SL", 2: "TIME"},
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "comparison")
        self.assertIsNotNone(result["baseline_run"])
        self.assertIsNotNone(result["filtered_run"])

        comparison = result["comparison"]
        for side in ("baseline", "filtered"):
            self.assertIn(side, comparison)
            snap = comparison[side]
            for key in (
                "total_trades",
                "win_rate_pct",
                "net_profit",
                "average_profit",
                "average_drawdown",
                "max_drawdown",
                "sharpe",
                "expectancy",
                "profit_factor",
            ):
                self.assertIn(key, snap)
        self.assertIn("delta", comparison)
        self.assertEqual(
            comparison["metric_order"],
            [
                "total_trades",
                "win_rate_pct",
                "net_profit",
                "average_profit",
                "average_drawdown",
                "max_drawdown",
                "sharpe",
                "expectancy",
                "profit_factor",
            ],
        )
        # No Precision/Recall anywhere in the Sim comparison payload.
        flat_keys = set(comparison["baseline"]) | set(comparison["filtered"])
        self.assertNotIn("precision", flat_keys)
        self.assertNotIn("recall", flat_keys)

        # Baseline must have TB inactive; filtered must have TB active, and
        # baseline's trade set must be the disabled-TB (unfiltered) run.
        baseline_metrics = (result["baseline_run"] or {}).get("metrics") or {}
        filtered_metrics = (result["filtered_run"] or {}).get("metrics") or {}
        self.assertFalse(baseline_metrics.get("tb_filter_active"))
        self.assertTrue(filtered_metrics.get("tb_filter_active"))

    def test_comparison_payload_none_when_tb_disabled(self) -> None:
        lab_path = os.path.join(self.tmp, "lab_cmp_off.db")
        self._seed_lab_with_tb(
            lab_path, tb_classes=[0, 0, 1, None], tb_probs=[0.85, 0.55, 0.9, None]
        )
        version_id = self._make_strategy()

        result = run_strategy_simulation_from_lab_with_tb_comparison(
            self.tmp,
            lab_db_path=lab_path,
            strategy_version_id=version_id,
            tb_filter_enabled=False,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "single")
        self.assertIsNone(result["comparison"])
        self.assertIsNone(result["baseline_run"])
        self.assertIsNone(result["filtered_run"])


if __name__ == "__main__":
    unittest.main()
