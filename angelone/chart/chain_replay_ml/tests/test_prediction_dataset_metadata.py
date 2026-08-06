"""Unit tests for the Prediction Dataset Metadata tab's backend (Phase 1.5/1.6).

Exercises ``prediction_dataset_metadata.compute_prediction_dataset_metadata``
against a tiny temp SQLite ``prediction_dataset`` with a deliberate mix of
NULL/populated columns across every stage (Regression, Probability Ladder,
Triple Barrier, Compute Outcomes, Confidence, Identity/Other) so the summary
counts, column coverage (incl. stage assignment), stage status classification,
expected/ready counts, and "why isn't this 100%" notes are all covered.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from chain_replay_ml.model_lab.prediction_dataset_metadata import (
    STATUS_NONE,
    STATUS_NOT_BUILT,
    STATUS_OK,
    STATUS_PARTIAL,
    compute_prediction_dataset_metadata,
    prediction_dataset_metadata_cache_path,
    read_cached_prediction_dataset_metadata,
    refresh_prediction_dataset_metadata,
)
from chain_replay_ml.model_lab.prediction_metadata_stages import (
    CONFIDENCE_PRED_COLUMNS,
    STAGE_REGISTRY,
)
from chain_replay_ml.model_lab.prediction_schema import (
    create_prediction_dataset_sql,
    create_prediction_day_metadata_sql,
)
from chain_replay_ml.training.prediction_packages import (
    PROBABILITY_LADDER,
    PROBABILITY_OUTPUT_COLUMNS,
)


def _make_schema(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(create_prediction_dataset_sql())
        conn.executescript(create_prediction_day_metadata_sql())
        conn.commit()
    finally:
        conn.close()


def _insert_row(conn: sqlite3.Connection, values: dict) -> None:
    cols = list(values.keys())
    placeholders = ", ".join("?" for _ in cols)
    col_sql = ", ".join(f'"{c}"' for c in cols)
    conn.execute(
        f'INSERT INTO prediction_dataset ({col_sql}) VALUES ({placeholders})',
        [values[c] for c in cols],
    )


def _seed_mixed_dataset(db_path: str) -> None:
    """10 rows across 2 trading days with a deliberate NULL/populated mix.

    - Regression (predicted_future_ltp, predicted_trend): populated in 8/10 → partial (~80%)
    - Probability Ladder: only pred_prob_up_2pct_5m populated (10/10); other
      5 ladder columns all NULL → partial (~16.67% average)
    - Triple Barrier (tb_pred_probability, tb_pred_class): all NULL → none (0%)
    - Compute Outcomes: every path-outcome column populated in all 10 rows → ok (100%)
    - Confidence (confidence_target_hit_pred): populated 10/10; other
      confidence_*_pred columns all NULL → partial (small average)
    """
    _make_schema(db_path)
    conn = sqlite3.connect(db_path)
    try:
        for i in range(10):
            day = "2026-01-01" if i < 5 else "2026-01-02"
            row = {
                "lab_uuid": "lab-1",
                "prediction_id": f"pred-{i}",
                "trading_day": day,
                "timestamp": float(1_700_000_000 + i),
                "token": "TOKEN",
                "predicted_future_ltp": 100.0 + i if i < 8 else None,
                "predicted_trend": "Up" if i < 8 else None,
                "pred_prob_up_2pct_5m": 0.5 + i * 0.01,
                "maximum_profit": 10.0,
                "maximum_drawdown": 5.0,
                "dd_before_target": 1.0,
                "time_to_max_profit": 30.0,
                "time_to_max_drawdown": 20.0,
                "time_to_dd_before_target": 10.0,
                "time_to_target": 40.0,
                "target_reached": 1,
                "target_reached_at": 1_700_000_040.0,
                "max_profit_at": 1_700_000_030.0,
                "max_drawdown_at": 1_700_000_020.0,
                "exit_at": 1_700_000_050.0,
                "rr_1_1_hit": 1,
                "rr_2_3_hit": 1,
                "rr_1_2_hit": 0,
                "rr_1_3_hit": 0,
                "rr_1_4_hit": 0,
                "confidence_target_hit_pred": 1 if i % 2 == 0 else 0,
            }
            _insert_row(conn, row)
        conn.execute(
            """
            INSERT INTO prediction_day_metadata
                (lab_uuid, trading_day, status, row_count)
            VALUES
                ('lab-1', '2026-01-01', 'completed', 5),
                ('lab-1', '2026-01-02', 'waiting', 5)
            """
        )
        conn.commit()
    finally:
        conn.close()


class ComputePredictionDatasetMetadataTests(unittest.TestCase):
    def test_missing_db_file_returns_empty_zeroed_metadata(self) -> None:
        meta = compute_prediction_dataset_metadata("/no/such/lab.db")
        self.assertEqual(meta["row_count"], 0)
        self.assertEqual(meta["column_count"], 0)
        self.assertEqual(meta["trading_days"], 0)
        self.assertEqual(meta["total_catalog_days"], 0)
        self.assertEqual(len(meta["stages"]), len(STAGE_REGISTRY))
        self.assertTrue(all(s["status"] == STATUS_NOT_BUILT for s in meta["stages"]))
        self.assertIn("Identity/Other", {s["name"] for s in meta["stages"]})

    def test_empty_table_zero_rows_is_not_built_for_every_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "lab.db")
            _make_schema(db_path)
            meta = compute_prediction_dataset_metadata(db_path)
            self.assertEqual(meta["row_count"], 0)
            self.assertGreater(meta["column_count"], 0)
            self.assertEqual(meta["completed_days"], 0)
            self.assertEqual(meta["pending_days"], 0)
            self.assertEqual(len(meta["stages"]), len(STAGE_REGISTRY))
            for stage in meta["stages"]:
                self.assertEqual(stage["status"], STATUS_NOT_BUILT)

    def test_dataset_summary_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "lab.db")
            _seed_mixed_dataset(db_path)
            meta = compute_prediction_dataset_metadata(db_path)
            self.assertEqual(meta["row_count"], 10)
            self.assertEqual(meta["trading_days"], 2)
            self.assertEqual(meta["completed_days"], 1)
            self.assertEqual(meta["pending_days"], 1)
            self.assertEqual(meta["total_catalog_days"], 2)
            self.assertGreater(meta["column_count"], 50)

    def test_column_coverage_populated_null_and_percent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "lab.db")
            _seed_mixed_dataset(db_path)
            meta = compute_prediction_dataset_metadata(db_path)
            by_name = {c["name"]: c for c in meta["columns"]}

            self.assertEqual(len(meta["columns"]), meta["column_count"])

            reg = by_name["predicted_future_ltp"]
            self.assertEqual(reg["populated"], 8)
            self.assertEqual(reg["null"], 2)
            self.assertEqual(reg["coverage_pct"], 80.0)

            ladder_hit = by_name["pred_prob_up_2pct_5m"]
            self.assertEqual(ladder_hit["populated"], 10)
            self.assertEqual(ladder_hit["coverage_pct"], 100.0)

            ladder_miss = by_name["pred_prob_up_3pct_5m"]
            self.assertEqual(ladder_miss["populated"], 0)
            self.assertEqual(ladder_miss["coverage_pct"], 0.0)

            tb = by_name["tb_pred_probability"]
            self.assertEqual(tb["populated"], 0)
            self.assertEqual(tb["coverage_pct"], 0.0)

            outcome = by_name["maximum_profit"]
            self.assertEqual(outcome["populated"], 10)
            self.assertEqual(outcome["coverage_pct"], 100.0)

    def test_column_coverage_stage_assignment(self) -> None:
        """Requirement #1 — every column maps to its owning pipeline stage."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "lab.db")
            _seed_mixed_dataset(db_path)
            meta = compute_prediction_dataset_metadata(db_path)
            by_name = {c["name"]: c for c in meta["columns"]}

            self.assertEqual(by_name["predicted_future_ltp"]["stage"], "Regression")
            self.assertEqual(by_name["predicted_trend"]["stage"], "Regression")
            self.assertEqual(by_name["pred_prob_up_2pct_5m"]["stage"], "Probability Ladder")
            self.assertEqual(by_name["pred_prob_up_6pct_5m"]["stage"], "Probability Ladder")
            self.assertEqual(by_name["tb_pred_probability"]["stage"], "Triple Barrier")
            self.assertEqual(by_name["maximum_profit"]["stage"], "Compute Outcomes")
            self.assertEqual(by_name["rr_1_1_hit"]["stage"], "Compute Outcomes")
            self.assertEqual(
                by_name["confidence_target_hit_pred"]["stage"], "Confidence"
            )
            # Identity/Other catch-all — keys, timestamps, tokens, joins.
            for col in ("trading_day", "timestamp", "token", "master_row_id", "lab_uuid"):
                self.assertEqual(by_name[col]["stage"], "Identity/Other")

    def test_column_coverage_bucket_and_emoji(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "lab.db")
            _seed_mixed_dataset(db_path)
            meta = compute_prediction_dataset_metadata(db_path)
            by_name = {c["name"]: c for c in meta["columns"]}

            self.assertEqual(by_name["maximum_profit"]["coverage_bucket"], "full")
            self.assertEqual(by_name["maximum_profit"]["coverage_emoji"], "\U0001F7E2")
            self.assertEqual(by_name["predicted_future_ltp"]["coverage_bucket"], "high")
            self.assertEqual(by_name["tb_pred_probability"]["coverage_bucket"], "empty")
            self.assertEqual(by_name["tb_pred_probability"]["coverage_emoji"], "\U0001F534")

    def test_stage_status_classification(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "lab.db")
            _seed_mixed_dataset(db_path)
            meta = compute_prediction_dataset_metadata(db_path)
            stages = {s["name"]: s for s in meta["stages"]}

            self.assertEqual(stages["Regression"]["status"], STATUS_PARTIAL)
            self.assertAlmostEqual(stages["Regression"]["coverage_pct"], 80.0, places=1)

            self.assertEqual(stages["Probability Ladder"]["status"], STATUS_PARTIAL)
            self.assertLess(stages["Probability Ladder"]["coverage_pct"], 20.0)
            self.assertGreater(stages["Probability Ladder"]["coverage_pct"], 0.0)

            self.assertEqual(stages["Triple Barrier"]["status"], STATUS_NONE)
            self.assertEqual(stages["Triple Barrier"]["coverage_pct"], 0.0)

            self.assertEqual(stages["Compute Outcomes"]["status"], STATUS_OK)
            self.assertEqual(stages["Compute Outcomes"]["coverage_pct"], 100.0)

            self.assertEqual(stages["Confidence"]["status"], STATUS_PARTIAL)
            self.assertGreater(stages["Confidence"]["coverage_pct"], 0.0)
            self.assertLess(stages["Confidence"]["coverage_pct"], 100.0)

    def test_stage_status_text_is_not_icon_only(self) -> None:
        """Requirement #6 — status shown as readable text, not a bare glyph."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "lab.db")
            _seed_mixed_dataset(db_path)
            meta = compute_prediction_dataset_metadata(db_path)
            stages = {s["name"]: s for s in meta["stages"]}

            self.assertEqual(stages["Compute Outcomes"]["status_label"], "\u2713 Complete")
            self.assertEqual(stages["Triple Barrier"]["status_label"], "\u2717 Empty/Failed")
            self.assertEqual(stages["Regression"]["status_label"], "\u26a0 Partial")

    def test_stage_expected_and_ready_counts(self) -> None:
        """Requirement #7 — Expected/Ready driven by the stage registry."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "lab.db")
            _seed_mixed_dataset(db_path)
            meta = compute_prediction_dataset_metadata(db_path)
            stages = {s["name"]: s for s in meta["stages"]}

            ladder = stages["Probability Ladder"]
            self.assertEqual(ladder["expected"], len(PROBABILITY_OUTPUT_COLUMNS))
            self.assertEqual(ladder["ready"], 1)  # only +2% populated

            tb = stages["Triple Barrier"]
            self.assertEqual(tb["expected"], 2)
            self.assertEqual(tb["ready"], 0)

            outcomes = stages["Compute Outcomes"]
            self.assertEqual(outcomes["ready"], outcomes["expected"])

            confidence = stages["Confidence"]
            self.assertEqual(confidence["expected"], len(CONFIDENCE_PRED_COLUMNS))
            self.assertEqual(confidence["ready"], 1)  # only confidence_target_hit_pred

    def test_probability_ladder_notes_infer_missing_members_from_columns(self) -> None:
        """Requirement #4 — explain WHY coverage isn't 100% for the ladder."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "lab.db")
            _seed_mixed_dataset(db_path)
            meta = compute_prediction_dataset_metadata(db_path)
            stages = {s["name"]: s for s in meta["stages"]}
            notes = stages["Probability Ladder"]["notes"]
            self.assertIn("Missing ladder models", notes)
            for label in ("+3%", "+4%", "+5%", "+6%", ">6%"):
                self.assertIn(label, notes)
            self.assertNotIn("+2%", notes)  # +2% is the one populated member

    def test_probability_ladder_notes_prefer_explicit_package_members(self) -> None:
        """Optional lab context overrides the DB-only inference when supplied."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "lab.db")
            _seed_mixed_dataset(db_path)
            # Deliberately contradicts column population (only +2% has data) to
            # prove the explicit package context wins over inference.
            package_members = [
                {**dict(item), "available": str(item["key"]) != "up_2pct"}
                for item in PROBABILITY_LADDER
            ]
            meta = compute_prediction_dataset_metadata(
                db_path, package_members=package_members
            )
            stages = {s["name"]: s for s in meta["stages"]}
            notes = stages["Probability Ladder"]["notes"]
            self.assertIn("Missing ladder models", notes)
            self.assertIn("+2%", notes)
            self.assertNotIn("+3%", notes)

    def test_triple_barrier_and_confidence_notes_when_partial_or_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "lab.db")
            _seed_mixed_dataset(db_path)
            meta = compute_prediction_dataset_metadata(db_path)
            stages = {s["name"]: s for s in meta["stages"]}

            tb_notes = stages["Triple Barrier"]["notes"]
            self.assertIn("not scored", tb_notes.lower())

            confidence_notes = stages["Confidence"]["notes"]
            self.assertTrue(confidence_notes)
            self.assertIn("Not scored yet", confidence_notes)

    def test_ok_and_not_built_stages_have_no_notes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "lab.db")
            _seed_mixed_dataset(db_path)
            meta = compute_prediction_dataset_metadata(db_path)
            stages = {s["name"]: s for s in meta["stages"]}
            self.assertEqual(stages["Compute Outcomes"]["notes"], "")

            empty_meta = compute_prediction_dataset_metadata("/no/such/lab.db")
            for stage in empty_meta["stages"]:
                self.assertEqual(stage["notes"], "")

    def test_registry_drives_stage_order_and_keys(self) -> None:
        """Requirement #10 — UI/compute both walk the same StageSpec registry."""
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "lab.db")
            _seed_mixed_dataset(db_path)
            meta = compute_prediction_dataset_metadata(db_path)
            self.assertEqual(
                [s["key"] for s in meta["stages"]],
                [spec.key for spec in STAGE_REGISTRY],
            )
            self.assertEqual(
                [s["name"] for s in meta["stages"]],
                [spec.label for spec in STAGE_REGISTRY],
            )

    def test_refresh_writes_and_reads_sidecar_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "lab.db")
            _seed_mixed_dataset(db_path)

            cache_path = prediction_dataset_metadata_cache_path(db_path)
            self.assertFalse(os.path.isfile(cache_path))
            self.assertIsNone(read_cached_prediction_dataset_metadata(db_path))

            metadata = refresh_prediction_dataset_metadata(db_path)
            self.assertEqual(metadata["row_count"], 10)
            self.assertTrue(os.path.isfile(cache_path))

            cached = read_cached_prediction_dataset_metadata(db_path)
            self.assertIsNotNone(cached)
            self.assertEqual(cached["row_count"], 10)
            self.assertEqual(cached["trading_days"], 2)


if __name__ == "__main__":
    unittest.main()
