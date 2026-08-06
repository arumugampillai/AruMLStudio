"""Unit tests for Non-Null filter Step 1 → Step 2 pipeline."""

from __future__ import annotations

import logging
import sqlite3
import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.non_null_filter import (
    apply_non_null_filter,
    apply_non_null_filter_frame,
    discover_kept_columns_step1,
    format_non_null_report,
)


def _conn_with_rows(rows: list[tuple]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE samples ("
        "trading_day TEXT, feature_a INTEGER, feature_b INTEGER, feature_c INTEGER)"
    )
    conn.executemany(
        "INSERT INTO samples (trading_day, feature_a, feature_b, feature_c) "
        "VALUES (?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    return conn


class TestNonNullStepOrder(unittest.TestCase):
    def test_dataframe_filter_runs_after_lag_and_removes_warmup_rows(self) -> None:
        """Regression: pre-transform filtering cannot see Lag's leading NULL."""
        transformed = pd.DataFrame({
            "trading_day": ["2026-07-01"] * 3,
            "token": ["A"] * 3,
            "ltp": [100.0, 101.0, 102.0],
            "ltp_lag_3s": [None, 100.0, 101.0],
        })
        result = apply_non_null_filter_frame(transformed)
        self.assertEqual(result["report"]["rows_before"], 3)
        self.assertEqual(result["report"]["rows_after"], 2)
        self.assertEqual(result["report"]["remaining_null_cells"], 0)
        self.assertEqual(result["frame"]["ltp_lag_3s"].tolist(), [100.0, 101.0])
        self.assertEqual(result["report"]["stage"], "post_transformation")

    def test_pipeline_step2_ignores_registry_nulls(self) -> None:
        """Analysis Pipeline stage: Registry NULLs must not drive a second drop."""
        transformed = pd.DataFrame({
            "trading_day": ["2026-07-01"] * 3,
            "token": ["A"] * 3,
            "option_low": [10.0, None, 12.0],  # registry incomplete on row 1
            "ltp_lag_3s": [None, 100.0, 101.0],  # pipeline warmup on row 0
        })
        result = apply_non_null_filter_frame(
            transformed,
            step2_columns=["ltp_lag_3s"],
        )
        # Row 1 kept despite option_low NULL — Registry already checked earlier.
        self.assertEqual(result["report"]["rows_after"], 2)
        self.assertEqual(result["report"]["step2_scope"], "pipeline_only")
        self.assertEqual(result["report"]["stage"], "pipeline_post_transformation")
        self.assertEqual(result["frame"]["ltp_lag_3s"].tolist(), [100.0, 101.0])
        self.assertTrue(pd.isna(result["frame"]["option_low"].iloc[0]))

    def test_dataframe_multi_day_empty_column_rule_matches_sql(self) -> None:
        transformed = pd.DataFrame({
            "trading_day": ["A", "A", "B", "B"],
            "feature_a": [None, None, 1.0, 2.0],
            "feature_b": [1.0, 2.0, 3.0, 4.0],
        })
        result = apply_non_null_filter_frame(transformed)
        self.assertIn("feature_a", result["dropped_columns"])
        self.assertEqual(result["report"]["rows_after"], 4)
        self.assertEqual(result["frame"].columns.tolist(), ["trading_day", "feature_b"])

    def test_step1_drops_100pct_null_column_before_step2(self) -> None:
        conn = _conn_with_rows(
            [
                ("2026-01-02", None, 1, None),
                ("2026-01-02", None, 2, 5),
                ("2026-01-02", None, 3, 6),
                ("2026-01-02", None, 4, 7),
            ]
        )
        cols = ["trading_day", "feature_a", "feature_b", "feature_c"]
        kept, dropped = discover_kept_columns_step1(conn, cols, "1=1", [])
        self.assertIn("feature_a", dropped)
        self.assertIn("feature_b", kept)
        self.assertIn("feature_c", kept)

        result = apply_non_null_filter(conn, cols, "1=1", [], log=False, debug=True)
        self.assertIn("feature_a", result["dropped_columns"])
        self.assertEqual(result["report"]["rows_after"], 3)
        self.assertEqual(result["report"]["remaining_null_cells"], 0)
        self.assertTrue(result["report"]["ok"])
        conn.close()

    def test_multi_day_does_not_collapse_to_zero(self) -> None:
        """
        Regression: union Step 1 re-activates columns empty on other days → 0 rows.

        Day A: feature_a empty, feature_b/c complete on some rows
        Day B: feature_a filled, feature_c empty on all → would resurrect feature_a
               under union discovery and wipe Day A.
        """
        conn = _conn_with_rows(
            [
                # Day A — feature_a 100% null; complete rows on b,c
                ("2026-07-14", None, 1, 10),
                ("2026-07-14", None, 2, 20),
                ("2026-07-14", None, None, 30),  # incomplete
                # Day B — feature_a filled; feature_c 100% null
                ("2026-07-13", 9, 3, None),
                ("2026-07-13", 8, 4, None),
            ]
        )
        cols = ["trading_day", "feature_a", "feature_b", "feature_c"]

        alone_a = apply_non_null_filter(
            conn, cols, "trading_day = ?", ["2026-07-14"], log=False, debug=True
        )
        alone_b = apply_non_null_filter(
            conn, cols, "trading_day = ?", ["2026-07-13"], log=False, debug=True
        )
        self.assertGreater(alone_a["report"]["rows_after"], 0)
        self.assertGreater(alone_b["report"]["rows_after"], 0)

        multi = apply_non_null_filter(
            conn,
            cols,
            "trading_day IN (?, ?)",
            ["2026-07-14", "2026-07-13"],
            log=False,
            debug=True,
        )
        # Must not be 0 when both single-day selections produce rows
        self.assertGreater(multi["report"]["rows_after"], 0)
        # feature_a empty on day A, feature_c empty on day B → both dropped
        self.assertIn("feature_a", multi["dropped_columns"])
        self.assertIn("feature_c", multi["dropped_columns"])
        expected = alone_a["report"]["rows_after"] + alone_b["report"]["rows_after"]
        # Intersection may keep slightly more rows than sum of alone (fewer constraints)
        # but must stay additive in spirit — never collapse below either day alone.
        self.assertGreaterEqual(
            multi["report"]["rows_after"],
            max(alone_a["report"]["rows_after"], alone_b["report"]["rows_after"]),
        )
        self.assertEqual(multi["report"]["remaining_null_cells"], 0)
        # Approx additive: within a small tolerance of sum (may be >= sum)
        self.assertGreaterEqual(multi["report"]["rows_after"], expected)
        conn.close()

    def test_operates_on_currently_filtered_dataset(self) -> None:
        conn = _conn_with_rows(
            [
                ("2026-01-02", None, 1, 1),
                ("2026-01-02", None, 2, 2),
                ("2026-01-03", 9, 3, 3),
                ("2026-01-03", 8, None, 4),
            ]
        )
        cols = ["trading_day", "feature_a", "feature_b", "feature_c"]
        where = "trading_day = ?"
        params: list = ["2026-01-02"]
        kept, dropped = discover_kept_columns_step1(conn, cols, where, params)
        self.assertIn("feature_a", dropped)

        result = apply_non_null_filter(conn, cols, where, params, log=False, debug=True)
        self.assertEqual(result["report"]["rows_before"], 2)
        self.assertEqual(result["report"]["rows_after"], 2)
        self.assertEqual(result["report"]["remaining_null_cells"], 0)
        conn.close()

    def test_remaining_null_cells_zero_and_report_format(self) -> None:
        conn = _conn_with_rows(
            [
                ("2026-01-02", None, 1, 2),
                ("2026-01-02", None, 3, 4),
            ]
        )
        result = apply_non_null_filter(
            conn,
            ["trading_day", "feature_a", "feature_b", "feature_c"],
            "1=1",
            [],
            log=False,
            debug=True,
        )
        report = result["report"]
        self.assertEqual(report["remaining_null_cells"], 0)
        self.assertEqual(report["empty_columns_removed"], 1)
        text = format_non_null_report(report)
        self.assertIn("Non-Null Filter", text)
        self.assertIn("Removed empty columns: 1", text)
        self.assertIn("Remaining NULL cells (mandatory): 0", text)
        conn.close()

    def test_format_report_fast_export_omits_fake_zeros(self) -> None:
        text = format_non_null_report(
            {
                "debug": False,
                "columns_before": 10,
                "columns_after": 8,
                "empty_columns_removed": 2,
                "columns_removed": ["a", "b"],
                "rows_before": 0,
                "rows_after": 0,
            }
        )
        self.assertIn("fast export mode", text)
        self.assertIn("Kept columns: 8", text)
        self.assertNotIn("Final rows: 0", text)
        self.assertNotIn("Rows before: 0", text)

    def test_logs_pipeline_summary(self) -> None:
        conn = _conn_with_rows(
            [
                ("2026-01-02", None, 1, None),
                ("2026-01-02", None, 2, 5),
            ]
        )
        with self.assertLogs(
            "chain_replay_ml.dataset_builder.non_null_filter", level=logging.INFO
        ) as cm:
            apply_non_null_filter(
                conn,
                ["trading_day", "feature_a", "feature_b", "feature_c"],
                "1=1",
                [],
                log=True,
                debug=True,
            )
        joined = "\n".join(cm.output)
        self.assertIn("Remove all-null columns", joined)
        self.assertIn("Remaining NULL cells (mandatory):", joined)
        conn.close()


class TestNullableFeatureListStep2(unittest.TestCase):
    def test_nullable_list_allows_nulls_without_dropping_rows(self) -> None:
        """gamma_flip_* may be NULL; mandatory columns still enforce completeness."""
        frame = pd.DataFrame({
            "trading_day": ["2026-07-24"] * 4,
            "ltp": [10.0, 11.0, 12.0, 13.0],
            "gamma_flip_spot": [None, 24000.0, None, 24100.0],
            "gamma_flip_distance": [None, 0.01, None, -0.02],
        })
        result = apply_non_null_filter_frame(frame)
        self.assertEqual(result["report"]["rows_after"], 4)
        self.assertEqual(
            set(result["report"]["nullable_features_ignored"]),
            {"gamma_flip_spot", "gamma_flip_distance"},
        )
        self.assertEqual(result["report"]["remaining_null_cells"], 0)
        self.assertTrue(result["report"]["ok"])
        # Columns kept (not 100% null) including nullable ones
        self.assertIn("gamma_flip_spot", result["kept_columns"])

    def test_nullable_does_not_rescue_mandatory_nulls(self) -> None:
        frame = pd.DataFrame({
            "trading_day": ["2026-07-24"] * 3,
            "ltp": [10.0, None, 12.0],
            "gamma_flip_spot": [None, None, 24000.0],
            "gamma_flip_distance": [None, None, 0.01],
        })
        result = apply_non_null_filter_frame(frame)
        self.assertEqual(result["report"]["rows_after"], 2)
        self.assertEqual(result["frame"]["ltp"].tolist(), [10.0, 12.0])

    def test_sql_path_ignores_nullable_list(self) -> None:
        conn = sqlite3.connect(":memory:")
        conn.execute(
            "CREATE TABLE samples ("
            "trading_day TEXT, ltp REAL, gamma_flip_spot REAL, gamma_flip_distance REAL)"
        )
        conn.executemany(
            "INSERT INTO samples VALUES (?, ?, ?, ?)",
            [
                ("2026-07-24", 10.0, None, None),
                ("2026-07-24", 11.0, 24000.0, 0.01),
                ("2026-07-24", None, None, None),  # mandatory ltp NULL → drop
            ],
        )
        conn.commit()
        result = apply_non_null_filter(
            conn,
            ["trading_day", "ltp", "gamma_flip_spot", "gamma_flip_distance"],
            "1=1",
            [],
            log=False,
            debug=True,
        )
        self.assertEqual(result["report"]["rows_after"], 2)
        self.assertIn("gamma_flip_spot", result["report"]["nullable_features_ignored"])
        self.assertTrue(result["report"]["ok"])
        conn.close()

    def test_nullable_list_includes_iv_and_higher_greeks(self) -> None:
        from chain_replay_ml.dataset_builder.nullable_features import (
            NULLABLE_FEATURE_LIST,
        )

        for name in ("current_iv", "vega", "vanna", "charm", "speed"):
            self.assertIn(name, NULLABLE_FEATURE_LIST)


if __name__ == "__main__":
    unittest.main()
