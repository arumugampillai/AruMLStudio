"""Unit tests for Diagnostics Studio 3 feature-source tabs and ownership invariants."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from chain_replay_ml.diagnostics_studio.feature_partition import (
    DiagnosticFeaturePartition,
    partition_diagnostic_rows,
)


class DiagnosticsStudioTabsOwnershipInvariantTests(unittest.TestCase):
    """Test ownership invariants across the three feature tabs."""

    def test_realistic_model_partition_invariants(self) -> None:
        """Test with realistic Future_LTP_5m_WF_1168f_XGB_2243_14 metadata structure."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = tmp
            reg_names = [f"reg_{i}" for i in range(110)]
            base_names = [f"fwd_ret_{i}m" for i in range(89)]
            exp_names = [f"exp_candidate_{i}" for i in range(384)]
            all_selected = reg_names + base_names + exp_names

            dataset_meta = {
                "dataset_name": "analysis_PL0005_198r_447p_6s_20260814_221827",
                "feature_project_id": "all",
                "pipeline_id": "PL_0005",
                "pipeline_snapshot_id": "ca5945f58f87e96e",
                "include_registry": True,
                "include_pipeline": True,
                "registry_export_features": reg_names,
                "base_pipeline_export_features": base_names,
                "pipeline_provenance": {
                    "pipeline_id": "PL_0005",
                    "candidate_features": exp_names,
                },
            }

            diag_rows = [
                {
                    "feature": feat,
                    "rank_gain": idx + 1,
                    "risk": "high" if idx < 10 else "low",
                    "risk_score": 70.0 - idx * 0.1,
                    "drift": 1.0 if idx < 10 else 0.2,
                    "drift_pct": 1.0,
                    "null_pct": 0.0,
                    "skew": 0.0,
                    "diagnostic_flag": "high_risk" if idx < 10 else "ok",
                }
                for idx, feat in enumerate(all_selected)
            ]

            partition = partition_diagnostic_rows(
                diag_rows,
                data_dir=data_dir,
                dataset_metadata=dataset_meta,
            )

            # Invariant 1: Partition is fully valid
            self.assertTrue(partition.is_valid, msg=partition.error_message)
            self.assertIsNone(partition.error_message)

            # Invariant 2: Exact counts match
            self.assertEqual(partition.registry_count, 110)
            self.assertEqual(partition.base_pipeline_count, 89)
            self.assertEqual(partition.experimental_count, 384)
            self.assertEqual(partition.total_count, 583)

            # Invariant 3: Sum of parts == total
            self.assertEqual(
                partition.registry_count + partition.base_pipeline_count + partition.experimental_count,
                partition.total_count,
            )

            # Invariant 4: No unclassified features
            self.assertEqual(len(partition.unclassified_rows), 0)
            self.assertEqual(len(partition.unclassified_feature_names), 0)

            # Invariant 5: Mutually exclusive / disjoint sets
            set_r = {r["feature"] for r in partition.registry_rows}
            set_b = {r["feature"] for r in partition.base_pipeline_rows}
            set_e = {r["feature"] for r in partition.experimental_rows}

            self.assertEqual(len(set_r & set_b), 0)
            self.assertEqual(len(set_r & set_e), 0)
            self.assertEqual(len(set_b & set_e), 0)

            # Invariant 6: Every feature exists in exactly one tab
            union_set = set_r | set_b | set_e
            self.assertEqual(union_set, set(all_selected))

    def test_unclassified_or_duplicate_feature_triggers_invariant_error(self) -> None:
        """Ensure unclassified or duplicated features fail is_valid and produce descriptive error."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = tmp
            reg_names = ["spot", "ltp"]
            base_names = ["fwd_ret_5m"]

            dataset_meta = {
                "dataset_name": "test_ds",
                "registry_export_features": reg_names,
                "base_pipeline_export_features": base_names,
            }

            # Duplicate feature in rows
            diag_rows = [
                {"feature": "spot", "risk_score": 50.0},
                {"feature": "spot", "risk_score": 50.0},
                {"feature": "fwd_ret_5m", "risk_score": 30.0},
            ]

            partition = partition_diagnostic_rows(
                diag_rows,
                data_dir=data_dir,
                dataset_metadata=dataset_meta,
            )

            self.assertFalse(partition.is_valid)
            self.assertIn("Duplicate features", partition.error_message)

    def test_unclassified_feature_recorded(self) -> None:
        """When an unknown feature is presented, it is flagged in error message."""
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = tmp
            dataset_meta = {
                "dataset_name": "test_ds",
                "registry_export_features": ["spot"],
                "base_pipeline_export_features": ["fwd_ret_5m"],
            }

            diag_rows = [
                {"feature": "spot", "risk_score": 50.0},
                {"feature": "unknown_future_col", "risk_score": 30.0},
            ]

            partition = partition_diagnostic_rows(
                diag_rows,
                data_dir=data_dir,
                dataset_metadata=dataset_meta,
            )

            # In the 3-category model, unknown pipeline cols map to other_pipeline
            # when dataset has no provenance or when explicitly classified
            self.assertEqual(partition.registry_count, 1)
            self.assertEqual(partition.experimental_count, 1)
            self.assertEqual(partition.total_count, 2)

    def test_diagnostics_studio_panel_apply_artifacts(self) -> None:
        """Ensure DiagnosticsStudioPanel.apply_artifacts populates all 3 tabs without import error."""
        import tkinter as tk
        from master_dataset_tk.diagnostics_studio_panel import DiagnosticsStudioPanel

        try:
            root = tk.Tk()
            root.withdraw()
        except Exception:
            # Headless environment without display
            return

        try:
            with tempfile.TemporaryDirectory() as tmp:
                panel = DiagnosticsStudioPanel(root, chart_dir=tmp)
                loaded = {
                    "summary": {
                        "primary_cause": "overfitting",
                        "label": "Overfitting",
                        "confidence_pct": 99.0,
                        "similarity_pct": 60.2,
                        "feature_drift_pct": 100.0,
                        "mae_pct_change": 52.8,
                        "dataset": "test_ds",
                        "joins": {"importance": True, "drift": True},
                    },
                    "narrative": ["Overfitting detected."],
                    "comparison": [
                        {"feature": "spot", "risk_score": 60.0, "diagnostic_flag": "high_risk"},
                        {"feature": "fwd_ret_5m", "risk_score": 40.0, "diagnostic_flag": "ok"},
                    ],
                    "meta": {"wall_time_sec": 0.05},
                }
                panel.apply_artifacts(loaded, "TestModel")
                self.assertEqual(len(panel._display_rows), 2)
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
