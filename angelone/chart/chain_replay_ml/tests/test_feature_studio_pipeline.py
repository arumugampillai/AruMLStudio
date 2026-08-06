"""Unit tests for Feature Studio unified load/compute pipeline."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from master_dataset_tk.feature_studio_pipeline import (
    PIPELINE_ORDER,
    PLANNER_SKIP_MSG,
    PipelineResult,
    StudioStageResult,
    planner_inputs_available,
    run_compute_pipeline,
    run_load_pipeline,
)


class PlannerInputsTests(unittest.TestCase):
    def test_needs_at_least_one_of_imp_dist_drift(self) -> None:
        self.assertFalse(planner_inputs_available({}))
        self.assertFalse(
            planner_inputs_available(
                {"importance": False, "distribution": False, "drift": False}
            )
        )
        self.assertTrue(planner_inputs_available({"importance": True}))
        self.assertTrue(planner_inputs_available({"drift": True}))
        self.assertFalse(planner_inputs_available({"diagnostics": True}))


class PipelineOrderTests(unittest.TestCase):
    def test_planner_is_last(self) -> None:
        self.assertEqual(PIPELINE_ORDER[-1], "planner")
        self.assertEqual(
            PIPELINE_ORDER,
            ("importance", "distribution", "drift", "diagnostics", "planner"),
        )


class LoadPipelineTests(unittest.TestCase):
    def test_missing_artifacts_continue_without_abort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = tmp
            pkg = os.path.join(data_dir, "models", "Demo")
            os.makedirs(pkg, exist_ok=True)
            # Only importance present
            imp_dir = os.path.join(pkg, "feature_importance_studio")
            os.makedirs(imp_dir)
            with open(os.path.join(imp_dir, "comparison.json"), "w", encoding="utf-8") as fh:
                json.dump({"rows": [{"feature": "a", "gain": 1.0}]}, fh)

            result = run_load_pipeline(data_dir=data_dir, model_name="Demo")
            self.assertEqual(result.model_name, "Demo")
            self.assertTrue(result.stages["importance"].available)
            self.assertFalse(result.stages["distribution"].available)
            self.assertFalse(result.stages["drift"].available)
            self.assertFalse(result.stages["diagnostics"].available)
            self.assertFalse(result.stages["planner"].available)
            # All stages recorded — load did not abort
            self.assertEqual(set(result.stages), set(PIPELINE_ORDER))
            marks = result.status_marks()
            self.assertIn("Importance ✓", marks)
            self.assertIn("Distribution ✗", marks)

    def test_status_marks_for_skipped_planner(self) -> None:
        result = PipelineResult(model_name="X")
        result.stages["importance"] = StudioStageResult(
            key="importance", available=True, payload={}
        )
        result.stages["planner"] = StudioStageResult(
            key="planner", skipped=True, skip_reason=PLANNER_SKIP_MSG
        )
        marks = result.status_marks()
        self.assertIn("Importance ✓", marks)
        self.assertIn("Experiment Planner skipped", marks)


class ComputePipelineTests(unittest.TestCase):
    def test_planner_skipped_when_no_soft_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = tmp
            pkg = os.path.join(data_dir, "models", "Empty")
            os.makedirs(pkg, exist_ok=True)

            def boom(*_a, **_k):
                return False, "should not run for empty"

            with mock.patch.dict(
                "master_dataset_tk.feature_studio_pipeline._COMPUTERS",
                {
                    "importance": lambda *_a, **_k: (False, "no model"),
                    "distribution": lambda *_a, **_k: (False, "no model"),
                    "drift": lambda *_a, **_k: (False, "no model"),
                    "diagnostics": lambda *_a, **_k: (False, "no model"),
                    "planner": boom,
                },
            ):
                result = run_compute_pipeline(data_dir=data_dir, model_name="Empty")

            self.assertFalse(result.stages["importance"].available)
            self.assertTrue(result.stages["planner"].skipped)
            self.assertEqual(result.stages["planner"].skip_reason, PLANNER_SKIP_MSG)

    def test_compute_continues_after_one_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = tmp
            pkg = os.path.join(data_dir, "models", "Partial")
            os.makedirs(pkg, exist_ok=True)
            calls: list[str] = []

            def make_ok(key: str):
                def _fn(*_a, **_k):
                    calls.append(key)
                    return True, None

                return _fn

            def fail_imp(*_a, **_k):
                calls.append("importance")
                return False, "imp failed"

            with mock.patch.dict(
                "master_dataset_tk.feature_studio_pipeline._COMPUTERS",
                {
                    "importance": fail_imp,
                    "distribution": make_ok("distribution"),
                    "drift": make_ok("drift"),
                    "diagnostics": make_ok("diagnostics"),
                    "planner": make_ok("planner"),
                },
            ):
                result = run_compute_pipeline(data_dir=data_dir, model_name="Partial")

            self.assertEqual(
                calls,
                ["importance", "distribution", "drift", "diagnostics", "planner"],
            )
            self.assertFalse(result.stages["importance"].available)
            self.assertEqual(result.stages["importance"].error, "imp failed")
            self.assertTrue(result.stages["distribution"].available)
            self.assertTrue(result.stages["planner"].available)


if __name__ == "__main__":
    unittest.main()
