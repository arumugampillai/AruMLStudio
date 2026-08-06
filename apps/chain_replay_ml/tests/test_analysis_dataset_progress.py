"""Analysis dataset progress helper tests (Phase 1A)."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.analysis_dataset_export import (
    STAGE_FINALIZE,
    STAGE_NO_NULL,
    STAGE_PIPELINE,
    STAGE_PREMIUM,
    STAGE_REGISTRY,
    _stage_states,
)


class AnalysisDatasetProgressTests(unittest.TestCase):
    def test_stage_order_both_sources(self) -> None:
        stages = _stage_states(
            STAGE_PIPELINE,
            include_registry=True,
            include_pipeline=True,
        )
        ids = [s["id"] for s in stages]
        self.assertEqual(ids, [STAGE_REGISTRY, STAGE_PIPELINE, STAGE_FINALIZE])
        self.assertEqual(stages[0]["status"], "done")
        self.assertEqual(stages[1]["status"], "running")
        self.assertEqual(stages[2]["status"], "pending")
        self.assertEqual(stages[0]["label"], "Registry Features")
        self.assertEqual(stages[1]["label"], "Pipeline Features")

    def test_stage_order_with_no_null(self) -> None:
        stages = _stage_states(
            STAGE_NO_NULL,
            include_registry=True,
            include_pipeline=True,
            no_null_data=True,
        )
        ids = [s["id"] for s in stages]
        self.assertEqual(
            ids,
            [STAGE_REGISTRY, STAGE_PIPELINE, STAGE_NO_NULL, STAGE_FINALIZE],
        )
        self.assertEqual(stages[2]["status"], "running")
        self.assertEqual(stages[2]["label"], "No-Null Filter")
        self.assertEqual(stages[3]["status"], "pending")

    def test_stage_order_no_null_then_premium(self) -> None:
        stages = _stage_states(
            STAGE_PREMIUM,
            include_registry=True,
            include_pipeline=True,
            no_null_data=True,
            premium_enabled=True,
        )
        ids = [s["id"] for s in stages]
        self.assertEqual(
            ids,
            [
                STAGE_REGISTRY,
                STAGE_PIPELINE,
                STAGE_NO_NULL,
                STAGE_PREMIUM,
                STAGE_FINALIZE,
            ],
        )
        self.assertEqual(stages[2]["status"], "done")
        self.assertEqual(stages[3]["status"], "running")
        self.assertEqual(stages[3]["label"], "Premium Filter")

    def test_stage_registry_only(self) -> None:
        stages = _stage_states(
            STAGE_FINALIZE,
            include_registry=True,
            include_pipeline=False,
        )
        ids = [s["id"] for s in stages]
        self.assertEqual(ids, [STAGE_REGISTRY, STAGE_FINALIZE])
        self.assertEqual(stages[0]["status"], "done")
        self.assertEqual(stages[1]["status"], "running")


if __name__ == "__main__":
    unittest.main()
