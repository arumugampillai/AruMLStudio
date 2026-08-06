"""Tests for versioned stage artifacts (stage isolation contract)."""

from __future__ import annotations

import tempfile
import unittest

from chain_replay_ml.dataset_builder.analysis_artifacts import (
    KIND_CORRELATION,
    KIND_DISCOVERY_BUNDLE,
    KIND_EXPERIMENT_HYPOTHESIS,
    KIND_HCA_FAMILIES,
    artifact_lineage,
    latest_artifact,
    load_artifact,
    publish_artifact,
    publish_discovery_bundle,
    require_artifact,
)
from chain_replay_ml.dataset_builder.analysis_lab_store import (
    ensure_analysis_run,
    register_dataset,
)
import json
import os

import numpy as np
import pandas as pd


class ArtifactContractTests(unittest.TestCase):
    def test_publish_version_and_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.parquet")
            pd.DataFrame({"x": [1, 2, 3]}).to_parquet(path, index=False)
            with open(path.replace(".parquet", ".json"), "w", encoding="utf-8") as f:
                json.dump({"columns": ["x"], "targets": [], "labels": []}, f)
            ds = register_dataset(tmp, path, name="a")
            run = ensure_analysis_run(tmp, ds["dataset_id"])
            rid = run["run_id"]

            a1 = publish_artifact(
                tmp,
                rid,
                KIND_CORRELATION,
                {"n_pairs": 10},
                label="corr",
            )
            self.assertEqual(len(str(a1.get("fingerprint") or "")), 64)
            a2 = publish_artifact(
                tmp,
                rid,
                KIND_CORRELATION,
                {"n_pairs": 10},
                label="corr",
            )
            self.assertEqual(a1["artifact_id"], a2["artifact_id"])
            self.assertEqual(a1["fingerprint"], a2["fingerprint"])
            self.assertEqual(int(a1["version"]), 1)

            a3 = publish_artifact(
                tmp,
                rid,
                KIND_CORRELATION,
                {"n_pairs": 11},
                label="corr2",
            )
            self.assertNotEqual(a3["artifact_id"], a1["artifact_id"])
            self.assertEqual(int(a3["version"]), 2)

    def test_wrong_parent_kind_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.parquet")
            pd.DataFrame({"x": [1.0]}).to_parquet(path, index=False)
            with open(path.replace(".parquet", ".json"), "w", encoding="utf-8") as f:
                json.dump({"columns": ["x"], "targets": [], "labels": []}, f)
            ds = register_dataset(tmp, path, name="a")
            run = ensure_analysis_run(tmp, ds["dataset_id"])
            rid = run["run_id"]

            corr = publish_artifact(tmp, rid, KIND_CORRELATION, {"ok": True})
            # HCA may consume correlation — OK
            hca = publish_artifact(
                tmp,
                rid,
                KIND_HCA_FAMILIES,
                {"n": 1},
                parent_ids=[corr["artifact_id"]],
            )
            self.assertEqual(hca["kind"], KIND_HCA_FAMILIES)

            # experiment_hypothesis must not take correlation as parent
            with self.assertRaises(ValueError):
                publish_artifact(
                    tmp,
                    rid,
                    KIND_EXPERIMENT_HYPOTHESIS,
                    {"family_reps": {}},
                    parent_ids=[corr["artifact_id"]],
                )

    def test_require_artifact_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "a.parquet")
            pd.DataFrame({"x": [1.0]}).to_parquet(path, index=False)
            with open(path.replace(".parquet", ".json"), "w", encoding="utf-8") as f:
                json.dump({"columns": ["x"], "targets": [], "labels": []}, f)
            ds = register_dataset(tmp, path, name="a")
            run = ensure_analysis_run(tmp, ds["dataset_id"])
            rid = run["run_id"]
            art = publish_artifact(tmp, rid, KIND_CORRELATION, {"n": 1})
            got = require_artifact(
                tmp, art["artifact_id"], expected_kind=KIND_CORRELATION
            )
            self.assertEqual(got["artifact_id"], art["artifact_id"])
            with self.assertRaises(ValueError):
                require_artifact(
                    tmp, art["artifact_id"], expected_kind=KIND_DISCOVERY_BUNDLE
                )
            with self.assertRaises(ValueError):
                require_artifact(tmp, "missing-id")


if __name__ == "__main__":
    unittest.main()
