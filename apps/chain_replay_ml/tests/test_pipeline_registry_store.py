"""Tests for pipeline_registry_store."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.dataset_builder.pipeline_registry_store import (
    add_candidate_features,
    create_pipeline,
    delete_pipeline,
    ensure_default_existing_pipeline,
    format_pipeline_id,
    get_pipeline_summary,
    load_store,
    save_store,
    set_registry_members,
)


class PipelineRegistryStoreTests(unittest.TestCase):
    def test_create_pipeline_assigns_unique_id(self) -> None:
        doc = load_store(self._tmpdir)
        rec = create_pipeline(doc, name="Pipeline_002", pipeline_type="auto")
        self.assertEqual(rec["pipeline_id"], format_pipeline_id(1))
        self.assertEqual(rec["name"], "Pipeline_002")
        self.assertEqual(rec["type"], "auto")
        self.assertEqual(rec["status"], "draft")
        save_store(self._tmpdir, doc)
        doc2 = load_store(self._tmpdir)
        summary = get_pipeline_summary(doc2, rec["pipeline_id"])
        self.assertIsNotNone(summary)
        self.assertEqual(summary["feature_count"], 0)

    def test_registry_members_and_candidates(self) -> None:
        doc = load_store(self._tmpdir)
        rec = create_pipeline(doc, pipeline_type="auto")
        pid = rec["pipeline_id"]
        set_registry_members(doc, pid, ["FR0001", "FR0002"])
        add_candidate_features(doc, pid, ["ltp_lag_60", "iv_zscore_60"])
        summary = get_pipeline_summary(doc, pid)
        self.assertEqual(summary["registry_feature_count"], 2)
        self.assertEqual(summary["candidate_count"], 2)
        self.assertEqual(summary["feature_count"], 4)

    def test_ensure_default_existing_pipeline(self) -> None:
        doc = ensure_default_existing_pipeline(self._tmpdir)
        rows = list((doc.get("pipelines") or {}).values())
        self.assertEqual(len(rows), 1)
        rec = rows[0]
        self.assertEqual(rec["type"], "existing")
        self.assertEqual(rec["status"], "ready")
        self.assertGreater(len(rec.get("candidate_features") or []), 0)

    def test_delete_pipeline(self) -> None:
        doc = load_store(self._tmpdir)
        rec = create_pipeline(doc, pipeline_type="auto")
        pid = rec["pipeline_id"]
        self.assertTrue(delete_pipeline(doc, pid))
        self.assertIsNone(get_pipeline_summary(doc, pid))

    def test_delete_existing_pipeline_rejected(self) -> None:
        doc = ensure_default_existing_pipeline(self._tmpdir)
        pid = list((doc.get("pipelines") or {}).keys())[0]
        with self.assertRaises(ValueError):
            delete_pipeline(doc, pid)

    def setUp(self) -> None:
        self._tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        path = os.path.join(self._tmpdir, "pipeline_registry_store.json")
        if os.path.isfile(path):
            os.remove(path)
        os.rmdir(self._tmpdir)


if __name__ == "__main__":
    unittest.main()
