"""Tests for permanently deleted Pipeline Features prefs + config prune."""

from __future__ import annotations

import tempfile
import unittest

from chain_replay_ml.dataset_builder.feature_sources_catalog import (
    feature_sources_catalog,
    pipeline_feature_names,
)
from chain_replay_ml.dataset_builder.pipeline_features_config import (
    build_pipeline_features_transformation_config,
    prune_pipeline_transformation_config,
)
from chain_replay_ml.dataset_builder.pipeline_features_prefs import (
    is_excluded_pipeline_feature,
    load_pipeline_transform_prune_features,
    load_retired_pipeline_features,
    retire_pipeline_features,
)


class PipelineFeaturesPrefsTests(unittest.TestCase):
    def test_retire_persists_and_filters_catalogue(self) -> None:
        with tempfile.TemporaryDirectory() as data_dir:
            before = pipeline_feature_names(data_dir=data_dir)
            self.assertGreater(len(before), 10)
            victim = before[0]
            retired = retire_pipeline_features(data_dir, [victim])
            self.assertIn(victim, retired)
            after = pipeline_feature_names(data_dir=data_dir)
            self.assertNotIn(victim, after)
            self.assertEqual(len(after), len(before) - 1)
            cat = feature_sources_catalog(data_dir=data_dir)
            pipe = next(s for s in cat["sources"] if s["id"] == "pipeline")
            self.assertEqual(pipe["total"], len(after))
            self.assertEqual(pipe["retired_count"], 1)

    def test_prune_removes_interaction_outputs(self) -> None:
        cfg = build_pipeline_features_transformation_config(sample_interval_sec=3.0)
        # Pick a known packaging output if present.
        victim = "ltp_ema20_to_ltp_ratio"
        pruned = prune_pipeline_transformation_config(cfg, {victim})
        for stage in pruned.get("transformations") or []:
            if str(stage.get("id") or "") != "interaction":
                continue
            outputs = {
                str((p or {}).get("output") or "")
                for p in (stage.get("params") or {}).get("pairs") or []
            }
            self.assertNotIn(victim, outputs)

    def test_load_empty_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as data_dir:
            self.assertEqual(load_retired_pipeline_features(data_dir), frozenset())

    def test_transform_prune_includes_computed_base(self) -> None:
        with tempfile.TemporaryDirectory() as data_dir:
            skip = load_pipeline_transform_prune_features(data_dir)
            self.assertIn("bs_reiv_pred", skip)
            self.assertIn("dgt_reiv_pred", skip)
            self.assertTrue(is_excluded_pipeline_feature("bs_reiv_pred_lag_6s", data_dir))


if __name__ == "__main__":
    unittest.main()
