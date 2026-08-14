"""Tests for auto_candidate_generation prefs and config builder."""

from __future__ import annotations

import unittest

from master_dataset_tk.auto_candidate_generation import (
    build_auto_candidate_transformation_config,
    candidate_generation_prefs_snapshot,
    default_candidate_generation_prefs,
    normalize_candidate_generation_prefs,
    _filter_active_source_features,
    _filter_registry_source_features,
)
from chain_replay_ml.dataset_builder.pipeline_features_config import (
    expected_pipeline_outputs_from_config,
)


class AutoCandidateGenerationTests(unittest.TestCase):
    def test_default_prefs_all_transforms_enabled(self) -> None:
        prefs = default_candidate_generation_prefs()
        self.assertEqual(prefs["source"], "registry")
        self.assertTrue(all(prefs["transformations"].values()))
        self.assertEqual(prefs["horizons_sec"], [6, 12, 30, 60, 120, 300])
        self.assertTrue(all(prefs["interaction_ops"].values()))

    def test_normalize_rejects_invalid_source(self) -> None:
        out = normalize_candidate_generation_prefs({"source": "invalid"})
        self.assertEqual(out["source"], "registry")

    def test_snapshot_roundtrip(self) -> None:
        snap = candidate_generation_prefs_snapshot(
            source="both",
            transformations={"lag": False, "difference": True},
            horizons_sec=[12, 30],
            interaction_ops={"multiply": True, "divide": False},
        )
        self.assertEqual(snap["source"], "both")
        self.assertFalse(snap["transformations"]["lag"])
        self.assertTrue(snap["transformations"]["difference"])
        self.assertEqual(snap["horizons_sec"], [12, 30])
        self.assertTrue(snap["interaction_ops"]["multiply"])
        self.assertFalse(snap["interaction_ops"]["divide"])

    def test_build_config_with_lag_only(self) -> None:
        prefs = candidate_generation_prefs_snapshot(
            source="registry",
            transformations={k: (k == "lag") for k in default_candidate_generation_prefs()["transformations"]},
            horizons_sec=[6, 12],
            interaction_ops={k: False for k in default_candidate_generation_prefs()["interaction_ops"]},
        )
        config = build_auto_candidate_transformation_config(
            features=["ltp", "iv"],
            interval_sec=3,
            candidate_prefs=prefs,
        )
        transforms = config.get("transformations") or []
        self.assertGreater(len(transforms), 0)
        ids = {t.get("id") for t in transforms}
        self.assertIn("lag", ids)

    def test_build_config_empty_features(self) -> None:
        config = build_auto_candidate_transformation_config(
            features=[],
            interval_sec=3,
            candidate_prefs=default_candidate_generation_prefs(),
        )
        self.assertEqual(config.get("transformations"), [])

    def test_expected_outputs_from_auto_config(self) -> None:
        feats = ["ltp", "current_iv"]
        prefs = default_candidate_generation_prefs()
        config = build_auto_candidate_transformation_config(
            features=feats,
            interval_sec=3,
            candidate_prefs=prefs,
        )
        names = expected_pipeline_outputs_from_config(config, master_features=feats)
        self.assertGreater(len(names), 50)
        self.assertIn("ltp_diff_6s", names)
        self.assertIn("current_iv_diff_6s", names)

    def test_filter_registry_source_excludes_pipeline_owned(self) -> None:
        names = _filter_registry_source_features(
            ["ltp", "bs_reiv_pred", "dgt_reiv_pred", "dgt_prediction_error", "iv"],
            "/nonexistent",
        )
        self.assertEqual(names, ["ltp", "iv"])
        import os
        import tempfile

        from chain_replay_ml.dataset_builder.pipeline_features_prefs import (
            save_retired_pipeline_features,
        )

        tmp = tempfile.mkdtemp()
        save_retired_pipeline_features(tmp, ["retired_feat"])
        names = _filter_active_source_features(["ltp", "retired_feat", "iv"], tmp)
        self.assertEqual(names, ["ltp", "iv"])
        path = os.path.join(tmp, "pipeline_features_retired.json")
        if os.path.isfile(path):
            os.remove(path)
        os.rmdir(tmp)


if __name__ == "__main__":
    unittest.main()
