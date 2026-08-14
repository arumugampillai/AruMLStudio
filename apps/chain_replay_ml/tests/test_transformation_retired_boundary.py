"""Retired registry features must not enter the transformation system."""

from __future__ import annotations

import os
import tempfile
import unittest

from chain_replay_ml.dataset_builder.feature_registry_store import (
    load_store,
    set_feature_registry_active,
)
from chain_replay_ml.dataset_builder.feature_sources_catalog import (
    get_active_feature_names,
    registry_feature_names,
    transformation_forbidden_feature_names,
)
from chain_replay_ml.dataset_builder.pipeline_features_config import (
    build_pipeline_features_transformation_config,
    collect_transformation_source_names,
    sanitize_transformation_config_before_execution,
)
from chain_replay_ml.dataset_builder.pipeline_features_prefs import (
    load_pipeline_output_prune_features,
    load_transformation_forbidden_features,
)
from chain_replay_ml.dataset_builder.transformations.config import normalize_transformation_config
from chain_replay_ml.dataset_builder.transformations.pipeline import run_transformation_pipeline
from chain_replay_ml.dataset_builder.transformations.base import TransformContext

RETIRED = ("bs_reiv_pred", "dgt_reiv_pred", "dgt_prediction_error")


class TransformationRetiredBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.data_dir = self._tmp.name
        os.makedirs(self.data_dir, exist_ok=True)
        store = load_store(self.data_dir)
        for name in RETIRED:
            set_feature_registry_active(
                self.data_dir,
                store,
                name,
                active=False,
                home_group_id="price",
            )
            store = load_store(self.data_dir)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_active_feature_names_exclude_retired(self) -> None:
        active = set(get_active_feature_names(self.data_dir))
        for name in RETIRED:
            self.assertNotIn(name, active)
            self.assertNotIn(name, registry_feature_names(data_dir=self.data_dir))

    def test_forbidden_set_includes_retired_registry_names(self) -> None:
        forbidden = transformation_forbidden_feature_names(self.data_dir)
        for name in RETIRED:
            self.assertIn(name, forbidden)
        self.assertEqual(forbidden, load_transformation_forbidden_features(self.data_dir))

    def test_pipeline_build_never_requests_retired_sources(self) -> None:
        forbidden = load_transformation_forbidden_features(self.data_dir)
        output_skip = load_pipeline_output_prune_features(self.data_dir)
        cfg = build_pipeline_features_transformation_config(
            sample_interval_sec=6.0,
            exclude_features=output_skip,
            interaction_operand_skip=forbidden,
            source_forbidden=forbidden,
        )
        sources = collect_transformation_source_names(cfg)
        for name in RETIRED:
            self.assertNotIn(name, sources)

    def test_execution_skips_retired_without_feature_not_found(self) -> None:
        import pandas as pd

        cfg = normalize_transformation_config({
            "transformations": [{
                "id": "lag",
                "enabled": True,
                "params": {
                    "features": list(RETIRED),
                    "horizons": [{"seconds": 30.0, "suffix": "30s"}],
                    "partition_by": ["trading_day", "token"],
                    "sample_interval_sec": 6.0,
                },
            }],
        })
        df = pd.DataFrame({
            "trading_day": ["2026-01-01"],
            "token": ["T1"],
            "ltp": [100.0],
        })
        logs: list[str] = []
        ctx = TransformContext(
            config=cfg,
            data_dir=self.data_dir,
            sample_interval_sec=6.0,
            logger=logs.append,
        )
        result = run_transformation_pipeline(df, cfg, context=ctx, log_fn=logs.append)
        self.assertTrue(any("SKIPPED_RETIRED_SOURCE" in line for line in logs))
        self.assertNotIn("dgt_reiv_pred_lag_30s", result.frame.columns)

    def test_sanitize_before_execution_strips_stale_candidates(self) -> None:
        cfg = normalize_transformation_config({
            "transformations": [{
                "id": "difference",
                "enabled": True,
                "params": {
                    "features": ["bs_reiv_pred"],
                    "horizons": [{"seconds": 60.0, "suffix": "1m"}],
                    "partition_by": ["trading_day", "token"],
                    "sample_interval_sec": 6.0,
                },
            }],
        })
        pruned, skipped = sanitize_transformation_config_before_execution(cfg, self.data_dir)
        self.assertIn("bs_reiv_pred", skipped)
        sources = collect_transformation_source_names(pruned)
        self.assertNotIn("bs_reiv_pred", sources)


if __name__ == "__main__":
    unittest.main()
