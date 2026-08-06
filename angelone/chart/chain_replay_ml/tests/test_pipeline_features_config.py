"""Pipeline features regeneration config coverage."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.feature_migration import PIPELINE_OWNED_FEATURES
from chain_replay_ml.dataset_builder.pipeline_features_config import (
    build_pipeline_features_transformation_config,
    prune_pipeline_transformation_config,
)


class PipelineFeaturesConfigTests(unittest.TestCase):
    def test_config_covers_most_catalogue(self) -> None:
        cfg = build_pipeline_features_transformation_config(sample_interval_sec=3.0)
        outs: set[str] = set()
        for t in cfg["transformations"]:
            tid = str(t.get("id") or "")
            params = t.get("params") or {}
            if tid == "interaction":
                for pair in params.get("pairs") or []:
                    outs.add(str(pair.get("output") or ""))
            elif tid in ("lag", "difference", "return", "difference_clip"):
                for h in params.get("horizons") or []:
                    col = h.get("column")
                    if col:
                        outs.add(str(col))
                    elif h.get("suffix") and params.get("features"):
                        suf = str(h["suffix"])
                        for feat in params["features"]:
                            if tid == "lag":
                                outs.add(f"{feat}_lag_{suf}")
            elif tid == "rolling_statistics":
                for w in params.get("windows") or []:
                    if w.get("column"):
                        outs.add(str(w["column"]))
            elif tid == "rolling_ohlc":
                outs.update(str(v) for v in (params.get("column_map") or {}).values())
            elif tid in ("derived", "anchor_return"):
                for o in params.get("outputs") or []:
                    if o.get("column"):
                        outs.add(str(o["column"]))

        hits = PIPELINE_OWNED_FEATURES & outs
        # 212 - 2 known blockers (ATM6 flow core + weighted_iv_zscore product)
        self.assertGreaterEqual(len(hits), 210)
        self.assertIn("interaction", [t["id"] for t in cfg["transformations"]])

    def test_time_shift_bases_exist_on_registry(self) -> None:
        from chain_replay_ml.dataset_builder.feature_ownership import canonical_registry_features

        registry = canonical_registry_features()
        cfg = build_pipeline_features_transformation_config(sample_interval_sec=3.0)
        created: set[str] = set()
        for t in cfg["transformations"]:
            params = t.get("params") or {}
            tid = str(t.get("id") or "")
            if tid in ("lag", "difference", "return", "difference_clip", "rolling_statistics", "rolling_ohlc"):
                for feat in params.get("features") or []:
                    if feat not in registry and feat not in created:
                        self.fail(f"{tid} base {feat!r} not in Registry")
                for h in (params.get("horizons") or params.get("windows") or []):
                    if h.get("column"):
                        created.add(str(h["column"]))
            elif tid in ("derived", "anchor_return"):
                for o in params.get("outputs") or []:
                    feat = o.get("feature")
                    if feat and feat not in registry and feat not in created:
                        self.fail(f"{tid} base {feat!r} not in Registry")
                    if o.get("column"):
                        created.add(str(o["column"]))
        self.assertIn("option_oi", registry)
        self.assertIn("option_day_volume", registry)

    def test_prune_removes_retired_rolling_statistics_windows(self) -> None:
        """UI delete must stop regenerating rolling_statistics column outputs."""
        cfg = build_pipeline_features_transformation_config(sample_interval_sec=3.0)
        raw = str(cfg)
        self.assertIn("atm_straddle_zscore_30m", raw)
        pruned = prune_pipeline_transformation_config(
            cfg,
            {"atm_straddle_zscore_30m"},
        )
        text = str(pruned)
        self.assertNotIn("atm_straddle_zscore_30m", text)
        # Dependent difference of the retired base must also drop.
        self.assertNotIn("atm_straddle_zscore_change_5m", text)

        via_build = build_pipeline_features_transformation_config(
            sample_interval_sec=3.0,
            exclude_features={"atm_straddle_zscore_30m"},
        )
        self.assertNotIn("atm_straddle_zscore_30m", str(via_build))


if __name__ == "__main__":
    unittest.main()
