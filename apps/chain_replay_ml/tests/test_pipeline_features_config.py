"""Pipeline features regeneration config coverage."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.feature_migration import PIPELINE_OWNED_FEATURES
from chain_replay_ml.dataset_builder.pipeline_features_config import (
    build_pipeline_features_transformation_config,
    collect_transformation_source_names,
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
            elif tid == "current_to_atm6_flow":
                col = params.get("column")
                if col:
                    outs.add(str(col))

        hits = PIPELINE_OWNED_FEATURES & outs
        self.assertGreaterEqual(len(hits), 207)
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
        pruned = prune_pipeline_transformation_config(
            cfg,
            {"atm_straddle_zscore_30m"},
        )
        for stage in pruned.get("transformations") or []:
            if str(stage.get("id") or "") != "rolling_statistics":
                continue
            params = stage.get("params") or {}
            for window in params.get("windows") or []:
                self.assertNotEqual(
                    str((window or {}).get("column") or ""),
                    "atm_straddle_zscore_30m",
                )

        via_build = build_pipeline_features_transformation_config(
            sample_interval_sec=3.0,
            exclude_features={"atm_straddle_zscore_30m"},
        )
        for stage in via_build.get("transformations") or []:
            if str(stage.get("id") or "") != "rolling_statistics":
                continue
            for window in (stage.get("params") or {}).get("windows") or []:
                self.assertNotEqual(
                    str((window or {}).get("column") or ""),
                    "atm_straddle_zscore_30m",
                )


    def test_prune_registry_retired_base_keeps_source_drops_outputs(self) -> None:
        from chain_replay_ml.dataset_builder.transformations.config import normalize_transformation_config

        cfg = normalize_transformation_config({
            "transformations": [{
                "id": "lag",
                "enabled": True,
                "params": {
                    "features": ["dgt_reiv_pred"],
                    "horizons": [{"seconds": 30.0, "suffix": "30s"}],
                    "partition_by": ["trading_day", "token"],
                    "sample_interval_sec": 6.0,
                },
            }],
        })
        kept = prune_pipeline_transformation_config(
            cfg,
            set(),
            interaction_operand_skip=set(),
        )
        lag_sources = {
            str(f)
            for t in kept.get("transformations") or []
            if str(t.get("id") or "") == "lag"
            for f in (t.get("params") or {}).get("features") or []
        }
        self.assertIn("dgt_reiv_pred", lag_sources)

        dropped_outputs = prune_pipeline_transformation_config(cfg, {"dgt_reiv_pred"})
        lag_stages = [
            t for t in dropped_outputs.get("transformations") or []
            if str(t.get("id") or "") == "lag"
        ]
        self.assertEqual(lag_stages, [])

        dropped_sources = prune_pipeline_transformation_config(
            cfg,
            set(),
            source_exclude={"dgt_reiv_pred"},
        )
        lag_stages_src = [
            t for t in dropped_sources.get("transformations") or []
            if str(t.get("id") or "") == "lag"
        ]
        self.assertEqual(lag_stages_src, [])

    def test_build_pipeline_config_excludes_retired_registry_sources(self) -> None:
        retired = {"bs_reiv_pred", "dgt_reiv_pred", "dgt_prediction_error"}
        cfg = build_pipeline_features_transformation_config(
            sample_interval_sec=6.0,
            exclude_features=retired,
            interaction_operand_skip=retired,
            source_forbidden=retired,
        )
        sources = collect_transformation_source_names(cfg)
        for name in retired:
            self.assertNotIn(name, sources)

    def test_prune_interaction_drops_orphaned_intermediate_pairs(self) -> None:
        cfg = build_pipeline_features_transformation_config(sample_interval_sec=6.0)
        skip = {"spot_ema9", "spot_ema20"}
        pruned = prune_pipeline_transformation_config(
            cfg,
            skip,
            interaction_operand_skip=skip,
        )
        for stage in pruned.get("transformations") or []:
            if str(stage.get("id") or "") != "interaction":
                continue
            pairs = (stage.get("params") or {}).get("pairs") or []
            outputs = {str(p.get("output") or "") for p in pairs}
            self.assertNotIn("ema9_minus_ema20", outputs)
            self.assertNotIn("ema_spread_pct", outputs)
            self.assertNotIn("ema_spread_vs_spot_pct", outputs)
            for p in pairs:
                left = str(p.get("left") or "")
                self.assertNotIn(left, ("ema9_minus_ema20", "spot_minus_spot_ema20"))

    def test_prune_keeps_registry_operands_for_interaction_when_not_operand_skip(self) -> None:
        cfg = build_pipeline_features_transformation_config(sample_interval_sec=6.0)
        pruned = prune_pipeline_transformation_config(
            cfg,
            set(),
            interaction_operand_skip=set(),
        )
        for stage in pruned.get("transformations") or []:
            if str(stage.get("id") or "") != "interaction":
                continue
            outputs = {
                str(p.get("output") or "")
                for p in (stage.get("params") or {}).get("pairs") or []
            }
            self.assertIn("ema9_minus_ema20", outputs)
            self.assertIn("ema_spread_pct", outputs)

    def test_prune_keeps_skip_set_names_as_lag_sources(self) -> None:
        from chain_replay_ml.dataset_builder.transformations.config import normalize_transformation_config

        cfg = normalize_transformation_config({
            "transformations": [{
                "id": "lag",
                "enabled": True,
                "params": {
                    "features": ["ask_depth_l1_5", "atm_pcr"],
                    "horizons": [{"seconds": 60.0}, {"seconds": 300.0}],
                    "partition_by": ["trading_day", "token"],
                    "sample_interval_sec": 6.0,
                },
            }],
        })
        pruned = prune_pipeline_transformation_config(
            cfg,
            {"spot_ema9"},
            interaction_operand_skip=set(),
        )
        lag_sources: set[str] = set()
        for stage in pruned.get("transformations") or []:
            if str(stage.get("id") or "") != "lag":
                continue
            lag_sources.update(
                str(f) for f in (stage.get("params") or {}).get("features") or []
            )
        self.assertIn("ask_depth_l1_5", lag_sources)
        self.assertIn("atm_pcr", lag_sources)

    def test_prune_keeps_registry_operands_with_retired_operand_skip_only(self) -> None:
        cfg = build_pipeline_features_transformation_config(sample_interval_sec=6.0)
        from chain_replay_ml.dataset_builder.feature_ownership import non_registry_transform_source_names

        skip = set(non_registry_transform_source_names())
        pruned = prune_pipeline_transformation_config(
            cfg,
            skip,
            interaction_operand_skip=set(),
        )
        for stage in pruned.get("transformations") or []:
            if str(stage.get("id") or "") != "interaction":
                continue
            outputs = {
                str(p.get("output") or "")
                for p in (stage.get("params") or {}).get("pairs") or []
            }
            self.assertIn("ltp_ema9_to_ltp_ratio", outputs)
            self.assertIn("ema_spread_pct", outputs)


if __name__ == "__main__":
    unittest.main()
