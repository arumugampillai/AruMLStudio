"""Tests for feature ownership catalog + Difference / Return transforms."""

from __future__ import annotations

import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.feature_migration import (
    PIPELINE_OWNED_FEATURES,
    RETIRED_FEATURES,
    is_pipeline_owned,
    is_retired,
)
from chain_replay_ml.dataset_builder.feature_ownership import (
    HISTORICAL_FEATURES,
    OWNERSHIP_BASE,
    OWNERSHIP_COMPUTED_BASE,
    OWNERSHIP_HISTORICAL,
    OWNERSHIP_PIPELINE_OWNED,
    OWNERSHIP_RETIRED,
    canonical_registry_features,
    future_generator_of,
    is_canonical,
    ownership_of,
    ownership_summary,
)
from chain_replay_ml.dataset_builder.transformations import (
    run_transformation_pipeline,
)
from chain_replay_ml.dataset_builder.transformations.base import TransformContext
from chain_replay_ml.dataset_builder.transformations.difference import (
    difference_column_name,
)
from chain_replay_ml.dataset_builder.transformations.lag_ui import (
    build_lag_transformation_config,
)
from chain_replay_ml.dataset_builder.transformations.return_transform import (
    return_column_name,
)


class FeatureOwnershipTests(unittest.TestCase):
    def test_historical_catalog_size(self) -> None:
        # Phase 3: all remaining Historical migrated or retired
        self.assertEqual(len(HISTORICAL_FEATURES), 0)

    def test_examples(self) -> None:
        self.assertEqual(ownership_of("ltp"), OWNERSHIP_BASE)
        self.assertEqual(ownership_of("current_iv"), OWNERSHIP_BASE)
        self.assertEqual(ownership_of("dgt_reiv_pred"), OWNERSHIP_COMPUTED_BASE)
        self.assertEqual(ownership_of("ltp_ema20"), OWNERSHIP_COMPUTED_BASE)
        self.assertEqual(ownership_of("iv_ema100"), OWNERSHIP_COMPUTED_BASE)
        # Wave 3: weighted blend levels are Computed Base; packaging is Pipeline Owned.
        self.assertEqual(ownership_of("weighted_ltp_ema"), OWNERSHIP_COMPUTED_BASE)
        self.assertEqual(ownership_of("weighted_spot_ema"), OWNERSHIP_COMPUTED_BASE)
        self.assertEqual(ownership_of("weighted_spot_high_ema"), OWNERSHIP_COMPUTED_BASE)
        self.assertEqual(ownership_of("weighted_spot_low_ema"), OWNERSHIP_COMPUTED_BASE)
        self.assertEqual(ownership_of("weighted_spot_close_ema"), OWNERSHIP_COMPUTED_BASE)
        self.assertEqual(ownership_of("weighted_ltp_ema_to_ltp_ratio"), OWNERSHIP_PIPELINE_OWNED)
        self.assertEqual(ownership_of("weighted_spot_ema_to_ltp_ratio"), OWNERSHIP_PIPELINE_OWNED)
        # Wave 4: sharp momentum levels are Computed Base; packaging is Pipeline Owned.
        self.assertEqual(ownership_of("spot_up_score_5m"), OWNERSHIP_COMPUTED_BASE)
        self.assertEqual(ownership_of("spot_down_score_10m"), OWNERSHIP_COMPUTED_BASE)
        self.assertEqual(ownership_of("spot_up_sample_count_5m"), OWNERSHIP_COMPUTED_BASE)
        self.assertEqual(ownership_of("spot_down_sample_count_1m"), OWNERSHIP_COMPUTED_BASE)
        self.assertEqual(
            ownership_of("spot_up_score_5m_to_ltp_ratio"),
            OWNERSHIP_PIPELINE_OWNED,
        )
        self.assertEqual(
            ownership_of("ltp_to_5m_spot_up_sample_count_ratio"),
            OWNERSHIP_PIPELINE_OWNED,
        )
        # Wave 5: channel width levels are Computed Base; packaging is Pipeline Owned.
        self.assertEqual(ownership_of("spot_ema20_channel_width"), OWNERSHIP_COMPUTED_BASE)
        self.assertEqual(ownership_of("spot_ema300_channel_width"), OWNERSHIP_COMPUTED_BASE)
        self.assertEqual(
            ownership_of("ltp_to_spot_ema20_channel_width_ratio"),
            OWNERSHIP_PIPELINE_OWNED,
        )
        # Wave 6: registry % packaging is Pipeline Owned; canonical levels stay.
        self.assertEqual(ownership_of("spot_ema20"), OWNERSHIP_COMPUTED_BASE)
        self.assertEqual(ownership_of("ce_atm6_ltp_sum"), OWNERSHIP_BASE)
        self.assertEqual(ownership_of("pe_atm6_ltp_sum"), OWNERSHIP_BASE)
        self.assertEqual(ownership_of("spot_vs_ema20_pct"), OWNERSHIP_PIPELINE_OWNED)
        self.assertEqual(ownership_of("ema_spread_pct"), OWNERSHIP_PIPELINE_OWNED)
        self.assertEqual(ownership_of("ema_spread_vs_spot_pct"), OWNERSHIP_PIPELINE_OWNED)
        self.assertEqual(ownership_of("ce_pe_atm6_ltp_diff_pct"), OWNERSHIP_PIPELINE_OWNED)
        self.assertEqual(
            ownership_of("weighted_spot_high_ema_to_ltp_ratio"),
            OWNERSHIP_PIPELINE_OWNED,
        )
        self.assertEqual(
            ownership_of("weighted_spot_low_ema_to_ltp_ratio"),
            OWNERSHIP_PIPELINE_OWNED,
        )
        self.assertEqual(
            ownership_of("weighted_spot_high_ema_to_weighted_spot_low_ema"),
            OWNERSHIP_PIPELINE_OWNED,
        )
        self.assertEqual(
            ownership_of("weighted_spot_ema_to_weighted_spot_low_ema"),
            OWNERSHIP_PIPELINE_OWNED,
        )
        self.assertEqual(
            ownership_of("weighted_spot_ema_to_weighted_spot_high_ema"),
            OWNERSHIP_PIPELINE_OWNED,
        )
        self.assertEqual(ownership_of("ltp_ema20_to_ltp_ratio"), OWNERSHIP_PIPELINE_OWNED)
        self.assertEqual(ownership_of("ltp_return_30s"), OWNERSHIP_PIPELINE_OWNED)
        self.assertTrue(is_retired("dgt_reiv_pred_lag_10s"))
        self.assertTrue(is_retired("ltp_return_5s"))
        self.assertEqual(ownership_of("ltp_to_spot_ratio_lag_30s"), OWNERSHIP_PIPELINE_OWNED)
        self.assertEqual(ownership_of("ltp_to_spot_ratio_change_30s"), OWNERSHIP_PIPELINE_OWNED)
        self.assertEqual(ownership_of("iv_zscore_1m"), OWNERSHIP_PIPELINE_OWNED)
        self.assertEqual(ownership_of("atm_straddle_slope_5m"), OWNERSHIP_PIPELINE_OWNED)
        self.assertEqual(ownership_of("opt_volume_flow_1m"), OWNERSHIP_PIPELINE_OWNED)
        self.assertTrue(is_retired("opt_volume_flow_5s"))
        self.assertEqual(
            ownership_of("atm_straddle_zscore_change_5m"),
            OWNERSHIP_PIPELINE_OWNED,
        )
        self.assertEqual(ownership_of("spot_body_pct_prev1"), OWNERSHIP_RETIRED)
        self.assertEqual(ownership_of("opt_volume_acc_5s_1m"), OWNERSHIP_RETIRED)
        self.assertEqual(ownership_of("spot_body_pct_10s"), OWNERSHIP_RETIRED)
        self.assertFalse(is_canonical("ltp_return_30s"))
        self.assertTrue(is_canonical("dgt_reiv_pred"))
        self.assertEqual(future_generator_of("ltp_return_30s"), "return")
        self.assertEqual(future_generator_of("ltp_to_spot_ratio_lag_30s"), "lag")
        self.assertEqual(future_generator_of("oi_change_1m"), "return")
        self.assertEqual(future_generator_of("oi_change_5m"), "difference")
        self.assertEqual(future_generator_of("iv_zscore_1m"), "rolling_zscore")
        self.assertEqual(future_generator_of("atm_straddle_slope_5m"), "derived")
        self.assertEqual(
            future_generator_of("atm_straddle_zscore_change_5m"),
            "difference",
        )

    def test_registry_admission_blocks_historical_names(self) -> None:
        from chain_replay_ml.dataset_builder.feature_ownership import (
            RegistryAdmissionError,
            assert_registry_admission,
            evaluate_registry_admission,
        )

        blocked = evaluate_registry_admission("foo_lag_30s")
        self.assertFalse(blocked["allowed"])
        blocked2 = evaluate_registry_admission("bar", requires_prior_rows=True)
        self.assertFalse(blocked2["allowed"])
        ok = evaluate_registry_admission("spot_microstructure_score", ownership="base")
        self.assertTrue(ok["allowed"])
        with self.assertRaises(RegistryAdmissionError):
            assert_registry_admission("iv_zscore_1m")
        allowed_exc = evaluate_registry_admission(
            "one_off_derived_x",
            allow_historical=True,
            historical_exception_reason="Cannot express as reusable transform yet",
        )
        self.assertTrue(allowed_exc["allowed"])

    def test_registry_admission_semantic_gates(self) -> None:
        from chain_replay_ml.dataset_builder.feature_ownership import (
            PLACEMENT_BASE,
            PLACEMENT_COMPUTED_BASE,
            PLACEMENT_PIPELINE,
            PLACEMENT_REVIEW_NEW_MODEL,
            evaluate_registry_admission,
            walk_feature_placement_tree,
        )

        # Not “does it multiply?” — caller declares Interaction / config / generic math.
        by_plugin = evaluate_registry_admission(
            "delta_ltp_to_spot_ratio",
            produced_by="interaction",
        )
        self.assertFalse(by_plugin["allowed"])
        self.assertEqual(by_plugin["category"], "interaction")

        by_config = evaluate_registry_admission(
            "experiment_combo_score",
            dataset_builder_configurable=True,
        )
        self.assertFalse(by_config["allowed"])
        self.assertEqual(by_config["category"], "dataset_builder_configurable")

        by_math = evaluate_registry_admission(
            "delta_ltp_to_spot_ratio",
            generic_registry_math=True,
        )
        self.assertFalse(by_math["allowed"])
        self.assertEqual(by_math["category"], "generic_registry_math")

        # Placement decision tree — every feature has exactly one home.
        # Foundational Base vs ordinary composition (both may be ratios).
        self.assertEqual(
            walk_feature_placement_tree(foundational_market_observation=True)["placement"],
            PLACEMENT_BASE,
        )
        self.assertEqual(
            walk_feature_placement_tree(raw_market_observation=True)["placement"],
            PLACEMENT_BASE,
        )  # alias
        self.assertEqual(
            walk_feature_placement_tree(
                foundational_market_observation=False,
                canonical_controller_or_market_model=True,
            )["placement"],
            PLACEMENT_COMPUTED_BASE,
        )
        # Foundational ratios stay Base; ordinary ratios go Pipeline.
        keep_ltp_spot = evaluate_registry_admission(
            "ltp_to_spot_ratio",
            foundational_market_observation=True,
        )
        self.assertTrue(keep_ltp_spot["allowed"])
        self.assertEqual(keep_ltp_spot["placement"], PLACEMENT_BASE)
        pipeline = evaluate_registry_admission(
            "spot_rv_ratio",
            foundational_market_observation=False,
            canonical_controller_or_market_model=False,
            recreatable_from_registry_or_helpers=True,
        )
        self.assertFalse(pipeline["allowed"])
        self.assertEqual(pipeline["placement"], PLACEMENT_PIPELINE)

        review = evaluate_registry_admission(
            "hypothetical_new_microstructure_model",
            foundational_market_observation=False,
            canonical_controller_or_market_model=False,
            recreatable_from_registry_or_helpers=False,
        )
        self.assertFalse(review["allowed"])
        self.assertEqual(review["placement"], PLACEMENT_REVIEW_NEW_MODEL)

        # Separated ATM6 layers: flow core → helper/review; scaling → pipeline.
        flow_core = walk_feature_placement_tree(
            foundational_market_observation=False,
            canonical_controller_or_market_model=False,
            recreatable_from_registry_or_helpers=True,  # Dataset Builder helper today
        )
        self.assertEqual(flow_core["placement"], PLACEMENT_PIPELINE)
        scaling = evaluate_registry_admission(
            "atm6_flow_x_delta_ltp_spot",
            foundational_market_observation=False,
            canonical_controller_or_market_model=False,
            recreatable_from_registry_or_helpers=True,
        )
        self.assertFalse(scaling["allowed"])

        market_state = evaluate_registry_admission(
            "premium_weighted_delta_index",
            ownership="computed_base",
            generic_registry_math=False,
        )
        self.assertTrue(market_state["allowed"])

    def test_registry_partition_sums(self) -> None:
        summary = ownership_summary()
        # Registry = base + computed_base only (no historical / interaction)
        # Wave 3/4/5: weighted + sharp + channel width levels in computed_base;
        # packaging → pipeline_owned.
        self.assertEqual(sum(summary.values()), 206)  # base 69 + computed_base 137
        self.assertEqual(summary[OWNERSHIP_BASE], 69)
        self.assertEqual(summary[OWNERSHIP_COMPUTED_BASE], 137)
        self.assertEqual(summary[OWNERSHIP_HISTORICAL], 0)
        self.assertEqual(summary.get(OWNERSHIP_PIPELINE_OWNED, 0), 0)
        self.assertEqual(len(PIPELINE_OWNED_FEATURES), 212)
        self.assertEqual(len(RETIRED_FEATURES), 21)
        canon = canonical_registry_features()
        self.assertEqual(len(canon), summary[OWNERSHIP_BASE] + summary[OWNERSHIP_COMPUTED_BASE])
        self.assertNotIn("ltp_return_30s", canon)
        self.assertNotIn("ltp_to_spot_ratio_lag_30s", canon)
        self.assertNotIn("dgt_reiv_pred_lag_30s", canon)
        self.assertNotIn("iv_zscore_1m", canon)
        self.assertNotIn("atm_straddle_zscore_change_5m", canon)
        self.assertNotIn("delta_x_spot", canon)
        self.assertNotIn("spot_ema50_to_ltp_ratio_x_moneyness", canon)
        self.assertNotIn("delta_ltp_to_spot_ratio", canon)
        self.assertNotIn("current_to_atm6_flow_delta_ltp_to_spot_ratio", canon)
        self.assertNotIn("ltp_to_dgt_reiv_ratio", canon)
        self.assertNotIn("spot_rv_ratio", canon)
        # Wave 2/3 packaged ratios are pipeline-owned, not canonical registry.
        self.assertNotIn("ltp_ema20_to_ltp_ratio", canon)
        self.assertNotIn("iv_ema9_to_spot_ratio", canon)
        self.assertNotIn("spot_ema50_to_ltp_ratio", canon)
        self.assertNotIn("spot_high_ema20_to_ltp_ratio", canon)
        self.assertNotIn("weighted_ltp_ema_to_ltp_ratio", canon)
        self.assertNotIn("weighted_spot_ema_to_ltp_ratio", canon)
        self.assertNotIn("weighted_spot_high_ema_to_ltp_ratio", canon)
        self.assertNotIn("weighted_spot_high_ema_to_weighted_spot_low_ema", canon)
        # Wave 4 packaging is pipeline-owned, not canonical registry.
        self.assertNotIn("spot_up_score_5m_to_ltp_ratio", canon)
        self.assertNotIn("ltp_to_5m_spot_up_sample_count_ratio", canon)
        # Wave 5 packaging is pipeline-owned; width levels are canonical.
        self.assertNotIn("ltp_to_spot_ema20_channel_width_ratio", canon)
        self.assertIn("spot_ema20_channel_width", canon)
        self.assertIn("spot_ema300_channel_width", canon)
        self.assertIn("ltp_ema20", canon)
        self.assertIn("iv_ema100", canon)
        self.assertIn("spot_ema300", canon)
        self.assertIn("weighted_ltp_ema", canon)
        self.assertIn("weighted_spot_ema", canon)
        self.assertIn("weighted_spot_high_ema", canon)
        self.assertIn("weighted_spot_low_ema", canon)
        self.assertIn("weighted_spot_close_ema", canon)
        self.assertIn("spot_up_score_5m", canon)
        self.assertIn("spot_down_score_10m", canon)
        self.assertIn("spot_up_sample_count_5m", canon)
        self.assertIn("spot_down_sample_count_1m", canon)
        self.assertIn("ltp", canon)
        self.assertIn("dgt_prediction_error", canon)
        self.assertIn("ltp_to_spot_ratio", canon)
        self.assertIn("moneyness", canon)
        self.assertIn("dgt_reiv_pred", canon)
    def test_retired_not_pipeline_owned(self) -> None:
        for name in RETIRED_FEATURES:
            self.assertTrue(is_retired(name))
            self.assertFalse(is_pipeline_owned(name))
            self.assertNotIn(name, HISTORICAL_FEATURES)


class DifferenceReturnTests(unittest.TestCase):
    def test_column_names(self) -> None:
        self.assertEqual(difference_column_name("ltp", 30), "ltp_diff_30s")
        self.assertEqual(return_column_name("spot", 60), "spot_return_60s")
        self.assertEqual(
            difference_column_name("ltp", 60, column="ltp_change_1m"),
            "ltp_change_1m",
        )
        self.assertEqual(
            return_column_name("spot", 15, column="spot_change_15s"),
            "spot_change_15s",
        )

    def test_difference_and_return_create_columns(self) -> None:
        df = pd.DataFrame({
            "trading_day": ["2024-01-01"] * 4,
            "token": ["A"] * 4,
            "ltp": [10.0, 12.0, 15.0, 20.0],
        })
        cfg = {
            "transformation_pipeline_version": 1,
            "transformations": [
                {
                    "id": "difference",
                    "enabled": True,
                    "params": {
                        "features": ["ltp"],
                        "lag_seconds": [3],
                        "partition_by": ["trading_day", "token"],
                    },
                },
                {
                    "id": "return",
                    "enabled": True,
                    "params": {
                        "features": ["ltp"],
                        "lag_seconds": [3],
                        "partition_by": ["trading_day", "token"],
                    },
                },
            ],
        }
        ctx = TransformContext(config=cfg, sample_interval_sec=3.0)
        result = run_transformation_pipeline(df, cfg, context=ctx)
        self.assertEqual(result.executed, 2)
        self.assertIn("ltp_diff_3s", result.frame.columns)
        self.assertIn("ltp_return_3s", result.frame.columns)
        self.assertTrue(pd.isna(result.frame.loc[0, "ltp_diff_3s"]))
        self.assertEqual(result.frame.loc[1, "ltp_diff_3s"], 2.0)
        self.assertEqual(result.frame.loc[1, "ltp_return_3s"], 0.2)

    def test_master_compatible_column_and_scale(self) -> None:
        """Difference column= and Return scale=100 produce Master names / pct×100."""
        df = pd.DataFrame({
            "trading_day": ["2024-01-01"] * 25,
            "token": ["A"] * 25,
            "ltp": [100.0 + i for i in range(25)],
            "spot": [200.0 + i for i in range(25)],
            "oi": [1000.0 + 10 * i for i in range(25)],
        })
        # Difference: Master abs name via column=
        diff_cfg = {
            "transformation_pipeline_version": 1,
            "transformations": [{
                "id": "difference",
                "enabled": True,
                "params": {
                    "features": ["ltp"],
                    "horizons": [
                        {"seconds": 60, "suffix": "1m", "column": "ltp_change_1m"},
                    ],
                    "partition_by": ["trading_day", "token"],
                    "sample_interval_sec": 3.0,
                },
            }],
        }
        # Return: Master pct name via column= + scale=100
        ret_cfg = {
            "transformation_pipeline_version": 1,
            "transformations": [{
                "id": "return",
                "enabled": True,
                "params": {
                    "features": ["spot"],
                    "horizons": [
                        {"seconds": 15, "suffix": "15s", "column": "spot_change_15s"},
                    ],
                    "scale": 100,
                    "partition_by": ["trading_day", "token"],
                    "sample_interval_sec": 3.0,
                },
            }],
        }
        oi_cfg = {
            "transformation_pipeline_version": 1,
            "transformations": [{
                "id": "return",
                "enabled": True,
                "params": {
                    "features": ["oi"],
                    "horizons": [
                        {"seconds": 60, "suffix": "1m", "column": "oi_change_1m"},
                    ],
                    "scale": 100,
                    "partition_by": ["trading_day", "token"],
                    "sample_interval_sec": 3.0,
                },
            }],
        }
        frame = df
        for cfg in (diff_cfg, ret_cfg, oi_cfg):
            ctx = TransformContext(config=cfg, sample_interval_sec=3.0)
            frame = run_transformation_pipeline(frame, cfg, context=ctx).frame
        self.assertIn("ltp_change_1m", frame.columns)
        self.assertIn("spot_change_15s", frame.columns)
        self.assertIn("oi_change_1m", frame.columns)
        # 60s / 3s = 20 rows; row 20: ltp 120 − 100 = 20
        self.assertEqual(frame.loc[20, "ltp_change_1m"], 20.0)
        # 15s / 3s = 5 rows; row 5: (205−200)/200 * 100 = 2.5
        self.assertAlmostEqual(frame.loc[5, "spot_change_15s"], 2.5)
        # oi row 20: (1200−1000)/1000 * 100 = 20
        self.assertAlmostEqual(frame.loc[20, "oi_change_1m"], 20.0)

    def test_pipeline_config_builder_multi(self) -> None:
        cfg = build_lag_transformation_config(
            enabled=True,
            features=["ltp"],
            lag_seconds=[30, 60],
            sample_interval_sec=3,
            difference_enabled=True,
            return_enabled=True,
        )
        ids = [t["id"] for t in cfg["transformations"]]
        self.assertEqual(ids, ["lag", "difference", "return"])


if __name__ == "__main__":
    unittest.main()
