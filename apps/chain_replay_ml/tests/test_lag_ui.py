"""Unit tests for Lag transform UI helpers."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.transformations.lag_ui import (
    _GROUP_TO_CATEGORY,
    build_lag_transformation_config,
    classify_feature,
    clear_feature_category_cache,
    default_selected_lag_features,
    features_for_preset,
    filter_features_by_search,
    filter_laggable_features,
    format_lag_preview_text,
    group_features_by_category,
    lag_preview_counts,
    lag_seconds_label,
    lag_warmup_warning,
    validate_lag_for_export,
    validate_time_shift_settings,
)


class TestLagUiHelpers(unittest.TestCase):
    def test_build_config_disabled(self) -> None:
        cfg = build_lag_transformation_config(
            enabled=False,
            features=["ltp"],
            lag_seconds=[30, 60],
        )
        self.assertEqual(cfg["transformation_pipeline_version"], 1)
        self.assertEqual(cfg["transformations"], [])

    def test_build_config_enabled(self) -> None:
        cfg = build_lag_transformation_config(
            enabled=True,
            features=["ltp", "spot"],
            lag_seconds=[60, 30, 60],
            sample_interval_sec=3,
        )
        self.assertEqual(len(cfg["transformations"]), 1)
        step = cfg["transformations"][0]
        self.assertEqual(step["id"], "lag")
        self.assertTrue(step["enabled"])
        self.assertEqual(step["params"]["features"], ["ltp", "spot"])
        self.assertEqual(step["params"]["lag_seconds"], [30, 60])
        self.assertEqual(step["params"]["partition_by"], ["trading_day", "token"])
        self.assertEqual(step["params"]["sample_interval_sec"], 3)

    def test_build_config_independent_difference_return(self) -> None:
        cfg = build_lag_transformation_config(
            enabled=True,
            features=["ltp"],
            lag_seconds=[30],
            difference_enabled=True,
            difference_features=["spot"],
            difference_lag_seconds=[60],
            return_enabled=True,
            return_features=["delta"],
            return_lag_seconds=[90],
            sample_interval_sec=10,
        )
        by_id = {t["id"]: t for t in cfg["transformations"]}
        self.assertEqual(set(by_id), {"lag", "difference", "return"})
        self.assertEqual(by_id["lag"]["params"]["features"], ["ltp"])
        self.assertEqual(by_id["lag"]["params"]["lag_seconds"], [30])
        self.assertEqual(by_id["difference"]["params"]["features"], ["spot"])
        self.assertEqual(by_id["difference"]["params"]["lag_seconds"], [60])
        self.assertEqual(by_id["return"]["params"]["features"], ["delta"])
        self.assertEqual(by_id["return"]["params"]["lag_seconds"], [90])

    def test_validate_time_shift_independent_stages(self) -> None:
        err = validate_time_shift_settings(
            lag_enabled=False,
            difference_enabled=True,
            return_enabled=False,
            features=["ltp"],
            lag_seconds=[30],
            sample_interval_sec=10,
            available_features=["ltp", "spot"],
            difference_features=[],
            difference_lag_seconds=[60],
        )
        self.assertIsNotNone(err)
        self.assertIn("Difference", err or "")

        self.assertIsNone(
            validate_time_shift_settings(
                lag_enabled=True,
                difference_enabled=True,
                return_enabled=True,
                features=["ltp"],
                lag_seconds=[30],
                sample_interval_sec=10,
                available_features=["ltp", "spot", "delta"],
                difference_features=["spot"],
                difference_lag_seconds=[60],
                return_features=["delta"],
                return_lag_seconds=[90],
            )
        )

    def test_preview_counts(self) -> None:
        c = lag_preview_counts(
            enabled=True,
            selected_features=["ltp", "spot"],
            lag_seconds=[30, 60, 90],
            current_columns=100,
            estimated_rows=1_000_000,
        )
        self.assertEqual(c["columns_to_add"], 6)
        self.assertEqual(c["final_columns"], 106)
        self.assertEqual(c["current_columns"], 100)
        self.assertEqual(c["lag_columns"], 6)
        self.assertIsNotNone(c["estimated_memory_mb"])
        self.assertNotEqual(c["estimated_memory_label"], "—")
        off = lag_preview_counts(
            enabled=False,
            selected_features=["ltp"],
            lag_seconds=[30],
            current_columns=100,
        )
        self.assertEqual(off["columns_to_add"], 0)
        self.assertEqual(off["final_columns"], 100)

    def test_preview_text_includes_final_and_memory(self) -> None:
        c = lag_preview_counts(
            enabled=True,
            selected_features=["ltp"],
            lag_seconds=[30],
            current_columns=262,
            estimated_rows=1000,
        )
        text = format_lag_preview_text(c, enabled=True)
        self.assertIn("Current Columns   : 262", text)
        self.assertIn("New Columns       : 1", text)
        self.assertIn("Final Columns     : 263", text)
        self.assertIn("Estimated Memory", text)

    def test_validate_requires_features_and_lags(self) -> None:
        self.assertIn(
            "no features",
            (validate_lag_for_export(
                enabled=True,
                features=[],
                lag_seconds=[30],
                sample_interval_sec=10,
            ) or "").lower(),
        )
        self.assertIn(
            "no lag",
            (validate_lag_for_export(
                enabled=True,
                features=["ltp"],
                lag_seconds=[],
                sample_interval_sec=10,
            ) or "").lower(),
        )

    def test_validate_exact_multiple(self) -> None:
        self.assertIsNone(
            validate_lag_for_export(
                enabled=True,
                features=["ltp"],
                lag_seconds=[30, 60],
                sample_interval_sec=10,
            )
        )
        err = validate_lag_for_export(
            enabled=True,
            features=["ltp"],
            lag_seconds=[45],
            sample_interval_sec=10,
        )
        self.assertIsNotNone(err)
        self.assertIn("exact multiples", err or "")

    def test_validate_missing_features(self) -> None:
        err = validate_lag_for_export(
            enabled=True,
            features=["ltp", "missing_x"],
            lag_seconds=[30],
            sample_interval_sec=10,
            available_features=["ltp", "spot"],
        )
        self.assertIsNotNone(err)
        self.assertIn("missing_x", err or "")

    def test_filter_and_defaults(self) -> None:
        # Default: registry-only — unknown / non-registry names are dropped.
        cols = filter_laggable_features(["ltp", "trading_day", "token", "spot", "foo"])
        self.assertEqual(cols, ["ltp", "spot"])
        self.assertEqual(default_selected_lag_features(cols), ["ltp", "spot"])
        # Opt-out keeps any non-meta column (legacy / tests).
        all_cols = filter_laggable_features(
            ["ltp", "trading_day", "token", "spot", "foo"],
            registry_only=False,
        )
        self.assertEqual(all_cols, ["foo", "ltp", "spot"])

    def test_search_filter(self) -> None:
        feats = ["current_ltp", "current_iv", "delta", "oi"]
        self.assertEqual(filter_features_by_search(feats, "current"), ["current_ltp", "current_iv"])
        self.assertEqual(filter_features_by_search(feats, ""), feats)

    def test_categories_and_presets(self) -> None:
        feats = [
            "ltp",
            "spot",
            "delta",
            "gamma",
            "current_iv",
            "option_oi",
            "option_day_volume",
            "atm_straddle",
        ]
        grouped = group_features_by_category(feats)
        self.assertIn("Price & Premium", grouped)
        self.assertIn("Spot & Futures", grouped)
        self.assertIn("Greeks", grouped)
        self.assertIn("ltp", grouped["Price & Premium"])
        self.assertIn("spot", grouped["Spot & Futures"])
        self.assertIn("delta", grouped["Greeks"])
        self.assertEqual(classify_feature("theta"), "Greeks")
        self.assertEqual(classify_feature("ltp"), "Price & Premium")
        self.assertEqual(classify_feature("atm_straddle"), "Chain Analytics")
        self.assertEqual(classify_feature("current_iv"), "Implied Volatility")

        price = features_for_preset("Price & Premium", feats)
        self.assertEqual(set(price), {"ltp"})
        greeks = features_for_preset("Greeks", feats)
        self.assertEqual(set(greeks), {"delta", "gamma"})
        oi = features_for_preset("Open Interest", feats)
        self.assertEqual(set(oi), {"option_oi"})
        dynamic = features_for_preset("dynamic", feats)
        self.assertIn("ltp", dynamic)
        self.assertIn("spot", dynamic)

    def test_registry_groups_mapped_to_domains(self) -> None:
        """Non-empty _REGISTRY_FEATURES groups must resolve via primary domain."""
        from chain_replay_ml.dataset_builder.feature_plugins import _REGISTRY_FEATURES

        clear_feature_category_cache()
        unmapped = [
            str(gid)
            for gid, feats in _REGISTRY_FEATURES.items()
            if feats and str(gid) not in _GROUP_TO_CATEGORY
        ]
        self.assertEqual(unmapped, [])
        self.assertEqual(classify_feature("mid_price"), "Volume & Liquidity")
        self.assertEqual(classify_feature("spot_1m_ema9"), "Historical Context")
        self.assertEqual(classify_feature("atm_iv_ce"), "Implied Volatility")
        self.assertEqual(classify_feature("oi_abs_delta_40_60_ce"), "Open Interest")
        self.assertNotEqual(classify_feature("ema9_slope"), "Engineered")
        self.assertNotEqual(classify_feature("atm_straddle"), "ATM")
        self.assertEqual(_GROUP_TO_CATEGORY["market_microstructure"], "Volume & Liquidity")
        self.assertEqual(_GROUP_TO_CATEGORY["historic_spot_ema"], "Historical Context")

    def test_filter_laggable_registry_only(self) -> None:
        cols = ["ltp", "spot", "ltp_return_5s", "timestamp", "atm_straddle_change_1m"]
        out = filter_laggable_features(cols, registry_only=True)
        self.assertIn("ltp", out)
        self.assertIn("spot", out)
        self.assertNotIn("timestamp", out)
        # Migrated/retired names must not appear even if still on Master disk.
        self.assertNotIn("ltp_return_5s", out)
        self.assertNotIn("atm_straddle_change_1m", out)
        all_cols = filter_laggable_features(cols, registry_only=False)
        self.assertIn("ltp_return_5s", all_cols)

    def test_registry_feature_count_from_master(self) -> None:
        from chain_replay_ml.dataset_builder.transformations.lag_ui import (
            registry_feature_count_from_master,
        )
        from chain_replay_ml.dataset_builder.schema_registry import (
            canonical_plugin_feature_names,
        )

        plugins = canonical_plugin_feature_names()
        # Mix of registry + stale historical.
        cols = sorted(plugins)[:10] + ["atm_straddle_change_1m", "ltp_return_5s"]
        self.assertEqual(registry_feature_count_from_master(cols), 10)
        self.assertEqual(
            registry_feature_count_from_master([]),
            len(plugins),
        )

        self.assertEqual(lag_seconds_label(30, 3), "30 sec (10 rows)")
        self.assertEqual(lag_seconds_label(60, 3), "60 sec (20 rows)")
        self.assertEqual(lag_seconds_label(90, 10), "90 sec (9 rows)")

    def test_warmup_warning(self) -> None:
        self.assertIsNone(
            lag_warmup_warning(enabled=True, lag_seconds=[30, 60], warmup_seconds=900)
        )
        warn = lag_warmup_warning(
            enabled=True, lag_seconds=[300, 900], warmup_seconds=300,
        )
        self.assertIsNotNone(warn)
        self.assertIn("exceeds warm-up", warn or "")
        self.assertIsNone(
            lag_warmup_warning(enabled=False, lag_seconds=[900], warmup_seconds=300)
        )


if __name__ == "__main__":
    unittest.main()
