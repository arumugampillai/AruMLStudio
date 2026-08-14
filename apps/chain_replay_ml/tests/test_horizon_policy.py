"""Tests for shared interval-specific horizon policy."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.transformations.horizon_policy import (
    default_horizons_for_interval,
    get_horizon_policy,
    load_horizon_policy,
    warmup_seconds_for_interval,
)
from chain_replay_ml.dataset_builder.transformations.lag_ui import (
    DEFAULT_LAG_SECONDS,
    default_lag_seconds_for_interval,
    resolve_warmup_seconds,
)
from chain_replay_ml.dataset_builder.transformations.time_shift import (
    LagConfigError,
    parse_features_and_horizons,
)


class HorizonPolicyTests(unittest.TestCase):
    def test_default_profiles(self) -> None:
        h3 = default_horizons_for_interval(3)
        self.assertEqual(h3[0], 30)
        self.assertEqual(h3[-1], 900)
        self.assertEqual(h3, tuple(range(30, 901, 30)))
        self.assertEqual(warmup_seconds_for_interval(3), 900.0)

        h6 = default_horizons_for_interval(6)
        self.assertEqual(h6[0], 60)
        self.assertEqual(h6[-1], 1800)
        self.assertEqual(h6, tuple(range(60, 1801, 60)))
        self.assertEqual(warmup_seconds_for_interval(6), 1800.0)

        h9 = default_horizons_for_interval(9)
        self.assertEqual(h9[0], 90)
        self.assertEqual(h9[-1], 2700)
        self.assertEqual(h9, tuple(range(90, 2701, 90)))
        self.assertEqual(warmup_seconds_for_interval(9), 2700.0)

    def test_lag_ui_defaults_follow_3s_policy(self) -> None:
        self.assertEqual(DEFAULT_LAG_SECONDS, default_horizons_for_interval(3))
        self.assertEqual(
            default_lag_seconds_for_interval(6),
            default_horizons_for_interval(6),
        )

    def test_parse_features_derives_horizons_when_omitted(self) -> None:
        features, offsets = parse_features_and_horizons(
            transform_name="Lag Transformation",
            params={"features": ["ltp"]},
            sample_interval_sec=3.0,
        )
        self.assertEqual(features, ["ltp"])
        self.assertEqual([int(s) for s, _, _, _ in offsets], list(range(30, 901, 30)))

    def test_parse_features_empty_list_still_errors(self) -> None:
        with self.assertRaises(LagConfigError):
            parse_features_and_horizons(
                transform_name="Lag Transformation",
                params={"features": ["ltp"], "lag_seconds": []},
                sample_interval_sec=3.0,
            )

    def test_parse_preserves_duplicate_seconds_with_distinct_columns(self) -> None:
        features, offsets = parse_features_and_horizons(
            transform_name="Return Transformation",
            params={
                "features": ["option_oi"],
                "horizons": [
                    {"seconds": 60.0, "column": "oi_change_1m"},
                    {"seconds": 60.0, "column": "oi_change_pct_1m"},
                ],
            },
            sample_interval_sec=3.0,
        )
        self.assertEqual(features, ["option_oi"])
        cols = [column for _, _, _, column in offsets]
        self.assertEqual(cols, ["oi_change_1m", "oi_change_pct_1m"])

    def test_custom_document(self) -> None:
        policies = load_horizon_policy(
            document={
                "intervals": {
                    "3": {
                        "min_horizon_sec": 30,
                        "step_sec": 30,
                        "max_horizon_sec": 90,
                        "warmup_sec": 90,
                    }
                }
            }
        )
        self.assertEqual(
            get_horizon_policy(3, policies=policies).horizons(),
            (30, 60, 90),
        )

    def test_unknown_interval_fails(self) -> None:
        with self.assertRaises(LagConfigError):
            default_horizons_for_interval(12)

    def test_resolve_warmup_prefers_policy_when_meta_missing(self) -> None:
        self.assertEqual(
            resolve_warmup_seconds({}, sample_interval_sec=6),
            1800.0,
        )


if __name__ == "__main__":
    unittest.main()
