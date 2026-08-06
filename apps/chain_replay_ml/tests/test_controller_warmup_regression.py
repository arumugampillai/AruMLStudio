"""Permanent regression: first valid sample for all controller-owned warmup features."""

from __future__ import annotations

import unittest

from chain_replay_ml.dataset_builder.controller_warmup_regression import (
    CONTROLLER_WARMUP_SPEC,
    WARMUP_REGRESSION_SPEC,
    assert_spec_features_are_controller_owned,
    expected_first_valid_for_feature,
    extract_feature_series_from_result,
    first_valid_sample_from_series,
    is_valid_feature_value,
    run_permanent_warmup_regression,
    validate_all_controller_warmups_from_result,
    validate_all_warmup_regressions_from_result,
    validate_warmup_row_from_result,
)
from chain_replay_ml.feature_policy.warmup_simulator import WarmupSimulationResult


class ControllerWarmupRegressionTests(unittest.TestCase):
    def test_spec_features_are_controller_owned(self) -> None:
        assert_spec_features_are_controller_owned()

    def test_permanent_first_valid_from_builder(self) -> None:
        """Mandatory CI: every CONTROLLER_WARMUP_SPEC row matches expected first sample."""
        results = run_permanent_warmup_regression()
        failures = [(label, exp, act) for label, exp, act, st in results if st != "PASS"]
        for label, expected, actual, status in [
            (label, exp, act, st) for label, exp, act, st in results
        ]:
            self.assertEqual(
                status,
                "PASS",
                f"{label}: expected first valid sample {expected}, got {actual}",
            )
        self.assertEqual(len(results), len(CONTROLLER_WARMUP_SPEC))

    def test_is_valid_feature_value(self) -> None:
        self.assertFalse(is_valid_feature_value(None))
        self.assertFalse(is_valid_feature_value("NULL"))
        self.assertTrue(is_valid_feature_value(1.23))
        self.assertTrue(is_valid_feature_value("1.23"))

    def test_first_valid_sample_from_series(self) -> None:
        series = [(1, None), (2, None), (3, 0.5)]
        self.assertEqual(first_valid_sample_from_series(series), 3)

    def test_validate_from_simulation_pass(self) -> None:
        spec = CONTROLLER_WARMUP_SPEC[0]  # EMA9
        result = WarmupSimulationResult(
            ok=True,
            feature_name=spec.feature_name,
            sampling_interval_sec=3,
            full_trace=[
                {"samples": i, "time": f"t{i}", "ts": 1000.0 + i * 3}
                for i in range(1, 12)
            ],
            all_features_lookup={
                int(1000.0 + i * 3): {
                    spec.feature_name: None if i < 9 else 1.01,
                }
                for i in range(1, 12)
            },
        )
        row = validate_warmup_row_from_result(result, spec)
        self.assertEqual(row["status"], "PASS")
        self.assertEqual(row["actual"], 9)

    def test_validate_from_simulation_fail_premature(self) -> None:
        spec = CONTROLLER_WARMUP_SPEC[0]  # EMA9
        result = WarmupSimulationResult(
            ok=True,
            feature_name=spec.feature_name,
            sampling_interval_sec=3,
            full_trace=[{"samples": 1, "time": "t1", "ts": 1000.0}],
            all_features_lookup={1000: {spec.feature_name: 1.01}},
        )
        row = validate_warmup_row_from_result(result, spec)
        self.assertEqual(row["status"], "FAIL")
        self.assertIn("Premature", row["note"])

    def test_validate_from_simulation_skip_without_lookup(self) -> None:
        result = WarmupSimulationResult(ok=True, feature_name="ltp_ema9")
        rows = validate_all_controller_warmups_from_result(result)
        self.assertTrue(all(r["status"] == "SKIP" for r in rows))

    def test_extract_feature_series_from_result(self) -> None:
        result = WarmupSimulationResult(
            ok=True,
            feature_name="opt_rv_5m",
            sampling_interval_sec=3,
            full_trace=[
                {"samples": 1, "ts": 1000.0},
                {"samples": 2, "ts": 1003.0},
            ],
            all_features_lookup={
                1000: {"opt_rv_5m": None},
                1003: {"opt_rv_5m": 0.42},
            },
        )
        series = extract_feature_series_from_result(result, "opt_rv_5m")
        self.assertEqual(len(series), 2)
        self.assertFalse(is_valid_feature_value(series[0][1]))
        self.assertTrue(is_valid_feature_value(series[1][1]))

    def test_expected_first_valid_for_feature(self) -> None:
        self.assertEqual(expected_first_valid_for_feature("ltp_ema20", 3), 20)
        self.assertEqual(expected_first_valid_for_feature("ltp_std20", 3), 20)
        self.assertEqual(expected_first_valid_for_feature("opt_rv_5m", 3), 31)
        self.assertEqual(expected_first_valid_for_feature("opt_rv_10m", 3), 61)
        self.assertEqual(expected_first_valid_for_feature("iv_zscore_1m", 3), 20)
        self.assertEqual(expected_first_valid_for_feature("iv_zscore_30m", 3), 600)
        self.assertEqual(expected_first_valid_for_feature("iv_change_1m", 3), 21)
        self.assertEqual(expected_first_valid_for_feature("ltp_to_spot_ratio_lag_30s", 3), 11)
        self.assertEqual(expected_first_valid_for_feature("dgt_reiv_pred_lag_30s", 3), 11)
        self.assertEqual(expected_first_valid_for_feature("spot_rv_ratio", 3), 61)
        self.assertEqual(expected_first_valid_for_feature("roll_iv", 3), 1)

    def test_warmup_regression_spec_covers_controller_spec(self) -> None:
        controller_names = {s.feature_name for s in CONTROLLER_WARMUP_SPEC}
        regression_names = {s.feature_name for s in WARMUP_REGRESSION_SPEC}
        self.assertTrue(controller_names <= regression_names)

    def test_validate_all_warmup_regressions_pending_without_result(self) -> None:
        rows = validate_all_warmup_regressions_from_result(None)
        self.assertEqual(len(rows), len(WARMUP_REGRESSION_SPEC))
        self.assertTrue(all(r["status"] == "PENDING" for r in rows))
        self.assertEqual(rows[0]["feature"], WARMUP_REGRESSION_SPEC[0].feature_name)


if __name__ == "__main__":
    unittest.main()
