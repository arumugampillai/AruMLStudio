"""Tests for dataset build feature readiness enforcement."""

from __future__ import annotations

import unittest

from chain_replay_ml.feature_policy.build_readiness import (
    build_feature_readiness_manifest,
    enforce_readiness_on_rows,
    validate_readiness_compliance,
)
from chain_replay_ml.feature_policy.engine import FeaturePolicyEngine
from chain_replay_ml.feature_policy.registry import load_feature_policy_registry


class BuildReadinessTests(unittest.TestCase):
    def test_manifest_fields(self) -> None:
        doc = build_feature_readiness_manifest(gap_max_sec=30.0)
        self.assertEqual(doc["version"], "1")
        self.assertEqual(doc["gap_max_sec"], 30.0)
        self.assertTrue(doc["rolling_enforcement"])
        self.assertEqual(doc["not_ready_output"], "NULL")
        self.assertTrue(doc["dependency_propagation"])

    def test_immature_ema_ratio_nulled(self) -> None:
        feat = "ltp_ema200"
        rows = [
            {
                "trading_day": "2025-01-01",
                "timestamp": 1000.0,
                feat: 1.05,
            },
        ]
        enforce_readiness_on_rows(
            rows,
            feature_names=[feat],
            sampling_interval_sec=10.0,
            gap_max_sec=30.0,
        )
        self.assertIsNone(rows[0][feat])

    def test_gap_resets_to_null(self) -> None:
        feat = "ltp_ema200"
        rows = [
            {"trading_day": "d", "timestamp": 0.0, feat: 1.0},
            {"trading_day": "d", "timestamp": 10.0, feat: 1.0},
            {"trading_day": "d", "timestamp": 50.0, feat: 1.0},
        ]
        enforce_readiness_on_rows(
            rows,
            feature_names=[feat],
            sampling_interval_sec=10.0,
            gap_max_sec=30.0,
        )
        for row in rows:
            self.assertIsNone(row[feat])

    def test_compliance_zero_after_enforce(self) -> None:
        feat = "ltp_ema20"
        rows = [
            {"trading_day": "d", "timestamp": float(i * 10), feat: 1.0 + i * 0.01}
            for i in range(25)
        ]
        enforce_readiness_on_rows(
            rows,
            feature_names=[feat],
            sampling_interval_sec=10.0,
            gap_max_sec=30.0,
        )
        report = validate_readiness_compliance(
            rows,
            feature_names=[feat],
            sampling_interval_sec=10.0,
            gap_max_sec=30.0,
        )
        self.assertEqual(report["violations"], 0)
        self.assertEqual(report["policy_compliance_pct"], 100.0)

    def test_ready_after_warmup_writes_value(self) -> None:
        feat = "ltp_ema20"
        reg = load_feature_policy_registry(feature_names=[feat])
        eng = FeaturePolicyEngine(reg, sampling_interval_sec=10.0, gap_max_sec=30.0)
        eng.on_session_start()
        for i in range(25):
            eng.on_sample(float(i * 10))
        self.assertTrue(eng.is_ready(feat))
        self.assertEqual(eng.value_or_null(feat, 1.23), 1.23)


if __name__ == "__main__":
    unittest.main()
