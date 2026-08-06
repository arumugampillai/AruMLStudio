"""Family migration parity tests — Phase-1 ltp_to_spot_ratio Lag."""

from __future__ import annotations

import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.feature_migration import (
    RETIRED_FEATURES,
    get_migration_family,
    horizons_compatible_with_interval,
    is_pipeline_owned,
    is_retired,
    migration_status_table,
)
from chain_replay_ml.dataset_builder.feature_ownership import (
    OWNERSHIP_PIPELINE_OWNED,
    ownership_of,
)
from chain_replay_ml.dataset_builder.migration_parity import compare_lag_family_parity
from chain_replay_ml.dataset_builder.transformations import run_transformation_pipeline
from chain_replay_ml.dataset_builder.transformations.base import TransformContext
from chain_replay_ml.dataset_builder.transformations.lag import lag_column_name


def _uniform_ratio_frame(*, n: int = 400, interval: float = 3.0) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "trading_day": "2026-07-23",
            "token": "T1",
            "timestamp": 1_000_000.0 + i * interval,
            "ltp_to_spot_ratio": 0.01 + i * 0.0001,
        })
    return pd.DataFrame(rows)


class LagNamingTests(unittest.TestCase):
    def test_master_compatible_suffix(self) -> None:
        self.assertEqual(
            lag_column_name("ltp_to_spot_ratio", 60, suffix="1m"),
            "ltp_to_spot_ratio_lag_1m",
        )
        self.assertEqual(lag_column_name("ltp", 60), "ltp_lag_60s")


class LtpToSpotRatioFamilyMigrationTests(unittest.TestCase):
    def test_10s_retired(self) -> None:
        self.assertIn("ltp_to_spot_ratio_lag_10s", RETIRED_FEATURES)
        self.assertIn("ltp_to_spot_ratio_change_10s", RETIRED_FEATURES)
        self.assertTrue(is_retired("ltp_to_spot_ratio_lag_10s"))
        fam = get_migration_family("ltp_to_spot_ratio")
        self.assertNotIn("10s", [s for s, _ in fam.horizons])
        self.assertNotIn("ltp_to_spot_ratio_lag_10s", fam.features)
        change = get_migration_family("ltp_to_spot_ratio_change")
        self.assertTrue(change.pipeline_owned)
        self.assertNotIn("ltp_to_spot_ratio_change_10s", change.features)

    def test_family_pipeline_owned(self) -> None:
        fam = get_migration_family("ltp_to_spot_ratio")
        self.assertEqual(fam.parity, "passed")
        self.assertTrue(fam.pipeline_owned)
        self.assertTrue(fam.removed_from_master)
        self.assertTrue(fam.removed_from_registry)
        for name in fam.features:
            self.assertTrue(is_pipeline_owned(name))
            self.assertEqual(ownership_of(name), OWNERSHIP_PIPELINE_OWNED)

    def test_horizons_all_compatible_at_3s(self) -> None:
        fam = get_migration_family("ltp_to_spot_ratio")
        ok, bad = horizons_compatible_with_interval(fam.horizons, 3.0)
        self.assertEqual(bad, [])
        self.assertEqual(
            [s for s, _ in ok],
            ["30s", "1m", "3m", "5m", "15m"],
        )

    def test_parity_passes_at_default_3s(self) -> None:
        frame = _uniform_ratio_frame()
        result = compare_lag_family_parity(
            frame,
            family_id="ltp_to_spot_ratio",
            sample_interval_sec=3.0,
            update_status=True,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["detail"]["blocked_horizons"], [])
        for col, info in result["detail"]["columns"].items():
            self.assertTrue(info["pass"], msg=f"{col}: {info}")
        fam = get_migration_family("ltp_to_spot_ratio")
        self.assertTrue(fam.pipeline_owned)
        self.assertTrue(fam.removed_from_master)

    def test_pipeline_emits_master_names(self) -> None:
        frame = _uniform_ratio_frame()
        fam = get_migration_family("ltp_to_spot_ratio")
        cfg = fam.pipeline_config(sample_interval_sec=3.0)
        ctx = TransformContext(config=cfg, sample_interval_sec=3.0)
        out = run_transformation_pipeline(
            frame[["trading_day", "token", "ltp_to_spot_ratio"]],
            cfg,
            context=ctx,
        )
        self.assertIn("ltp_to_spot_ratio_lag_1m", out.frame.columns)
        self.assertIn("ltp_to_spot_ratio_lag_30s", out.frame.columns)
        self.assertNotIn("ltp_to_spot_ratio_lag_10s", out.frame.columns)
        self.assertNotIn("ltp_to_spot_ratio_lag_60s", out.frame.columns)

    def test_status_table_has_families(self) -> None:
        rows = migration_status_table()
        ids = {r["feature_family"] for r in rows}
        self.assertIn("ltp_to_spot_ratio", ids)
        self.assertIn("ltp_to_spot_ratio_change", ids)
        self.assertIn("oi", ids)
        self.assertIn("volume", ids)
        self.assertIn("dgt_reiv_pred_lag", ids)
        self.assertIn("ltp_return", ids)
        pilot = next(r for r in rows if r["feature_family"] == "ltp_to_spot_ratio")
        self.assertTrue(pilot["pipeline_owned"])
        self.assertTrue(pilot["removed_from_master"])
        self.assertTrue(pilot["removed_from_registry"])
        oi = next(r for r in rows if r["feature_family"] == "oi")
        self.assertTrue(oi["pipeline_owned"])
        self.assertEqual(oi["parity"], "passed")


if __name__ == "__main__":
    unittest.main()
