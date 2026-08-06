"""Unit tests for Why-driven fold diagnostics."""

from __future__ import annotations

import unittest

import pandas as pd

from chain_replay_ml.training.fold_diagnostics import (
    build_feature_distribution_shift,
    build_fold_context,
    build_fold_outlier_scores,
    build_why_explanations,
)


def _day_frame(
    *,
    day: str,
    n: int = 40,
    spot0: float = 24800.0,
    spot_move: float = 100.0,
    ltp0: float = 20.0,
    ltp_hi: float = 45.0,
    iv: float = 0.4,
    dte: float = 2.0,
    is_expiry: int = 0,
    hour_offset: float = 11.0,
) -> pd.DataFrame:
    spots = [spot0 + spot_move * (i / max(n - 1, 1)) for i in range(n)]
    ltps = [ltp0 + (ltp_hi - ltp0) * (i / max(n - 1, 1)) for i in range(n)]
    # Fake epoch around IST session
    base_ts = 1_700_000_000.0 + hour_offset * 3600
    return pd.DataFrame({
        "trading_day": [day] * n,
        "timestamp": [base_ts + i * 60 for i in range(n)],
        "spot": spots,
        "ltp": ltps,
        "current_iv": [iv + 0.01 * (i / n) for i in range(n)],
        "iv_zscore_5m": [iv] * n,
        "delta": [0.25 + 0.002 * i for i in range(n)],
        "volume": [1000 + 10 * i for i in range(n)],
        "oi": [50000 + 100 * i for i in range(n)],
        "days_to_expiry": [dte] * n,
        "is_expiry_day": [is_expiry] * n,
    })


class FoldContextTests(unittest.TestCase):
    def test_context_has_regime_and_ranges(self) -> None:
        df = _day_frame(day="2026-06-25", spot_move=280, ltp0=8, ltp_hi=54, iv=18.6, dte=2)
        full = pd.concat([_day_frame(day="2026-06-24", spot0=24700), df], ignore_index=True)
        ctx = build_fold_context(df, fold=10, label="Fold 10", trading_days=["2026-06-25"], full_df=full)
        self.assertTrue(ctx["available"])
        self.assertIn("Trending", ctx["market_regime"] or "")
        labels = {r["label"]: r["value"] for r in ctx["rows"]}
        self.assertEqual(labels["Validation day(s)"], "2026-06-25")
        self.assertIn("→", labels["Spot range"])
        self.assertEqual(labels["Expiry distance"], "T-2")
        self.assertEqual(labels["Sample count"], "40")


class DistributionShiftTests(unittest.TestCase):
    def test_flags_huge_iv_shift(self) -> None:
        a = _day_frame(day="2026-06-01", iv=0.3, ltp0=18, ltp_hi=22, spot_move=40)
        b = _day_frame(day="2026-06-25", iv=2.8, ltp0=30, ltp_hi=50, spot_move=150)
        b["volume"] = b["volume"] * 4
        dist = build_feature_distribution_shift(
            a, b, ["iv_zscore_5m", "ltp", "volume", "delta"],
            label_a="Fold 6", label_b="Fold 10",
        )
        self.assertTrue(dist["available"])
        by_feat = {r["feature"]: r for r in dist["rows"]}
        self.assertIn("iv_zscore_5m", by_feat)
        self.assertEqual(by_feat["iv_zscore_5m"]["severity"], "huge")
        self.assertTrue(dist["largest_shifts"])


class OutlierScoreTests(unittest.TestCase):
    def test_ranks_unusual_features(self) -> None:
        train = _day_frame(day="2026-05-01", iv=0.3, n=80)
        train = pd.concat([train, _day_frame(day="2026-05-02", iv=0.35, n=80)], ignore_index=True)
        fold = _day_frame(day="2026-06-25", iv=2.5, spot_move=200, ltp0=40, ltp_hi=70)
        out = build_fold_outlier_scores(
            fold, train, ["iv_zscore_5m", "ltp", "spot", "delta"],
            fold=10, label="Fold 10",
        )
        self.assertTrue(out["available"])
        self.assertEqual(out["rows"][0]["feature"], "iv_zscore_5m")
        self.assertGreater(abs(out["rows"][0]["z_score"]), 2.0)


class WhyNarrativeTests(unittest.TestCase):
    def test_why_bullets_answer_without_tables(self) -> None:
        ctx_a = build_fold_context(
            _day_frame(day="2026-06-01", iv=0.3, spot_move=40, ltp0=15, ltp_hi=25),
            fold=6, label="Fold 6", trading_days=["2026-06-01"],
        )
        ctx_b = build_fold_context(
            _day_frame(day="2026-06-25", iv=2.0, spot_move=300, ltp0=20, ltp_hi=80, dte=0, is_expiry=1, hour_offset=14),
            fold=10, label="Fold 10", trading_days=["2026-06-25"],
        )
        a = _day_frame(day="2026-06-01", iv=0.3)
        b = _day_frame(day="2026-06-25", iv=2.0, spot_move=300, ltp0=20, ltp_hi=80)
        dist = build_feature_distribution_shift(
            a, b, ["iv_zscore_5m", "ltp"], label_a="Fold 6", label_b="Fold 10",
        )
        out_b = build_fold_outlier_scores(
            b, a, ["iv_zscore_5m", "ltp"], fold=10, label="Fold 10",
        )
        why = build_why_explanations(
            label_a="Fold 6",
            label_b="Fold 10",
            metrics_a={"mae": 1.1, "directional_accuracy_pct": 96.0},
            metrics_b={"mae": 2.75, "directional_accuracy_pct": 78.0},
            context_a=ctx_a,
            context_b=ctx_b,
            distribution=dist,
            outliers_a={"available": False, "rows": []},
            outliers_b=out_b,
            error_histograms={
                "available": True,
                "fold_a": {"tail_gt5_pct": 16.0},
                "fold_b": {"tail_gt5_pct": 58.0},
            },
        )
        self.assertEqual(why["worse_label"], "Fold 10")
        joined = " ".join(why["bullets"])
        self.assertIn("Fold 10 differs from Fold 6", why["headline"])
        self.assertTrue(
            any("IV" in b or "Spot" in b or "Premium" in b or "Expiry" in b for b in why["bullets"]),
            joined,
        )
        self.assertTrue(any("Direction accuracy fell" in b for b in why["bullets"]), joined)
        self.assertTrue(why["metric_cards"])
        mae_card = why["metric_cards"][0]
        self.assertTrue(mae_card["why"], "MAE card should carry Why bullets")
        self.assertIn("↑", mae_card["value_display"])


if __name__ == "__main__":
    unittest.main()
