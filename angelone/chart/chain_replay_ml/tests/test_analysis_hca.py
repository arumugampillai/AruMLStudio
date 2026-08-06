"""Tests for HCA Feature Families + Family Review decision log."""

from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from chain_replay_ml.dataset_builder.analysis_correlation import (
    run_correlation_analysis,
)
from chain_replay_ml.dataset_builder.analysis_family_review import (
    STATUS_ACCEPTED,
    load_family_review,
    load_families_with_reviews,
    review_summary,
    upsert_family_review,
)
from chain_replay_ml.dataset_builder.analysis_hca import (
    compute_hca_families,
    load_families,
    run_hca_analysis,
)
from chain_replay_ml.dataset_builder.analysis_lab_store import (
    STATUS_COMPLETED,
    ensure_analysis_run,
    module_statuses,
    register_dataset,
)


class HcaComputeTests(unittest.TestCase):
    def test_two_clear_families(self) -> None:
        # Block A: a1/a2/a3 highly correlated; Block B: b1/b2; noise alone
        rng = np.random.default_rng(0)
        n = 200
        base_a = rng.normal(size=n)
        base_b = rng.normal(size=n)
        pairs = [
            ("a1", "a2", 0.99),
            ("a1", "a3", 0.98),
            ("a2", "a3", 0.97),
            ("b1", "b2", 0.96),
            ("a1", "b1", 0.05),
            ("a1", "noise", 0.02),
            ("b1", "noise", 0.01),
            ("a2", "b1", 0.04),
            ("a2", "b2", 0.03),
            ("a2", "noise", 0.01),
            ("a3", "b1", 0.02),
            ("a3", "b2", 0.02),
            ("a3", "noise", 0.01),
            ("b2", "noise", 0.01),
            ("a1", "b2", 0.03),
        ]
        # Add remaining near-zero if needed — compute uses only listed pairs; missing → d=1
        families = compute_hca_families(pairs, distance_threshold=0.15)
        multi = [f for f in families if f["size"] >= 2]
        self.assertGreaterEqual(len(multi), 2)
        member_sets = [set(f["members"]) for f in multi]
        self.assertTrue(any({"a1", "a2", "a3"} <= s for s in member_sets))
        self.assertTrue(any({"b1", "b2"} <= s for s in member_sets))
        # Candidates exist but no final representative field
        for f in multi:
            self.assertIn("candidates", f)
            self.assertTrue(f["candidates"])
            self.assertNotIn("selected_representative", f)


class HcaIntegrationTests(unittest.TestCase):
    def test_run_hca_and_review_log(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            n = 120
            rng = np.random.default_rng(3)
            spot = pd.Series(np.linspace(100, 140, n)) + rng.normal(0, 0.1, n)
            df = pd.DataFrame(
                {
                    "trading_day": ["2026-01-02"] * n,
                    "spot": spot,
                    "ltp": spot * 1.0001,
                    "mid_price": spot * 1.00005,
                    "current_iv": np.linspace(0.1, 0.2, n),
                    "iv_ema": np.linspace(0.11, 0.21, n),
                    "noise_feat": rng.normal(size=n),
                    "future_ltp_5m": spot.shift(-1).bfill(),
                }
            )
            path = os.path.join(tmp, "hca_demo.parquet")
            df.to_parquet(path, index=False)
            ds = register_dataset(tmp, path, name="hca_demo")
            run = ensure_analysis_run(tmp, ds["dataset_id"])
            run_id = run["run_id"]

            run_correlation_analysis(tmp, run_id, ds)
            out = run_hca_analysis(tmp, run_id)
            self.assertGreater(out["n_families"], 0)
            statuses = {
                r["module_id"]: r["status"] for r in module_statuses(tmp, run_id)
            }
            self.assertEqual(statuses.get("hca"), STATUS_COMPLETED)

            families = load_families(tmp, run_id, min_size=2)
            self.assertTrue(families)
            fam = families[0]
            fid = str(fam["family_id"])
            members = list(fam["members"])
            self.assertGreaterEqual(len(members), 2)

            # Researcher picks a member (possibly not top candidate)
            pick = members[-1]
            saved = upsert_family_review(
                tmp,
                run_id,
                fid,
                selected_representative=pick,
                reason_code="Interpretability",
                reason_text="Easier to explain than top candidate.",
                status=STATUS_ACCEPTED,
            )
            self.assertEqual(saved.get("selected_representative"), pick)
            self.assertEqual(saved.get("status"), STATUS_ACCEPTED)

            loaded = load_family_review(tmp, run_id, fid)
            assert loaded is not None
            self.assertEqual(loaded["reason_code"], "Interpretability")

            from chain_replay_ml.dataset_builder.analysis_family_review import (
                FILTER_ALL,
            )

            enriched = load_families_with_reviews(
                tmp, run_id, min_size=2, status_filter=FILTER_ALL
            )
            row = next(r for r in enriched if r["family_id"] == fid)
            self.assertEqual(row["selected_representative"], pick)

            summary = review_summary(tmp, run_id, min_size=2)
            self.assertGreaterEqual(summary["counts"][STATUS_ACCEPTED], 1)

            # Reject non-member
            with self.assertRaises(ValueError):
                upsert_family_review(
                    tmp,
                    run_id,
                    fid,
                    selected_representative="not_a_member",
                    reason_code="Stability",
                    status=STATUS_ACCEPTED,
                )


if __name__ == "__main__":
    unittest.main()
