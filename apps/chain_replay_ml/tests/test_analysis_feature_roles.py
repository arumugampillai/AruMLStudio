"""Tests for Research Lab feature_role classification."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from chain_replay_ml.dataset_builder.analysis_correlation import compute_correlation_frame
from chain_replay_ml.dataset_builder.analysis_feature_profiles import (
    build_feature_profiles,
    load_feature_profile,
    load_feature_scorecard,
)
from chain_replay_ml.dataset_builder.analysis_feature_roles import (
    ROLE_LABEL,
    ROLE_METADATA,
    ROLE_PREDICTOR,
    ROLE_TARGET,
    classify_feature_role,
    predictor_columns,
    role_banner,
)
from chain_replay_ml.dataset_builder.analysis_lab_store import (
    ensure_analysis_run,
    register_dataset,
)
from chain_replay_ml.dataset_builder.analysis_mutual_information import (
    compute_mutual_information,
)


class FeatureRoleTests(unittest.TestCase):
    def test_classify_roles(self) -> None:
        self.assertEqual(classify_feature_role("spot"), ROLE_PREDICTOR)
        self.assertEqual(classify_feature_role("future_ltp_5m"), ROLE_TARGET)
        self.assertEqual(classify_feature_role("label_up_2pct_5m"), ROLE_LABEL)
        self.assertEqual(classify_feature_role("trading_day"), ROLE_METADATA)
        self.assertEqual(classify_feature_role("master_row_id"), ROLE_METADATA)
        self.assertEqual(classify_feature_role("market"), ROLE_METADATA)
        self.assertEqual(
            classify_feature_role(
                "custom_target",
                sidecar={"prediction_target_columns": ["custom_target"]},
            ),
            ROLE_TARGET,
        )
        title, body = role_banner(ROLE_TARGET) or ("", "")
        self.assertIn("Target", title)
        self.assertIn("prediction target", body)

    def test_scorecard_predictors_only_and_explorer_keeps_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            n = 80
            df = pd.DataFrame(
                {
                    "trading_day": ["2026-07-24"] * n,
                    "spot": np.linspace(100, 120, n),
                    "current_iv": np.linspace(0.1, 0.2, n),
                    "future_ltp_5m": np.linspace(101, 121, n),
                    "label_up_5m": [i % 2 for i in range(n)],
                }
            )
            path = os.path.join(tmp, "roles.parquet")
            df.to_parquet(path, index=False)
            with open(os.path.join(tmp, "roles.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "prediction_target_columns": [
                            "future_ltp_5m",
                            "label_up_5m",
                        ]
                    },
                    f,
                )
            register_dataset(tmp, path, name="roles")
            run = ensure_analysis_run(tmp, "roles")
            ds = {"path": path, "dataset_id": "roles", "name": "roles"}

            corr, cols = compute_correlation_frame(path)
            self.assertIn("spot", cols)
            self.assertNotIn("future_ltp_5m", cols)
            self.assertNotIn("label_up_5m", cols)
            self.assertNotIn("trading_day", cols)

            mi = compute_mutual_information(path, "future_ltp_5m", max_rows=80)
            mi_feats = {r["feature"] for r in mi}
            self.assertIn("spot", mi_feats)
            self.assertNotIn("future_ltp_5m", mi_feats)
            self.assertNotIn("label_up_5m", mi_feats)

            build_feature_profiles(tmp, run["run_id"], ds)
            card = load_feature_scorecard(tmp, run["run_id"])
            names = {r["feature_name"] for r in card}
            self.assertIn("spot", names)
            self.assertIn("current_iv", names)
            self.assertNotIn("future_ltp_5m", names)
            self.assertNotIn("label_up_5m", names)
            self.assertNotIn("trading_day", names)

            tgt = load_feature_profile(tmp, run["run_id"], "future_ltp_5m")
            self.assertIsNotNone(tgt)
            assert tgt is not None
            self.assertEqual(tgt.get("feature_role"), ROLE_TARGET)
            self.assertEqual(tgt.get("recommendation"), "Excluded")

            lab = load_feature_profile(tmp, run["run_id"], "label_up_5m")
            self.assertIsNotNone(lab)
            assert lab is not None
            self.assertEqual(lab.get("feature_role"), ROLE_LABEL)

            preds = predictor_columns(list(df.columns), sidecar={
                "prediction_target_columns": ["future_ltp_5m", "label_up_5m"]
            })
            self.assertEqual(set(preds), {"spot", "current_iv"})


if __name__ == "__main__":
    unittest.main()
