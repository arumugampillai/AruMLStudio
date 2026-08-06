from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from chain_replay_ml.training.prediction_packages import (
    PROBABILITY_LADDER,
    PROBABILITY_OUTPUT_COLUMNS,
    build_prediction_package,
    discover_prediction_package_members,
    is_package_anchor_target,
    package_members_summary,
    package_registry_rows,
    probability_ladder_slot,
    target_horizon,
)
from live_inference.pipeline import _build_prediction_package_results
from live_inference.snapshot import PredictionResult


def _row(
    name: str,
    target: str,
    *,
    dataset: str = "MS_same",
    trained_at: str = "2026-07-24T10:00:00+00:00",
    package_anchor: str | None = None,
) -> dict:
    row = {
        "model_name": name,
        "target": target,
        "dataset": dataset,
        "trained_at": trained_at,
        "status": "ready",
        "prediction_type": (
            "binary" if str(target).startswith("label_up_") else "regression"
        ),
    }
    if package_anchor:
        row["package_anchor"] = package_anchor
    return row


class TestPredictionPackages(unittest.TestCase):
    def test_target_horizon_and_slots(self) -> None:
        self.assertEqual(target_horizon("future_ltp_5m"), "5m")
        self.assertEqual(target_horizon("label_up_gt6pct_5m"), "5m")
        self.assertIsNone(target_horizon("target_reached"))
        self.assertEqual(
            probability_ladder_slot("label_up_4pct_5m")["label"],
            "+4%",
        )

    def test_partial_package_has_explicit_missing_slots(self) -> None:
        anchor = _row("reg", "future_ltp_5m")
        classifiers = [
            _row("c2", "label_up_2pct_5m"),
            _row("c3", "label_up_3pct_5m"),
            _row("c5", "label_up_5pct_5m"),
            _row("cgt6", "label_up_gt6pct_5m"),
        ]
        package = build_prediction_package(anchor, classifiers)
        classification = package["classification"]
        self.assertEqual(classification["available"], 4)
        self.assertEqual(classification["total"], 6)
        self.assertFalse(classification["complete"])
        by_label = {row["label"]: row for row in classification["members"]}
        self.assertTrue(by_label["+2%"]["available"])
        self.assertFalse(by_label["+4%"]["available"])
        self.assertFalse(by_label["+6%"]["available"])
        self.assertEqual(len(by_label), len(PROBABILITY_LADDER))

    def test_registry_assigns_classifiers_by_temporal_ownership(self) -> None:
        older = _row(
            "reg_old",
            "future_ltp_5m",
            trained_at="2026-07-23T10:00:00+00:00",
        )
        newer = _row(
            "reg_new",
            "future_ltp_5m",
            trained_at="2026-07-24T10:00:00+00:00",
        )
        c2_old = _row(
            "c2_old",
            "label_up_2pct_5m",
            trained_at="2026-07-23T11:00:00+00:00",
        )
        c2_new = _row(
            "c2_new",
            "label_up_2pct_5m",
            trained_at="2026-07-24T11:00:00+00:00",
        )
        orphan = _row(
            "orphan",
            "label_up_3pct_5m",
            dataset="MS_without_regression",
        )

        visible = package_registry_rows([older, newer, c2_old, c2_new, orphan])
        names = [row["model_name"] for row in visible]
        self.assertEqual(names, ["reg_old", "reg_new", "orphan"])

        old_pkg = visible[0]["prediction_package"]
        new_pkg = visible[1]["prediction_package"]
        old_member = next(
            row
            for row in old_pkg["classification"]["members"]
            if row["label"] == "+2%"
        )
        new_member = next(
            row
            for row in new_pkg["classification"]["members"]
            if row["label"] == "+2%"
        )
        self.assertEqual(old_member["model_name"], "c2_old")
        self.assertEqual(new_member["model_name"], "c2_new")
        self.assertEqual(visible[0]["package_badge"], "Reg + 1/6 cls")
        self.assertEqual(visible[1]["package_badge"], "Reg + 1/6 cls")

    def test_newer_regression_does_not_steal_older_classifiers(self) -> None:
        """Reproduce the registry bug: later Future_LTP on same dataset must
        not inherit classifiers trained under an earlier package."""
        older = _row(
            "Future_LTP_5m_WF_262f_XGB_0507_25",
            "future_ltp_5m",
            dataset="MS_262f_3s_0053",
            trained_at="2026-07-24T23:43:10+00:00",
        )
        newer = _row(
            "Future_LTP_5m_WF_244f_XGB_0707_25",
            "future_ltp_5m",
            dataset="MS_262f_3s_0053",
            trained_at="2026-07-25T01:41:48+00:00",
        )
        classifiers = [
            _row(
                "label_up_2pct_5m_WF_122f_XGB_0541_25",
                "label_up_2pct_5m",
                dataset="MS_262f_3s_0053",
                trained_at="2026-07-25T00:12:07+00:00",
            ),
            _row(
                "label_up_3pct_5m_WF_122f_XGB_0559_25",
                "label_up_3pct_5m",
                dataset="MS_262f_3s_0053",
                trained_at="2026-07-25T00:30:37+00:00",
            ),
            _row(
                "label_up_4pct_5m_WF_122f_XGB_0602_25",
                "label_up_4pct_5m",
                dataset="MS_262f_3s_0053",
                trained_at="2026-07-25T00:33:16+00:00",
            ),
        ]
        visible = package_registry_rows([newer, older, *classifiers])
        by_name = {row["model_name"]: row for row in visible}
        self.assertEqual(
            by_name[older["model_name"]]["package_badge"], "Reg + 3/6 cls"
        )
        self.assertEqual(
            by_name[newer["model_name"]]["package_badge"], "Reg + 0/6 cls"
        )
        newer_members = by_name[newer["model_name"]]["prediction_package"][
            "classification"
        ]["members"]
        self.assertTrue(all(not m["available"] for m in newer_members))

    def test_explicit_package_anchor_overrides_temporal_rule(self) -> None:
        older = _row(
            "reg_old",
            "future_ltp_5m",
            trained_at="2026-07-23T10:00:00+00:00",
        )
        newer = _row(
            "reg_new",
            "future_ltp_5m",
            trained_at="2026-07-24T10:00:00+00:00",
        )
        # Trained after newer, but explicitly stamped to older.
        linked = _row(
            "c2_linked",
            "label_up_2pct_5m",
            trained_at="2026-07-24T12:00:00+00:00",
            package_anchor="reg_old",
        )
        visible = package_registry_rows([older, newer, linked])
        by_name = {row["model_name"]: row for row in visible}
        old_member = next(
            m
            for m in by_name["reg_old"]["prediction_package"]["classification"][
                "members"
            ]
            if m["label"] == "+2%"
        )
        new_member = next(
            m
            for m in by_name["reg_new"]["prediction_package"]["classification"][
                "members"
            ]
            if m["label"] == "+2%"
        )
        self.assertEqual(old_member["model_name"], "c2_linked")
        self.assertFalse(new_member["available"])

    def test_different_dataset_or_horizon_never_links(self) -> None:
        anchor = _row("reg", "future_ltp_5m", dataset="MS_A")
        classifiers = [
            _row("wrong_ds", "label_up_2pct_5m", dataset="MS_B"),
            _row("wrong_horizon", "label_up_2pct_3m", dataset="MS_A"),
        ]
        package = build_prediction_package(anchor, classifiers)
        self.assertEqual(package["classification"]["available"], 0)
        visible = package_registry_rows(
            [_row("reg_3m", "future_ltp_3m"), *classifiers]
        )
        self.assertNotIn("prediction_package", visible[0])

    def test_prediction_output_contains_future_ltp_and_partial_ladder(self) -> None:
        specs = [
            {
                "model_name": "reg",
                "target": "future_ltp_5m",
                "dataset": "MS_A",
                "tier": "regression",
            },
            {
                "model_name": "c2",
                "target": "label_up_2pct_5m",
                "dataset": "MS_A",
                "tier": "classification",
                "probability_ladder_slot": probability_ladder_slot(
                    "label_up_2pct_5m"
                ),
            },
        ]
        common = {
            "mae": None,
            "rmse": None,
            "prediction_time_ms": 1.0,
            "status": "ok",
            "feature_version": "v1",
        }
        results = {
            "reg": PredictionResult(
                prediction=108.4,
                model_id="reg",
                target="future_ltp_5m",
                tier="regression",
                **common,
            ),
            "c2": PredictionResult(
                prediction=0.97,
                model_id="c2",
                target="label_up_2pct_5m",
                tier="classification",
                **common,
            ),
        }
        packages = _build_prediction_package_results(specs, results)
        self.assertEqual(packages[0]["regression"]["future_ltp"], 108.4)
        self.assertEqual(packages[0]["classification"]["available"], 1)
        ladder = {row["label"]: row for row in packages[0]["probability_ladder"]}
        self.assertEqual(ladder["+2%"]["probability_pct"], 97.0)
        self.assertEqual(ladder["+3%"]["status"], "missing")


def _write_model_package(
    data_dir: str,
    name: str,
    *,
    target: str,
    dataset: str,
    features: list[str],
    trained_at: str,
) -> None:
    base = os.path.join(data_dir, "models", name)
    os.makedirs(base, exist_ok=True)
    with open(os.path.join(base, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "target": target,
                "dataset": dataset,
                "prediction_type": "binary",
                "algorithm": "xgboost",
                "features": features,
            },
            fh,
        )
    with open(os.path.join(base, "registry.json"), "w", encoding="utf-8") as fh:
        json.dump({"trained_at": trained_at}, fh)
    with open(os.path.join(base, "model.ubj"), "wb") as fh:
        fh.write(b"fake")


class TestPackageMemberContract(unittest.TestCase):
    def test_ladder_slots_carry_full_member_contract(self) -> None:
        for slot in PROBABILITY_LADDER:
            self.assertTrue(str(slot["role"]).startswith("probability_"))
            self.assertEqual(slot["prediction_type"], "probability")
            self.assertTrue(str(slot["output_column"]).startswith("pred_prob_"))
        self.assertEqual(len(set(PROBABILITY_OUTPUT_COLUMNS)), len(PROBABILITY_LADDER))

    def test_probability_columns_are_in_prediction_schema(self) -> None:
        from chain_replay_ml.model_lab.prediction_schema import CORE_COLUMN_NAMES

        for column in PROBABILITY_OUTPUT_COLUMNS:
            self.assertIn(column, CORE_COLUMN_NAMES)

    def test_discover_members_from_disk(self) -> None:
        with tempfile.TemporaryDirectory() as data_dir:
            _write_model_package(
                data_dir,
                "cls2_old",
                target="label_up_2pct_5m",
                dataset="MS_A",
                features=["f1", "f2"],
                trained_at="2026-07-23T10:00:00+00:00",
            )
            _write_model_package(
                data_dir,
                "cls2_new",
                target="label_up_2pct_5m",
                dataset="MS_A",
                features=["f1", "f3"],
                trained_at="2026-07-24T10:00:00+00:00",
            )
            _write_model_package(
                data_dir,
                "cls4_wrong_ds",
                target="label_up_4pct_5m",
                dataset="MS_B",
                features=["f1"],
                trained_at="2026-07-24T10:00:00+00:00",
            )
            _write_model_package(
                data_dir,
                "cls5",
                target="label_up_5pct_5m",
                dataset="MS_A",
                features=["f9"],
                trained_at="2026-07-24T10:00:00+00:00",
            )

            members = discover_prediction_package_members(
                data_dir,
                dataset="MS_A",
                anchor_target="future_ltp_5m",
                anchor_model_name="Future_LTP_anchor",
                anchor_trained_at="2026-07-22T10:00:00+00:00",
            )
            self.assertEqual(len(members), len(PROBABILITY_LADDER))
            by_key = {m["key"]: m for m in members}

            chosen = by_key["up_2pct"]
            self.assertTrue(chosen["available"])
            self.assertEqual(chosen["model_name"], "cls2_new")
            self.assertEqual(chosen["features"], ["f1", "f3"])
            self.assertEqual(chosen["output_column"], "pred_prob_up_2pct_5m")
            self.assertTrue(str(chosen["model_path"]).endswith("model.ubj"))

            self.assertFalse(by_key["up_4pct"]["available"])  # wrong dataset
            self.assertTrue(by_key["up_5pct"]["available"])
            self.assertFalse(by_key["up_gt6pct"]["available"])

            self.assertIn("2/6 classifiers", package_members_summary(members))

    def test_non_anchor_target_yields_all_missing(self) -> None:
        self.assertFalse(is_package_anchor_target("label_up_2pct_5m"))
        with tempfile.TemporaryDirectory() as data_dir:
            members = discover_prediction_package_members(
                data_dir, dataset="MS_A", anchor_target="label_up_2pct_5m"
            )
        self.assertTrue(all(not m["available"] for m in members))


class _FakeProbaModel:
    """Positive-class probability = 0.25 * f1 (clipped by the executor)."""

    def predict_proba(self, X):  # noqa: N803
        import numpy as np

        p = np.clip(np.asarray(X["f1"], dtype=float) * 0.25, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


class _FakeRawModel:
    """Plain predict that can exceed [0, 1] to exercise probability clipping."""

    def predict(self, X):  # noqa: N803
        import numpy as np

        arr = np.asarray(X, dtype=float)
        return arr[:, 0] * 0.6 if arr.ndim > 1 else arr * 0.6


class TestGenericMemberExecutor(unittest.TestCase):
    def _day_df(self):
        import pandas as pd

        return pd.DataFrame({"f1": [0.0, 1.0, 2.0, 4.0], "f2": [1.0, 1.0, 1.0, 1.0]})

    def _member(self, **overrides) -> dict:
        member = {
            "role": "probability_up_2pct",
            "key": "up_2pct",
            "target": "label_up_2pct_5m",
            "prediction_type": "probability",
            "output_column": "pred_prob_up_2pct_5m",
            "available": True,
            "model_path": "X:/fake/model.ubj",
            "algorithm": "xgboost",
            "features": ["f1", "f2"],
        }
        member.update(overrides)
        return member

    def test_members_predict_and_missing_stay_null(self) -> None:
        from chain_replay_ml.model_lab.prediction_parallel import (
            run_package_member_predictions,
        )

        members = [
            self._member(),
            self._member(
                role="probability_up_3pct",
                output_column="pred_prob_up_3pct_5m",
                available=False,
                model_path=None,
            ),
            self._member(
                role="probability_up_4pct",
                output_column="pred_prob_up_4pct_5m",
                features=["f1", "not_in_frame"],
            ),
        ]
        with mock.patch(
            "chain_replay_ml.training.model_runtime.load_prediction_model_cached",
            return_value=(_FakeProbaModel(), 0.0, False),
        ):
            outputs, notes = run_package_member_predictions(
                members=members, day_df=self._day_df()
            )

        self.assertEqual(
            set(outputs),
            {"pred_prob_up_2pct_5m", "pred_prob_up_3pct_5m", "pred_prob_up_4pct_5m"},
        )
        self.assertIsNone(outputs["pred_prob_up_3pct_5m"])  # unavailable member
        self.assertIsNone(outputs["pred_prob_up_4pct_5m"])  # features missing
        self.assertEqual(len(notes), 1)
        values = list(outputs["pred_prob_up_2pct_5m"])
        self.assertAlmostEqual(values[1], 0.25)
        self.assertAlmostEqual(values[3], 1.0)

    def test_probability_clipped_for_plain_predict_models(self) -> None:
        from chain_replay_ml.model_lab.prediction_parallel import (
            run_package_member_predictions,
        )

        with mock.patch(
            "chain_replay_ml.training.model_runtime.load_prediction_model_cached",
            return_value=(_FakeRawModel(), 0.0, False),
        ):
            outputs, notes = run_package_member_predictions(
                members=[self._member(features=["f1"])], day_df=self._day_df()
            )
        self.assertEqual(notes, [])
        values = list(outputs["pred_prob_up_2pct_5m"])
        self.assertAlmostEqual(values[1], 0.6)
        self.assertAlmostEqual(values[3], 1.0)  # 2.4 clipped to 1.0


if __name__ == "__main__":
    unittest.main()

