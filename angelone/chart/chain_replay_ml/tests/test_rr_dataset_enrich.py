"""Tests for RR label enrichment of existing training datasets."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.writer import datasets_dir
from chain_replay_ml.model_lab.prediction_schema import (
    DATASET_TYPE_SEEN,
    DATASET_TYPE_UNSEEN,
    compute_rr_hit_labels,
)
from chain_replay_ml.model_lab.rr_dataset_enrich import (
    enrich_training_dataset_with_rr_labels,
    format_rr_enrichment_report,
)
from chain_replay_ml.model_lab.store import ModelLabStore


def _write_train_dataset(data_dir: str, name: str, rows: list[dict]) -> None:
    out = datasets_dir(data_dir)
    df = pd.DataFrame(rows)
    pq = os.path.join(out, f"{name}.parquet")
    meta = os.path.join(out, f"{name}.json")
    df.to_parquet(pq, index=False)
    with open(meta, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "dataset_name": name,
                "feature_columns": ["f1"],
                "selected_features": ["f1"],
                "feature_count": 1,
                "prediction_target_columns": ["future_ltp_5m"],
                "target_count": 1,
                "row_count": len(rows),
            },
            fh,
            indent=2,
        )


def _seed_lab(
    path: str,
    rows: list[dict],
    *,
    parent_dataset: str,
    day_types: dict[str, str] | None = None,
) -> None:
    with ModelLabStore(path) as store:
        store.ensure_prediction_schema()
        store.write_prediction_summary(
            lab_uuid="u1",
            status="ready",
            row_count=len(rows),
            trading_days=len({r.get("trading_day") for r in rows}),
            target_column="future_ltp_5m",
            parent_dataset=parent_dataset,
            parent_model_name="Model_X",
            created_at="2026-07-16T10:00:00+00:00",
        )
        store.insert_prediction_rows(rows)
        days = sorted({str(r.get("trading_day")) for r in rows if r.get("trading_day")})
        store.ensure_build_days(
            "u1",
            days,
            day_dataset_types=day_types
            or {d: DATASET_TYPE_SEEN for d in days},
        )


class RrDatasetEnrichTests(unittest.TestCase):
    def test_enrich_join_and_save_seen_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            os.makedirs(data_dir, exist_ok=True)
            train_rows = [
                {
                    "trading_day": "2026-07-01",
                    "timestamp": 1000.0,
                    "token": "T1",
                    "symbol": "NIFTY",
                    "master_row_id": 1,
                    "f1": 1.5,
                    "future_ltp_5m": 101.0,
                },
                {
                    "trading_day": "2026-07-01",
                    "timestamp": 1001.0,
                    "token": "T2",
                    "symbol": "NIFTY",
                    "master_row_id": 2,
                    "f1": 2.5,
                    "future_ltp_5m": 102.0,
                },
            ]
            _write_train_dataset(data_dir, "MS_test_rr_src", train_rows)

            lab_path = os.path.join(tmp, "lab.db")
            pred_rows = []
            # Seen day rows (joined)
            for i, (mid, hit, profit, dd) in enumerate(
                ((1, 1, 12.0, 3.0), (2, 1, 6.0, 3.0)), start=1
            ):
                rr = compute_rr_hit_labels(
                    target_reached=hit, maximum_profit=profit, maximum_drawdown=dd
                )
                pred_rows.append(
                    {
                        "lab_uuid": "u1",
                        "prediction_id": f"p{i}",
                        "trading_day": "2026-07-01",
                        "timestamp": 999.0 + i,
                        "token": f"T{i}",
                        "master_row_id": mid,
                        "target_reached": hit,
                        "maximum_profit": profit,
                        "maximum_drawdown": dd,
                        **rr,
                    }
                )
            # Unseen day row (must be ignored)
            rr_u = compute_rr_hit_labels(
                target_reached=1, maximum_profit=20.0, maximum_drawdown=2.0
            )
            pred_rows.append(
                {
                    "lab_uuid": "u1",
                    "prediction_id": "unseen1",
                    "trading_day": "2026-07-15",
                    "timestamp": 2000.0,
                    "token": "U1",
                    "master_row_id": 999,
                    "target_reached": 1,
                    "maximum_profit": 20.0,
                    "maximum_drawdown": 2.0,
                    **rr_u,
                }
            )
            _seed_lab(
                lab_path,
                pred_rows,
                parent_dataset="MS_test_rr_src",
                day_types={
                    "2026-07-01": DATASET_TYPE_SEEN,
                    "2026-07-15": DATASET_TYPE_UNSEEN,
                },
            )

            result = enrich_training_dataset_with_rr_labels(
                data_dir, "MS_test_rr_src", lab_path
            )
            self.assertTrue(result["ok"], result.get("error"))
            self.assertEqual(result["dataset_name"], "MS_test_rr_src_rr")
            rep = result["report"]
            self.assertEqual(rep["matched"], 2)
            self.assertEqual(rep["missing"], 0)
            self.assertEqual(rep["duplicates"], 0)
            self.assertEqual(rep["seen_prediction_rows"], 2)
            self.assertEqual(rep["unseen_prediction_rows"], 1)

            out_pq = os.path.join(datasets_dir(data_dir), "MS_test_rr_src_rr.parquet")
            out_json = os.path.join(datasets_dir(data_dir), "MS_test_rr_src_rr.json")
            self.assertTrue(os.path.isfile(out_pq))
            enriched = pd.read_parquet(out_pq)
            self.assertEqual(len(enriched), 2)
            self.assertEqual(int(enriched["rr_1_2_hit"].sum()), 2)
            self.assertEqual(int(enriched["rr_1_4_hit"].sum()), 1)

            with open(out_json, encoding="utf-8") as fh:
                meta = json.load(fh)
            self.assertTrue(meta["rr_enrichment"]["seen_only"])
            self.assertEqual(meta["rr_enrichment"]["unseen_ignored"], 1)

            text = format_rr_enrichment_report(result)
            self.assertIn("Seen Prediction Rows", text)
            self.assertIn("Unseen Prediction Rows", text)
            self.assertIn("(ignored)", text)
            self.assertIn("✓ Enrichment completed", text)

    def test_abort_when_train_matches_unseen(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            os.makedirs(data_dir, exist_ok=True)
            _write_train_dataset(
                data_dir,
                "MS_leak",
                [
                    {
                        "trading_day": "2026-07-15",
                        "timestamp": 1.0,
                        "token": "U1",
                        "master_row_id": 50,
                        "f1": 1.0,
                        "future_ltp_5m": 1.0,
                    },
                ],
            )
            lab_path = os.path.join(tmp, "lab.db")
            rr = compute_rr_hit_labels(
                target_reached=1, maximum_profit=10.0, maximum_drawdown=2.0
            )
            _seed_lab(
                lab_path,
                [
                    {
                        "lab_uuid": "u1",
                        "prediction_id": "seen",
                        "trading_day": "2026-07-01",
                        "timestamp": 1.0,
                        "token": "S1",
                        "master_row_id": 1,
                        "target_reached": 1,
                        "maximum_profit": 10.0,
                        "maximum_drawdown": 2.0,
                        **rr,
                    },
                    {
                        "lab_uuid": "u1",
                        "prediction_id": "unseen",
                        "trading_day": "2026-07-15",
                        "timestamp": 1.0,
                        "token": "U1",
                        "master_row_id": 50,
                        "target_reached": 1,
                        "maximum_profit": 10.0,
                        "maximum_drawdown": 2.0,
                        **rr,
                    },
                ],
                parent_dataset="MS_leak",
                day_types={
                    "2026-07-01": DATASET_TYPE_SEEN,
                    "2026-07-15": DATASET_TYPE_UNSEEN,
                },
            )
            result = enrich_training_dataset_with_rr_labels(data_dir, "MS_leak", lab_path)
            self.assertFalse(result["ok"])
            self.assertIn("Unseen", result.get("error") or "")
            self.assertEqual(result["report"]["unseen_matches"], 1)

    def test_abort_on_parent_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            os.makedirs(data_dir, exist_ok=True)
            _write_train_dataset(
                data_dir,
                "MS_other",
                [
                    {
                        "trading_day": "2026-07-01",
                        "timestamp": 1.0,
                        "token": "A",
                        "master_row_id": 1,
                        "f1": 1.0,
                        "future_ltp_5m": 1.0,
                    },
                ],
            )
            lab_path = os.path.join(tmp, "lab.db")
            rr = compute_rr_hit_labels(
                target_reached=1, maximum_profit=10.0, maximum_drawdown=2.0
            )
            _seed_lab(
                lab_path,
                [
                    {
                        "lab_uuid": "u1",
                        "prediction_id": "p1",
                        "trading_day": "2026-07-01",
                        "timestamp": 1.0,
                        "token": "A",
                        "master_row_id": 1,
                        "target_reached": 1,
                        "maximum_profit": 10.0,
                        "maximum_drawdown": 2.0,
                        **rr,
                    }
                ],
                parent_dataset="MS_expected_parent",
            )
            result = enrich_training_dataset_with_rr_labels(
                data_dir, "MS_other", lab_path
            )
            self.assertFalse(result["ok"])
            self.assertIn("inconsistent", (result.get("error") or "").lower())

    def test_abort_on_duplicate_prediction_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            os.makedirs(data_dir, exist_ok=True)
            _write_train_dataset(
                data_dir,
                "MS_dup",
                [
                    {
                        "trading_day": "2026-07-01",
                        "timestamp": 1.0,
                        "token": "A",
                        "master_row_id": 10,
                        "f1": 1.0,
                        "future_ltp_5m": 1.0,
                    },
                ],
            )
            lab_path = os.path.join(tmp, "lab.db")
            rr = compute_rr_hit_labels(
                target_reached=1, maximum_profit=10.0, maximum_drawdown=2.0
            )
            shared = {
                "lab_uuid": "u1",
                "trading_day": "2026-07-01",
                "timestamp": 1.0,
                "token": "A",
                "master_row_id": 10,
                "target_reached": 1,
                "maximum_profit": 10.0,
                "maximum_drawdown": 2.0,
                **rr,
            }
            _seed_lab(
                lab_path,
                [
                    {**shared, "prediction_id": "a"},
                    {**shared, "prediction_id": "b"},
                ],
                parent_dataset="MS_dup",
            )
            result = enrich_training_dataset_with_rr_labels(data_dir, "MS_dup", lab_path)
            self.assertFalse(result["ok"])
            self.assertGreater(int(result["report"]["duplicates"]), 0)

    def test_abort_on_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = os.path.join(tmp, "data")
            os.makedirs(data_dir, exist_ok=True)
            _write_train_dataset(
                data_dir,
                "MS_miss",
                [
                    {
                        "trading_day": "2026-07-01",
                        "timestamp": 1.0,
                        "token": "A",
                        "master_row_id": 10,
                        "f1": 1.0,
                        "future_ltp_5m": 1.0,
                    },
                    {
                        "trading_day": "2026-07-01",
                        "timestamp": 2.0,
                        "token": "B",
                        "master_row_id": 11,
                        "f1": 2.0,
                        "future_ltp_5m": 2.0,
                    },
                ],
            )
            lab_path = os.path.join(tmp, "lab.db")
            rr = compute_rr_hit_labels(
                target_reached=1, maximum_profit=10.0, maximum_drawdown=2.0
            )
            _seed_lab(
                lab_path,
                [
                    {
                        "lab_uuid": "u1",
                        "prediction_id": "only",
                        "trading_day": "2026-07-01",
                        "timestamp": 1.0,
                        "token": "A",
                        "master_row_id": 10,
                        "target_reached": 1,
                        "maximum_profit": 10.0,
                        "maximum_drawdown": 2.0,
                        **rr,
                    }
                ],
                parent_dataset="MS_miss",
            )
            result = enrich_training_dataset_with_rr_labels(data_dir, "MS_miss", lab_path)
            self.assertFalse(result["ok"])
            self.assertEqual(result["report"]["missing"], 1)
            self.assertFalse(
                os.path.isfile(os.path.join(datasets_dir(data_dir), "MS_miss_rr.parquet"))
            )


if __name__ == "__main__":
    unittest.main()
