"""Tests for Feature Test UI backend."""

from __future__ import annotations

import os
import tempfile
import unittest

import pandas as pd

from chain_replay_ml.dataset_builder.feature_test import (
    _evaluate_formula,
    _extract_formula_references,
    get_feature_inspection_meta,
    list_dataset_tokens,
    load_feature_group_catalog,
    preview_dataset_features,
    sample_dataset_features,
    validate_formula_dependencies,
)


class TestFeatureTestBackend(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.tmp, "data")
        self.ds_dir = os.path.join(self.data_dir, "datasets")
        os.makedirs(self.ds_dir)
        self.name = "test_feat_sample"
        self.df = pd.DataFrame([
            {
                "trading_day": "2026-07-07",
                "timestamp": 1000.0,
                "token": "T1",
                "strike": 24000.0,
                "option_type": "CE",
                "spot": 23815.0,
                "ltp": 210.5,
                "delta": 0.42,
                "gamma": 0.0018,
                "theta": -0.08,
                "spot_down_score_10m": 15.82,
                "spot_down_score_10m_to_ltp_ratio": 15.82 / 210.5,
                "bs_reiv_pred": 212.0,
                "spot_change_1m": 12.4,
                "roll_age_min": 5.0,
                "dgt_reiv_pred": 219.41,
            },
            {
                "trading_day": "2026-07-07",
                "timestamp": 1003.0,
                "token": "T2",
                "strike": 24100.0,
                "option_type": "PE",
                "spot": 23816.0,
                "ltp": 206.1,
                "delta": 0.43,
                "gamma": 0.0017,
                "theta": -0.08,
                "spot_down_score_10m": 14.55,
                "spot_down_score_10m_to_ltp_ratio": 14.55 / 206.1,
                "spot_up_score_1m_to_ltp_ratio": 0.11,
                "bs_reiv_pred": 213.0,
                "spot_change_1m": 9.8,
                "roll_age_min": 5.0,
                "dgt_reiv_pred": 220.02,
            },
            {
                "trading_day": "2026-07-07",
                "timestamp": 1006.0,
                "token": "T3",
                "strike": 24200.0,
                "option_type": "CE",
                "spot": 23818.0,
                "ltp": 221.3,
                "delta": 0.45,
                "gamma": 0.0019,
                "theta": -0.08,
                "spot_down_score_10m": 17.98,
                "spot_down_score_10m_to_ltp_ratio": 17.98 / 221.3,
                "spot_up_score_1m_to_ltp_ratio": 0.15,
                "bs_reiv_pred": 214.0,
                "spot_change_1m": 14.1,
                "roll_age_min": 5.0,
                "dgt_reiv_pred": 220.31,
            },
        ])
        self.df.to_parquet(os.path.join(self.ds_dir, f"{self.name}.parquet"), index=False)
        with open(os.path.join(self.ds_dir, f"{self.name}.json"), "w", encoding="utf-8") as fh:
            fh.write('{"row_count": 3, "has_parquet": true}')

    def test_load_feature_group_catalog_has_sharp_momentum(self) -> None:
        groups = load_feature_group_catalog()
        ids = {g["id"] for g in groups}
        self.assertIn("sharp_momentum", ids)
        sharp = next(g for g in groups if g["id"] == "sharp_momentum")
        self.assertEqual(sharp["feature_count"], 18)

    def test_sample_random_row(self) -> None:
        out = sample_dataset_features(
            self.data_dir,
            dataset_name=self.name,
            group_ids=["sharp_momentum"],
            seed=0,
        )
        self.assertEqual(out["total_rows"], 3)
        self.assertIn(out["row_index"], (0, 1, 2))
        names = [v["name"] for v in out["values"]]
        # Wave 4: sharp_momentum group is canonical levels (not packaging ratios).
        self.assertIn("spot_down_score_10m", names)

    def test_requires_selection(self) -> None:
        with self.assertRaises(ValueError):
            sample_dataset_features(self.data_dir, dataset_name=self.name, group_ids=[], feature_names=[])

    def test_raw_preview_multiple_rows(self) -> None:
        out = preview_dataset_features(
            self.data_dir,
            dataset_name=self.name,
            feature_names=["spot", "ltp", "delta"],
            mode="raw",
            row_count=10,
            sampling="first",
        )
        self.assertEqual(out["mode"], "raw")
        self.assertEqual(len(out["rows"]), 3)
        self.assertEqual([c["name"] for c in out["columns"]], ["spot", "ltp", "delta"])
        self.assertEqual(out["rows"][0]["values"]["spot"], 23815.0)

    def test_formula_preview_with_verification(self) -> None:
        # Wave 4: score levels are registry Computed Base (not packaging).
        meta = get_feature_inspection_meta("spot_down_score_10m")
        self.assertTrue(meta["is_derived"])
        self.assertIsNone(meta.get("ratio_inspection"))

        # Packaging ratios whose numerator is not in registry still use derived-numerator UI.
        self.df["side_to_ltp_ratio"] = self.df["ltp"] / 100.0
        self.df.to_parquet(os.path.join(self.ds_dir, f"{self.name}.parquet"), index=False)

        meta_ratio = get_feature_inspection_meta(
            "side_to_ltp_ratio",
            parquet_cols=set(self.df.columns),
        )
        self.assertTrue(meta_ratio["is_derived"])
        dep_names = [d["name"] for d in meta_ratio["direct_dependencies"]]
        self.assertIn("ltp", dep_names)
        self.assertIn("side", dep_names)

        out = preview_dataset_features(
            self.data_dir,
            dataset_name=self.name,
            feature_names=["side_to_ltp_ratio"],
            mode="formula",
            row_count=10,
            sampling="first",
            verify_formula=True,
        )
        self.assertEqual(out["mode"], "formula")
        self.assertEqual(len(out["rows"]), 3)
        self.assertTrue(out.get("inverse_verify"))
        self.assertAlmostEqual(float(out["rows"][0]["derived_numerator"]), 210.5 * (210.5 / 100.0), places=4)

    def test_formula_mode_rejects_multiple_features(self) -> None:
        with self.assertRaises(ValueError):
            preview_dataset_features(
                self.data_dir,
                dataset_name=self.name,
                feature_names=["spot", "ltp"],
                mode="formula",
            )

    def test_dgt_reiv_inspection_dependencies(self) -> None:
        meta = get_feature_inspection_meta(
            "dgt_reiv_pred",
            parquet_cols=set(self.df.columns),
        )
        self.assertTrue(meta["is_derived"])
        dep_names = [d["name"] for d in meta["direct_dependencies"]]
        self.assertIn("bs_reiv_pred", dep_names)
        self.assertIn("delta", dep_names)
        self.assertTrue(any(n.startswith("spot_change") for n in dep_names))

    def test_formula_preview_skips_missing_parquet_columns(self) -> None:
        out = preview_dataset_features(
            self.data_dir,
            dataset_name=self.name,
            feature_names=["dgt_reiv_pred"],
            mode="formula",
            row_count=10,
            sampling="first",
        )
        self.assertEqual(out["mode"], "formula")
        self.assertNotIn("roll_ltp", [c["name"] for c in out["columns"]])

    def test_ltp_to_dgt_reiv_ratio_includes_ltp_dependency(self) -> None:
        meta = get_feature_inspection_meta(
            "ltp_to_dgt_reiv_ratio",
            parquet_cols=set(self.df.columns),
        )
        dep_names = [d["name"] for d in meta["direct_dependencies"]]
        self.assertIn("ltp", dep_names)
        self.assertIn("dgt_reiv_pred", dep_names)

        self.df["ltp_to_dgt_reiv_ratio"] = self.df["ltp"] / self.df["dgt_reiv_pred"]
        self.df.to_parquet(os.path.join(self.ds_dir, f"{self.name}.parquet"), index=False)

        out = preview_dataset_features(
            self.data_dir,
            dataset_name=self.name,
            feature_names=["ltp_to_dgt_reiv_ratio"],
            mode="formula",
            row_count=10,
            sampling="first",
            verify_formula=True,
        )
        col_names = [c["name"] for c in out["columns"] if c.get("role") != "result"]
        self.assertIn("ltp", col_names)
        self.assertIn("dgt_reiv_pred", col_names)
        # Pipeline-owned packaging may not expose verify_formula via registry view.
        self.assertEqual(len(out["rows"]), 3)
        self.assertTrue(all(r.get("result") is not None for r in out["rows"]))

    def test_evaluate_formula(self) -> None:
        val = _evaluate_formula("spot_down_score_10m / ltp", {"spot_down_score_10m": 15.82, "ltp": 210.5})
        self.assertIsNotNone(val)
        self.assertAlmostEqual(float(val), 15.82 / 210.5, places=6)

    def test_formula_ast_extracts_all_references(self) -> None:
        refs = _extract_formula_references("(delta * ltp + gamma * spot) / iv")
        self.assertEqual(refs, ["delta", "ltp", "gamma", "spot", "iv"])

    def test_abs_delta_resolves_delta_dependency(self) -> None:
        from chain_replay_ml.dataset_builder.schema_registry import columns_map, load_schema_registry, enrich_column_view

        cols = columns_map(load_schema_registry())
        view = enrich_column_view("abs_delta", load_schema_registry())
        formula = str(view.get("formula_doc") or "")
        v = validate_formula_dependencies("abs_delta", formula, cols)
        self.assertTrue(v["passed"])
        self.assertIn("delta", [r.get("registry_column") for r in v["references"]])

    def test_load_feature_catalog_includes_ratio_meta(self) -> None:
        from chain_replay_ml.dataset_builder.feature_test import load_feature_catalog

        rows = load_feature_catalog(self.data_dir, dataset_name=self.name)
        self.assertGreater(len(rows), 100)
        # Packaging with non-registry numerator still exposes ratio_inspection.
        row = next(r for r in rows if r["name"] == "side_to_ltp_ratio")
        self.assertIsNotNone(row.get("ratio_inspection"))
        self.assertEqual(row["ratio_inspection"]["numerator"], "side")
        self.assertFalse(row["ratio_inspection"]["numerator_in_registry"])
        # Wave 4 sharp levels are registry features without ratio packaging UI.
        level = next(r for r in rows if r["name"] == "spot_down_score_10m")
        self.assertIsNone(level.get("ratio_inspection"))

    def test_side_to_ltp_ratio_raw_preview_includes_ltp_and_derived(self) -> None:
        self.df["side_to_ltp_ratio"] = self.df["ltp"] / 100.0
        self.df.to_parquet(os.path.join(self.ds_dir, f"{self.name}.parquet"), index=False)

        meta = get_feature_inspection_meta(
            "side_to_ltp_ratio",
            parquet_cols=set(self.df.columns),
        )
        self.assertIsNotNone(meta.get("ratio_inspection"))
        self.assertEqual(meta["ratio_inspection"]["numerator"], "side")

        out = preview_dataset_features(
            self.data_dir,
            dataset_name=self.name,
            feature_names=["side_to_ltp_ratio"],
            mode="raw",
            row_count=10,
            sampling="first",
        )
        col_names = [c["name"] for c in out["columns"]]
        self.assertIn("ltp", col_names)
        self.assertIn("side", col_names)
        self.assertIn("side_to_ltp_ratio", col_names)
        row = out["rows"][0]
        ratio = row["values"]["side_to_ltp_ratio"]
        ltp = row["values"]["ltp"]
        derived = row["values"]["side"]
        self.assertIsNotNone(ratio)
        self.assertIsNotNone(ltp)
        self.assertAlmostEqual(float(derived), float(ratio) * float(ltp), places=4)

    def test_spot_up_score_level_in_registry_not_packaging_ui(self) -> None:
        # Wave 4: score level is registry Computed Base; packaging no longer uses
        # derived-numerator UI because the numerator is now in the registry.
        level = get_feature_inspection_meta(
            "spot_up_score_1m",
            parquet_cols={"ltp", "spot_up_score_1m"},
        )
        self.assertTrue(level["is_derived"])
        self.assertIsNone(level.get("ratio_inspection"))

        meta = get_feature_inspection_meta(
            "spot_up_score_1m_to_ltp_ratio",
            parquet_cols={"ltp", "spot_up_score_1m_to_ltp_ratio", "spot_up_score_1m"},
        )
        self.assertIsNone(meta.get("ratio_inspection"))

        self.df["spot_up_score_1m"] = [6.46, 0.301191 * 21.45, 0.15 * 221.3]
        self.df.loc[0, "ltp"] = 21.45
        self.df.to_parquet(os.path.join(self.ds_dir, f"{self.name}.parquet"), index=False)

        out = preview_dataset_features(
            self.data_dir,
            dataset_name=self.name,
            feature_names=["spot_up_score_1m"],
            mode="raw",
            row_count=10,
            sampling="first",
        )
        self.assertIn("spot_up_score_1m", [c["name"] for c in out["columns"]])
        self.assertAlmostEqual(float(out["rows"][0]["values"]["spot_up_score_1m"]), 6.46, places=4)
    def test_registry_validation_pass_and_missing(self) -> None:
        from chain_replay_ml.dataset_builder.schema_registry import columns_map, load_schema_registry

        cols = columns_map(load_schema_registry())
        ok = validate_formula_dependencies(
            "test_feat",
            "(delta * ltp + gamma * spot) / iv",
            cols,
        )
        self.assertTrue(ok["passed"])
        self.assertEqual(ok["expected_count"], 5)
        self.assertEqual(ok["detected_count"], 5)

        bad = validate_formula_dependencies(
            "dgt_reiv_pred",
            "roll_ltp + delta * spot_change",
            cols,
        )
        self.assertFalse(bad["passed"])
        self.assertIn("roll_ltp", bad["missing"])

    def test_dgt_reiv_validation_flags_roll_ltp(self) -> None:
        from chain_replay_ml.dataset_builder.schema_registry import columns_map, load_schema_registry

        cols = columns_map(load_schema_registry())
        meta = get_feature_inspection_meta(
            "dgt_reiv_pred",
            columns=cols,
            parquet_cols=set(self.df.columns),
        )
        val = meta["dependency_validation"]
        self.assertIn("roll_ltp", val["missing"])
        self.assertFalse(val["passed"])

    def test_list_dataset_tokens(self) -> None:
        out = list_dataset_tokens(self.data_dir, self.name)
        self.assertTrue(out["has_token_column"])
        self.assertEqual(out["token_count"], 3)
        tokens = {t["token"] for t in out["tokens"]}
        self.assertEqual(tokens, {"T1", "T2", "T3"})
        t1 = next(t for t in out["tokens"] if t["token"] == "T1")
        self.assertEqual(t1["row_count"], 1)

    def test_preview_token_filter_single_token(self) -> None:
        rows = []
        for i in range(60):
            rows.append({
                "trading_day": "2026-07-07",
                "timestamp": 1000.0 + i,
                "token": "T1",
                "strike": 24000.0,
                "option_type": "CE",
                "spot": 23815.0 + i,
                "ltp": 210.5,
                "delta": 0.42,
            })
        for i in range(40):
            rows.append({
                "trading_day": "2026-07-07",
                "timestamp": 2000.0 + i,
                "token": "T2",
                "strike": 24100.0,
                "option_type": "PE",
                "spot": 23816.0,
                "ltp": 206.1,
                "delta": 0.43,
            })
        token_ds = "token_filter_ds"
        pd.DataFrame(rows).to_parquet(
            os.path.join(self.ds_dir, f"{token_ds}.parquet"), index=False,
        )

        out = preview_dataset_features(
            self.data_dir,
            dataset_name=token_ds,
            feature_names=["spot", "ltp", "delta"],
            mode="raw",
            row_count=50,
            sampling="first",
            token="T1",
        )
        self.assertEqual(out["token"], "T1")
        self.assertEqual(out["token_row_count"], 60)
        self.assertEqual(len(out["rows"]), 50)
        self.assertTrue(all(r["identity"]["token"] == "T1" for r in out["rows"]))

        out_rand = preview_dataset_features(
            self.data_dir,
            dataset_name=token_ds,
            feature_names=["spot"],
            mode="raw",
            row_count=100,
            sampling="random",
            seed=7,
            token="T2",
        )
        self.assertEqual(len(out_rand["rows"]), 40)
        self.assertTrue(all(r["identity"]["token"] == "T2" for r in out_rand["rows"]))


if __name__ == "__main__":
    unittest.main()
