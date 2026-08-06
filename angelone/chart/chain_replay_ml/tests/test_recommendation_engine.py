"""Unit tests for Recommendation Engine (Phase 5.3)."""

from __future__ import annotations

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path


def _names(affected: list) -> list[str]:
    from chain_replay_ml.recommendation_engine.rules import feature_names

    return feature_names(affected)


class RuleUnitTests(unittest.TestCase):
    """Every deterministic rule has at least one positive hit test."""

    def setUp(self) -> None:
        from chain_replay_ml.recommendation_engine.config import merge_thresholds

        self.th = merge_thresholds()

    def test_r1_high_drift_low_importance(self) -> None:
        from chain_replay_ml.recommendation_engine.rules import (
            rule_high_drift_low_importance,
        )

        rows = [
            {"feature": f"mid{i}", "rank_gain": 10 + i, "drift": 0.05, "risk_score": 5}
            for i in range(40)
        ]
        rows.append(
            {"feature": "noise", "rank_gain": 80, "drift": 0.6, "risk_score": 40}
        )
        rows.append(
            {"feature": "keep", "rank_gain": 2, "drift": 0.6, "risk_score": 40}
        )
        out = rule_high_drift_low_importance(rows, self.th)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["category"], "Feature Review")
        self.assertEqual(
            out[0]["title"], "Review 1 drifting low-importance features"
        )
        self.assertIn("noise", _names(out[0]["affected_features"]))
        self.assertNotIn("keep", _names(out[0]["affected_features"]))

    def test_r2_high_null_drift(self) -> None:
        from chain_replay_ml.recommendation_engine.rules import rule_high_null_drift

        rows = [
            {"feature": "sparse", "null_drift_pp": 12.0},
            {"feature": "ok", "null_drift_pp": 1.0},
        ]
        out = rule_high_null_drift(rows, self.th)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["category"], "Data Collection")
        self.assertEqual(_names(out[0]["affected_features"]), ["sparse"])
        self.assertIn("1", out[0]["title"])

    def test_r3_high_importance_high_drift(self) -> None:
        from chain_replay_ml.recommendation_engine.rules import (
            rule_high_importance_high_drift,
        )

        rows = [
            {"feature": "key", "rank_gain": 3, "drift": 0.5, "risk_score": 55},
            {"feature": "low", "rank_gain": 90, "drift": 0.5, "risk_score": 55},
        ]
        out = rule_high_importance_high_drift(rows, self.th)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["category"], "Model Refresh")
        self.assertEqual(_names(out[0]["affected_features"]), ["key"])
        self.assertIn("1 important", out[0]["title"])

    def test_r4_high_rank_high_risk(self) -> None:
        from chain_replay_ml.recommendation_engine.rules import (
            rule_high_rank_gain_high_risk,
        )

        rows = [
            {"feature": "top", "rank_gain": 1, "risk": "high", "risk_score": 70},
            {"feature": "bot", "rank_gain": 99, "risk": "high", "risk_score": 70},
        ]
        out = rule_high_rank_gain_high_risk(rows, self.th)
        self.assertEqual(len(out), 1)
        self.assertIn("1 high-risk top", out[0]["title"])
        self.assertEqual(_names(out[0]["affected_features"]), ["top"])

    def test_r5_large_ks_small_mean_drift(self) -> None:
        from chain_replay_ml.recommendation_engine.rules import (
            rule_large_ks_small_mean_drift,
        )

        rows = [
            {"feature": "shape", "ks_statistic": 0.8, "drift_pct": 2.0},
            {"feature": "loc", "ks_statistic": 0.8, "drift_pct": 30.0},
            {"feature": "flat", "ks_statistic": 0.1, "drift_pct": 1.0},
        ]
        out = rule_large_ks_small_mean_drift(rows, self.th)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["title"], "Review 1 high-KS features")
        self.assertEqual(_names(out[0]["affected_features"]), ["shape"])

    def test_r6_high_wasserstein_low_ks(self) -> None:
        from chain_replay_ml.recommendation_engine.rules import (
            rule_high_wasserstein_low_ks,
        )

        rows = [
            {
                "feature": "scale",
                "wasserstein_normalized": 2.5,
                "ks_statistic": 0.05,
            },
            {
                "feature": "shape",
                "wasserstein_normalized": 2.5,
                "ks_statistic": 0.9,
            },
        ]
        out = rule_high_wasserstein_low_ks(rows, self.th)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["title"], "Review 1 scale-shift features")
        self.assertEqual(_names(out[0]["affected_features"]), ["scale"])

    def test_r7_feature_removal(self) -> None:
        from chain_replay_ml.recommendation_engine.rules import (
            rule_feature_removal_candidates,
        )

        # 8 features so bottom quartile is clear
        rows = [
            {"feature": f"f{i}", "rank_gain": i + 1, "drift": 0.1, "gain": 1.0}
            for i in range(7)
        ]
        rows.append(
            {"feature": "junk", "rank_gain": 8, "drift": 0.7, "gain": 0.0, "risk_score": 60}
        )
        out = rule_feature_removal_candidates(rows, self.th)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["category"], "Feature Removal")
        self.assertEqual(out[0]["title"], "Review 1 bottom-ranked features")
        self.assertIn("junk", _names(out[0]["affected_features"]))

    def test_no_false_positive_when_clean(self) -> None:
        from chain_replay_ml.recommendation_engine.rules import apply_rules

        rows = [
            {
                "feature": "stable",
                "rank_gain": 5,
                "drift": 0.05,
                "risk_score": 5,
                "risk": "low",
                "ks_statistic": 0.05,
                "drift_pct": 1.0,
                "wasserstein_normalized": 0.1,
                "null_drift_pp": 0.0,
                "gain": 10.0,
            }
        ]
        out = apply_rules(rows)
        self.assertEqual(out, [])


class SuggestionSchemaTests(unittest.TestCase):
    """Title counts, evidence_score, reason_bullets, benefit structure."""

    def setUp(self) -> None:
        from chain_replay_ml.recommendation_engine.config import merge_thresholds

        self.th = merge_thresholds()

    def test_title_includes_hit_count(self) -> None:
        from chain_replay_ml.recommendation_engine.rules import (
            rule_high_drift_low_importance,
            rule_large_ks_small_mean_drift,
        )

        rows = [
            {"feature": f"mid{i}", "rank_gain": 10 + i, "drift": 0.05, "risk_score": 5}
            for i in range(40)
        ]
        for i in range(3):
            rows.append(
                {
                    "feature": f"noise{i}",
                    "rank_gain": 80 + i,
                    "drift": 0.6,
                    "risk_score": 40,
                    "ks_statistic": 0.05,
                    "drift_pct": 20.0,
                }
            )
        out = rule_high_drift_low_importance(rows, self.th)
        self.assertEqual(len(out), 1)
        self.assertEqual(
            out[0]["title"], "Review 3 drifting low-importance features"
        )
        self.assertEqual(out[0]["evidence"]["feature_count"], 3)

        ks_rows = [
            {"feature": f"ks{i}", "ks_statistic": 0.7, "drift_pct": 1.0}
            for i in range(5)
        ]
        ks_out = rule_large_ks_small_mean_drift(ks_rows, self.th)
        self.assertEqual(ks_out[0]["title"], "Review 5 high-KS features")

    def test_evidence_score_0_100_integer(self) -> None:
        from chain_replay_ml.recommendation_engine.rules import rule_high_null_drift

        rows = [{"feature": "sparse", "null_drift_pp": 12.0}]
        out = rule_high_null_drift(rows, self.th)
        self.assertEqual(len(out), 1)
        score = out[0]["evidence_score"]
        self.assertIsInstance(score, int)
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 100)
        self.assertAlmostEqual(out[0]["confidence"], score / 100.0, places=2)

    def test_reason_bullets_structured(self) -> None:
        from chain_replay_ml.recommendation_engine.rules import (
            rule_high_drift_low_importance,
        )

        rows = [
            {"feature": f"mid{i}", "rank_gain": 10 + i, "drift": 0.05, "risk_score": 5}
            for i in range(40)
        ]
        rows.append(
            {"feature": "noise", "rank_gain": 80, "drift": 0.6, "risk_score": 40}
        )
        out = rule_high_drift_low_importance(rows, self.th)
        bullets = out[0]["reason_bullets"]
        self.assertIsInstance(bullets, list)
        self.assertGreaterEqual(len(bullets), 2)
        self.assertTrue(all(isinstance(b, str) and b for b in bullets))
        self.assertIn("Low importance", bullets)
        self.assertIn("High drift", bullets)
        for b in bullets:
            self.assertIn(b, out[0]["reason"])

    def test_expected_benefit_structure(self) -> None:
        from chain_replay_ml.recommendation_engine.rules import (
            rule_feature_removal_candidates,
        )

        rows = [
            {"feature": f"f{i}", "rank_gain": i + 1, "drift": 0.1, "gain": 1.0}
            for i in range(7)
        ]
        rows.append(
            {"feature": "junk", "rank_gain": 8, "drift": 0.7, "gain": 0.0, "risk_score": 60}
        )
        out = rule_feature_removal_candidates(rows, self.th)
        benefit = out[0]["expected_benefit"]
        self.assertIsInstance(benefit, dict)
        for key in (
            "model_stability",
            "prediction_accuracy",
            "training_speed",
            "summary",
        ):
            self.assertIn(key, benefit)
            self.assertTrue(str(benefit[key]).strip())
        self.assertIn(
            benefit["model_stability"],
            {"high", "medium", "low", "slightly_improved", "unknown", "none"},
        )

    def test_affected_features_include_evidence(self) -> None:
        from chain_replay_ml.recommendation_engine.rules import (
            rule_high_importance_high_drift,
        )

        rows = [
            {
                "feature": "key",
                "rank_gain": 3,
                "drift": 0.5,
                "risk_score": 55,
                "ks_statistic": 0.4,
            },
        ]
        out = rule_high_importance_high_drift(rows, self.th)
        feats = out[0]["affected_features"]
        self.assertEqual(len(feats), 1)
        self.assertIsInstance(feats[0], dict)
        self.assertEqual(feats[0]["feature"], "key")
        self.assertEqual(feats[0]["rank_gain"], 3)
        self.assertEqual(feats[0]["risk_score"], 55)

    def test_normalize_legacy_confidence(self) -> None:
        from chain_replay_ml.recommendation_engine.writer import normalize_suggestion

        legacy = {
            "id": "legacy",
            "title": "Old",
            "reason": "Because A; Because B",
            "confidence": 0.82,
            "expected_benefit": "Faster training",
            "affected_features": ["a", "b"],
            "priority": "High",
            "category": "Feature Review",
            "evidence": {},
        }
        norm = normalize_suggestion(legacy)
        self.assertEqual(norm["evidence_score"], 82)
        self.assertEqual(norm["reason_bullets"], ["Because A", "Because B"])
        self.assertEqual(norm["expected_benefit"]["summary"], "Faster training")
        self.assertEqual(
            [f["feature"] for f in norm["affected_features"]], ["a", "b"]
        )


class PersistRoundtripTests(unittest.TestCase):
    def _write_studio(
        self, pkg: str, dirname: str, rows: list[dict], meta: dict | None = None
    ) -> None:
        path = os.path.join(pkg, dirname)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, "comparison.json"), "w", encoding="utf-8") as fh:
            json.dump({"rows": rows}, fh)
        with open(os.path.join(path, "run_meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta or {"ok": True}, fh)

    def _pkg(self, data_dir: str, name: str = "Planner_Model") -> tuple[str, str]:
        from chain_replay_ml.training.paths import model_package_dir, safe_model_name

        model_name = safe_model_name(name)
        pkg = model_package_dir(data_dir, model_name)
        os.makedirs(pkg, exist_ok=True)
        with open(os.path.join(pkg, "metadata.json"), "w", encoding="utf-8") as fh:
            json.dump({"model_name": model_name}, fh)
        return model_name, pkg

    def test_end_to_end_persist_load(self) -> None:
        from chain_replay_ml.recommendation_engine import run_recommendation_engine
        from chain_replay_ml.recommendation_engine.writer import load_studio_artifacts

        with tempfile.TemporaryDirectory() as tmp:
            model_name, pkg = self._pkg(tmp)
            self._write_studio(
                pkg,
                "feature_importance_studio",
                [
                    {"feature": "key", "rank_gain": 1, "gain": 20},
                    {"feature": "noise", "rank_gain": 80, "gain": 0.0},
                ],
            )
            self._write_studio(
                pkg,
                "feature_distribution_studio",
                [
                    {"feature": "key", "null_pct": 0.0, "skew": 0.1},
                    {"feature": "noise", "null_pct": 8.0, "skew": 1.0},
                ],
            )
            self._write_studio(
                pkg,
                "feature_drift_studio",
                [
                    {
                        "feature": "key",
                        "drift": 0.55,
                        "drift_pct": 3.0,
                        "risk": "high",
                        "risk_score": 62.0,
                        "ks_statistic": 0.85,
                        "wasserstein_normalized": 0.2,
                        "null_drift_pp": 0.0,
                    },
                    {
                        "feature": "noise",
                        "drift": 0.5,
                        "drift_pct": 40.0,
                        "risk": "high",
                        "risk_score": 55.0,
                        "ks_statistic": 0.1,
                        "wasserstein_normalized": 2.0,
                        "null_drift_pp": 10.0,
                    },
                ],
            )

            result = run_recommendation_engine(data_dir=tmp, model_name=model_name)
            self.assertTrue(result.ok, result.error)
            for name in ("planner.json", "summary.json", "run_meta.json"):
                self.assertTrue(
                    os.path.isfile(os.path.join(result.artifacts_dir, name)), name
                )

            loaded = load_studio_artifacts(pkg)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(len(loaded["suggestions"]), len(result.suggestions))
            self.assertEqual(
                loaded["summary"].get("total_suggestions"),
                result.summary.get("total_suggestions"),
            )
            self.assertIn("planner_version", loaded["meta"])
            self.assertEqual(loaded["meta"].get("schema_version"), 5)
            self.assertIn("input_artifacts", loaded["meta"])
            self.assertTrue(loaded["meta"]["input_artifacts"]["importance"])
            self.assertTrue(loaded["meta"]["input_artifacts"]["drift"])

            # Required experiment fields (Planner v2 + refinements)
            for s in result.suggestions:
                for key in (
                    "id",
                    "experiment_id",
                    "category",
                    "title",
                    "reason",
                    "reason_bullets",
                    "evidence",
                    "evidence_score",
                    "hypothesis",
                    "expected_experiment",
                    "suggested_next_steps",
                    "estimated_effort",
                    "status",
                    "experiment_scope",
                    "priority",
                    "affected_features",
                    "findings",
                    "recommendations",
                    "created_from",
                    "generated_at",
                    "planner_version",
                ):
                    self.assertIn(key, s, key)
                self.assertIsInstance(s["evidence_score"], int)
                self.assertIsInstance(s["reason_bullets"], list)
                self.assertTrue(str(s["hypothesis"]).strip())
                self.assertTrue(str(s["expected_experiment"]).strip())
                self.assertEqual(s["status"], "Not Started")
                self.assertRegex(str(s["experiment_id"]), r"^EXP-\d{3}$")
                self.assertEqual(s["created_from"], model_name)
                self.assertTrue(str(s["generated_at"] or "").strip())
                self.assertTrue(isinstance(s["findings"], list) and s["findings"])
                self.assertLessEqual(len(result.suggestions), 10)

            # experiment_state.json created alongside planner
            state_path = os.path.join(result.artifacts_dir, "experiment_state.json")
            self.assertTrue(os.path.isfile(state_path))
            self.assertIn("experiment_state", loaded)
            self.assertIn("highest_evidence_suggestion", result.summary)

            # Core rules still fire (ids are family-bucket after merge)
            ids = {s["id"] for s in result.suggestions}
            families = {s.get("family") for s in result.suggestions if s.get("family")}
            # At least one feature-family experiment or model experiment present
            self.assertTrue(ids)
            # Findings reference original rule ids when features matched
            all_rule_ids = set()
            for s in result.suggestions:
                for f in s.get("findings") or []:
                    if isinstance(f, dict) and f.get("rule_id"):
                        all_rule_ids.add(str(f["rule_id"]))
            self.assertTrue(
                all_rule_ids
                & {
                    "R1_high_drift_low_importance",
                    "R2_high_null_drift",
                    "R3_high_importance_high_drift",
                    "R4_high_rank_high_risk",
                    "R5_large_ks_small_mean_drift",
                    "R6_high_wasserstein_low_ks",
                }
                or families
            )

    def test_missing_artifacts_error(self) -> None:
        from chain_replay_ml.recommendation_engine import run_recommendation_engine

        with tempfile.TemporaryDirectory() as tmp:
            model_name, _pkg = self._pkg(tmp)
            result = run_recommendation_engine(data_dir=tmp, model_name=model_name)
            self.assertFalse(result.ok)
            self.assertIn("artifacts", (result.error or "").lower())


class NoDatasetEngineImportTests(unittest.TestCase):
    def test_recommendation_engine_has_no_dataset_engine_imports(self) -> None:
        root = Path(__file__).resolve().parents[1] / "recommendation_engine"
        self.assertTrue(root.is_dir(), str(root))
        banned = (
            "dataset_builder",
            "dataset_loader",
            "dataset_engine",
            "chain_replay_ml.training.dataset_loader",
        )
        for path in root.rglob("*.py"):
            src = path.read_text(encoding="utf-8")
            tree = ast.parse(src, filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        mod = alias.name
                        for b in banned:
                            self.assertNotIn(
                                b,
                                mod,
                                f"{path.name} imports {mod}",
                            )
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    for b in banned:
                        self.assertNotIn(
                            b,
                            mod,
                            f"{path.name} from-imports {mod}",
                        )


class ExperimentPlannerV2Tests(unittest.TestCase):
    """Family grouping, split experiments, ranking/cap, hypothesis, no new rules."""

    def test_no_new_rules_in_order(self) -> None:
        from chain_replay_ml.recommendation_engine.rules import RULE_ORDER

        expected = [
            "high_drift_low_importance",
            "high_null_drift",
            "high_importance_high_drift",
            "high_rank_gain_high_risk",
            "large_ks_small_mean_drift",
            "high_wasserstein_low_ks",
            "feature_removal_candidates",
        ]
        self.assertEqual([name for name, _ in RULE_ORDER], expected)

    def test_family_heuristics(self) -> None:
        from chain_replay_ml.recommendation_engine.families import (
            resolve_feature_family,
        )

        cases = {
            "atm_iv_ce": "IV",
            "iv_ema100": "IV",
            "ltp_ema200_to_ltp_ratio": "EMA",
            "max_call_oi_pct": "OI",
            "atm_pcr": "PCR",
            "delta_w_volume_flow_5m": "Greeks",
            "option_volume": "Volume",
            "spot_change_5m": "Spot",
            "days_to_expiry": "Time",
            "atm_straddle": "Straddle",
            "chain_pcr": "PCR",  # PCR before Chain when both match
            "chain_gex": "GEX",
            "call_gex": "GEX",
            "future_ltp_5m": "Targets",
            "mystery_feat": "Other",
        }
        for name, fam in cases.items():
            self.assertEqual(
                resolve_feature_family(name), fam, msg=name
            )

    def test_family_registry_map_preferred(self) -> None:
        from chain_replay_ml.recommendation_engine.families import (
            map_registry_families,
            resolve_feature_family,
        )

        mapped = map_registry_families({"weird_x": "iv", "other_y": "Greeks"})
        self.assertEqual(mapped["weird_x"], "IV")
        self.assertEqual(
            resolve_feature_family("weird_x", family_by_name=mapped), "IV"
        )
        self.assertEqual(
            resolve_feature_family("other_y", family_by_name=mapped), "Greeks"
        )

    def test_split_rule_match_by_family(self) -> None:
        from chain_replay_ml.recommendation_engine.experiments import (
            split_suggestion_by_family,
        )
        from chain_replay_ml.recommendation_engine.rules import (
            rule_high_drift_low_importance,
        )
        from chain_replay_ml.recommendation_engine.config import merge_thresholds

        th = merge_thresholds()
        rows = [
            {"feature": f"mid{i}", "rank_gain": 10 + i, "drift": 0.05, "risk_score": 5}
            for i in range(40)
        ]
        rows.extend(
            [
                {
                    "feature": "atm_iv_ce",
                    "rank_gain": 80,
                    "drift": 0.7,
                    "risk_score": 55,
                    "ks_statistic": 0.5,
                },
                {
                    "feature": "atm_pcr",
                    "rank_gain": 81,
                    "drift": 0.6,
                    "risk_score": 40,
                    "ks_statistic": 0.4,
                },
                {
                    "feature": "call_gex",
                    "rank_gain": 82,
                    "drift": 0.65,
                    "risk_score": 50,
                    "ks_statistic": 0.3,
                },
            ]
        )
        raw = rule_high_drift_low_importance(rows, th)
        self.assertEqual(len(raw), 1)
        self.assertGreaterEqual(len(raw[0]["affected_features"]), 3)

        experiments = split_suggestion_by_family(raw[0], th=th)
        families = {e["family"] for e in experiments}
        self.assertIn("IV", families)
        self.assertIn("PCR", families)
        self.assertIn("GEX", families)
        for exp in experiments:
            self.assertTrue(str(exp.get("hypothesis") or "").strip())
            self.assertEqual(exp["evidence"]["family"], exp["family"])
            self.assertIn("rule_id", exp["evidence"])
            self.assertIn("matched_features", exp["evidence"])
            self.assertIn("top_contributors", exp["evidence"])
            self.assertTrue(str(exp["title"]).startswith("Review "))
            # Small groups — not the mega list
            self.assertLess(len(exp["affected_features"]), 72)

    def test_ranking_and_cap(self) -> None:
        from chain_replay_ml.recommendation_engine.experiments import (
            refine_to_experiments,
        )

        # Fabricate several family-like raw suggestions with different strength.
        raw = []
        for i, fam_feats in enumerate(
            [
                [
                    {
                        "feature": "atm_iv_ce",
                        "risk_score": 90,
                        "drift": 0.9,
                        "ks_statistic": 0.8,
                        "rank_gain": 80,
                    }
                ],
                [
                    {
                        "feature": "atm_pcr",
                        "risk_score": 10,
                        "drift": 0.4,
                        "ks_statistic": 0.2,
                        "rank_gain": 60,
                    }
                ],
                [
                    {
                        "feature": "call_gex",
                        "risk_score": 50,
                        "drift": 0.5,
                        "ks_statistic": 0.4,
                        "rank_gain": 70,
                    }
                ],
            ]
        ):
            raw.append(
                {
                    "id": f"R1_high_drift_low_importance",
                    "category": "Feature Review",
                    "title": f"Review {i}",
                    "reason": "test",
                    "reason_bullets": ["Low importance", "High drift"],
                    "evidence": {
                        "rule": "high_drift_low_importance",
                        "thresholds": {"high_drift": 0.35},
                    },
                    "evidence_score": 90 - i * 5,
                    "confidence": 0.9 - i * 0.05,
                    "priority": "High",
                    "affected_features": fam_feats,
                    "expected_benefit": {"summary": "x"},
                }
            )
        # Duplicate id raw entries would collapse after split; use distinct rule ids
        raw[1]["id"] = "R5_large_ks_small_mean_drift"
        raw[1]["evidence"] = {
            "rule": "large_ks_small_mean_drift",
            "thresholds": {"large_ks": 0.4},
        }
        raw[2]["id"] = "R7_feature_removal_candidates"
        raw[2]["evidence"] = {"rule": "feature_removal_candidates", "thresholds": {}}

        capped = refine_to_experiments(raw, thresholds={"max_experiments": 2})
        self.assertEqual(len(capped), 2)
        # Highest aggregate first
        self.assertGreaterEqual(
            float(capped[0]["rank_score"]), float(capped[1]["rank_score"])
        )
        for exp in capped:
            self.assertTrue(exp.get("hypothesis"))

    def test_apply_rules_emits_hypothesis_not_mega_list(self) -> None:
        from chain_replay_ml.recommendation_engine.rules import apply_rules

        rows = [
            {"feature": f"mid{i}", "rank_gain": 10 + i, "drift": 0.05, "risk_score": 5}
            for i in range(40)
        ]
        for name, risk in (
            ("atm_iv_ce", 60),
            ("iv_skew_atm", 55),
            ("atm_pcr", 45),
            ("chain_pcr", 40),
            ("call_gex", 50),
            ("atm_straddle", 35),
            ("max_call_oi_pct", 42),
        ):
            rows.append(
                {
                    "feature": name,
                    "rank_gain": 80,
                    "drift": 0.7,
                    "risk_score": risk,
                    "ks_statistic": 0.5,
                    "drift_pct": 20.0,
                    "wasserstein_normalized": 0.2,
                    "null_drift_pp": 0.0,
                    "gain": 0.0,
                }
            )
        out = apply_rules(rows, thresholds={"max_experiments": 10})
        self.assertGreaterEqual(len(out), 1)
        self.assertLessEqual(len(out), 10)
        for exp in out:
            self.assertTrue(str(exp.get("hypothesis") or "").strip())
            self.assertNotIn("prediction_accuracy", exp.get("hypothesis", ""))
            # No single mega experiment with all features
            self.assertLessEqual(len(exp.get("affected_features") or []), 20)
            if exp.get("family"):
                self.assertIn(exp["family"], exp["title"])

    def test_normalize_hypothesis_from_legacy_benefit(self) -> None:
        from chain_replay_ml.recommendation_engine.writer import normalize_suggestion

        legacy = {
            "id": "R1_high_drift_low_importance__IV",
            "title": "Review IV Features",
            "reason": "Low importance · High drift",
            "confidence": 0.9,
            "expected_benefit": {
                "model_stability": "medium",
                "prediction_accuracy": "unknown",
                "training_speed": "slightly_improved",
                "summary": "Legacy summary text",
            },
            "affected_features": [{"feature": "atm_iv_ce", "risk_score": 50}],
            "priority": "High",
            "category": "Feature Review",
            "evidence": {"rule": "high_drift_low_importance", "family": "IV"},
        }
        norm = normalize_suggestion(legacy)
        self.assertEqual(norm["hypothesis"], "Legacy summary text")
        self.assertEqual(norm["family"], "IV")
        self.assertEqual(norm["evidence_score"], 90)


class ExperimentPlannerRefinementTests(unittest.TestCase):
    """IDs, effort, expected_experiment, status, family aggregates (no new rules)."""

    def test_stable_experiment_ids_after_cap(self) -> None:
        from chain_replay_ml.recommendation_engine.experiments import (
            refine_to_experiments,
        )

        raw = []
        for i, feat in enumerate(("atm_iv_ce", "atm_pcr", "call_gex")):
            raw.append(
                {
                    "id": f"R{i + 1}_high_drift_low_importance",
                    "category": "Feature Review",
                    "title": "Review",
                    "reason": "test",
                    "reason_bullets": ["Low importance"],
                    "evidence": {"rule": "high_drift_low_importance", "thresholds": {}},
                    "evidence_score": 80 - i,
                    "confidence": 0.8,
                    "priority": "High",
                    "affected_features": [
                        {
                            "feature": feat,
                            "risk_score": 70 - i * 10,
                            "drift": 0.7,
                            "ks_statistic": 0.5,
                            "rank_gain": 80,
                        }
                    ],
                }
            )
        out = refine_to_experiments(raw, thresholds={"max_experiments": 10})
        self.assertGreaterEqual(len(out), 2)
        ids = [s["experiment_id"] for s in out]
        self.assertEqual(ids[0], "EXP-001")
        self.assertEqual(ids[1], "EXP-002")
        # Internal id preserved and distinct from display id (family bucket after merge)
        for s in out:
            self.assertNotEqual(s["id"], s["experiment_id"])
            self.assertTrue(
                str(s["id"]).startswith("feature__")
                or str(s["id"]).startswith("model__")
                or str(s["id"]).startswith("R")
            )

    def test_estimated_effort_heuristics(self) -> None:
        from chain_replay_ml.recommendation_engine.experiments import estimate_effort

        self.assertEqual(
            estimate_effort(category="Retraining", feature_count=0), "High"
        )
        self.assertEqual(
            estimate_effort(category="Model Refresh", feature_count=3), "High"
        )
        self.assertEqual(
            estimate_effort(category="Data Collection", feature_count=2), "High"
        )
        self.assertEqual(
            estimate_effort(category="Feature Review", feature_count=2), "Easy"
        )
        self.assertEqual(
            estimate_effort(category="Feature Review", feature_count=5), "Easy"
        )
        self.assertEqual(
            estimate_effort(category="Feature Review", feature_count=12), "Medium"
        )
        self.assertEqual(
            estimate_effort(category="Feature Removal", feature_count=30), "High"
        )
        self.assertEqual(
            estimate_effort(category="Threshold Review", feature_count=0), "Medium"
        )

    def test_expected_experiment_distinct_from_hypothesis(self) -> None:
        from chain_replay_ml.recommendation_engine.experiments import (
            build_expected_experiment,
            build_hypothesis,
            refine_to_experiments,
        )

        hyp = build_hypothesis("high_drift_low_importance", "EMA")
        expected = build_expected_experiment("high_drift_low_importance", "EMA")
        self.assertIn("EMA", hyp)
        self.assertIn("EMA", expected)
        self.assertNotEqual(hyp, expected)
        self.assertIn("Holdout MAE", expected)
        self.assertIn("Train without", expected)

        raw = [
            {
                "id": "R1_high_drift_low_importance",
                "category": "Feature Review",
                "title": "Review",
                "reason": "t",
                "reason_bullets": ["Low importance"],
                "evidence": {"rule": "high_drift_low_importance", "thresholds": {}},
                "evidence_score": 70,
                "confidence": 0.7,
                "priority": "Medium",
                "affected_features": [
                    {
                        "feature": "ltp_ema200_to_ltp_ratio",
                        "risk_score": 40,
                        "drift": 0.5,
                        "ks_statistic": 0.4,
                        "rank_gain": 60,
                    }
                ],
            }
        ]
        out = refine_to_experiments(raw)
        self.assertEqual(len(out), 1)
        self.assertTrue(str(out[0]["hypothesis"]).strip())
        self.assertTrue(str(out[0]["expected_experiment"]).strip())
        self.assertNotEqual(out[0]["hypothesis"], out[0]["expected_experiment"])
        self.assertIn("EMA", out[0]["expected_experiment"])

    def test_status_default_not_started(self) -> None:
        from chain_replay_ml.recommendation_engine.experiments import (
            refine_to_experiments,
        )

        raw = [
            {
                "id": "R8_retraining_overfitting",
                "category": "Retraining",
                "title": "Retrain",
                "reason": "overfit",
                "reason_bullets": ["Overfitting"],
                "evidence": {"rule": "diagnostics_retraining", "thresholds": {}},
                "evidence_score": 90,
                "confidence": 0.9,
                "priority": "High",
                "affected_features": [],
            }
        ]
        out = refine_to_experiments(raw)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["status"], "Not Started")
        self.assertEqual(out[0]["experiment_scope"], "model")
        self.assertEqual(out[0]["estimated_effort"], "High")

    def test_family_aggregates_include_highest_risk(self) -> None:
        from chain_replay_ml.recommendation_engine.experiments import (
            split_suggestion_by_family,
        )

        parent = {
            "id": "R1_high_drift_low_importance",
            "category": "Feature Review",
            "title": "Review",
            "reason": "t",
            "reason_bullets": ["Low importance", "High drift"],
            "evidence": {"rule": "high_drift_low_importance", "thresholds": {}},
            "evidence_score": 80,
            "confidence": 0.8,
            "priority": "High",
            "affected_features": [
                {
                    "feature": "atm_iv_ce",
                    "risk_score": 55.0,
                    "drift": 0.8,
                    "ks_statistic": 0.6,
                    "rank_gain": 70,
                },
                {
                    "feature": "iv_skew_atm",
                    "risk_score": 40.0,
                    "drift": 0.5,
                    "ks_statistic": 0.4,
                    "rank_gain": 75,
                },
            ],
        }
        exps = split_suggestion_by_family(parent)
        self.assertEqual(len(exps), 1)
        agg = exps[0]["evidence"]["aggregate"]
        self.assertEqual(agg["highest_risk_feature"], "atm_iv_ce")
        self.assertAlmostEqual(float(agg["highest_risk_score"]), 55.0)
        self.assertIsNotNone(agg["avg_drift"])
        self.assertIsNotNone(agg["avg_ks"])
        self.assertIsNotNone(agg["avg_rank_gain"])
        self.assertEqual(exps[0]["evidence"]["feature_count"], 2)
        self.assertEqual(exps[0]["family"], "IV")

    def test_model_vs_feature_scope(self) -> None:
        from chain_replay_ml.recommendation_engine.experiments import experiment_scope

        self.assertEqual(experiment_scope("Retraining"), "model")
        self.assertEqual(experiment_scope("Model Refresh"), "model")
        self.assertEqual(experiment_scope("Feature Review", "IV"), "feature")
        self.assertEqual(experiment_scope("Feature Removal"), "feature")

    def test_status_sidecar_roundtrip(self) -> None:
        from chain_replay_ml.recommendation_engine.writer import (
            apply_experiment_state,
            load_experiment_state,
            load_experiment_statuses,
            normalize_suggestion,
            record_experiment_action,
            save_experiment_statuses,
            sync_experiment_state_on_recompute,
        )

        with tempfile.TemporaryDirectory() as tmp:
            # Legacy flat map API still works (writes experiment_state.json)
            save_experiment_statuses(tmp, {"EXP-001": "In Progress", "EXP-002": "bogus"})
            loaded = load_experiment_statuses(tmp)
            self.assertEqual(loaded["EXP-001"], "In Progress")
            self.assertEqual(loaded["EXP-002"], "Not Started")  # invalid → default

            state = load_experiment_state(tmp)
            self.assertIn("EXP-001", state["experiments"])
            self.assertTrue(
                os.path.isfile(
                    os.path.join(tmp, "experiment_planner", "experiment_state.json")
                )
            )

            rows = [
                normalize_suggestion(
                    {
                        "experiment_id": "EXP-001",
                        "id": "feature__IV",
                        "title": "Review IV",
                        "hypothesis": "h",
                        "status": "Not Started",
                        "evidence_score": 70,
                    }
                )
            ]
            merged = apply_experiment_state(rows, state)
            self.assertEqual(merged[0]["status"], "In Progress")

            # Manual action with required reject note
            record_experiment_action(
                tmp,
                experiment_id="EXP-001",
                action="reject",
                note="Not actionable",
                internal_id="feature__IV",
            )
            state2 = load_experiment_state(tmp)
            self.assertEqual(state2["experiments"]["EXP-001"]["status"], "Rejected")
            self.assertTrue(state2["experiments"]["EXP-001"]["notes"])

            # Recompute with different EXP set → old id Superseded; rematch by internal_id
            new_suggestions = [
                {
                    "experiment_id": "EXP-009",
                    "id": "feature__IV",
                    "title": "IV Features",
                    "status": "Not Started",
                }
            ]
            synced = sync_experiment_state_on_recompute(tmp, new_suggestions)
            # Rematched onto EXP-009 keeping Rejected
            self.assertEqual(synced["experiments"]["EXP-009"]["status"], "Rejected")
            # Orphan EXP-002 (Not Started stub) may remain; EXP-001 remapped away
            if "EXP-001" in synced["experiments"]:
                self.assertEqual(
                    synced["experiments"]["EXP-001"]["status"], "Superseded"
                )

    def test_merge_family_findings(self) -> None:
        from chain_replay_ml.recommendation_engine.experiments import (
            refine_to_experiments,
        )

        raw = [
            {
                "id": "R1_high_drift_low_importance",
                "category": "Feature Review",
                "title": "Review drifting",
                "reason": "Low importance · High drift",
                "reason_bullets": ["Low importance", "High drift"],
                "evidence": {"rule": "high_drift_low_importance", "thresholds": {}},
                "evidence_score": 80,
                "confidence": 0.8,
                "priority": "High",
                "affected_features": [
                    {
                        "feature": "days_to_expiry",
                        "risk_score": 60,
                        "drift": 0.7,
                        "ks_statistic": 0.2,
                        "rank_gain": 80,
                    }
                ],
            },
            {
                "id": "R5_large_ks_small_mean_drift",
                "category": "Feature Review",
                "title": "Review high-KS",
                "reason": "Large KS · Small mean drift",
                "reason_bullets": ["Large KS", "Small mean drift"],
                "evidence": {"rule": "large_ks_small_mean_drift", "thresholds": {}},
                "evidence_score": 70,
                "confidence": 0.7,
                "priority": "Medium",
                "affected_features": [
                    {
                        "feature": "session_minute",
                        "risk_score": 40,
                        "drift": 0.1,
                        "ks_statistic": 0.8,
                        "rank_gain": 55,
                        "drift_pct": 2.0,
                    }
                ],
            },
        ]
        out = refine_to_experiments(raw, thresholds={"max_experiments": 10})
        # Both Time-family → one merged experiment
        time_exps = [e for e in out if e.get("family") == "Time"]
        self.assertEqual(len(time_exps), 1)
        exp = time_exps[0]
        self.assertEqual(exp["id"], "feature__Time")
        self.assertIn("Time Features", exp["title"])
        self.assertGreaterEqual(len(exp["findings"]), 2)
        recs = exp.get("recommendations") or []
        self.assertTrue(any("low importance" in r.lower() for r in recs))
        self.assertTrue(any("shape shift" in r.lower() for r in recs))
        names = {
            (f.get("feature") if isinstance(f, dict) else f)
            for f in exp["affected_features"]
        }
        self.assertIn("days_to_expiry", names)
        self.assertIn("session_minute", names)

    def test_legacy_status_json_migrates(self) -> None:
        from chain_replay_ml.recommendation_engine.writer import (
            ARTIFACT_DIRNAME,
            LEGACY_STATUS_FILENAME,
            load_experiment_state,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ARTIFACT_DIRNAME)
            os.makedirs(path, exist_ok=True)
            with open(
                os.path.join(path, LEGACY_STATUS_FILENAME), "w", encoding="utf-8"
            ) as fh:
                json.dump({"statuses": {"EXP-003": "Completed"}}, fh)
            state = load_experiment_state(tmp)
            self.assertEqual(state["experiments"]["EXP-003"]["status"], "Completed")
            self.assertTrue(
                os.path.isfile(os.path.join(path, "experiment_state.json"))
            )

    def test_hypothesis_still_present_with_refinements(self) -> None:
        from chain_replay_ml.recommendation_engine.rules import apply_rules

        rows = [
            {
                "feature": "atm_iv_ce",
                "rank_gain": 80,
                "drift": 0.7,
                "risk_score": 60,
                "ks_statistic": 0.5,
                "drift_pct": 20.0,
                "wasserstein_normalized": 0.2,
                "null_drift_pp": 0.0,
                "gain": 0.0,
            }
        ]
        out = apply_rules(rows, thresholds={"max_experiments": 10})
        self.assertGreaterEqual(len(out), 1)
        for exp in out:
            self.assertTrue(str(exp.get("hypothesis") or "").strip())
            self.assertTrue(str(exp.get("expected_experiment") or "").strip())
            self.assertRegex(str(exp.get("experiment_id") or ""), r"^EXP-\d{3}$")
            self.assertEqual(exp.get("status"), "Not Started")
            self.assertIn(exp.get("estimated_effort"), ("Easy", "Medium", "High"))


if __name__ == "__main__":
    unittest.main()
