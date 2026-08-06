"""Unit tests for Feature Inspector formatting / search-plan helpers."""

from __future__ import annotations

import unittest

from feature_intelligence.ui.inspect_format import (
    ABSENT,
    SCOPE_FEAT,
    SCOPE_NAME,
    SCOPE_ONTOLOGY,
    SCOPE_OPERATOR,
    SCOPE_PRIMITIVE,
    architecture_strip,
    build_search_plan,
    compiler_stack_summary,
    feature_display_name,
    filter_hits_by_plan,
    header_summary_lines,
    hit_list_label,
    identity_fields,
    lineage_tree_text,
    merge_hit_lists,
    ontology_chip_labels,
    overview_fields,
    references_fields,
    research_fields,
)


def _sample_payload(**overrides):
    base = {
        "research_uuid": "FRR_abc",
        "feature_uuid": "FEAT_abc",
        "canonical_name": "ema20_ratio",
        "sections_present": {
            "identity": True,
            "compiler": True,
            "ast": True,
            "ontology": True,
            "lineage": True,
            "research": True,
            "references": True,
        },
        "identity": {
            "feature_uuid": "FEAT_abc",
            "canonical_name": "ema20_ratio",
            "display_name": "EMA20 Ratio",
            "definition_version": "1.0",
            "implementation_version": "1.0",
            "definition_hash": "deadbeef",
            "primitive_ids": ["PR_CLOSE"],
            "transformation_uuid": "TR_1",
        },
        "compiler": {
            "transformation_uuid": "TR_1",
            "grammar_version": "tl_v1",
            "compiler_version": "1.0.0",
            "canonical_text": "ema(period=20)",
            "manifest_summary": {
                "compilation_uuid": "CMP_1",
                "ast_hash": "hash1",
                "cache_hit": False,
            },
        },
        "ast": {
            "ast_hash": "hash1",
            "node_count": 3,
            "root_operator": "OP_EMA",
        },
        "ontology": {
            "ontology_uuid": "ONT_1",
            "domain": "DOM_PRICE",
            "domain_display": "Price",
            "signal_type": ["SIG_TREND"],
            "signal_type_display": [
                {"vocabulary_id": "SIG_TREND", "display_name": "Trend"}
            ],
            "mathematical_family": ["MATH_RATIO"],
            "output_type": "OUT_SCALAR",
            "frequency": "FREQ_BAR",
            "stability": "STAB_MED",
        },
        "lineage": {
            "parent_count": 1,
            "child_count": 0,
            "ancestor_count": 2,
            "sample_parents": ["FEAT_parent"],
            "sample_children": [],
            "primitive_ancestors": ["PR_CLOSE"],
            "operator_ancestors": ["OP_EMA"],
        },
        "research": {
            "research_uuid": "FRR_abc",
            "research_status": "ACTIVE",
            "validation_status": "pending",
            "experiment_ids": ["EXP_1"],
            "notes": None,
            "evidence_json": None,
            "created_at": "2026-01-01T00:00:00Z",
            "transformation_uuid": "TR_1",
            "ontology_uuid": "ONT_1",
            "compiler_version": "1.0.0",
            "grammar_version": "tl_v1",
        },
        "references": {
            "research_uuid": "FRR_abc",
            "feature_uuid": "FEAT_abc",
            "transformation_uuid": "TR_1",
            "ontology_uuid": "ONT_1",
            "models": [],
            "datasets": [],
            "experiments": ["EXP_1"],
            "research_programs": [],
        },
    }
    base.update(overrides)
    return base


class TestSearchPlan(unittest.TestCase):
    def test_structured_pass_through(self) -> None:
        plan = build_search_plan("status:ACTIVE domain:price", {SCOPE_NAME})
        self.assertEqual(plan.mode, "structured")
        self.assertEqual(plan.structured_query, "status:ACTIVE domain:price")

    def test_empty_match_all(self) -> None:
        plan = build_search_plan("", {SCOPE_NAME})
        self.assertTrue(plan.match_all)

    def test_name_substring_plan(self) -> None:
        plan = build_search_plan("ema", {SCOPE_NAME})
        self.assertEqual(plan.name_substring, "ema")
        self.assertTrue(plan.match_all)

    def test_prefix_feat(self) -> None:
        plan = build_search_plan("FEAT_abc", {SCOPE_NAME})
        self.assertEqual(plan.engine_queries, ("feature:FEAT_abc",))

    def test_prefix_frr(self) -> None:
        plan = build_search_plan("FRR_xyz", set())
        self.assertTrue(plan.match_all)
        self.assertEqual(plan.research_substring, "FRR_xyz")

    def test_ontology_scope_union_queries(self) -> None:
        plan = build_search_plan("price", {SCOPE_ONTOLOGY})
        self.assertIn("domain:price", plan.engine_queries)
        self.assertIn("signal:price", plan.engine_queries)

    def test_multi_scope_or_fields(self) -> None:
        plan = build_search_plan(
            "ema",
            {SCOPE_NAME, SCOPE_PRIMITIVE, SCOPE_OPERATOR, SCOPE_FEAT},
        )
        self.assertEqual(plan.name_substring, "ema")
        self.assertEqual(plan.feat_substring, "ema")
        self.assertIn("primitive:ema", plan.engine_queries)
        self.assertIn("operator:ema", plan.engine_queries)

    def test_filter_and_merge_hits(self) -> None:
        items = [
            {
                "research_uuid": "FRR_1",
                "feature_uuid": "FEAT_AAA",
                "canonical_name": "ema20",
            },
            {
                "research_uuid": "FRR_2",
                "feature_uuid": "FEAT_BBB",
                "canonical_name": "rsi14",
            },
        ]
        plan = build_search_plan("ema", {SCOPE_NAME})
        filtered = filter_hits_by_plan(items, plan)
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["research_uuid"], "FRR_1")
        merged = merge_hit_lists(filtered, items)
        self.assertEqual(len(merged), 2)
        self.assertEqual(hit_list_label(filtered[0]), f"ema20  [{ABSENT}]")

    def test_hit_grid_columns(self) -> None:
        from feature_intelligence.ui.inspect_format import (
            HIT_GRID_COLUMNS,
            hit_grid_values,
            platform_summary_count_rows,
            platform_summary_version_rows,
            references_are_empty,
        )

        hit = {
            "canonical_name": "ema20",
            "feature_uuid": "FEAT_1",
            "research_status": "ACTIVE",
            "domain": "Price",
            "primary_operator": "OP_EMA",
            "primary_primitive": "PR_SPOT",
            "compiler_version": "1.0.0",
            "ontology_version": "1.0.0",
        }
        vals = hit_grid_values(hit)
        self.assertEqual(len(vals), len(HIT_GRID_COLUMNS))
        self.assertEqual(vals[0], "ema20")
        summary = {
            "counts": {"primitives": 1, "features": 2, "operators": 3,
                       "transformations": 0, "ontology_records": 4, "research_records": 2},
            "versions": {
                "compiler_version": "1.0.0",
                "grammar_pack_version": "1.0.0",
                "grammar_version": "1.0",
                "ontology_version": "1.0.0",
            },
        }
        self.assertEqual(platform_summary_count_rows(summary)[1].value, "2")
        self.assertEqual(platform_summary_version_rows(summary)[0].value, "1.0.0")
        empty_refs = _sample_payload(
            references={
                "models": [],
                "datasets": [],
                "experiments": [],
                "research_programs": [],
            },
            research={"experiment_ids": []},
        )
        self.assertTrue(references_are_empty(empty_refs))


class TestInspectFormatters(unittest.TestCase):
    def test_display_name_and_header(self) -> None:
        p = _sample_payload()
        self.assertEqual(feature_display_name(p), "EMA20 Ratio")
        lines = header_summary_lines(p)
        self.assertTrue(lines[0].startswith("Feature : EMA20 Ratio"))

    def test_architecture_strip(self) -> None:
        parts = architecture_strip(_sample_payload())
        labels = [x[0] for x in parts]
        self.assertEqual(labels, ["FEAT", "TR", "ONT", "FRR"])
        self.assertTrue(all(ok for _, _, ok in parts))

    def test_overview_has_created(self) -> None:
        rows = {r.label: r for r in overview_fields(_sample_payload())}
        self.assertIn("2026-01-01", rows["Created"].value)
        self.assertIn("1 parent", rows["Lineage summary"].value)
        self.assertIn("Price", rows["Ontology summary"].value)

    def test_identity_checksum(self) -> None:
        rows = {r.label: r for r in identity_fields(_sample_payload())}
        self.assertEqual(rows["Checksum (definition_hash)"].value, "deadbeef")

    def test_compiler_stack(self) -> None:
        rows = compiler_stack_summary(_sample_payload())
        labels = [r.label for r in rows]
        self.assertIn("Transformation", labels)
        self.assertIn("AST root operator", labels)
        self.assertIn("Manifest compilation", labels)

    def test_ontology_chips(self) -> None:
        chips = dict((a, b) for a, b, _ in ontology_chip_labels(_sample_payload()))
        self.assertEqual(chips["Domain"], "Price")
        self.assertIn("Trend", chips["Signal"])

    def test_lineage_tree(self) -> None:
        text = lineage_tree_text(_sample_payload())
        self.assertIn("FEAT_parent", text)
        self.assertIn("FEAT_abc", text)

    def test_lineage_empty(self) -> None:
        p = _sample_payload(
            lineage=None,
            sections_present={
                "identity": True,
                "compiler": False,
                "ast": False,
                "ontology": False,
                "lineage": False,
                "research": True,
                "references": True,
            },
        )
        self.assertIn("No lineage", lineage_tree_text(p))

    def test_research_and_references(self) -> None:
        res = {r.label: r for r in research_fields(_sample_payload())}
        self.assertEqual(res["Experiments count"].value, "1")
        refs = {r.label: r for r in references_fields(_sample_payload())}
        # Sample has experiment_ids → Experiments present; models empty
        self.assertEqual(refs["Models"].value, "No references found.")
        self.assertFalse(refs["Models"].present)
        self.assertEqual(refs["Experiments"].value, "1")

    def test_absent_identity(self) -> None:
        p = _sample_payload(
            identity=None,
            sections_present={
                "identity": False,
                "compiler": False,
                "ast": False,
                "ontology": False,
                "lineage": False,
                "research": True,
                "references": True,
            },
        )
        rows = identity_fields(p)
        self.assertEqual(rows[0].value, ABSENT)
        self.assertFalse(rows[0].present)


class TestUiSmokeImport(unittest.TestCase):
    def test_import_panels_without_mainloop(self) -> None:
        from feature_intelligence.ui import (
            FeatureInspectorPanel,
            FeatureIntelligenceSearchBar,
            build_search_plan,
        )
        from master_dataset_tk.feature_intelligence_studio_panel import (
            FeatureIntelligenceStudioPanel,
        )

        self.assertTrue(callable(FeatureInspectorPanel))
        self.assertTrue(callable(FeatureIntelligenceSearchBar))
        self.assertTrue(callable(FeatureIntelligenceStudioPanel))
        self.assertEqual(build_search_plan("x", {SCOPE_NAME}).name_substring, "x")


if __name__ == "__main__":
    unittest.main()
