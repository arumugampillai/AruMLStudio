"""Unit tests for Feature Ontology (Sprint 6)."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from feature_intelligence.migrations.runner import MigrationRunner
from feature_intelligence.ontology.catalog import (
    EXPECTED_ONTOLOGY_SEED_HASH,
    EXPECTED_VOCAB_SEED_HASH,
    ONTOLOGY_VERSION,
    SEED_ONTOLOGY_ROWS,
    SEED_VOCABULARIES,
    compute_ontology_seed_hash,
    compute_vocab_seed_hash,
)
from feature_intelligence.ontology.error_codes import ALL_ERROR_CODES
from feature_intelligence.ontology.identity import derive_ontology_uuid
from feature_intelligence.ontology.import_export import (
    export_ontology,
    import_ontology,
    record_to_envelope,
)
from feature_intelligence.ontology.models import (
    OBJECT_TYPE_TABLE,
    OntologyRecord,
    normalize_id_list,
)
from feature_intelligence.ontology.service import OntologyService
from feature_intelligence.registry.models import ValidationReport


class TestOntologyVocabFreeze(unittest.TestCase):
    def test_vocab_seed_hash_matches_expected(self) -> None:
        self.assertEqual(compute_vocab_seed_hash(), EXPECTED_VOCAB_SEED_HASH)

    def test_ontology_seed_hash_matches_expected(self) -> None:
        self.assertEqual(compute_ontology_seed_hash(), EXPECTED_ONTOLOGY_SEED_HASH)

    def test_vocab_count_and_prefixes(self) -> None:
        self.assertEqual(len(SEED_VOCABULARIES), 64)
        by_type: dict[str, int] = {}
        for v in SEED_VOCABULARIES:
            by_type[v.vocabulary_type] = by_type.get(v.vocabulary_type, 0) + 1
            self.assertTrue(v.active)
            self.assertEqual(v.ontology_version, ONTOLOGY_VERSION)
        self.assertEqual(by_type["DOMAIN"], 11)
        self.assertEqual(by_type["SIGNAL_TYPE"], 15)
        self.assertEqual(by_type["MATH_FAMILY"], 13)
        self.assertEqual(by_type["HORIZON"], 8)
        self.assertEqual(by_type["OUTPUT_TYPE"], 5)
        self.assertEqual(by_type["FREQUENCY"], 9)
        self.assertEqual(by_type["STABILITY"], 3)

    def test_required_seed_counts(self) -> None:
        pr = [r for r in SEED_ONTOLOGY_ROWS if r.object_type == "PRIMITIVE"]
        op = [r for r in SEED_ONTOLOGY_ROWS if r.object_type == "OPERATOR"]
        self.assertEqual(len(pr), 14)
        self.assertEqual(len(op), 31)

    def test_error_codes_catalog_complete(self) -> None:
        self.assertGreaterEqual(len(ALL_ERROR_CODES), 19)

    def test_normalize_id_list_dedupe_sort(self) -> None:
        self.assertEqual(
            normalize_id_list(["SIG_RAW", "SIG_LEVEL", "SIG_RAW"]),
            ["SIG_LEVEL", "SIG_RAW"],
        )


class TestOntologyIdentity(unittest.TestCase):
    def test_ont_determinism(self) -> None:
        a = derive_ontology_uuid("PRIMITIVE", "PR_SPOT")
        b = derive_ontology_uuid("PRIMITIVE", "PR_SPOT")
        self.assertEqual(a, b)
        self.assertRegex(a, r"^ONT_[0-9A-F]{32}$")

    def test_ont_differs_by_type_and_id(self) -> None:
        self.assertNotEqual(
            derive_ontology_uuid("PRIMITIVE", "PR_SPOT"),
            derive_ontology_uuid("OPERATOR", "OP_EMA"),
        )
        self.assertNotEqual(
            derive_ontology_uuid("PRIMITIVE", "PR_SPOT"),
            derive_ontology_uuid("PRIMITIVE", "PR_ASK"),
        )

    def test_object_type_table_map_single(self) -> None:
        self.assertEqual(OBJECT_TYPE_TABLE["PRIMITIVE"], "primitive_ontology")
        self.assertEqual(len(OBJECT_TYPE_TABLE), 4)


class TestOntologyMigrateCoverage(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "feature_intelligence.db"
        MigrationRunner(self.db).upgrade()
        self.svc = OntologyService(self.db)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_required_coverage_100_after_migrate(self) -> None:
        cov = self.svc.compute_coverage_metrics()
        self.assertEqual(cov.by_type["PRIMITIVE"].coverage_pct, 100.0)
        self.assertEqual(cov.by_type["OPERATOR"].coverage_pct, 100.0)
        self.assertEqual(cov.by_type["PRIMITIVE"].expected, 14)
        self.assertEqual(cov.by_type["OPERATOR"].expected, 31)
        self.assertEqual(cov.by_type["PRIMITIVE"].classified, 14)
        self.assertEqual(cov.by_type["OPERATOR"].classified, 31)

    def test_strict_validate_passes(self) -> None:
        report = self.svc.validate_ontology(mode="strict")
        self.assertIsInstance(report, ValidationReport)
        self.assertTrue(report.passed, report.failed_rules)
        self.assertEqual(report.seed_hash, EXPECTED_VOCAB_SEED_HASH)

    def test_validate_always_writes_snapshot(self) -> None:
        self.assertEqual(self.svc.store.count_statistics(), 0)
        self.svc.validate_ontology(mode="strict")
        self.assertEqual(self.svc.store.count_statistics(), 1)
        self.svc.validate_ontology(mode="present")
        self.assertEqual(self.svc.store.count_statistics(), 2)

    def test_coverage_reads_then_miss_regen(self) -> None:
        self.assertEqual(self.svc.store.count_statistics(), 0)
        first = self.svc.coverage_ontology()
        self.assertTrue(first.from_snapshot)
        self.assertEqual(self.svc.store.count_statistics(), 1)
        second = self.svc.coverage_ontology()
        self.assertTrue(second.from_snapshot)
        self.assertEqual(self.svc.store.count_statistics(), 1)
        self.assertEqual(second.snapshot_created_at, first.snapshot_created_at)

    def test_import_never_writes_snapshot(self) -> None:
        self.assertEqual(self.svc.store.count_statistics(), 0)
        # Re-import an existing primitive row via JSON — must not create stats
        row = self.svc.get_ontology("PRIMITIVE", "PR_SPOT")
        envelope = {
            "schema_version": "1.0",
            "ontology_version": ONTOLOGY_VERSION,
            "vocab_pack_version": "1.0.0",
            "records": [record_to_envelope(row)],
        }
        path = Path(self._tmp.name) / "ont.json"
        path.write_text(json.dumps(envelope), encoding="utf-8")
        report = import_ontology(self.svc, path, fmt="json")
        self.assertTrue(report.passed, report.failed_rules)
        self.assertEqual(self.svc.store.count_statistics(), 0)

    def test_import_export_roundtrip_json(self) -> None:
        out1 = Path(self._tmp.name) / "e1.json"
        out2 = Path(self._tmp.name) / "e2.json"
        export_ontology(self.svc, out1, fmt="json")
        import_ontology(self.svc, out1, fmt="json")
        export_ontology(self.svc, out2, fmt="json")
        a = json.loads(out1.read_text(encoding="utf-8"))
        b = json.loads(out2.read_text(encoding="utf-8"))
        # Ignore timestamps
        a.pop("exported_at", None)
        b.pop("exported_at", None)
        self.assertEqual(a["records"], b["records"])

    def test_import_export_roundtrip_csv(self) -> None:
        out1 = Path(self._tmp.name) / "e1.csv"
        out2 = Path(self._tmp.name) / "e2.csv"
        export_ontology(self.svc, out1, fmt="csv")
        import_ontology(self.svc, out1, fmt="csv")
        export_ontology(self.svc, out2, fmt="csv")
        self.assertEqual(
            out1.read_text(encoding="utf-8"),
            out2.read_text(encoding="utf-8"),
        )

    def test_reimport_updates_in_place_same_ont(self) -> None:
        before = self.svc.get_ontology("OPERATOR", "OP_EMA")
        ont = before.ontology_uuid
        updated = OntologyRecord(
            ontology_uuid=ont,
            object_type="OPERATOR",
            object_id="OP_EMA",
            ontology_version=ONTOLOGY_VERSION,
            domain="DOM_DERIVED",
            signal_type=["SIG_TREND"],
            mathematical_family=["MATH_MOVING_AVERAGE"],
            horizon="HOR_INTRADAY",
            output_type="OUT_NUMERIC",
            frequency="FREQ_ANY",
            stability="STAB_EXPERIMENTAL",
            input_dependencies=[],
            meaning="reclass test",
            confidence=None,
            classification_source="IMPORT",
        )
        after = self.svc.upsert_ontology(updated)
        self.assertEqual(after.ontology_uuid, ont)
        self.assertEqual(after.stability, "STAB_EXPERIMENTAL")
        self.assertEqual(after.classification_source, "IMPORT")

    def test_vocabulary_pk_not_in_export(self) -> None:
        out = Path(self._tmp.name) / "exp.json"
        export_ontology(self.svc, out, fmt="json")
        text = out.read_text(encoding="utf-8")
        self.assertNotIn("vocabulary_pk", text)

    def test_confidence_null(self) -> None:
        for row in self.svc.list_ontology():
            self.assertIsNone(row.confidence)

    def test_classification_source_seed(self) -> None:
        row = self.svc.get_ontology("PRIMITIVE", "PR_SPOT")
        self.assertEqual(row.classification_source, "SEED")


if __name__ == "__main__":
    unittest.main()
