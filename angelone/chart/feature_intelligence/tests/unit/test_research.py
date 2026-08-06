"""Unit tests for Feature Research Record (Sprint 8)."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from feature_intelligence.migrations.runner import MigrationRunner
from feature_intelligence.registry.feature_service import FeatureRegistryService
from feature_intelligence.registry.models import ValidationReport
from feature_intelligence.research import error_codes as ec
from feature_intelligence.research.identity import derive_research_uuid
from feature_intelligence.research.import_export import (
    export_research,
    import_research,
    record_to_envelope,
)
from feature_intelligence.research.models import (
    RESEARCH_EXPORT_VERSION,
    RESEARCH_VERSION,
    SCHEMA_VERSION,
    SOURCE_IMPORT,
    STATUS_ACTIVE,
    FeatureResearchRecord,
    ResearchSyncSummary,
)
from feature_intelligence.research.service import ResearchService
from feature_intelligence.research.store import (
    ResearchStore,
    compute_research_checksum,
    empty_research_checksum,
)


def _register_feature(db: Path, name: str = "spot_ema20") -> str:
    svc = FeatureRegistryService(db)
    rec = svc.register_feature(
        canonical_name=name,
        display_name=name,
        primitive_ids=["PR_SPOT"],
        created_by="test",
        controller_owner="test",
        warmup_periods=20,
        gap_policy="RESET_ON_GAP",
        memory_model="SLIDING_WINDOW",
    )
    return rec.feature_uuid


class TestResearchIdentity(unittest.TestCase):
    def test_frr_determinism(self) -> None:
        feat = "FEAT_" + "A" * 32
        a = derive_research_uuid(feat)
        b = derive_research_uuid(feat)
        self.assertEqual(a, b)
        self.assertRegex(a, r"^FRR_[0-9A-F]{32}$")

    def test_frr_formula(self) -> None:
        feat = "FEAT_0123456789ABCDEF0123456789ABCDEF"
        expected = "FRR_" + hashlib.sha256(feat.encode("utf-8")).hexdigest()[:32].upper()
        self.assertEqual(derive_research_uuid(feat), expected)

    def test_one_frr_per_feature_mapping(self) -> None:
        f1 = "FEAT_" + "1" * 32
        f2 = "FEAT_" + "2" * 32
        self.assertNotEqual(derive_research_uuid(f1), derive_research_uuid(f2))
        self.assertEqual(derive_research_uuid(f1), derive_research_uuid(f1))

    def test_checksum_empty_and_sorted(self) -> None:
        self.assertEqual(empty_research_checksum(), hashlib.sha256(b"").hexdigest())
        r1 = FeatureResearchRecord(
            research_uuid="FRR_" + "B" * 32,
            feature_uuid="FEAT_" + "B" * 32,
            research_status="EMPTY",
            validation_status="pending",
        )
        r2 = FeatureResearchRecord(
            research_uuid="FRR_" + "A" * 32,
            feature_uuid="FEAT_" + "A" * 32,
            research_status="EMPTY",
            validation_status="pending",
        )
        c1 = compute_research_checksum([r1, r2])
        c2 = compute_research_checksum([r2, r1])
        self.assertEqual(c1, c2)


class TestResearchMigrateAndSync(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.db = Path(self._tmp.name) / "feature_intelligence.db"
        MigrationRunner(self.db).upgrade()
        self.svc = ResearchService(self.db)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_migration_seeds_pack_no_frr_rows(self) -> None:
        self.assertEqual(self.svc.store.count_records(), 0)
        pack = self.svc.store.get_pack()
        assert pack is not None
        self.assertEqual(pack["research_version"], RESEARCH_VERSION)
        self.assertEqual(pack["schema_version"], SCHEMA_VERSION)
        self.assertEqual(pack["checksum"], empty_research_checksum())

    def test_sync_summary_and_one_per_feature(self) -> None:
        feat = _register_feature(self.db)
        summary = self.svc.sync_research()
        self.assertIsInstance(summary, ResearchSyncSummary)
        self.assertEqual(summary.created, 1)
        self.assertEqual(summary.updated, 0)
        self.assertEqual(summary.unchanged, 0)
        self.assertEqual(summary.skipped, 0)

        row = self.svc.get_research_by_feature(feat)
        self.assertEqual(row.research_uuid, derive_research_uuid(feat))
        self.assertEqual(row.research_status, "EMPTY")
        self.assertEqual(row.validation_status, "pending")
        self.assertIsNone(row.evidence_json)
        self.assertEqual(row.record_source, "SYNC")

        again = self.svc.sync_research()
        self.assertEqual(again.created, 0)
        self.assertEqual(again.unchanged, 1)

        self.assertEqual(self.svc.store.count_records(), 1)

    def test_evidence_null_default_on_sync(self) -> None:
        _register_feature(self.db, "ema_null_evidence")
        self.svc.sync_research()
        rows = self.svc.list_research()
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0].evidence_json)
        self.assertIsNone(rows[0].strengths_json)
        self.assertIsNone(rows[0].weaknesses_json)

    def test_stats_policy_validate_sync_import(self) -> None:
        _register_feature(self.db, "stats_feat")
        self.assertEqual(self.svc.store.count_statistics(), 0)

        # stats miss-only regen
        first = self.svc.research_stats()
        self.assertTrue(first.from_snapshot)
        self.assertEqual(self.svc.store.count_statistics(), 1)
        second = self.svc.research_stats()
        self.assertEqual(self.svc.store.count_statistics(), 1)
        self.assertEqual(second.snapshot_created_at, first.snapshot_created_at)

        # sync refreshes stats
        self.svc.sync_research()
        self.assertEqual(self.svc.store.count_statistics(), 2)
        snap = self.svc.store.latest_statistics()
        assert snap is not None
        self.assertIsNotNone(snap["last_sync_at"])

        # validate always writes
        before = self.svc.store.count_statistics()
        report = self.svc.validate_research(mode="strict")
        self.assertTrue(report.passed, report.failed_rules)
        self.assertEqual(self.svc.store.count_statistics(), before + 1)

        # import never writes stats
        out = Path(self._tmp.name) / "frr.json"
        export_research(self.svc, out, fmt="json")
        before_imp = self.svc.store.count_statistics()
        imp = import_research(self.svc, out, fmt="json")
        self.assertTrue(imp.passed, imp.failed_rules)
        self.assertEqual(self.svc.store.count_statistics(), before_imp)

    def test_completeness_reporting_not_validity_gate(self) -> None:
        _register_feature(self.db, "gap_feat")
        self.svc.sync_research()
        comp = self.svc.research_completeness()
        self.assertEqual(comp.total_frr, 1)
        self.assertEqual(comp.incomplete, 1)
        self.assertIn("ontology_uuid", comp.gaps[0].missing_fields)

        report = self.svc.validate_research(mode="strict")
        self.assertTrue(report.passed, report.failed_rules)
        self.assertNotIn("MISSING_LINK", report.failed_rules)

    def test_strict_missing_frr(self) -> None:
        _register_feature(self.db, "need_sync")
        report = self.svc.validate_research(mode="strict")
        self.assertFalse(report.passed)
        self.assertIn(ec.MISSING_FRR, report.failed_rules)

    def test_present_mode_warns_not_fails(self) -> None:
        _register_feature(self.db, "present_mode")
        report = self.svc.validate_research(mode="present")
        self.assertTrue(report.passed, report.failed_rules)
        self.assertTrue(any("missing_frr" in w for w in report.warnings))

    def test_import_export_roundtrip(self) -> None:
        feat = _register_feature(self.db, "roundtrip")
        self.svc.sync_research()
        frr = self.svc.get_research_by_feature(feat)
        # Import may set evidence
        updated = FeatureResearchRecord(
            research_uuid=frr.research_uuid,
            feature_uuid=frr.feature_uuid,
            research_status=STATUS_ACTIVE,
            validation_status="pending",
            evidence_json='{"src":"import"}',
            notes="imported",
            record_source=SOURCE_IMPORT,
            created_at=frr.created_at,
        )
        self.svc.store.upsert_record(updated)

        out = Path(self._tmp.name) / "export.json"
        export_research(self.svc, out, fmt="json")
        data = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(data["research_export_version"], RESEARCH_EXPORT_VERSION)
        self.assertEqual(data["schema_version"], SCHEMA_VERSION)

        # Fresh DB with same feature
        db2 = Path(self._tmp.name) / "other.db"
        MigrationRunner(db2).upgrade()
        FeatureRegistryService(db2).register_feature(
            canonical_name="roundtrip",
            display_name="roundtrip",
            primitive_ids=["PR_SPOT"],
            created_by="test",
            controller_owner="test",
            warmup_periods=20,
            gap_policy="RESET_ON_GAP",
            memory_model="SLIDING_WINDOW",
        )
        # Force same FEAT by using exported feature_uuid — re-register may differ.
        # Instead insert via first feature's uuid already in export; recreate feature
        # with matching uuid by syncing after importing feature registry path is hard.
        # Simpler: import into same DB after wipe of FRR is unnecessary —
        # import into db2 after inserting the exact feature_uuid.
        store2 = ResearchStore(db2)
        # Copy feature row identity from export
        feat_uuid = data["records"][0]["feature_uuid"]
        # Ensure feature exists with that uuid
        import sqlite3

        conn = sqlite3.connect(str(db2))
        try:
            # Delete auto-registered feature and insert exact uuid
            conn.execute("DELETE FROM feature_primitives")
            conn.execute("DELETE FROM feature_registry")
            conn.execute(
                """
                INSERT INTO feature_registry(
                    feature_uuid, canonical_name, display_name,
                    definition_version, implementation_version, feature_version,
                    definition_hash, created_by, controller_owner,
                    warmup_periods, gap_policy, memory_model, research_state
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    feat_uuid,
                    "roundtrip",
                    "roundtrip",
                    "1.0.0",
                    "1.0.0",
                    "1.0.0",
                    "hash",
                    "test",
                    "test",
                    20,
                    "RESET_ON_GAP",
                    "SLIDING_WINDOW",
                    "EXPERIMENTAL",
                ),
            )
            conn.execute(
                "INSERT INTO feature_primitives(feature_uuid, primitive_id, ordinal) VALUES (?,?,?)",
                (feat_uuid, "PR_SPOT", 0),
            )
            conn.commit()
        finally:
            conn.close()

        svc2 = ResearchService(db2)
        report = import_research(svc2, out, fmt="json")
        self.assertTrue(report.passed, report.failed_rules)
        again = svc2.get_research_by_feature(feat_uuid)
        self.assertEqual(again.research_uuid, frr.research_uuid)
        self.assertEqual(again.evidence_json, '{"src":"import"}')
        self.assertEqual(again.research_status, STATUS_ACTIVE)

        out2 = Path(self._tmp.name) / "export2.json"
        export_research(svc2, out2, fmt="json")
        d2 = json.loads(out2.read_text(encoding="utf-8"))
        # Compare records ignoring timestamps (not in envelope)
        self.assertEqual(
            record_to_envelope(again),
            d2["records"][0],
        )

    def test_error_codes_catalog(self) -> None:
        self.assertIn(ec.MISSING_FRR, ec.ALL_ERROR_CODES)
        self.assertIn(ec.EVIDENCE_COMPUTED, ec.ALL_ERROR_CODES)
        self.assertGreaterEqual(len(ec.ALL_ERROR_CODES), 15)

    def test_no_frr_for_non_feat(self) -> None:
        summary = self.svc.sync_research(feature_uuid="PR_SPOT")
        self.assertEqual(summary.skipped, 1)
        self.assertEqual(self.svc.store.count_records(), 0)


class TestResearchValidateReport(unittest.TestCase):
    def test_validate_returns_shared_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "feature_intelligence.db"
            MigrationRunner(db).upgrade()
            svc = ResearchService(db)
            report = svc.validate_research(mode="present")
            self.assertIsInstance(report, ValidationReport)
            self.assertTrue(report.passed)


if __name__ == "__main__":
    unittest.main()
