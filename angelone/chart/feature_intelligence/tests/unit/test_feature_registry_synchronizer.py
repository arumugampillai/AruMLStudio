"""Unit tests for Feature Registry Synchronizer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from feature_intelligence.core.config import (
    DatabaseConfig,
    FeatureIntelligenceConfig,
    FicConfig,
    LoggingConfig,
)
from feature_intelligence.core.database import init_database
from feature_intelligence.registry.feature_ids import generate_feature_uuid
from feature_intelligence.registry.feature_registry_synchronizer import (
    extract_legacy_feature_id,
    extract_source_feature_uuid,
    synchronize_feature_registry,
)
from feature_intelligence.registry.feature_service import FeatureRegistryService
from feature_intelligence.registry.primitive_mapping import PrimitiveMappingProvider


def _init_db(tmp: str) -> Path:
    db = Path(tmp) / "feature_intelligence.db"
    cfg = FicConfig(
        feature_intelligence=FeatureIntelligenceConfig(
            schema_version="0.0.1",
            environment="test",
            data_dir=Path(tmp),
        ),
        database=DatabaseConfig(path=db),
        logging=LoggingConfig(level="WARNING"),
    )
    init_database(cfg, apply_migrations=True)
    return db


def _feat(
    name: str,
    *,
    feature_id: str | None = "FR0001",
    feature_uuid: str | None = None,
    status: str = "implemented",
    active: bool = True,
    display_name: str | None = None,
) -> dict:
    row: dict = {
        "name": name,
        "display_name": display_name or name,
        "implementation_status": status,
        "registry_active": active,
        "owner": "test",
        "created_by": "test",
        "feature_version": "1.0",
        "policy": {
            "lifecycle": "tick",
            "intrinsic_warmup_samples": 0,
            "policy_version": "1",
        },
    }
    if feature_id is not None:
        row["feature_id"] = feature_id
    if feature_uuid is not None:
        row["feature_uuid"] = feature_uuid
    return row


class TestExtractIds(unittest.TestCase):
    def test_legacy_fr_not_feat(self) -> None:
        self.assertIsNone(extract_source_feature_uuid({"feature_id": "FR0001"}))
        self.assertEqual(extract_legacy_feature_id({"feature_id": "FR0001"}), "FR0001")

    def test_feat_from_feature_uuid(self) -> None:
        fu = generate_feature_uuid()
        self.assertEqual(
            extract_source_feature_uuid({"feature_uuid": fu, "feature_id": "FR0001"}),
            fu,
        )
        self.assertEqual(
            extract_legacy_feature_id({"feature_uuid": fu, "feature_id": "FR0001"}),
            "FR0001",
        )


class TestSynchronize(unittest.TestCase):
    def test_import_skip_idempotent_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = _init_db(tmp)
            svc = FeatureRegistryService(db)
            features = [
                _feat("spot_change_5m", feature_id="FR0100"),
                _feat("unknown_no_map_xyz", feature_id="FR9999"),
            ]
            s1 = synchronize_feature_registry(
                svc,
                features=features,
                mode="lenient",
                research_sync=False,
            )
            self.assertEqual(s1.total_source, 2)
            self.assertEqual(s1.newly_imported, 1)
            self.assertEqual(s1.already_registered, 0)
            self.assertEqual(s1.failed, 1)
            self.assertEqual(s1.failures[0].reason, "UNMAPPED_PRIMITIVES")
            first = svc.get_by_name("spot_change_5m")
            self.assertTrue(first.feature_uuid.startswith("FEAT_"))
            self.assertEqual(first.legacy_feature_id, "FR0100")

            s2 = synchronize_feature_registry(
                svc,
                features=features,
                mode="lenient",
                research_sync=False,
            )
            self.assertEqual(s2.newly_imported, 0)
            self.assertEqual(s2.already_registered, 1)
            self.assertEqual(s2.failed, 1)
            again = svc.get_by_name("spot_change_5m")
            self.assertEqual(again.feature_uuid, first.feature_uuid)

    def test_preserve_source_feat_uuid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = _init_db(tmp)
            svc = FeatureRegistryService(db)
            preserved = generate_feature_uuid()
            features = [
                _feat(
                    "bid_ask_spread",
                    feature_id="FR0200",
                    feature_uuid=preserved,
                )
            ]
            summary = synchronize_feature_registry(
                svc,
                features=features,
                research_sync=False,
            )
            self.assertEqual(summary.newly_imported, 1)
            rec = svc.get_by_name("bid_ask_spread")
            self.assertEqual(rec.feature_uuid, preserved)
            self.assertEqual(rec.legacy_feature_id, "FR0200")

            # Re-run matches by FEAT id / name — no second insert
            summary2 = synchronize_feature_registry(
                svc,
                features=features,
                research_sync=False,
            )
            self.assertEqual(summary2.already_registered, 1)
            self.assertEqual(summary2.newly_imported, 0)
            self.assertEqual(svc.get_by_uuid(preserved).canonical_name, "bid_ask_spread")

    def test_match_by_feat_when_name_lookup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = _init_db(tmp)
            svc = FeatureRegistryService(db)
            preserved = generate_feature_uuid()
            synchronize_feature_registry(
                svc,
                features=[
                    _feat("spot", feature_id="FR0001", feature_uuid=preserved)
                ],
                research_sync=False,
            )
            # Source omits uuid on second pass — name match preserves FEAT
            s2 = synchronize_feature_registry(
                svc,
                features=[_feat("spot", feature_id="FR0001")],
                research_sync=False,
            )
            self.assertEqual(s2.already_registered, 1)
            self.assertEqual(svc.get_by_name("spot").feature_uuid, preserved)

    def test_research_sync_creates_frr(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = _init_db(tmp)
            svc = FeatureRegistryService(db)
            summary = synchronize_feature_registry(
                svc,
                features=[_feat("spot_change_5m", feature_id="FR0100")],
                research_sync=True,
            )
            self.assertEqual(summary.newly_imported, 1)
            self.assertGreaterEqual(summary.research_created, 1)
            from feature_intelligence.research.service import ResearchService

            rs = ResearchService(db)
            feat = svc.get_by_name("spot_change_5m")
            fr = rs.get_research_by_feature(feat.feature_uuid)
            self.assertTrue(fr.research_uuid.startswith("FRR_"))


class TestPrimitiveMappingExtensions(unittest.TestCase):
    def test_reasonable_mappings(self) -> None:
        p = PrimitiveMappingProvider()
        self.assertEqual(p.resolve({"name": "atm_iv_ce"}), ["PR_IV"])
        self.assertEqual(p.resolve({"name": "charm"}), ["PR_DELTA"])
        self.assertEqual(p.resolve({"name": "futures_bid"}), ["PR_BID"])
        self.assertEqual(p.resolve({"name": "ltq"}), ["PR_VOLUME"])
        self.assertEqual(
            p.resolve({"name": "book_imbalance_l1"}), ["PR_BID", "PR_ASK"]
        )
        # Still leave ambiguous / non-seed features unmapped
        self.assertEqual(p.resolve({"name": "ltp"}), [])
        self.assertEqual(p.resolve({"name": "unknown_xyz"}), [])
        self.assertEqual(
            p.resolve({"name": "whatever", "primary_domain": "implied_volatility"}),
            ["PR_IV"],
        )


if __name__ == "__main__":
    unittest.main()
