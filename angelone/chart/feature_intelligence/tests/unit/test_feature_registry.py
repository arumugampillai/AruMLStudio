"""Unit tests for Feature Registry service."""

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
from feature_intelligence.registry.feature_definition_hash import compute_definition_hash
from feature_intelligence.registry.feature_import_export import (
    export_features,
    import_features,
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


class TestFeatureRegistry(unittest.TestCase):
    def test_register_validate_find_by_primitive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = _init_db(tmp)
            svc = FeatureRegistryService(db)
            rec = svc.register_feature(
                canonical_name="spot_ema20",
                display_name="Spot EMA20",
                primitive_ids=["PR_SPOT"],
                created_by="test",
                controller_owner="test",
                warmup_periods=20,
                gap_policy="RESET_ON_GAP",
                memory_model="SLIDING_WINDOW",
            )
            self.assertTrue(rec.feature_uuid.startswith("FEAT_"))
            expected = compute_definition_hash(
                canonical_name="spot_ema20",
                warmup_periods=20,
                gap_policy="RESET_ON_GAP",
                memory_model="SLIDING_WINDOW",
                primitive_ids=["PR_SPOT"],
            )
            self.assertEqual(rec.definition_hash, expected)
            report = svc.validate_registry()
            self.assertTrue(report.passed, report.failed_rules)
            self.assertEqual(report.validated_objects, "1 features")
            found = svc.find_by_primitive("PR_SPOT")
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].canonical_name, "spot_ema20")

    def test_export_import_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db = _init_db(tmp)
            svc = FeatureRegistryService(db)
            original = svc.register_feature(
                canonical_name="bid_ask_spread",
                display_name="Bid Ask Spread",
                primitive_ids=["PR_BID", "PR_ASK"],
                created_by="test",
                controller_owner="test",
                warmup_periods=0,
                gap_policy="CONTINUOUS",
                memory_model="TICK",
            )
            out = Path(tmp) / "features.json"
            export_features(svc, out, fmt="json")

            db2 = Path(tmp) / "other.db"
            cfg = FicConfig(
                feature_intelligence=FeatureIntelligenceConfig(
                    schema_version="0.0.1",
                    environment="test",
                    data_dir=Path(tmp),
                ),
                database=DatabaseConfig(path=db2),
                logging=LoggingConfig(level="WARNING"),
            )
            init_database(cfg, apply_migrations=True)
            svc2 = FeatureRegistryService(db2)
            import_features(svc2, out, fmt="json")
            again = svc2.get_by_name("bid_ask_spread")
            self.assertEqual(again.feature_uuid, original.feature_uuid)
            self.assertEqual(set(again.primitive_ids), {"PR_BID", "PR_ASK"})

    def test_mapping_provider(self) -> None:
        p = PrimitiveMappingProvider()
        self.assertEqual(p.resolve({"name": "bid_ask_spread"}), ["PR_BID", "PR_ASK"])
        self.assertEqual(p.resolve({"name": "spot_change_5m"}), ["PR_SPOT"])
        self.assertEqual(p.resolve({"name": "unknown_xyz"}), [])


if __name__ == "__main__":
    unittest.main()
