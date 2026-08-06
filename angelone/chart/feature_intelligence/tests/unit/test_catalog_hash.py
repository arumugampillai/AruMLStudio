"""Unit tests for seed catalog hash integrity."""

from __future__ import annotations

import unittest

from feature_intelligence.registry.catalog import (
    EXPECTED_SEED_CATALOG_HASH,
    SEED_PRIMITIVES,
    compute_seed_catalog_hash,
)
from feature_intelligence.registry.traceability import is_valid_primitive_id, looks_like_uuid


class TestCatalogHash(unittest.TestCase):
    def test_expected_hash_matches_seed(self) -> None:
        self.assertEqual(compute_seed_catalog_hash(), EXPECTED_SEED_CATALOG_HASH)

    def test_hash_detects_seed_edit(self) -> None:
        tampered = list(SEED_PRIMITIVES)
        first = tampered[0]
        tampered[0] = first._replace(description=first.description + " X")
        self.assertNotEqual(
            compute_seed_catalog_hash(tuple(tampered)),
            EXPECTED_SEED_CATALOG_HASH,
        )

    def test_seed_ids_are_deterministic(self) -> None:
        self.assertEqual(len(SEED_PRIMITIVES), 14)
        for p in SEED_PRIMITIVES:
            self.assertTrue(is_valid_primitive_id(p.primitive_id))
            self.assertFalse(looks_like_uuid(p.primitive_id))
            self.assertEqual(p.catalog_version, "1.0")


if __name__ == "__main__":
    unittest.main()
