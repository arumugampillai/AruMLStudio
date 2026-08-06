"""Tests for FEAT_ UUIDv7 identity."""

from __future__ import annotations

import re
import unittest
import uuid

from feature_intelligence.registry.feature_ids import (
    FEATURE_UUID_PATTERN,
    generate_feature_uuid,
    is_valid_feature_uuid,
    normalize_feature_uuid,
)


class TestFeatureIds(unittest.TestCase):
    def test_generate_matches_pattern_and_v7(self) -> None:
        fu = generate_feature_uuid()
        self.assertRegex(fu, FEATURE_UUID_PATTERN.pattern)
        body = fu.removeprefix("FEAT_")
        u = uuid.UUID(hex=body)
        self.assertEqual(u.version, 7)
        self.assertTrue(is_valid_feature_uuid(fu))

    def test_normalize_hyphenated(self) -> None:
        fu = generate_feature_uuid()
        body = fu.removeprefix("FEAT_")
        hex_part = (
            f"{body[0:8]}-{body[8:12]}-{body[12:16]}-"
            f"{body[16:20]}-{body[20:32]}"
        ).lower()
        hyphen = f"FEAT_{hex_part}"
        self.assertEqual(normalize_feature_uuid(hyphen), fu)

    def test_reject_v4(self) -> None:
        v4 = "FEAT_" + uuid.uuid4().hex.upper()
        with self.assertRaises(ValueError):
            normalize_feature_uuid(v4)


if __name__ == "__main__":
    unittest.main()
