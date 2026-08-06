"""Primitive catalog data models (Sprint 1)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PrimitiveRecord:
    primitive_id: str
    name: str
    primitive_type: str
    description: str | None
    data_source: str
    units: str
    catalog_version: str
    created_at: str = ""


@dataclass
class ValidationReport:
    passed: bool
    failed_rules: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    seed_hash: str = ""
    expected_seed_hash: str = ""
    validated_objects: str = ""
    timestamp: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "failed_rules": list(self.failed_rules),
            "warnings": list(self.warnings),
            "seed_hash": self.seed_hash,
            "expected_seed_hash": self.expected_seed_hash,
            "validated_objects": self.validated_objects,
            "timestamp": self.timestamp,
        }
