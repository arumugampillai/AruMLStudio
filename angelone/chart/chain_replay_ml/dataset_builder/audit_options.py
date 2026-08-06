"""Skip flags for dataset audit / validation in production vs fast-experiment builds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def audit_validation_required_for_dataset(meta: dict[str, Any] | None) -> bool:
    """When False, Model Builder may train without completed audit and validation."""
    meta = meta or {}
    if "audit_validation_required" in meta:
        return bool(meta["audit_validation_required"])
    export_src = str(meta.get("export_source") or meta.get("source") or "").lower()
    if export_src == "master_filter_export":
        return False
    if export_src == "ormp_dataset_builder" or str(meta.get("dataset_kind") or "").upper() == "ORMP":
        return False
    if str(meta.get("build_profile") or "").lower() == "fast_experiment":
        return False
    if bool(meta.get("validation_skipped")):
        return False
    if bool(meta.get("allow_training_without_audit")):
        return False
    return True


@dataclass
class AuditOptions:
    build_profile: str = "production"
    skip_feature_audit: bool = False
    skip_data_validation: bool = False
    skip_dataset_statistics: bool = False
    skip_distribution_report: bool = False
    skip_leakage_audit: bool = False
    skip_quality_report: bool = False
    allow_training_without_audit: bool = False

    @classmethod
    def from_mapping(cls, doc: dict[str, Any] | None) -> AuditOptions:
        doc = doc or {}
        profile = str(doc.get("build_profile") or "production").lower()
        fast = profile == "fast_experiment"

        def _flag(key: str, default: bool = False) -> bool:
            if key in doc:
                return bool(doc.get(key))
            return default

        return cls(
            build_profile=profile,
            skip_feature_audit=_flag("skip_feature_audit", fast),
            skip_data_validation=_flag("skip_data_validation", fast),
            skip_dataset_statistics=_flag("skip_dataset_statistics", fast),
            skip_distribution_report=_flag("skip_distribution_report", fast),
            skip_leakage_audit=_flag("skip_leakage_audit", fast),
            skip_quality_report=_flag("skip_quality_report", fast),
            allow_training_without_audit=_flag(
                "allow_training_without_audit",
                fast or profile == "fast_experiment",
            ),
        )

    def audit_heavy_work_skipped(self) -> bool:
        return (
            self.skip_feature_audit
            and self.skip_dataset_statistics
            and self.skip_distribution_report
            and self.skip_leakage_audit
            and self.skip_quality_report
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "build_profile": self.build_profile,
            "skip_feature_audit": self.skip_feature_audit,
            "skip_data_validation": self.skip_data_validation,
            "skip_dataset_statistics": self.skip_dataset_statistics,
            "skip_distribution_report": self.skip_distribution_report,
            "skip_leakage_audit": self.skip_leakage_audit,
            "skip_quality_report": self.skip_quality_report,
            "allow_training_without_audit": self.allow_training_without_audit,
        }
