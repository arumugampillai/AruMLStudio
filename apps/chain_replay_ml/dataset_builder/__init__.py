"""ML dataset build pipeline — orchestrates replay DB → Parquet + metadata JSON."""

from .auditor import audit_dataset, compare_datasets, delete_dataset, list_datasets
from .feature_merge_ops import (
    get_feature_merge_job,
    merge_features_into_dataset,
    plan_feature_merge,
    start_feature_merge_job,
)
from .dataset_summary import build_dataset_summary
from .audit_rca import investigate_audit_failure
from .expected_spec import build_expected_spec, write_expected_spec
from .golden_regression import (
    build_manifest_from_dataset,
    golden_regression_status,
    run_golden_regression,
)
from .orchestrator import DatasetBuildConfig, DatasetBuildOrchestrator
from .progress import BuildProgress

__all__ = [
    "DatasetBuildConfig",
    "DatasetBuildOrchestrator",
    "BuildProgress",
    "audit_dataset",
    "compare_datasets",
    "plan_feature_merge",
    "merge_features_into_dataset",
    "start_feature_merge_job",
    "get_feature_merge_job",
    "delete_dataset",
    "list_datasets",
    "investigate_audit_failure",
    "build_expected_spec",
    "write_expected_spec",
    "golden_regression_status",
    "run_golden_regression",
    "build_dataset_summary",
    "build_manifest_from_dataset",
]
