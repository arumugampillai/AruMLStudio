"""Post-training Feature Studio pipeline (Phase 5.1).

Model Builder only calls ``post_training.run(package_dir, data_dir)``.
Studios stay independent compute modules (Compute → Artifacts → UI).
"""

from __future__ import annotations

from .config import normalize_post_training_config, resolve_post_training_config
from .orchestrator import run, run_safe
from .status import (
    STATUS_FILENAME,
    format_readiness_line,
    load_feature_studio_status,
    status_path,
    write_feature_studio_status,
)

__all__ = [
    "STATUS_FILENAME",
    "format_readiness_line",
    "load_feature_studio_status",
    "normalize_post_training_config",
    "resolve_post_training_config",
    "run",
    "run_safe",
    "status_path",
    "write_feature_studio_status",
]
