"""Phase X — Label Runs: decoupled labels for immutable feature datasets.

Feature Dataset (parquet) stays unchanged. OLE writes small Label Run artifacts
under ``data/label_runs/``. Create Model joins Feature Dataset + one Label Run
once during ``load_training_xy``.
"""

from __future__ import annotations

from .join import join_feature_frame_with_label_run, resolve_join_keys
from .paths import label_runs_dir, mint_label_run_id
from .promote import promote_feature_column_to_label_run
from .registry import (
    get_label_run,
    label_run_exists,
    list_label_runs,
    load_label_run_frame,
    load_label_run_meta,
)
from .triple_barrier_run import create_triple_barrier_label_run
from .types import JOIN_KEY_PREFERENCE, LabelRunRecord, label_run_meta_template
from .writer import write_label_run

__all__ = [
    "JOIN_KEY_PREFERENCE",
    "LabelRunRecord",
    "create_triple_barrier_label_run",
    "get_label_run",
    "join_feature_frame_with_label_run",
    "label_run_exists",
    "label_run_meta_template",
    "label_runs_dir",
    "list_label_runs",
    "load_label_run_frame",
    "load_label_run_meta",
    "mint_label_run_id",
    "promote_feature_column_to_label_run",
    "resolve_join_keys",
    "write_label_run",
]
