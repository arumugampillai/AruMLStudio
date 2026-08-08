"""Drop unselected Registry columns from Analysis Dataset Parquet (post-transform)."""

from __future__ import annotations

from typing import Any, Callable

from .feature_sources_catalog import registry_feature_names


def prune_registry_columns_in_parquet(
    parquet_path: str,
    *,
    selected_registry: frozenset[str],
    on_progress: Callable[[str], None] | None = None,
) -> list[str]:
    """Remove registry feature columns not in ``selected_registry``. Returns dropped names."""
    reg_all = frozenset(registry_feature_names())
    if not selected_registry or selected_registry >= reg_all:
        return []

    try:
        from .transformations.lag_ui import META_SKIP_COLUMNS
    except Exception:
        META_SKIP_COLUMNS = frozenset()

    import pandas as pd

    from .writer import _write_parquet, ensure_parquet_engine

    ensure_parquet_engine()
    frame = pd.read_parquet(parquet_path)
    drop_cols = [
        c
        for c in frame.columns
        if c in reg_all and c not in selected_registry and c not in META_SKIP_COLUMNS
    ]
    if not drop_cols:
        return []
    if on_progress:
        on_progress(
            f"Registry export selection: dropping {len(drop_cols)} unselected "
            f"Registry Features from Analysis Dataset…"
        )
    frame = frame.drop(columns=drop_cols, errors="ignore")
    _write_parquet(frame, parquet_path)
    return drop_cols


__all__ = ["prune_registry_columns_in_parquet"]
