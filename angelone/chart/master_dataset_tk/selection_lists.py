"""Global sort standard for Dataset & Model dropdowns (Tkinter app).

Single source of truth for how every Dataset / Model combobox across the
``master_dataset_tk`` app is ordered, and for how the current selection is
preserved (or defaulted to the newest item) when a combobox is refreshed.

Do NOT reimplement this sort/selection logic in individual panels — import
``get_sorted_datasets`` / ``get_sorted_models`` (or the ``*_names`` variants)
and ``refresh_combobox`` / ``pick_preserved_or_default`` instead.

Dataset sort order
-------------------
1. Creation time (newest first)
2. Else last modified time (newest first)
3. Name (A → Z) as the final tie-breaker

Model sort order
-----------------
1. Most recently trained/created first
2. Else last modified time
3. Name as the final tie-breaker

Behavior contract for combobox refresh
---------------------------------------
- Values are set newest-first (index 0 == newest).
- If the currently selected item still exists in the refreshed list, it is
  kept selected.
- Otherwise the newest item (index 0) is selected automatically.
- If the list is empty, the selection is cleared.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover - typing only
    import tkinter as tk
    from tkinter import ttk


def _parse_timestamp(value: Any) -> float | None:
    """Best-effort parse of a timestamp field into a comparable epoch float."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        try:
            return float(text)
        except (TypeError, ValueError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _path_mtime(path: Any) -> float | None:
    if not path:
        return None
    try:
        return os.path.getmtime(str(path))
    except OSError:
        return None


def _dataset_effective_ts(row: dict[str, Any]) -> float:
    """Creation time if present, else newest file mtime, else -inf (sorts last)."""
    ts = _parse_timestamp(row.get("created_at"))
    if ts is not None:
        return ts
    for key in ("parquet_path", "metadata_path"):
        ts = _path_mtime(row.get(key))
        if ts is not None:
            return ts
    return float("-inf")


def _dataset_sort_key(row: dict[str, Any]) -> tuple[float, str]:
    name = str(row.get("dataset_name") or row.get("name") or "")
    return (-_dataset_effective_ts(row), name.lower())


def sort_dataset_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort dataset rows per the global standard (newest first, name tie-break)."""
    return sorted((dict(r) for r in rows), key=_dataset_sort_key)


def get_sorted_datasets(
    data_dir: str | None = None,
    *,
    rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Return dataset registry rows sorted by the global standard.

    Pass ``rows`` to re-sort an already-fetched / filtered row list (e.g. the
    output of ``model_builder.service.list_builder_datasets``). Otherwise
    ``data_dir`` is used to fetch the full dataset registry via
    ``chain_replay_ml.dataset_builder.list_datasets``.
    """
    if rows is None:
        if not data_dir:
            return []
        from chain_replay_ml.dataset_builder import list_datasets

        rows = list_datasets(data_dir)
    return sort_dataset_rows(rows)


def get_sorted_dataset_names(
    data_dir: str | None = None,
    *,
    rows: list[dict[str, Any]] | None = None,
    name_key: str = "dataset_name",
) -> list[str]:
    """Convenience wrapper returning just the sorted, de-duplicated names."""
    sorted_rows = get_sorted_datasets(data_dir, rows=rows)
    return _names_from_rows(sorted_rows, name_key)


def _model_effective_ts(data_dir: str | None, row: dict[str, Any]) -> float:
    """Trained/created time if present, else package dir mtime, else -inf."""
    ts = _parse_timestamp(row.get("trained_at") or row.get("created_at"))
    if ts is not None:
        return ts
    if data_dir:
        name = str(row.get("model_name") or row.get("name") or "").strip()
        if name:
            try:
                from chain_replay_ml.training.paths import model_package_dir

                ts = _path_mtime(model_package_dir(data_dir, name))
                if ts is not None:
                    return ts
            except Exception:
                pass
    return float("-inf")


def _model_sort_key(data_dir: str | None, row: dict[str, Any]) -> tuple[float, str]:
    name = str(row.get("model_name") or row.get("name") or "")
    return (-_model_effective_ts(data_dir, row), name.lower())


def sort_model_rows(
    rows: Iterable[dict[str, Any]], *, data_dir: str | None = None
) -> list[dict[str, Any]]:
    """Sort model rows per the global standard (newest trained/created first)."""
    return sorted((dict(r) for r in rows), key=lambda r: _model_sort_key(data_dir, r))


def get_sorted_models(
    data_dir: str | None = None,
    *,
    rows: list[dict[str, Any]] | None = None,
    lightweight: bool = False,
    include_experiments: bool = False,
) -> list[dict[str, Any]]:
    """Return trained-model rows sorted by the global standard.

    Pass ``rows`` to re-sort an already-fetched / filtered row list. Otherwise
    ``data_dir`` is used to fetch models via
    ``chain_replay_ml.training.registry.list_trained_models``.
    """
    if rows is None:
        if not data_dir:
            return []
        from chain_replay_ml.training.registry import list_trained_models

        rows = list_trained_models(
            data_dir, lightweight=lightweight, include_experiments=include_experiments
        )
    return sort_model_rows(rows, data_dir=data_dir)


def get_sorted_model_names(
    data_dir: str | None = None,
    *,
    rows: list[dict[str, Any]] | None = None,
    lightweight: bool = False,
    include_experiments: bool = False,
    name_key: str = "model_name",
) -> list[str]:
    """Convenience wrapper returning just the sorted, de-duplicated names."""
    sorted_rows = get_sorted_models(
        data_dir,
        rows=rows,
        lightweight=lightweight,
        include_experiments=include_experiments,
    )
    return _names_from_rows(sorted_rows, name_key)


def _names_from_rows(rows: list[dict[str, Any]], name_key: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for row in rows:
        name = str(row.get(name_key) or row.get("name") or "").strip()
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return names


def pick_preserved_or_default(items: Sequence[str], current: str | None) -> str:
    """Preserve ``current`` if still present in ``items``; else the newest item.

    ``items`` is assumed sorted newest-first (as returned by
    ``get_sorted_dataset_names`` / ``get_sorted_model_names``), so "newest" is
    simply ``items[0]``. Returns ``""`` when ``items`` is empty.
    """
    current = str(current or "").strip()
    if current and current in items:
        return current
    return items[0] if items else ""


def refresh_combobox(
    combo: "ttk.Combobox",
    items: Sequence[str],
    *,
    var: "tk.StringVar | None" = None,
    current: str | None = None,
) -> str:
    """Apply a freshly sorted item list to a combobox, preserving selection.

    Sets ``combo["values"]`` to ``items`` (expected newest-first), then keeps
    the current selection if it still exists in ``items``; otherwise selects
    the newest item (``items[0]``); otherwise clears the selection.

    ``current`` overrides the value read from ``var``/``combo`` when given
    (useful when the caller has a saved/preferred value to restore).

    Returns the value that ends up selected — callers do not need to read it
    back from ``var``/``combo`` immediately afterwards.
    """
    combo["values"] = list(items)
    if current is None:
        current = var.get() if var is not None else str(combo.get() or "")
    picked = pick_preserved_or_default(items, current)
    if var is not None:
        var.set(picked)
    else:
        combo.set(picked)
    return picked


__all__ = [
    "get_sorted_datasets",
    "get_sorted_dataset_names",
    "get_sorted_models",
    "get_sorted_model_names",
    "sort_dataset_rows",
    "sort_model_rows",
    "pick_preserved_or_default",
    "refresh_combobox",
]
