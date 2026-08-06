"""Recognized training target column kinds (Master + ORMP research labels)."""

from __future__ import annotations

from typing import Any


def is_ormp_return_target(name: str) -> bool:
    """Continuous ORMP label: ormp_return_{N}m_points|percent."""
    n = str(name or "").strip()
    if not n.startswith("ormp_return_"):
        return False
    return n.endswith("_points") or n.endswith("_percent")


def is_ormp_direction_target(name: str) -> bool:
    """Directional ORMP label: ormp_direction_{N}m (−1/0/+1)."""
    return str(name or "").strip().startswith("ormp_direction_")


def is_label_up_target(name: str) -> bool:
    """Binary Master export label: label_up_{N}pct_5m / label_up_gt6pct_5m."""
    return str(name or "").strip().startswith("label_up_")


def is_ole_class_target(name: str) -> bool:
    """OLE categorical training column (strategy primary_target, e.g. label_id).

    Column-name based — not ``if strategy == ...``. Strategies declare this via
    ``get_target_definitions().primary_target``.
    """
    return str(name or "").strip() in {"label_id"}


def is_regression_target(name: str) -> bool:
    n = str(name or "").strip()
    return n.startswith("future_ltp") or is_ormp_return_target(n)


def is_binary_hit_target(name: str) -> bool:
    return str(name or "").strip() in ("target_reached", "hit") or is_label_up_target(name)


def is_classification_target(name: str) -> bool:
    return (
        is_binary_hit_target(name)
        or is_ormp_direction_target(name)
        or is_ole_class_target(name)
    )


def prediction_type_for_target(name: str) -> str | None:
    if is_regression_target(name):
        return "regression"
    if is_binary_hit_target(name):
        return "binary"
    if is_ormp_direction_target(name) or is_ole_class_target(name):
        return "classification"
    return None

def target_listed_in_meta(name: str, meta: dict[str, Any] | None) -> bool:
    meta = meta or {}
    t = str(name or "").strip()
    if not t:
        return False
    for key in ("prediction_target_columns", "target_columns", "targets", "prediction_targets"):
        raw = meta.get(key) or []
        if isinstance(raw, list) and t in {str(x) for x in raw}:
            return True
    return False


def is_allowed_training_target(
    name: str,
    *,
    registry_targets: set[str] | None = None,
    meta: dict[str, Any] | None = None,
) -> bool:
    """True if the name is a known Master/ORMP target or declared in dataset metadata."""
    t = str(name or "").strip()
    if not t:
        return False
    if registry_targets and t in registry_targets:
        return True
    if is_regression_target(t) or is_classification_target(t):
        return True
    if target_listed_in_meta(t, meta):
        return True
    return False


def target_prediction_type_compatible(prediction_type: str, target: str) -> bool:
    pred = str(prediction_type or "").strip().lower()
    t = str(target or "").strip()
    if pred == "regression" and is_regression_target(t):
        return True
    if pred in ("binary", "classification") and is_classification_target(t):
        return True
    return False
