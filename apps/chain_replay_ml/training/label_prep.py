"""Adapt raw training targets to the selected prediction_type.

Triple Barrier ``label_id`` is multiclass {TP=0, SL=1, TIME=2}. Create Model
"Binary Classification (Hit)" means TP-hit vs not-TP — XGBoost ``binary:logistic``
requires labels in {0, 1}.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .target_kinds import is_ole_class_target

# Default OLE Triple Barrier encoding (strategy contract).
_DEFAULT_TB_ENCODING: dict[str, int] = {"TP": 0, "SL": 1, "TIME": 2}


def _int_classes(y: pd.Series) -> list[int]:
    vals = pd.to_numeric(y, errors="coerce").dropna()
    return sorted({int(round(float(v))) for v in vals.tolist()})


def _resolve_encoding(
    *,
    target: str | None,
    label_encoding: dict[str, Any] | None,
) -> dict[str, int]:
    raw = dict(label_encoding or {})
    out: dict[str, int] = {}
    for k, v in raw.items():
        try:
            out[str(k)] = int(v)
        except (TypeError, ValueError):
            continue
    if out:
        return out
    if is_ole_class_target(str(target or "")):
        return dict(_DEFAULT_TB_ENCODING)
    return {}


def adapt_target_for_prediction_type(
    y: pd.Series,
    *,
    prediction_type: str,
    target: str | None = None,
    label_encoding: dict[str, Any] | None = None,
) -> tuple[pd.Series, dict[str, Any]]:
    """Return (possibly remapped) ``y`` and a small audit dict.

    Binary + OLE ``label_id`` / encoding with TP → ``1`` if TP else ``0``.
    Already-{0,1} binary targets pass through (idempotent).
    Multiclass / classification with >2 classes remaps ids to ``0..K-1``.
    """
    pred = str(prediction_type or "regression").strip().lower()
    meta: dict[str, Any] = {
        "prediction_type": pred,
        "target": str(target or "").strip() or None,
        "mode": "passthrough",
    }
    if pred not in ("binary", "classification", "multiclass"):
        return y, meta

    y_num = pd.to_numeric(y, errors="coerce")
    classes = _int_classes(y_num)
    meta["raw_classes"] = list(classes)
    encoding = _resolve_encoding(target=target, label_encoding=label_encoding)
    if encoding:
        meta["label_encoding"] = dict(encoding)

    if pred == "binary" or (pred == "classification" and len(classes) <= 2):
        if set(classes).issubset({0, 1}):
            out = y_num.fillna(0).astype("float32")
            meta["mode"] = "binary_passthrough"
            meta["n_classes"] = 2
            return out, meta

        if "TP" in encoding:
            pos = int(encoding["TP"])
            out = (y_num == float(pos)).astype("float32")
            meta["mode"] = "binary_tp_hit"
            meta["positive_raw_class"] = pos
            meta["n_classes"] = 2
            meta["adapted_prediction_type"] = "binary"
            return out, meta

        if len(classes) == 2:
            lo, hi = classes[0], classes[1]
            out = (y_num == float(hi)).astype("float32")
            meta["mode"] = "binary_two_class"
            meta["positive_raw_class"] = hi
            meta["n_classes"] = 2
            return out, meta

        raise ValueError(
            f"Binary training requires labels in {{0, 1}}; got classes {classes}. "
            "For Triple Barrier label_id, keep prediction_type=binary (TP hit) "
            "or switch to Classification for TP/SL/TIME multiclass."
        )

    # Multiclass path
    class_map = {c: i for i, c in enumerate(classes)}
    out = y_num.map(
        lambda v: float(class_map.get(int(round(float(v))), 0)) if pd.notna(v) else float("nan")
    ).astype("float32")
    meta["mode"] = "multiclass_remap"
    meta["class_map"] = {str(k): v for k, v in class_map.items()}
    meta["n_classes"] = max(2, len(classes))
    meta["adapted_prediction_type"] = "multiclass"
    return out, meta
