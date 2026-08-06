"""Prediction-package discovery for regression anchors and classifier ladders.

The physical model artifacts stay independent.  This module builds a logical
package view so one future-LTP regression model can expose any available
``label_up_*`` classifiers trained from the same dataset and horizon.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any


# Formal package-member contract fields (generic executor input):
#   role             — unique member role inside a package (e.g. probability_up_2pct)
#   prediction_type  — how the wrapper output is interpreted (regression | probability)
#   output_column    — prediction-dataset column the member writes
#   target           — training target that identifies the member slot
# The executor never special-cases roles: it slices features, predicts, and
# writes ``output_column`` for every member according to ``prediction_type``.
PREDICTION_TYPE_REGRESSION = "regression"
PREDICTION_TYPE_PROBABILITY = "probability"

PROBABILITY_LADDER: tuple[dict[str, Any], ...] = (
    {
        "key": "up_2pct", "label": "+2%", "target": "label_up_2pct_5m", "order": 2,
        "role": "probability_up_2pct", "output_column": "pred_prob_up_2pct_5m",
        "prediction_type": PREDICTION_TYPE_PROBABILITY,
    },
    {
        "key": "up_3pct", "label": "+3%", "target": "label_up_3pct_5m", "order": 3,
        "role": "probability_up_3pct", "output_column": "pred_prob_up_3pct_5m",
        "prediction_type": PREDICTION_TYPE_PROBABILITY,
    },
    {
        "key": "up_4pct", "label": "+4%", "target": "label_up_4pct_5m", "order": 4,
        "role": "probability_up_4pct", "output_column": "pred_prob_up_4pct_5m",
        "prediction_type": PREDICTION_TYPE_PROBABILITY,
    },
    {
        "key": "up_5pct", "label": "+5%", "target": "label_up_5pct_5m", "order": 5,
        "role": "probability_up_5pct", "output_column": "pred_prob_up_5pct_5m",
        "prediction_type": PREDICTION_TYPE_PROBABILITY,
    },
    {
        "key": "up_6pct", "label": "+6%", "target": "label_up_6pct_5m", "order": 6,
        "role": "probability_up_6pct", "output_column": "pred_prob_up_6pct_5m",
        "prediction_type": PREDICTION_TYPE_PROBABILITY,
    },
    {
        "key": "up_gt6pct", "label": ">6%", "target": "label_up_gt6pct_5m", "order": 7,
        "role": "probability_up_gt6pct", "output_column": "pred_prob_up_gt6pct_5m",
        "prediction_type": PREDICTION_TYPE_PROBABILITY,
    },
)

# Stable prediction-dataset schema: these columns always exist; missing
# members write NULL.
PROBABILITY_OUTPUT_COLUMNS: tuple[str, ...] = tuple(
    str(item["output_column"]) for item in PROBABILITY_LADDER
)

_LADDER_BY_TARGET = {str(item["target"]): item for item in PROBABILITY_LADDER}
_REGRESSION_HORIZON_RE = re.compile(r"^future_ltp_(\d+[ms])$", re.IGNORECASE)
_CLASSIFICATION_HORIZON_RE = re.compile(
    r"^label_up_(?:[2-6]|gt6)pct_(\d+[ms])$",
    re.IGNORECASE,
)


def target_horizon(target: Any) -> str | None:
    """Return a normalized target horizon (for example ``5m``)."""
    text = str(target or "").strip()
    for pattern in (_REGRESSION_HORIZON_RE, _CLASSIFICATION_HORIZON_RE):
        match = pattern.match(text)
        if match:
            return match.group(1).lower()
    return None


def probability_ladder_slot(target: Any) -> dict[str, Any] | None:
    """Return the canonical ladder slot for a supported classification target."""
    slot = _LADDER_BY_TARGET.get(str(target or "").strip().lower())
    return dict(slot) if slot else None


def is_package_anchor_target(target: Any) -> bool:
    """True when the target alone qualifies as a package anchor (5m regression)."""
    text = str(target or "").strip()
    return bool(_REGRESSION_HORIZON_RE.match(text)) and target_horizon(text) == "5m"


def is_prediction_package_anchor(row: dict[str, Any]) -> bool:
    prediction_type = str(row.get("prediction_type") or "").strip().lower()
    return is_package_anchor_target(row.get("target")) and prediction_type in (
        "",
        "regression",
    )


def _trained_sort_value(row: dict[str, Any]) -> tuple[float, str]:
    return _trained_stamp(row), str(row.get("model_name") or "")


def _trained_stamp(row: dict[str, Any]) -> float:
    raw = str(row.get("trained_at") or "").strip()
    if not raw:
        return 0.0
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return 0.0


def _package_identity(row: dict[str, Any]) -> tuple[str, str] | None:
    dataset = str(row.get("dataset") or "").strip()
    horizon = target_horizon(row.get("target"))
    if not dataset or dataset == "—" or not horizon:
        return None
    return dataset, horizon


def _explicit_package_anchor(row: dict[str, Any]) -> str:
    """Classifier → regression package link stamped at train time."""
    for key in ("package_anchor", "prediction_package_id"):
        text = str(row.get(key) or "").strip()
        if text:
            return text
    return ""


def resolve_classifier_package_anchor(
    classifier: dict[str, Any],
    anchors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the regression package that owns this classifier, or ``None``.

    Membership rules (in order):
    1. Explicit ``package_anchor`` / ``prediction_package_id`` matching an
       anchor model name on the same dataset + horizon.
    2. Legacy fallback: newest same-(dataset, horizon) regression whose
       ``trained_at`` is at or before the classifier's. This keeps older
       classifiers under the package that existed when they were trained,
       instead of letting a later regression on the same dataset steal them.
    """
    identity = _package_identity(classifier)
    if identity is None:
        return None

    same = [row for row in anchors if _package_identity(row) == identity]
    if not same:
        return None

    by_name = {
        str(row.get("model_name") or "").strip(): row
        for row in same
        if str(row.get("model_name") or "").strip()
    }
    explicit = _explicit_package_anchor(classifier)
    if explicit:
        return by_name.get(explicit)

    cls_stamp = _trained_stamp(classifier)
    eligible = [row for row in same if _trained_stamp(row) <= cls_stamp]
    if not eligible:
        return None
    return max(eligible, key=_trained_sort_value)


def build_prediction_package(
    anchor: dict[str, Any],
    classification_rows: list[dict[str, Any]],
    *,
    anchors: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build one package manifest. Missing ladder members are explicit slots.

    ``classification_rows`` may be the full registry classifier list; ownership
    is resolved against ``anchors`` (defaults to ``[anchor]`` only).
    """
    anchor_pool = list(anchors) if anchors is not None else [anchor]
    anchor_name = str(anchor.get("model_name") or "").strip()
    candidates: dict[str, list[dict[str, Any]]] = {}
    for row in classification_rows:
        owner = resolve_classifier_package_anchor(row, anchor_pool)
        if owner is None or str(owner.get("model_name") or "").strip() != anchor_name:
            continue
        slot = probability_ladder_slot(row.get("target"))
        if slot is None:
            continue
        candidates.setdefault(str(slot["key"]), []).append(row)

    members: list[dict[str, Any]] = []
    for canonical in PROBABILITY_LADDER:
        matches = candidates.get(str(canonical["key"])) or []
        chosen = max(matches, key=_trained_sort_value) if matches else None
        members.append(
            {
                **canonical,
                "available": chosen is not None,
                "model_name": (
                    str(chosen.get("model_name") or "") if chosen is not None else None
                ),
                "trained_at": chosen.get("trained_at") if chosen is not None else None,
                "status": chosen.get("status") if chosen is not None else "missing",
                "package_anchor": (
                    _explicit_package_anchor(chosen) or anchor_name
                    if chosen is not None
                    else None
                ),
            }
        )

    available = sum(1 for member in members if member["available"])
    return {
        "package_id": anchor_name,
        "anchor_model_name": anchor_name,
        "dataset": anchor.get("dataset"),
        "horizon": target_horizon(anchor.get("target")),
        "regression": {
            "available": True,
            "model_name": anchor_name,
            "target": anchor.get("target"),
        },
        "classification": {
            "available": available,
            "total": len(PROBABILITY_LADDER),
            "complete": available == len(PROBABILITY_LADDER),
            "members": members,
        },
    }


def package_registry_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach a package manifest to every regression anchor.

    Classifiers owned by an anchor are collapsed under that package. Orphans
    (no matching ownership) remain standalone rows. Partial packages are always
    retained — including empty ones so the Classification tab can offer Build.
    """
    anchors = [row for row in rows if is_prediction_package_anchor(row)]
    classifiers = [
        row for row in rows if probability_ladder_slot(row.get("target")) is not None
    ]

    owned_classifier_names: set[str] = set()
    for classifier in classifiers:
        owner = resolve_classifier_package_anchor(classifier, anchors)
        if owner is None:
            continue
        name = str(classifier.get("model_name") or "").strip()
        if name:
            owned_classifier_names.add(name)

    result: list[dict[str, Any]] = []
    for row in rows:
        if probability_ladder_slot(row.get("target")) is not None:
            name = str(row.get("model_name") or "").strip()
            if name in owned_classifier_names:
                continue

        enriched = dict(row)
        if is_prediction_package_anchor(row) and _package_identity(row) is not None:
            manifest = build_prediction_package(row, classifiers, anchors=anchors)
            enriched["prediction_package"] = manifest
            available = int(manifest["classification"]["available"])
            total = int(manifest["classification"]["total"])
            enriched["package_badge"] = f"Reg + {available}/{total} cls"
        result.append(enriched)

    return result


def _load_json_doc(path: str) -> dict[str, Any]:
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else {}
    except Exception:
        return {}


def _missing_member(canonical: dict[str, Any]) -> dict[str, Any]:
    return {
        **canonical,
        "available": False,
        "model_name": None,
        "model_path": None,
        "algorithm": None,
        "features": [],
        "trained_at": None,
    }


def discover_prediction_package_members(
    data_dir: str,
    *,
    dataset: str,
    anchor_target: str,
    anchor_model_name: str | None = None,
    anchor_trained_at: str | None = None,
) -> list[dict[str, Any]]:
    """Build the probability-ladder member contracts for one package anchor.

    Scans on-disk model packages and assigns classifiers to this anchor using
    the same ownership rules as the Model Registry (explicit package_anchor,
    else temporal legacy). Returns one JSON-safe contract per canonical slot
    (missing slots stay explicit with ``available=False``).
    """
    members = [_missing_member(dict(item)) for item in PROBABILITY_LADDER]
    if not is_package_anchor_target(anchor_target):
        return members

    from .model_runtime import resolve_production_model_path
    from .paths import model_artifact_paths, models_dir

    dataset_name = str(dataset or "").strip()
    horizon = target_horizon(anchor_target)
    anchor_name = str(anchor_model_name or "").strip()
    if not dataset_name or not horizon:
        return members

    base = models_dir(data_dir)
    try:
        entries = sorted(
            e for e in os.listdir(base)
            if not e.startswith(".") and os.path.isdir(os.path.join(base, e))
        )
    except OSError:
        entries = []

    anchors: list[dict[str, Any]] = []
    classifier_candidates: list[dict[str, Any]] = []
    for entry in entries:
        paths = model_artifact_paths(data_dir, entry)
        config = _load_json_doc(paths["config_json"])
        if not config:
            continue
        if str(config.get("dataset") or "").strip() != dataset_name:
            continue
        if target_horizon(config.get("target")) != horizon:
            continue

        registry_doc = _load_json_doc(paths["registry_json"])
        trained_at = registry_doc.get("trained_at") or config.get("trained_at")
        prediction_type = str(config.get("prediction_type") or "regression").strip().lower()
        row = {
            "model_name": entry,
            "target": config.get("target"),
            "dataset": dataset_name,
            "trained_at": trained_at,
            "prediction_type": prediction_type,
            "package_anchor": config.get("package_anchor")
            or config.get("prediction_package_id"),
            "algorithm": config.get("algorithm") or registry_doc.get("algorithm"),
            "features": list(
                config.get("features") or config.get("selected_features") or []
            ),
        }

        if is_prediction_package_anchor(row):
            anchors.append(row)
            continue

        slot = probability_ladder_slot(config.get("target"))
        if slot is None or not row["features"]:
            continue
        production_name = str(
            _load_json_doc(paths["training_metadata_json"]).get("production_model") or ""
        ).strip()
        model_path = resolve_production_model_path(
            paths["package_dir"],
            algorithm=config.get("algorithm"),
            production_name=production_name or None,
        )
        if not model_path:
            continue
        row["model_path"] = model_path
        row["_slot_key"] = str(slot["key"])
        classifier_candidates.append(row)

    # Ensure the requesting anchor is in the pool even if its package dir name
    # differs slightly from on-disk scan edge cases.
    if anchor_name and not any(
        str(a.get("model_name") or "") == anchor_name for a in anchors
    ):
        anchors.append(
            {
                "model_name": anchor_name,
                "target": anchor_target,
                "dataset": dataset_name,
                "trained_at": anchor_trained_at,
                "prediction_type": "regression",
            }
        )

    by_slot: dict[str, dict[str, Any]] = {}
    for candidate in classifier_candidates:
        owner = resolve_classifier_package_anchor(candidate, anchors)
        if owner is None:
            continue
        if anchor_name and str(owner.get("model_name") or "") != anchor_name:
            continue
        key = str(candidate.get("_slot_key") or "")
        previous = by_slot.get(key)
        if previous is None or _trained_sort_value(candidate) > _trained_sort_value(previous):
            by_slot[key] = candidate

    for member in members:
        chosen = by_slot.get(str(member["key"]))
        if chosen is None:
            continue
        member.update(
            {
                "available": True,
                "model_name": chosen["model_name"],
                "model_path": chosen["model_path"],
                "algorithm": chosen["algorithm"],
                "features": list(chosen["features"]),
                "trained_at": chosen["trained_at"],
            }
        )
    return members


def package_members_summary(members: list[dict[str, Any]]) -> str:
    """Short human-readable ladder status, e.g. ``3/6 classifiers (+2%, +3%, +5%)``."""
    available = [m for m in members if m.get("available")]
    total = len(members)
    if not available:
        return f"0/{total} classifiers"
    labels = ", ".join(str(m.get("label") or m.get("key")) for m in available)
    return f"{len(available)}/{total} classifiers ({labels})"


def attach_prediction_package(
    detail: dict[str, Any],
    registry_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Attach the selected anchor's already-discovered manifest to detail data."""
    name = str(detail.get("model_name") or "")
    row = next(
        (item for item in registry_rows if str(item.get("model_name") or "") == name),
        None,
    )
    if row and isinstance(row.get("prediction_package"), dict):
        detail["prediction_package"] = dict(row["prediction_package"])
    return detail

