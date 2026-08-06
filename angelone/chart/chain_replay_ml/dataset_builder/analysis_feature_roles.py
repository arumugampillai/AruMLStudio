"""Feature role classification for Research Lab scoring.

Roles
-----
predictor — included in Correlation, MI, SHAP, VIF, Permutation, Scorecard
target    — prediction targets only (MI/SHAP target pickers); excluded from scoring
label     — supervised labels only; excluded from scoring
metadata  — identity / join columns; excluded from scoring and ranking
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

ROLE_PREDICTOR = "predictor"
ROLE_TARGET = "target"
ROLE_LABEL = "label"
ROLE_METADATA = "metadata"

FEATURE_ROLES = (
    ROLE_PREDICTOR,
    ROLE_TARGET,
    ROLE_LABEL,
    ROLE_METADATA,
)

META_COLS = frozenset(
    {
        "trading_day",
        "timestamp",
        "ts",
        "token",
        "symbol",
        "expiry",
        "strike",
        "option_type",
        "right",
        "side",
        "row_id",
        "sample_id",
        "index",
        "datetime",
        "date",
        "time",
        # Master / analysis identity (never predictors)
        "master_row_id",
        "market",
        "underlying",
        "instrument_type",
        "exchange",
        "segment",
        "interval_sec",
        "sample_ts",
        "row_hash",
    }
)

# Short horizon tokens that appear in sidecars but are not column names.
_HORIZON_TOKENS = frozenset({"10s", "30s", "1m", "3m", "5m", "10m", "15m", "30m"})

ROLE_BANNERS: dict[str, tuple[str, str]] = {
    ROLE_TARGET: (
        "Role: Target",
        "This column is a prediction target and is not included in feature evaluation.",
    ),
    ROLE_LABEL: (
        "Role: Label",
        "This column is a supervised learning label and is excluded from feature scoring.",
    ),
    ROLE_METADATA: (
        "Role: Metadata",
        "This column is dataset metadata and is excluded from feature scoring.",
    ),
}


def _sidecar_named_columns(sidecar: Mapping[str, Any] | None) -> tuple[set[str], set[str]]:
    """Return (target_names, label_names) declared in a dataset sidecar."""
    targets: set[str] = set()
    labels: set[str] = set()
    if not isinstance(sidecar, Mapping):
        return targets, labels

    for key in ("prediction_target_columns", "prediction_targets"):
        raw = sidecar.get(key) or []
        if isinstance(raw, list):
            for x in raw:
                s = str(x).strip()
                if not s or s in _HORIZON_TOKENS:
                    continue
                low = s.lower()
                if low.startswith("label_"):
                    labels.add(s)
                else:
                    targets.add(s)

    clf = sidecar.get("classification_labels_5m") or {}
    if isinstance(clf, Mapping):
        for x in clf.get("columns") or []:
            s = str(x).strip()
            if s:
                labels.add(s)
    return targets, labels


def classify_feature_role(
    name: str,
    *,
    sidecar: Mapping[str, Any] | None = None,
    known_targets: Iterable[str] | None = None,
    known_labels: Iterable[str] | None = None,
) -> str:
    """Classify a column into predictor | target | label | metadata."""
    n = str(name or "").strip()
    if not n or n.startswith("_"):
        return ROLE_METADATA

    low = n.lower()
    if n in META_COLS or low in META_COLS or low.endswith("_row_id"):
        return ROLE_METADATA

    side_targets, side_labels = _sidecar_named_columns(sidecar)
    if known_targets:
        side_targets |= {str(x).strip() for x in known_targets if str(x).strip()}
    if known_labels:
        side_labels |= {str(x).strip() for x in known_labels if str(x).strip()}

    if n in side_labels or low.startswith("label_"):
        return ROLE_LABEL
    if n in side_targets or low.startswith("future_ltp_"):
        return ROLE_TARGET
    return ROLE_PREDICTOR


def classify_columns(
    columns: Sequence[str],
    *,
    sidecar: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    return {
        str(c): classify_feature_role(str(c), sidecar=sidecar) for c in columns
    }


def is_predictor(name: str, *, sidecar: Mapping[str, Any] | None = None) -> bool:
    return classify_feature_role(name, sidecar=sidecar) == ROLE_PREDICTOR


def is_scorable(name: str, *, sidecar: Mapping[str, Any] | None = None) -> bool:
    """True when the column participates in Research Lab scoring modules."""
    return is_predictor(name, sidecar=sidecar)


def is_explorer_column(name: str, *, sidecar: Mapping[str, Any] | None = None) -> bool:
    """Explorer lists predictors, targets, and labels — not metadata."""
    return classify_feature_role(name, sidecar=sidecar) != ROLE_METADATA


def predictor_columns(
    columns: Sequence[str],
    *,
    sidecar: Mapping[str, Any] | None = None,
) -> list[str]:
    return [
        str(c)
        for c in columns
        if classify_feature_role(str(c), sidecar=sidecar) == ROLE_PREDICTOR
    ]


def explorer_columns(
    columns: Sequence[str],
    *,
    sidecar: Mapping[str, Any] | None = None,
) -> list[str]:
    return [
        str(c)
        for c in columns
        if is_explorer_column(str(c), sidecar=sidecar)
    ]


def role_banner(role: str) -> tuple[str, str] | None:
    return ROLE_BANNERS.get(str(role or ""))


__all__ = [
    "FEATURE_ROLES",
    "META_COLS",
    "ROLE_BANNERS",
    "ROLE_LABEL",
    "ROLE_METADATA",
    "ROLE_PREDICTOR",
    "ROLE_TARGET",
    "classify_columns",
    "classify_feature_role",
    "explorer_columns",
    "is_explorer_column",
    "is_predictor",
    "is_scorable",
    "predictor_columns",
    "role_banner",
]
