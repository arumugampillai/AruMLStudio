"""Model Registry → XGBoost artifact resolution for Model Explorer."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# Strict XGBoost keys — do NOT use normalize_algorithm_id alone (it defaults unknowns to xgboost).
_XGB_KEYS = frozenset({"xgboost", "xgb"})
_NON_XGB_KEYS = frozenset({
    "lightgbm",
    "lgb",
    "light_gbm",
    "catboost",
    "cat",
    "random_forest",
    "rf",
    "randomforest",
    "extra_trees",
    "et",
    "extratrees",
    "extra_tree",
})
_XGB_ARTIFACT_EXTS = frozenset({".ubj", ".json", ".bst", ".model", ".pkl", ".pickle"})
_MISSING = "—"
_TOP_FEATURES_LIMIT = 20


def _algo_key(algorithm: str | None) -> str:
    return str(algorithm or "").strip().lower().replace(" ", "_").replace("-", "_")


def is_xgboost_algorithm(algorithm: str | None) -> bool:
    """True only when ``algorithm`` is explicitly XGBoost (id or display label)."""
    raw = str(algorithm or "").strip()
    if not raw or raw in {"—", "-", "n/a", "na", "none", "unknown", "?"}:
        return False
    return _algo_key(raw) in _XGB_KEYS


def is_non_xgboost_algorithm(algorithm: str | None) -> bool:
    """True when ``algorithm`` is a known non-XGBoost trainer."""
    key = _algo_key(algorithm)
    if not key or key in {"—", "-", "n/a", "na", "none", "unknown"}:
        return False
    if key in _XGB_KEYS:
        return False
    return key in _NON_XGB_KEYS or key in {
        "lightgbm",
        "catboost",
        "random_forest",
        "extra_trees",
    }


def filter_xgboost_registry_rows(
    rows: list[dict[str, Any]],
    *,
    data_dir: str | None = None,
    require_artifact: bool = True,
) -> list[dict[str, Any]]:
    """Keep registry rows that are XGBoost and (optionally) have a resolvable artifact.

    Each returned row is a shallow copy with ``artifact_path`` set when resolved.
    Primary display label remains ``model_name``.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        name = str(row.get("model_name") or row.get("name") or "").strip()
        if not name:
            continue

        algo = row.get("algorithm")
        artifact_path = ""
        resolved_algo = algo

        if require_artifact:
            if not data_dir:
                continue
            resolved = resolve_registry_xgboost_artifact(data_dir, name, row_algorithm=algo)
            if not resolved.get("ok"):
                continue
            artifact_path = str(resolved.get("artifact_path") or "")
            resolved_algo = resolved.get("algorithm") or algo
        elif not is_xgboost_algorithm(algo):
            continue

        entry = dict(row)
        entry["model_name"] = name
        entry["algorithm"] = resolved_algo or algo
        if artifact_path:
            entry["artifact_path"] = artifact_path
        out.append(entry)
    return out


def resolve_registry_xgboost_artifact(
    data_dir: str,
    model_name: str,
    *,
    row_algorithm: str | None = None,
) -> dict[str, Any]:
    """Resolve a registered model's production artifact path (XGBoost only).

    Returns ``{"ok": True, "artifact_path", "algorithm", "model_name"}`` or
    ``{"ok": False, "error": ...}``.
    """
    from chain_replay_ml.training.model_runtime import resolve_prediction_model_package

    name = str(model_name or "").strip()
    if not name:
        return {"ok": False, "error": "Model name is empty"}
    if not data_dir or not os.path.isdir(data_dir):
        return {"ok": False, "error": "Data directory not found"}

    pkg = resolve_prediction_model_package(data_dir, name)
    if not pkg.get("ok"):
        return {
            "ok": False,
            "error": str(pkg.get("error") or f"Model artifact not found for '{name}'"),
            "model_name": name,
        }

    config_algo = pkg.get("algorithm")
    path = str(pkg.get("model_path") or "")
    if not path or not os.path.isfile(path):
        return {
            "ok": False,
            "error": f"Model artifact not found for '{name}'",
            "model_name": name,
            "algorithm": config_algo,
        }

    # Prefer explicit XGBoost on config or registry row; reject known non-XGB.
    if is_non_xgboost_algorithm(config_algo) or is_non_xgboost_algorithm(row_algorithm):
        bad = config_algo if is_non_xgboost_algorithm(config_algo) else row_algorithm
        return {
            "ok": False,
            "error": f"Model '{name}' is {bad}, not XGBoost",
            "model_name": name,
            "algorithm": bad,
        }

    if not (
        is_xgboost_algorithm(config_algo)
        or is_xgboost_algorithm(row_algorithm)
    ):
        # Unknown / missing algorithm: allow only XGBoost-native artifact extensions.
        ext = os.path.splitext(path)[1].lower()
        if ext not in _XGB_ARTIFACT_EXTS:
            return {
                "ok": False,
                "error": (
                    f"Model '{name}' algorithm is unknown "
                    f"({config_algo or row_algorithm!r}) and artifact is not XGBoost format"
                ),
                "model_name": name,
                "algorithm": config_algo or row_algorithm,
            }

    return {
        "ok": True,
        "artifact_path": path,
        "model_path": path,
        "algorithm": config_algo or row_algorithm or "xgboost",
        "model_name": name,
        "features": list(pkg.get("features") or []),
    }


def list_xgboost_registry_models(
    data_dir: str,
    *,
    include_experiments: bool = False,
) -> list[dict[str, Any]]:
    """List Model Registry XGBoost packages that have a valid production artifact.

    Sorted newest-first via ``selection_lists.get_sorted_models``.
    Each item: ``model_name``, ``algorithm``, ``artifact_path``, plus registry fields.
    """
    if not data_dir or not os.path.isdir(data_dir):
        return []

    from chain_replay_ml.training.registry import list_trained_models

    from ..selection_lists import get_sorted_models

    rows = list_trained_models(
        data_dir, lightweight=False, include_experiments=include_experiments
    )
    sorted_rows = get_sorted_models(data_dir, rows=rows)
    return filter_xgboost_registry_rows(
        sorted_rows, data_dir=data_dir, require_artifact=True
    )


def registry_model_labels(models: list[dict[str, Any]]) -> list[str]:
    """Primary dropdown labels — model name only."""
    labels: list[str] = []
    seen: set[str] = set()
    for row in models:
        name = str(row.get("model_name") or "").strip()
        if name and name not in seen:
            labels.append(name)
            seen.add(name)
    return labels


# ------------------------------------------------------------------ Registry summary strip


@dataclass(frozen=True)
class RegistrySummaryView:
    """Read-only Model Registry fields for the Explorer summary strip."""

    model_name: str = ""
    rows: Any = None
    selected_features: Any = None
    target: str | None = None
    algorithm: str | None = None
    created: str | None = None
    holdout_accuracy_pct: float | None = None
    # (feature_name, importance_pct) — pct is None when registry has names only
    top_features: list[tuple[str, float | None]] = field(default_factory=list)


def _display_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text in {"—", "-", "n/a", "na", "none", "unknown", "?"}:
        return None
    return text


def _first_present(*vals: Any) -> Any:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and not str(v).strip():
            continue
        return v
    return None


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def format_registry_rows(value: Any) -> str:
    """Format training row count with thousands separators."""
    n = _as_int(value)
    return f"{n:,}" if n is not None else _MISSING


def format_registry_feature_count(value: Any) -> str:
    n = _as_int(value)
    return f"{n:,}" if n is not None else _MISSING


def format_registry_created(value: Any) -> str:
    """Prefer ``YYYY-MM-DD`` from ``trained_at`` / created timestamps."""
    text = _display_or_none(value)
    if not text:
        return _MISSING
    # ISO-ish: 2026-07-29T12:34:56 → 2026-07-29
    if len(text) >= 10 and text[4] == "-" and text[7] == "-":
        return text[:10]
    return text


def format_registry_holdout_accuracy(value: Any) -> str:
    pct = _as_float(value)
    if pct is None:
        return _MISSING
    return f"{pct:.1f}%"


def extract_holdout_accuracy_pct(row: dict[str, Any] | None) -> float | None:
    """Holdout / production accuracy from a registry row or ``get_model_summary`` blob.

    Preference: classification ``accuracy_pct``, then ``directional_accuracy_pct``.
    Looks in ``production_metrics``, then top-level / ``metrics``.
    """
    if not isinstance(row, dict):
        return None
    prod = row.get("production_metrics") if isinstance(row.get("production_metrics"), dict) else {}
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    return _as_float(
        _first_present(
            prod.get("accuracy_pct"),
            row.get("accuracy_pct"),
            metrics.get("accuracy_pct"),
            prod.get("directional_accuracy_pct"),
            row.get("directional_accuracy_pct"),
            metrics.get("directional_accuracy_pct"),
        )
    )


_PCT_FIELD_KEYS = (
    "importance_pct",
    "Importance",  # feature_importance.csv — already gain share %
    "pct",
    "percent",
    "percentage",
    "gain_share",
    "gain_importance_pct",
)
_RAW_SCORE_KEYS = ("importance", "gain", "score", "weight", "value")


def _feature_name_from_item(item: Any) -> str:
    if isinstance(item, dict):
        return str(
            item.get("feature") or item.get("Feature") or item.get("name") or ""
        ).strip()
    return str(item or "").strip()


def _score_from_feature_item(item: dict[str, Any]) -> tuple[float | None, bool]:
    """Return ``(value, is_percentage)`` from a registry / importance row.

    Prefers explicit percentage fields (``importance_pct``, CSV ``Importance``).
    Falls back to raw gain/importance scores for later normalization.
    """
    for key in _PCT_FIELD_KEYS:
        if key not in item:
            continue
        raw = item.get(key)
        if raw in (None, ""):
            continue
        val = _as_float(raw)
        if val is not None:
            return val, True
    for key in _RAW_SCORE_KEYS:
        if key not in item:
            continue
        raw = item.get(key)
        if raw in (None, ""):
            continue
        val = _as_float(raw)
        if val is not None:
            return val, False
    return None, False


def _normalize_raw_scores(
    entries: list[tuple[str, float | None, bool]],
) -> list[tuple[str, float | None]]:
    """Convert raw scores to % of total when no percentage fields were present."""
    if not entries:
        return []
    any_pct = any(is_pct and score is not None for _, score, is_pct in entries)
    if any_pct:
        return [(name, score if is_pct else None) for name, score, is_pct in entries]

    total = sum(float(score) for _, score, is_pct in entries if score is not None and score > 0)
    if total <= 0:
        return [(name, None) for name, _, _ in entries]
    out: list[tuple[str, float | None]] = []
    for name, score, _ in entries:
        if score is None:
            out.append((name, None))
        else:
            out.append((name, (float(score) / total) * 100.0))
    return out


def extract_top_features(
    source: dict[str, Any] | list[Any] | None,
    *,
    limit: int = _TOP_FEATURES_LIMIT,
) -> list[tuple[str, float | None]]:
    """Top features with optional importance % from registry summary / importance lists.

    When Model Registry provides ``importance_pct`` (or CSV ``Importance``), that
    share is kept as-is. When only raw scores exist, values are normalized to % of
    total (sum ≈ 100) across the source list before applying ``limit``.
    """
    if limit <= 0:
        return []

    def _from_list(blob: list[Any]) -> list[tuple[str, float | None]]:
        staged: list[tuple[str, float | None, bool]] = []
        seen: set[str] = set()
        for item in blob:
            if isinstance(item, dict):
                name = _feature_name_from_item(item)
                if not name or name in seen:
                    continue
                score, is_pct = _score_from_feature_item(item)
                staged.append((name, score, is_pct))
                seen.add(name)
            else:
                name = _feature_name_from_item(item)
                if not name or name in seen:
                    continue
                staged.append((name, None, False))
                seen.add(name)
        ranked = _normalize_raw_scores(staged)
        if any(pct is not None for _, pct in ranked):
            ranked = sorted(
                ranked,
                key=lambda t: (t[1] is not None, float(t[1] or 0.0)),
                reverse=True,
            )
        return ranked[:limit]

    if isinstance(source, list):
        return _from_list(source)

    if not isinstance(source, dict):
        return []

    for key in ("top_features", "feature_importance"):
        blob = source.get(key)
        if isinstance(blob, list) and blob:
            return _from_list(blob)

    for key in ("selected_features", "features"):
        blob = source.get(key)
        if isinstance(blob, list) and blob:
            return _from_list(blob)

    return []


def extract_top_feature_names(
    source: dict[str, Any] | list[Any] | None,
    *,
    limit: int = _TOP_FEATURES_LIMIT,
) -> list[str]:
    """Top feature names from registry summary / importance / selected lists."""
    return [name for name, _pct in extract_top_features(source, limit=limit)]


def format_top_feature_line(index: int, name: str, importance_pct: float | None) -> str:
    """Format one Top Features strip line: ``1. spot_ema200  12.4%``."""
    label = str(name or "").strip() or _MISSING
    if importance_pct is None:
        return f"{index}. {label}"
    return f"{index}. {label}  {float(importance_pct):.1f}%"


def extract_registry_summary(
    row: dict[str, Any] | None,
    *,
    enrich: dict[str, Any] | None = None,
    top_limit: int = _TOP_FEATURES_LIMIT,
) -> RegistrySummaryView:
    """Build a summary view from a registry list row and optional ``get_model_summary``.

    Missing fields stay ``None`` / empty so the UI can show ``—`` gracefully.
    Does not read booster dumps — registry metadata only.
    """
    base = dict(row or {})
    extra = dict(enrich or {}) if isinstance(enrich, dict) else {}

    model_name = str(
        _first_present(base.get("model_name"), extra.get("model_name"), base.get("name")) or ""
    ).strip()

    rows = _first_present(extra.get("rows"), base.get("rows"))
    selected = _first_present(
        extra.get("feature_count"),
        base.get("feature_count"),
        _as_int(len(extra.get("selected_features")))
        if isinstance(extra.get("selected_features"), list)
        else None,
        _as_int(len(base.get("selected_features")))
        if isinstance(base.get("selected_features"), list)
        else None,
    )
    target = _display_or_none(_first_present(extra.get("target"), base.get("target")))
    algorithm = _display_or_none(_first_present(extra.get("algorithm"), base.get("algorithm")))
    created_raw = _first_present(
        extra.get("trained_at"),
        base.get("trained_at"),
        extra.get("created_at"),
        base.get("created_at"),
        extra.get("created"),
        base.get("created"),
    )
    created = format_registry_created(created_raw)
    if created == _MISSING:
        created = None
    else:
        # Keep date-only string already formatted.
        created = created

    holdout = extract_holdout_accuracy_pct(extra) or extract_holdout_accuracy_pct(base)
    top = extract_top_features(extra, limit=top_limit)
    if not top:
        top = extract_top_features(base, limit=top_limit)

    return RegistrySummaryView(
        model_name=model_name,
        rows=rows,
        selected_features=selected,
        target=target,
        algorithm=algorithm,
        created=created,
        holdout_accuracy_pct=holdout,
        top_features=top,
    )


def format_registry_summary_text(view: RegistrySummaryView | None) -> str:
    """Render the read-only Model Registry summary strip (mockup layout)."""
    if view is None:
        return ""
    has_any = bool(
        view.model_name
        or view.target
        or view.algorithm
        or view.created
        or view.rows is not None
        or view.selected_features is not None
        or view.holdout_accuracy_pct is not None
        or view.top_features
    )
    if not has_any:
        return ""

    lines = [
        f"Rows              : {format_registry_rows(view.rows)}",
        f"Selected Features : {format_registry_feature_count(view.selected_features)}",
        f"Target            : {view.target or _MISSING}",
        f"Algorithm         : {view.algorithm or _MISSING}",
        f"Created           : {view.created or _MISSING}",
        f"Holdout Accuracy  : {format_registry_holdout_accuracy(view.holdout_accuracy_pct)}",
        "",
        "Top Features (from Model Registry)",
        "",
    ]
    if view.top_features:
        for i, entry in enumerate(view.top_features[:_TOP_FEATURES_LIMIT], start=1):
            if isinstance(entry, tuple) and len(entry) >= 2:
                name, pct = entry[0], entry[1]
            else:
                name, pct = str(entry), None
            lines.append(format_top_feature_line(i, str(name), pct))
    else:
        lines.append("(none available)")
    return "\n".join(lines)


def load_registry_summary_for_model(
    data_dir: str,
    model_name: str,
    *,
    cached_row: dict[str, Any] | None = None,
) -> RegistrySummaryView:
    """Resolve registry summary for ``model_name``, enriching with ``get_model_summary`` when possible."""
    name = str(model_name or "").strip()
    row = dict(cached_row or {})
    if name and not row.get("model_name"):
        row["model_name"] = name

    enrich: dict[str, Any] | None = None
    if data_dir and name and os.path.isdir(data_dir):
        try:
            from chain_replay_ml.training.registry import get_model_summary

            enrich = get_model_summary(data_dir, name)
        except Exception:
            enrich = None

    return extract_registry_summary(row, enrich=enrich)
