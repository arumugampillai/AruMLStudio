"""Aggregate feature rankings from all ready registry models (historical selection + gain)."""

from __future__ import annotations

import csv
import json
import os
import re
from typing import Any

from chain_replay_ml.dataset_builder.schema_registry import load_schema_registry
from chain_replay_ml.training.paths import model_artifact_paths, model_package_dir
from chain_replay_ml.training.registry import list_trained_models

FREQ_WEIGHT = 0.50
GAIN_WEIGHT = 0.30
RANK_WEIGHT = 0.20

TOP_CHOICES = (25, 50, 75, 100, 125)

# Models below this fraction of the cohort max quality weight are excluded.
_MIN_QUALITY_WEIGHT_RATIO = 0.15

_UNDERLYING_ALIASES = {
    "NIFTY": "NIFTY",
    "NIFTY50": "NIFTY",
    "NIFTY 50": "NIFTY",
    "SENSEX": "SENSEX",
    "BANKNIFTY": "BANKNIFTY",
    "BANK_NIFTY": "BANKNIFTY",
    "BANK NIFTY": "BANKNIFTY",
    "NIFTYBANK": "BANKNIFTY",
    "BNF": "BANKNIFTY",
}


def normalize_underlying(raw: Any) -> str | None:
    key = str(raw or "").strip().upper()
    if not key or key in ("BOTH", "MIXED", "ALL", "ANY", "—", "-"):
        return None
    if key in _UNDERLYING_ALIASES:
        return _UNDERLYING_ALIASES[key]
    if "SENSEX" in key:
        return "SENSEX"
    if "BANKNIFTY" in key or ("BANK" in key and "NIFTY" in key):
        return "BANKNIFTY"
    if "NIFTY" in key:
        return "NIFTY"
    return key


def infer_underlying_from_text(text: str) -> str | None:
    low = str(text or "").lower()
    if "sensex" in low:
        return "SENSEX"
    if "banknifty" in low or "bank_nifty" in low or "bank-nifty" in low:
        return "BANKNIFTY"
    if re.search(r"\bbnf\b", low):
        return "BANKNIFTY"
    if "nifty" in low:
        return "NIFTY"
    return None


def resolve_model_underlying(data_dir: str, model: dict[str, Any]) -> str | None:
    dataset = str(model.get("dataset") or "").strip()
    if dataset and dataset != "—":
        try:
            from chain_replay_ml.dataset_builder.append_ops import load_dataset_metadata

            meta, _ = load_dataset_metadata(data_dir, dataset)
            market = normalize_underlying(meta.get("market"))
            if market:
                return market
            days = meta.get("days") or []
            markets = {
                normalize_underlying(d.get("market"))
                for d in days
                if isinstance(d, dict)
            }
            markets.discard(None)
            if len(markets) == 1:
                return next(iter(markets))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            pass
        inferred = infer_underlying_from_text(dataset)
        if inferred:
            return inferred
    return infer_underlying_from_text(str(model.get("model_name") or ""))


def _model_matches_underlying(
    data_dir: str,
    model: dict[str, Any],
    underlying_filter: str | None,
) -> bool:
    if not underlying_filter:
        return True
    model_ul = resolve_model_underlying(data_dir, model)
    if not model_ul:
        return False
    return model_ul == underlying_filter


def _float_val(raw: Any, default: float = 0.0) -> float:
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _int_val(raw: Any, default: int = 9999) -> int:
    if raw in (None, ""):
        return default
    try:
        return int(float(raw))
    except (TypeError, ValueError):
        return default


def _is_selected(row: dict[str, Any]) -> bool:
    raw = str(row.get("selected") or row.get("Selected") or "").strip().lower()
    if raw in ("yes", "1", "true", "y"):
        return True
    if raw in ("no", "0", "false", "n"):
        return False
    return True


def _parse_selected_features_csv(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                feat = row.get("feature") or row.get("Feature")
                if not feat:
                    continue
                out.append({
                    "feature": str(feat).strip(),
                    "selected": _is_selected(row),
                    "gain_importance_pct": _float_val(
                        row.get("gain_importance_pct") or row.get("Gain_Importance_Pct"),
                    ),
                    "final_rank": _int_val(row.get("final_rank") or row.get("Final_Rank")),
                })
    except (OSError, csv.Error):
        return []
    return out


def _load_gain_map_from_importance_csv(path: str) -> dict[str, float]:
    gains: dict[str, float] = {}
    if not os.path.isfile(path):
        return gains
    try:
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                feat = row.get("feature") or row.get("Feature")
                if not feat:
                    continue
                imp = (
                    row.get("gain_importance_pct")
                    or row.get("Gain_Importance_Pct")
                    or row.get("Importance")
                    or row.get("importance_pct")
                    or row.get("importance")
                )
                gains[str(feat).strip()] = _float_val(imp)
    except (OSError, csv.Error):
        pass
    return gains


def _load_model_selection_rows(data_dir: str, model_name: str) -> list[dict[str, Any]]:
    paths = model_artifact_paths(data_dir, model_name)
    pkg = paths["package_dir"]
    wf_path = os.path.join(pkg, "walk_forward", "selected_features.csv")
    if os.path.isfile(wf_path):
        return _parse_selected_features_csv(wf_path)

    config_path = paths["config_json"]
    features: list[str] = []
    if os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as fh:
                cfg = json.load(fh)
            features = [str(f).strip() for f in (cfg.get("features") or []) if str(f).strip()]
        except (OSError, json.JSONDecodeError):
            features = []

    if not features:
        return []

    gains = _load_gain_map_from_importance_csv(paths["feature_importance_csv"])
    ranked = sorted(
        features,
        key=lambda f: (-gains.get(f, 0.0), f),
    )
    return [
        {
            "feature": feat,
            "selected": True,
            "gain_importance_pct": gains.get(feat, 0.0),
            "final_rank": idx,
        }
        for idx, feat in enumerate(ranked, start=1)
    ]


def _groups_for_features(feature_names: list[str], registry: dict[str, Any] | None = None) -> list[str]:
    reg = registry or load_schema_registry()
    group_order = list(reg.get("groupOrder") or [])
    groups_meta = reg.get("groups") or {}
    selected = set(feature_names)
    enabled: list[str] = []
    for gid in group_order:
        feats = [str(f) for f in ((groups_meta.get(gid) or {}).get("features") or [])]
        if any(f in selected for f in feats):
            enabled.append(gid)
    return enabled


def _model_quality_raw(model: dict[str, Any]) -> float:
    """Higher = better model. Prefer production composite; fallback RMSE/MAE/direction."""
    prod = model.get("production_metrics") if isinstance(model.get("production_metrics"), dict) else {}
    metrics = model.get("metrics") if isinstance(model.get("metrics"), dict) else {}

    composite = _float_val(prod.get("composite_score") or metrics.get("composite_score"))
    if composite > 0:
        return composite

    rmse = _float_val(prod.get("rmse") or metrics.get("rmse"))
    mae = _float_val(prod.get("mae") or metrics.get("mae"))
    dir_acc = _float_val(prod.get("directional_accuracy_pct") or metrics.get("directional_accuracy_pct"))

    parts: list[float] = []
    if rmse > 0:
        parts.append(1.0 / rmse)
    if mae > 0:
        parts.append(1.0 / mae)
    if dir_acc > 0:
        parts.append(dir_acc / 100.0)
    if parts:
        return sum(parts) / len(parts)
    return 1.0


def _compute_model_weights(models: list[dict[str, Any]]) -> dict[str, float]:
    """Normalize quality scores to weights that sum to 1; down-rank weak models."""
    raw: dict[str, float] = {}
    for model in models:
        name = str(model.get("model_name") or "").strip()
        if not name:
            continue
        raw[name] = max(_model_quality_raw(model), 0.0)

    if not raw:
        return {}

    max_raw = max(raw.values())
    if max_raw <= 0:
        n = len(raw)
        return {k: 1.0 / n for k in raw}

    floor = max_raw * _MIN_QUALITY_WEIGHT_RATIO
    scaled = {k: max(v, floor) for k, v in raw.items()}
    total = sum(scaled.values())
    return {k: v / total for k, v in scaled.items()}


def _rank_quality_pct(weighted_avg_rank: float, *, max_rank: float) -> float:
    """Convert weighted final rank (lower is better) to a 0–100 score."""
    if weighted_avg_rank >= 9999 or max_rank <= 1:
        return 0.0
    span = max(max_rank - 1.0, 1.0)
    return max(0.0, min(100.0, 100.0 * (1.0 - (weighted_avg_rank - 1.0) / span)))


def _normalize_top_limit(top: Any) -> int | None:
    if top is None:
        return 75
    if isinstance(top, str):
        t = top.strip().lower()
        if t in ("all", "0", ""):
            return None
        try:
            top = int(t)
        except ValueError:
            return 75
    try:
        n = int(top)
    except (TypeError, ValueError):
        return 75
    if n <= 0:
        return None
    return n


def build_registry_auto_features(
    data_dir: str,
    *,
    top: Any = 75,
    underlying: str | None = None,
) -> dict[str, Any]:
    """Rank features from ready models; return selected set + summary stats."""
    top_limit = _normalize_top_limit(top)
    underlying_filter = normalize_underlying(underlying)
    ready_models = [
        m for m in list_trained_models(data_dir, lightweight=False)
        if str(m.get("status") or "").strip().lower() == "ready"
    ]
    matched_models = [
        m for m in ready_models
        if _model_matches_underlying(data_dir, m, underlying_filter)
    ]
    model_weights = _compute_model_weights(matched_models)

    agg: dict[str, dict[str, Any]] = {}
    models_analysed = 0
    models_used: list[str] = []
    max_rank_seen = 1.0

    for model in matched_models:
        name = str(model.get("model_name") or "").strip()
        if not name:
            continue
        mw = float(model_weights.get(name) or 0.0)
        if mw <= 0:
            continue
        rows = _load_model_selection_rows(data_dir, name)
        if not rows:
            continue
        selected_rows = [r for r in rows if r.get("selected")]
        if not selected_rows:
            continue
        models_analysed += 1
        models_used.append(name)
        for row in selected_rows:
            feat = str(row.get("feature") or "").strip()
            if not feat:
                continue
            gain = _float_val(row.get("gain_importance_pct"))
            rank = float(_int_val(row.get("final_rank")))
            max_rank_seen = max(max_rank_seen, rank)
            slot = agg.setdefault(feat, {
                "feature": feat,
                "weighted_select": 0.0,
                "weighted_gain_sum": 0.0,
                "weighted_rank_sum": 0.0,
                "weight_denom": 0.0,
                "select_count": 0,
            })
            slot["weighted_select"] += mw
            slot["weighted_gain_sum"] += mw * gain
            slot["weighted_rank_sum"] += mw * rank
            slot["weight_denom"] += mw
            slot["select_count"] += 1

    total_model_weight = sum(float(model_weights.get(n) or 0.0) for n in models_used)
    if total_model_weight <= 0:
        total_model_weight = 1.0

    ranked_rows: list[dict[str, Any]] = []
    for feat, slot in agg.items():
        weight_denom = float(slot["weight_denom"])
        weighted_selection_frequency_pct = round(
            100.0 * float(slot["weighted_select"]) / total_model_weight,
            2,
        )
        weighted_average_gain_pct = round(
            float(slot["weighted_gain_sum"]) / weight_denom,
            2,
        ) if weight_denom else 0.0
        weighted_average_final_rank = round(
            float(slot["weighted_rank_sum"]) / weight_denom,
            2,
        ) if weight_denom else 9999.0
        rank_quality_pct = round(
            _rank_quality_pct(weighted_average_final_rank, max_rank=max_rank_seen),
            2,
        )
        final_score = round(
            FREQ_WEIGHT * weighted_selection_frequency_pct
            + GAIN_WEIGHT * weighted_average_gain_pct
            + RANK_WEIGHT * rank_quality_pct,
            4,
        )
        ranked_rows.append({
            "feature": feat,
            "selection_frequency_pct": weighted_selection_frequency_pct,
            "average_gain_pct": weighted_average_gain_pct,
            "average_final_rank": weighted_average_final_rank,
            "rank_quality_pct": rank_quality_pct,
            "final_score": final_score,
            "models_selected": int(slot["select_count"]),
            "weighted_vote_pct": weighted_selection_frequency_pct,
        })

    ranked_rows.sort(
        key=lambda r: (
            -float(r["final_score"]),
            -float(r["selection_frequency_pct"]),
            -float(r["rank_quality_pct"]),
            -float(r["average_gain_pct"]),
            float(r["average_final_rank"]),
            str(r["feature"]),
        ),
    )

    if top_limit is None:
        picked = ranked_rows
    else:
        picked = ranked_rows[:top_limit]

    feature_names = [str(r["feature"]) for r in picked]
    registry = load_schema_registry()
    enabled_groups = _groups_for_features(feature_names, registry)

    avg_vote = (
        round(sum(float(r["selection_frequency_pct"]) for r in picked) / len(picked), 1)
        if picked else 0.0
    )
    avg_gain = (
        round(sum(float(r["average_gain_pct"]) for r in picked) / len(picked), 1)
        if picked else 0.0
    )
    avg_rank_quality = (
        round(sum(float(r.get("rank_quality_pct") or 0) for r in picked) / len(picked), 1)
        if picked else 0.0
    )

    model_weight_rows = [
        {
            "model_name": name,
            "quality_weight": round(float(model_weights.get(name) or 0.0), 4),
            "quality_raw": round(_model_quality_raw(m), 4),
        }
        for m in matched_models
        for name in [str(m.get("model_name") or "").strip()]
        if name and name in model_weights
    ]
    model_weight_rows.sort(key=lambda r: -float(r["quality_weight"]))

    return {
        "source": "all_ready_models",
        "ranking": "quality_weighted_registry_score",
        "underlying_filter": underlying_filter,
        "ready_models_total": len(ready_models),
        "ready_models_matched": len(matched_models),
        "ready_models_analysed": models_analysed,
        "formula": {
            "weighted_selection_frequency_weight": FREQ_WEIGHT,
            "weighted_average_gain_weight": GAIN_WEIGHT,
            "rank_quality_weight": RANK_WEIGHT,
            "model_weight": "production composite score (fallback: RMSE+MAE+direction)",
            "description": (
                "final_score = 0.50 × weighted_selection_frequency "
                "+ 0.30 × weighted_average_gain + 0.20 × rank_quality"
            ),
        },
        "top_limit": top_limit,
        "top_requested": top,
        "features_selected": len(feature_names),
        "average_vote_pct": avg_vote,
        "average_gain_pct": avg_gain,
        "average_rank_quality_pct": avg_rank_quality,
        "features": feature_names,
        "enabled_groups": enabled_groups,
        "model_weights": model_weight_rows,
        "ranking_rows": ranked_rows,
        "selected_rows": picked,
    }
