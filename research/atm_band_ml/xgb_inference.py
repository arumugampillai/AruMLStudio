"""
Registry-model inference for ATM-band live scoring.

Loads the active/latest model from ``data/models/`` (Model Builder registry).
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.atm_band_ml.band_evaluator import BandEvalRow, BandEvalSnapshot
from research.atm_band_ml.feature_builder import (
    model_feature_vector,
    replay_feature_columns,
)

_CHART_DIR = Path(__file__).resolve().parents[2] / "apps"
DEFAULT_DATA_DIR = _CHART_DIR / "data"
DEFAULT_SCORE_THRESHOLD = 3.0
DELTA_BANDS = ("A", "B", "C")


def _ensure_chart_on_path() -> None:
    chart_dir = str(_CHART_DIR)
    if chart_dir not in sys.path:
        sys.path.insert(0, chart_dir)


def default_model_dir() -> Path:
    """Deprecated alias — registry packages live under data/models."""
    return DEFAULT_DATA_DIR / "models"


def default_data_dir() -> Path:
    return DEFAULT_DATA_DIR


def discover_model_stamps(model_dir: str | Path | None = None) -> list[str]:
    """Newest-first registry model names (compat alias for UI)."""
    return discover_registry_models(model_dir)


def discover_registry_models(data_dir: str | Path | None = None) -> list[str]:
    _ensure_chart_on_path()
    from chain_replay_ml.training.registry import list_trained_models

    root = _normalize_data_dir(data_dir)
    rows = list_trained_models(root, lightweight=True)
    return [str(r.get("model_name") or "") for r in rows if r.get("model_name")]


def stamp_has_complete_models(stamp: str, model_dir: str | Path | None = None) -> bool:
    _ensure_chart_on_path()
    from chain_replay_ml.training.registry import get_trained_model

    root = _normalize_data_dir(model_dir)
    return get_trained_model(root, stamp) is not None


def delta_band_name(delta: float | None) -> str | None:
    if delta is None:
        return None
    try:
        abs_delta = abs(float(delta))
    except (TypeError, ValueError):
        return None
    if 0.40 <= abs_delta <= 0.50:
        return "A"
    if 0.25 <= abs_delta < 0.40:
        return "B"
    if 0.15 <= abs_delta < 0.25:
        return "C"
    return None


def compute_expectancy_score(
    p_hit: float,
    pred_max_return: float,
    pred_min_return: float,
) -> float:
    return float(p_hit) * float(pred_max_return) - (1.0 - float(p_hit)) * abs(float(pred_min_return))


def _registry_score(pred_ltp: float, spot_ltp: float, target: str) -> float:
    if spot_ltp <= 0:
        return 0.0
    if str(target).startswith("future_ltp"):
        return (float(pred_ltp) - float(spot_ltp)) / float(spot_ltp) * 100.0
    return float(pred_ltp)


def _import_xgboost():
    try:
        from xgboost import XGBRegressor
    except ImportError as exc:
        raise ImportError(
            "xgboost is required for registry inference (pip install xgboost)"
        ) from exc
    return XGBRegressor


def _prefer_gpu_inference() -> bool:
    """
    GPU is preferred by default for live inference.

    Set ``XGB_USE_GPU=0`` to force CPU.
    """
    raw = str(os.environ.get("XGB_USE_GPU", "1")).strip().lower()
    return raw not in ("0", "false", "no", "off")


def _apply_inference_runtime(model: Any) -> None:
    """
    Configure loaded XGBoost model for fast inference.

    GPU-first with CPU fallback so environments without CUDA keep working.
    """
    use_gpu = _prefer_gpu_inference()
    try:
        if use_gpu:
            # XGBoost 2.x path: ``device='cuda'``.
            model.set_params(device="cuda", tree_method="hist")
        else:
            model.set_params(device="cpu", tree_method="hist")
    except Exception:
        # Keep model usable even if runtime params are unsupported.
        pass

    # Booster-level params are accepted across wrapper/runtime versions.
    try:
        booster = model.get_booster()
        if use_gpu:
            try:
                booster.set_param({"device": "cuda"})
            except Exception:
                pass
            try:
                booster.set_param({"predictor": "gpu_predictor"})
            except Exception:
                pass
            booster.set_param({"tree_method": "hist"})
        else:
            try:
                booster.set_param({"device": "cpu"})
            except Exception:
                pass
            try:
                booster.set_param({"predictor": "cpu_predictor"})
            except Exception:
                pass
            booster.set_param({"tree_method": "hist"})
    except Exception:
        pass


def _normalize_data_dir(path: str | Path | None) -> str:
    root = Path(path or DEFAULT_DATA_DIR)
    if root.name in ("ml_models", "models"):
        return str(root.parent)
    return str(root)


def load_registry_model(
    model_name: str | None = None,
    *,
    data_dir: str | Path | None = None,
) -> tuple[str, Any, list[str], str]:
    """Return (model_name, booster, feature_columns, target)."""
    _ensure_chart_on_path()
    from chain_replay_ml.training.paths import model_artifact_paths
    from chain_replay_ml.training.registry import resolve_default_model_name

    root = _normalize_data_dir(data_dir)
    name = resolve_default_model_name(root, model_name)
    if not name:
        raise FileNotFoundError(f"No registry model found under {root}/models")
    paths = model_artifact_paths(root, name)
    config_path = paths["config_json"]
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Registry model package incomplete: {name}")

    model_path = ""
    tmeta_path = paths.get("training_metadata_json")
    if tmeta_path and os.path.isfile(tmeta_path):
        try:
            with open(tmeta_path, encoding="utf-8") as fh:
                tmeta = json.load(fh)
            production_name = str(tmeta.get("production_model") or "").strip()
            if production_name:
                candidate_path = os.path.join(paths["package_dir"], production_name)
                if os.path.isfile(candidate_path):
                    model_path = candidate_path
        except Exception:
            pass
    if not model_path and os.path.isfile(paths.get("model_ubj", "")):
        model_path = paths["model_ubj"]
    if not model_path and os.path.isfile(paths["model_json"]):
        model_path = paths["model_json"]
    if not model_path:
        raise FileNotFoundError(f"Registry model package missing production model: {name}")
    with open(config_path, encoding="utf-8") as fh:
        config = json.load(fh)
    features = list(config.get("features") or [])
    target = str(config.get("target") or "")
    if not features or not target:
        raise FileNotFoundError(f"Registry model config missing features/target: {name}")
    XGBRegressor = _import_xgboost()
    model = XGBRegressor()
    model.load_model(model_path)
    _apply_inference_runtime(model)
    return name, model, features, target


def load_models_for_stamp(
    stamp: str,
    *,
    model_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Compat shim — ``stamp`` is treated as registry ``model_name``."""
    name, model, features, target = load_registry_model(stamp, data_dir=model_dir)
    bundle = {"reg": model, "features": features, "target": target, "model_name": name}
    return {band: bundle for band in DELTA_BANDS}


@dataclass(frozen=True, slots=True)
class ScoredStrike:
    ts: float
    token: str
    symbol: str
    option_type: str
    strike: float
    delta: float | None
    delta_band: str | None
    ltp: float | None
    P_hit: float | None
    pred_max_return: float | None
    pred_min_return: float | None
    score: float | None
    model_complete: bool
    reason: str = ""

    @property
    def scorable(self) -> bool:
        return self.score is not None and self.delta_band is not None

    @property
    def above_threshold(self) -> bool:
        return self.score is not None and float(self.score) >= DEFAULT_SCORE_THRESHOLD


class AtmBandModelScorer:
    """Lazy-loaded registry XGB model for live band scoring."""

    def __init__(
        self,
        *,
        data_dir: str | Path | None = None,
        model_dir: str | Path | None = None,
        stamp: str | None = None,
        model_name: str | None = None,
        feature_columns: Sequence[str] | None = None,
    ) -> None:
        self.data_dir = Path(_normalize_data_dir(data_dir or model_dir))
        self.model_name = model_name or stamp
        self.stamp = self.model_name
        self.feature_columns = list(feature_columns or [])
        self._model = None
        self._registry_features: list[str] = []
        self._target = ""

    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, stamp: str | None = None, model_name: str | None = None) -> str:
        use_name = model_name or stamp or self.model_name
        name, model, features, target = load_registry_model(
            use_name, data_dir=self.data_dir,
        )
        self._model = model
        self._registry_features = list(features)
        if not self.feature_columns:
            self.feature_columns = list(features)
        self._target = target
        self.model_name = name
        self.stamp = name
        return name

    def ensure_loaded(self) -> None:
        if self._model is None:
            self.load()

    def _feature_vector(
        self,
        features: Mapping[str, Any],
        *,
        fill_missing: float | None = None,
    ) -> list[float] | None:
        cols = self._registry_features or self.feature_columns or replay_feature_columns()
        vec: list[float] = []
        for col in cols:
            val = features.get(col)
            if val is None:
                if fill_missing is not None:
                    vec.append(float(fill_missing))
                    continue
                return None
            try:
                fval = float(val)
            except (TypeError, ValueError):
                if fill_missing is not None:
                    vec.append(float(fill_missing))
                    continue
                return None
            if math.isnan(fval):
                if fill_missing is not None:
                    vec.append(float(fill_missing))
                    continue
                return None
            vec.append(fval)
        return vec

    def score_features(
        self,
        features: Mapping[str, Any],
        *,
        ts: float = 0.0,
        token: str = "",
        symbol: str = "",
        option_type: str = "",
        strike: float = 0.0,
        fill_missing: float | None = None,
    ) -> ScoredStrike:
        self.ensure_loaded()
        assert self._model is not None

        delta_raw = features.get("delta")
        try:
            delta = float(delta_raw) if delta_raw is not None else None
        except (TypeError, ValueError):
            delta = None
        ltp_raw = features.get("ltp")
        try:
            ltp = float(ltp_raw) if ltp_raw is not None else None
        except (TypeError, ValueError):
            ltp = None

        band = delta_band_name(delta)
        vec = self._feature_vector(features, fill_missing=fill_missing)
        if vec is None:
            return ScoredStrike(
                ts=float(ts),
                token=str(token),
                symbol=str(symbol),
                option_type=str(option_type),
                strike=float(strike),
                delta=delta,
                delta_band=band,
                ltp=ltp,
                P_hit=None,
                pred_max_return=None,
                pred_min_return=None,
                score=None,
                model_complete=False,
                reason="incomplete_registry_features",
            )
        if band is None:
            return ScoredStrike(
                ts=float(ts),
                token=str(token),
                symbol=str(symbol),
                option_type=str(option_type),
                strike=float(strike),
                delta=delta,
                delta_band=None,
                ltp=ltp,
                P_hit=None,
                pred_max_return=None,
                pred_min_return=None,
                score=None,
                model_complete=True,
                reason="delta_out_of_band",
            )
        if ltp is None or ltp <= 0:
            return ScoredStrike(
                ts=float(ts),
                token=str(token),
                symbol=str(symbol),
                option_type=str(option_type),
                strike=float(strike),
                delta=delta,
                delta_band=band,
                ltp=ltp,
                P_hit=None,
                pred_max_return=None,
                pred_min_return=None,
                score=None,
                model_complete=True,
                reason="no_ltp",
            )

        x_row = np.array([vec], dtype=float)
        pred_ltp = float(self._model.predict(x_row)[0])
        score = _registry_score(pred_ltp, ltp, self._target)
        return ScoredStrike(
            ts=float(ts),
            token=str(token),
            symbol=str(symbol),
            option_type=str(option_type),
            strike=float(strike),
            delta=delta,
            delta_band=band,
            ltp=ltp,
            P_hit=0.5,
            pred_max_return=score,
            pred_min_return=0.0,
            score=score,
            model_complete=True,
            reason="",
        )

    def score_band_row(self, row: BandEvalRow) -> ScoredStrike:
        contract = row.contract
        return self.score_features(
            row.result.features,
            ts=row.ts,
            token=contract.token,
            symbol=contract.symbol,
            option_type=contract.option_type,
            strike=contract.strike,
        )

    def score_band_snapshot(self, snapshot: BandEvalSnapshot) -> list[ScoredStrike]:
        return [self.score_band_row(r) for r in snapshot.rows]

    def pick_top_scored(
        self,
        scored: Sequence[ScoredStrike],
        *,
        min_score: float = DEFAULT_SCORE_THRESHOLD,
    ) -> ScoredStrike | None:
        candidates = [
            s for s in scored if s.scorable and s.score is not None and float(s.score) >= min_score
        ]
        if not candidates:
            return None
        return max(candidates, key=lambda s: float(s.score or 0.0))

    def pick_top_from_snapshot(
        self,
        snapshot: BandEvalSnapshot,
        *,
        min_score: float = DEFAULT_SCORE_THRESHOLD,
    ) -> ScoredStrike | None:
        return self.pick_top_scored(self.score_band_snapshot(snapshot), min_score=min_score)


# Back-compat aliases
DEFAULT_MODEL_DIR = DEFAULT_DATA_DIR / "models"
DEFAULT_LIVE_MODEL_STAMP: str | None = None
