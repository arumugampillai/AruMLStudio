"""Meta Engine — extensible PredictionMeta dict from PredictionSnapshot history."""

from __future__ import annotations

import statistics
from typing import Any

from .history import PredictionSnapshotHistory
from .snapshot import PredictionMeta, PredictionSnapshot


def _ok_predictions(snapshot: PredictionSnapshot) -> list[float]:
    out: list[float] = []
    for result in snapshot.results.values():
        if result.status == "ok" and result.prediction is not None:
            out.append(float(result.prediction))
    return out


def _agreement_count(values: list[float], mean: float, std: float) -> int:
    if not values:
        return 0
    if std <= 0:
        return len(values)
    return sum(1 for v in values if abs(v - mean) <= std)


def _slope_over(history: PredictionSnapshotHistory, key: str, window_sec: float) -> float | None:
    points = history.meta_series(key, window_sec)
    if len(points) < 2:
        return None
    t0, v0 = points[0]
    t1, v1 = points[-1]
    dt = t1 - t0
    if dt <= 0:
        return None
    return round((v1 - v0) / dt, 6)


class MetaEngine:
    """Publish PredictionMeta dict — add keys without changing consumers."""

    def compute(
        self,
        snapshot: PredictionSnapshot,
        history: PredictionSnapshotHistory,
    ) -> PredictionMeta:
        values = _ok_predictions(snapshot)
        total_models = len(snapshot.results)
        models_ok = snapshot.models_ok
        models_failed = snapshot.models_failed

        meta: dict[str, Any] = {
            "models_ok": models_ok,
            "models_failed": models_failed,
            "models_total": total_models,
            "agreement_denominator": models_ok,
        }

        if values:
            mean = statistics.fmean(values)
            median = statistics.median(values)
            std = statistics.pstdev(values) if len(values) > 1 else 0.0
            vmin = min(values)
            vmax = max(values)
            spread = vmax - vmin
            agree = _agreement_count(values, mean, std)

            meta.update({
                "mean": round(mean, 4),
                "median": round(float(median), 4),
                "std": round(std, 4),
                "min": round(vmin, 4),
                "max": round(vmax, 4),
                "spread": round(spread, 4),
                "agreement": round(agree / models_ok, 4) if models_ok else 0.0,
                "agreement_count": agree,
            })
        else:
            meta.update({
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
                "spread": None,
                "agreement": 0.0,
                "agreement_count": 0,
            })

        for key in ("mean", "std", "spread"):
            current = meta.get(key)
            if current is None:
                continue
            for window in (1.0, 5.0, 10.0):
                past = history.meta_value_at_offset(key, window)
                if past is not None:
                    meta[f"{key}_change_{int(window)}s"] = round(float(current) - past, 4)
            slope = _slope_over(history, key, 10.0)
            if slope is not None and current is not None:
                series = history.meta_series(key, 10.0)
                if series:
                    series = list(series) + [(snapshot.timestamp, float(current))]
                    t0, v0 = series[0]
                    t1, v1 = series[-1]
                    dt = t1 - t0
                    if dt > 0:
                        meta[f"{key}_slope"] = round((v1 - v0) / dt, 6)

        vel = _slope_over(history, "mean", 1.0)
        if vel is not None:
            meta["prediction_velocity"] = vel
        prior_vel = history.meta_value_at_offset("prediction_velocity", 1.0)
        if vel is not None and prior_vel is not None:
            meta["prediction_acceleration"] = round(vel - prior_vel, 6)
        trend = _slope_over(history, "mean", 5.0)
        if trend is not None:
            meta["prediction_trend"] = trend
        vol = history.rolling_std("mean", 10.0)
        if vol is not None:
            meta["prediction_volatility"] = round(vol, 4)

        history.append(snapshot, meta)
        return PredictionMeta.create(timestamp=snapshot.timestamp, values=meta)
