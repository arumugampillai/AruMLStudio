"""Rolling live metrics for the prediction meta build dashboard."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds != seconds:
        return "—"
    sec = int(seconds)
    if sec < 60:
        return f"{sec}s"
    mins, s = divmod(sec, 60)
    if mins < 60:
        return f"{mins}m {s:02d}s"
    hrs, mins = divmod(mins, 60)
    return f"{hrs}h {mins:02d}m"


@dataclass
class LiveMetricsTracker:
    """Accumulates batch stats and exposes dashboard snapshot."""

    models_per_row: int = 0
    started_monotonic: float = field(default_factory=time.perf_counter)
    rows_at_start: int = 0

    rows_done: int = 0
    rows_total: int | None = None
    batches: int = 0

    predictions_ok: int = 0
    predictions_failed: int = 0
    skipped_rows: int = 0
    failed_model_rows: int = 0

    outcome_completed: int = 0
    outcome_pending: int = 0

    _sum_agreement: float = 0.0
    _count_agreement: int = 0
    _sum_spread: float = 0.0
    _count_spread: int = 0
    _sum_direction: float = 0.0
    _count_direction: int = 0

    _ema_feature_ms: float | None = None
    _ema_prediction_ms: float | None = None
    _ema_sqlite_ms: float | None = None
    _ema_rows_per_sec: float | None = None

    registry_cache_pct: float = 100.0
    model_cache_pct: float = 100.0
    feature_cache_pct: float = 100.0

    prediction_version: int | None = None
    status: str = "running"

    def _ema(self, prev: float | None, value: float, *, alpha: float = 0.25) -> float:
        if prev is None:
            return value
        return prev * (1.0 - alpha) + value * alpha

    def set_cache_stats(
        self,
        *,
        registry_cache_hit: bool,
        models_loaded_from_disk: int,
        model_count: int,
    ) -> None:
        self.registry_cache_pct = 100.0 if registry_cache_hit else 99.0
        if model_count > 0:
            self.model_cache_pct = round(100.0 * (1.0 - models_loaded_from_disk / model_count), 1)

    def update_batch(
        self,
        *,
        batch_rows: int,
        feature_ms: float,
        prediction_ms: float,
        sqlite_ms: float,
        feature_valid_pct: float,
        rows_done: int,
        batch_stats: dict[str, Any],
    ) -> None:
        self.batches += 1
        self.rows_done = rows_done
        per_row_feat = feature_ms / max(batch_rows, 1)
        per_row_pred = prediction_ms / max(batch_rows, 1)
        per_row_sql = sqlite_ms / max(batch_rows, 1)
        self._ema_feature_ms = self._ema(self._ema_feature_ms, per_row_feat)
        self._ema_prediction_ms = self._ema(self._ema_prediction_ms, per_row_pred)
        self._ema_sqlite_ms = self._ema(self._ema_sqlite_ms, per_row_sql)
        self.feature_cache_pct = self._ema(self.feature_cache_pct, feature_valid_pct, alpha=0.15)

        elapsed = max(time.perf_counter() - self.started_monotonic, 0.001)
        processed = max(rows_done - self.rows_at_start, 0)
        instant_rps = processed / elapsed
        self._ema_rows_per_sec = self._ema(self._ema_rows_per_sec, instant_rps, alpha=0.12)

        self.predictions_ok += int(batch_stats.get("predictions_ok") or 0)
        self.predictions_failed += int(batch_stats.get("predictions_failed") or 0)
        self.skipped_rows += int(batch_stats.get("skipped_rows") or 0)
        self.failed_model_rows += int(batch_stats.get("failed_model_rows") or 0)
        self.outcome_completed += int(batch_stats.get("outcome_completed") or 0)
        self.outcome_pending += int(batch_stats.get("outcome_pending") or 0)

        for val in batch_stats.get("agreement_values") or []:
            if val is not None:
                self._sum_agreement += float(val)
                self._count_agreement += 1
        for val in batch_stats.get("spread_values") or []:
            if val is not None:
                self._sum_spread += float(val)
                self._count_spread += 1
        for val in batch_stats.get("direction_values") or []:
            if val is not None:
                self._sum_direction += float(val)
                self._count_direction += 1

    def progress_pct(self) -> float:
        total = self.rows_total or 0
        if total <= 0:
            return 0.0
        return round(min(100.0, 100.0 * self.rows_done / total), 1)

    def eta_sec(self) -> float | None:
        total = self.rows_total
        rps = self._ema_rows_per_sec
        if not total or not rps or rps <= 0:
            return None
        remaining = max(total - self.rows_done, 0)
        return remaining / rps

    def snapshot(self) -> dict[str, Any]:
        rps = self._ema_rows_per_sec or 0.0
        mpr = max(self.models_per_row, 1)
        eta = self.eta_sec()
        avg_agreement = (
            round(100.0 * self._sum_agreement / self._count_agreement, 1)
            if self._count_agreement else None
        )
        avg_spread = (
            round(self._sum_spread / self._count_spread, 2)
            if self._count_spread else None
        )
        direction_accuracy = (
            round(100.0 * self._sum_direction / self._count_direction, 1)
            if self._count_direction else None
        )
        return {
            "status": self.status,
            "progress_pct": self.progress_pct(),
            "eta_sec": round(eta, 1) if eta is not None else None,
            "eta_label": _fmt_eta(eta),
            "rows_done": self.rows_done,
            "rows_total": self.rows_total,
            "rows_per_sec": round(rps),
            "predictions_per_sec": round(rps * mpr),
            "models_per_row": self.models_per_row,
            "timing_ms": {
                "feature_build": round(self._ema_feature_ms or 0.0, 2),
                "prediction": round(self._ema_prediction_ms or 0.0, 2),
                "sqlite": round(self._ema_sqlite_ms or 0.0, 2),
            },
            "cache_pct": {
                "registry_cache": round(self.registry_cache_pct, 1),
                "model_cache": round(self.model_cache_pct, 1),
                "feature_cache": round(self.feature_cache_pct, 1),
            },
            "outcomes": {
                "completed": self.outcome_completed,
                "pending": self.outcome_pending,
            },
            "failed_models": self.predictions_failed,
            "skipped_rows": self.skipped_rows,
            "failed_model_rows": self.failed_model_rows,
            "quality": {
                "avg_agreement_pct": avg_agreement,
                "avg_spread": avg_spread,
                "direction_accuracy_pct": direction_accuracy,
            },
            "prediction_version": self.prediction_version,
            "batches": self.batches,
            "updated_at": _utc_now(),
        }
