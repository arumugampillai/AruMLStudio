"""Live progress events for shared FeatureSnapshot + prediction pipeline."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable

from .timing_profiler import InferenceTimingProfiler
from .versions import feature_version

EmitFn = Callable[[dict[str, Any]], None]

_MARKET_STATE_ITEMS: tuple[tuple[str, str], ...] = (
    ("spot", "Spot"),
    ("option_chain", "Option Chain"),
    ("oi", "OI"),
    ("volume", "Volume"),
    ("greeks", "Greeks"),
)


def _short_group_label(gid: str, label: str) -> str:
    text = str(label or gid).strip()
    if not text:
        return str(gid)
    for sep in ("—", "-", "/", ":"):
        if sep in text:
            text = text.split(sep, 1)[0].strip()
    return text.split()[0] if text else str(gid)


class InferenceProgressReporter:
    """Push compact progress suitable for Model Prediction UI."""

    def __init__(
        self,
        *,
        union_features: list[str],
        model_count: int,
        groups: list[dict[str, Any]],
        on_emit: EmitFn,
        profiler: InferenceTimingProfiler | None = None,
    ) -> None:
        self._on_emit = on_emit
        self._profiler = profiler
        self._t0 = time.perf_counter()
        self._phase = "starting"
        self._union_count = len(union_features)
        self._registry_total = self._union_count
        self._built_features = 0
        self._model_count = int(model_count)
        self._models_done = 0
        self._groups: dict[str, dict[str, Any]] = {
            str(g["id"]): {
                "id": g["id"],
                "label": g.get("label") or g["id"],
                "short": g.get("short") or _short_group_label(str(g["id"]), str(g.get("label") or g["id"])),
                "status": "pending",
                "built": 0,
                "total": int(g.get("total") or 0),
            }
            for g in groups
        }
        self._group_order = [str(g["id"]) for g in groups]
        self._market_state = [
            {"id": item_id, "label": label, "done": False}
            for item_id, label in _MARKET_STATE_ITEMS
        ]
        self._waiting_on: list[str] = []
        self._status_detail = ""
        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread: threading.Thread | None = None
        self._spot_ticks = 0
        self._chain_ticks = 0

    def _elapsed(self) -> float:
        return round(time.perf_counter() - self._t0, 2)

    def _groups_completed(self) -> int:
        return sum(1 for g in self._groups.values() if g["status"] == "done")

    def _groups_total(self) -> int:
        return len(self._groups)

    def _pct(self) -> int:
        if self._phase == "done":
            return 100
        if self._phase == "meta":
            return 98
        if self._phase == "predict":
            base = 82
            if self._model_count <= 0:
                return base
            return min(97, base + int(15 * self._models_done / self._model_count))
        if self._phase == "snapshot":
            return 80
        if self._phase == "feature_build":
            done = self._groups_completed()
            total = max(self._groups_total(), 1)
            return min(79, 8 + int(72 * done / total))
        if self._phase == "market_state":
            done = sum(1 for m in self._market_state if m["done"])
            return min(7, 1 + done)
        return 1

    def _eta_sec(self) -> float | None:
        elapsed = self._elapsed()
        pct = self._pct()
        if pct <= 2 or pct >= 99:
            return None
        remaining = elapsed * (100.0 - pct) / max(pct, 1)
        return round(remaining, 1)

    def _payload(self, *, event: str = "PROGRESS", extra: dict[str, Any] | None = None) -> dict[str, Any]:
        groups_list = [self._groups[gid] for gid in self._group_order if gid in self._groups]
        active = next((g for g in groups_list if g["status"] == "active"), None)
        payload: dict[str, Any] = {
            "type": "MODEL_INFERENCE",
            "event": event,
            "phase": self._phase,
            "title": self._title(),
            "pct": self._pct(),
            "built_features": self._built_features,
            "union_features": self._union_count,
            "registry_total_features": self._registry_total,
            "groups_completed": self._groups_completed(),
            "groups_total": self._groups_total(),
            "groups": groups_list,
            "active_group": active,
            "market_state": list(self._market_state),
            "models_loaded": self._model_count,
            "models_waiting": max(0, self._model_count - self._models_done),
            "models_done": self._models_done,
            "elapsed_sec": self._elapsed(),
            "eta_sec": self._eta_sec(),
            "waiting_on": list(self._waiting_on),
            "feature_version": feature_version(),
            "status_detail": self._status_detail,
            "spot_ticks": self._spot_ticks,
            "chain_ticks": self._chain_ticks,
        }
        if extra:
            payload.update(extra)
        return payload

    def _title(self) -> str:
        titles = {
            "starting": "Starting inference…",
            "market_state": "Loading Market State…",
            "feature_build": "Building Shared FeatureSnapshot…",
            "snapshot": "FeatureSnapshot ready",
            "predict": "Running Predictions…",
            "meta": "Computing Prediction Meta…",
            "done": "Prediction Snapshot ✓ Ready",
        }
        return titles.get(self._phase, "Working…")

    def emit(self, *, event: str = "PROGRESS", extra: dict[str, Any] | None = None) -> None:
        t0 = time.perf_counter()
        self._on_emit(self._payload(event=event, extra=extra))
        emit_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        if self._profiler is not None:
            bucket = f"progress_emit_{self._phase}"
            self._profiler.add_ms(bucket, emit_ms, event=event)
            self._profiler.add_ms("progress_emit_total", emit_ms, event=event, phase=self._phase)

    def set_detail(self, detail: str) -> None:
        self._status_detail = str(detail or "")
        self.emit()

    def set_tick_counts(self, *, spot_ticks: int = 0, chain_ticks: int = 0) -> None:
        self._spot_ticks = int(spot_ticks)
        self._chain_ticks = int(chain_ticks)
        self.emit()

    def start_heartbeat(self, interval_sec: float = 1.0) -> None:
        self.stop_heartbeat()
        self._heartbeat_stop.clear()

        def _beat() -> None:
            while not self._heartbeat_stop.wait(interval_sec):
                self.emit(event="HEARTBEAT")

        self._heartbeat_thread = threading.Thread(target=_beat, daemon=True, name="inference-progress-hb")
        self._heartbeat_thread.start()

    def stop_heartbeat(self) -> None:
        self._heartbeat_stop.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=0.2)
        self._heartbeat_thread = None

    def started(self, *, model_count: int, union_features: int) -> None:
        self._model_count = model_count
        self._union_count = union_features
        self._phase = "starting"
        self.emit()

    def market_state_begin(self) -> None:
        self._phase = "market_state"
        self.emit()

    def market_state_item(self, item_id: str) -> None:
        for item in self._market_state:
            if item["id"] == item_id:
                item["done"] = True
        self.emit()

    def market_state_ready(self) -> None:
        for item in self._market_state:
            item["done"] = True
        self.emit()

    def feature_build_begin(self) -> None:
        self._phase = "feature_build"
        self.emit()

    def mark_all_groups_done(self) -> None:
        for gid in self._group_order:
            self.group_done(gid)

    def group_start(self, gid: str, label: str = "") -> None:
        g = self._groups.get(str(gid))
        if not g:
            return
        g["status"] = "active"
        if label:
            g["label"] = label
            g["short"] = _short_group_label(str(gid), label)
        self._waiting_on = [g["short"] for g in self._groups.values() if g["status"] in ("pending", "active")]
        self.emit()

    def group_progress(self, gid: str, current: int, total: int) -> None:
        g = self._groups.get(str(gid))
        if not g:
            return
        g["status"] = "active"
        g["built"] = min(int(current), int(g["total"] or total))
        if total > 0 and not g["total"]:
            g["total"] = int(total)
        self._recompute_built_features()
        self.emit()

    def group_done(self, gid: str) -> None:
        g = self._groups.get(str(gid))
        if not g:
            return
        g["status"] = "done"
        if g["total"]:
            g["built"] = g["total"]
        self._recompute_built_features()
        self._waiting_on = [x["short"] for x in self._groups.values() if x["status"] in ("pending", "active")]
        self.emit()

    def _recompute_built_features(self) -> None:
        self._built_features = sum(int(g.get("built") or 0) for g in self._groups.values())

    def snapshot_ready(self, *, built_count: int, build_sec: float, warmup_complete: bool) -> None:
        self._phase = "snapshot"
        self._built_features = min(int(built_count), self._union_count)
        self.emit(extra={
            "snapshot_ready": True,
            "build_sec": round(build_sec, 3),
            "warmup_complete": warmup_complete,
        })

    def predict_begin(self) -> None:
        self._phase = "predict"
        self._models_done = 0
        self.emit()

    def predict_model(self, index: int, model_id: str, *, ok: bool, prediction: float | None = None, error: str | None = None) -> None:
        self._models_done = int(index)
        self.emit(extra={
            "prediction_index": index,
            "prediction_model": model_id,
            "prediction_ok": ok,
            "prediction_value": prediction,
            "prediction_error": error,
        })

    def meta_begin(self) -> None:
        self._phase = "meta"
        self.emit()

    def done(self, result: dict[str, Any]) -> None:
        self.stop_heartbeat()
        self._phase = "done"
        self._models_done = self._model_count
        self._built_features = min(
            int(result.get("shared_feature_count") or self._built_features),
            self._union_count,
        )
        meta = result.get("prediction_meta") or {}
        self.emit(event="DONE", extra={
            "result": result,
            "mean_prediction": meta.get("mean"),
            "median_prediction": meta.get("median"),
            "spread": meta.get("spread"),
            "agreement": meta.get("agreement"),
            "models_ok": result.get("models_ok"),
            "models_failed": result.get("models_failed"),
            "build_sec": result.get("feature_build_sec"),
            "total_sec": result.get("total_sec"),
            "warmup_complete": result.get("warmup_complete"),
            "warmup_label": result.get("warmup_label"),
        })

    def error(self, detail: str) -> None:
        self.stop_heartbeat()
        self.emit(event="ERROR", extra={"detail": detail})


def build_group_plan(
    replay_config: dict[str, Any],
    union_features: list[str],
) -> list[dict[str, Any]]:
    from chain_replay_ml.dataset_builder.schema_registry import load_feature_registry
    from chain_replay_ml.replay_feature_scoring import merge_replay_feature_build_plan

    registry = load_feature_registry()
    enabled = list(
        replay_config.get("feature_groups_implemented") or replay_config.get("feature_groups") or []
    )
    enabled, _implemented, _pending, per_group = merge_replay_feature_build_plan(
        enabled, registry, union_features,
    )
    groups: list[dict[str, Any]] = []
    for gid in enabled:
        feats = per_group.get(gid) or []
        if not feats:
            continue
        label = str((registry.get("groups") or {}).get(gid, {}).get("label") or gid)
        groups.append({
            "id": gid,
            "label": label,
            "short": _short_group_label(gid, label),
            "total": len(feats),
        })
    return groups
