"""Single-day master dataset build — direct HTTP (no WebSocket daemon thread)."""

from __future__ import annotations

import threading
from typing import Any

from .master_build import MasterDatasetBuildOrchestrator
from .master_defaults import build_master_dataset_config
from .orchestrator import DatasetBuildConfig

_lock = threading.Lock()
_active = False


def master_build_in_progress() -> bool:
    return _active


def run_master_day_build(body: dict[str, Any]) -> dict[str, Any]:
    """Build one selected trading day into the master SQLite DB."""
    global _active
    with _lock:
        if _active:
            raise RuntimeError("A master day build is already running")
        _active = True

    progress_log: list[dict[str, Any]] = []

    try:
        market = str(body.get("market") or "NIFTY").upper()
        interval_sec = int(body.get("interval_sec") or 10)
        trading_day = str(body.get("trading_day") or "").strip()
        expiry = str(body.get("expiry") or "").strip()
        if not trading_day:
            raise ValueError("trading_day is required")
        if not expiry:
            raise ValueError("expiry is required")

        from tick_pipeline import DATA_DIR

        source = {
            "source_id": body.get("source_id")
            or f"{trading_day}|{market}|{expiry}",
            "trading_day": trading_day,
            "market": market,
            "expiry": expiry,
            "date": body.get("date") or body.get("date_label") or trading_day,
        }

        use_page_config = bool(
            body.get("sampling")
            and body.get("strike_selection")
            and body.get("prediction_targets")
            and body.get("feature_selection")
        )
        if use_page_config:
            from .master_naming import master_dataset_slug, resolve_master_db_path
            from .orchestrator import _load_feature_registry

            registry = _load_feature_registry()
            master_path = body.get("master_db_path") or resolve_master_db_path(
                DATA_DIR,
                market=market,
                sampling_interval_sec=interval_sec,
            )
            config = DatasetBuildConfig(
                dataset_name=master_dataset_slug(market=market, sampling_interval_sec=interval_sec),
                sources=[source],
                sampling=dict(body.get("sampling") or {}),
                strike_selection=dict(body.get("strike_selection") or {}),
                prediction_targets=dict(body.get("prediction_targets") or {}),
                feature_selection=dict(body.get("feature_selection") or {}),
                feature_registry=registry,
                data_dir=DATA_DIR,
                build_mode="append",
                storage_backend="master_sqlite",
                master_db_path=master_path,
                skip_parquet_export=True,
                skip_data_validation=bool(body.get("skip_data_validation", False)),
            )
        else:
            cfg = build_master_dataset_config(
                data_dir=DATA_DIR,
                market=market,
                interval_sec=interval_sec,
                sources=[source],
            )
            config = DatasetBuildConfig(
                dataset_name=cfg.dataset_name,
                sources=cfg.sources,
                sampling=cfg.sampling,
                strike_selection=cfg.strike_selection,
                prediction_targets=cfg.prediction_targets,
                feature_selection=cfg.feature_selection,
                feature_registry=cfg.feature_registry,
                data_dir=cfg.data_dir,
                build_mode=cfg.build_mode,
                storage_backend=cfg.storage_backend,
                master_db_path=cfg.master_db_path,
                skip_parquet_export=bool(body.get("skip_parquet_export", True)),
            )

        def on_progress(payload: dict[str, Any]) -> None:
            progress_log.append({
                "message": payload.get("message") or "",
                "stage": payload.get("stage"),
                "percent": payload.get("percent"),
                "rows": payload.get("rows"),
                "pipeline": payload.get("pipeline"),
            })

        result = MasterDatasetBuildOrchestrator(config).run(on_progress=on_progress)
        out: dict[str, Any] = {
            "status": result.status,
            "trading_day": trading_day,
            "market": market,
            "interval_sec": interval_sec,
            "master_db_path": config.master_db_path,
            "progress_log": progress_log,
            "warnings": list(result.warnings or []),
            "error": result.error,
        }
        if result.dataset_stats:
            out["dataset_stats"] = result.dataset_stats
        if result.status not in ("completed",):
            err = result.error or "; ".join(result.warnings or []) or "Build did not complete"
            raise RuntimeError(err)
        return out
    finally:
        with _lock:
            _active = False
