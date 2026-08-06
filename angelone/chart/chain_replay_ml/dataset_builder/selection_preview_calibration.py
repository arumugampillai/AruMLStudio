"""Record preview-vs-actual statistics to calibrate selection estimates."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from .dataset_selection_engine import DatasetSelectionEngine, DatasetSelectionSpec, SelectionPreviewResult


def _calibration_path(data_dir: str) -> str:
    return os.path.join(data_dir, "dataset_selection_calibration.jsonl")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_error_pct(estimated: int, actual: int) -> float | None:
    if actual <= 0:
        return None if estimated <= 0 else 100.0
    return round((estimated - actual) / actual * 100.0, 2)


def record_selection_calibration(
    data_dir: str,
    *,
    build_kind: str,
    spec: DatasetSelectionSpec,
    preview: SelectionPreviewResult | dict[str, Any] | None,
    actual_rows: int,
    actual_tokens: int | None = None,
    actual_days: int | None = None,
    actual_size_mb: float | None = None,
    build_job_id: str | None = None,
    master_db_path: str | None = None,
    metadata_version: int | None = None,
) -> dict[str, Any]:
    """Append one preview-vs-actual calibration record."""
    preview_doc: dict[str, Any]
    if isinstance(preview, SelectionPreviewResult):
        preview_doc = preview.to_api_dict()
    elif isinstance(preview, dict):
        preview_doc = dict(preview)
    else:
        preview_doc = {}

    est_rows = int(preview_doc.get("estimated_rows") or 0)
    record: dict[str, Any] = {
        "recorded_at": _utc_now(),
        "build_kind": str(build_kind),
        "build_job_id": build_job_id,
        "spec_fingerprint": spec.fingerprint(),
        "spec": spec.to_dict(),
        "master_db_path": master_db_path,
        "metadata_version": metadata_version or preview_doc.get("metadata_version"),
        "preview": preview_doc,
        "actual": {
            "rows": int(actual_rows),
            "tokens": actual_tokens,
            "days": actual_days,
            "size_mb": actual_size_mb,
        },
        "deltas": {
            "row_error_pct": _row_error_pct(est_rows, int(actual_rows)),
            "row_delta": est_rows - int(actual_rows),
            "preview_accuracy": preview_doc.get("accuracy"),
        },
    }

    path = _calibration_path(data_dir)
    os.makedirs(os.path.dirname(path) or data_dir, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")
    return record


def load_selection_calibration(
    data_dir: str,
    *,
    limit: int = 100,
    build_kind: str | None = None,
) -> list[dict[str, Any]]:
    path = _calibration_path(data_dir)
    if not os.path.isfile(path):
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError:
                continue
            if build_kind and doc.get("build_kind") != build_kind:
                continue
            rows.append(doc)
    return rows[-limit:]


def record_build_calibration(
    data_dir: str,
    *,
    build_kind: str,
    strike_selection: dict[str, Any],
    sources: list[dict[str, Any]],
    actual_rows: int,
    market: str = "NIFTY",
    interval_sec: int | None = None,
    master_db_path: str | None = None,
    build_job_id: str | None = None,
    preview_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Compute preview (if needed) and record calibration after a build completes."""
    spec = DatasetSelectionSpec.from_strike_selection(
        strike_selection,
        selected_days=[str(s.get("trading_day") or "") for s in sources if s.get("trading_day")],
        market=market,
        interval_sec=interval_sec,
    )
    spec.master_db_path = master_db_path

    preview_result: SelectionPreviewResult | dict[str, Any] | None = preview_snapshot
    resolved_path = master_db_path
    if preview_snapshot is None and master_db_path and os.path.isfile(master_db_path):
        engine = DatasetSelectionEngine(spec, master_db_path)
        preview_result = engine.preview()
        resolved_path = master_db_path
    elif preview_snapshot is None and interval_sec is not None:
        try:
            resolved = DatasetSelectionEngine.resolve_master_db_path(data_dir, spec)
            if os.path.isfile(resolved):
                engine = DatasetSelectionEngine(spec, resolved)
                preview_result = engine.preview()
                resolved_path = resolved
        except ValueError:
            preview_result = None

    if preview_result is None:
        return None

    actual_days = len({str(s.get("trading_day") or "") for s in sources if s.get("trading_day")})
    return record_selection_calibration(
        data_dir,
        build_kind=build_kind,
        spec=spec,
        preview=preview_result,
        actual_rows=actual_rows,
        actual_days=actual_days,
        build_job_id=build_job_id,
        master_db_path=resolved_path,
    )
