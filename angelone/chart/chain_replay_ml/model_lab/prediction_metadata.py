"""Prediction Dataset UI metadata sidecar (not build truth).

``{lab_stem}.prediction_metadata.json`` next to the lab DB is the Trading Days
grid source. ``prediction.db`` remains the source of build/compute data.
Refresh Days may rebuild this file from the DB; normal startup must not.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def prediction_metadata_path(lab_db_path: str) -> str:
    base, _ = os.path.splitext(os.path.abspath(lab_db_path))
    return f"{base}.prediction_metadata.json"


def empty_prediction_metadata() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": None,
        "days": {},
    }


def _parse_iso(raw: Any) -> datetime | None:
    if not raw:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _build_time_sec(started_at: Any, finished_at: Any) -> float | None:
    a = _parse_iso(started_at)
    b = _parse_iso(finished_at)
    if a is None or b is None:
        return None
    if a.tzinfo is None:
        a = a.replace(tzinfo=timezone.utc)
    if b.tzinfo is None:
        b = b.replace(tzinfo=timezone.utc)
    sec = (b - a).total_seconds()
    return round(sec, 1) if sec >= 0 else None


def default_day_entry(
    *,
    status: str = "waiting",
    dataset_rows: int | None = None,
    prediction_rows: int = 0,
    build_time_sec: float | None = None,
    completed_at: str | None = None,
    started_at: str | None = None,
    note: str = "",
    dataset_type: str | None = None,
) -> dict[str, Any]:
    return {
        "status": str(status or "waiting"),
        "dataset_rows": int(dataset_rows) if dataset_rows is not None else None,
        "prediction_rows": int(prediction_rows or 0),
        "build_time_sec": build_time_sec,
        "completed_at": completed_at,
        "started_at": started_at,
        "note": str(note or ""),
        "dataset_type": dataset_type,
    }


def read_prediction_metadata(lab_db_path: str) -> dict[str, Any]:
    """Read sidecar JSON. Missing/corrupt → empty doc (no DB fallback)."""
    path = prediction_metadata_path(lab_db_path)
    if not os.path.isfile(path):
        return empty_prediction_metadata()
    try:
        with open(path, encoding="utf-8") as fh:
            loaded = json.load(fh)
        if not isinstance(loaded, dict):
            return empty_prediction_metadata()
        days = loaded.get("days")
        if not isinstance(days, dict):
            loaded["days"] = {}
        loaded.setdefault("schema_version", SCHEMA_VERSION)
        return loaded
    except (OSError, json.JSONDecodeError, TypeError):
        return empty_prediction_metadata()


def write_prediction_metadata(lab_db_path: str, doc: dict[str, Any]) -> str:
    """Atomic write (temp file in same dir, then replace)."""
    path = prediction_metadata_path(lab_db_path)
    payload = dict(doc)
    payload["schema_version"] = int(payload.get("schema_version") or SCHEMA_VERSION)
    if not isinstance(payload.get("days"), dict):
        payload["days"] = {}
    payload["updated_at"] = _utc_now()
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pred_meta_", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise
    return path


class _MetaFileLock:
    """Best-effort exclusive lock for read-modify-write of the sidecar."""

    def __init__(self, lab_db_path: str) -> None:
        self._lock_path = prediction_metadata_path(lab_db_path) + ".lock"
        self._fh: Any = None

    def __enter__(self) -> _MetaFileLock:
        parent = os.path.dirname(self._lock_path) or "."
        os.makedirs(parent, exist_ok=True)
        self._fh = open(self._lock_path, "a+", encoding="utf-8")
        deadline = time.time() + 15.0
        while True:
            try:
                if os.name == "nt":
                    import msvcrt

                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.time() >= deadline:
                    break
                time.sleep(0.05)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._fh is None:
            return
        try:
            if os.name == "nt":
                import msvcrt

                self._fh.seek(0)
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            self._fh.close()
        except OSError:
            pass
        self._fh = None


def upsert_prediction_day_metadata(
    lab_db_path: str,
    trading_day: str,
    **fields: Any,
) -> dict[str, Any]:
    """Merge fields into one day entry and write atomically (locked)."""
    day = str(trading_day or "").strip()
    if not day or not lab_db_path:
        return empty_prediction_metadata()
    with _MetaFileLock(lab_db_path):
        doc = read_prediction_metadata(lab_db_path)
        days = doc.setdefault("days", {})
        if not isinstance(days, dict):
            days = {}
            doc["days"] = days
        cur = dict(days.get(day) or default_day_entry())
        for key, val in fields.items():
            if val is None and key not in (
                "completed_at",
                "started_at",
                "build_time_sec",
                "dataset_rows",
                "dataset_type",
                "note",
            ):
                continue
            if key == "prediction_rows" and val is not None:
                cur["prediction_rows"] = int(val)
            elif key == "dataset_rows" and val is not None:
                cur["dataset_rows"] = int(val)
            elif key == "status" and val is not None:
                cur["status"] = str(val)
            elif key == "note":
                cur["note"] = str(val or "")
            elif key == "build_time_sec":
                cur["build_time_sec"] = val
            elif key == "completed_at":
                cur["completed_at"] = val
            elif key == "started_at":
                cur["started_at"] = val
            elif key == "dataset_type":
                cur["dataset_type"] = val
        st = str(cur.get("status") or "waiting")
        if st == "running" and not cur.get("started_at"):
            cur["started_at"] = _utc_now()
        # "partial" is a terminal (day finished building) state too — just
        # without full coverage — so it gets a completed_at/build duration
        # exactly like "completed" does.
        if st in ("completed", "partial"):
            if not cur.get("completed_at"):
                cur["completed_at"] = _utc_now()
            if cur.get("build_time_sec") is None:
                bt = _build_time_sec(cur.get("started_at"), cur.get("completed_at"))
                if bt is not None:
                    cur["build_time_sec"] = bt
        if st in ("failed", "cancelled", "waiting"):
            if st != "waiting":
                cur["completed_at"] = None
        days[day] = cur
        write_prediction_metadata(lab_db_path, doc)
        return doc


def clear_prediction_metadata_days(lab_db_path: str) -> str:
    doc = empty_prediction_metadata()
    return write_prediction_metadata(lab_db_path, doc)


def day_entry_from_store_row(row: dict[str, Any]) -> dict[str, Any]:
    """Map ``list_build_days`` row → sidecar day entry."""
    started = row.get("started_at")
    finished = row.get("finished_at")
    status = str(row.get("status") or "waiting")
    note = str(row.get("error_message") or "")
    pred_n = int(row.get("row_count") or 0)
    ds_n = row.get("rows_expected")
    ds_n_i = int(ds_n) if ds_n is not None else None
    if not note:
        if ds_n_i and not pred_n:
            note = "pred missing — build needed"
        elif pred_n and ds_n_i and pred_n < ds_n_i:
            note = f"partial vs dataset ({pred_n:,}/{ds_n_i:,})"
        elif status == "completed" or (pred_n and ds_n_i and pred_n >= ds_n_i):
            note = "complete"
    return default_day_entry(
        status=status,
        dataset_rows=ds_n_i,
        prediction_rows=pred_n,
        build_time_sec=_build_time_sec(started, finished),
        completed_at=str(finished) if finished else None,
        started_at=str(started) if started else None,
        note=note,
        dataset_type=row.get("dataset_type"),
    )


def rebuild_prediction_metadata_from_db(lab_db_path: str) -> dict[str, Any]:
    """Refresh Days only: rebuild sidecar from lab day catalog + pred counts."""
    from .store import ModelLabStore

    if not lab_db_path or not os.path.isfile(lab_db_path):
        doc = empty_prediction_metadata()
        if lab_db_path:
            write_prediction_metadata(lab_db_path, doc)
        return doc

    with ModelLabStore(lab_db_path) as store:
        store.ensure_prediction_schema()
        lab = store.read_info()
        if lab is None:
            doc = empty_prediction_metadata()
            write_prediction_metadata(lab_db_path, doc)
            return doc
        rows = store.list_build_days(lab.lab_uuid)
        counts = store.prediction_row_counts_by_day()

    days_out: dict[str, Any] = {}
    for row in rows:
        day = str(row.get("trading_day") or "").strip()
        if not day:
            continue
        entry = day_entry_from_store_row(row)
        if day in counts:
            entry["prediction_rows"] = int(counts[day])
            if entry["prediction_rows"] > 0 and entry["status"] in (
                "waiting",
                "skipped",
                "",
            ):
                ds_n_i = entry.get("dataset_rows")
                pred_n_i = int(entry["prediction_rows"])
                if ds_n_i and pred_n_i < int(ds_n_i):
                    entry["status"] = "partial"
                    if not entry.get("note"):
                        entry["note"] = f"partial vs dataset ({pred_n_i:,}/{int(ds_n_i):,})"
                else:
                    entry["status"] = "completed"
                    if not entry.get("note"):
                        entry["note"] = "complete"
        days_out[day] = entry

    doc = empty_prediction_metadata()
    doc["days"] = days_out
    with _MetaFileLock(lab_db_path):
        write_prediction_metadata(lab_db_path, doc)
    return doc


def normalize_stale_running_days(doc: dict[str, Any]) -> dict[str, Any]:
    """On UI open: stuck ``running`` → cancelled so the grid never hangs."""
    days = doc.get("days")
    if not isinstance(days, dict):
        return doc
    changed = False
    for day, raw in list(days.items()):
        if not isinstance(raw, dict):
            continue
        if str(raw.get("status") or "") != "running":
            continue
        entry = dict(raw)
        entry["status"] = "cancelled"
        if not entry.get("note"):
            entry["note"] = "interrupted — was still running"
        days[day] = entry
        changed = True
    if changed:
        doc = dict(doc)
        doc["days"] = days
    return doc


def merge_master_and_metadata(
    master_rows: dict[str, int],
    meta_doc: dict[str, Any],
) -> list[dict[str, Any]]:
    """Merge Master trading days with sidecar UI state (no DB)."""
    meta_doc = normalize_stale_running_days(meta_doc)
    meta_days = meta_doc.get("days") if isinstance(meta_doc.get("days"), dict) else {}
    all_days = sorted(
        {
            str(d).strip()
            for d in list(master_rows.keys()) + list(meta_days.keys())
            if str(d or "").strip()
        }
    )
    out: list[dict[str, Any]] = []
    for day in all_days:
        master_n = int(master_rows.get(day) or 0)
        raw = meta_days.get(day) if isinstance(meta_days.get(day), dict) else None
        if raw:
            status = str(raw.get("status") or "waiting")
            pred_n = int(raw.get("prediction_rows") or 0)
            ds_n = raw.get("dataset_rows")
            if ds_n is None:
                ds_n = master_n if master_n > 0 else None
            else:
                ds_n = int(ds_n)
            note = str(raw.get("note") or "")
            started = raw.get("started_at")
            finished = raw.get("completed_at")
            build_sec = raw.get("build_time_sec")
            out.append(
                {
                    "trading_day": day,
                    "status": status,
                    "row_count": pred_n,
                    "rows_expected": ds_n,
                    "progress_pct": 100.0 if status in ("completed", "partial") else None,
                    "error_message": note if status == "failed" else None,
                    "started_at": started,
                    "finished_at": finished,
                    "selected": True,
                    "updated_at": meta_doc.get("updated_at"),
                    "dataset_type": raw.get("dataset_type"),
                    "note": note,
                    "build_time_sec": build_sec,
                    "ui_meta_ready": True,
                }
            )
        else:
            out.append(
                {
                    "trading_day": day,
                    "status": "waiting",
                    "row_count": 0,
                    "rows_expected": master_n if master_n > 0 else None,
                    "progress_pct": None,
                    "error_message": None,
                    "started_at": None,
                    "finished_at": None,
                    "selected": False,
                    "updated_at": None,
                    "dataset_type": None,
                    "note": "pred missing — build needed" if master_n else "",
                    "build_time_sec": None,
                    "ui_meta_ready": True,
                }
            )
    return out


def sync_day_metadata_from_status(
    lab_db_path: str,
    trading_day: str,
    *,
    status: str,
    row_count: int | None = None,
    rows_expected: int | None = None,
    error_message: str | None = None,
    started: bool = False,
    finished: bool = False,
    dataset_type: str | None = None,
) -> None:
    """Called after ``set_build_day_status`` — keep UI sidecar in sync."""
    fields: dict[str, Any] = {"status": status}
    if row_count is not None:
        fields["prediction_rows"] = int(row_count)
    if rows_expected is not None:
        fields["dataset_rows"] = int(rows_expected)
    if error_message is not None:
        fields["note"] = error_message
    if started:
        fields["started_at"] = _utc_now()
    if finished:
        fields["completed_at"] = _utc_now() if status in ("completed", "partial") else None
        if status in ("completed", "partial") and "started_at" not in fields:
            # build_time filled inside upsert when both timestamps exist
            pass
    if dataset_type is not None:
        fields["dataset_type"] = dataset_type
    try:
        upsert_prediction_day_metadata(lab_db_path, trading_day, **fields)
    except OSError:
        pass
