"""Day-at-a-time helpers: schema lock, checkpoint/resume, progress cards."""

from __future__ import annotations

import json
import os
import re
from typing import Any


class SchemaMismatchError(ValueError):
    """Raised when a later trading day does not match Day-1 locked schema."""

    def __init__(
        self,
        *,
        day: str,
        missing: list[str] | None = None,
        extra: list[str] | None = None,
        order_mismatch: bool = False,
        type_mismatches: list[str] | None = None,
    ) -> None:
        self.day = str(day)
        self.missing = list(missing or [])
        self.extra = list(extra or [])
        self.order_mismatch = bool(order_mismatch)
        self.type_mismatches = list(type_mismatches or [])
        parts = [f"Schema mismatch on Day {self.day}"]
        if self.missing:
            parts.append("Missing:\n  " + "\n  ".join(self.missing[:40]))
        if self.extra:
            parts.append("Extra:\n  " + "\n  ".join(self.extra[:40]))
        if self.order_mismatch:
            parts.append("Column order differs from Day-1 locked schema")
        if self.type_mismatches:
            parts.append("Type mismatches:\n  " + "\n  ".join(self.type_mismatches[:40]))
        super().__init__("\n".join(parts))


_SAFE_DAY_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_day_filename(day: str) -> str:
    return _SAFE_DAY_RE.sub("_", str(day).strip()) or "day"


def serialize_schema(schema: Any) -> list[dict[str, str]]:
    return [{"name": f.name, "type": str(f.type)} for f in schema]


def schema_names(schema_doc: list[dict[str, str]]) -> list[str]:
    return [str(f.get("name") or "") for f in schema_doc]


def validate_table_against_locked_schema(
    table: Any,
    locked: list[dict[str, str]],
    *,
    day: str,
) -> None:
    """Fail hard unless names, order, and types match the Day-1 lock."""
    got_names = list(table.schema.names)
    want_names = schema_names(locked)
    missing = [n for n in want_names if n not in got_names]
    extra = [n for n in got_names if n not in want_names]
    order_mismatch = (not missing and not extra and got_names != want_names)
    type_mismatches: list[str] = []
    if not missing and not extra:
        got_types = {f.name: str(f.type) for f in table.schema}
        for field in locked:
            name = str(field.get("name") or "")
            want_t = str(field.get("type") or "")
            got_t = got_types.get(name)
            if got_t is not None and want_t and got_t != want_t:
                type_mismatches.append(f"{name}: expected {want_t}, got {got_t}")
    if missing or extra or order_mismatch or type_mismatches:
        raise SchemaMismatchError(
            day=day,
            missing=missing,
            extra=extra,
            order_mismatch=order_mismatch,
            type_mismatches=type_mismatches,
        )


def parts_dir_for(parquet_path: str) -> str:
    return f"{parquet_path}.by_day.parts"


def checkpoint_path_for(parquet_path: str) -> str:
    return f"{parquet_path}.by_day.checkpoint.json"


def source_fingerprint(parquet_path: str) -> dict[str, Any]:
    try:
        st = os.stat(parquet_path)
        return {"size": int(st.st_size), "mtime_ns": int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)))}
    except OSError:
        return {}


def load_checkpoint(parquet_path: str) -> dict[str, Any]:
    path = checkpoint_path_for(parquet_path)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_checkpoint(parquet_path: str, doc: dict[str, Any]) -> None:
    path = checkpoint_path_for(parquet_path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)


def clear_day_artifacts(parquet_path: str) -> None:
    import shutil

    for path in (checkpoint_path_for(parquet_path), f"{parquet_path}.by_day.tmp"):
        try:
            if os.path.isfile(path):
                os.remove(path)
        except OSError:
            pass
    parts = parts_dir_for(parquet_path)
    if os.path.isdir(parts):
        try:
            shutil.rmtree(parts)
        except OSError:
            pass


def build_transformation_summary(
    *,
    days: list[str],
    mode_by_day: dict[str, str],
    day_stats: dict[str, dict[str, Any]],
    created_columns: list[str],
    feature_count: int | None,
    output_columns: int | None,
    elapsed_sec: float,
    codec: str,
    warmup_mode: str = "within_day",
    resumed: bool = False,
    peak_ram_bytes: int | None = None,
) -> dict[str, Any]:
    fast_days = sum(1 for d in days if str(mode_by_day.get(d) or "").startswith("fast"))
    safe_days = sum(
        1
        for d in days
        if str(mode_by_day.get(d) or "").startswith("safe")
    )
    empty_days = sum(1 for d in days if mode_by_day.get(d) == "empty")
    total_rows = 0
    for d in days:
        stats = day_stats.get(d) or {}
        try:
            total_rows += int(stats.get("rows") or 0)
        except (TypeError, ValueError):
            pass
    return {
        "kind": "transformation_summary",
        "days": len(days),
        "rows": total_rows,
        "features": feature_count,
        "output_columns": output_columns,
        "created_columns": len(created_columns),
        "fast_days": fast_days,
        "safe_days": safe_days,
        "empty_days": empty_days,
        "elapsed_sec": round(float(elapsed_sec), 3),
        "peak_ram_bytes": peak_ram_bytes,
        "codec": codec,
        "warmup_mode": warmup_mode,
        "resumed": bool(resumed),
        "day_stats": {d: dict(day_stats.get(d) or {}) for d in days},
    }


def format_transformation_summary_text(summary: dict[str, Any] | None) -> str:
    s = dict(summary or {})
    if not s:
        return ""
    peak = s.get("peak_ram_bytes")
    if isinstance(peak, (int, float)) and peak > 0:
        peak_txt = f"{peak / (1024**3):.1f} GB"
    else:
        peak_txt = "—"
    lines = [
        "Transformation Summary",
        f"Days               {s.get('days') if s.get('days') is not None else '—'}",
        f"Rows               {s.get('rows') if s.get('rows') is not None else '—'}",
        f"Features           {s.get('features') if s.get('features') is not None else '—'}",
        f"Output Columns     {s.get('output_columns') if s.get('output_columns') is not None else '—'}",
        f"Fast Days          {s.get('fast_days') if s.get('fast_days') is not None else '—'}",
        f"Safe Days          {s.get('safe_days') if s.get('safe_days') is not None else '—'}",
        f"Elapsed            {s.get('elapsed_sec')}s" if s.get("elapsed_sec") is not None else "Elapsed            —",
        f"Peak RAM           {peak_txt}",
        f"Codec              {s.get('codec') or '—'}",
        f"Warmup             {s.get('warmup_mode') or 'within_day'}",
    ]
    if s.get("resumed"):
        lines.append("Resume             yes")
    return "\n".join(lines)
