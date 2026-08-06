"""Persist and load ``feature_studio_status.json`` under a model package."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

STATUS_FILENAME = "feature_studio_status.json"
STAGE_ORDER: tuple[str, ...] = ("importance", "distribution", "drift")

_STATUS_GLYPH = {
    "completed": "✓",
    "running": "…",
    "pending": "·",
    "failed": "✗",
    "skipped": "–",
    "error": "✗",
}


def status_path(package_dir: str) -> str:
    return os.path.join(os.path.abspath(str(package_dir or "").strip()), STATUS_FILENAME)


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def empty_stages(*, status: str = "pending") -> dict[str, dict[str, Any]]:
    return {
        s: {"status": status, "duration_sec": 0.0, "error": None} for s in STAGE_ORDER
    }


def write_feature_studio_status(package_dir: str, payload: dict[str, Any]) -> str:
    """Atomically write status JSON. Returns path written (or empty on no-op)."""
    base = os.path.abspath(str(package_dir or "").strip())
    if not base or not os.path.isdir(base):
        return ""
    path = os.path.join(base, STATUS_FILENAME)

    body = dict(payload or {})
    body["updated_at"] = _iso_now()
    stages = body.get("stages")
    if isinstance(stages, dict):
        for key in STAGE_ORDER:
            st = stages.get(key) or {}
            if isinstance(st, dict):
                body[key] = st.get("status", body.get(key, "pending"))

    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(body, fh, indent=2, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.isfile(tmp):
                os.remove(tmp)
        except OSError:
            pass
        raise
    return path


def load_feature_studio_status(package_dir: str) -> dict[str, Any] | None:
    path = status_path(package_dir)
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def format_readiness_line(status: dict[str, Any] | None) -> str:
    """Short banner line, e.g. ``✓ Importance · ✓ Distribution · … Drift``."""
    if not status:
        return "Feature Studio: no auto-run status yet"
    overall = str(status.get("status") or "").strip().lower()
    parts: list[str] = []
    stages = status.get("stages") if isinstance(status.get("stages"), dict) else {}
    for key in STAGE_ORDER:
        label = key.capitalize()
        st = ""
        if isinstance(stages.get(key), dict):
            st = str(stages[key].get("status") or "").strip().lower()
        if not st:
            st = str(status.get(key) or "pending").strip().lower()
        glyph = _STATUS_GLYPH.get(st, "·")
        parts.append(f"{glyph} {label}")
    joined = " · ".join(parts)
    if overall and overall not in ("completed",):
        return f"Feature Studio ({overall}): {joined}"
    return f"Feature Studio: {joined}"
