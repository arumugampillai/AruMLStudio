"""Persistent storage for ML paper-simulated trades."""
from __future__ import annotations

import json
import os
from typing import Any

from shared.config.config import ML_TRADES_FILE_NAME, TRADE_STORAGE_BASE_DIR


def ml_trades_file_path() -> str:
    return os.path.join(TRADE_STORAGE_BASE_DIR, ML_TRADES_FILE_NAME)


def load_ml_entries() -> list[dict[str, Any]]:
    path = ml_trades_file_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        print(f"❌ ERROR loading ML trades from {path}: {exc}")
        return []
    if isinstance(raw, dict):
        entries = raw.get("entries")
        if isinstance(entries, list):
            return [e for e in entries if isinstance(e, dict)]
        return []
    if isinstance(raw, list):
        return [e for e in raw if isinstance(e, dict)]
    return []


def save_ml_entries(entries: list[dict[str, Any]]) -> None:
    path = ml_trades_file_path()
    if not os.path.exists(TRADE_STORAGE_BASE_DIR):
        try:
            os.makedirs(TRADE_STORAGE_BASE_DIR)
        except OSError as exc:
            print(f"❌ ERROR: Could not create directory {TRADE_STORAGE_BASE_DIR}: {exc}")
            return
    payload = {"entries": list(entries or [])}
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, default=str)
    except Exception as exc:
        print(f"❌ ERROR saving ML trades to {path}: {exc}")
