"""Child-process entry point for Create Dataset builds (Windows spawn-safe)."""

from __future__ import annotations

from typing import Any


def run_master_build_process(
    build_kwargs: dict[str, Any],
    progress_queue: Any,
    cancel_event: Any,
) -> None:
    """Execute one master dataset build in an isolated process."""
    import sys
    from pathlib import Path

    code_root = str(Path(__file__).resolve().parent.parent)
    if code_root not in sys.path:
        sys.path.insert(0, code_root)

    from .build_service import build_master_insert_config, run_master_build_streaming
    from .progress_adapter import enrich_master_build_payload

    def on_progress(payload: dict[str, Any]) -> None:
        progress_queue.put(enrich_master_build_payload(payload))

    result: dict[str, Any]
    try:
        config = build_master_insert_config(**build_kwargs)
        result = run_master_build_streaming(
            config,
            cancel_event=cancel_event,
            on_progress=on_progress,
        )
    except Exception as exc:
        result = {"status": "failed", "error": str(exc)}

    progress_queue.put({**result, "_done": True})
