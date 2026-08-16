"""Child-process entry point for Create Dataset builds (Windows spawn-safe)."""

from __future__ import annotations

from typing import Any


def run_master_build_process(
    build_kwargs: dict[str, Any],
    progress_queue: Any,
    cancel_event: Any,
) -> None:
    """Execute one master dataset build in an isolated process."""
    import inspect
    import sys
    from pathlib import Path

    code_root = str(Path(__file__).resolve().parent.parent)
    while code_root in sys.path:
        sys.path.remove(code_root)
    sys.path.insert(0, code_root)

    try:
        from path_config import ensure_ml_studio_paths
        ensure_ml_studio_paths()
    except Exception:
        pass

    from .build_service import build_master_insert_config, run_master_build_streaming
    from .progress_adapter import enrich_master_build_payload

    # Explicitly verify build_service is loaded from AruMLStudio
    mod_file = getattr(sys.modules.get("master_dataset_tk.build_service"), "__file__", "")
    if "aruneo" in mod_file.lower():
        err_msg = f"Fatal process isolation error: build_service loaded from AruNeo path: {mod_file}"
        progress_queue.put({"status": "failed", "error": err_msg, "_done": True})
        return

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
        mod = getattr(build_master_insert_config, "__module__", "<unknown>")
        try:
            file_loc = inspect.getfile(build_master_insert_config)
            sig = str(inspect.signature(build_master_insert_config))
        except Exception:
            file_loc = "<unknown>"
            sig = "<unknown>"
        err_msg = f"{exc} [Runtime: {mod} from {file_loc} with signature {sig}]"
        result = {"status": "failed", "error": err_msg}

    progress_queue.put({**result, "_done": True})
