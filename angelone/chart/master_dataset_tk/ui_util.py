"""Shared UI helpers for master dataset Tk app."""

from __future__ import annotations

import os
import subprocess
import sys


def open_path(path: str) -> None:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if sys.platform == "win32":
        os.startfile(path if os.path.isdir(path) else os.path.dirname(path))  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)


def model_builder_url(
    *,
    new: bool = False,
    dataset: str | None = None,
    model_name: str | None = None,
    mode: str | None = None,
    base_url: str = "http://127.0.0.1:8000",
) -> str:
    from urllib.parse import urlencode

    params: dict[str, str] = {}
    if new:
        params["new"] = "1"
    if dataset:
        params["dataset"] = str(dataset).strip()
    if model_name:
        params["model"] = str(model_name).strip()
    if mode:
        params["mode"] = str(mode).strip().lower()
    if model_name and mode:
        params["from"] = "model_lifecycle"
    qs = urlencode(params)
    root = f"{base_url.rstrip('/')}/ml/model-builder"
    return f"{root}?{qs}" if qs else root


def open_web_model_builder(
    *,
    new: bool = False,
    dataset: str | None = None,
    base_url: str = "http://127.0.0.1:8000",
) -> None:
    """Open web Model Builder (Create Model uses ?new=1)."""
    import webbrowser

    try:
        webbrowser.open(model_builder_url(new=new, dataset=dataset, base_url=base_url))
    except OSError:
        pass


def open_model_builder_lifecycle(model_name: str, mode: str, *, base_url: str = "http://127.0.0.1:8000") -> None:
    """Open web Model Builder with lifecycle preset (matches web Retrain tab actions)."""
    import webbrowser

    name = str(model_name or "").strip()
    if not name:
        return
    try:
        webbrowser.open(
            model_builder_url(model_name=name, mode=str(mode or "retrain").strip().lower(), base_url=base_url)
        )
    except OSError:
        pass


def open_web_model_registry(*, base_url: str = "http://127.0.0.1:8000") -> None:
    """Open web Model Registry tab (for HPO trial history, etc.)."""
    import webbrowser

    try:
        webbrowser.open(f"{base_url.rstrip('/')}/ml/create-dataset#models")
    except OSError:
        pass
