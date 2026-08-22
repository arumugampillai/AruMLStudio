"""Resolve Model Lab SQLite directory (default D:\\data\\model_research)."""

from __future__ import annotations

import json
import os
import re

from chain_replay_ml.training.paths import safe_model_name

DEFAULT_MODEL_RESEARCH_DIR = r"D:\data\predictions\datasets"


def _load_config_model_research_dir() -> str:
    from chain_replay_ml.core.data_root import load_application_config
    doc = load_application_config()
    return str(doc.get("model_research_dir") or "").strip()


def resolve_model_research_dir() -> str:
    """Canonical predictions datasets directory: D:\\data\\predictions\\datasets."""
    env = str(
        os.environ.get("ARUMLSTUDIO_MODEL_RESEARCH_DIR")
        or os.environ.get("ARUNEO_MODEL_RESEARCH_DIR")
        or ""
    ).strip()
    if env:
        research_dir = os.path.abspath(os.path.normpath(env))
    else:
        saved = _load_config_model_research_dir()
        if saved and saved != r"D:\data\model_research":
            research_dir = os.path.abspath(os.path.normpath(saved))
        else:
            from chain_replay_ml.core.data_root import get_data_root_service
            research_dir = get_data_root_service().get_predictions_dir("datasets")
    os.makedirs(research_dir, exist_ok=True)
    return research_dir


def resolve_prediction_artifacts_dir() -> str:
    r"""Canonical predictions artifacts directory: D:\data\predictions\artifacts."""
    from chain_replay_ml.core.data_root import get_data_root_service
    return get_data_root_service().get_predictions_dir("artifacts")


def resolve_prediction_logs_dir() -> str:
    r"""Canonical application logs directory: D:\data\logs."""
    from chain_replay_ml.core.data_root import get_data_root_service
    return get_data_root_service().get_logs_dir()


def lab_db_stem(model_name: str) -> str:
    return f"model_lab_{safe_model_name(model_name)}"


def next_lab_version(model_name: str, research_dir: str | None = None) -> int:
    root = research_dir or resolve_model_research_dir()
    stem = lab_db_stem(model_name)
    max_v = 0
    if not os.path.isdir(root):
        return 1
    for name in os.listdir(root):
        m = re.match(rf"^{re.escape(stem)}_v(\d+)\.db$", name, re.I)
        if m:
            max_v = max(max_v, int(m.group(1)))
    return max_v + 1


def lab_db_filename(model_name: str, version: int) -> str:
    return f"{lab_db_stem(model_name)}_v{int(version)}.db"


def lab_db_path(model_name: str, version: int, research_dir: str | None = None) -> str:
    root = research_dir or resolve_model_research_dir()
    return os.path.join(root, lab_db_filename(model_name, version))


def list_lab_db_paths(model_name: str, research_dir: str | None = None) -> list[tuple[int, str]]:
    """Return (version, path) ascending for labs of this parent model."""
    root = research_dir or resolve_model_research_dir()
    stem = lab_db_stem(model_name)
    found: list[tuple[int, str]] = []
    if not os.path.isdir(root):
        return found
    for name in os.listdir(root):
        m = re.match(rf"^{re.escape(stem)}_v(\d+)\.db$", name, re.I)
        if not m:
            continue
        path = os.path.join(root, name)
        if os.path.isfile(path):
            found.append((int(m.group(1)), path))
    found.sort(key=lambda item: item[0])
    return found


def iter_all_lab_db_paths(research_dir: str | None = None) -> list[str]:
    """All model_lab_*.db files under the research directory (unsorted)."""
    root = research_dir or resolve_model_research_dir()
    if not os.path.isdir(root):
        return []
    out: list[str] = []
    for name in os.listdir(root):
        if not re.match(r"^model_lab_.+\.db$", name, re.I):
            continue
        path = os.path.join(root, name)
        if os.path.isfile(path):
            out.append(path)
    return out


def latest_lab_path(model_name: str, research_dir: str | None = None) -> tuple[int, str] | None:
    labs = list_lab_db_paths(model_name, research_dir=research_dir)
    return labs[-1] if labs else None
