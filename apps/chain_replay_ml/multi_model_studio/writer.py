"""Write Multi-model Feature Studio pair artifacts."""

from __future__ import annotations

import json
import os
from typing import Any

from chain_replay_ml.training.paths import safe_model_name


ARTIFACT_ROOT = "multi_model_studio"


def pair_dirname(model_a: str, model_b: str) -> str:
    a, b = sorted([safe_model_name(model_a), safe_model_name(model_b)])
    return f"{a}__vs__{b}"


def studio_pair_dir(data_dir: str, model_a: str, model_b: str) -> str:
    path = os.path.join(data_dir, ARTIFACT_ROOT, pair_dirname(model_a, model_b))
    os.makedirs(path, exist_ok=True)
    return path


def write_studio_artifacts(
    data_dir: str,
    model_a: str,
    model_b: str,
    *,
    comparison: list[dict[str, Any]],
    run_meta: dict[str, Any],
) -> str:
    out = studio_pair_dir(data_dir, model_a, model_b)
    payloads = {
        "comparison.json": {"rows": comparison},
        "run_meta.json": run_meta,
    }
    for name, doc in payloads.items():
        with open(os.path.join(out, name), "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, default=str)
    return out


def load_studio_artifacts(
    data_dir: str, model_a: str, model_b: str
) -> dict[str, Any] | None:
    out = os.path.join(data_dir, ARTIFACT_ROOT, pair_dirname(model_a, model_b))
    comparison_path = os.path.join(out, "comparison.json")
    meta_path = os.path.join(out, "run_meta.json")
    if not os.path.isfile(comparison_path):
        return None
    with open(comparison_path, encoding="utf-8") as fh:
        comparison_doc = json.load(fh)
    meta: dict[str, Any] = {}
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as fh:
            loaded = json.load(fh)
            if isinstance(loaded, dict):
                meta = loaded
    rows = comparison_doc.get("rows") if isinstance(comparison_doc, dict) else None
    if not isinstance(rows, list):
        return None
    return {"artifacts_dir": out, "comparison": rows, "meta": meta}
