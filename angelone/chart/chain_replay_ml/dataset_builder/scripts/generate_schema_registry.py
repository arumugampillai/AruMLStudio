#!/usr/bin/env python3
"""Full-rebuild ``ml_schema_registry.json`` from ``_REGISTRY_FEATURES``.

``ml_schema_registry.json`` is a **generated artifact**. Do not edit it by hand.

Source of truth:
    feature_plugins._REGISTRY_FEATURES
        → rebuild_schema_registry_from_plugins()
        → angelone/chart/static/ml_schema_registry.json
        → Feature Registry UI

This script always replaces the previous schema (never an additive merge).
"""

from __future__ import annotations

import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_SCRIPT_DIR)
_CHART_ROOT = os.path.dirname(os.path.dirname(_PKG_DIR))  # angelone/chart

if _CHART_ROOT not in sys.path:
    sys.path.insert(0, _CHART_ROOT)

from chain_replay_ml.dataset_builder.feature_ownership import is_interaction_feature
from chain_replay_ml.dataset_builder.feature_plugins import _REGISTRY_FEATURES
from chain_replay_ml.dataset_builder.schema_registry import (
    rebuild_schema_registry_from_plugins,
    schema_feature_column_names,
    validate_schema_plugin_parity,
    write_schema_registry,
)

STATIC = os.path.join(_CHART_ROOT, "static")
FEATURE_REG = os.path.join(STATIC, "ml_feature_registry.json")
OUT = os.path.join(STATIC, "ml_schema_registry.json")


def main() -> None:
    legacy_meta: dict = {}
    if os.path.isfile(FEATURE_REG):
        with open(FEATURE_REG, encoding="utf-8") as fh:
            legacy = json.load(fh)
        # Only structural group metadata — feature lists are ignored (full rebuild).
        legacy_meta = {
            "hardMandatory": legacy.get("hardMandatory") or ["price"],
            "dependencies": legacy.get("dependencies") or {},
            "profiles": legacy.get("profiles") or {},
        }

    interaction_in_plugins = sorted(
        f
        for feats in _REGISTRY_FEATURES.values()
        for f in feats
        if is_interaction_feature(f)
    )
    if interaction_in_plugins:
        raise SystemExit(
            "FAIL: InteractionTransformation products must never be regenerated "
            "into the Feature Registry (semantic admission rule — evidence via "
            "auto-name / generator=interaction):\n  "
            + "\n  ".join(interaction_in_plugins)
        )

    schema = rebuild_schema_registry_from_plugins(legacy_meta=legacy_meta)
    validate_schema_plugin_parity(schema, raise_on_error=True)
    write_schema_registry(schema, OUT)

    plugin_n = sum(len(v) for v in _REGISTRY_FEATURES.values())
    feature_n = len(schema_feature_column_names(schema))
    meta_n = sum(
        1
        for c in (schema.get("columns") or {}).values()
        if str(c.get("type") or "").lower() == "metadata"
    )
    target_n = sum(
        1
        for c in (schema.get("columns") or {}).values()
        if str(c.get("type") or "").lower() == "target"
    )
    print(f"Wrote {OUT}")
    print(f"Canonical plugins: {plugin_n}")
    print(f"Schema features:   {feature_n}")
    print(f"Metadata: {meta_n}, targets: {target_n}, total columns: {len(schema.get('columns') or {})}")
    print(f"generated_from={schema.get('generated_from')}")
    if plugin_n != feature_n:
        raise SystemExit(f"FAIL: plugin/schema feature count mismatch {plugin_n} != {feature_n}")


if __name__ == "__main__":
    main()
