"""Build structured feature detail payloads for the unified detail panel."""

from __future__ import annotations

import re
from typing import Any

from chain_replay_ml.dataset_builder.feature_grid_policy import build_feature_parity_spec
from chain_replay_ml.dataset_builder.schema_column_docs import RICH_COLUMN_DOCS
from chain_replay_ml.dataset_builder.schema_implementation import implementation_for_column
from chain_replay_ml.dataset_builder.schema_registry import enrich_column_view, load_schema_registry
from chain_replay_ml.feature_policy.registry import load_feature_policy_registry
from chain_replay_ml.feature_policy.warmup_calc_debug import build_formula_spec, formula_doc_for

from .source_navigation import resolve_source_location


def _input_lines(deps: list[str]) -> list[dict[str, str]]:
    skip = frozenset({"timestamp", "token", "symbol", "feature_grid"})
    out: list[dict[str, str]] = []
    for dep in deps:
        if not dep or dep in skip:
            continue
        label = dep.replace("_", " ").title()
        if dep.startswith("__roll."):
            label = dep.replace("__roll.", "").replace(".", " ").upper()
        out.append({"name": dep, "label": label, "role": "dependency"})
    return out


def _python_snippet(
    name: str,
    *,
    formula_doc: str,
    deps: list[str],
    impl: dict[str, str],
    parity: dict[str, Any],
    policy: dict[str, Any] | None,
) -> str:
    mod = impl.get("module") or "chain_replay_ml/dataset_builder/feature_plugins.py"
    fn = impl.get("function") or "compute()"
    header = f"# {name}\n# Source: {mod} → {fn}\n"

    spec = build_formula_spec(name)
    kind = spec.get("kind") or "generic"
    period_m = re.search(r"ema(\d+)", name, re.I)
    period = period_m.group(1) if period_m else "?"

    if kind == "ltp_ema_to_spot":
        return header + f"""
ltp = inputs["ltp"]                    # option last traded price
spot = inputs["spot"]                  # index spot price
ema = rolling_ema(ltp, period={period}) # session EMA; warm-up = {period} samples

if not policy_ready(f"__roll.ltp.ema{period}"):
    {name} = None
elif spot:
    {name} = ema / spot
else:
    {name} = None
""".strip()

    if kind == "ltp_ema_to_ltp":
        return header + f"""
ltp = inputs["ltp"]
ema = rolling_ema(ltp, period={period})

if not policy_ready(f"__roll.ltp.ema{period}"):
    {name} = None
elif ltp:
    {name} = ema / ltp
else:
    {name} = None
""".strip()

    if kind == "spot_ema_to_ltp":
        return header + f"""
ltp = inputs["ltp"]
spot = inputs["spot"]
spot_ema = rolling_ema(spot, period={period})

if not policy_ready(f"__roll.spot.ema{period}"):
    {name} = None
elif ltp:
    {name} = spot_ema / ltp
else:
    {name} = None
""".strip()

    if kind == "channel_width":
        return header + f"""
ltp = inputs["ltp"]
high_ratio = inputs["spot_high_ema{period}_to_ltp_ratio"]
low_ratio = inputs["spot_low_ema{period}_to_ltp_ratio"]

high = high_ratio * ltp
low = low_ratio * ltp
width = high - low

if width is None or abs(width) < 1e-6:
    {name} = None
else:
    {name} = ltp / (abs(width) + 1e-6)
""".strip()

    lag_m = re.match(r"^(.+)_lag_(\d+)(s|m)$", name)
    if lag_m:
        base, n, unit = lag_m.groups()
        sec = int(n) * (60 if unit == "m" else 1)
        return header + f"""
current = inputs["{base}"]             # value at this sample
past = inputs_lagged("{base}", {sec})  # value {n}{unit} earlier on grid

if not policy_ready("{name}"):
    {name} = None
else:
    {name} = past
""".strip()

    chg_m = re.match(r"^(.+)_change_(\d+)(s|m)$", name)
    if chg_m:
        base, n, unit = chg_m.groups()
        sec = int(n) * (60 if unit == "m" else 1)
        return header + f"""
now = inputs["{base}"]
before = inputs_lagged("{base}", {sec})

if not policy_ready("{name}"):
    {name} = None
else:
    {name} = now - before
""".strip()

    if name == "chain_pcr" or formula_doc.lower().startswith("sum(put"):
        return header + f"""
# Formula: {formula_doc}
put_oi = sum(row["oi"] for row in chain if row["option_type"] == "PE")
call_oi = sum(row["oi"] for row in chain if row["option_type"] == "CE")

{name} = put_oi / call_oi if call_oi else None
""".strip()

    if name == "roll_age_min" or "roll_anchor" in formula_doc.lower():
        return header + f"""
# Formula: {formula_doc}
sample_ts = inputs["timestamp"]
roll_anchor_ts = roll_state.anchor_ts   # last IV re-anchor for this contract

if roll_anchor_ts is None:
    {name} = None
else:
    {name} = (sample_ts - roll_anchor_ts) / 60.0
""".strip()

    if deps:
        dep_lines = "\n".join(f'{d} = inputs["{d}"]' for d in deps[:12])
        return header + f"""
# Formula: {formula_doc or name}
{dep_lines}

# Pseudocode — see implementation in {mod}
{name} = compute_{name}({", ".join(deps[:8])})
""".strip()

    return header + f"""
# Formula: {formula_doc or "—"}
# Rule: {parity.get("rule") or "—"}

{name} = compute_{name}(inputs)
""".strip()


def build_feature_detail(
    feature_name: str,
    *,
    features_by_name: dict[str, dict[str, Any]] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble formula, inputs, code, and policy for one feature."""
    name = str(feature_name or "").strip()
    if not name:
        return {"ok": False, "error": "Feature name is required"}

    schema = load_schema_registry()
    view = enrich_column_view(name, schema)
    doc = dict(RICH_COLUMN_DOCS.get(name) or {})
    catalog_row = (features_by_name or {}).get(name) or {}
    group_id = str(catalog_row.get("group") or view.get("group") or "")

    formula_doc = str(
        view.get("formula_doc")
        or doc.get("formula_doc")
        or catalog_row.get("formula")
        or formula_doc_for(name)
        or "—"
    )
    description = str(
        view.get("description")
        or doc.get("description")
        or catalog_row.get("description")
        or ""
    )
    interpretation = str(view.get("interpretation") or doc.get("interpretation") or "")
    parity = build_feature_parity_spec(name, group_id).as_dict()
    deps = list(parity.get("depends_on") or view.get("depends_on") or [])
    impl = implementation_for_column(
        name,
        formula_ref=view.get("formula_ref"),
        group_id=group_id,
        doc=doc,
    )

    policy_meta: dict[str, Any] = {}
    try:
        reg = load_feature_policy_registry(feature_names=[name])
        meta = reg.get(name)
        if meta:
            policy_meta = meta.as_dict()
    except Exception:
        policy_meta = {}

    inputs = _input_lines([str(d) for d in deps])
    if not inputs and catalog_row.get("inputs_required"):
        inputs = _input_lines([str(d) for d in catalog_row["inputs_required"]])

    python_code = _python_snippet(
        name,
        formula_doc=formula_doc,
        deps=[i["name"] for i in inputs],
        impl=impl,
        parity=parity,
        policy=policy_meta,
    )
    module_path = str(impl.get("module") or "")
    source_location = resolve_source_location(
        feature_name=name,
        module_path=module_path,
        function_ref=str(impl.get("function") or ""),
    )

    return {
        "ok": True,
        "name": name,
        "display_name": view.get("display_name") or catalog_row.get("display_name") or name,
        "group": group_id or "—",
        "category": view.get("category") or catalog_row.get("category") or "—",
        "description": description,
        "interpretation": interpretation,
        "formula_doc": formula_doc,
        "formula_ref": view.get("formula_ref") or doc.get("formula_ref") or name,
        "unit": view.get("unit") or doc.get("unit") or "—",
        "example": view.get("example") or doc.get("example") or "",
        "expected_range": view.get("expected_range") or doc.get("expected_range") or "",
        "nullable": view.get("nullable"),
        "expected_null_reason": view.get("expected_null_reason") or doc.get("expected_null_reason") or "",
        "inputs": inputs,
        "parity": parity,
        "implementation": impl,
        "source_location": source_location,
        "policy": policy_meta,
        "python_code": python_code,
        "context": dict(context or {}),
    }
