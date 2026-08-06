"""Production contracts for OLE artifacts, validity, and session defaults."""

from __future__ import annotations

from typing import Any

from .types import LabelRunMeta

REQUIRED_RUN_META_FIELDS = (
    "strategy",
    "version",
    "engine_version",
    "source",
    "params",
    "rows",
    "compute_time_sec",
    "supported_problem_types",
    "target_columns",
    "target_definitions",
)

FORBIDDEN_SENTINEL_LABEL_IDS = frozenset({-1})


class OleContractError(ValueError):
    """Raised when a production OLE contract is violated."""


def assert_run_meta_complete(meta: LabelRunMeta | dict[str, Any]) -> dict[str, Any]:
    """Validate audit fields required for long-term reproducibility (§5.4)."""
    data = meta.to_dict() if isinstance(meta, LabelRunMeta) else dict(meta)
    missing = [k for k in REQUIRED_RUN_META_FIELDS if k not in data or data[k] is None]
    if missing:
        raise OleContractError(f"run_meta missing required fields: {missing}")
    if not str(data.get("strategy") or "").strip():
        raise OleContractError("run_meta.strategy must be non-empty")
    if not str(data.get("version") or "").strip():
        raise OleContractError("run_meta.version must be non-empty")
    if not str(data.get("engine_version") or "").strip():
        raise OleContractError("run_meta.engine_version must be non-empty")
    defs = data.get("target_definitions") or {}
    if not isinstance(defs, dict) or not defs.get("primary_target"):
        raise OleContractError("run_meta.target_definitions.primary_target is required")
    # Encoding optional for regression FH; when present must not include sentinels.
    enc = defs.get("label_encoding")
    if isinstance(enc, dict):
        bad = FORBIDDEN_SENTINEL_LABEL_IDS.intersection(enc.values())
        if bad:
            raise OleContractError(
                f"label_encoding must not include sentinel ids {sorted(bad)}"
            )
    return data


def assert_truncate_at_close_default(schema: dict[str, Any]) -> None:
    """Triple Barrier (and similar) must default truncate_at_close=True (§5.2)."""
    if "truncate_at_close" not in schema:
        raise OleContractError("schema missing truncate_at_close")
    spec = schema["truncate_at_close"]
    if not isinstance(spec, dict) or spec.get("default") is not True:
        raise OleContractError("truncate_at_close default must be True")


def sanitize_label_row(row: dict[str, Any]) -> dict[str, Any]:
    """Enforce validity contract: no sentinel class ids; clear labels when invalid (§5.3)."""
    out = dict(row)
    lid = out.get("label_id")
    if lid in FORBIDDEN_SENTINEL_LABEL_IDS:
        raise OleContractError(
            "sentinel label_id=-1 is forbidden; use is_valid + invalid_reason"
        )
    if out.get("is_valid") is False:
        out["label_id"] = None
        out["label_name"] = None
        if not out.get("invalid_reason"):
            out["invalid_reason"] = "unspecified"
    return out
