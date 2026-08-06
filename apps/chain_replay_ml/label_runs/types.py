"""Label Run metadata contract (Phase X)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Prefer master_row_id; fall back to composite identity used by Create Model.
JOIN_KEY_PREFERENCE: tuple[str, ...] = (
    "master_row_id",
    "sample_id",
    "trading_day",
    "timestamp",
    "token",
)

_LABEL_ONLY_ALLOWED = frozenset(
    {
        "master_row_id",
        "sample_id",
        "trading_day",
        "timestamp",
        "token",
        "label_id",
        "label_name",
        "entry_price",
        "exit_price",
        "exit_reason",
        "holding_seconds",
        "is_valid",
        "invalid_reason",
        "realized_return",
        # Fixed-horizon promote may keep the original target column name.
        "primary_target_value",
    }
)


@dataclass
class LabelRunRecord:
    """Registry row + on-disk identity for one Label Run."""

    run_id: str
    strategy: str
    strategy_version: str
    engine_version: str
    dataset_id: str
    dataset_hash: str | None
    created_at: str
    rows: int
    valid_rows: int
    invalid_rows: int
    parameters: dict[str, Any] = field(default_factory=dict)
    primary_target: str = "label_id"
    display_target: str | None = None
    label_encoding: dict[str, int] | None = None
    join_keys: list[str] = field(default_factory=list)
    status: str = "ready"
    parquet_path: str = ""
    meta_path: str = ""
    exists: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_line(self) -> str:
        params = self.parameters or {}
        bits: list[str] = [str(self.strategy)]
        if params.get("tp_value") is not None and params.get("sl_value") is not None:
            unit = "%" if str(params.get("barrier_type") or "") == "percentage" else ""
            bits.append(f"TP={params['tp_value']}{unit} SL={params['sl_value']}{unit}")
        if params.get("holding_seconds") is not None:
            bits.append(f"hold={params['holding_seconds']}s")
        if self.primary_target and self.primary_target not in ("label_id",):
            bits.append(self.primary_target)
        return f"{self.run_id}  ·  " + "  ·  ".join(bits)


def label_run_meta_template(
    *,
    run_id: str,
    strategy: str,
    strategy_version: str,
    engine_version: str,
    dataset_id: str,
    dataset_hash: str | None,
    parameters: dict[str, Any],
    rows: int,
    valid_rows: int,
    invalid_rows: int,
    primary_target: str,
    display_target: str | None = None,
    label_encoding: dict[str, int] | None = None,
    join_keys: list[str] | None = None,
    created_at: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from datetime import datetime, timezone

    doc: dict[str, Any] = {
        "run_id": run_id,
        "strategy": strategy,
        "strategy_version": strategy_version,
        "engine_version": engine_version,
        "dataset_id": dataset_id,
        "dataset_hash": dataset_hash,
        "parameters": dict(parameters or {}),
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
        "rows": int(rows),
        "valid_rows": int(valid_rows),
        "invalid_rows": int(invalid_rows),
        "primary_target": primary_target,
        "display_target": display_target,
        "label_encoding": dict(label_encoding) if label_encoding else None,
        "join_keys": list(join_keys or []),
        "artifact_kind": "label_run",
        "schema_version": 1,
    }
    if extra:
        doc.update(extra)
    return doc


def assert_label_only_columns(columns: list[str], *, primary_target: str) -> None:
    """Refuse feature-column smuggling into Label Runs."""
    allowed = set(_LABEL_ONLY_ALLOWED) | {str(primary_target)}
    bad = [c for c in columns if c not in allowed and not str(c).startswith("_")]
    # Identity + label contract columns only; primary_target always allowed.
    # Soft allow: any column listed as target/meta that is not an obvious feature dump.
    # Hard fail if hundreds of columns (feature duplication).
    if len(columns) > 40:
        raise ValueError(
            f"Label Run has {len(columns)} columns — looks like a feature dump. "
            "Store labels only (identity + label fields)."
        )
    # Allow unknown small set of label enrichments, but block common feature prefixes.
    blocked_prefixes = ("feat_", "f_", "spot_", "iv_", "oi_", "delta_", "gamma_", "theta_", "vega_")
    smuggled = [
        c
        for c in columns
        if c not in allowed and any(str(c).lower().startswith(p) for p in blocked_prefixes)
    ]
    if smuggled:
        raise ValueError(f"Label Run must not include feature columns: {smuggled[:8]}")
