"""Interaction Builder UI helpers — groups, bulk pairs, lineage, preview ledger."""

from __future__ import annotations

from typing import Any

from .difference import difference_column_name
from .interaction import (
    interaction_column_name,
    normalize_interaction_op,
    normalize_interaction_pair,
)
from .lag import lag_column_name
from .return_transform import return_column_name

SOURCE_ORDER: tuple[str, ...] = (
    "Base",
    "Computed Base",
    "Lag",
    "Difference",
    "Return",
    "Rolling",
    "Exponential Rolling",
    "OHLC Aggregation",
    "Interaction",
    "Other",
)

OP_DISPLAY_LABELS: dict[str, str] = {
    "multiply": "Multiply",
    "divide": "Divide",
    "add": "Add",
    "subtract": "Subtract",
    "min": "Min",
    "max": "Max",
    "absolute_difference": "Absolute Difference",
}

OP_CHOICES: tuple[str, ...] = tuple(OP_DISPLAY_LABELS.keys())

BULK_CONFIRM_THRESHOLD = 50


def display_op_label(op: str) -> str:
    try:
        return OP_DISPLAY_LABELS[normalize_interaction_op(op)]
    except Exception:
        return str(op or "Multiply")


def classify_feature_source(
    name: str,
    *,
    master_features: set[str] | None = None,
    interaction_outputs: set[str] | None = None,
) -> str:
    """Best-effort source group for a column name."""
    n = str(name or "").strip()
    if not n:
        return "Other"
    if interaction_outputs and n in interaction_outputs:
        return "Interaction"
    if "_lag_" in n:
        return "Lag"
    if "_diff_" in n or "_change_" in n:
        return "Difference"
    if "_return_" in n:
        return "Return"
    # OHLC Aggregation: ``{feature}_{3m|5m|15m}_{index}_{open|high|low|close}``
    if (
        n.endswith(("_open", "_high", "_low", "_close"))
        and any(tok in n for tok in ("_3m_", "_5m_", "_15m_"))
    ):
        return "OHLC Aggregation"
    if "_roll_" in n:
        return "Rolling"
    # Transform EWM/EMA use ``_ewm_*_`` / ``_ema_<period>``; controller EMAs are ``*_ema20``.
    if "_ewm_" in n or "_ema_" in n:
        return "Exponential Rolling"
    if any(
        tok in n
        for tok in (
            "_zscore_",
            "_mean_",
            "_std_",
            "_min_",
            "_max_",
            "_body_pct_",
            "_range_pct_",
            "_range_pos_",
            "_dist_high_",
            "_dist_low_",
        )
    ):
        return "Rolling"
    if master_features is not None and n in master_features:
        try:
            from chain_replay_ml.dataset_builder.feature_ownership import (
                OWNERSHIP_COMPUTED_BASE,
                ownership_of,
            )

            if ownership_of(n) == OWNERSHIP_COMPUTED_BASE:
                return "Computed Base"
            return "Base"
        except Exception:
            return "Base"
    # Heuristic without master set
    try:
        from chain_replay_ml.dataset_builder.feature_ownership import (
            OWNERSHIP_COMPUTED_BASE,
            ownership_of,
        )

        if ownership_of(n) == OWNERSHIP_COMPUTED_BASE:
            return "Computed Base"
    except Exception:
        pass
    return "Other"


def group_features_by_source(
    features: list[str],
    *,
    master_features: set[str] | None = None,
    interaction_outputs: set[str] | None = None,
) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {s: [] for s in SOURCE_ORDER}
    for name in features:
        src = classify_feature_source(
            name,
            master_features=master_features,
            interaction_outputs=interaction_outputs,
        )
        if src not in buckets:
            src = "Other"
        buckets[src].append(str(name))
    return {k: sorted(v) for k, v in buckets.items() if v}


def planned_time_shift_outputs(
    *,
    features: list[str],
    lag_seconds: list[int],
    lag_enabled: bool = False,
    difference_enabled: bool = False,
    return_enabled: bool = False,
) -> dict[str, list[str]]:
    """Planned column names from Lag / Difference / Return UI settings."""
    feats = [str(f).strip() for f in features if str(f).strip()]
    lags = sorted({int(s) for s in lag_seconds if int(s) > 0})
    out: dict[str, list[str]] = {"Lag": [], "Difference": [], "Return": []}
    for feat in feats:
        for sec in lags:
            if lag_enabled:
                out["Lag"].append(lag_column_name(feat, sec))
            if difference_enabled:
                out["Difference"].append(difference_column_name(feat, sec))
            if return_enabled:
                out["Return"].append(return_column_name(feat, sec))
    return out


def describe_interaction_catalog(
    config: dict[str, Any] | None,
    *,
    master_features: list[str],
    sample_interval_sec: float | int | None = None,
):
    """Pipeline description for Interaction Builder (shared catalog contract)."""
    from .describe import describe_pipeline_stages

    return describe_pipeline_stages(
        config,
        master_features=list(master_features or []),
        sample_interval_sec=sample_interval_sec,
        include_disabled=True,
    )


def interaction_source_choices(
    config: dict[str, Any] | None,
    *,
    master_features: list[str],
    sample_interval_sec: float | int | None = None,
) -> list[tuple[str, str]]:
    """Source dropdown choices: Master + earlier enabled stages + Interaction pairs."""
    from .describe import MASTER_STAGE_ID

    desc = describe_interaction_catalog(
        config,
        master_features=master_features,
        sample_interval_sec=sample_interval_sec,
    )
    choices = desc.source_choices(before_stage_id="interaction", require_outputs=True)
    # Always keep Master even if empty (fresh panel).
    if not any(sid == MASTER_STAGE_ID for sid, _ in choices):
        choices.insert(0, (MASTER_STAGE_ID, "Master Features"))
    ix = desc.stage("interaction")
    if ix is not None and ix.output_names:
        if not any(sid == "interaction" for sid, _ in choices):
            choices.append(("interaction", "Interaction Outputs"))
    return choices


def columns_for_interaction_source(
    config: dict[str, Any] | None,
    source_id: str,
    *,
    master_features: list[str],
    sample_interval_sec: float | int | None = None,
) -> list[str]:
    """Columns belonging to one Interaction source stage."""
    from .describe import MASTER_STAGE_ID

    desc = describe_interaction_catalog(
        config,
        master_features=master_features,
        sample_interval_sec=sample_interval_sec,
    )
    sid = str(source_id or MASTER_STAGE_ID).strip() or MASTER_STAGE_ID
    st = desc.stage(sid)
    if st is None:
        return []
    if sid == MASTER_STAGE_ID:
        return list(st.output_names)
    if not st.enabled and sid != "interaction":
        return []
    return list(st.output_names)


def available_interaction_features_from_config(
    config: dict[str, Any] | None,
    *,
    master_features: list[str],
    sample_interval_sec: float | int | None = None,
) -> list[str]:
    """Universe of columns Interaction may pick (Master + earlier stages + pairs)."""
    desc = describe_interaction_catalog(
        config,
        master_features=master_features,
        sample_interval_sec=sample_interval_sec,
    )
    cols = desc.available_before("interaction", enabled_only=True)
    ix = desc.stage("interaction")
    if ix is not None and ix.enabled:
        cols = list(dict.fromkeys([*cols, *ix.output_names]))
    elif ix is not None and ix.output_names:
        # Pairs may be staged while Interaction toggle is off — still list for editing.
        cols = list(dict.fromkeys([*cols, *ix.output_names]))
    return cols


def available_interaction_features(
    *,
    master_features: list[str],
    lag_features: list[str] | None = None,
    lag_seconds: list[int] | None = None,
    lag_enabled: bool = False,
    difference_enabled: bool = False,
    return_enabled: bool = False,
    interaction_pairs: list[dict[str, Any]] | None = None,
    ohlc_enabled: bool = False,
    ohlc_features: list[str] | None = None,
    ohlc_timeframes: list[str] | None = None,
    ohlc_outputs: list[str] | None = None,
    sample_interval_sec: float | int | None = None,
    transformation_config: dict[str, Any] | None = None,
    rolling_enabled: bool = False,
    rolling_features: list[str] | None = None,
    rolling_windows: list[int] | None = None,
    rolling_operations: list[str] | None = None,
    exponential_rolling_enabled: bool = False,
    exponential_features: list[str] | None = None,
    exponential_periods: list[int] | None = None,
    exponential_operations: list[str] | None = None,
) -> list[str]:
    """Universe of columns the Interaction Builder may pick from.

    Prefer ``transformation_config`` (full pipeline). Keyword args remain for
    tests / callers that assemble settings piecemeal.
    """
    if transformation_config is not None:
        return available_interaction_features_from_config(
            transformation_config,
            master_features=master_features,
            sample_interval_sec=sample_interval_sec,
        )

    from .exponential_rolling_ui import merge_exponential_rolling_into_config
    from .lag_ui import build_lag_transformation_config
    from .ohlc_aggregation_ui import merge_ohlc_aggregation_into_config
    from .rolling_ui import merge_rolling_into_config

    feats = list(lag_features or [])
    interval = sample_interval_sec
    base = build_lag_transformation_config(
        enabled=lag_enabled,
        features=feats,
        lag_seconds=list(lag_seconds or []),
        partition_by=["trading_day", "token"],
        sample_interval_sec=interval,
        difference_enabled=difference_enabled,
        return_enabled=return_enabled,
    )
    with_rolling = merge_rolling_into_config(
        base,
        enabled=rolling_enabled,
        features=list(rolling_features if rolling_features is not None else feats),
        windows=list(rolling_windows or []),
        operations=list(rolling_operations or []),
        partition_by=["trading_day", "token"],
        sample_interval_sec=interval,
    )
    with_exp = merge_exponential_rolling_into_config(
        with_rolling,
        enabled=exponential_rolling_enabled,
        features=list(exponential_features if exponential_features is not None else feats),
        periods=list(exponential_periods or []),
        operations=list(exponential_operations or ["ema"]),
        partition_by=["trading_day", "token"],
        sample_interval_sec=interval,
    )
    with_ohlc = merge_ohlc_aggregation_into_config(
        with_exp,
        enabled=ohlc_enabled,
        features=list(ohlc_features if ohlc_features is not None else feats),
        timeframes=list(ohlc_timeframes or []),
        outputs=list(ohlc_outputs or ["open", "high", "low", "close"]),
        partition_by=["trading_day", "token"],
        sample_interval_sec=interval,
    )
    cfg = merge_interaction_into_config(
        with_ohlc,
        enabled=True,
        pairs=list(interaction_pairs or []),
    )
    return available_interaction_features_from_config(
        cfg,
        master_features=master_features,
        sample_interval_sec=interval,
    )


def bulk_interaction_pairs(
    feature_a: list[str],
    feature_b: list[str],
    *,
    op: str = "multiply",
    scale: float = 1.0,
    skip_identical: bool = True,
) -> list[dict[str, Any]]:
    """Cartesian product A × B → pair configs."""
    op_key = normalize_interaction_op(op)
    pairs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for left in feature_a:
        left_s = str(left).strip()
        if not left_s:
            continue
        for right in feature_b:
            right_s = str(right).strip()
            if not right_s:
                continue
            if skip_identical and left_s == right_s:
                continue
            output = interaction_column_name(left_s, right_s, op_key)
            if output in seen:
                continue
            seen.add(output)
            pairs.append({
                "left": left_s,
                "right": right_s,
                "op": op_key,
                "output": output,
                "scale": float(scale),
            })
    return pairs


def format_lineage_tree(
    feature: str,
    *,
    parent_map: dict[str, dict[str, Any]] | None = None,
) -> str:
    """Human-readable lineage for one feature (ASCII tree).

    Example::

        spot_return_30s_x_moneyness
        └── Interaction (multiply)
            ├── spot_return_30s
            └── moneyness
    """
    parents = parent_map or {}
    node = parents.get(feature)
    if not node:
        return str(feature)
    left = str(node.get("left") or "")
    right = str(node.get("right") or "")
    op = str(node.get("op") or "multiply")
    try:
        op_key = normalize_interaction_op(op)
    except Exception:
        op_key = op
    lines = [
        str(feature),
        f"└── Interaction ({op_key})",
        f"    ├── {left}",
    ]
    if left in parents:
        for sub in format_lineage_tree(left, parent_map=parents).splitlines()[1:]:
            lines.append(f"    │   {sub}")
    lines.append(f"    └── {right}")
    if right in parents:
        for sub in format_lineage_tree(right, parent_map=parents).splitlines()[1:]:
            lines.append(f"        {sub}")
    return "\n".join(lines)


def format_pair_chain_entry(
    index: int,
    pair: dict[str, Any],
) -> str:
    """Visual chain block for the configured-pairs list."""
    try:
        norm = normalize_interaction_pair(pair)
    except Exception:
        return f"{index}. (invalid pair)"
    left, right, op, output = norm["left"], norm["right"], norm["op"], norm["output"]
    symbol = {
        "multiply": "×",
        "divide": "÷",
        "add": "+",
        "subtract": "−",
        "min": "min",
        "max": "max",
        "absolute_difference": "absdiff",
    }.get(op, op)
    return (
        f"{index}.\n"
        f"{left}\n"
        f"{symbol}\n"
        f"{right}\n"
        f"↓\n"
        f"{output}"
    )


def format_pair_lineage_brief(pair: dict[str, Any]) -> str:
    """Compact lineage one-liner for list rows."""
    try:
        norm = normalize_interaction_pair(pair)
    except Exception:
        return str(pair)
    left, right, op, output = norm["left"], norm["right"], norm["op"], norm["output"]
    return (
        f"{left}\n"
        f"└── Interaction ({op})\n"
        f"    ├── {right}\n"
        f"    └── {output}"
    )


def pipeline_feature_ledger(
    *,
    master_count: int,
    lag_enabled: bool = False,
    difference_enabled: bool = False,
    return_enabled: bool = False,
    rolling_count: int = 0,
    exponential_rolling_count: int = 0,
    ohlc_aggregation_count: int = 0,
    interaction_count: int = 0,
    math_count: int = 0,
    normalization_count: int = 0,
    regime_count: int = 0,
    selected_features: list[str] | None = None,
    lag_seconds: list[int] | None = None,
    lag_count: int | None = None,
    difference_count: int | None = None,
    return_count: int | None = None,
) -> dict[str, Any]:
    """Counts for the Feature Preview ledger."""
    n_feat = len([f for f in (selected_features or []) if str(f).strip()])
    n_lags = len([int(s) for s in (lag_seconds or []) if int(s) > 0])
    lag_n = int(lag_count) if lag_count is not None else (n_feat * n_lags if lag_enabled else 0)
    diff_n = (
        int(difference_count)
        if difference_count is not None
        else (n_feat * n_lags if difference_enabled else 0)
    )
    ret_n = (
        int(return_count)
        if return_count is not None
        else (n_feat * n_lags if return_enabled else 0)
    )
    roll_n = max(0, int(rolling_count or 0))
    exp_n = max(0, int(exponential_rolling_count or 0))
    ohlc_n = max(0, int(ohlc_aggregation_count or 0))
    ix_n = max(0, int(interaction_count or 0))
    math_n = max(0, int(math_count or 0))
    norm_n = max(0, int(normalization_count or 0))
    regime_n = max(0, int(regime_count or 0))
    master = max(0, int(master_count or 0))
    final = (
        master + lag_n + diff_n + ret_n + roll_n + exp_n + ohlc_n + ix_n
        + math_n + norm_n + regime_n
    )
    return {
        "master": master,
        "lag": lag_n,
        "difference": diff_n,
        "return": ret_n,
        "rolling": roll_n,
        "exponential_rolling": exp_n,
        "ohlc_aggregation": ohlc_n,
        "interaction": ix_n,
        "math": math_n,
        "normalization": norm_n,
        "regime": regime_n,
        "final": final,
    }


def format_pipeline_ledger_text(ledger: dict[str, Any]) -> str:
    lines = [
        f"Master Features              : {ledger.get('master', 0)}",
        f"Lag                          : +{ledger.get('lag', 0)}",
        f"Difference                   : +{ledger.get('difference', 0)}",
        f"Return                       : +{ledger.get('return', 0)}",
        f"Rolling                      : +{ledger.get('rolling', 0)}",
        f"Exponential Rolling          : +{ledger.get('exponential_rolling', 0)}",
        f"OHLC Aggregation             : +{ledger.get('ohlc_aggregation', 0)}",
        f"Interaction                  : +{ledger.get('interaction', 0)}",
        f"Math (Unary)                 : +{ledger.get('math', 0)}",
        f"Normalization                : +{ledger.get('normalization', 0)}",
        f"Regime / Bucket              : +{ledger.get('regime', 0)}",
        "--------------------------------",
        f"Final Dataset                : {ledger.get('final', 0)} features",
    ]
    return "\n".join(lines)


def merge_interaction_into_config(
    base_config: dict[str, Any] | None,
    *,
    enabled: bool,
    pairs: list[dict[str, Any]],
    div_zero: str = "null",
    eps: float = 1e-12,
) -> dict[str, Any]:
    """Attach / replace the interaction entry on a pipeline config."""
    from .config import normalize_transformation_config

    cfg = normalize_transformation_config(base_config)
    transforms = [
        t for t in (cfg.get("transformations") or [])
        if str((t or {}).get("id") or "") != "interaction"
    ]
    if enabled and pairs:
        normalized = [normalize_interaction_pair(p) for p in pairs]
        transforms.append({
            "id": "interaction",
            "enabled": True,
            "order": 50,
            "params": {
                "pairs": normalized,
                "div_zero": str(div_zero or "null"),
                "eps": float(eps),
                "fail_on_duplicate_output": True,
            },
        })
    cfg["transformations"] = transforms
    return cfg


def validate_interaction_for_export(
    *,
    enabled: bool,
    pairs: list[dict[str, Any]],
    available_features: list[str] | None = None,
) -> str | None:
    if not enabled:
        return None
    if not pairs:
        return "Interaction is enabled but no pairs are configured."
    avail = set(available_features or ())
    try:
        from .interaction import validate_interaction_pairs

        validate_interaction_pairs(
            pairs,
            existing_columns=avail if avail else None,
            fail_on_duplicate_output=True,
            allow_overwrite=False,
        )
    except Exception as exc:
        return str(exc)
    return None
