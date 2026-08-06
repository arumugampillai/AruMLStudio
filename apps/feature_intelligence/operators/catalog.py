"""Frozen Operator Catalog — pack 1.0.0 / catalog 1.0 (Sprint 3 v1.2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, NamedTuple

OPERATOR_CATALOG_VERSION = "1.0"
OPERATOR_PACK_VERSION = "1.0.0"
OPERATOR_VERSION_DEFAULT = "1.0"

OPERATOR_ID_PATTERN = r"^OP_[A-Z][A-Z0-9_]*$"


def _schema(required: list[str], properties: dict[str, Any]) -> str:
    obj = {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def _window_period(desc: str = "Lookback window") -> dict[str, Any]:
    return {"type": "window", "minimum": 1, "description": desc}


def _feat(name: str, desc: str) -> dict[str, Any]:
    return {name: {"type": "feature", "description": desc}}


class SeedOperator(NamedTuple):
    operator_id: str
    canonical_name: str
    display_name: str
    category: str
    formula: str
    definition_text: str
    parameter_schema_json: str
    input_arity_min: int
    input_arity_max: int | None
    warmup_policy: str
    missing_data_policy: str
    deterministic: int
    stateful: int
    streaming_supported: int
    incremental_supported: int
    complexity_class: str
    description: str = ""


def _op(
    oid: str,
    name: str,
    category: str,
    formula: str,
    definition: str,
    schema: str,
    *,
    amin: int,
    amax: int | None,
    warmup: str,
    missing: str = "NAN",
    det: int = 1,
    stateful: int,
    streaming: int,
    incremental: int,
    complexity: str,
    display: str | None = None,
    description: str = "",
) -> SeedOperator:
    return SeedOperator(
        operator_id=oid,
        canonical_name=name,
        display_name=display or name.replace("_", " ").title(),
        category=category,
        formula=formula,
        definition_text=definition,
        parameter_schema_json=schema,
        input_arity_min=amin,
        input_arity_max=amax,
        warmup_policy=warmup,
        missing_data_policy=missing,
        deterministic=det,
        stateful=stateful,
        streaming_supported=streaming,
        incremental_supported=incremental,
        complexity_class=complexity,
        description=description,
    )


_EMPTY = _schema([], {})
_PERIOD = _schema(["period"], {"period": _window_period()})
_LAG = _schema(["periods"], {"periods": _window_period("Lag/lead periods")})
_RATIO = _schema(
    ["left", "right"],
    {**_feat("left", "Numerator"), **_feat("right", "Denominator")},
)
_BINARY = _schema(
    ["left", "right"],
    {**_feat("left", "Left input"), **_feat("right", "Right input")},
)
_CLIP = _schema(
    ["lo", "hi"],
    {
        "lo": {"type": "float", "description": "Lower bound"},
        "hi": {"type": "float", "description": "Upper bound"},
    },
)
_PERCENTILE = _schema(
    ["period", "q"],
    {
        "period": _window_period(),
        "q": {"type": "float", "minimum": 0, "maximum": 1, "description": "Quantile in [0,1]"},
    },
)

SEED_OPERATORS: tuple[SeedOperator, ...] = (
    _op("OP_EMA", "ema", "TREND", "ema_t=a*x_t+(1-a)*ema_{t-1}", "Exponential moving average with smoothing alpha derived from period.", _PERIOD, amin=1, amax=1, warmup="WINDOW", stateful=1, streaming=1, incremental=1, complexity="O(1)"),
    _op("OP_SMA", "sma", "TREND", "sma_t=mean(x_{t-p+1..t})", "Simple moving average over a fixed window.", _PERIOD, amin=1, amax=1, warmup="WINDOW", stateful=1, streaming=1, incremental=0, complexity="O(window)"),
    _op("OP_WMA", "wma", "TREND", "wma_t=weighted_mean(x_{t-p+1..t})", "Linearly weighted moving average over a fixed window.", _PERIOD, amin=1, amax=1, warmup="WINDOW", stateful=1, streaming=1, incremental=0, complexity="O(window)"),
    _op("OP_RATIO", "ratio", "ARITHMETIC", "y=left/right", "Elementwise ratio of two series.", _RATIO, amin=2, amax=2, warmup="NONE", stateful=0, streaming=1, incremental=1, complexity="O(1)"),
    _op("OP_DIFFERENCE", "difference", "ARITHMETIC", "y=left-right", "Elementwise difference of two series.", _BINARY, amin=2, amax=2, warmup="NONE", stateful=0, streaming=1, incremental=1, complexity="O(1)"),
    _op("OP_SUM", "sum", "ARITHMETIC", "y=sum(inputs)", "Elementwise sum of two or more series.", _EMPTY, amin=2, amax=None, warmup="NONE", stateful=0, streaming=1, incremental=1, complexity="O(n)"),
    _op("OP_PRODUCT", "product", "ARITHMETIC", "y=prod(inputs)", "Elementwise product of two or more series.", _EMPTY, amin=2, amax=None, warmup="NONE", stateful=0, streaming=1, incremental=1, complexity="O(n)"),
    _op("OP_DIVIDE", "divide", "ARITHMETIC", "y=left/right", "Elementwise division of two series.", _BINARY, amin=2, amax=2, warmup="NONE", stateful=0, streaming=1, incremental=1, complexity="O(1)"),
    _op("OP_LAG", "lag", "TIME", "y_t=x_{t-k}", "Shift series backward by k periods.", _LAG, amin=1, amax=1, warmup="WINDOW", stateful=1, streaming=1, incremental=1, complexity="O(1)"),
    _op("OP_LEAD", "lead", "TIME", "y_t=x_{t+k}", "Shift series forward by k periods (lookahead; research-only).", _LAG, amin=1, amax=1, warmup="WINDOW", stateful=1, streaming=0, incremental=0, complexity="O(1)"),
    _op("OP_DELTA", "delta", "TIME", "y_t=x_t-x_{t-1}", "One-step discrete difference.", _EMPTY, amin=1, amax=1, warmup="NONE", stateful=1, streaming=1, incremental=1, complexity="O(1)"),
    _op("OP_ROC", "roc", "TIME", "y_t=(x_t-x_{t-k})/x_{t-k}", "Rate of change over k periods.", _LAG, amin=1, amax=1, warmup="WINDOW", stateful=1, streaming=1, incremental=1, complexity="O(1)"),
    _op("OP_MIN", "min", "COMPARISON", "y=min(inputs)", "Elementwise minimum across inputs.", _EMPTY, amin=2, amax=None, warmup="NONE", stateful=0, streaming=1, incremental=1, complexity="O(n)"),
    _op("OP_MAX", "max", "COMPARISON", "y=max(inputs)", "Elementwise maximum across inputs.", _EMPTY, amin=2, amax=None, warmup="NONE", stateful=0, streaming=1, incremental=1, complexity="O(n)"),
    _op("OP_MEAN", "mean", "STATISTICAL", "y_t=mean(x_{t-p+1..t})", "Rolling mean over window.", _PERIOD, amin=1, amax=1, warmup="WINDOW", stateful=1, streaming=1, incremental=0, complexity="O(window)"),
    _op("OP_MEDIAN", "median", "STATISTICAL", "y_t=median(x_{t-p+1..t})", "Rolling median over window.", _PERIOD, amin=1, amax=1, warmup="WINDOW", stateful=1, streaming=0, incremental=0, complexity="O(window)"),
    _op("OP_STDDEV", "stddev", "STATISTICAL", "y_t=std(x_{t-p+1..t})", "Rolling standard deviation over window.", _PERIOD, amin=1, amax=1, warmup="WINDOW", stateful=1, streaming=1, incremental=0, complexity="O(window)"),
    _op("OP_VARIANCE", "variance", "STATISTICAL", "y_t=var(x_{t-p+1..t})", "Rolling variance over window.", _PERIOD, amin=1, amax=1, warmup="WINDOW", stateful=1, streaming=1, incremental=0, complexity="O(window)"),
    _op("OP_ZSCORE", "zscore", "NORMALIZATION", "y_t=(x_t-mean)/std", "Rolling z-score normalization.", _PERIOD, amin=1, amax=1, warmup="WINDOW", stateful=1, streaming=1, incremental=0, complexity="O(window)"),
    _op("OP_PERCENTILE", "percentile", "STATISTICAL", "y_t=percentile(x_{t-p+1..t},q)", "Rolling percentile at quantile q.", _PERCENTILE, amin=1, amax=1, warmup="WINDOW", stateful=1, streaming=0, incremental=0, complexity="O(window)"),
    _op("OP_ABS", "abs", "TRANSFORMATION", "y=|x|", "Elementwise absolute value.", _EMPTY, amin=1, amax=1, warmup="NONE", stateful=0, streaming=1, incremental=1, complexity="O(1)"),
    _op("OP_LOG", "log", "TRANSFORMATION", "y=log(x)", "Elementwise natural logarithm.", _EMPTY, amin=1, amax=1, warmup="NONE", stateful=0, streaming=1, incremental=1, complexity="O(1)"),
    _op("OP_EXP", "exp", "TRANSFORMATION", "y=exp(x)", "Elementwise exponential.", _EMPTY, amin=1, amax=1, warmup="NONE", stateful=0, streaming=1, incremental=1, complexity="O(1)"),
    _op("OP_SQRT", "sqrt", "TRANSFORMATION", "y=sqrt(x)", "Elementwise square root.", _EMPTY, amin=1, amax=1, warmup="NONE", stateful=0, streaming=1, incremental=1, complexity="O(1)"),
    _op("OP_CLIP", "clip", "NORMALIZATION", "y=clip(x,lo,hi)", "Clip values to [lo, hi].", _CLIP, amin=1, amax=1, warmup="NONE", stateful=0, streaming=1, incremental=1, complexity="O(1)"),
    _op("OP_NORMALIZE", "normalize", "NORMALIZATION", "y=(x-min)/(max-min)", "Min-max normalize over a window.", _PERIOD, amin=1, amax=1, warmup="WINDOW", stateful=1, streaming=0, incremental=0, complexity="O(window)"),
    _op("OP_ROLLING_MIN", "rolling_min", "ROLLING", "y_t=min(x_{t-p+1..t})", "Rolling minimum.", _PERIOD, amin=1, amax=1, warmup="WINDOW", stateful=1, streaming=1, incremental=0, complexity="O(window)"),
    _op("OP_ROLLING_MAX", "rolling_max", "ROLLING", "y_t=max(x_{t-p+1..t})", "Rolling maximum.", _PERIOD, amin=1, amax=1, warmup="WINDOW", stateful=1, streaming=1, incremental=0, complexity="O(window)"),
    _op("OP_ROLLING_MEAN", "rolling_mean", "ROLLING", "y_t=mean(x_{t-p+1..t})", "Rolling mean (alias family of mean).", _PERIOD, amin=1, amax=1, warmup="WINDOW", stateful=1, streaming=1, incremental=0, complexity="O(window)"),
    _op("OP_SLOPE", "slope", "TREND", "y_t=slope(x_{t-p+1..t})", "Rolling linear regression slope.", _PERIOD, amin=1, amax=1, warmup="WINDOW", stateful=1, streaming=0, incremental=0, complexity="O(window)"),
    _op("OP_INTERACTION", "interaction", "INTERACTION", "y=left*right", "Elementwise product interaction of two features.", _BINARY, amin=2, amax=2, warmup="NONE", stateful=0, streaming=1, incremental=1, complexity="O(1)"),
)

SEED_BY_ID: dict[str, SeedOperator] = {o.operator_id: o for o in SEED_OPERATORS}


def canonical_operator_catalog_document(
    seeds: tuple[SeedOperator, ...] | None = None,
) -> str:
    rows = sorted(seeds or SEED_OPERATORS, key=lambda o: o.operator_id)
    lines: list[str] = []
    for o in rows:
        amax = "" if o.input_arity_max is None else str(o.input_arity_max)
        # Ensure schema canonical form
        schema_obj = json.loads(o.parameter_schema_json)
        schema_canon = json.dumps(schema_obj, sort_keys=True, separators=(",", ":"))
        lines.append(
            f"{o.operator_id}|{o.canonical_name}|{o.category}|{OPERATOR_CATALOG_VERSION}|"
            f"{OPERATOR_PACK_VERSION}|{OPERATOR_VERSION_DEFAULT}|{o.input_arity_min}|{amax}|"
            f"1|{o.warmup_policy}|{o.missing_data_policy}|{o.deterministic}|"
            f"{o.complexity_class}|{schema_canon}"
        )
    return "\n".join(lines) + "\n"


def compute_operator_catalog_hash(
    seeds: tuple[SeedOperator, ...] | None = None,
) -> str:
    return hashlib.sha256(
        canonical_operator_catalog_document(seeds).encode("utf-8")
    ).hexdigest()


# Locked for pack 1.0.0 — update only with an intentional freeze bump.
EXPECTED_OPERATOR_CATALOG_HASH = (
    "1f410b3a44f3bc499af33d3c44211543ea42929bbe6bd9d88b36b534b070f4be"
)


def catalog_artifact_path() -> Path:
    return Path(__file__).resolve().parent / "operator_catalog.json"


def build_catalog_artifact_dict() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "catalog_version": OPERATOR_CATALOG_VERSION,
        "operator_pack_version": OPERATOR_PACK_VERSION,
        "expected_catalog_hash": EXPECTED_OPERATOR_CATALOG_HASH,
        "operators": [
            {
                "operator_id": o.operator_id,
                "canonical_name": o.canonical_name,
                "display_name": o.display_name,
                "category": o.category,
                "formula": o.formula,
                "definition_text": o.definition_text,
                "description": o.description,
                "parameter_schema": json.loads(o.parameter_schema_json),
                "depends_on_operator_ids": None,
                "input_arity_min": o.input_arity_min,
                "input_arity_max": o.input_arity_max,
                "output_count": 1,
                "warmup_policy": o.warmup_policy,
                "missing_data_policy": o.missing_data_policy,
                "deterministic": bool(o.deterministic),
                "stateful": bool(o.stateful),
                "streaming_supported": bool(o.streaming_supported),
                "incremental_supported": bool(o.incremental_supported),
                "complexity_class": o.complexity_class,
                "operator_version": OPERATOR_VERSION_DEFAULT,
                "catalog_version": OPERATOR_CATALOG_VERSION,
                "operator_pack_version": OPERATOR_PACK_VERSION,
            }
            for o in sorted(SEED_OPERATORS, key=lambda x: x.operator_id)
        ],
    }


def write_catalog_artifacts() -> tuple[Path, Path]:
    """Write operator_catalog.json and summary CSV next to this module."""
    root = Path(__file__).resolve().parent
    json_path = root / "operator_catalog.json"
    csv_path = root / "operator_catalog.csv"
    data = build_catalog_artifact_dict()
    json_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    headers = [
        "operator_id",
        "canonical_name",
        "category",
        "operator_pack_version",
        "warmup_policy",
        "missing_data_policy",
        "deterministic",
        "complexity_class",
    ]
    lines = [",".join(headers)]
    for o in sorted(SEED_OPERATORS, key=lambda x: x.operator_id):
        lines.append(
            ",".join(
                [
                    o.operator_id,
                    o.canonical_name,
                    o.category,
                    OPERATOR_PACK_VERSION,
                    o.warmup_policy,
                    o.missing_data_policy,
                    str(bool(o.deterministic)),
                    o.complexity_class,
                ]
            )
        )
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path

