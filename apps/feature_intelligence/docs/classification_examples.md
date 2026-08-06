# Classification Examples (Sprint 6)

Worked examples of ontology rows. Multi-value fields are stored as sorted JSON arrays.

## Primitive — `PR_SPOT`

| Field | Value |
|-------|-------|
| object_type | PRIMITIVE |
| object_id | PR_SPOT |
| domain | DOM_PRICE |
| signal_type | SIG_LEVEL, SIG_RAW (sorted) |
| mathematical_family | MATH_IDENTITY |
| horizon | HOR_INTRADAY |
| output_type | OUT_NUMERIC |
| frequency | FREQ_ANY |
| stability | STAB_STABLE |
| input_dependencies | [] |
| classification_source | SEED |

## Operator — `OP_EMA`

| Field | Value |
|-------|-------|
| object_type | OPERATOR |
| object_id | OP_EMA |
| domain | DOM_DERIVED |
| signal_type | SIG_TREND |
| mathematical_family | MATH_MOVING_AVERAGE |
| horizon | HOR_INTRADAY |
| output_type | OUT_NUMERIC |
| frequency | FREQ_ANY |
| stability | STAB_STABLE |
| classification_source | SEED |

## Operator — `OP_RATIO`

| Field | Value |
|-------|-------|
| domain | DOM_DERIVED |
| signal_type | SIG_RATIO |
| mathematical_family | MATH_ARITHMETIC |

## Optional feature / transformation

`FEAT_*` and `TR_*` ontology rows are **optional** in Sprint 6. When present they use the
same field model and are validated; coverage % may be low. Classify via import:

```json
{
  "schema_version": "1.0",
  "ontology_version": "1.0.0",
  "records": [
    {
      "object_type": "FEATURE",
      "object_id": "FEAT_0123456789ABCDEF0123456789ABCDEF",
      "ontology_version": "1.0.0",
      "domain": "DOM_DERIVED",
      "signal_type": ["SIG_TREND"],
      "mathematical_family": ["MATH_MOVING_AVERAGE"],
      "horizon": "HOR_INTRADAY",
      "output_type": "OUT_NUMERIC",
      "frequency": "FREQ_ANY",
      "stability": "STAB_EXPERIMENTAL",
      "input_dependencies": ["OP_EMA", "PR_SPOT"],
      "classification_source": "IMPORT"
    }
  ]
}
```

`input_dependencies` are **references only** — no graph traversal in Sprint 6.
