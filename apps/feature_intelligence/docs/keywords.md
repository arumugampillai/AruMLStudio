# TL Tokens & Conventional Names

Grammar 1.0 does **not** reserve English words like `source` / `period` globally.

## Frozen prefixes

| Prefix | Meaning |
|--------|---------|
| `PR_` | Primitive |
| `FEAT_` | Feature UUID |
| `OP_` | Operator |
| `TR_` | Transformation (reserved; not issued in Sprint 4) |

## Punctuators

`(` `)` `,` `=` `[` `]` — `:` is reserved and **forbidden** in expression text.

## Boolean literals

`true` · `false` (lowercase only)

## Conventional parameter names

Common in operator schemas / series slots (not language keywords):

`source`, `left`, `right`, `period`, `periods`, `lo`, `hi`, `q`

Series inputs may appear as named args (e.g. `source = PR_SPOT`) even when Sprint 3 `parameter_schema_json` lists only hyperparameters such as `period` — arity is tracked separately on the operator row.
