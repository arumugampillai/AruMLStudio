# TL Examples

Corpus under `grammar/examples/`.

## Valid

| File | Notes |
|------|-------|
| `valid/ema_spot.tl` | `OP_EMA` with `source=PR_SPOT`, `period=20` |
| `valid/ratio_nested.tl` | Nested `OP_EMA` inside `OP_RATIO` |
| `valid/ema_period_only.tl` | Schema-only params |

Validate with `bound` against a migrated FIC DB (operators + primitives seeded).

## Invalid

Each file declares `# expect: CODE` (or a `.expected` sidecar).

| Code | Example |
|------|---------|
| `TRAILING_COMMA` | `trailing_comma.tl` |
| `POSITIONAL_ARG` | `positional.tl` |
| `FORBIDDEN_COLON` | `forbidden_colon.tl` |
| `UNKNOWN_OPERATOR` | `unknown_operator.tl` (bound) |
| `MISSING_PARAM` | `missing_param.tl` (bound) |
| `DUPLICATE_PARAM` | `duplicate_param.tl` |
| `BAD_LITERAL` | `bad_literal.tl` |
| `UNBALANCED_PAREN` | `unbalanced_paren.tl` |

No arrow syntax appears under `valid/`.
