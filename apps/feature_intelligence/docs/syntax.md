# TL Syntax (Grammar 1.0)

Canonical surface is **function-call form only**:

```text
OP_EMA(
    source = PR_SPOT,
    period = 20
)
```

Arrow chains (`A → B`) are documentation sugar — **not** accepted.

## Rules

| Rule | Detail |
|------|--------|
| Callee | `OP_*` only |
| Arguments | Named only: `name = value` |
| Trailing comma | Forbidden |
| Colon `:` | Forbidden in expressions |
| Nesting | Max call depth **64** |
| Feature ids | `FEAT_` + 32 uppercase hex |

## Values

`Call` · `PR_*` · `FEAT_*` · int / float / `true`/`false` / `"string"` · `[...]`

## Validation modes

| Mode | Checks |
|------|--------|
| `syntax_only` | Lexical form, shapes, patterns, nesting |
| `bound` | Plus registry existence + operator param schema |

## Canonical layout

- No space before `(`
- Args on indented lines (`name = value`), comma except last
- Closing `)` on its own line
- Param order: schema `required` order, then remaining names alphabetically
