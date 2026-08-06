# Transformation Compiler

Sprint 5 compiles **canonical Transformation Language (TL) 1.0** into a
content-addressed **Transformation** (`TR_*`) plus a persisted **Feature AST**,
and records each compile as a **Compilation** event (`COMP_*`).

## Pipeline

```text
source_text
  → grammar.format (canonical_text)
  → grammar.validate (bound)
  → derive TR_* / expression_hash
  → cache lookup by expression_hash
  → build AST → subtree_hash / ast_hash
  → mint COMP_* + Manifest (+ optional persist)
```

There is **no execution**, optimization, folding, or bytecode.

## Identity

| Id | Meaning |
|----|---------|
| `TR_*` | Semantic identity = `"TR_" + SHA256(canonical_text)[:32].upper()` |
| `expression_hash` | Full SHA-256 hex of canonical UTF-8 |
| `COMP_*` | UUIDv7 compile event (may differ across runs) |
| `ast_hash` | Integrity fingerprint: `SHA256(grammar\|compiler\|pack\|root.subtree_hash)` — **not** used for `TR_*` |

## Versions

- `ast_schema_version` = `1.0` (node model / serialization shape)
- `compiler_version` = `1.0.0` (implementation)
- These evolve independently.

## Public API

```python
from feature_intelligence.compiler import parse, compile, decompile

parse(text, mode="bound", db=path)      # AST + TR_*; no COMP_*; no persist
compile(text, mode="bound", db=path, persist=False, metrics=False, ...)
decompile(ast_root) → TL text
```

## Cache

Keyed by `expression_hash`. Hits reuse the AST without rebuild. Optional
lightweight `COMP_*` with `cache_hit=true` may be recorded; AST rows are
**not** rewritten.

## CLI

```bash
python -m feature_intelligence compiler compile --expr "OP_EMA(period=20)" [--db PATH] [--metrics]
python -m feature_intelligence compiler validate --expr "..." [--roundtrip]
python -m feature_intelligence compiler manifest --expr "..."
python -m feature_intelligence compiler export --format json --out out.json --expr "..."
python -m feature_intelligence compiler import --format json --in out.json [--db PATH]
```

No `compiler execute` command exists.
