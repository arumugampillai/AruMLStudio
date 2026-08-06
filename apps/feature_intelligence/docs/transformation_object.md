# Transformation Object

In-memory product of `parse` / `compile`.

## Fields

| Field | Notes |
|-------|-------|
| `transformation_uuid` | Derived `TR_*` from `canonical_text` |
| `expression_hash` | Full SHA-256 of canonical UTF-8 |
| `compilation_uuid` | `COMP_*` on compile; `None` on parse-only |
| `ast_hash` | Versioned AST integrity fingerprint |
| `ast_schema_version` | `"1.0"` — shape (≠ `compiler_version`) |
| `grammar_version` / `compiler_version` / `operator_pack_version` | Version triple |
| `source_text` / `canonical_text` | As submitted / after formatter |
| `root` | In-memory `AstNode` |
| `manifest` | `CompilerManifest` on compile |
| `cache_hit` | `True` when AST served from cache; `None` on parse |
| `metrics` | Optional `CompileMetrics` when enabled |

## Related ids

```text
expression_hash     = SHA256(canonical_text UTF-8)          # 64 lowercase hex
transformation_uuid = "TR_" + expression_hash[:32].upper()
compilation_uuid    = "COMP_" + UUIDv7_32_uppercase_hex
```

See [compiler.md](./compiler.md), [ast.md](./ast.md), [manifest.md](./manifest.md).
