# Compiler Manifest

Emitted on every successful `compile` that mints a compilation event. Ties the
event (`COMP_*`) to the stable transformation identity (`TR_*`).

## Fields

```yaml
transformation_manifest:
  transformation_uuid: TR_...
  compilation_uuid: COMP_...
  expression_hash: ...
  ast_schema_version: "1.0"
  grammar_version: "1.0"
  grammar_pack_version: "1.0.0"
  formatter_version: "1.0.0"
  compiler_version: "1.0.0"
  operator_pack_version: "1.0.0"
  definition_version: null
  implementation_version: null
  ast_hash: ...
  root_subtree_hash: ...
  operator_versions: { OP_EMA: "1.0", ... }
  canonical_text: ...
  cache_hit: false
  created_at: "..."   # per invocation
```

## Determinism equality

When comparing manifests across runs, **ignore** `created_at`,
`compilation_uuid`, and `cache_hit`. All other fields must match for the same
canonical text and version triple.

## CLI

```bash
python -m feature_intelligence compiler manifest --expr "OP_EMA(period=20)"
```
