# Feature AST

The Feature AST is the compiled tree for a Transformation. It is **not** the
Sprint 4 internal syntax tree.

## Node types

| `node_type` | Fields |
|-------------|--------|
| `operator` | `operator_id`, named-arg `children` |
| `primitive` | `primitive_id` |
| `feature` | `feature_uuid` |
| `literal` | `literal: {kind, value}` |
| `list` | ordered `children` |

Every node has `node_id` (`N0`, `N1`, … DFS preorder), `parent`, `subtree_hash`,
and reserved `stable_node_hash` (**null / omitted** in Sprint 5 — not computed).

Operator children follow Sprint 4 formatter parameter order (required schema
order, then remaining names alphabetically). Only schema-declared parameters
appear — undeclared slots such as `source=` on `OP_EMA` are rejected.

## Hashing

- `subtree_hash` — bottom-up SHA-256 over canonical JSON payload
  `{node_type, payload, child_subtree_hashes}` (excludes `node_id` / `parent` /
  `stable_node_hash`)
- `ast_hash` — versioned envelope over
  `grammar_version|compiler_version|operator_pack_version|root.subtree_hash`

## Storage (single AST store)

| Table | Role |
|-------|------|
| `ast_nodes` | Normalized projection keyed by `(transformation_uuid, ast_hash, node_id)` |
| `feature_ast.ast_json` | Sole full AST document when a feature is linked |
| `compilation_registry` | Event metadata only — **no** `compiled_json` blob |

See also: [compiler.md](./compiler.md), [transformation_object.md](./transformation_object.md).
