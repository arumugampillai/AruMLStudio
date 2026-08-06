# Lineage Graph Model (Sprint 7)

Directed **acyclic** graph of FIC identity objects. Edge direction is
`parent_object → child_object` (upstream → downstream).

## Allowed nodes

| Prefix | Pattern |
|--------|---------|
| `PR_*` | `^PR_[A-Z][A-Z0-9_]*$` |
| `OP_*` | `^OP_[A-Z][A-Z0-9_]*$` |
| `TR_*` | `^TR_[0-9A-F]{32}$` |
| `FEAT_*` | `^FEAT_[0-9A-F]{32}$` (hyphens stripped) |

Forbidden as nodes: `ONT_*`, `COMP_*`, manifests, vocabulary / `REL_*` ids, AST node ids.

## DAG rules

- Multiple parents and multiple children allowed
- Self-edges forbidden
- Duplicate `(parent, child, relationship_id)` forbidden
- Parallel edges with different `REL_*` allowed
- Cycles forbidden — detect and reject only (no cycle optimization)

## Navigation API

| API | Behavior |
|-----|----------|
| `parents(id)` | Direct parents (ASCII ascending) |
| `children(id)` | Direct children (ASCII ascending) |
| `ancestors(id)` | Transitive parents; exclude self; ASCII sorted set |
| `descendants(id)` | Transitive children; exclude self; ASCII sorted set |

No shortest-path, LCA, weighted, or scored traversal in Sprint 7.

## Graph checksum

1. Collect triples `(parent, child, relationship_id)`
2. Sort ASCII ascending on the tuple
3. Serialize UTF-8 lines: `parent\tchild\trelationship_id\n` (LF only; trailing newline)
4. `graph_checksum = SHA256(...).hexdigest()`

Empty graph → SHA-256 of empty bytes.

`graph_schema_version` (`1.0`) describes this model shape and evolves independently of
`lineage_version` (`1.0.0`).

`lineage validate` always recomputes and writes `graph_checksum`.
