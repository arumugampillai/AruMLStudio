# Relationship Types (Sprint 7)

Edges store a foreign key `relationship_id` (`REL_*`) — never free-text labels.

## Frozen vocabulary (pack 1.0.0 — exactly five)

| `relationship_id` | `canonical_name` | Meaning (parent → child) |
|-------------------|------------------|--------------------------|
| `REL_USES` | `uses` | Parent used in constructing child (typical: `OP_*` → `TR_*`) |
| `REL_GENERATED_BY` | `generated_by` | Parent generated child (typical: `TR_*` → `FEAT_*`) |
| `REL_DEPENDS_ON` | `depends_on` | Parent is a dependency of child (typical: `FEAT_*` → `TR_*`) |
| `REL_DERIVED_FROM` | `derived_from` | Child derived from parent (typical: `PR_*` → `FEAT_*` closure) |
| `REL_INPUT_TO` | `input_to` | Parent is an input into child (typical: `PR_*` → `TR_*`) |

Soft-retire: set `active=0`; new edges reject inactive ids; historical edges may remain.

## Phase 1 derive map

For each `TR_*` with persisted `ast_nodes` (+ `feature_ast` links):

| # | Condition | Edge |
|---|-----------|------|
| D1 | `node_type=primitive` | `PR_*` → `TR_*` / `REL_INPUT_TO` |
| D2 | `node_type=operator` | `OP_*` → `TR_*` / `REL_USES` |
| D3 | `node_type=feature` | `FEAT_*` → `TR_*` / `REL_DEPENDS_ON` |
| D4 | `feature_ast` link | `TR_*` → `FEAT_*` / `REL_GENERATED_BY` |
| D5 | Closure (default on) | `PR_*` → `FEAT_*` / `REL_DERIVED_FROM` |

Literals and list nodes emit no edges. Closure disabled with `--no-closure`.
