# Primitive-to-Feature Traceability Contract (Sprint 1)

Canonical freeze: Sprint 1 §10.

## Rules

1. **Leaf rule:** AST market-data leaves must reference a `primitive_id` present in `primitive_registry`.
2. **Root rule:** Lineage roots are the set of reachable `primitive_id` values.
3. **Reference form:** JSON field name is always `primitive_id` (`PR_*` string).
4. **Resolution:** Use `primitive_exists` / `get_primitive` — do not hardcode unchecked atoms.
5. **Stability:** Reference by `primitive_id`, never by description text.

## Helpers

```python
from feature_intelligence.registry import PRIMITIVE_ID_FIELD, is_valid_primitive_id
```

Lineage / AST tables are **not** implemented in Sprint 1.
