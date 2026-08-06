# Operator Registry (Sprint 3)

Deterministic `OP_*` IDs (no UUIDs). Pack `1.0.0`, catalog `1.0`, **31** operators.

Semantics of an `OP_*` ID are immutable. Evolution uses new IDs / pack versions.

```python
from feature_intelligence.operators import OperatorRegistryService

svc = OperatorRegistryService(db_path)
svc.get_by_id("OP_EMA")
svc.validate_registry()
```

CLI: `operators list|get|validate|export|import` (no register/edit/execute).

Artifacts: `operators/operator_catalog.json`, `operators/operator_catalog.csv`.

Freeze: `docs/antigravity-doc/roadmap_1_feature_intelligence/sprint_3_operator_intelligence.md`
