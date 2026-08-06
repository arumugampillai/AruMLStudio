# Primitive Catalog (Sprint 1)

Architecture name: **Primitive Registry** (Step 0).  
Implementation: curated **Primitive Catalog** — seed-driven, no runtime registration.

## IDs

Deterministic `PR_*` only (never UUIDs), e.g. `PR_SPOT`, `PR_VOLUME`.

## Catalog version

Every row has `catalog_version`. Seed set is **`1.0`** (14 primitives).

## Seed hash

`EXPECTED_SEED_CATALOG_HASH` in `registry/catalog.py` must match `compute_seed_catalog_hash()`.

## APIs

```python
from feature_intelligence.registry import PrimitiveCatalogService

svc = PrimitiveCatalogService(db_path)
svc.list_primitives()
svc.get_primitive("PR_SPOT")
svc.validate_primitives()  # -> ValidationReport
```

## CLI

```bash
python -m feature_intelligence primitives list --db PATH
python -m feature_intelligence primitives get --id PR_SPOT --db PATH
python -m feature_intelligence primitives validate --db PATH
```

See freeze: `docs/antigravity-doc/roadmap_1_feature_intelligence/sprint_1_primitive_registry.md`.
