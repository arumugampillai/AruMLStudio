# Feature Registry (Sprint 2)

Authoritative FIC catalog of generated features. DB: `feature_intelligence.db`.

## Identity

- `feature_uuid`: `FEAT_` + UUIDv7 (32 uppercase hex)
- `canonical_name`: lowercase snake_case, immutable
- `definition_hash`: SHA-256 over UTF-8 definition line
- Primitives: junction table `feature_primitives`

## Service

```python
from feature_intelligence.registry import FeatureRegistryService

svc = FeatureRegistryService(db_path)
svc.register_feature(...)  # internal / adapter / import — not CLI
svc.find_by_primitive("PR_SPOT")
svc.validate_registry()
```

## CLI (read-only + i/o + sync)

```bash
python -m feature_intelligence features list --db PATH
python -m feature_intelligence features get --name spot_ema20 --db PATH
python -m feature_intelligence features validate --db PATH
python -m feature_intelligence features export --format json --out out.json --db PATH
python -m feature_intelligence features import --format json --in out.json --db PATH
python -m feature_intelligence features sync --data-dir ./data --mode lenient --db PATH
```

`features sync` pulls the legacy Feature Registry catalog into FIC (idempotent; preserves `FEAT_*` when the source provides one). See `docs/adapter.md`.

Freeze: `docs/antigravity-doc/roadmap_1_feature_intelligence/sprint_2_feature_identity.md`
