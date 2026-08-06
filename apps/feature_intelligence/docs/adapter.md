# Legacy Adapter / Feature Registry Synchronizer (one-way pull)

```text
build_feature_registry_catalog(data_dir)  →  FIC Feature Registry
```

- Never writes legacy overlays from the synchronizer itself (catalog build may ensure `FR####` identities)
- **Idempotent** on preserved `FEAT_*` (when source supplies one) and/or `canonical_name`
- Legacy `FR####` stored as `legacy_feature_id` (not used as `FEAT_*`)
- Primitive binding via `PrimitiveMappingProvider` (explicit map → rules → domain hints → empty)
- Unmapped primitives are reported as **Failed** with reason `UNMAPPED_PRIMITIVES` (no invented primitives)
- Modes: `strict` / `lenient` (both report unmapped as Failed; reserved for future soft behavior)
- Optional **research sync** (default on for CLI/UI) creates missing FRR shells so Feature Explorer List All works

## FEAT ID preservation

| Source field | Behavior |
|--------------|----------|
| `feature_uuid` / `fic_feature_uuid` / `feature_id` matching `FEAT_*` UUIDv7 | Used as `feature_uuid` on insert; re-sync matches by that id |
| Legacy `feature_id` = `FR####` only | Stored as `legacy_feature_id`; FIC mints a new `FEAT_*` once; re-sync matches by `canonical_name` |

Today’s legacy `feature_registry_store.json` has **no** `FEAT_*` — only `FR####`. Cross-machine FEAT stability still uses `features export` / `import`.

## API

```python
from feature_intelligence.registry import FeatureRegistryService

svc = FeatureRegistryService(db_path)
summary = svc.synchronize_from_feature_registry(
    data_dir, mode="lenient", research_sync=True
)
# summary.total_source / already_registered / newly_imported / failed / failures
```

Legacy adapter still returns `SyncReport`:

```python
svc.sync_from_legacy(data_dir, mode="lenient")
```

## CLI

```bash
# from angelone/chart
python -m feature_intelligence features sync --data-dir ./data --mode lenient
python -m feature_intelligence features sync --chart-dir . --db PATH --no-research-sync
```

## UI

Feature Explorer → **Sync from Feature Registry** (toolbar). Shows a summary dialog; then refreshes List All.
