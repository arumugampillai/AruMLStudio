# Public API (Python)

Sprint 9 — thin read-only surface in `feature_intelligence.api`.

## Envelope (`schema_version` = `1.0`)

```json
{
  "schema_version": "1.0",
  "query_engine_version": "1.0.0",
  "query_language_version": "1.0",
  "ok": true,
  "error": null,
  "execution_ms": 1.23,
  "data": {}
}
```

`execution_ms` is **required** on `search_features` and `inspect_feature` (diagnostics). Other callables may include it.

## Callables

| Function | Role |
|----------|------|
| `get_feature` | Identity + FRR pointer summary |
| `search_features` | Structured query or `match_all=True` |
| `get_research` | FRR row |
| `get_lineage` | Lineage summary (`direction`) |
| `get_ontology` | Ontology classification |
| `inspect_feature` | Inspector aggregate + `sections_present` |
| `get_capabilities` | Versions, filters, modes, exports; `read_only` / `frr_mandatory` |

All subject selectors resolve through **FRR** first. Missing FRR → error (never auto-sync).

```python
from feature_intelligence.api import search_features, inspect_feature, get_capabilities

caps = get_capabilities()
hits = search_features(query="status:EMPTY", db_path=db)
detail = inspect_feature(canonical_name="spot_ema_20", db_path=db)
```

## Must NOT

No create/update/delete methods. No NL. No ranking. No HTTP/REST gateway in Phase 1.

## Reserved

- `SavedQuery` persistence / CRUD
- UI deep-link navigation APIs
