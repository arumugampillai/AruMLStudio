# Semantic Query Engine

Sprint 9 — structured, FRR-centric, **read-only** metadata retrieval.

## Versions

| Identifier | Value |
|------------|-------|
| `query_engine_version` | `1.0.0` |
| `query_language_version` | `1.0` |
| `schema_version` | `1.0` |
| `query_export_version` | `1.0` |

## Core principles

1. **Semantic Query retrieves metadata. It never creates metadata.**
2. **Every public query starts from `FRR_*`.** Callers may supply `FEAT_*`, `FRR_*`, or `canonical_name`; the engine resolves to FRR first.
3. **Missing FRRs are not auto-created.** Populate shells with `python -m feature_intelligence research sync` after features are registered. Until then, `match_all` / List All returns 0 rows even if `feature_registry` is non-empty.

## Query language (`1.0`)

Structured tokens only: `field:value` (whitespace-separated, **AND** semantics).

```text
domain:DOM_PRICE signal:SIG_MOMENTUM status:ACTIVE
```

| Field | Value form | Notes |
|-------|------------|-------|
| `feature` | `FEAT_*` or exact `canonical_name` | Missing feature → empty search hits |
| `domain` | `DOM_*` or active `canonical_name` alias | Prefer vocabulary_id |
| `signal` | `SIG_*` or active alias | Prefer vocabulary_id |
| `operator` | `OP_*` or unique name / short token | Ambiguous → validation error |
| `primitive` | `PR_*` or unique name | Ambiguous → error |
| `transformation` | `TR_*` | Exact |
| `status` | `EMPTY` \| `ACTIVE` \| `ARCHIVED` | Case-normalized |
| `validation` | `validated` \| `pending` \| `failed` | Case-normalized |
| `grammar` / `compiler` / `ontology_version` | Exact version string | |

**Forbidden:** natural language, fuzzy / wildcards, ranking, Query Result Hash.

**Sort:** `research_uuid` ASCII ascending, then `feature_uuid`.

## CLI

```bash
python -m feature_intelligence query search --query "domain:DOM_PRICE status:ACTIVE"
python -m feature_intelligence query inspect --name spot_ema_20
python -m feature_intelligence query validate --query "signal:momentum"
python -m feature_intelligence query export --query "status:EMPTY" --format json --out hits.json
python -m feature_intelligence query capabilities
```

## Package

Behavior lives in `feature_intelligence/query/`. Public callers should prefer `feature_intelligence.api`.

## Reserved (not implemented)

- **`SavedQuery`** — no table / API / CLI in Phase 1
- Query Result Hash — deferred to Experiment Intelligence
