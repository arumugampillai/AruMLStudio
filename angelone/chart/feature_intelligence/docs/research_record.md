# Feature Research Record (Sprint 8)

Every registered feature (`FEAT_*`) becomes addressable as a **Research Object**
through a permanent **Feature Research Record (`FRR_*`)**.

> **Invariant: Research Record is metadata about research, not research itself.**  
> FRR holds identity, status, version pointers, nullable cross-registry links, and
> Phase 1–empty evidence / experiment placeholders. No AI, scoring, KG, or evidence
> computation in Sprint 8.

> **Invariant: Migration ≠ Business rows.**  
> Schema lives in migrations; FRR shells are materialized by `research sync` / import.

## Identity

```text
research_uuid = FRR_ + SHA256(full FEAT_* UTF-8)[:32].upper()
```

Pattern: `^FRR_[0-9A-F]{32}$`. Never a random UUID. Immutable once created.

| Version | Value |
|---------|-------|
| `research_version` | `1.0.0` |
| `schema_version` | `1.0` (≠ pack version) |
| `research_export_version` | `1.0` (on export envelopes) |

## Coverage

- Exactly one FRR per registered `FEAT_*`
- No FRR for `PR_*` / `OP_*` / `TR_*`
- Orphan FRR (feature missing) is rejected

## Materialization

| Path | Role |
|------|------|
| Migration `0009` | Schema + pack row only — **no** FRR business seed |
| **`research sync`** | Create missing EMPTY/`pending` shells; optional link fill |
| **Import** | Upsert (may set evidence placeholders) |
| Validate | Checks only — does **not** create FRRs |
| CLI edit | Forbidden |

## Evidence

`evidence_json` (and other evidence-shaped columns) are **NULL** unless populated by
import. Sprint 8 never computes evidence. No AI.

## Status vocabularies

| Field | Values | Sync default |
|-------|--------|--------------|
| `research_status` | `EMPTY` \| `ACTIVE` \| `ARCHIVED` | `EMPTY` |
| `validation_status` | `validated` \| `pending` \| `failed` | `pending` |

## CLI

```bash
python -m feature_intelligence research list [--status EMPTY|ACTIVE|ARCHIVED]
python -m feature_intelligence research get --id FRR_* | --feature FEAT_*
python -m feature_intelligence research validate [--mode strict|present] [--strict-refs]
python -m feature_intelligence research sync [--feature FEAT_*]
python -m feature_intelligence research stats
python -m feature_intelligence research completeness
python -m feature_intelligence research export --format json|yaml|csv --out PATH
python -m feature_intelligence research import --format json|yaml|csv --path PATH
```

Sync prints: `created=… updated=… unchanged=… skipped=…`

### Snapshot write policy

| Command | `research_statistics` |
|---------|----------------------|
| `validate` | **Always** appends snapshot |
| `sync` (success) | **Writes/refreshes** (`last_sync_at`) |
| `stats` | Reads latest; writes **only if empty** |
| `import` | **Never** writes stats |
| `completeness` | Read-only report — not a validity gate |

## Related docs

- [research_schema.md](./research_schema.md)
- [research_examples.md](./research_examples.md)

Freeze: `docs/antigravity-doc/roadmap_1_feature_intelligence/sprint_8_feature_research_record.md` (v1.1)
